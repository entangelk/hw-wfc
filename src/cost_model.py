"""HW-WFC 경량 비용 모델.

각 HWState의 예상 성능 점수를 계산한다.
점수가 높을수록 해당 상태가 더 효율적임을 의미한다.

v2 개선사항 (2026-03-14):
- 타일 크기에 따른 데이터 재사용률(data reuse) 반영
- 레이어 타입별 레이아웃 선호도 반영
- cache_efficiency를 연속 함수로 변경
"""

from __future__ import annotations

import math

from .state import HWState, LayerNode, LayerType, MemoryLayout, ComputeLocation, working_memory_multiplier
from .constraint import HardwareSpec


def _matmul_memory_traffic(node: LayerNode, state: HWState, dtype_bytes: int) -> int:
    """MatMul/Linear의 실제 메모리 트래픽 추정 (타일링 고려).

    C[M,N] = A[M,K] × B[K,N] 에서:
    - A의 각 행은 tile_n 번 재사용 (B의 열 타일 수만큼)
    - B의 각 열은 tile_m 번 재사용 (A의 행 타일 수만큼)
    - 타일이 클수록 재사용이 많아져 메모리 트래픽이 줄어듦
    """
    m = node.dims.get("M", 1)
    n = node.dims.get("N", 1)
    k = node.dims.get("K", 1)

    tm, tn = state.tile_m, state.tile_n

    # A[M,K] 로드: M*K 원소를 (N/tn)번 나눠 읽지만, 타일 내에서 tn번 재사용
    # 실제 로드 횟수 = ceil(N/tn)번 × M*K
    tiles_n = math.ceil(n / tn)
    tiles_m = math.ceil(m / tm)

    a_traffic = tiles_n * m * k * dtype_bytes
    b_traffic = tiles_m * k * n * dtype_bytes
    c_traffic = m * n * dtype_bytes  # 출력은 1번 쓰기

    return a_traffic + b_traffic + c_traffic


def _softmax_memory_traffic(node: LayerNode, state: HWState, dtype_bytes: int) -> int:
    """Softmax의 메모리 트래픽 추정.

    Softmax는 행 단위 reduction이므로:
    - 각 행의 전체 데이터를 한 번에 읽어야 max/sum 계산 가능
    - 타일의 tile_n이 행 길이(N)보다 작으면, 같은 행을 여러 번 읽어야 함 (2-pass)
    - tile_m이 작을수록 한 번에 처리하는 행 수가 적어 SRAM 효율적
    """
    m = node.dims.get("M", 1)
    n = node.dims.get("N", 1)

    passes = math.ceil(n / state.tile_n)
    # pass가 1이면 1번 읽기, 2 이상이면 2-pass (max→sum→normalize)
    read_multiplier = 1 if passes == 1 else 2 * passes
    traffic = m * n * dtype_bytes * read_multiplier + m * n * dtype_bytes  # +write
    return traffic


def _elementwise_memory_traffic(node: LayerNode, dtype_bytes: int) -> int:
    """Element-wise 연산(ReLU 등)의 메모리 트래픽.

    타일 크기와 무관하게 입출력 텐서를 1번씩 읽고 쓴다.
    """
    total_elements = 1
    for v in node.dims.values():
        total_elements *= v
    return total_elements * dtype_bytes * 2  # read + write


def _conv2d_memory_traffic(node: LayerNode, state: HWState, dtype_bytes: int) -> int:
    """Conv2D의 im2col 기반 메모리 트래픽 추정.

    im2col은 입력을 [H_out*W_out, C_in*KH*KW] 행렬로 변환한 뒤
    커널 [C_out, C_in*KH*KW]과 MatMul하는 방식.
    타일링은 이 변환된 행렬에 적용된다.
    """
    h = node.dims.get("H", 1)
    w = node.dims.get("W", 1)
    c_in = node.dims.get("C_in", 1)
    c_out = node.dims.get("C_out", 1)
    kh = node.dims.get("KH", 3)
    kw = node.dims.get("KW", 3)

    # im2col 후의 행렬 크기
    m = h * w          # 출력 spatial 크기
    k = c_in * kh * kw  # 펼쳐진 커널 크기
    n = c_out           # 출력 채널 수

    tm, tn = state.tile_m, state.tile_n
    tiles_n = math.ceil(n / tn)
    tiles_m = math.ceil(m / tm)

    # im2col 입력 행렬 로드 (커널 오버랩으로 인한 중복 읽기 포함)
    im2col_traffic = tiles_n * m * k * dtype_bytes
    # 커널 행렬 로드
    kernel_traffic = tiles_m * k * n * dtype_bytes
    # 출력 쓰기
    output_traffic = m * n * dtype_bytes

    return im2col_traffic + kernel_traffic + output_traffic


def _depthwise_conv_memory_traffic(node: LayerNode, state: HWState, dtype_bytes: int) -> int:
    """Depthwise Conv의 메모리 트래픽 추정.

    채널별 독립 연산이므로 데이터 재사용이 제한적.
    타일이 공간 차원(H,W)을 커버할수록 커널 재사용이 증가한다.
    """
    h = node.dims.get("H", 1)
    w = node.dims.get("W", 1)
    c = node.dims.get("C", 1)
    kh = node.dims.get("KH", 3)
    kw = node.dims.get("KW", 3)

    # 타일이 공간 차원을 커버하는 비율 → 커널 로드 횟수에 반영
    spatial = h * w
    tile_spatial = state.tile_m * state.tile_n
    tiles = math.ceil(spatial / tile_spatial) if tile_spatial > 0 else spatial

    # 각 채널에 대해: 입력 + 커널 + 출력
    # 입력은 halo(커널 오버랩) 포함으로 약간 더 읽음
    halo_factor = (kh * kw) / max(tile_spatial, 1)
    input_traffic = c * spatial * (1.0 + min(halo_factor, 1.0)) * dtype_bytes
    kernel_traffic = c * kh * kw * tiles * dtype_bytes  # 타일마다 커널 재로드
    output_traffic = c * spatial * dtype_bytes

    return int(input_traffic + kernel_traffic + output_traffic)


def estimate_memory_traffic(
    node: LayerNode, state: HWState, dtype_bytes: int = 4
) -> int:
    """레이어 타입에 따른 메모리 트래픽 추정."""
    if node.layer_type == LayerType.LINEAR:
        return _matmul_memory_traffic(node, state, dtype_bytes)
    if node.layer_type == LayerType.CONV2D:
        return _conv2d_memory_traffic(node, state, dtype_bytes)
    if node.layer_type == LayerType.DEPTHWISE_CONV:
        return _depthwise_conv_memory_traffic(node, state, dtype_bytes)
    if node.layer_type == LayerType.SOFTMAX:
        return _softmax_memory_traffic(node, state, dtype_bytes)
    return _elementwise_memory_traffic(node, dtype_bytes)


def roofline_score(
    node: LayerNode,
    state: HWState,
    spec: HardwareSpec,
    dtype_bytes: int = 4,
) -> float:
    """Roofline 모델 기반 성능 점수 (0~1 범위로 정규화)."""
    flops = node.flops()
    traffic = estimate_memory_traffic(node, state, dtype_bytes)

    if traffic == 0:
        return 0.0

    oi = flops / traffic  # Operational Intensity

    if state.location == ComputeLocation.SRAM:
        bandwidth = spec.sram_bandwidth_gbps * 1e9
    else:
        bandwidth = spec.dram_bandwidth_gbps * 1e9

    peak_compute = spec.compute_gflops * 1e9
    achievable = min(peak_compute, oi * bandwidth)
    score = achievable / peak_compute if peak_compute > 0 else 0.0
    return min(1.0, max(0.0, score))


def layout_affinity(node: LayerNode, state: HWState) -> float:
    """레이어 타입별 메모리 레이아웃 친화도.

    - LINEAR/MatMul: ROW_MAJOR가 출력 행렬 write에 유리
    - SOFTMAX: ROW_MAJOR 필수 (행 단위 reduction)
    - RELU/LAYERNORM: 레이아웃 무관 (element-wise)
    - CONV2D: BLOCK_TILED이 공간적 지역성에 유리
    """
    layout = state.layout

    if node.layer_type == LayerType.LINEAR:
        if layout == MemoryLayout.ROW_MAJOR:
            return 0.15
        elif layout == MemoryLayout.BLOCK_TILED:
            return 0.08
        else:  # COL_MAJOR
            return 0.0

    if node.layer_type == LayerType.SOFTMAX:
        if layout == MemoryLayout.ROW_MAJOR:
            return 0.2  # 강한 선호
        elif layout == MemoryLayout.BLOCK_TILED:
            return 0.03
        else:
            return 0.0  # COL_MAJOR는 softmax에 최악

    if node.layer_type == LayerType.CONV2D:
        if layout == MemoryLayout.BLOCK_TILED:
            return 0.15
        elif layout == MemoryLayout.ROW_MAJOR:
            return 0.08
        else:
            return 0.0

    if node.layer_type == LayerType.DEPTHWISE_CONV:
        # Depthwise: 채널 독립 → BLOCK_TILED이 공간 지역성에 강하게 유리
        if layout == MemoryLayout.BLOCK_TILED:
            return 0.20
        elif layout == MemoryLayout.ROW_MAJOR:
            return 0.05
        else:
            return 0.0

    # RELU, LAYERNORM 등 element-wise: 레이아웃 무관
    return 0.05


def cache_efficiency(
    node: LayerNode, state: HWState, spec: HardwareSpec, dtype_bytes: int = 4,
) -> float:
    """SRAM 활용률 점수 (연속 함수, working memory 고려).

    실제 SRAM 점유 = tile_bytes × working_memory_multiplier
    최적점 ~0.75 (75%)에서 피크, 양쪽으로 감소.
    """
    if state.location != ComputeLocation.SRAM:
        return 0.0

    multiplier = working_memory_multiplier(node.layer_type)
    actual_usage_bytes = state.memory_bytes(dtype_bytes) * multiplier
    usage = actual_usage_bytes / spec.sram_bytes

    if usage > 1.0:
        return -0.3  # 페널티: working memory 기준 SRAM 초과

    # 가우시안 형태: 0.75에서 피크, sigma=0.25
    optimal = 0.75
    sigma = 0.25
    efficiency = math.exp(-((usage - optimal) ** 2) / (2 * sigma ** 2))
    return efficiency * 0.2  # 최대 0.2 보너스


def compute_score(
    node: LayerNode,
    state: HWState,
    spec: HardwareSpec,
    dtype_bytes: int = 4,
) -> float:
    """종합 성능 점수."""
    base = roofline_score(node, state, spec, dtype_bytes)
    affinity = layout_affinity(node, state)
    cache = cache_efficiency(node, state, spec, dtype_bytes)
    return base + affinity + cache


def score_all_candidates(
    node: LayerNode,
    spec: HardwareSpec,
    dtype_bytes: int = 4,
) -> list[tuple[HWState, float]]:
    """노드의 모든 후보에 대해 점수를 계산하고, 점수 내림차순으로 정렬."""
    scored = [
        (state, compute_score(node, state, spec, dtype_bytes))
        for state in node.candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def scores_to_probabilities(scored: list[tuple[HWState, float]]) -> list[tuple[HWState, float]]:
    """점수를 확률로 변환 (softmax 스타일 정규화)."""
    if not scored:
        return []

    max_score = max(s for _, s in scored)
    exp_scores = [(state, math.exp(score - max_score)) for state, score in scored]
    total = sum(e for _, e in exp_scores)

    if total == 0:
        uniform = 1.0 / len(exp_scores)
        return [(state, uniform) for state, _ in exp_scores]

    return [(state, e / total) for state, e in exp_scores]
