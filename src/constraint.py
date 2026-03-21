"""HW-WFC 제약 엔진.

2-Phase 제약 시스템:
  Phase 1 (Hard Constraints): 물리적으로 불가능한 상태를 즉시 소거
  Phase 2 (Soft Constraints): 비용 모델 기반 페널티 부여
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import HWState, LayerNode, LayerType, MemoryLayout, ComputeLocation, working_memory_multiplier


@dataclass
class HardwareSpec:
    """타겟 하드웨어 스펙 (JSON에서 로드)."""
    name: str
    sram_bytes: int          # L1/Shared Memory 용량
    dram_bandwidth_gbps: float  # DRAM 대역폭 (GB/s)
    sram_bandwidth_gbps: float  # SRAM 대역폭 (GB/s)
    max_registers: int       # 레지스터 개수 한계
    alignment_bytes: int     # 메모리 정렬 요구사항
    compute_gflops: float    # 피크 연산 성능 (GFLOPS)

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareSpec:
        data = json.loads(Path(path).read_text())
        return cls(
            name=data["name"],
            sram_bytes=data["sram_bytes"],
            dram_bandwidth_gbps=data["dram_bandwidth_gbps"],
            sram_bandwidth_gbps=data["sram_bandwidth_gbps"],
            max_registers=data["max_registers"],
            alignment_bytes=data["alignment_bytes"],
            compute_gflops=data["compute_gflops"],
        )


# ─── Hard Constraints ────────────────────────────────────────────

def check_sram_capacity(state: HWState, spec: HardwareSpec, dtype_bytes: int = 4) -> bool:
    """SRAM 용량 초과 여부 (단일 타일). True = 유효(통과)."""
    if state.location == ComputeLocation.DRAM:
        return True  # DRAM이면 SRAM 제약 무관
    return state.memory_bytes(dtype_bytes) <= spec.sram_bytes


def check_working_memory(
    state: HWState,
    layer_type: LayerType,
    spec: HardwareSpec,
    dtype_bytes: int = 4,
) -> bool:
    """Working memory가 SRAM에 들어가는지 검사. True = 유효(통과).

    타일 하나를 처리할 때 필요한 총 버퍼(input + weight + output 등)가
    동시에 SRAM에 올라가야 한다. 이 총량이 SRAM을 초과하면 물리적으로 불가능.
    """
    if state.location == ComputeLocation.DRAM:
        return True
    multiplier = working_memory_multiplier(layer_type)
    actual_usage = state.memory_bytes(dtype_bytes) * multiplier
    return actual_usage <= spec.sram_bytes


def check_alignment(state: HWState, spec: HardwareSpec, dtype_bytes: int = 4) -> bool:
    """메모리 정렬 요구사항 충족 여부."""
    row_bytes = state.tile_n * dtype_bytes
    return row_bytes % spec.alignment_bytes == 0


def apply_hard_constraints(
    node: LayerNode,
    spec: HardwareSpec,
    dtype_bytes: int = 4,
) -> int:
    """노드의 후보군에서 물리적으로 불가능한 상태를 제거.

    Returns: 제거된 상태 수.
    """
    if node.is_collapsed:
        return 0

    original_count = len(node.candidates)
    node.candidates = [
        s for s in node.candidates
        if check_sram_capacity(s, spec, dtype_bytes)
        and check_alignment(s, spec, dtype_bytes)
        and check_working_memory(s, node.layer_type, spec, dtype_bytes)
    ]
    return original_count - len(node.candidates)


# ─── Relational Constraints (인접 레이어 간) ─────────────────────

def layout_transition_penalty(src: HWState, dst: HWState) -> float:
    """인접 레이어 간 메모리 레이아웃 전환 페널티.

    같은 레이아웃이면 0, 다르면 전환 비용을 반환.
    """
    if src.layout == dst.layout:
        return 0.0
    # Row ↔ Col 전환이 가장 비싸고, Block-tiled은 중간
    if {src.layout, dst.layout} == {MemoryLayout.ROW_MAJOR, MemoryLayout.COL_MAJOR}:
        return 1.0  # 전치(Transpose) 필요
    return 0.5  # Block-tiled 전환은 상대적으로 저렴


def location_transition_penalty(src: HWState, dst: HWState) -> float:
    """SRAM ↔ DRAM 전환 페널티."""
    if src.location == dst.location:
        return 0.0
    return 0.8  # 메모리 계층 이동 비용


def tile_shape_transition_penalty(src: HWState, dst: HWState) -> float:
    """타일 크기 불일치 페널티.

    같은 타일이면 0. 다르면 re-tiling(재배열) 비용.
    - 같은 면적, 다른 형태 (64x64 vs 32x128): 0.3 (stride 재배열)
    - 다른 면적: 면적 비율에 비례 (최대 0.6)
    """
    if src.tile_m == dst.tile_m and src.tile_n == dst.tile_n:
        return 0.0

    src_area = src.tile_elements
    dst_area = dst.tile_elements

    if src_area == dst_area:
        # 같은 면적, 다른 형태 — stride 변환만 필요
        return 0.3

    # 다른 면적 — 면적 비율이 클수록 비용 증가
    ratio = max(src_area, dst_area) / min(src_area, dst_area)
    # ratio 1→0, ratio 4→0.4, ratio 16→0.6 (로그 스케일)
    import math
    return min(0.6, 0.3 * math.log2(ratio))


def derive_transition_weight(spec: HardwareSpec, dtype_bytes: int = 4) -> float:
    """SRAM 용량 기반 전환 비용 가중치 자동 파생.

    reference 타일(32x32)의 SRAM 점유율을 기반으로 계산.
    SRAM이 작을수록 전환 시 데이터 재배치가 전체 시간의 큰 비율을 차지함.
    """
    reference_tile_bytes = 32 * 32 * dtype_bytes  # 4096 bytes for float32
    ratio = reference_tile_bytes / spec.sram_bytes
    return max(0.01, min(0.5, ratio))


def total_transition_penalty(src: HWState, dst: HWState) -> float:
    """인접 레이어 간 총 전환 페널티."""
    return (
        layout_transition_penalty(src, dst)
        + location_transition_penalty(src, dst)
        + tile_shape_transition_penalty(src, dst)
    )


def propagate_constraints(
    collapsed_node: LayerNode,
    neighbor: LayerNode,
    penalty_threshold: float = 1.5,
) -> int:
    """붕괴된 노드의 상태를 기반으로 인접 노드의 후보를 제약 전파.

    페널티가 threshold를 초과하는 후보를 제거한다.

    Returns: 제거된 후보 수.
    """
    if not collapsed_node.is_collapsed or neighbor.is_collapsed:
        return 0

    src_state = collapsed_node.collapsed_state
    original_count = len(neighbor.candidates)
    neighbor.candidates = [
        c for c in neighbor.candidates
        if total_transition_penalty(src_state, c) <= penalty_threshold
    ]
    return original_count - len(neighbor.candidates)
