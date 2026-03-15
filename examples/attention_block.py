"""HW-WFC PoC: Transformer Attention Block.

ideation_2.md Section 4의 검증 계획을 실현한다:
  QKV Projection → QK MatMul → Softmax → AV MatMul → Output Projection → LayerNorm

검증 포인트:
  1. Softmax가 MatMul과 다른 최적 타일을 선택하는가?
  2. 전파가 레이아웃 충돌을 감지하는가?
  3. Grid Search 대비 탐색 공간 감소율과 선택 품질 비교
"""

import sys
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import LayerNode, LayerType, generate_default_candidates
from src.constraint import HardwareSpec, apply_hard_constraints
from src.cost_model import compute_score
from src.scheduler import HWWFCScheduler


def load_spec() -> HardwareSpec:
    return HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json"
    )


def build_attention_block() -> list[LayerNode]:
    """Transformer Self-Attention Block.

    seq_len=2048, d_model=512, d_head=64, num_heads=8
    """
    return [
        LayerNode(
            name="QKV_Proj",
            layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 192, "K": 512},  # 3 * d_head
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="QK_MatMul",
            layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 2048, "K": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Softmax",
            layer_type=LayerType.SOFTMAX,
            dims={"M": 2048, "N": 2048},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="AV_MatMul",
            layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 64, "K": 2048},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Out_Proj",
            layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 512, "K": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="LayerNorm",
            layer_type=LayerType.LAYERNORM,
            dims={"M": 2048, "N": 512},
            candidates=generate_default_candidates(),
        ),
    ]


def grid_search_best(nodes: list[LayerNode], spec: HardwareSpec) -> float:
    """Grid Search로 전체 파이프라인 최적 점수를 찾는다.

    단순화: 각 노드의 best score 합으로 upper bound 계산.
    (실제 Grid Search는 조합 폭발이지만, 독립 최적이 상한)
    """
    total = 0.0
    for node in nodes:
        best = max(compute_score(node, s, spec) for s in node.candidates)
        total += best
    return total


def hwwfc_total_score(nodes: list[LayerNode], spec: HardwareSpec) -> float:
    """HW-WFC가 선택한 결과의 총 점수."""
    total = 0.0
    for node in nodes:
        if node.collapsed_state:
            total += compute_score(node, node.collapsed_state, spec)
    return total


def main():
    print("=" * 70)
    print("  HW-WFC PoC: Transformer Attention Block")
    print("=" * 70)

    spec = load_spec()
    nodes = build_attention_block()

    print(f"\n[Model] {len(nodes)} layers")
    for node in nodes:
        print(f"  {node.name}: {node.layer_type.name}, "
              f"FLOPs={node.flops():,}, candidates={node.num_candidates}")

    # Grid Search upper bound (각 레이어 독립 최적)
    nodes_for_gs = build_attention_block()
    for n in nodes_for_gs:
        apply_hard_constraints(n, spec)
    gs_best = grid_search_best(nodes_for_gs, spec)
    gs_candidates_per_node = [len(n.candidates) for n in nodes_for_gs]
    gs_combinations = 1
    for c in gs_candidates_per_node:
        gs_combinations *= c

    # HW-WFC 실행
    print(f"\n{'─' * 70}")
    print("  HW-WFC Scheduling")
    print(f"{'─' * 70}")

    t0 = time.perf_counter()
    scheduler = HWWFCScheduler(spec, penalty_threshold=1.5)
    result = scheduler.schedule(nodes)
    t_wfc = time.perf_counter() - t0

    for step in result.steps:
        print(f"  {step}")

    wfc_score = hwwfc_total_score(nodes, spec)

    # 결과 비교
    print(f"\n{'=' * 70}")
    print("  Result Comparison")
    print(f"{'=' * 70}")

    print(f"\n  [HW-WFC]")
    print(f"    Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"    Time: {t_wfc*1000:.2f} ms")
    print(f"    Search space: {result.initial_search_space} → {result.final_search_space}")
    reduction = (1 - result.final_search_space / result.initial_search_space) * 100
    print(f"    Reduction: {reduction:.1f}%")
    print(f"    Backtracks: {result.backtracks_used}")
    print(f"    Total score: {wfc_score:.4f}")

    print(f"\n  [Grid Search (upper bound)]")
    print(f"    Combinations to evaluate: {gs_combinations:,}")
    print(f"    Best possible score (independent): {gs_best:.4f}")

    # 품질 비교
    if gs_best > 0:
        quality = wfc_score / gs_best * 100
        print(f"\n  [Quality]")
        print(f"    HW-WFC / Grid Search best: {quality:.1f}%")
        if quality >= 95:
            print(f"    → 독립 최적의 {quality:.1f}% 달성 (우수)")
        elif quality >= 85:
            print(f"    → 전환 비용 반영으로 인한 차이 (정상)")
        else:
            print(f"    !! 품질이 낮음 — Cost Model 또는 붕괴 순서 검토 필요")

    # 최종 선택 요약
    print(f"\n  [Final Layout]")
    unique_states = set()
    for node in nodes:
        s = node.collapsed_state
        if s:
            unique_states.add(s)
            print(f"    {node.name:12s} → tile={s.tile_m}x{s.tile_n}, "
                  f"{s.layout.name}, {s.location.name}")
    print(f"\n    Unique states: {len(unique_states)} / {len(nodes)} layers")
    if len(unique_states) > 1:
        print(f"    → 레이어별 차별화된 최적해 선택 확인")
    else:
        print(f"    !! 모든 레이어 동일 상태 — 복사기 문제")


if __name__ == "__main__":
    main()
