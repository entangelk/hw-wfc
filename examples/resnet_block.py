"""HW-WFC PoC: ResNet Residual Block (DAG 전파 검증).

Skip connection이 있는 DAG 구조에서 BFS 전파가 동작하는지 검증한다:

  Conv1 → BN1 → ReLU1 → Conv2 → BN2 → Add → ReLU2
    ↑                                    ↑
    └──────── (skip connection) ──────────┘

검증 포인트:
  1. Skip connection을 통해 Conv1의 붕괴가 Add 노드에도 전파되는가?
  2. 양방향 BFS가 cycle 없이 동작하는가?
  3. 1D 체인 대비 DAG에서의 탐색 공간 감소율 차이
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import LayerNode, LayerType, generate_default_candidates
from src.constraint import HardwareSpec, apply_hard_constraints
from src.cost_model import compute_score
from src.scheduler import HWWFCScheduler


def load_spec() -> HardwareSpec:
    return HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / "stress_gpu.json"
    )


def build_resnet_block() -> list[LayerNode]:
    """ResNet Residual Block (skip connection 포함).

    H=56, W=56, C=64 → Conv3x3 → BN → ReLU → Conv3x3 → BN → Add → ReLU

    그래프 구조 (neighbors):
      Conv1 ──→ BN1 ──→ ReLU1 ──→ Conv2 ──→ BN2 ──→ Add ──→ ReLU2
        │                                              ↑
        └────────────── skip connection ────────────────┘
    """
    nodes = [
        LayerNode(
            name="Conv1",
            layer_type=LayerType.CONV2D,
            dims={"H": 56, "W": 56, "C_in": 64, "C_out": 64, "KH": 3, "KW": 3},
            candidates=generate_default_candidates(),
            neighbors=["BN1", "Add"],  # 순방향 + skip
        ),
        LayerNode(
            name="BN1",
            layer_type=LayerType.LAYERNORM,  # BN ≈ LayerNorm for cost model
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
            neighbors=["Conv1", "ReLU1"],
        ),
        LayerNode(
            name="ReLU1",
            layer_type=LayerType.RELU,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
            neighbors=["BN1", "Conv2"],
        ),
        LayerNode(
            name="Conv2",
            layer_type=LayerType.CONV2D,
            dims={"H": 56, "W": 56, "C_in": 64, "C_out": 64, "KH": 3, "KW": 3},
            candidates=generate_default_candidates(),
            neighbors=["ReLU1", "BN2"],
        ),
        LayerNode(
            name="BN2",
            layer_type=LayerType.LAYERNORM,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
            neighbors=["Conv2", "Add"],
        ),
        LayerNode(
            name="Add",
            layer_type=LayerType.RELU,  # element-wise add ≈ ReLU for cost model
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
            neighbors=["Conv1", "BN2", "ReLU2"],  # skip + main + output
        ),
        LayerNode(
            name="ReLU2",
            layer_type=LayerType.RELU,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
            neighbors=["Add"],
        ),
    ]
    return nodes


def build_resnet_block_linear() -> list[LayerNode]:
    """같은 레이어를 1D 체인으로 연결한 버전 (비교용). neighbors 미설정."""
    nodes = [
        LayerNode(
            name="Conv1",
            layer_type=LayerType.CONV2D,
            dims={"H": 56, "W": 56, "C_in": 64, "C_out": 64, "KH": 3, "KW": 3},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="BN1",
            layer_type=LayerType.LAYERNORM,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="ReLU1",
            layer_type=LayerType.RELU,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Conv2",
            layer_type=LayerType.CONV2D,
            dims={"H": 56, "W": 56, "C_in": 64, "C_out": 64, "KH": 3, "KW": 3},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="BN2",
            layer_type=LayerType.LAYERNORM,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Add",
            layer_type=LayerType.RELU,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="ReLU2",
            layer_type=LayerType.RELU,
            dims={"M": 56 * 56, "N": 64},
            candidates=generate_default_candidates(),
        ),
    ]
    return nodes


def total_score(nodes: list[LayerNode], spec: HardwareSpec) -> float:
    return sum(
        compute_score(n, n.collapsed_state, spec)
        for n in nodes if n.collapsed_state
    )


def main():
    print("=" * 70)
    print("  HW-WFC PoC: ResNet Residual Block (DAG vs Linear)")
    print("=" * 70)

    spec = load_spec()

    # ── DAG 버전 (skip connection) ──
    print(f"\n{'─' * 70}")
    print("  [1] DAG 구조 (skip connection: Conv1 → Add)")
    print(f"{'─' * 70}")

    nodes_dag = build_resnet_block()
    print(f"\n  그래프 연결:")
    for n in nodes_dag:
        print(f"    {n.name}: neighbors={n.neighbors}")

    t0 = time.perf_counter()
    scheduler = HWWFCScheduler(spec, penalty_threshold=1.5)
    result_dag = scheduler.schedule(nodes_dag)
    t_dag = time.perf_counter() - t0

    for step in result_dag.steps:
        print(f"  {step}")

    print(f"\n  결과:")
    for n in nodes_dag:
        print(f"    {n.name:8s} → {n.collapsed_state}")

    # ── Linear 버전 (1D 체인, 비교용) ──
    print(f"\n{'─' * 70}")
    print("  [2] Linear 구조 (1D 체인, skip connection 없음)")
    print(f"{'─' * 70}")

    nodes_lin = build_resnet_block_linear()

    t0 = time.perf_counter()
    scheduler2 = HWWFCScheduler(spec, penalty_threshold=1.5)
    result_lin = scheduler2.schedule(nodes_lin)
    t_lin = time.perf_counter() - t0

    for step in result_lin.steps:
        print(f"  {step}")

    print(f"\n  결과:")
    for n in nodes_lin:
        print(f"    {n.name:8s} → {n.collapsed_state}")

    # ── 비교 ──
    print(f"\n{'=' * 70}")
    print("  비교 결과")
    print(f"{'=' * 70}")

    dag_score = total_score(nodes_dag, spec)
    lin_score = total_score(nodes_lin, spec)

    print(f"\n  {'':20s} {'DAG':>12s} {'Linear':>12s}")
    print(f"  {'─' * 46}")
    print(f"  {'Status':20s} {'OK' if result_dag.success else 'FAIL':>12s} "
          f"{'OK' if result_lin.success else 'FAIL':>12s}")
    print(f"  {'Time (ms)':20s} {t_dag*1000:>12.2f} {t_lin*1000:>12.2f}")
    print(f"  {'Total Score':20s} {dag_score:>12.4f} {lin_score:>12.4f}")
    print(f"  {'Backtracks':20s} {result_dag.backtracks_used:>12d} "
          f"{result_lin.backtracks_used:>12d}")
    print(f"  {'Search Reduction':20s} "
          f"{(1 - result_dag.final_search_space / result_dag.initial_search_space) * 100:>11.1f}% "
          f"{(1 - result_lin.final_search_space / result_lin.initial_search_space) * 100:>11.1f}%")

    # skip connection으로 인한 제약 차이 확인
    dag_states = {n.name: n.collapsed_state for n in nodes_dag}
    lin_states = {n.name: n.collapsed_state for n in nodes_lin}
    diff = sum(1 for k in dag_states if dag_states[k] != lin_states.get(k))
    print(f"\n  DAG vs Linear 다른 선택: {diff}/{len(dag_states)} 노드")
    if diff > 0:
        for k in dag_states:
            if dag_states[k] != lin_states.get(k):
                print(f"    {k}: DAG={dag_states[k]}  /  Linear={lin_states[k]}")


if __name__ == "__main__":
    main()
