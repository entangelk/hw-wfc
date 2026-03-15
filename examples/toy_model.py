"""HW-WFC PoC: Linear → ReLU → Linear (Toy Model).

ideation.md의 Section 3 시뮬레이션을 실제로 실행한다.
SRAM 64KB 제약 하에서 제약 전파만으로 탐색 공간이
얼마나 줄어드는지 확인하는 최소 검증.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import LayerNode, LayerType, generate_default_candidates
from src.constraint import HardwareSpec
from src.scheduler import HWWFCScheduler


def build_toy_model(spec: HardwareSpec) -> list[LayerNode]:
    """Linear(A) → ReLU(B) → Linear(C) 모델 생성."""

    # 각 레이어에 동일한 기본 후보군 부여 (Superposition)
    candidates_a = generate_default_candidates()
    candidates_b = generate_default_candidates()
    candidates_c = generate_default_candidates()

    nodes = [
        LayerNode(
            name="Linear_A",
            layer_type=LayerType.LINEAR,
            dims={"M": 256, "N": 512, "K": 256},
            candidates=candidates_a,
        ),
        LayerNode(
            name="ReLU_B",
            layer_type=LayerType.RELU,
            dims={"M": 256, "N": 512},
            candidates=candidates_b,
        ),
        LayerNode(
            name="Linear_C",
            layer_type=LayerType.LINEAR,
            dims={"M": 256, "N": 128, "K": 512},
            candidates=candidates_c,
        ),
    ]
    return nodes


def main():
    print("=" * 60)
    print("  HW-WFC PoC: Toy Model (Linear → ReLU → Linear)")
    print("=" * 60)

    # 하드웨어 스펙 로드
    spec_path = Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json"
    spec = HardwareSpec.from_json(spec_path)
    print(f"\n[Hardware] {spec.name}")
    print(f"  SRAM: {spec.sram_bytes // 1024}KB")
    print(f"  Alignment: {spec.alignment_bytes}B")

    # 모델 구성
    nodes = build_toy_model(spec)
    print(f"\n[Model] {len(nodes)} layers")
    for node in nodes:
        print(f"  {node.name}: {node.layer_type.name}, "
              f"FLOPs={node.flops():,}, "
              f"candidates={node.num_candidates}")

    # 스케줄링 실행
    print("\n" + "-" * 60)
    print("  Scheduling Start")
    print("-" * 60)

    scheduler = HWWFCScheduler(spec, penalty_threshold=1.5)
    result = scheduler.schedule(nodes)

    # 과정 출력
    for step in result.steps:
        print(f"  {step}")

    # 최종 결과
    print("\n" + "=" * 60)
    print("  Final Result")
    print("=" * 60)
    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Search space: {result.initial_search_space} → {result.final_search_space}")

    reduction = (1 - result.final_search_space / result.initial_search_space) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Backtracks: {result.backtracks_used}")

    if result.success:
        print(f"\n  Optimized Layout:")
        for node in nodes:
            print(f"    {node.name}: {node.collapsed_state}")

    # Grid Search와 비교
    print(f"\n  [Comparison]")
    print(f"    Grid Search would evaluate: {result.initial_search_space ** len(nodes):,} combinations")
    print(f"    HW-WFC collapse steps: {len(result.steps)}")


if __name__ == "__main__":
    main()
