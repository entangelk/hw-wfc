"""HW-WFC PoC: MobileNetV2 Inverted Residual Block.

Depthwise Separable Convolution의 타일 선택 차별화를 검증한다:

  Pointwise Conv1x1 (expand) → Depthwise Conv3x3 → Pointwise Conv1x1 (project)
       ↑                                                      ↑
       └────────────────── skip connection ────────────────────┘

검증 포인트:
  1. Depthwise Conv가 Conv2D/Linear과 다른 레이아웃을 선택하는가? (BLOCK_TILED 강선호)
  2. im2col 기반 Conv2D traffic이 Linear traffic과 다른 타일 최적해를 유도하는가?
  3. Depthwise의 낮은 FLOPs가 엔트로피 기반 붕괴 순서에 영향을 주는가?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import LayerNode, LayerType, generate_default_candidates
from src.constraint import HardwareSpec
from src.cost_model import compute_score
from src.scheduler import HWWFCScheduler


def load_spec(name: str = "stress_gpu") -> HardwareSpec:
    return HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / f"{name}.json"
    )


def build_mobilenet_block(
    h: int = 56, w: int = 56, c_in: int = 32, expand_ratio: int = 6,
) -> list[LayerNode]:
    """MobileNetV2 Inverted Residual Block.

    1. Pointwise Conv 1x1 (expand): C_in → C_in*expand_ratio
    2. Depthwise Conv 3x3: 채널별 독립 연산
    3. Pointwise Conv 1x1 (project): C_in*expand_ratio → C_in
    4. Add (skip connection)
    """
    c_exp = c_in * expand_ratio
    candidates = generate_default_candidates()

    nodes = [
        LayerNode(
            name="PW_Expand",
            layer_type=LayerType.CONV2D,
            dims={"H": h, "W": w, "C_in": c_in, "C_out": c_exp, "KH": 1, "KW": 1},
            candidates=list(candidates),
            neighbors=["Add", "DW_Conv"],
        ),
        LayerNode(
            name="DW_Conv",
            layer_type=LayerType.DEPTHWISE_CONV,
            dims={"H": h, "W": w, "C": c_exp, "KH": 3, "KW": 3},
            candidates=list(candidates),
            neighbors=["PW_Expand", "ReLU6"],
        ),
        LayerNode(
            name="ReLU6",
            layer_type=LayerType.RELU,
            dims={"M": h * w, "N": c_exp},
            candidates=list(candidates),
            neighbors=["DW_Conv", "PW_Project"],
        ),
        LayerNode(
            name="PW_Project",
            layer_type=LayerType.CONV2D,
            dims={"H": h, "W": w, "C_in": c_exp, "C_out": c_in, "KH": 1, "KW": 1},
            candidates=list(candidates),
            neighbors=["ReLU6", "Add"],
        ),
        LayerNode(
            name="Add",
            layer_type=LayerType.RELU,  # element-wise add
            dims={"M": h * w, "N": c_in},
            candidates=list(candidates),
            neighbors=["PW_Expand", "PW_Project"],
        ),
    ]
    return nodes


def main():
    print("=" * 70)
    print("  HW-WFC PoC: MobileNetV2 Inverted Residual Block")
    print("=" * 70)

    for spec_name in ["stress_gpu", "a100"]:
        spec = load_spec(spec_name)
        nodes = build_mobilenet_block()

        print(f"\n{'─' * 70}")
        print(f"  GPU: {spec.name} (SRAM: {spec.sram_bytes // 1024}KB)")
        print(f"{'─' * 70}")

        print(f"\n  레이어 구성:")
        for n in nodes:
            print(f"    {n.name:14s} {n.layer_type.name:16s} FLOPs={n.flops():>12,}")

        t0 = time.perf_counter()
        scheduler = HWWFCScheduler(spec, penalty_threshold="auto")
        result = scheduler.schedule(nodes)
        elapsed = (time.perf_counter() - t0) * 1000

        for step in result.steps:
            print(f"  {step}")

        print(f"\n  결과 ({elapsed:.1f}ms):")
        for n in nodes:
            s = n.collapsed_state
            if s:
                score = compute_score(n, s, spec)
                print(f"    {n.name:14s} → tile={s.tile_m}x{s.tile_n}, "
                      f"{s.layout.name:12s}, {s.location.name:4s}  "
                      f"(score={score:.4f})")

        # Depthwise vs Pointwise 레이아웃 비교
        dw = next(n for n in nodes if n.name == "DW_Conv")
        pw = next(n for n in nodes if n.name == "PW_Expand")
        if dw.collapsed_state and pw.collapsed_state:
            dw_layout = dw.collapsed_state.layout.name
            pw_layout = pw.collapsed_state.layout.name
            if dw_layout != pw_layout:
                print(f"\n  OK: Depthwise({dw_layout}) ≠ Pointwise({pw_layout}) "
                      f"→ 레이아웃 차별화 확인")
            else:
                print(f"\n  주의: Depthwise와 Pointwise가 같은 레이아웃({dw_layout}) "
                      f"→ 전환 비용이 layout affinity를 상쇄했을 수 있음")


if __name__ == "__main__":
    main()
