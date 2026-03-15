"""각 레이어별 점수 분해 — 복사기 문제 원인 정밀 추적.

"왜 모든 레이어가 64x64 ROW_MAJOR SRAM을 고르는가?"를
roofline / affinity / cache 세 요소로 분해해서 확인한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import (
    LayerNode, LayerType, HWState,
    MemoryLayout, ComputeLocation, generate_default_candidates,
)
from src.constraint import HardwareSpec, apply_hard_constraints
from src.cost_model import (
    roofline_score, layout_affinity, cache_efficiency,
    compute_score, estimate_memory_traffic,
)


def load_spec() -> HardwareSpec:
    return HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json"
    )


def analyze_layer(name: str, node: LayerNode, spec: HardwareSpec):
    """단일 레이어의 점수 분해."""
    apply_hard_constraints(node, spec)

    print(f"\n  ── {name} ({node.layer_type.name}) ──")
    print(f"  dims={node.dims}, FLOPs={node.flops():,}")
    print(f"  후보 {len(node.candidates)}개 (hard constraint 후)")

    # SRAM 후보만 상위 5개 출력
    sram_candidates = [c for c in node.candidates if c.location == ComputeLocation.SRAM]
    print(f"\n  {'타일':>10} {'layout':>12} {'roofline':>10} {'affinity':>10} {'cache':>10} {'total':>10} {'traffic':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    scores = []
    for s in sram_candidates:
        rf = roofline_score(node, s, spec)
        af = layout_affinity(node, s)
        ce = cache_efficiency(node, s, spec)
        total = rf + af + ce
        traffic = estimate_memory_traffic(node, s)
        scores.append((s, rf, af, ce, total, traffic))

    scores.sort(key=lambda x: x[4], reverse=True)
    for s, rf, af, ce, total, traffic in scores:
        marker = " ★" if total == scores[0][4] else ""
        print(f"  {s.tile_m:>3}x{s.tile_n:<5} {s.layout.name:>12} "
              f"{rf:>10.4f} {af:>10.4f} {ce:>10.4f} {total:>10.4f} "
              f"{traffic:>12,}{marker}")


def main():
    spec = load_spec()
    print("=" * 90)
    print("  Score Breakdown — 복사기 문제 정밀 추적")
    print("=" * 90)

    layers = [
        ("QK_MatMul", LayerNode(
            name="QK_MatMul", layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 2048, "K": 64},
            candidates=generate_default_candidates(),
        )),
        ("Softmax", LayerNode(
            name="Softmax", layer_type=LayerType.SOFTMAX,
            dims={"M": 2048, "N": 2048},
            candidates=generate_default_candidates(),
        )),
        ("AV_MatMul", LayerNode(
            name="AV_MatMul", layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 64, "K": 2048},
            candidates=generate_default_candidates(),
        )),
        ("LayerNorm", LayerNode(
            name="LayerNorm", layer_type=LayerType.LAYERNORM,
            dims={"M": 2048, "N": 64},
            candidates=generate_default_candidates(),
        )),
    ]

    for name, node in layers:
        analyze_layer(name, node, spec)

    # 직사각 타일이 있었다면?
    print("\n\n" + "=" * 90)
    print("  가상 실험: 직사각 타일 추가 시 점수 변화")
    print("=" * 90)

    rect_tiles = [(32, 64), (64, 32), (32, 128), (128, 32), (64, 128), (128, 64)]
    for name, layer_type, dims in [
        ("QK_MatMul", LayerType.LINEAR, {"M": 2048, "N": 2048, "K": 64}),
        ("Softmax", LayerType.SOFTMAX, {"M": 2048, "N": 2048}),
        ("AV_MatMul", LayerType.LINEAR, {"M": 2048, "N": 64, "K": 2048}),
    ]:
        print(f"\n  ── {name} + 직사각 타일 (ROW_MAJOR, SRAM만) ──")
        print(f"  {'타일':>10} {'roofline':>10} {'affinity':>10} {'cache':>10} {'total':>10} {'traffic':>12}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

        all_tiles = [(32, 32), (64, 64), (128, 128)] + rect_tiles
        results = []
        for tm, tn in all_tiles:
            s = HWState(tm, tn, MemoryLayout.ROW_MAJOR, ComputeLocation.SRAM)
            node = LayerNode(name=name, layer_type=layer_type, dims=dims, candidates=[s])
            rf = roofline_score(node, s, spec)
            af = layout_affinity(node, s)
            ce = cache_efficiency(node, s, spec)
            total = rf + af + ce
            traffic = estimate_memory_traffic(node, s)
            results.append((tm, tn, rf, af, ce, total, traffic))

        results.sort(key=lambda x: x[5], reverse=True)
        best_total = results[0][5]
        for tm, tn, rf, af, ce, total, traffic in results:
            marker = " ★" if total == best_total else ""
            print(f"  {tm:>3}x{tn:<5} {rf:>10.4f} {af:>10.4f} {ce:>10.4f} {total:>10.4f} {traffic:>12,}{marker}")


if __name__ == "__main__":
    main()
