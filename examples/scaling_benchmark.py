"""HW-WFC Phase 2: 대규모 모델 스케일링 벤치마크.

실제 GPU 스펙(A100, H100)에서 GPT-2 Small(72 layers),
BERT-Base(84 layers)를 스케줄링하여 WFC의 확장성을 검증한다.

검증 포인트:
  1. 1000+ 초기 상태에서 WFC가 몇 ms 안에 완료되는가?
  2. 레이어별 차별화된 선택이 이루어지는가?
  3. A100 vs H100 스펙 차이가 결과에 반영되는가?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import generate_default_candidates
from src.constraint import HardwareSpec, apply_hard_constraints
from src.cost_model import compute_score
from src.scheduler import HWWFCScheduler
from src.model_zoo import gpt2_small, bert_base, generate_large_candidates


SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"


def run_benchmark(model_name, nodes, spec, threshold="auto"):
    """단일 벤치마크 실행."""
    initial_total = sum(n.num_candidates for n in nodes)

    t0 = time.perf_counter()
    scheduler = HWWFCScheduler(spec, penalty_threshold=threshold)
    result = scheduler.schedule(nodes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 유니크 상태 수
    states = set()
    for n in nodes:
        if n.collapsed_state:
            states.add(n.collapsed_state)

    # 총 점수
    total_score = sum(
        compute_score(n, n.collapsed_state, spec)
        for n in nodes if n.collapsed_state
    )

    return {
        "model": model_name,
        "spec": spec.name,
        "layers": len(nodes),
        "initial_states": initial_total,
        "final_states": result.final_search_space,
        "success": result.success,
        "time_ms": elapsed_ms,
        "backtracks": result.backtracks_used,
        "unique_states": len(states),
        "total_score": total_score,
        "threshold": result.final_threshold,
    }


def main():
    print("=" * 78)
    print("  HW-WFC Phase 2: Scaling Benchmark")
    print("=" * 78)

    specs = [
        HardwareSpec.from_json(SPECS_DIR / "a100.json"),
        HardwareSpec.from_json(SPECS_DIR / "h100.json"),
    ]

    models = [
        ("GPT-2 Small (72L)", gpt2_small),
        ("BERT-Base (84L)", bert_base),
    ]

    results = []

    for spec in specs:
        print(f"\n{'─' * 78}")
        print(f"  GPU: {spec.name}")
        print(f"  SRAM: {spec.sram_bytes // 1024}KB, "
              f"DRAM BW: {spec.dram_bandwidth_gbps:.0f} GB/s, "
              f"Compute: {spec.compute_gflops / 1000:.0f} TFLOPS")
        print(f"{'─' * 78}")

        for model_name, model_fn in models:
            nodes = model_fn()
            r = run_benchmark(model_name, nodes, spec)
            results.append(r)

            print(f"\n  [{model_name}]")
            print(f"    Layers: {r['layers']}")
            print(f"    Initial states: {r['initial_states']:,}")
            print(f"    Status: {'OK' if r['success'] else 'FAIL'}")
            print(f"    Time: {r['time_ms']:.1f} ms")
            print(f"    Backtracks: {r['backtracks']}")
            print(f"    Unique states: {r['unique_states']}/{r['layers']}")
            print(f"    Total score: {r['total_score']:.2f}")
            print(f"    Threshold: {r['threshold']:.2f}")

            # 레이어 타입별 대표 선택 표시 (첫 블록만)
            print(f"    Block 0 selections:")
            for n in nodes[:7]:  # 첫 블록
                if n.collapsed_state:
                    s = n.collapsed_state
                    short_name = n.name.replace("B0_", "")
                    print(f"      {short_name:12s} → "
                          f"tile={s.tile_m}x{s.tile_n}, "
                          f"{s.layout.name}, {s.location.name}")

    # ── 비교 표 ──
    print(f"\n{'=' * 78}")
    print("  Summary")
    print(f"{'=' * 78}")
    print(f"\n  {'Model':<22s} {'GPU':<20s} {'Layers':>6s} {'States':>8s} "
          f"{'Time':>8s} {'Unique':>7s} {'Score':>8s}")
    print(f"  {'─' * 80}")
    for r in results:
        gpu_short = "A100" if "A100" in r["spec"] else "H100"
        print(f"  {r['model']:<22s} {gpu_short:<20s} {r['layers']:>6d} "
              f"{r['initial_states']:>8,} {r['time_ms']:>7.1f}ms "
              f"{r['unique_states']:>4d}/{r['layers']:<3d} "
              f"{r['total_score']:>8.2f}")

    # ── Grid Search 비교 (추정) ──
    print(f"\n  Grid Search 비교 (조합 수):")
    for r in results:
        per_layer = r["initial_states"] // r["layers"]
        combinations = per_layer ** r["layers"]
        gpu_short = "A100" if "A100" in r["spec"] else "H100"
        print(f"    {r['model']:<22s} {gpu_short}: "
              f"{per_layer}^{r['layers']} = {combinations:.2e} 조합 "
              f"vs WFC {r['time_ms']:.1f}ms")


if __name__ == "__main__":
    main()
