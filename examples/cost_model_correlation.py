"""HW-WFC Cost Model Correlation Experiment.

Cost model 점수 순위가 실제 GPU 실행 시간 순위와 일치하는지 측정한다.
Spearman rank correlation으로 정량 평가.

실행: python examples/cost_model_correlation.py
요구사항: RTX 3060 + torch + triton + scipy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl
from scipy.stats import spearmanr

from src.state import (
    LayerNode, LayerType, HWState,
    MemoryLayout, ComputeLocation, generate_default_candidates,
)
from src.constraint import HardwareSpec, apply_hard_constraints
from src.cost_model import compute_score, roofline_score, layout_affinity, cache_efficiency
from src.codegen import _estimate_tile_k, _estimate_num_warps, _estimate_num_stages


# ─── Triton Kernels ───────────────────────────────────────────


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c = acc.to(tl.float16)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    row_max = -float("inf")
    for start in range(0, N, BLOCK_N):
        cols = start + offs_n
        mask = cols < N
        x = tl.load(X_ptr + row * stride_m + cols * stride_n, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    row_sum = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + offs_n
        mask = cols < N
        x = tl.load(X_ptr + row * stride_m + cols * stride_n, mask=mask, other=-float("inf"))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    for start in range(0, N, BLOCK_N):
        cols = start + offs_n
        mask = cols < N
        x = tl.load(X_ptr + row * stride_m + cols * stride_n, mask=mask, other=-float("inf"))
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + row * stride_m + cols * stride_n, y, mask=mask)


# ─── Candidate Enumeration ────────────────────────────────────


def enumerate_matmul_configs(dims, spec):
    """Hard constraint를 통과한 MatMul 타일 설정을 열거하고 점수를 매긴다.

    Triton 실행은 SRAM + ROW_MAJOR만 가능하므로 해당 조합만 반환.
    """
    node = LayerNode(
        name="MatMul", layer_type=LayerType.LINEAR,
        dims=dims, candidates=generate_default_candidates(),
    )
    apply_hard_constraints(node, spec)

    # SRAM + ROW_MAJOR만 필터, (tile_m, tile_n)으로 중복 제거
    seen = set()
    configs = []
    for s in node.candidates:
        if s.location != ComputeLocation.SRAM or s.layout != MemoryLayout.ROW_MAJOR:
            continue
        key = (s.tile_m, s.tile_n)
        if key in seen:
            continue
        seen.add(key)

        score = compute_score(node, s, spec)
        r = roofline_score(node, s, spec)
        a = layout_affinity(node, s)
        c = cache_efficiency(node, s, spec)

        tile_k_node = LayerNode(name="tmp", layer_type=LayerType.LINEAR, dims=dims)
        tile_k = _estimate_tile_k(tile_k_node)

        configs.append({
            "tile_m": s.tile_m, "tile_n": s.tile_n, "tile_k": tile_k,
            "num_warps": _estimate_num_warps(s.tile_m, s.tile_n),
            "num_stages": _estimate_num_stages(s.tile_m, s.tile_n),
            "score": score,
            "roofline": r, "affinity": a, "cache": c,
        })

    configs.sort(key=lambda x: x["score"], reverse=True)
    return configs


def enumerate_softmax_configs(dims, spec):
    """Hard constraint를 통과한 Softmax 타일 설정을 열거하고 점수를 매긴다."""
    node = LayerNode(
        name="Softmax", layer_type=LayerType.SOFTMAX,
        dims=dims, candidates=generate_default_candidates(),
    )
    apply_hard_constraints(node, spec)

    seen = set()
    configs = []
    for s in node.candidates:
        if s.location != ComputeLocation.SRAM or s.layout != MemoryLayout.ROW_MAJOR:
            continue
        # Softmax 커널은 BLOCK_N만 사용
        key = s.tile_n
        if key in seen:
            continue
        seen.add(key)

        score = compute_score(node, s, spec)
        r = roofline_score(node, s, spec)
        a = layout_affinity(node, s)
        c = cache_efficiency(node, s, spec)

        configs.append({
            "tile_n": s.tile_n,
            "score": score,
            "roofline": r, "affinity": a, "cache": c,
        })

    configs.sort(key=lambda x: x["score"], reverse=True)
    return configs


# ─── Timing Functions ─────────────────────────────────────────


def time_matmul(M, N, K, cfg, A, B):
    """MatMul 타일 설정의 실행 시간 측정. 동일한 입력 텐서 사용."""
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, cfg["tile_m"]), triton.cdiv(N, cfg["tile_n"]))

    def run():
        matmul_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=cfg["tile_m"], BLOCK_N=cfg["tile_n"], BLOCK_K=cfg["tile_k"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

    # Correctness check
    run()
    C_ref = torch.matmul(A.float(), B.float()).half()
    rel_err = (C.float() - C_ref.float()).abs().max().item() / (C_ref.float().abs().mean().item() + 1e-6)
    correct = rel_err < 0.05

    ms = triton.testing.do_bench(run, warmup=25, rep=100)
    return ms, correct


def time_softmax(M, N, cfg, X):
    """Softmax 타일 설정의 실행 시간 측정. 동일한 입력 텐서 사용."""
    Y = torch.empty_like(X)

    def run():
        softmax_kernel[(M,)](
            X, Y, M, N,
            X.stride(0), X.stride(1),
            BLOCK_N=cfg["tile_n"],
        )

    # Correctness check
    run()
    Y_ref = torch.softmax(X, dim=-1)
    max_diff = (Y - Y_ref).abs().max().item()
    correct = max_diff < 1e-5

    ms = triton.testing.do_bench(run, warmup=25, rep=100)
    return ms, correct


# ─── Correlation Measurement ──────────────────────────────────


def measure_matmul_correlation(workload, spec):
    """단일 MatMul workload에 대한 cost model ↔ 실행 시간 상관관계 측정."""
    M, N, K = workload["M"], workload["N"], workload["K"]
    configs = enumerate_matmul_configs({"M": M, "N": N, "K": K}, spec)

    if len(configs) < 3:
        return None  # 상관관계 계산에 최소 3개 필요

    # 동일 입력 텐서
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    for cfg in configs:
        ms, correct = time_matmul(M, N, K, cfg, A, B)
        cfg["time_ms"] = ms
        cfg["correct"] = correct
        cfg["speed"] = 1.0 / ms  # 높을수록 빠름

    scores = [c["score"] for c in configs]
    speeds = [c["speed"] for c in configs]
    rho, p_value = spearmanr(scores, speeds)

    return {"configs": configs, "rho": rho, "p_value": p_value, "n": len(configs)}


def measure_softmax_correlation(workload, spec):
    """단일 Softmax workload에 대한 cost model ↔ 실행 시간 상관관계 측정."""
    M, N = workload["M"], workload["N"]
    configs = enumerate_softmax_configs({"M": M, "N": N}, spec)

    if len(configs) < 3:
        return None

    torch.manual_seed(42)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32)

    for cfg in configs:
        ms, correct = time_softmax(M, N, cfg, X)
        cfg["time_ms"] = ms
        cfg["correct"] = correct
        cfg["speed"] = 1.0 / ms

    scores = [c["score"] for c in configs]
    speeds = [c["speed"] for c in configs]
    rho, p_value = spearmanr(scores, speeds)

    return {"configs": configs, "rho": rho, "p_value": p_value, "n": len(configs)}


# ─── Workload Definitions ────────────────────────────────────


MATMUL_WORKLOADS = [
    {"name": "Small (256x512x256)",     "M": 256,  "N": 512,  "K": 256},
    {"name": "Medium (1024x1024x1024)", "M": 1024, "N": 1024, "K": 1024},
    {"name": "Large (2048x2048x2048)",  "M": 2048, "N": 2048, "K": 2048},
    {"name": "GPT-2 FFN (768x3072x768)","M": 768,  "N": 3072, "K": 768},
    {"name": "BERT QKV (512x768x768)",  "M": 512,  "N": 768,  "K": 768},
]

SOFTMAX_WORKLOADS = [
    {"name": "Small (256x512)",   "M": 256,  "N": 512},
    {"name": "Medium (512x1024)", "M": 512,  "N": 1024},
    {"name": "Large (1024x2048)", "M": 1024, "N": 2048},
    {"name": "Attn (512x512)",    "M": 512,  "N": 512},
]


# ─── Main ─────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  HW-WFC Cost Model Correlation Experiment")
    print("=" * 70)
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[CUDA] {torch.version.cuda}")

    spec_path = Path(__file__).resolve().parent.parent / "specs" / "rtx3060.json"
    spec = HardwareSpec.from_json(spec_path)
    print(f"[Spec] {spec.name} (SRAM {spec.sram_bytes // 1024}KB, align {spec.alignment_bytes}B)")

    all_results = []

    # ── MatMul ──
    print(f"\n{'═' * 70}")
    print("  MatMul (LINEAR) Correlation")
    print(f"{'═' * 70}")

    for wl in MATMUL_WORKLOADS:
        print(f"\n  [{wl['name']}]")
        result = measure_matmul_correlation(wl, spec)
        if result is None:
            print("    후보 부족 (< 3개), 건너뜀")
            continue

        # 상세 출력: score rank vs time rank
        configs = result["configs"]
        # score 내림차순 정렬 (이미 정렬됨)
        for i, cfg in enumerate(configs):
            tile_str = f"{cfg['tile_m']:>3}x{cfg['tile_n']:<3}"
            correct_str = "OK" if cfg["correct"] else "FAIL"
            print(f"    {tile_str}  score={cfg['score']:.4f} "
                  f"(R={cfg['roofline']:.3f} A={cfg['affinity']:.3f} C={cfg['cache']:.3f})  "
                  f"time={cfg['time_ms']:.4f}ms  {correct_str}")

        rho_str = f"{result['rho']:+.3f}" if result['rho'] is not None else "N/A"
        sig_str = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"    → Spearman ρ = {rho_str} (p={result['p_value']:.4f}) {sig_str}  [n={result['n']}]")

        all_results.append({"type": "MatMul", "workload": wl["name"], **result})

    # ── Softmax ──
    print(f"\n{'═' * 70}")
    print("  Softmax Correlation")
    print(f"{'═' * 70}")

    for wl in SOFTMAX_WORKLOADS:
        print(f"\n  [{wl['name']}]")
        result = measure_softmax_correlation(wl, spec)
        if result is None:
            print("    후보 부족 (< 3개), 건너뜀")
            continue

        configs = result["configs"]
        for cfg in configs:
            correct_str = "OK" if cfg["correct"] else "FAIL"
            print(f"    BLOCK_N={cfg['tile_n']:>3}  score={cfg['score']:.4f} "
                  f"(R={cfg['roofline']:.3f} A={cfg['affinity']:.3f} C={cfg['cache']:.3f})  "
                  f"time={cfg['time_ms']:.4f}ms  {correct_str}")

        rho_str = f"{result['rho']:+.3f}" if result['rho'] is not None else "N/A"
        sig_str = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"    → Spearman ρ = {rho_str} (p={result['p_value']:.4f}) {sig_str}  [n={result['n']}]")

        all_results.append({"type": "Softmax", "workload": wl["name"], **result})

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("  Summary")
    print(f"{'═' * 70}")
    print(f"\n  {'Type':<10} {'Workload':<30} {'ρ':>6} {'p':>8} {'n':>3} {'Sig':>4}")
    print(f"  {'─'*10} {'─'*30} {'─'*6} {'─'*8} {'─'*3} {'─'*4}")

    rho_values = []
    for r in all_results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        rho_str = f"{r['rho']:+.3f}" if r['rho'] is not None else "  N/A"
        print(f"  {r['type']:<10} {r['workload']:<30} {rho_str:>6} {r['p_value']:>8.4f} {r['n']:>3} {sig:>4}")
        if r["rho"] is not None:
            rho_values.append(r["rho"])

    if rho_values:
        avg_rho = sum(rho_values) / len(rho_values)
        print(f"\n  Average Spearman ρ = {avg_rho:+.3f} (across {len(rho_values)} workloads)")
        print(f"  ρ > 0: cost model 점수가 높을수록 실제로 빠름")
        print(f"  ρ ≈ 0: 점수와 실행 시간 사이에 상관관계 없음")
        print(f"  ρ < 0: 점수가 높을수록 오히려 느림")

    # ── Interpretation ──
    print(f"\n{'═' * 70}")
    print("  Interpretation & Limitations")
    print(f"{'═' * 70}")
    print("""
  이 실험은 HW-WFC cost model이 실제 GPU 성능의 유효한 proxy인지 검증한다.

  실험 조건:
  - SRAM + ROW_MAJOR 후보만 측정 (Triton은 layout/location을 구분하지 않음)
  - 동일 입력 텐서, correctness 확인 후 timing
  - triton.testing.do_bench (warmup=25, rep=100)

  한계:
  1. Layout/location은 Triton 실행에서 구분 불가 → cost model의 해당 차원은 미검증
  2. num_warps/num_stages는 codegen 휴리스틱으로 파생 → 실제 최적값과 다를 수 있음
  3. RTX 3060 단일 GPU 결과 → 다른 아키텍처에서는 다를 수 있음
  4. Roofline score가 포화되면 cache_efficiency만으로 순위가 결정됨
""")


if __name__ == "__main__":
    main()
