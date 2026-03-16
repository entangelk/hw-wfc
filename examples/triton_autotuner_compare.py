"""HW-WFC Phase 4: HW-WFC 스케줄 vs Triton Autotuner 성능 비교.

동일한 MatMul 워크로드에 대해:
  1. HW-WFC가 선택한 타일 설정의 실행 시간
  2. Triton autotuner가 탐색한 최적 설정의 실행 시간
을 비교하여 WFC 스케줄의 품질을 정량 평가한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl

from src.state import LayerNode, LayerType, generate_default_candidates
from src.constraint import HardwareSpec
from src.scheduler import HWWFCScheduler
from src.codegen import extract_all_configs


# ─── Fixed-config 커널 (HW-WFC 스케줄 결과 사용) ────────────────


@triton.jit
def matmul_fixed(
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


# ─── Autotuned 커널 (Triton이 최적 설정 탐색) ──────────────────


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_autotuned(
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


# ─── 벤치마크 ────────────────────────────────────────────────


def bench_matmul_fixed(M, N, K, cfg, warmup=10, rep=50):
    """HW-WFC config로 고정 실행."""
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    grid = (triton.cdiv(M, cfg.tile_m), triton.cdiv(N, cfg.tile_n))

    def run():
        matmul_fixed[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=cfg.tile_m, BLOCK_N=cfg.tile_n, BLOCK_K=cfg.tile_k,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )

    ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
    tflops = 2 * M * N * K / ms * 1e-9
    return ms, tflops


def bench_matmul_autotuned(M, N, K, warmup=10, rep=50):
    """Triton autotuner로 최적 config 탐색 후 실행."""
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    def run():
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        matmul_autotuned[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )

    ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
    tflops = 2 * M * N * K / ms * 1e-9

    # autotuner가 선택한 config 추출
    best = matmul_autotuned.best_config
    return ms, tflops, best


# ─── 메인 ────────────────────────────────────────────────────


WORKLOADS = [
    {"name": "Small (256x512x256)", "M": 256, "N": 512, "K": 256},
    {"name": "Medium (1024x1024x1024)", "M": 1024, "N": 1024, "K": 1024},
    {"name": "Large (2048x2048x2048)", "M": 2048, "N": 2048, "K": 2048},
    {"name": "GPT-2 FFN (768x3072x768)", "M": 768, "N": 3072, "K": 768},
    {"name": "BERT QKV (512x768x768)", "M": 512, "N": 768, "K": 768},
]


def main():
    print("=" * 70)
    print("  HW-WFC Schedule vs Triton Autotuner Comparison")
    print("=" * 70)
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")

    # RTX 3060 스펙으로 스케줄링
    spec_path = Path(__file__).resolve().parent.parent / "specs" / "rtx3060.json"
    spec = HardwareSpec.from_json(spec_path)
    print(f"[Spec] {spec.name} (SRAM {spec.sram_bytes // 1024}KB)")

    print(f"\n{'─' * 70}")
    print(f"  {'Workload':<30} {'WFC (ms)':>10} {'Auto (ms)':>10} "
          f"{'WFC TFLOPS':>11} {'Auto TFLOPS':>12} {'Ratio':>7}")
    print(f"{'─' * 70}")

    results = []

    for wl in WORKLOADS:
        M, N, K = wl["M"], wl["N"], wl["K"]

        # HW-WFC 스케줄링
        nodes = [
            LayerNode(
                name="MatMul",
                layer_type=LayerType.LINEAR,
                dims={"M": M, "N": N, "K": K},
                candidates=generate_default_candidates(),
            ),
        ]
        scheduler = HWWFCScheduler(spec, penalty_threshold="auto")
        sched_result = scheduler.schedule(nodes)
        configs = extract_all_configs(nodes)
        cfg = configs[0]

        # 벤치마크
        wfc_ms, wfc_tflops = bench_matmul_fixed(M, N, K, cfg)
        auto_ms, auto_tflops, auto_best = bench_matmul_autotuned(M, N, K)

        ratio = wfc_tflops / auto_tflops

        print(f"  {wl['name']:<30} {wfc_ms:>9.3f}  {auto_ms:>9.3f}  "
              f"{wfc_tflops:>10.2f}  {auto_tflops:>11.2f}  {ratio:>6.1%}")

        results.append({
            "name": wl["name"],
            "wfc_tile": f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}",
            "auto_tile": (f"{auto_best.kwargs['BLOCK_M']}x"
                          f"{auto_best.kwargs['BLOCK_N']}x"
                          f"{auto_best.kwargs['BLOCK_K']}"),
            "wfc_ms": wfc_ms,
            "auto_ms": auto_ms,
            "ratio": ratio,
        })

    # 상세 config 비교
    print(f"\n{'─' * 70}")
    print(f"  {'Workload':<30} {'WFC Tile':>15} {'Autotuner Tile':>15}")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['name']:<30} {r['wfc_tile']:>15} {r['auto_tile']:>15}")

    avg_ratio = sum(r["ratio"] for r in results) / len(results)
    print(f"\n{'=' * 70}")
    print(f"  Average WFC/Autotuner ratio: {avg_ratio:.1%}")
    print(f"  (100% = same, >100% = WFC faster, <100% = Autotuner faster)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
