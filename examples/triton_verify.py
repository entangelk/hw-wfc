"""HW-WFC Phase 4: Triton 커널 실행 검증.

RTX 3060에서 codegen이 생성한 Triton 커널을 실제 실행하고
PyTorch 참조 결과와 비교하여 correctness를 검증한다.
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


# ─── Triton Kernels (codegen 출력과 동일 구조) ─────────────────


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
def relu_kernel(
    X_ptr, Y_ptr, N_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elements
    x = tl.load(X_ptr + offs, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


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


# ─── 검증 함수 ──────────────────────────────────────────────


def verify_matmul(cfg, dims: dict) -> dict:
    """MatMul 커널 검증: Triton vs torch.matmul"""
    M, N, K = dims["M"], dims["N"], dims["K"]
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C_triton = torch.empty(M, N, device="cuda", dtype=torch.float16)

    grid = (triton.cdiv(M, cfg.tile_m), triton.cdiv(N, cfg.tile_n))
    matmul_kernel[grid](
        A, B, C_triton,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_triton.stride(0), C_triton.stride(1),
        BLOCK_M=cfg.tile_m, BLOCK_N=cfg.tile_n, BLOCK_K=cfg.tile_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )

    C_ref = torch.matmul(A.float(), B.float()).half()
    max_diff = (C_triton.float() - C_ref.float()).abs().max().item()
    # fp16 matmul tolerance: relative to magnitude
    mean_val = C_ref.float().abs().mean().item()
    rel_err = max_diff / (mean_val + 1e-6)
    passed = rel_err < 0.05  # 5% relative tolerance for fp16

    return {"passed": passed, "max_diff": max_diff, "rel_err": rel_err}


def verify_relu(cfg, dims: dict) -> dict:
    """ReLU 커널 검증: Triton vs torch.relu"""
    n_elements = 1
    for v in dims.values():
        n_elements *= v
    X = torch.randn(n_elements, device="cuda", dtype=torch.float16)
    Y_triton = torch.empty_like(X)

    block_size = cfg.tile_m * cfg.tile_n
    grid = (triton.cdiv(n_elements, block_size),)
    relu_kernel[grid](X, Y_triton, n_elements, BLOCK_SIZE=block_size)

    Y_ref = torch.relu(X)
    max_diff = (Y_triton - Y_ref).abs().max().item()
    passed = max_diff == 0.0

    return {"passed": passed, "max_diff": max_diff}


def verify_softmax(cfg, dims: dict) -> dict:
    """Softmax 커널 검증: Triton vs torch.softmax"""
    M = dims.get("M", 1)
    N = dims.get("N", 512)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32)
    Y_triton = torch.empty_like(X)

    softmax_kernel[(M,)](
        X, Y_triton, M, N,
        X.stride(0), X.stride(1),
        BLOCK_N=cfg.tile_n,
    )

    Y_ref = torch.softmax(X, dim=-1)
    max_diff = (Y_triton - Y_ref).abs().max().item()
    passed = max_diff < 1e-5

    return {"passed": passed, "max_diff": max_diff}


VERIFIERS = {
    LayerType.LINEAR: verify_matmul,
    LayerType.CONV2D: verify_matmul,  # im2col → matmul
    LayerType.RELU: verify_relu,
    LayerType.SOFTMAX: verify_softmax,
}


# ─── 메인 ────────────────────────────────────────────────────


def build_test_model(spec: HardwareSpec) -> list[LayerNode]:
    """Linear → ReLU → Softmax → Linear 테스트 모델."""
    return [
        LayerNode(
            name="Linear_0",
            layer_type=LayerType.LINEAR,
            dims={"M": 256, "N": 512, "K": 256},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="ReLU_1",
            layer_type=LayerType.RELU,
            dims={"M": 256, "N": 512},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Softmax_2",
            layer_type=LayerType.SOFTMAX,
            dims={"M": 256, "N": 512},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Linear_3",
            layer_type=LayerType.LINEAR,
            dims={"M": 256, "N": 128, "K": 512},
            candidates=generate_default_candidates(),
        ),
    ]


def main():
    print("=" * 60)
    print("  HW-WFC Phase 4: Triton Kernel Verification")
    print("=" * 60)

    # GPU 확인
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")

    # RTX 3060 스펙 로드
    spec_path = Path(__file__).resolve().parent.parent / "specs" / "rtx3060.json"
    spec = HardwareSpec.from_json(spec_path)
    print(f"\n[Spec] {spec.name}")
    print(f"  SRAM: {spec.sram_bytes // 1024}KB, Alignment: {spec.alignment_bytes}B")

    # 스케줄링
    nodes = build_test_model(spec)
    scheduler = HWWFCScheduler(spec, penalty_threshold="auto")
    result = scheduler.schedule(nodes)

    if not result.success:
        print("\n[ERROR] Scheduling failed!")
        return

    # codegen config 추출
    configs = extract_all_configs(nodes)
    print(f"\n[Schedule] {len(configs)} layers scheduled")
    for cfg in configs:
        print(f"  {cfg.layer_name}: {cfg.tile_m}x{cfg.tile_n} "
              f"(warps={cfg.num_warps}, stages={cfg.num_stages})")

    # 각 커널 검증
    print(f"\n{'─' * 60}")
    print("  Kernel Correctness Verification")
    print(f"{'─' * 60}")

    all_passed = True
    for cfg, node in zip(configs, nodes):
        verifier = VERIFIERS.get(cfg.layer_type)
        if verifier is None:
            print(f"  {cfg.layer_name}: SKIP (no verifier for {cfg.layer_type.name})")
            continue

        result = verifier(cfg, node.dims)
        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            all_passed = False
        detail = ", ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in result.items() if k != "passed")
        print(f"  {cfg.layer_name} ({cfg.layer_type.name}): {status} [{detail}]")

    print(f"\n{'=' * 60}")
    print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
