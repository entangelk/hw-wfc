<p align="center">
  <a href="./result.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./result.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>

# HW-WFC Research Results

> **Version**: v2.9 (2026-03-21)
> **Objective**: Evaluate the feasibility of applying the Wave Function Collapse (WFC) algorithm to AI hardware compiler auto-scheduling, within a controlled software environment.

---

## 1. Research Question

**Can a constraint-based collapsing search (WFC) produce competitive hardware scheduling decisions without executing kernels on real hardware?**

The hypothesis: if a cost model can accurately rank tile configurations by expected performance, then WFC's constraint propagation and entropy-based collapse can efficiently navigate the combinatorial search space to find near-optimal schedules.

---

## 2. What Worked

### 2.1 Algorithm Correctness

The WFC heuristic produces results identical to exact dynamic programming (Viterbi algorithm) on the benchmark problem.

| Metric | Value |
|--------|-------|
| WFC vs Exact DP quality | **100%** (identical states on both `stress_gpu` and `toy_gpu`) |
| WFC scheduling time | ~2-3ms |
| Exact DP time | ~7-10ms |
| Search space | 33^1 x 36^5 ~ 2 billion combinations |

Both algorithms choose `16x16` for Softmax (working memory hard constraint) and `32x32` for all other layers on 12KB SRAM. The WFC heuristic is greedy and does not guarantee optimality in general, but on this chain-structured benchmark it reaches the exact optimum.

### 2.2 Cost Model Correlation with Real GPU

Spearman rank correlation between cost model scores and actual Triton kernel execution times on RTX 3060:

| Layer Type | Workload | Spearman ρ | p-value | Sig |
|------------|----------|-----------|---------|-----|
| Softmax | Medium (512x1024) | **+1.00** | < 0.001 | *** |
| Softmax | Large (1024x2048) | **+1.00** | < 0.001 | *** |
| Softmax | Attn (512x512) | **+1.00** | < 0.001 | *** |
| MatMul | Large (2048x2048) | **+0.72** | < 0.05 | * |
| MatMul | GPT-2 FFN (768x3072) | +0.66 | 0.08 | ns |
| MatMul | Medium (1024x1024) | +0.60 | 0.10 | ns |
| MatMul | BERT QKV (512x768) | +0.40 | 0.33 | ns |
| MatMul | Small (256x512) | -0.69 | 0.07 | ns |
| Softmax | Small (256x512) | +0.50 | 0.67 | ns |

**Average ρ = +0.52** across 9 workloads.

Interpretation:
- **Softmax**: Perfect rank agreement. The cost model's memory traffic estimation correctly captures that larger `BLOCK_N` reduces row iteration passes.
- **MatMul (large workloads)**: Positive correlation. The model correctly predicts that larger tiles are generally faster due to better data reuse.
- **MatMul (small workloads)**: Measurement noise dominates at ~0.01ms execution time, producing inverted rankings.

Reproduced by: `python examples/cost_model_correlation.py` on RTX 3060, CUDA 12.8.

### 2.3 Hard Constraints

Physical constraints are enforced correctly and produce meaningful differentiation:

| Constraint | Mechanism | Effect (stress_gpu 12KB) |
|------------|-----------|--------------------------|
| SRAM capacity | `tile_bytes > sram_bytes` → eliminate | Removes large tiles from SRAM candidates |
| Alignment | `row_bytes % alignment != 0` → eliminate | Removes 8x8, 16x16 on 128B-aligned specs |
| Working memory | `tile_bytes × multiplier > sram_bytes` → eliminate | Softmax 32x32 (4x × 4KB = 16KB > 12KB) eliminated |

The working memory constraint is the key differentiator: it forces Softmax to use a smaller tile than compute-bound layers, producing genuine per-layer-type differentiation confirmed by both WFC and exact DP.

---

## 3. What Did Not Work

### 3.1 No Advantage Over Existing Methods

| Comparison | Result | Assessment |
|------------|--------|------------|
| WFC vs Exact DP | Same quality, WFC ~3-5x faster | Both are trivially fast (ms-scale). Speed difference is not meaningful. |
| WFC vs Triton Autotuner | Autotuner finds 5-20% faster configs on larger workloads | Autotuner wins because it measures actual execution time. |
| WFC vs Naive Exhaustive | ~3500x speedup | **Misleading.** See §3.2. |

The WFC algorithm does not outperform any well-implemented baseline on this problem.

### 3.2 The Naive Exhaustive Speedup Is Not Meaningful

The ~3500x speedup figure (WFC ~2ms vs naive exhaustive ~7s) is an artifact of implementation quality, not algorithmic superiority:

1. **Unoptimized Python iteration**: The naive baseline recomputes scores in a tight Python loop without vectorization, memoization, or early termination.
2. **No pruning**: All `8^6 = 262,144` combinations are evaluated exhaustively.
3. **Exact DP solves this in ~7-10ms**: The Viterbi algorithm finds the provably optimal solution for the same chain objective, making the naive exhaustive approach entirely redundant.

To make a fair speedup claim, the baseline must be optimized:
- **Branch-and-bound** with cost model upper bounds for pruning
- **Beam search** retaining top-k states per layer
- **Vectorized DP** with precomputed transition cost matrices (NumPy/C++)
- **C++/CUDA implementation** to eliminate Python overhead

Until these baselines exist, no speedup claim is valid.

### 3.3 Cost Model Saturation

The roofline score saturates at 1.0 for all MatMul tile configurations on GPUs with sufficient memory bandwidth. When this happens, the only discriminating factor is `cache_efficiency` — a secondary heuristic component — which limits the model's ability to recommend workload-specific tile sizes.

| Workload | Roofline scores | Discrimination source |
|----------|----------------|----------------------|
| MatMul (any size) | All 1.0 | cache_efficiency only |
| Softmax | Varies by BLOCK_N | memory traffic (correct) |

This is the primary reason the cost model's average correlation (ρ = +0.52) falls short of being a reliable predictor.

### 3.4 Virtual Spec Dependency

The key differentiation scenario (12KB SRAM) does not exist on any real GPU. On actual hardware:
- A100 (192KB SRAM): All tiles fit comfortably. No constraints activate. The problem is trivial.
- H100 (256KB SRAM): Same situation.

The algorithm demonstrates its value only under artificially tight constraints. Whether real-world scheduling problems produce similarly constrained scenarios (e.g., multi-tenant SRAM partitioning, fusion-expanded working sets) is an open question.

---

## 4. Experimental Conditions

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 3060 (GA106, SM86) |
| CUDA | 12.8 |
| Triton | OpenAI Triton (pip install) |
| dtype | float16 (MatMul), float32 (Softmax) |
| Benchmark method | `triton.testing.do_bench` (warmup=25, rep=100) |
| Correctness check | Before timing; rel_err < 5% (MatMul), max_diff < 1e-5 (Softmax) |
| Input tensors | Shared across all configs per workload; `torch.manual_seed(42)` |
| Virtual specs | `stress_gpu.json` (12KB SRAM, 64B align), `toy_gpu.json` (64KB SRAM, 128B align) |

### Measurement Limitations

- Only SRAM + ROW_MAJOR configurations are measured. Triton does not expose layout or compute location as tuning knobs.
- `num_warps` and `num_stages` are derived by a fixed heuristic, not optimized per configuration.
- Small workloads (~0.01ms) are noise-sensitive. Rankings on these workloads are unreliable.
- Results are from a single GPU model. Correlation may differ on other architectures.

---

## 5. Conclusions

### What This Research Demonstrates

1. **WFC is a valid search algorithm for constrained hardware scheduling.** It correctly navigates the search space, respects physical constraints, and matches exact DP on the benchmark problem.
2. **A physics-based cost model can serve as a directional proxy for GPU performance.** Softmax achieves perfect rank correlation (ρ = 1.0). MatMul achieves positive correlation on non-trivial workloads (ρ = +0.72).
3. **Hard constraints — not heuristic scoring — drive the most meaningful scheduling decisions.** The Softmax tile differentiation is entirely determined by the working memory constraint, not by score optimization.

### What This Research Does Not Demonstrate

1. **No advantage over existing methods.** The WFC heuristic does not outperform exact DP, Triton autotuner, or any well-optimized baseline.
2. **No evidence of practical value on real hardware.** The interesting constraint scenarios occur only on virtual specs with artificially small SRAM.
3. **Cost model is not precise enough for workload-specific recommendations.** Average ρ = +0.52 is directionally useful but insufficient for production scheduling decisions.

### Path to Practical Value

The research identifies one clear path: **cost model calibration**.

If the cost model's rank correlation can be improved from ρ = +0.52 to ρ ≥ +0.8 through GPU profiling data fitting, the WFC approach would offer a concrete advantage: **predicting good tile configurations without running kernels on hardware**. This would be valuable for:
- Cross-compilation targeting GPUs not physically available
- Rapid design-space exploration during hardware/compiler co-design
- Initial schedule generation before autotuner refinement

This calibration requires real GPU profiling data across diverse workloads, which is beyond the scope of this software-only experiment.

---

## 6. Reproduction

```bash
# Core benchmark (WFC vs exact DP)
python examples/attention_exhaustive_benchmark.py

# Cost model correlation with GPU
python examples/cost_model_correlation.py

# Safety checks (5/5)
python tests/test_safety_checks.py

# Triton kernel correctness
python examples/triton_verify.py
```

Requirements: Python 3.10+, scipy, PyTorch, Triton, CUDA-capable GPU.
