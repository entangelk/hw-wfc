<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# HW-WFC

**Hardware Wave Function Collapse** — constraint-based collapsing search applied to AI hardware compiler auto-scheduling.

A research prototype that applies the Wave Function Collapse (WFC) algorithm to determine per-layer tile size, memory layout, and compute location for AI model inference on GPU hardware.

**Status**: Research complete (v2.9). The algorithm is correct and the cost model shows directional correlation with real GPU performance, but no advantage over existing methods has been demonstrated. See [Research Results](docs/result.md) for the full analysis.

**Documentation**: [Research Results](docs/result.md) | [Architecture Guide](docs/architecture_guide.md) | [Concept](docs/concept.md)

---

## Algorithm

![HW-WFC Collapse Animation](assets/collapse_animation.gif)

1. **Superposition** — Each layer holds all possible HW states (tile size × layout × location)
2. **Hard Constraints** — Eliminate physically impossible candidates (SRAM capacity, working memory overflow, alignment)
3. **Bottleneck-First Collapse** — Highest-FLOPs layer collapses first
4. **Constraint Propagation** — Transition-cost-based pruning propagates to neighbors
5. **Entropy-Ordered Collapse** — Remaining layers collapse by Shannon entropy
6. **Backtracking** — Snapshot-based rollback on contradiction

---

## Results Summary

6-layer Transformer Attention Block on `stress_gpu` (12KB SRAM).

| Metric | Value |
|--------|-------|
| WFC vs Exact DP (Viterbi) | **100% match** — identical states |
| WFC time | ~2-3ms |
| Exact DP time | ~7-10ms |
| Search space | ~2 billion combinations |
| Cost model ↔ GPU correlation | **Avg Spearman ρ = +0.52** (Softmax: 1.0, MatMul large: +0.72) |

### What worked

- WFC correctly navigates the constrained search space and matches the exact DP optimum on this benchmark.
- The cost model shows positive rank correlation with actual Triton kernel execution time — Softmax achieves perfect agreement (ρ = 1.0).
- Hard constraints (SRAM, alignment, working memory) drive the most meaningful scheduling differentiation.

### What did not work

- **No advantage over existing methods.** WFC does not outperform exact DP, Triton autotuner, or any well-optimized baseline.
- **Cost model saturates for MatMul.** Roofline scores are all 1.0 → discrimination comes solely from the secondary `cache_efficiency` component.
- **Virtual spec dependency.** The key scenario (12KB SRAM) does not exist on real GPUs. On A100 (192KB), the problem is trivial.
- **Naive exhaustive speedup (~3500x) is misleading.** The baseline is unoptimized Python; exact DP solves the same problem in ~7-10ms.

Full analysis: **[Research Results](docs/result.md)**

---

## WFC vs Exact DP (12KB SRAM)

```
Layer       WFC heuristic       Exact DP
QKV_Proj    32x32 ROW SRAM      32x32 ROW SRAM
QK_MatMul   32x32 ROW SRAM      32x32 ROW SRAM
Softmax     16x16 ROW SRAM      16x16 ROW SRAM   ← working memory constraint
AV_MatMul   32x32 ROW SRAM      32x32 ROW SRAM
Out_Proj    32x32 ROW SRAM      32x32 ROW SRAM
LayerNorm   32x32 ROW SRAM      32x32 ROW SRAM
```

Softmax working memory: 4x × 4KB = 16KB > 12KB SRAM → `32x32` eliminated by hard constraint. Both algorithms select `16x16`.

---

## Cost Model ↔ GPU Correlation

Spearman rank correlation between cost model scores and actual Triton execution time (RTX 3060, CUDA 12.8).

| Layer | Workload | ρ | Sig |
|-------|----------|---|-----|
| Softmax | Medium–Large | **+1.00** | *** |
| MatMul | Large (2048²) | **+0.72** | * |
| MatMul | GPT-2 FFN | +0.66 | ns |
| MatMul | Small (256x512) | -0.69 | ns |

Average ρ = +0.52. Directionally useful but not precise. The roofline component saturates for MatMul, leaving `cache_efficiency` as the only discriminator.

Reproduced by: `python examples/cost_model_correlation.py`

---

## Visualizations

Regenerated via `python tools/generate_all_visuals.py`.

| | |
|---|---|
| ![Score Distribution](assets/score_distribution.png) | ![Propagation Sweep](assets/propagation_sweep.png) |
| Cost model score decomposition (Roofline + Affinity + Cache) | Candidate removal rate by threshold (t=1.5 → 43%) |
| ![Copier Problem](assets/copier_problem.png) | ![SRAM Comparison](assets/sram_comparison.png) |
| 64KB: all same state. 12KB: working memory forces differentiation | SRAM size impact on scheduling decisions |

---

## Architecture

```
LayerNode ──▶ HardConstraint ──▶ CostModel ──▶ CollapseEngine ──▶ Propagation ──▶ Scheduler
(candidates)  (SRAM, align,     (roofline +    (entropy,         (layout +       (bottleneck
 HWState[]     working memory)   affinity +      backtrack)        tile_shape +     first,
                                 cache)                            location)        bidirect)
```

### Cost Model

Three-component additive scoring:
- **Roofline Score** — Data reuse from tiling (saturates at 1.0 for MatMul on large SRAM)
- **Layout Affinity** — Per-layer-type layout preference
- **Cache Efficiency** — Gaussian over SRAM utilization with working memory multiplier

### Hardware Specs

| Spec | SRAM | Role |
|------|------|------|
| `stress_gpu.json` | 12KB | Primary benchmark — forces differentiation |
| `toy_gpu.json` | 64KB | Baseline — exposes copier problem |
| `a100.json` / `h100.json` | 192–256KB | Scaling validation |
| `rtx3060.json` | 100KB | GPU execution verification |

---

## Safety Checks (5/5 Pass)

| # | Check | Criteria | Result |
|---|-------|----------|--------|
| 1 | Hard Constraint | No false positives/negatives | Pass |
| 2 | Cost Model discrimination | Score spread > 0.01 | 1.19 |
| 3 | Propagation | Removal > 25% at t=1.5 | 43% |
| 4 | Backtracking | Triggers on contradiction | Verified |
| 5 | No copier problem | Heterogeneous layers differentiate | WFC = DP |

Run: `python tests/test_safety_checks.py`

---

## Limitations

- **No demonstrated advantage** over exact DP or autotuner on this problem.
- **Cost model precision**: ρ = +0.52 average. Directional, not reliable for production scheduling.
- **Virtual spec dependency**: Key results require artificially constrained SRAM (12KB). Real GPUs (192–256KB) make the problem trivial.
- **Roofline saturation**: MatMul scores converge, limiting workload-specific tile recommendation.
- **Single-kernel scope**: No multi-kernel fusion, operator scheduling, or cross-layer memory planning.

## Path Forward

The primary bottleneck is cost model precision. If rank correlation can be improved from ρ = +0.52 to ρ ≥ +0.8 through GPU profiling data calibration, the approach could predict competitive tile configurations without hardware execution — enabling cross-compilation, design-space exploration, and autotuner warm-starting.

This calibration requires real GPU profiling data across diverse workloads, which is beyond the scope of this software experiment.

---

## Quick Start

```bash
python examples/attention_block.py                # Schedule an Attention Block
python examples/attention_exhaustive_benchmark.py  # Reproduce key results
python tests/test_safety_checks.py                 # Run safety checks
python examples/cost_model_correlation.py          # Cost model ↔ GPU correlation
python examples/triton_verify.py                   # Triton kernel correctness (GPU required)
```

## Project Structure

```
hw-wfc/
├── src/                    # Core algorithm
│   ├── state.py            # HWState, LayerNode, superposition
│   ├── constraint.py       # Hard/soft constraints, propagation
│   ├── cost_model.py       # Roofline + affinity + cache
│   ├── collapse.py         # Entropy-based collapse + backtracking
│   └── scheduler.py        # Main pipeline
├── specs/                  # Hardware specifications (JSON)
├── examples/               # Benchmarks and experiments
├── tests/                  # Safety checks
├── docs/
│   ├── result.md / .ko.md              # Research results (EN/KO)
│   ├── concept.md / .ko.md             # Algorithm concept (EN/KO)
│   ├── architecture_guide.md / .ko.md  # Design rationale (EN/KO)
│   └── daily_logs/                     # Work logs
└── HANDOFF.md              # Project status and future work
```

## Requirements

- Python 3.10+
- scipy (correlation analysis)
- Pillow (visualization)
- PyTorch + Triton (GPU verification, `pip install torch triton`)

## License

Research prototype. Not for production use.
