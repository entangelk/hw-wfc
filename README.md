<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# HW-WFC

**Hardware Wave Function Collapse** — AI hardware compiler auto-scheduling using constraint-based collapse.

A research prototype that applies the Wave Function Collapse (WFC) algorithm to AI hardware compiler auto-scheduling. It determines the optimal tile size, memory layout, and compute location for each layer through constraint-driven search.

> **New here?** Start with the **[Architecture Guide](docs/architecture_guide.md)** — it explains *why* every design decision was made. For the algorithm concept and motivation, see the **[Concept Document](docs/concept.md)**.

## How It Works

![HW-WFC Collapse Animation](assets/collapse_animation.gif)

1. **Superposition** — Each layer starts with all possible HW implementation states (tile size × layout × location) as candidates
2. **Hard Constraints** — Physically impossible candidates are pruned (SRAM capacity overflow, alignment violation)
3. **Bottleneck-First Collapse** — The layer with the highest FLOPs collapses to its optimal state first
4. **Constraint Propagation** — Transition-cost-based candidate pruning propagates from collapsed nodes to neighbors
5. **Entropy-Ordered Collapse** — Remaining nodes collapse in Shannon entropy order (most certain first)
6. **Backtracking** — Snapshot-based state restoration on contradiction

## Key Results

| Metric | Value |
|--------|-------|
| Search space reduction | 396 → 6 states (98.5%) |
| Match vs truncated exhaustive | 100% (transition-aware, top-8/layer) |
| Quality vs exact full DP | 98.3% on `stress_gpu`, 100% on `toy_gpu` |
| WFC scheduling time | ~3ms (6-layer attention block, `stress_gpu`) |
| Naive truncated exhaustive | 262K combinations, ~7-9s |
| Speedup vs naive truncated exhaustive | **~2400x** |

These numbers are reproduced by `python examples/attention_exhaustive_benchmark.py`.
Important: the `~2400x` figure is against a naive top-k exhaustive enumerator, not against the best exact solver for this chain objective.
The truncated exhaustive baseline keeps only the top 8 unary survivors per layer (`8^6 = 262,144`), so it can miss the true optimum once pairwise transition costs are considered.
The same script also reports a full exact dynamic-programming baseline over all hard-constrained candidates; on `stress_gpu`, the current greedy WFC heuristic reaches `98.3%` of that exact objective and does not match the full optimum.

### Current Heuristic Differentiation (12KB SRAM)

```
QKV_Proj   → 32x32  ROW_MAJOR  SRAM
QK_MatMul  → 32x32  ROW_MAJOR  SRAM
Softmax    → 16x16  ROW_MAJOR  SRAM  ← different in the current WFC run
AV_MatMul  → 32x32  ROW_MAJOR  SRAM
Out_Proj   → 32x32  ROW_MAJOR  SRAM
LayerNorm  → 32x32  ROW_MAJOR  SRAM
```

Under `stress_gpu`, the current greedy WFC run collapses Softmax to `16x16` while the other layers stay at `32x32`.
Important: working-memory overflow is currently modeled as a strong cache penalty in [src/cost_model.py](/mnt/d/devel/hw-wfc/src/cost_model.py), not as a hard elimination in [src/constraint.py](/mnt/d/devel/hw-wfc/src/constraint.py). So this differentiation is a heuristic result, not proof that `32x32` Softmax is physically impossible.

## Visualizations

All visualizations are regenerated via `python tools/generate_all_visuals.py`.
Updating visuals after code changes provides a visual diff against previous results.
History is auto-appended to [assets/VISUALS_LOG.md](assets/VISUALS_LOG.md).

### Cost Model Score Distribution

Per-layer-type candidate scores decomposed into Roofline (blue) + Affinity (orange) + Cache (green). Red indicates cache penalty.

![Score Distribution](assets/score_distribution.png)

### Constraint Propagation Sweep

Candidate removal rate as threshold varies from 0.3 to 3.0. Red line (25%) is the passing criterion. Current t=1.5 yields 43%.

![Propagation Sweep](assets/propagation_sweep.png)

### The Copier Problem (Failure → Fix)

With ample SRAM (64KB), all layers converge to the same state — the "copier problem."
Reducing SRAM to 12KB makes the current WFC heuristic pick a different Softmax tile.

![Copier Problem](assets/copier_problem.png)

### Backtracking: Contradiction & Recovery

Intentionally constructed unsolvable constraint configurations verify that backtracking actually fires.
In a ROW → ? → COL chain with threshold=0.5, contradiction occurs → 1 backtrack → confirmed failure.

![Backtracking](assets/backtracking.png)

### SRAM Size Impact

The same 6-layer Attention Block scheduled under 64KB / 16KB / 12KB SRAM.
Only in the current 12KB heuristic run does Softmax select a smaller tile (16×16), producing differentiation.

![SRAM Comparison](assets/sram_comparison.png)

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  LayerNode  │────▶│ HardConstraint│────▶│  CostModel   │
│ (candidates)│     │  (SRAM, align)│     │ (roofline +  │
│             │     │               │     │  affinity +  │
│ HWState[]   │     └───────────────┘     │  cache)      │
└─────────────┘                           └──────┬───────┘
                                                 │
       ┌──────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│CollapseEngine│────▶│  Propagation  │────▶│  Scheduler   │
│ (entropy,    │     │ (layout +     │     │ (bottleneck  │
│  backtrack)  │     │  tile_shape + │     │  first,      │
│              │     │  location)    │     │  bidirect)   │
└──────────────┘     └───────────────┘     └──────────────┘
```

## GPU Execution Verification (v2.7)

Codegen-generated Triton kernels were re-verified in `.venv` on RTX 3060 (SM86, CUDA 12.8):

| Kernel | Type | Result | Error |
|--------|------|--------|-------|
| Linear (MatMul) | GEMM | PASS | rel_err < 0.5% (fp16) |
| ReLU | Element-wise | PASS | Exact match |
| Softmax | Row-parallel | PASS | max_diff < 1e-8 |

### HW-WFC vs Triton Autotuner

This section is currently under re-audit.

Repeated reruns in the same `.venv` and on the same RTX 3060 already showed materially different averages (`118%` and `84%` WFC/autotuner).
We then repeated the script 5 more times in isolated processes with separate `TRITON_CACHE_DIR` values and no other visible GPU compute processes; the average ratio still ranged from `86.0%` to `94.6%`, so the old fixed TFLOPS table is still not stable enough for README use.

What is verified today:
- The Triton kernels execute correctly on the target GPU.
- The comparison script runs end-to-end in `.venv`.
- A pattern does emerge under isolated reruns: the tiny `Small (256x512x256)` case is unstable at ~`0.01-0.02ms` and Triton flips between `32x64x32` and `32x32x32`, while the larger workloads are more consistent and Triton usually stays about `5-20%` ahead of the current WFC choice.
- The current benchmarking methodology still needs stabilization before we can claim a single authoritative WFC/autotuner ratio.

## Quick Start

```bash
# Schedule a Transformer Attention Block
python examples/attention_block.py

# Reproduce the README benchmark (top-8 exhaustive baseline)
python examples/attention_exhaustive_benchmark.py

# Run Safety Checks
python tests/test_safety_checks.py

# Verify Triton kernels on GPU (requires torch + triton)
python examples/triton_verify.py

# Compare WFC vs Triton autotuner
python examples/triton_autotuner_compare.py

# Generate / update all visualizations
python tools/generate_all_visuals.py
```

## Cost Model

Three-component additive scoring:

- **Roofline Score** — Reflects data reuse from tiling. Larger tiles improve MatMul OI; wider tile_n helps Softmax
- **Layout Affinity** — Per-layer-type layout preference (LINEAR→ROW_MAJOR, SOFTMAX→ROW_MAJOR, CONV2D→BLOCK_TILED)
- **Cache Efficiency** — Gaussian over SRAM utilization (peaks at 75%, applies working memory multiplier per layer type)

## Hardware Specs

| Spec | SRAM | Purpose |
|------|------|---------|
| `toy_gpu.json` | 64KB | Comfortable environment, basic validation |
| `tight_gpu.json` | 16KB | Moderate constraint |
| `stress_gpu.json` | 12KB | Tight SRAM, forces per-layer-type differentiation |
| `a100.json` | 192KB | NVIDIA A100 (HBM2e 2039 GB/s) |
| `h100.json` | 256KB | NVIDIA H100 (HBM3 3352 GB/s) |
| `rtx3060.json` | 100KB | NVIDIA RTX 3060 (GDDR6 360 GB/s) — GPU execution verified |

## Safety Checks (5/5 Pass)

| # | Check | Status | Criteria |
|---|-------|--------|----------|
| 1 | Hard Constraint correctness | Pass | No false positives/negatives |
| 2 | Cost Model discrimination | Pass | Score spread > 0.01 (actual: 1.19) |
| 3 | Propagation effectiveness | Pass | Removal > 25% at t=1.5 (actual: 43%) |
| 4 | Backtracking liveness | Pass | Triggers on contradiction (verified) |
| 5 | No copier problem | Pass | Heterogeneous layers get different states |

## Project Structure

```
hw-wfc/
├── src/                    # Core algorithm
│   ├── state.py            # HWState, LayerNode, superposition
│   ├── constraint.py       # Hard/soft constraints, propagation, auto-weight
│   ├── cost_model.py       # Roofline + affinity + cache
│   ├── collapse.py         # Entropy-based collapse + backtracking
│   └── scheduler.py        # Main pipeline
├── specs/                  # Hardware specifications (JSON)
├── examples/               # PoC scripts
├── tests/                  # Safety checks
├── tools/                  # Visualization generators
├── assets/                 # Generated images/GIFs
├── docs/
│   ├── concept.md / .ko.md             # Algorithm concept & motivation (EN/KO)
│   ├── architecture_guide.md / .ko.md  # Design rationale guide (EN/KO)
│   └── daily_logs/                     # Work logs
└── HANDOFF.md              # Next tasks & ideas
```

## Requirements

- Python 3.10+
- Pillow (visualization only)
- PyTorch + Triton (GPU verification only, `pip install torch triton`)

## License

Research prototype. Not for production use.
