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
| Quality vs Grid Search | 100% (transition-aware) |
| Speed | ~2.5ms (6-layer attention block) |
| Grid Search equivalent | 262K combinations, 6000ms |
| Speedup | **~2400x** |

### Layer-Type Differentiation (12KB SRAM)

```
QKV_Proj   → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
QK_MatMul  → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
Softmax    → 16x16  ROW_MAJOR  SRAM   (SOFTMAX, working memory 4x)  ← different!
AV_MatMul  → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
Out_Proj   → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
LayerNorm  → 32x32  ROW_MAJOR  SRAM   (LAYERNORM, working memory 3x)
```

Under tight SRAM, the working memory multiplier difference between layer types forces the algorithm to select different optimal tiles — Softmax (4× buffer) cannot use 32×32 in 12KB SRAM, while LINEAR (3× buffer) can.

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
Reducing SRAM to 12KB forces Softmax to pick a different tile due to working memory differences.

![Copier Problem](assets/copier_problem.png)

### Backtracking: Contradiction & Recovery

Intentionally constructed unsolvable constraint configurations verify that backtracking actually fires.
In a ROW → ? → COL chain with threshold=0.5, contradiction occurs → 1 backtrack → confirmed failure.

![Backtracking](assets/backtracking.png)

### SRAM Size Impact

The same 6-layer Attention Block scheduled under 64KB / 16KB / 12KB SRAM.
Only at 12KB does Softmax select a smaller tile (16×16), producing differentiation.

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

## Quick Start

```bash
# Schedule a Transformer Attention Block
python examples/attention_block.py

# Run Safety Checks
python tests/test_safety_checks.py

# Generate / update all visualizations
python tools/generate_all_visuals.py

# Regenerate collapse GIF only
python tools/visualize_collapse.py
```

## Cost Model

Three-component additive scoring:

- **Roofline Score** — Reflects data reuse from tiling. Larger tiles improve MatMul OI; wider tile_n helps Softmax
- **Layout Affinity** — Per-layer-type layout preference (LINEAR→ROW_MAJOR, SOFTMAX→ROW_MAJOR, CONV2D→BLOCK_TILED)
- **Cache Efficiency** — Gaussian over SRAM utilization (peaks at 75%, applies working memory multiplier per layer type)

## Hardware Specs

Three virtual GPU specs with real A100 performance numbers but varying SRAM sizes to stress-test the algorithm:

| Spec | SRAM | Purpose |
|------|------|---------|
| `toy_gpu.json` | 64KB | Comfortable environment, basic validation |
| `tight_gpu.json` | 16KB | Moderate constraint |
| `stress_gpu.json` | 12KB | Tight SRAM, forces per-layer-type differentiation |

> **Why virtual specs?** Real GPUs (A100: 192KB, H100: 256KB) have enough SRAM that all tiles fit comfortably — no constraint fires, no differentiation occurs. Virtual specs with small SRAM create the hard trade-offs that actually exercise the algorithm. See [Architecture Guide §2](docs/architecture_guide.md#2-gpu-스펙-설계-의도) for full rationale.

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

## License

Research prototype. Not for production use.
