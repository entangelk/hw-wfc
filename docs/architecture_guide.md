<p align="center">
  <a href="./architecture_guide.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./architecture_guide.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>

# HW-WFC Architecture Guide

> This document is for anyone new to the project who wants to understand **"why it was designed this way."**
> It does not cover code structure or APIs. You can learn those by reading the code.
> What it does cover is **intent, background, and tradeoffs** -- things you cannot learn from the code alone.

---

## 1. The Problem This Project Solves

When running AI models on a GPU, each layer (MatMul, Softmax, ReLU, etc.) requires many decisions:

- **Tile size**: How should the matrix be partitioned for processing? (8x8, 32x32, 64x128...)
- **Memory layout**: Should data be arranged row-major or column-major?
- **Compute location**: Should execution happen in fast but small SRAM, or slow but large DRAM?

Each layer has tens to hundreds of possible combinations, and with tens of layers, the total search space reaches **billions of combinations**.

### Limitations of Existing Approaches

| Approach | Problem |
|----------|---------|
| Grid Search | Tries every combination. Search space explodes exponentially |
| RL-based Search | Requires thousands to tens of thousands of actual executions for training. Must restart from scratch when hardware changes |
| Manual Tuning | Expert-dependent. Easy to miss inter-layer interactions |

### The HW-WFC Approach

**"Instead of trying every possible combination, eliminate the impossible ones first."**

This borrows the core idea from the Wave Function Collapse (WFC) algorithm:
1. Every layer starts with all possible states simultaneously (Superposition)
2. Physically impossible states are removed (Hard Constraint)
3. The most certain layer is collapsed to a single state first (Collapse)
4. The collapsed result propagates to neighboring layers (Propagation)
5. If a dead end is reached, the algorithm backtracks (Backtracking)

---

## 2. GPU Spec Design Intent

### 2.1 Why Virtual Specs?

Reasons for not using actual A100/H100 specs:

- **Real GPU SRAM (192-256KB) is too generous.** Our test tiles (up to 128x128 = 64KB) all fit into SRAM, so constraints never activate and the algorithm goes unverified.
- **We need to prevent the illusion that the algorithm is only solving "easy problems."** With ample SRAM, every layer picks the same optimal tile, making propagation and backtracking unnecessary. That is not the algorithm working well -- it is the problem being easy.

### 2.2 The Role of Three Specs

```
Spec design starts from the question: "Which aspect of the algorithm do we want to test?"
```

| Spec | SRAM | Alignment | Role |
|------|------|-----------|------|
| `toy_gpu` | 64KB | 128B | **Basic behavior check**. Verify the algorithm finds the correct optimum with ample SRAM |
| `tight_gpu` | 16KB | 64B | **Transition point exploration**. Some tiles start exceeding SRAM capacity |
| `stress_gpu` | 12KB | 64B | **Core verification spec**. Extremely limited SRAM forces different optimal solutions per layer |

### 2.3 SRAM Size Intent in Detail

#### toy_gpu (64KB)

```
32x32 float32 tile = 4KB   -> 6% of SRAM. Plenty of room
64x64 float32 tile = 16KB  -> 25% of SRAM. Still comfortable
128x128 float32 tile = 64KB -> 100% of SRAM. Boundary
```

- Most tiles fit in SRAM.
- Purpose: Verify that the **basic algorithm flow** (collapse -> propagation -> completion) works correctly.
- Limitation: Every layer picks the same optimal tile (64x64) -- no differentiation.

#### stress_gpu (12KB) -- The Most Important Spec

```
32x32 float32 tile = 4KB
  -> LINEAR (working memory 3x) = 12KB -> 100% of SRAM
  -> SOFTMAX (working memory 4x) = 16KB -> exceeds the working-memory target used by the cache penalty

16x16 float32 tile = 1KB
  -> SOFTMAX (4x) = 4KB -> 33% of SRAM
```

What this means:
- The current heuristic strongly prefers **32x32 for LINEAR** under stress_gpu
- The current heuristic often prefers **16x16 for SOFTMAX** because the cache term penalizes working-memory overflow
- Within the same model, **different layers can end up with different chosen tiles**
- Whether the heuristic discovers this on its own is the core verification point (solving the "copier problem")

#### What Is the Working Memory Multiplier?

The number of buffers that must reside in SRAM simultaneously to process a single tile:

| Layer | Buffers | Multiplier |
|-------|---------|------------|
| LINEAR (MatMul) | input_tile + weight_tile + output_tile | 3x |
| SOFTMAX | input + exp_buffer + sum_buffer + output | 4x |
| RELU | input + output | 2x |

Example: Processing LINEAR with a 32x32 tile (4KB) requires 4KB x 3 = 12KB simultaneously in SRAM.
On stress_gpu (12KB), this occupies 100% of SRAM.

### 2.4 Alignment Setting Intent

```
alignment_bytes = 128  (toy_gpu)
```

GPU memory controllers read data efficiently only at addresses aligned to specific byte boundaries.
The 128B alignment is based on the actual A100 value.

What this does in practice:
- tile_n x dtype_bytes must be a multiple of the alignment
- 8x8 tile: row = 8 x 4 = 32B -> 32 % 128 != 0 -> **eliminated**
- 16x16 tile: row = 16 x 4 = 64B -> 64 % 128 != 0 -> **eliminated**
- 32x32 tile: row = 32 x 4 = 128B -> 128 % 128 = 0 -> **passes**

This is intentional behavior. On real GPUs, small tiles also waste memory bandwidth.

### 2.5 Remaining Numbers

| Parameter | Value | Source |
|-----------|-------|--------|
| `compute_gflops: 312000` | NVIDIA A100 FP32 peak | Official spec sheet |
| `sram_bandwidth_gbps: 19000` | A100 Shared Memory theoretical bandwidth | Architecture documentation |
| `dram_bandwidth_gbps: 900` | A100 HBM2e | Official spec sheet |

These values are used in the Roofline model to distinguish "compute-bound vs memory-bound."
Only the SRAM size is virtual; the remaining performance figures are based on real GPU specs.

---

## 3. How the Cost Model Scores Candidates

Each candidate state (HWState) receives a score. Score = "How efficient is this tile configuration?"

### 3.1 Three Scoring Components

```python
total_score = roofline_score + layout_affinity + cache_efficiency
```

#### Roofline Score (0-1.0)

Determines whether the operation is compute-bound or memory-bound.

```
Operational Intensity = FLOPs / Memory Traffic
```

- High OI -> compute-bound -> GPU cores fully utilized -> high score
- Low OI -> memory-bound -> GPU waiting for data -> low score

Larger tiles increase data reuse, reducing traffic and raising OI.
This is the basis for "bigger tile = better," but it trades off against SRAM capacity constraints.

#### Layout Affinity (0-0.2)

Different layer types prefer different memory layouts:

- **Softmax**: Row-wise reduction -> ROW_MAJOR required (+0.2)
- **LINEAR**: ROW_MAJOR favorable for output writes (+0.15)
- **Conv2D**: BLOCK_TILED favorable for spatial locality (+0.15)
- **RELU**: Element-wise, layout-agnostic (+0.05)

#### Cache Efficiency (-0.3 to +0.2)

Higher bonus when SRAM utilization is close to the sweet spot (75%).

```
Too little usage (20%) -> SRAM wasted -> low score
Just right (75%)       -> ideal -> highest score
Exceeded (>100%)       -> strong penalty for working-memory overflow
```

Modeled as a Gaussian curve peaking around 0.75.

### 3.2 Why These Three Components?

| Component | Phenomenon It Captures | Problem If Missing |
|-----------|----------------------|-------------------|
| Roofline | Compute/memory efficiency by tile size | All tiles get the same score -> entropy-based selection = random |
| Layout Affinity | Optimal layout differences per layer type | Softmax and LINEAR pick the same layout -> copier problem |
| Cache Efficiency | SRAM utilization and working-memory overflow pressure | overflowed tiles can look attractive unless the score penalizes them |

---

## 4. Transition Cost and Automatic Weighting

### 4.1 What Is Transition Cost?

When adjacent layers choose different configurations, data conversion is needed between them:

| Transition Type | Example | Cost |
|-----------------|---------|------|
| Layout transition | ROW_MAJOR -> COL_MAJOR | 1.0 (matrix transpose required) |
| Layout transition | ROW_MAJOR -> BLOCK_TILED | 0.5 (partial rearrangement) |
| Location transition | SRAM -> DRAM | 0.8 (memory hierarchy movement) |
| Tile size transition | 32x32 -> 64x64 | 0.0-0.6 (proportional to area ratio) |

These costs are summed into a **total transition penalty** (maximum ~2.4).

### 4.2 Why Incorporate Transition Cost?

Optimizing each layer independently can cause data rearrangement at every layer boundary.
Giving up the per-layer optimum may be worthwhile if it reduces total pipeline transition cost.

Example:
```
Layer A's individual optimum: ROW_MAJOR (score = 1.2)
Layer B's individual optimum: COL_MAJOR (score = 1.1)
-> A->B transition cost: 1.0 (transpose required)

If Layer A chooses COL_MAJOR instead (score = 1.05):
-> A->B transition cost: 0.0
-> Net result: 0.15 loss vs 1.0 saved = net gain of 0.85
```

### 4.3 The Role of transition_weight

The transition penalty is a dimensionless number (0-2.4), and the base score is also dimensionless (0-1.4).
The weight controls their relative importance:

```python
adjusted_score = base_score - penalty * weight
```

- weight = 0.01 -> transition cost nearly ignored -> each layer optimized independently
- weight = 0.5 -> transition cost strongly reflected -> preference for consistent settings with neighbors

### 4.4 Automatic Derivation Formula

```python
weight = clamp(reference_tile_bytes / sram_bytes, 0.01, 0.5)
```

| Variable | Meaning |
|----------|---------|
| `reference_tile_bytes` | 32x32 float32 = 4096 bytes (mid-sized reference tile) |
| `sram_bytes` | Hardware SRAM capacity |

**Physical intuition**: When SRAM is small, a tile occupies a large fraction of it. In this situation, a layout transition is close to a complete replacement of data in SRAM, making the cost relatively high.

| Spec | SRAM | weight | Interpretation |
|------|------|--------|----------------|
| toy_gpu | 64KB | 0.0625 | Reference tile is 6% of SRAM -> low transition burden |
| tight_gpu | 16KB | 0.25 | Reference tile is 25% of SRAM -> transitions become noticeable |
| stress_gpu | 12KB | 0.333 | Reference tile is 33% of SRAM -> transition = major cost |

### 4.5 Why Automatic Derivation Instead of a Fixed 0.1?

Problems with a manual value:
- The real burden of transition cost differs by 5x between toy_gpu (64KB) and stress_gpu (12KB), but using a flat 0.1 underestimates transition cost on stress_gpu
- Every time a new hardware spec is added, the appropriate weight must be found manually
- Automatic derivation starts from the physical basis of "tile occupancy relative to SRAM," so it produces a reasonable value for any spec

Validation results:
- On existing scenarios (toy_gpu, stress_gpu), the auto-derived weight produces the same final selections as the manual 0.1 -> backward compatible
- Differences appear only under extremely small SRAM (8KB): automatic derivation correctly prioritizes transition cost, choosing the same layout as neighbors

---

## 5. Constraint Propagation and Threshold

### 5.1 What Propagation Does

When Layer A collapses to `32x32 ROW_MAJOR SRAM`, candidates in adjacent Layer B whose transition penalty exceeds the threshold are removed.

```
A = 32x32 ROW_MAJOR SRAM (collapsed)
Among B's candidates:
  32x32 ROW_MAJOR SRAM -> penalty = 0.0 -> kept
  32x32 BLOCK_TILED SRAM -> penalty = 0.5 -> below threshold (1.5) -> kept
  64x64 COL_MAJOR DRAM -> penalty = 0.8+1.0+0.3 = 2.1 -> exceeds threshold (1.5) -> removed
```

### 5.2 Rationale for Threshold = 1.5

| Threshold | Removal Rate | Meaning |
|-----------|-------------|---------|
| 0.5 | 94% | Extreme. Only same layout allowed. Search space collapses |
| 1.0 | 76% | Aggressive. Most DRAM + different layout combinations removed |
| **1.5** | **43%** | **Balance point. Only clearly inefficient candidates removed** |
| 2.0 | 15% | Lenient. Almost no propagation effect |
| 3.0 | 0% | Propagation disabled. All candidates kept |

Why 1.5 was chosen:
- layout transition (1.0) + location transition (0.8) = 1.8 -> removed. This means **only extreme transitions where both layout and memory location differ are eliminated**
- Layout-only difference (1.0) or location-only difference (0.8) -> kept. **One dimension of difference is allowed**
- 43% removal rate meets the CLAUDE.md criterion (>25%)

---

## 6. What Safety Checks Verify

Safety Checks originate from the research principle: "The results look good, but are they genuinely good? Be skeptical."

### Check #1: Hard Constraint Accuracy

**Suspicion**: "It says 12 candidates were removed, but were only physically impossible ones removed? Could valid candidates have been incorrectly eliminated?"

Verification method: Manually compute SRAM capacity and alignment for each candidate and cross-check against the function's results.

Key checks:
- If a DRAM candidate is rejected by SRAM constraints, that is a bug (DRAM is independent of SRAM)
- Determine whether 8x8 tiles being rejected by alignment is excessive (currently this is intentional behavior)

### Check #2: Cost Model Discriminative Power

**Suspicion**: "If all candidates receive similar scores, is entropy-based selection effectively random?"

Verification method: Measure the max-min score difference (spread).

Pass criteria: spread > 0.01, with at least 3 distinct tiers.
Current result: spread = 1.19, with 54 candidates fully differentiated.

### Check #3: Propagation Strength

**Suspicion**: "Is threshold=1.5 too lenient, making propagation nominal? Or is it too strict, eliminating valid candidates?"

Verification method: Compare removal rates as threshold varies from 0.5 to 3.0.
Pass criteria: Removal rate > 25% at threshold=1.5.
Current result: 43%.

### Check #4: Backtracking

**Suspicion**: "backtrack_count = 0 means the algorithm is perfect?" No -- it could mean the backtracking code is dead code that never executes.

Verification method: Construct scenarios that intentionally induce contradictions.
- **4a**: Force conflicting layouts on adjacent layers (ROW vs COL conflict)
- **4b**: Extreme constraints with no valid solution (ROW->COL impossible at threshold=0.5)

Pass criteria: backtrack_count > 0 in at least one scenario.

### Check #5: Scaling and the Copier Problem

**Suspicion**: "98% reduction with 3 layers is impressive, but does it hold with 50 layers? And are all layers picking the same answer?"

Verification method:
- **5a**: Scale homogeneous layers (Linear+ReLU repeating) to 3/10/50 layers
- **5b**: Run heterogeneous layers (MatMul+Softmax+LayerNorm) on stress_gpu

Pass criteria: `all_same_state = False` for heterogeneous layers.
Current heuristic result: MatMul=32x32, Softmax=16x16 -- differentiation achieved.

---

## 7. Full Pipeline Walkthrough

Scheduling 4 layers of an Attention Block on stress_gpu (12KB):

```
[Initial State]
  QK_MatMul:  66 candidates (11 tiles x 3 layouts x 2 locations)
  Softmax:    66 candidates
  AV_MatMul:  66 candidates
  LayerNorm:  66 candidates
  Total search space: 264 states

[Step 1: Hard Constraint]
  Remove tiles exceeding 12KB SRAM + alignment violations
  -> 96 removed, 168 states remaining

[Step 2: Bottleneck-First Collapse]
  QK_MatMul has highest FLOPs (536M) -> collapse first
  Compute cost model scores -> 32x32 ROW_MAJOR SRAM scores highest
  -> QK_MatMul collapsed

[Step 3: Propagation]
  QK_MatMul (32x32 ROW) -> Softmax: 20 candidates with transition penalty > 1.5 removed
  -> Softmax reduced to 22 candidates

[Step 4: Remaining Collapses]
  Softmax: Entropy calculation -> 16x16 ROW_MAJOR scores highest under the current heuristic
  AV_MatMul -> 32x32 ROW_MAJOR (same physical reasoning as QK_MatMul)
  LayerNorm -> 32x32 ROW_MAJOR

[Result]
  264 -> 4 states (98.5% reduction)
  Only Softmax selects a different tile in this heuristic run -> copier problem resolved
```

Important note:
- Working-memory overflow is currently modeled as a strong soft penalty in `cost_model.py`, not as a hard elimination in `constraint.py`
- Because of that, this walkthrough reflects the current WFC heuristic path, not a proof of the global optimum over the full candidate set

---

## 8. Current Limitations and Known Constraints

### Algorithm Limitations
- **1D chain only**: Propagation currently supports only linear chains (layer 0->1->2->3). DAG structures like ResNet skip connections or Transformer multi-head parallelism are not supported.
- **Fixed threshold**: penalty_threshold=1.5 is not optimal for all models. Propagation may weaken with 100+ layers.

### Cost Model Limitations
- **Gap with actual execution time**: The Roofline model represents a theoretical upper bound. Cache conflicts, warp divergence, etc. are not captured.
- **Immature Conv2D support**: No im2col-based memory traffic model yet. Currently treated identically to MatMul.

### Validation Limitations
- **No real GPU benchmarks**: Whether the cost model scores correlate with actual execution times has not been verified.
- **Virtual spec based**: stress_gpu's 12KB SRAM does not exist on any real GPU. On an actual A100 (192KB), the problem may be too easy.

---

## Appendix: Glossary

| Term | Meaning |
|------|---------|
| **Superposition** | A layer in an undecided state. Holds all possible HWStates simultaneously |
| **Collapse** | Selecting one candidate from a layer's possibilities and discarding the rest |
| **Propagation** | The process by which a collapsed layer's constraints reduce the candidates of neighboring layers |
| **Backtracking** | A safety mechanism that reverts to a previous state when a collapse produces a contradiction |
| **Entropy** | How "certain" the score distribution of candidates is. Lower entropy means one candidate dominates |
| **Roofline Model** | An analysis framework that determines performance bottlenecks by comparing compute volume vs memory bandwidth |
| **Working Memory** | Total memory that must reside in SRAM simultaneously to process a single tile |
| **Transition Penalty** | Data conversion cost incurred when adjacent layers have different configurations |
| **HWState** | A single hardware implementation candidate. A combination of (tile_m, tile_n, layout, location) |
