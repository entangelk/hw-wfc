<p align="center">
  <a href="./concept.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./concept.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>

# HW-WFC: Automated AI Hardware Scheduling via Constraint-Based Collapsing Search

## 1. Algorithm Overview

### 1.1 What is HW-WFC?

HW-WFC is an AI compiler scheduling algorithm that applies the **core principles of Circle-WFC** (plugin constraints, bidirectional collapse from midpoints) -- originally designed for 2D pathfinding -- to **multidimensional hardware optimization spaces**.

Instead of exhaustively searching billions of hardware parameter combinations, it **cascading-collapses infeasible states using physical constraints** to find optimal memory layouts and compute configurations.

### 1.2 Core Idea Mapping

| Circle-WFC (2D Pathfinding) | HW-WFC (Hardware Optimization) | Essence |
|---|---|---|
| Grid / Tile Map | Hardware Search Space | Tile size, loop unrolling, memory layout |
| Circle Constraint | Physical Hardware Limits | SRAM capacity, memory bandwidth, register limits |
| Start from Midpoint (Bidirectional Collapse) | Bottleneck-First Collapse | Fix the heaviest computation first, propagate both ways |
| Plugin Condition Injection | Target Hardware Spec Injection (JSON) | Swap specs when hardware changes |

---

## 2. Core Mechanisms

### 2.1 Hardware State Superposition

Before compilation, each layer (node) exists in a superposition of multiple hardware implementation options:

- **Memory Layout**: `Row-major`, `Col-major`, `Block-tiled`
- **Tiling Size**: `16x16`, `32x32`, `64x64`, `128x128`, rectangular tiles, etc.
- **Compute Location**: `SRAM (Shared Memory)`, `DRAM (Global Memory)`

### 2.2 Two-Phase Constraint Engine

A two-phase system that goes beyond simple "possible/impossible" to evaluate hardware performance tradeoffs:

**Phase 1 -- Hard Constraints (Physical Infeasibility Elimination)**
- Immediately removes states that are physically infeasible: SRAM/L1 cache capacity exceeded, memory alignment violations, etc.
- Purpose: Cut off explosive growth of the search space early

**Phase 2 -- Soft Constraints & Cost Model (Probabilistic Ranking)**
- A lightweight cost model based on the Roofline model computes an expected performance score for each state
- Scores are normalized into probabilities -> Shannon Entropy is computed -> the most certain state collapses first

### 2.3 Bottleneck-First Bidirectional Collapse

Unlike typical compilers that search sequentially from input to output:

1. **Bottleneck Identification**: Select the layer with the highest FLOPs across the entire model as the starting point
2. **Initial Collapse**: Force-collapse to the state with the highest Cost Model score
3. **Bidirectional Propagation**: Confirmed constraints propagate like a wave to preceding and following layers
4. **Cascading Collapse**: Surrounding layers automatically collapse to satisfy the propagated constraints

### 2.4 Backtracking

A safety mechanism to prevent initial collapse from breaking global optimality:
- If a contradiction occurs during propagation (zero feasible states remaining) -> undo the most recent collapse
- Exclude that state from candidates and re-collapse to the next-best state

### 2.5 Transition Cost Integration

Different configurations between adjacent layers incur data conversion costs:
- Layout transition: ROW<->COL (transpose), ROW<->BLOCK (partial rearrangement)
- Location transition: SRAM<->DRAM (memory hierarchy movement)
- Tile size transition: Re-tiling cost proportional to area ratio

During collapse, transition costs with adjacent nodes are subtracted from the base score, **favoring pipeline-wide optima over per-layer optima**. Weights are automatically derived from SRAM capacity.

---

## 3. Comparison with Existing Approaches

| Aspect | Grid Search / RL | HW-WFC |
|--------|-----------------|--------|
| **Search Strategy** | Random mutation + runtime measurement (Trial & Error) | Infeasible state elimination via constraints (Deductive) |
| **Time Complexity** | Thousands to tens of thousands of compilations required | Cascading collapse drastically reduces candidates (~2ms) |
| **Optimization Direction** | Forward (input -> output) | Bidirectional (bottleneck -> simultaneous propagation both ways) |
| **Hardware Change** | Full re-search from scratch | Instant adaptation by swapping spec (JSON) |
| **Inter-Layer Interaction** | Independent optimization (no interaction) | Global optimization via transition costs + propagation |

---

## 4. Validation Strategy

### 4.1 Toy Model (Basic Validation)

**Structure**: `Linear(A)` -> `ReLU(B)` -> `Linear(C)` / SRAM 64KB

Minimal validation that constraint propagation alone reduces the search space by 98.5%.

### 4.2 Transformer Attention Block (Core Validation)

**Structure**: `QKV Proj` -> `QK MatMul` -> `Softmax` -> `AV MatMul` -> `Out Proj` -> `LayerNorm`

Validation points:
- Softmax is a memory-bound layer. Does HW-WFC detect conflicts when it clashes with the layout chosen by MatMul?
- Does transition cost propagation find a "global optimum that sacrifices MatMul's local optimum to speed up the entire Attention Block"?
- When SRAM is tight (12KB), does it discover different optimal tiles for different layer types? (Solving the copier problem)

### 4.3 Safety Checks (5 Sanity Checks)

A research principle of questioning "the results look good, but are they actually good?":

1. **Hard Constraint Correctness** -- Does it avoid incorrectly eliminating valid candidates?
2. **Cost Model Discriminability** -- If scores are flat, entropy-based selection is effectively random
3. **Propagation Effectiveness** -- Does propagation produce real effects, not just nominal ones?
4. **Backtracking Activation** -- Is it actually triggered, or is it dead code?
5. **No Copier Problem** -- Do heterogeneous layers select different optimal states?

---

## 5. Roadmap

### Phase 1: Core Algorithm Refinement
- ~~Automatic transition cost weight derivation~~ (Done, v2.2)
- 2D graph propagation (DAG support: skip connections, multi-head parallelism)
- Adaptive penalty threshold

### Phase 2: Realistic Models & Specs
- Testing with real GPU specs (A100, H100)
- Large-scale model scaling (GPT-2 Small 72 layers, BERT-Base 84 layers)
- Conv2D / Depthwise Conv support

### Phase 3: Output & Integration
- Schedule results -> automatic Triton / CUDA kernel parameter generation
- Interactive visualization dashboard
