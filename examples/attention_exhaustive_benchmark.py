"""Reproducible HW-WFC vs exhaustive baseline benchmark for the attention block.

The exhaustive baseline is intentionally limited to the top-k candidates per
layer after hard constraints so that it remains runnable on a developer
machine. This is the benchmark referenced from the README.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constraint import (
    HardwareSpec,
    apply_hard_constraints,
    derive_transition_weight,
    total_transition_penalty,
)
from src.cost_model import compute_score, score_all_candidates
from src.scheduler import HWWFCScheduler

from attention_block import build_attention_block


def resolve_spec(spec_name: str) -> Path:
    path = Path(spec_name)
    if path.exists():
        return path
    return Path(__file__).resolve().parent.parent / "specs" / spec_name


def pipeline_score(nodes, states, spec: HardwareSpec, weight: float) -> float:
    score = sum(compute_score(node, state, spec) for node, state in zip(nodes, states))
    score -= sum(
        weight * total_transition_penalty(src, dst)
        for src, dst in zip(states, states[1:])
    )
    return score


def run_wfc(spec: HardwareSpec, runs: int) -> tuple[list[float], object, list[object], float]:
    times_ms: list[float] = []
    last_result = None
    last_nodes = None
    weight = derive_transition_weight(spec)

    for _ in range(runs):
        nodes = build_attention_block()
        t0 = time.perf_counter()
        result = HWWFCScheduler(spec, penalty_threshold=1.5).schedule(nodes)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_result = result
        last_nodes = nodes

    chosen_states = [node.collapsed_state for node in last_nodes]
    score = pipeline_score(last_nodes, chosen_states, spec, weight)
    return times_ms, last_result, last_nodes, score


def run_exhaustive(spec: HardwareSpec, top_k: int) -> tuple[list[int], int, float, tuple[object, ...], float]:
    nodes = build_attention_block()
    weight = derive_transition_weight(spec)

    candidate_lists = []
    for node in nodes:
        apply_hard_constraints(node, spec)
        scored = score_all_candidates(node, spec)[:top_k]
        candidate_lists.append([state for state, _ in scored])

    counts = [len(candidates) for candidates in candidate_lists]
    combinations = 1
    for count in counts:
        combinations *= count

    best_score = float("-inf")
    best_combo = None
    t0 = time.perf_counter()
    for combo in product(*candidate_lists):
        score = pipeline_score(nodes, combo, spec, weight)
        if score > best_score:
            best_score = score
            best_combo = combo
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return counts, combinations, best_score, best_combo, elapsed_ms


def format_state(state) -> str:
    return f"{state.tile_m}x{state.tile_n} {state.layout.name} {state.location.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="stress_gpu.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--wfc-runs", type=int, default=8)
    args = parser.parse_args()

    spec = HardwareSpec.from_json(resolve_spec(args.spec))
    weight = derive_transition_weight(spec)

    print("=" * 72)
    print("  HW-WFC vs Exhaustive Baseline")
    print("=" * 72)
    print(f"\n[Config]")
    print(f"  Spec: {spec.name}")
    print(f"  SRAM: {spec.sram_bytes // 1024}KB")
    print(f"  Top-k per layer: {args.top_k}")
    print(f"  Transition weight: {weight:.4f}")

    wfc_times, result, nodes, wfc_score = run_wfc(spec, args.wfc_runs)
    exhaustive_counts, combinations, best_score, best_combo, exhaustive_ms = run_exhaustive(
        spec, args.top_k
    )

    print(f"\n[HW-WFC]")
    print(f"  Search space: {result.initial_search_space} -> {result.final_search_space}")
    print(f"  Time avg: {statistics.mean(wfc_times):.3f} ms")
    print(f"  Time med: {statistics.median(wfc_times):.3f} ms")
    print(f"  Time min/max: {min(wfc_times):.3f} / {max(wfc_times):.3f} ms")
    print(f"  Score: {wfc_score:.4f}")
    for node in nodes:
        print(f"  {node.name:12s} -> {format_state(node.collapsed_state)}")

    print(f"\n[Exhaustive Baseline]")
    print(f"  Candidate counts: {exhaustive_counts}")
    print(f"  Combinations: {combinations:,}")
    print(f"  Time: {exhaustive_ms:.3f} ms")
    print(f"  Best score: {best_score:.4f}")
    for name, state in zip([node.name for node in nodes], best_combo):
        print(f"  {name:12s} -> {format_state(state)}")

    quality = (wfc_score / best_score * 100.0) if best_score > 0 else 0.0
    speedup = exhaustive_ms / statistics.mean(wfc_times) if wfc_times else 0.0
    same_choice = all(node.collapsed_state == state for node, state in zip(nodes, best_combo))

    print(f"\n[Comparison]")
    print(f"  Quality vs exhaustive: {quality:.4f}%")
    print(f"  Same final choice: {same_choice}")
    print(f"  Speedup: {speedup:.1f}x")
    print(
        "\nNote: the exhaustive baseline searches the top-k survivors per layer "
        "after hard constraints, not the full Cartesian product of all candidates."
    )


if __name__ == "__main__":
    main()
