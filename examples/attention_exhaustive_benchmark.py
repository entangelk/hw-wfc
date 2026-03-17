"""Audit HW-WFC against several exact baselines for the attention block.

This script reports:
1. HW-WFC's heuristic schedule
2. A naive exhaustive enumeration over the top-k unary candidates per layer
3. An exact dynamic-programming solver over the full hard-constrained set

The top-k exhaustive baseline is still useful for reproducing the original
README benchmark, but it can miss the true optimum because the candidate set is
truncated before pairwise transition costs are considered.
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


def prepare_nodes(spec: HardwareSpec):
    nodes = build_attention_block()
    for node in nodes:
        apply_hard_constraints(node, spec)
    return nodes


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
    nodes = prepare_nodes(spec)
    weight = derive_transition_weight(spec)

    candidate_lists = []
    for node in nodes:
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


def run_exact_dp(
    spec: HardwareSpec,
    runs: int,
) -> tuple[list[float], list[int], float, list[object]]:
    times_ms: list[float] = []
    counts: list[int] = []
    best_score = 0.0
    best_states: list[object] = []
    weight = derive_transition_weight(spec)

    for _ in range(runs):
        nodes = prepare_nodes(spec)
        candidate_lists = [list(node.candidates) for node in nodes]
        counts = [len(states) for states in candidate_lists]
        unary = [
            [compute_score(node, state, spec) for state in states]
            for node, states in zip(nodes, candidate_lists)
        ]

        t0 = time.perf_counter()
        dp = unary[0][:]
        backpointers: list[list[int]] = []

        for layer_idx in range(1, len(candidate_lists)):
            prev_states = candidate_lists[layer_idx - 1]
            cur_states = candidate_lists[layer_idx]
            cur_unary = unary[layer_idx]
            cur_dp: list[float] = []
            cur_back: list[int] = []

            for cur_idx, cur_state in enumerate(cur_states):
                best_val = None
                best_prev = 0
                for prev_idx, prev_state in enumerate(prev_states):
                    value = (
                        dp[prev_idx]
                        + cur_unary[cur_idx]
                        - weight * total_transition_penalty(prev_state, cur_state)
                    )
                    if best_val is None or value > best_val:
                        best_val = value
                        best_prev = prev_idx
                cur_dp.append(best_val)
                cur_back.append(best_prev)

            dp = cur_dp
            backpointers.append(cur_back)

        best_last = max(range(len(dp)), key=lambda idx: dp[idx])
        best_score = dp[best_last]

        states = [candidate_lists[-1][best_last]]
        for layer_idx in range(len(candidate_lists) - 2, -1, -1):
            best_last = backpointers[layer_idx][best_last]
            states.append(candidate_lists[layer_idx][best_last])
        states.reverse()
        best_states = states
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms, counts, best_score, best_states


def format_state(state) -> str:
    return f"{state.tile_m}x{state.tile_n} {state.layout.name} {state.location.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="stress_gpu.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--wfc-runs", type=int, default=8)
    parser.add_argument("--dp-runs", type=int, default=20)
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
    dp_times, dp_counts, dp_score, dp_states = run_exact_dp(spec, args.dp_runs)
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

    print(f"\n[Naive Truncated Exhaustive]")
    print(f"  Candidate counts: {exhaustive_counts}")
    print(f"  Combinations: {combinations:,}")
    print(f"  Time: {exhaustive_ms:.3f} ms")
    print(f"  Best score: {best_score:.4f}")
    for name, state in zip([node.name for node in nodes], best_combo):
        print(f"  {name:12s} -> {format_state(state)}")

    print(f"\n[Exact Full DP]")
    print(f"  Candidate counts: {dp_counts}")
    print(f"  Time avg: {statistics.mean(dp_times):.3f} ms")
    print(f"  Time med: {statistics.median(dp_times):.3f} ms")
    print(f"  Time min/max: {min(dp_times):.3f} / {max(dp_times):.3f} ms")
    print(f"  Best score: {dp_score:.4f}")
    for name, state in zip([node.name for node in nodes], dp_states):
        print(f"  {name:12s} -> {format_state(state)}")

    quality = (wfc_score / best_score * 100.0) if best_score > 0 else 0.0
    quality_vs_dp = (wfc_score / dp_score * 100.0) if dp_score > 0 else 0.0
    speedup = exhaustive_ms / statistics.mean(wfc_times) if wfc_times else 0.0
    same_choice = all(node.collapsed_state == state for node, state in zip(nodes, best_combo))
    same_as_dp = all(node.collapsed_state == state for node, state in zip(nodes, dp_states))

    print(f"\n[Comparison]")
    print(f"  Quality vs truncated exhaustive: {quality:.4f}%")
    print(f"  Same as truncated exhaustive: {same_choice}")
    print(f"  Quality vs exact full DP: {quality_vs_dp:.4f}%")
    print(f"  Same as exact full DP: {same_as_dp}")
    print(f"  Speedup vs truncated exhaustive: {speedup:.1f}x")
    print(
        "\nNote: the truncated exhaustive baseline searches only the top-k unary "
        "survivors per layer after hard constraints. It is not guaranteed to "
        "contain the true optimum once pairwise transition costs are considered."
    )
    if not same_as_dp:
        print(
            "Warning: the current WFC heuristic does not match the full exact-DP "
            "optimum for this configuration."
        )


if __name__ == "__main__":
    main()
