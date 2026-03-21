"""HW-WFC Safety Checks.

PoC 결과가 "진짜 의미 있는 것인지" 검증하는 5가지 안전 점검.
너무 잘 나오는 결과를 의심하고, 알고리즘의 실제 한계를 드러낸다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import (
    LayerNode, LayerType, HWState,
    MemoryLayout, ComputeLocation, generate_default_candidates,
)
from src.constraint import (
    HardwareSpec, apply_hard_constraints, propagate_constraints,
    derive_transition_weight, total_transition_penalty,
)
from src.cost_model import compute_score, score_all_candidates
from src.scheduler import HWWFCScheduler


def load_spec() -> HardwareSpec:
    return HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json"
    )


# ═══════════════════════════════════════════════════════════════
# Safety Check #1: Hard Constraint가 실제로 올바르게 동작하는가?
#
# 의심: "36개 제거"가 진짜 물리적으로 불가능한 것만 제거했는지,
#        아니면 유효한 후보까지 잘못 걸러내고 있지는 않은지?
# ═══════════════════════════════════════════════════════════════
def check_1_hard_constraint_correctness():
    print("\n" + "=" * 60)
    print("  Safety Check #1: Hard Constraint 정확성")
    print("=" * 60)

    spec = load_spec()
    candidates = generate_default_candidates()

    print(f"\n  전체 후보: {len(candidates)}개")
    print(f"  SRAM 한도: {spec.sram_bytes} bytes ({spec.sram_bytes // 1024}KB)")
    print(f"  Alignment: {spec.alignment_bytes} bytes")

    # 각 후보를 수동으로 검사
    passed = []
    failed_sram = []
    failed_align = []
    for s in candidates:
        mem = s.memory_bytes(4)
        row_bytes = s.tile_n * 4
        sram_ok = (s.location == ComputeLocation.DRAM) or (mem <= spec.sram_bytes)
        align_ok = (row_bytes % spec.alignment_bytes == 0)

        if sram_ok and align_ok:
            passed.append(s)
        else:
            if not sram_ok:
                failed_sram.append((s, mem))
            if not align_ok:
                failed_align.append((s, row_bytes))

    print(f"\n  통과: {len(passed)}개")
    print(f"  SRAM 초과로 제거: {len(failed_sram)}개")
    for s, mem in failed_sram:
        print(f"    {s} → {mem} bytes ({mem/1024:.1f}KB) > {spec.sram_bytes // 1024}KB")
    print(f"  Alignment 위반으로 제거: {len(failed_align)}개")
    for s, rb in failed_align:
        print(f"    {s} → row_bytes={rb}, {rb} % {spec.alignment_bytes} = {rb % spec.alignment_bytes}")

    # 핵심 검증: DRAM 후보는 SRAM 제약에 걸리면 안 됨
    dram_in_failed = [s for s, _ in failed_sram if s.location == ComputeLocation.DRAM]
    if dram_in_failed:
        print(f"\n  !! BUG: DRAM 후보 {len(dram_in_failed)}개가 SRAM 제약에 잘못 걸림")
    else:
        print(f"\n  OK: DRAM 후보는 SRAM 제약을 올바르게 우회함")

    # 핵심 검증: 8x8 타일은 반드시 통과해야 함 (256 bytes << 64KB)
    tile_8x8_sram = [s for s in candidates
                     if s.tile_m == 8 and s.tile_n == 8
                     and s.location == ComputeLocation.SRAM]
    tile_8x8_passed = [s for s in passed
                       if s.tile_m == 8 and s.tile_n == 8
                       and s.location == ComputeLocation.SRAM]

    # 8x8 타일: row_bytes = 8*4 = 32. 32 % 128 = 32 ≠ 0 → alignment 위반!
    print(f"\n  주의: 8x8 SRAM 후보 {len(tile_8x8_sram)}개 중 {len(tile_8x8_passed)}개 통과")
    if len(tile_8x8_passed) < len(tile_8x8_sram):
        print(f"  → 8x8 타일이 alignment({spec.alignment_bytes}B)에 걸림.")
        print(f"     row_bytes = 8*4 = 32, 32 % 128 ≠ 0")
        print(f"  → 이것이 과도한 제거인지? 작은 타일도 쓸 수 있어야 하는 상황이라면 문제.")


# ═══════════════════════════════════════════════════════════════
# Safety Check #2: Cost Model이 합리적인 점수를 주는가?
#
# 의심: 모든 후보에 비슷한 점수를 주면 "엔트로피 기반 선택"이
#        사실상 랜덤 선택과 다를 바 없음.
#        반대로 한 후보만 압도적이면, Cost Model이 아니라
#        그냥 "SRAM에 가장 큰 타일"을 고르는 것과 같음.
# ═══════════════════════════════════════════════════════════════
def _run_discrimination_check(spec: HardwareSpec, spec_label: str):
    """단일 스펙에 대한 Cost Model 변별력 검사."""
    node = LayerNode(
        name="Test_Linear",
        layer_type=LayerType.LINEAR,
        dims={"M": 256, "N": 512, "K": 256},
        candidates=generate_default_candidates(),
    )

    apply_hard_constraints(node, spec)
    scored = score_all_candidates(node, spec)

    print(f"\n  [{spec_label}] Hard constraint 후 남은 후보: {len(scored)}개")
    print(f"  점수 분포:")
    for state, score in scored:
        bar = "█" * int(score * 40)
        print(f"    {state} → {score:.4f} {bar}")

    scores = [s for _, s in scored]
    if not scores:
        print(f"  !! 후보 없음")
        return

    max_s, min_s = max(scores), min(scores)
    spread = max_s - min_s

    # 동점 그룹 카운트 (소수점 4자리 기준)
    distinct_scores = set(round(s, 4) for s in scores)
    num_distinct = len(distinct_scores)
    num_tied = len(scores) - num_distinct

    print(f"\n  점수 범위: {min_s:.4f} ~ {max_s:.4f} (spread: {spread:.4f})")
    print(f"  고유 점수 수: {num_distinct}개 / 전체 {len(scores)}개 (동점: {num_tied}개)")

    if spread < 0.01:
        print(f"  !! 경고: 점수 spread가 {spread:.4f}로 매우 좁음")
        print(f"     → Cost Model이 사실상 후보를 구분하지 못함")
    elif num_distinct < 3:
        print(f"  !! 경고: 고유 점수가 {num_distinct}단계뿐 → 구분력 부족")
    else:
        print(f"  OK: spread={spread:.2f}, {num_distinct}단계 구분")
    if num_tied > 0:
        print(f"  주의: 동점 후보 {num_tied}개 존재 (같은 면적 직사각 타일 등)")


def check_2_cost_model_discrimination():
    print("\n" + "=" * 60)
    print("  Safety Check #2: Cost Model 변별력")
    print("=" * 60)

    # toy_gpu (64KB) — 넉넉한 환경
    spec_toy = load_spec()
    _run_discrimination_check(spec_toy, "toy_gpu 64KB")

    # stress_gpu (12KB) — 핵심 검증 환경
    stress_spec_path = Path(__file__).resolve().parent.parent / "specs" / "stress_gpu.json"
    spec_stress = HardwareSpec.from_json(stress_spec_path)
    _run_discrimination_check(spec_stress, "stress_gpu 12KB")


# ═══════════════════════════════════════════════════════════════
# Safety Check #3: 제약 전파가 과도하게 후보를 제거하는가?
#
# 의심: penalty_threshold=1.5가 너무 관대하거나 너무 엄격할 수 있음.
#        관대하면 전파가 사실상 안 되고,
#        엄격하면 유효한 후보도 날아감.
# ═══════════════════════════════════════════════════════════════
def check_3_propagation_aggressiveness():
    print("\n" + "=" * 60)
    print("  Safety Check #3: 제약 전파 강도")
    print("=" * 60)

    spec = load_spec()

    # threshold별 제거 비율 비교
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

    for threshold in thresholds:
        src = LayerNode(
            name="Src", layer_type=LayerType.LINEAR,
            dims={"M": 256, "N": 512, "K": 256},
            candidates=generate_default_candidates(),
        )
        dst = LayerNode(
            name="Dst", layer_type=LayerType.RELU,
            dims={"M": 256, "N": 512},
            candidates=generate_default_candidates(),
        )

        apply_hard_constraints(src, spec)
        apply_hard_constraints(dst, spec)

        dst_before = len(dst.candidates)

        # src를 ROW_MAJOR로 붕괴
        row_major_sram = next(
            (s for s in src.candidates
             if s.layout == MemoryLayout.ROW_MAJOR
             and s.location == ComputeLocation.SRAM),
            None
        )
        if row_major_sram:
            src.collapse_to(row_major_sram)

        removed = propagate_constraints(src, dst, penalty_threshold=threshold)
        dst_after = len(dst.candidates)

        pct = (removed / dst_before * 100) if dst_before > 0 else 0
        print(f"  threshold={threshold:.1f}: {dst_before} → {dst_after} ({pct:.0f}% 제거)")

        # 남은 후보의 레이아웃 분포
        layouts = {}
        for c in dst.candidates:
            key = f"{c.layout.name}/{c.location.name}"
            layouts[key] = layouts.get(key, 0) + 1
        layout_str = ", ".join(f"{k}:{v}" for k, v in sorted(layouts.items()))
        print(f"           남은 분포: [{layout_str}]")

    print(f"\n  현재 사용 중인 threshold: 1.5")
    print(f"  → 전파가 실질적 효과를 발휘하는지, 아니면 명목상인지 확인 필요")


# ═══════════════════════════════════════════════════════════════
# Safety Check #4: 백트래킹이 실제로 작동하는 시나리오 구성
#
# 의심: Toy model에서 백트래킹 0회 = "필요 없었다"가 아니라
#        "테스트 안 된 코드"일 수 있음.
#        의도적으로 모순을 유발해서 백트래킹 경로를 검증한다.
# ═══════════════════════════════════════════════════════════════
def check_4_backtracking_actually_works():
    print("\n" + "=" * 60)
    print("  Safety Check #4: 백트래킹 실제 동작 검증")
    print("=" * 60)

    spec = load_spec()

    # ── 시나리오 4a: 양쪽 갈등 (이전과 동일) ──
    print(f"\n  [4a] A(ROW only) → B(3 options) → C(COL only)")
    node_a = LayerNode(
        name="Forced_RowMajor",
        layer_type=LayerType.LINEAR,
        dims={"M": 256, "N": 512, "K": 256},
        candidates=[
            HWState(32, 32, MemoryLayout.ROW_MAJOR, ComputeLocation.SRAM),
        ],
    )
    node_b = LayerNode(
        name="Middle_Layer",
        layer_type=LayerType.RELU,
        dims={"M": 256, "N": 512},
        candidates=[
            HWState(32, 32, MemoryLayout.ROW_MAJOR, ComputeLocation.SRAM),
            HWState(32, 32, MemoryLayout.COL_MAJOR, ComputeLocation.SRAM),
            HWState(32, 32, MemoryLayout.BLOCK_TILED, ComputeLocation.SRAM),
        ],
    )
    node_c = LayerNode(
        name="Forced_ColMajor",
        layer_type=LayerType.LINEAR,
        dims={"M": 256, "N": 128, "K": 512},
        candidates=[
            HWState(32, 32, MemoryLayout.COL_MAJOR, ComputeLocation.SRAM),
        ],
    )
    nodes = [node_a, node_b, node_c]
    scheduler = HWWFCScheduler(spec, penalty_threshold=1.0)
    result = scheduler.schedule(nodes)
    for step in result.steps:
        print(f"    {step}")
    print(f"  결과: {'SUCCESS' if result.success else 'FAILED'}, 백트래킹: {result.backtracks_used}회")
    if result.success:
        for n in nodes:
            print(f"    {n.name}: {n.collapsed_state}")

    # ── 시나리오 4b: 진짜 모순 (해가 없어야 함) ──
    print(f"\n  [4b] A(ROW only) → B(ROW only) → C(COL only), threshold=0.5")
    print(f"  → B가 ROW인데 C가 COL이면 penalty=1.0 > 0.5 → 전파 실패 → 백트래킹 필요")
    node_a2 = LayerNode(
        name="A_Row", layer_type=LayerType.LINEAR,
        dims={"M": 256, "N": 512, "K": 256},
        candidates=[HWState(32, 32, MemoryLayout.ROW_MAJOR, ComputeLocation.SRAM)],
    )
    node_b2 = LayerNode(
        name="B_Mid", layer_type=LayerType.RELU,
        dims={"M": 256, "N": 512},
        candidates=[
            HWState(32, 32, MemoryLayout.ROW_MAJOR, ComputeLocation.SRAM),
            HWState(32, 32, MemoryLayout.COL_MAJOR, ComputeLocation.SRAM),
        ],
    )
    node_c2 = LayerNode(
        name="C_Col", layer_type=LayerType.LINEAR,
        dims={"M": 256, "N": 128, "K": 512},
        candidates=[HWState(32, 32, MemoryLayout.COL_MAJOR, ComputeLocation.SRAM)],
    )
    nodes2 = [node_a2, node_b2, node_c2]
    scheduler2 = HWWFCScheduler(spec, penalty_threshold=0.5)
    result2 = scheduler2.schedule(nodes2)
    for step in result2.steps:
        print(f"    {step}")
    print(f"  결과: {'SUCCESS' if result2.success else 'FAILED'}, 백트래킹: {result2.backtracks_used}회")
    if result2.success:
        for n in nodes2:
            print(f"    {n.name}: {n.collapsed_state}")
    if result2.backtracks_used > 0:
        print(f"  OK: 백트래킹 실제 발동 확인됨")
    else:
        print(f"  !! 백트래킹이 여전히 발동하지 않음")


# ═══════════════════════════════════════════════════════════════
# Safety Check #5: 레이어 수 증가 시 스케일링
#
# 의심: 3개 레이어에서 96.7% 감소는 인상적이지만,
#        10개, 20개, 50개로 늘리면 어떻게 되는가?
#        제약 전파가 체인 끝까지 도달하는가?
#        아니면 중간에 끊기는가?
# ═══════════════════════════════════════════════════════════════
def check_5_scaling():
    print("\n" + "=" * 60)
    print("  Safety Check #5: 레이어 수 증가 + 이질적 레이어")
    print("=" * 60)

    spec = load_spec()

    # ── 5a: 동질적 레이어 (이전과 동일) ──
    print(f"\n  [5a] 동질적 레이어 (Linear+ReLU 반복)")
    for num_layers in [3, 10, 50]:
        nodes = []
        for i in range(num_layers):
            if i % 2 == 0:
                node = LayerNode(
                    name=f"Linear_{i}", layer_type=LayerType.LINEAR,
                    dims={"M": 256, "N": 512, "K": 256},
                    candidates=generate_default_candidates(),
                )
            else:
                node = LayerNode(
                    name=f"ReLU_{i}", layer_type=LayerType.RELU,
                    dims={"M": 256, "N": 512},
                    candidates=generate_default_candidates(),
                )
            nodes.append(node)

        scheduler = HWWFCScheduler(spec, penalty_threshold=1.5)
        result = scheduler.schedule(nodes)
        reduction = (1 - result.final_search_space / result.initial_search_space) * 100 if result.initial_search_space > 0 else 0
        all_same = all(n.collapsed_state == nodes[0].collapsed_state for n in nodes) if result.success else False

        print(f"    {num_layers:3d} layers: {'OK' if result.success else 'FAIL'} | "
              f"{result.initial_search_space} → {result.final_search_space} ({reduction:.1f}%) | "
              f"all_same={all_same}")

    # ── 5b: 이질적 레이어 (stress_gpu 12KB — SRAM이 tight한 환경) ──
    # 현재 heuristic에서는 12KB SRAM에서 레이어별 다른 타일이 선택되는지 확인한다.
    # 이 검사는 global optimum 증명이 아니라 "복사기 문제를 피했는가"를 본다.
    stress_spec_path = Path(__file__).resolve().parent.parent / "specs" / "stress_gpu.json"
    stress_spec = HardwareSpec.from_json(stress_spec_path)

    print(f"\n  [5b] 이질적 레이어 (stress_gpu 12KB SRAM):")
    print(f"       QK MatMul → Softmax(seq=2048) → AV MatMul → LayerNorm")
    nodes = [
        LayerNode(
            name="QK_MatMul", layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 2048, "K": 64},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="Softmax", layer_type=LayerType.SOFTMAX,
            dims={"M": 2048, "N": 2048},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="AV_MatMul", layer_type=LayerType.LINEAR,
            dims={"M": 2048, "N": 64, "K": 2048},
            candidates=generate_default_candidates(),
        ),
        LayerNode(
            name="LayerNorm", layer_type=LayerType.LAYERNORM,
            dims={"M": 2048, "N": 512},
            candidates=generate_default_candidates(),
        ),
    ]

    scheduler = HWWFCScheduler(stress_spec, penalty_threshold=1.5)
    result = scheduler.schedule(nodes)
    for step in result.steps:
        print(f"    {step}")

    print(f"\n  결과:")
    all_same = True
    if result.success:
        first_state = nodes[0].collapsed_state
        for n in nodes:
            same = "==" if n.collapsed_state == first_state else "!="
            print(f"    {n.name}: {n.collapsed_state} {same} first")
            if n.collapsed_state != first_state:
                all_same = False
    print(f"\n  all_same_state={all_same}")
    if all_same:
        print(f"  !! 이질적 레이어인데도 동일 상태 → 복사기 문제 미해결")
    else:
        print(f"  OK: 레이어별 다른 상태를 선택함 → 복사기 문제는 피함")

    # ── 5b 추가 검증: WFC 차별화가 "올바른 방향"인지 exact DP와 비교 ──
    if result.success:
        print(f"\n  [5b-DP] Exact DP로 같은 문제를 풀어 WFC 차별화 방향 검증:")
        weight = derive_transition_weight(stress_spec)

        # DP용 노드 재생성
        dp_nodes = [
            LayerNode(name="QK_MatMul", layer_type=LayerType.LINEAR,
                      dims={"M": 2048, "N": 2048, "K": 64},
                      candidates=generate_default_candidates()),
            LayerNode(name="Softmax", layer_type=LayerType.SOFTMAX,
                      dims={"M": 2048, "N": 2048},
                      candidates=generate_default_candidates()),
            LayerNode(name="AV_MatMul", layer_type=LayerType.LINEAR,
                      dims={"M": 2048, "N": 64, "K": 2048},
                      candidates=generate_default_candidates()),
            LayerNode(name="LayerNorm", layer_type=LayerType.LAYERNORM,
                      dims={"M": 2048, "N": 512},
                      candidates=generate_default_candidates()),
        ]
        for n in dp_nodes:
            apply_hard_constraints(n, stress_spec)

        candidate_lists = [list(n.candidates) for n in dp_nodes]
        unary = [
            [compute_score(n, s, stress_spec) for s in states]
            for n, states in zip(dp_nodes, candidate_lists)
        ]

        # Viterbi DP
        dp = unary[0][:]
        backpointers: list[list[int]] = []
        for layer_idx in range(1, len(candidate_lists)):
            prev_states = candidate_lists[layer_idx - 1]
            cur_states = candidate_lists[layer_idx]
            cur_unary = unary[layer_idx]
            cur_dp: list[float] = []
            cur_back: list[int] = []
            for ci, cs in enumerate(cur_states):
                best_val = None
                best_prev = 0
                for pi, ps in enumerate(prev_states):
                    val = dp[pi] + cur_unary[ci] - weight * total_transition_penalty(ps, cs)
                    if best_val is None or val > best_val:
                        best_val = val
                        best_prev = pi
                cur_dp.append(best_val)
                cur_back.append(best_prev)
            dp = cur_dp
            backpointers.append(cur_back)

        best_last = max(range(len(dp)), key=lambda i: dp[i])
        dp_states = [candidate_lists[-1][best_last]]
        for li in range(len(candidate_lists) - 2, -1, -1):
            best_last = backpointers[li][best_last]
            dp_states.append(candidate_lists[li][best_last])
        dp_states.reverse()

        dp_all_same = all(s == dp_states[0] for s in dp_states)
        for n, s in zip(dp_nodes, dp_states):
            print(f"    DP {n.name}: {s}")
        print(f"  DP all_same_state={dp_all_same}")

        # WFC와 DP 차별화 방향 비교
        wfc_diff_layers = [n.name for n in nodes if n.collapsed_state != nodes[0].collapsed_state]
        dp_diff_layers = [n.name for n, s in zip(dp_nodes, dp_states) if s != dp_states[0]]

        if set(wfc_diff_layers) == set(dp_diff_layers):
            print(f"  OK: WFC와 DP가 같은 레이어({wfc_diff_layers})에서 차별화 → 방향 일치")
        elif not dp_all_same and not all_same:
            print(f"  주의: 둘 다 차별화하지만 위치가 다름")
            print(f"    WFC 차별화: {wfc_diff_layers}")
            print(f"    DP  차별화: {dp_diff_layers}")
        elif dp_all_same and not all_same:
            print(f"  !! 경고: DP는 전부 같은 상태를 선택하지만 WFC만 다름 → WFC 차별화가 최적이 아닐 수 있음")
        else:
            print(f"  !! 경고: DP는 차별화하지만 WFC는 못 함")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  HW-WFC Safety Checks — 결과가 진짜인지 의심하기        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    check_1_hard_constraint_correctness()
    check_2_cost_model_discrimination()
    check_3_propagation_aggressiveness()
    check_4_backtracking_actually_works()
    check_5_scaling()

    print("\n" + "=" * 60)
    print("  Safety Checks 완료 — 위 결과를 바탕으로 연구 노트 작성 필요")
    print("=" * 60)


if __name__ == "__main__":
    main()
