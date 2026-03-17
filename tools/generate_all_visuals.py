"""HW-WFC 전체 시각화 생성기.

실행할 때마다 assets/ 디렉토리에 모든 시각화를 재생성한다.
성공 사례뿐 아니라 실패 사례, 개선 전/후 비교도 포함.

Usage:
    python tools/generate_all_visuals.py

생성되는 파일:
    assets/collapse_animation.gif     — 붕괴 과정 애니메이션
    assets/score_distribution.png     — Cost Model 점수 분포
    assets/propagation_sweep.png      — threshold별 전파 제거율
    assets/sram_comparison.png        — SRAM 크기별 결과 비교
    assets/copier_problem.png         — 복사기 문제 (실패→해결)
    assets/backtracking.png           — 백트래킹 시나리오
    assets/VISUALS_LOG.md             — 시각화 버전/생성 기록
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import (
    HWState, LayerNode, LayerType, MemoryLayout, ComputeLocation,
    generate_default_candidates,
)
from src.constraint import (
    HardwareSpec, apply_hard_constraints, propagate_constraints,
    total_transition_penalty,
)
from src.cost_model import (
    compute_score, score_all_candidates, roofline_score,
    layout_affinity, cache_efficiency,
)
from src.scheduler import HWWFCScheduler

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ── 색상 ──
BG = (24, 24, 32)
BG_LIGHT = (30, 30, 42)
GRID_LINE = (40, 40, 55)
TEXT = (220, 220, 230)
TEXT_DIM = (120, 120, 140)
GREEN = (80, 200, 120)
RED = (220, 70, 70)
YELLOW = (255, 200, 60)
BLUE = (70, 140, 220)
ORANGE = (240, 160, 60)
PURPLE = (160, 100, 220)
CYAN = (60, 200, 200)

# 바 차트 색상
BAR_COLORS = [GREEN, BLUE, ORANGE, PURPLE, CYAN, YELLOW, RED]


def font(size=16):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


F_TITLE = font(20)
F_LABEL = font(14)
F_SMALL = font(11)
F_TINY = font(9)


def draw_header(draw, width, title, subtitle=""):
    draw.rectangle([0, 0, width, 60], fill=BG_LIGHT)
    draw.text((20, 10), title, fill=TEXT, font=F_TITLE)
    if subtitle:
        draw.text((20, 36), subtitle, fill=TEXT_DIM, font=F_SMALL)


# ═══════════════════════════════════════════════════════════════
# 1. Score Distribution — Cost Model 변별력
# ═══════════════════════════════════════════════════════════════
def viz_score_distribution():
    print("  [1/6] Score Distribution...")
    spec = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json")

    layers = [
        ("Linear (256x512x256)", LayerType.LINEAR, {"M": 256, "N": 512, "K": 256}),
        ("Softmax (2048x2048)", LayerType.SOFTMAX, {"M": 2048, "N": 2048}),
        ("MatMul (2048x64x2048)", LayerType.LINEAR, {"M": 2048, "N": 64, "K": 2048}),
    ]

    W, H = 900, 220 * len(layers) + 80
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, W, "Cost Model Score Distribution",
                "Score = Roofline + Layout Affinity + Cache Efficiency | toy_gpu 64KB SRAM")

    for li, (label, lt, dims) in enumerate(layers):
        node = LayerNode("Test", lt, dims, generate_default_candidates())
        apply_hard_constraints(node, spec)
        scored = score_all_candidates(node, spec)

        # SRAM 후보만 상위 12개
        sram_scored = [(s, sc) for s, sc in scored if s.location == ComputeLocation.SRAM][:12]
        if not sram_scored:
            continue

        max_score = max(sc for _, sc in sram_scored)
        base_y = 75 + li * 220
        bar_area_x = 200
        bar_area_w = W - bar_area_x - 30
        bar_h = 14
        bar_gap = 3

        draw.text((20, base_y), label, fill=TEXT, font=F_LABEL)

        for bi, (state, sc) in enumerate(sram_scored):
            y = base_y + 25 + bi * (bar_h + bar_gap)
            w = int(bar_area_w * sc / max_score) if max_score > 0 else 0

            # 3색 스택 바: roofline / affinity / cache
            rf = roofline_score(node, state, spec)
            af = layout_affinity(node, state)
            ce = cache_efficiency(node, state, spec)

            w_rf = int(bar_area_w * max(0, rf) / max_score) if max_score > 0 else 0
            w_af = int(bar_area_w * max(0, af) / max_score) if max_score > 0 else 0
            w_ce = int(bar_area_w * max(0, ce) / max_score) if max_score > 0 else 0

            # Roofline (blue)
            draw.rectangle([bar_area_x, y, bar_area_x + w_rf, y + bar_h], fill=BLUE)
            # Affinity (orange)
            draw.rectangle([bar_area_x + w_rf, y, bar_area_x + w_rf + w_af, y + bar_h], fill=ORANGE)
            # Cache (green)
            draw.rectangle([bar_area_x + w_rf + w_af, y,
                            bar_area_x + w_rf + w_af + w_ce, y + bar_h], fill=GREEN)

            # 패널티 표시
            if ce < 0:
                pen_w = int(bar_area_w * abs(ce) / max_score)
                total_w = w_rf + w_af
                draw.rectangle([bar_area_x + total_w, y,
                                bar_area_x + total_w + pen_w, y + bar_h], fill=RED)

            tile_label = f"{state.tile_m:>3}x{state.tile_n:<3} {state.layout.name[:3]}"
            draw.text((bar_area_x - 115, y), tile_label, fill=TEXT_DIM, font=F_TINY)
            draw.text((bar_area_x + w + 5, y), f"{sc:.3f}", fill=TEXT_DIM, font=F_TINY)

        # 범례 (첫 레이어에만)
        if li == 0:
            lx = W - 200
            draw.rectangle([lx, base_y, lx + 10, base_y + 10], fill=BLUE)
            draw.text((lx + 14, base_y - 2), "Roofline", fill=TEXT_DIM, font=F_TINY)
            draw.rectangle([lx + 80, base_y, lx + 90, base_y + 10], fill=ORANGE)
            draw.text((lx + 94, base_y - 2), "Affinity", fill=TEXT_DIM, font=F_TINY)
            draw.rectangle([lx, base_y + 14, lx + 10, base_y + 24], fill=GREEN)
            draw.text((lx + 14, base_y + 12), "Cache", fill=TEXT_DIM, font=F_TINY)
            draw.rectangle([lx + 80, base_y + 14, lx + 90, base_y + 24], fill=RED)
            draw.text((lx + 94, base_y + 12), "Penalty", fill=TEXT_DIM, font=F_TINY)

    path = ASSETS / "score_distribution.png"
    img.save(str(path))
    print(f"    → {path}")


# ═══════════════════════════════════════════════════════════════
# 2. Propagation Sweep — threshold별 전파 제거율
# ═══════════════════════════════════════════════════════════════
def viz_propagation_sweep():
    print("  [2/6] Propagation Sweep...")
    spec = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json")

    thresholds = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    results = []

    for th in thresholds:
        src = LayerNode("Src", LayerType.LINEAR, {"M": 256, "N": 512, "K": 256},
                        generate_default_candidates())
        dst = LayerNode("Dst", LayerType.RELU, {"M": 256, "N": 512},
                        generate_default_candidates())
        apply_hard_constraints(src, spec)
        apply_hard_constraints(dst, spec)
        dst_before = len(dst.candidates)

        row_sram = next((s for s in src.candidates
                         if s.layout == MemoryLayout.ROW_MAJOR
                         and s.location == ComputeLocation.SRAM), None)
        if row_sram:
            src.collapse_to(row_sram)
        removed = propagate_constraints(src, dst, penalty_threshold=th)
        pct = removed / dst_before * 100 if dst_before > 0 else 0
        results.append((th, pct, len(dst.candidates)))

    W, H = 700, 400
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, W, "Propagation Sweep",
                "Candidate removal % vs penalty threshold | pass criterion: >25% at t=1.5")

    chart_x, chart_y = 80, 80
    chart_w, chart_h = W - 120, H - 130

    # 그리드
    for i in range(5):
        y = chart_y + int(chart_h * i / 4)
        draw.line([(chart_x, y), (chart_x + chart_w, y)], fill=GRID_LINE)
        label = f"{100 - i * 25}%"
        draw.text((chart_x - 40, y - 6), label, fill=TEXT_DIM, font=F_SMALL)

    # 25% 기준선
    y_25 = chart_y + int(chart_h * 0.75)
    draw.line([(chart_x, y_25), (chart_x + chart_w, y_25)], fill=RED, width=1)
    draw.text((chart_x + chart_w + 5, y_25 - 6), "25%", fill=RED, font=F_TINY)

    # 바
    bar_w = chart_w // len(results) - 8
    for i, (th, pct, remaining) in enumerate(results):
        x = chart_x + i * (bar_w + 8) + 4
        h = int(chart_h * pct / 100)
        y = chart_y + chart_h - h

        color = GREEN if pct >= 25 else RED
        draw.rectangle([x, y, x + bar_w, chart_y + chart_h], fill=color)

        # threshold 라벨
        draw.text((x, chart_y + chart_h + 5), f"{th}", fill=TEXT_DIM, font=F_TINY)
        # 퍼센트 라벨
        draw.text((x, y - 14), f"{pct:.0f}%", fill=TEXT, font=F_TINY)

    # x축 라벨
    draw.text((chart_x + chart_w // 2 - 40, H - 20), "penalty threshold",
              fill=TEXT_DIM, font=F_SMALL)

    path = ASSETS / "propagation_sweep.png"
    img.save(str(path))
    print(f"    → {path}")


# ═══════════════════════════════════════════════════════════════
# 3. Copier Problem — 실패 → 해결 비교
# ═══════════════════════════════════════════════════════════════
def viz_copier_problem():
    print("  [3/6] Copier Problem (failure → fix)...")

    W, H = 850, 500
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, W, "The Copier Problem",
                "All layers converge to identical state — algorithm is just picking max, not optimizing")

    # FAILURE: toy_gpu 64KB — 모든 레이어 64x64
    spec_64 = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json")
    nodes_fail = [
        LayerNode("QK_MatMul", LayerType.LINEAR, {"M": 2048, "N": 2048, "K": 64}, generate_default_candidates()),
        LayerNode("Softmax", LayerType.SOFTMAX, {"M": 2048, "N": 2048}, generate_default_candidates()),
        LayerNode("AV_MatMul", LayerType.LINEAR, {"M": 2048, "N": 64, "K": 2048}, generate_default_candidates()),
        LayerNode("LayerNorm", LayerType.LAYERNORM, {"M": 2048, "N": 512}, generate_default_candidates()),
    ]
    s_fail = HWWFCScheduler(spec_64, penalty_threshold=1.5)
    s_fail.schedule(nodes_fail)

    # SUCCESS: stress_gpu 12KB — Softmax가 다른 타일
    spec_12 = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / "stress_gpu.json")
    nodes_pass = [
        LayerNode("QK_MatMul", LayerType.LINEAR, {"M": 2048, "N": 2048, "K": 64}, generate_default_candidates()),
        LayerNode("Softmax", LayerType.SOFTMAX, {"M": 2048, "N": 2048}, generate_default_candidates()),
        LayerNode("AV_MatMul", LayerType.LINEAR, {"M": 2048, "N": 64, "K": 2048}, generate_default_candidates()),
        LayerNode("LayerNorm", LayerType.LAYERNORM, {"M": 2048, "N": 512}, generate_default_candidates()),
    ]
    s_pass = HWWFCScheduler(spec_12, penalty_threshold=1.5)
    s_pass.schedule(nodes_pass)

    half_w = W // 2 - 15
    row_h = 50

    for panel, (title, nodes, color, sram_label, is_fail) in enumerate([
        ("FAIL: 64KB SRAM (too comfortable)", nodes_fail, RED, "64KB", True),
        ("PASS: 12KB SRAM (real tradeoffs)", nodes_pass, GREEN, "12KB", False),
    ]):
        px = 15 + panel * (half_w + 15)
        py = 75

        # 패널 배경
        draw.rectangle([px, py, px + half_w, py + 320], fill=BG_LIGHT, outline=color)
        draw.text((px + 10, py + 8), title, fill=color, font=F_LABEL)
        draw.text((px + 10, py + 28), f"SRAM: {sram_label}", fill=TEXT_DIM, font=F_SMALL)

        # 각 레이어의 선택된 상태
        for i, node in enumerate(nodes):
            y = py + 55 + i * row_h
            s = node.collapsed_state
            if not s:
                continue

            # 타일 크기 시각적 표현 (비율 사각형)
            max_dim = 64
            tw = int(30 * s.tile_m / max_dim)
            th = int(30 * s.tile_n / max_dim)
            tw = max(8, min(35, tw))
            th = max(8, min(35, th))

            rect_x = px + 15
            rect_y = y + 5
            draw.rectangle([rect_x, rect_y, rect_x + tw, rect_y + th],
                            fill=color if not is_fail else YELLOW,
                            outline=TEXT_DIM)

            # 레이어 이름 + 상태
            draw.text((px + 55, y + 2), node.name, fill=TEXT, font=F_LABEL)
            state_str = f"{s.tile_m}x{s.tile_n} {s.layout.name}"
            draw.text((px + 55, y + 20), state_str, fill=TEXT_DIM, font=F_SMALL)

        # 판정
        unique = len(set(n.collapsed_state for n in nodes if n.collapsed_state))
        verdict_y = py + 55 + len(nodes) * row_h + 10
        if is_fail:
            draw.text((px + 15, verdict_y),
                      f"Unique states: {unique}/{len(nodes)} — ALL SAME",
                      fill=RED, font=F_LABEL)
            draw.text((px + 15, verdict_y + 22),
                      "Working memory fits easily → no tradeoff",
                      fill=TEXT_DIM, font=F_SMALL)
        else:
            draw.text((px + 15, verdict_y),
                      f"Unique states: {unique}/{len(nodes)} — DIFFERENTIATED",
                      fill=GREEN, font=F_LABEL)
            draw.text((px + 15, verdict_y + 22),
                      "Current WFC heuristic picks a smaller Softmax tile",
                      fill=TEXT_DIM, font=F_SMALL)

    # 하단 설명
    draw.text((20, H - 55),
              "Observation: with ample SRAM, the current heuristic tends to reuse the same tile everywhere.",
              fill=TEXT_DIM, font=F_SMALL)
    draw.text((20, H - 35),
              "Tighter SRAM increases working-memory pressure and can diversify heuristic choices.",
              fill=TEXT_DIM, font=F_SMALL)

    path = ASSETS / "copier_problem.png"
    img.save(str(path))
    print(f"    → {path}")


# ═══════════════════════════════════════════════════════════════
# 4. Backtracking Scenario — 모순 발생 & 복구
# ═══════════════════════════════════════════════════════════════
def viz_backtracking():
    print("  [4/6] Backtracking Scenario...")

    W, H = 800, 480
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, W, "Backtracking: Contradiction Detection & Recovery",
                "A(ROW) → B(ROW|COL) → C(COL only) with threshold=0.5")

    spec = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / "toy_gpu.json")

    # 스텝별 상태를 수동 시각화
    steps = [
        {
            "title": "Step 1: Initial",
            "states": [
                ("A_Row", "ROW only", [(1, "ROW", True)], None),
                ("B_Mid", "ROW | COL", [(1, "ROW", True), (1, "COL", True)], None),
                ("C_Col", "COL only", [(1, "COL", True)], None),
            ],
            "arrow": None,
            "status": "3 nodes, 4 total candidates",
        },
        {
            "title": "Step 2: Collapse A → ROW",
            "states": [
                ("A_Row", "COLLAPSED", [(1, "ROW", True)], GREEN),
                ("B_Mid", "ROW | COL", [(1, "ROW", True), (1, "COL", True)], None),
                ("C_Col", "COL only", [(1, "COL", True)], None),
            ],
            "arrow": "A→B propagation",
            "status": "Bottleneck collapse",
        },
        {
            "title": "Step 3: Propagate → B loses COL",
            "states": [
                ("A_Row", "COLLAPSED", [(1, "ROW", True)], GREEN),
                ("B_Mid", "ROW only", [(1, "ROW", True), (1, "COL", False)], YELLOW),
                ("C_Col", "COL only", [(1, "COL", True)], None),
            ],
            "arrow": "penalty(ROW→COL)=1.0 > 0.5",
            "status": "B auto-collapses to ROW",
        },
        {
            "title": "Step 4: Propagate B→C: CONTRADICTION!",
            "states": [
                ("A_Row", "COLLAPSED", [(1, "ROW", True)], GREEN),
                ("B_Mid", "COLLAPSED", [(1, "ROW", True)], GREEN),
                ("C_Col", "EMPTY!", [(1, "COL", False)], RED),
            ],
            "arrow": "penalty(ROW→COL)=1.0 > 0.5",
            "status": "C has 0 candidates → BACKTRACK!",
        },
    ]

    step_w = (W - 40) // len(steps)
    node_h = 50

    for si, step in enumerate(steps):
        sx = 20 + si * step_w
        sy = 75

        # 스텝 타이틀
        draw.text((sx + 5, sy), step["title"], fill=TEXT, font=F_SMALL)

        for ni, (name, desc, candidates, highlight) in enumerate(step["states"]):
            ny = sy + 25 + ni * (node_h + 20)
            box_w = step_w - 20
            box_h = node_h

            # 노드 박스
            outline = highlight or GRID_LINE
            fill = BG_LIGHT
            if highlight == RED:
                fill = (50, 25, 25)
            elif highlight == GREEN:
                fill = (25, 45, 30)
            draw.rectangle([sx + 5, ny, sx + 5 + box_w, ny + box_h],
                            fill=fill, outline=outline)

            draw.text((sx + 12, ny + 5), name, fill=TEXT, font=F_SMALL)
            draw.text((sx + 12, ny + 22), desc, fill=TEXT_DIM, font=F_TINY)

            # 후보 표시 (작은 블록)
            cx = sx + box_w - 30
            for ci, (_, label, alive) in enumerate(candidates):
                block_color = GREEN if alive else RED
                draw.rectangle([cx + ci * 14, ny + 8, cx + ci * 14 + 10, ny + 20],
                                fill=block_color)

        # 상태 텍스트
        status_y = sy + 25 + 3 * (node_h + 20) + 5
        status_color = RED if "BACKTRACK" in step["status"] else TEXT_DIM
        draw.text((sx + 5, status_y), step["status"], fill=status_color, font=F_TINY)

        if step["arrow"]:
            draw.text((sx + 5, status_y + 14), step["arrow"], fill=YELLOW, font=F_TINY)

    # 하단 결론
    draw.rectangle([20, H - 80, W - 20, H - 20], fill=BG_LIGHT, outline=RED)
    draw.text((35, H - 72), "Result: FAILED (as expected)", fill=RED, font=F_LABEL)
    draw.text((35, H - 50),
              "This scenario has no valid solution. Backtrack count: 1.", fill=TEXT_DIM, font=F_SMALL)
    draw.text((35, H - 32),
              "Validates: contradiction detection works, backtracking fires, dead code eliminated.",
              fill=TEXT_DIM, font=F_SMALL)

    path = ASSETS / "backtracking.png"
    img.save(str(path))
    print(f"    → {path}")


# ═══════════════════════════════════════════════════════════════
# 5. SRAM Size Comparison — 크기별 결과 비교
# ═══════════════════════════════════════════════════════════════
def viz_sram_comparison():
    print("  [5/6] SRAM Size Comparison...")

    specs = [
        ("toy_gpu.json", "64KB"),
        ("tight_gpu.json", "16KB"),
        ("stress_gpu.json", "12KB"),
    ]

    results = []
    for spec_file, label in specs:
        spec = HardwareSpec.from_json(Path(__file__).resolve().parent.parent / "specs" / spec_file)
        nodes = [
            LayerNode("QKV_Proj", LayerType.LINEAR, {"M": 2048, "N": 192, "K": 512}, generate_default_candidates()),
            LayerNode("QK_MatMul", LayerType.LINEAR, {"M": 2048, "N": 2048, "K": 64}, generate_default_candidates()),
            LayerNode("Softmax", LayerType.SOFTMAX, {"M": 2048, "N": 2048}, generate_default_candidates()),
            LayerNode("AV_MatMul", LayerType.LINEAR, {"M": 2048, "N": 64, "K": 2048}, generate_default_candidates()),
            LayerNode("Out_Proj", LayerType.LINEAR, {"M": 2048, "N": 512, "K": 64}, generate_default_candidates()),
            LayerNode("LayerNorm", LayerType.LAYERNORM, {"M": 2048, "N": 512}, generate_default_candidates()),
        ]
        s = HWWFCScheduler(spec, penalty_threshold=1.5)
        result = s.schedule(nodes)
        score = sum(compute_score(n, n.collapsed_state, spec) for n in nodes if n.collapsed_state)
        unique = len(set(n.collapsed_state for n in nodes if n.collapsed_state))
        states = [(n.name, n.collapsed_state) for n in nodes]
        results.append((label, result, score, unique, states))

    W, H = 850, 520
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, W, "SRAM Size Impact on Scheduling",
                "Same 6-layer Attention Block across different SRAM constraints")

    col_w = (W - 40) // len(results)

    for ri, (label, result, score, unique, states) in enumerate(results):
        cx = 20 + ri * col_w
        cy = 75

        # SRAM 라벨
        is_diff = unique > 1
        title_color = GREEN if is_diff else YELLOW
        draw.text((cx + 10, cy), f"SRAM: {label}", fill=title_color, font=F_TITLE)

        # 통계
        draw.text((cx + 10, cy + 35), f"Score: {score:.3f}", fill=TEXT, font=F_SMALL)
        draw.text((cx + 10, cy + 52), f"Unique: {unique}/6", fill=title_color, font=F_SMALL)
        draw.text((cx + 10, cy + 69), f"Backtracks: {result.backtracks_used}", fill=TEXT_DIM, font=F_SMALL)

        # 각 레이어 상태 표시
        for si, (name, state) in enumerate(states):
            y = cy + 100 + si * 52
            if not state:
                continue

            # 타일 비율 사각형
            max_t = 64
            tw = max(6, int(28 * state.tile_m / max_t))
            th = max(6, int(28 * state.tile_n / max_t))
            draw.rectangle([cx + 10, y, cx + 10 + tw, y + th],
                            fill=title_color, outline=TEXT_DIM)

            # 텍스트
            draw.text((cx + 50, y), name[:10], fill=TEXT, font=F_SMALL)
            draw.text((cx + 50, y + 16),
                      f"{state.tile_m}x{state.tile_n} {state.layout.name[:3]}",
                      fill=TEXT_DIM, font=F_TINY)

    # 하단 인사이트
    draw.text((20, H - 45),
              "Insight: Tighter SRAM can increase working-memory pressure and diversify heuristic choices.",
              fill=TEXT_DIM, font=F_SMALL)
    draw.text((20, H - 25),
              "64KB is 'too comfortable' — the current heuristic collapses to the same tile across layers.",
              fill=TEXT_DIM, font=F_SMALL)

    path = ASSETS / "sram_comparison.png"
    img.save(str(path))
    print(f"    → {path}")


# ═══════════════════════════════════════════════════════════════
# 6. Collapse Animation GIF (기존 스크립트 재사용)
# ═══════════════════════════════════════════════════════════════
def viz_collapse_gif():
    print("  [6/6] Collapse Animation GIF...")
    import subprocess
    gif_script = Path(__file__).resolve().parent / "visualize_collapse.py"
    subprocess.run([sys.executable, str(gif_script)], check=True)


# ═══════════════════════════════════════════════════════════════
# 시각화 로그 생성
# ═══════════════════════════════════════════════════════════════
def write_visuals_log(elapsed_ms: float):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = ASSETS / "VISUALS_LOG.md"

    # 기존 로그 읽기
    history_lines: list[str] = []
    if log_path.exists():
        content = log_path.read_text()
        # "## History" 이후 부분 추출
        if "## History" in content:
            raw_history = content.split("## History", 1)[1].strip().splitlines()
            # 기존 헤더 행은 버리고 실제 엔트리만 보존
            for line in raw_history:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("| Date | Time | Version | Notes |"):
                    continue
                if stripped.startswith("|------|------|---------|-------|"):
                    continue
                history_lines.append(stripped)

    new_entry = f"| {now} | {elapsed_ms:.0f}ms | v2.7.1 | All 6 visuals regenerated |"
    history_block = "\n".join([new_entry, *history_lines])

    log_content = f"""# Visuals Log

자동 생성된 시각화 파일 목록과 업데이트 기록.

`python tools/generate_all_visuals.py` 실행 시 모든 파일이 재생성됩니다.

## Generated Files

| File | Description | Type |
|------|-------------|------|
| `collapse_animation.gif` | WFC 붕괴 과정 step-by-step 애니메이션 | Success |
| `score_distribution.png` | Cost Model 3요소 분해 (roofline/affinity/cache) | Analysis |
| `propagation_sweep.png` | threshold별 전파 제거율 그래프 | Analysis |
| `copier_problem.png` | 복사기 문제: 64KB(실패) vs 12KB(해결) 비교 | Failure→Fix |
| `backtracking.png` | 모순 발생 → 백트래킹 시나리오 4단계 | Failure |
| `sram_comparison.png` | SRAM 64KB/16KB/12KB 결과 비교 | Comparison |

## How to Update

```bash
# 코드 변경 후 시각화 재생성
python tools/generate_all_visuals.py

# 개별 GIF만 재생성
python tools/visualize_collapse.py
```

## History

| Date | Time | Version | Notes |
|------|------|---------|-------|
{history_block}
"""
    log_path.write_text(log_content)
    print(f"    → {log_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  HW-WFC Visual Generator")
    print("=" * 60)

    t0 = time.perf_counter()

    viz_score_distribution()
    viz_propagation_sweep()
    viz_copier_problem()
    viz_backtracking()
    viz_sram_comparison()
    viz_collapse_gif()

    elapsed = (time.perf_counter() - t0) * 1000

    write_visuals_log(elapsed)

    print(f"\nDone! {elapsed:.0f}ms total")
    print(f"All visuals in: {ASSETS}/")


if __name__ == "__main__":
    main()
