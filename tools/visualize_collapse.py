"""HW-WFC 붕괴 과정 GIF 시각화.

각 프레임에서:
- 행: 레이어 노드
- 열: 후보 상태 (tile × layout × location)
- 밝은 셀 = 살아있는 후보, 어두운 셀 = 제거됨
- 붕괴된 셀 = 강조 색상
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import (
    HWState,
    LayerNode,
    LayerType,
    MemoryLayout,
    ComputeLocation,
    generate_default_candidates,
)
from src.constraint import HardwareSpec, apply_hard_constraints, propagate_constraints
from src.cost_model import compute_score, score_all_candidates
from src.collapse import CollapseEngine, select_best_state


# ── 색상 팔레트 ──
BG = (24, 24, 32)
CELL_ALIVE = (60, 70, 100)
CELL_DEAD = (35, 35, 45)
CELL_COLLAPSED = (80, 200, 120)
CELL_JUST_COLLAPSED = (255, 200, 60)
CELL_JUST_REMOVED = (180, 60, 60)
TEXT_COLOR = (220, 220, 230)
TEXT_DIM = (120, 120, 140)
HEADER_BG = (30, 30, 42)
STEP_BG = (40, 35, 55)
BORDER = (50, 50, 65)


@dataclass
class Frame:
    """시각화 프레임 상태."""
    title: str
    subtitle: str
    # 각 노드별 상태: (후보 수, collapsed 여부, collapsed_state)
    node_states: list[dict]
    # 강조 표시할 노드 인덱스
    highlight_collapsed: int | None = None
    highlight_removed: list[int] = field(default_factory=list)


def try_load_font(size: int):
    """시스템 폰트를 시도하고, 실패하면 기본 폰트 반환."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]
    for p in font_paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def capture_collapse_frames(spec: HardwareSpec, nodes: list[LayerNode]) -> list[Frame]:
    """스케줄링 과정을 프레임 단위로 캡처."""
    frames: list[Frame] = []
    all_candidates_ref = generate_default_candidates()
    total_possible = len(all_candidates_ref)

    def snap(title, subtitle, hi_col=None, hi_rem=None):
        states = []
        for n in nodes:
            states.append({
                "name": n.name,
                "type": n.layer_type.name,
                "num_candidates": n.num_candidates,
                "total": total_possible,
                "is_collapsed": n.is_collapsed,
                "collapsed_state": n.collapsed_state,
            })
        frames.append(Frame(
            title=title,
            subtitle=subtitle,
            node_states=states,
            highlight_collapsed=hi_col,
            highlight_removed=hi_rem or [],
        ))

    # Frame 0: 초기 상태
    snap("Initial Superposition",
         f"{len(nodes)} layers x {total_possible} candidates = {len(nodes) * total_possible} states")

    # Frame 1: Hard Constraints
    total_pruned = 0
    for node in nodes:
        total_pruned += apply_hard_constraints(node, spec)
    remaining = sum(n.num_candidates for n in nodes)
    snap("Hard Constraints Applied",
         f"{total_pruned} states pruned | {remaining} remaining")

    # Collapse loop
    engine = CollapseEngine(max_backtracks=10)
    penalty_threshold = 1.5

    # Bottleneck first
    bottleneck = engine.find_bottleneck_node(nodes)
    if bottleneck:
        bi = nodes.index(bottleneck)
        engine.collapse_node(bottleneck, spec, nodes)
        snap(f"Bottleneck Collapse: {bottleneck.name}",
             f"FLOPs: {bottleneck.flops():,.0f} | {bottleneck.collapsed_state}",
             hi_col=bi)

        # Propagation from bottleneck
        idx = nodes.index(bottleneck)
        removed_indices = []
        for i in range(idx + 1, len(nodes)):
            if nodes[i].is_collapsed:
                continue
            prev = nodes[i - 1]
            if not prev.is_collapsed:
                break
            before = len(nodes[i].candidates)
            propagate_constraints(prev, nodes[i], penalty_threshold)
            after = len(nodes[i].candidates)
            if before > after:
                removed_indices.append(i)
            if len(nodes[i].candidates) == 1 and not nodes[i].is_collapsed:
                nodes[i].collapse_to(nodes[i].candidates[0])

        for i in range(idx - 1, -1, -1):
            if nodes[i].is_collapsed:
                continue
            nxt = nodes[i + 1]
            if not nxt.is_collapsed:
                break
            before = len(nodes[i].candidates)
            propagate_constraints(nxt, nodes[i], penalty_threshold)
            after = len(nodes[i].candidates)
            if before > after:
                removed_indices.append(i)
            if len(nodes[i].candidates) == 1 and not nodes[i].is_collapsed:
                nodes[i].collapse_to(nodes[i].candidates[0])

        if removed_indices:
            snap("Constraint Propagation",
                 f"Propagated from {bottleneck.name} | {len(removed_indices)} neighbors affected",
                 hi_rem=removed_indices)

    # Remaining collapses
    while True:
        next_node = engine.find_lowest_entropy_node(nodes, spec)
        if next_node is None:
            break
        if not next_node.candidates:
            if engine.backtrack(nodes):
                snap("Backtrack!",
                     f"{next_node.name}: contradiction detected, rolling back")
                continue
            else:
                break

        ni = nodes.index(next_node)
        engine.collapse_node(next_node, spec, nodes)
        snap(f"Entropy Collapse: {next_node.name}",
             f"{next_node.collapsed_state}",
             hi_col=ni)

        # Propagation
        idx = nodes.index(next_node)
        removed_indices = []
        for i in range(idx + 1, len(nodes)):
            if nodes[i].is_collapsed:
                continue
            prev = nodes[i - 1]
            if not prev.is_collapsed:
                break
            before = len(nodes[i].candidates)
            propagate_constraints(prev, nodes[i], penalty_threshold)
            after = len(nodes[i].candidates)
            if before > after:
                removed_indices.append(i)
            if len(nodes[i].candidates) == 1 and not nodes[i].is_collapsed:
                nodes[i].collapse_to(nodes[i].candidates[0])

        for i in range(idx - 1, -1, -1):
            if nodes[i].is_collapsed:
                continue
            nxt = nodes[i + 1]
            if not nxt.is_collapsed:
                break
            before = len(nodes[i].candidates)
            propagate_constraints(nxt, nodes[i], penalty_threshold)
            after = len(nodes[i].candidates)
            if before > after:
                removed_indices.append(i)
            if len(nodes[i].candidates) == 1 and not nodes[i].is_collapsed:
                nodes[i].collapse_to(nodes[i].candidates[0])

        if removed_indices:
            snap("Constraint Propagation",
                 f"{len(removed_indices)} neighbors updated",
                 hi_rem=removed_indices)

    # Final frame
    total_score = sum(compute_score(n, n.collapsed_state, spec)
                      for n in nodes if n.collapsed_state)
    unique = len(set(n.collapsed_state for n in nodes if n.collapsed_state))
    snap("Schedule Complete",
         f"Score: {total_score:.3f} | {unique} unique states | {engine.backtrack_count} backtracks")

    return frames


def render_frame(frame: Frame, width: int, height: int, font, font_sm) -> Image.Image:
    """단일 프레임을 PIL Image로 렌더링."""
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    n_nodes = len(frame.node_states)
    if n_nodes == 0:
        return img

    # 상단 헤더
    header_h = 70
    draw.rectangle([0, 0, width, header_h], fill=HEADER_BG)
    draw.text((20, 12), frame.title, fill=TEXT_COLOR, font=font)
    draw.text((20, 40), frame.subtitle, fill=TEXT_DIM, font=font_sm)

    # 레이어 영역
    margin_x = 20
    margin_top = header_h + 15
    row_h = 70
    label_w = 140
    bar_x = margin_x + label_w + 10
    bar_max_w = width - bar_x - margin_x

    max_total = max(ns["total"] for ns in frame.node_states)

    for i, ns in enumerate(frame.node_states):
        y = margin_top + i * (row_h + 8)

        # 레이어 이름 + 타입
        name_color = TEXT_COLOR
        if frame.highlight_collapsed == i:
            name_color = CELL_JUST_COLLAPSED
        elif i in frame.highlight_removed:
            name_color = CELL_JUST_REMOVED

        draw.text((margin_x, y + 5), ns["name"], fill=name_color, font=font)
        draw.text((margin_x, y + 30), ns["type"], fill=TEXT_DIM, font=font_sm)

        # 후보 바 시각화
        total = ns["total"]
        alive = ns["num_candidates"]
        bar_w = bar_max_w

        # 배경 (전체 후보)
        draw.rectangle([bar_x, y + 5, bar_x + bar_w, y + row_h - 5],
                        fill=CELL_DEAD, outline=BORDER)

        if ns["is_collapsed"]:
            # 붕괴된 상태: 얇은 밝은 바
            collapsed_w = max(4, bar_w // total)
            color = CELL_JUST_COLLAPSED if frame.highlight_collapsed == i else CELL_COLLAPSED
            draw.rectangle([bar_x + 2, y + 7, bar_x + collapsed_w + 2, y + row_h - 7],
                            fill=color)
            # 상태 텍스트
            cs = ns["collapsed_state"]
            state_text = f"{cs.tile_m}x{cs.tile_n} {cs.layout.name} {cs.location.name}"
            draw.text((bar_x + collapsed_w + 10, y + 20), state_text,
                       fill=CELL_COLLAPSED, font=font_sm)
        else:
            # 살아있는 후보 바
            alive_w = max(1, int(bar_w * alive / total))
            color = CELL_JUST_REMOVED if i in frame.highlight_removed else CELL_ALIVE
            draw.rectangle([bar_x + 2, y + 7, bar_x + alive_w, y + row_h - 7],
                            fill=color)
            # 카운트 텍스트
            count_text = f"{alive}/{total}"
            draw.text((bar_x + alive_w + 8, y + 20), count_text,
                       fill=TEXT_DIM, font=font_sm)

    # 하단 스텝 표시
    step_y = height - 35
    draw.rectangle([0, step_y, width, height], fill=STEP_BG)
    step_num = frame.title.split(":")[0] if ":" in frame.title else frame.title
    draw.text((20, step_y + 8), f"Step {frames.index(frame) + 1}/{len(frames)}" if 'frames' in dir() else "",
              fill=TEXT_DIM, font=font_sm)

    return img


def generate_gif(
    frames: list[Frame],
    output_path: str,
    width: int = 800,
    duration_ms: int = 1200,
):
    """프레임 리스트를 GIF 파일로 저장."""
    n_nodes = len(frames[0].node_states) if frames else 6
    height = 70 + 15 + n_nodes * 78 + 50

    font = try_load_font(18)
    font_sm = try_load_font(13)

    images = []
    for fi, frame in enumerate(frames):
        img = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(img)

        # 상단 헤더
        header_h = 70
        draw.rectangle([0, 0, width, header_h], fill=HEADER_BG)
        draw.text((20, 12), frame.title, fill=TEXT_COLOR, font=font)
        draw.text((20, 40), frame.subtitle, fill=TEXT_DIM, font=font_sm)

        # 레이어 영역
        margin_x = 20
        margin_top = header_h + 15
        row_h = 55
        label_w = 130
        bar_x = margin_x + label_w + 10
        bar_max_w = width - bar_x - 140  # 오른쪽 텍스트 공간 확보

        for i, ns in enumerate(frame.node_states):
            y = margin_top + i * (row_h + 10)

            # 레이어 이름
            name_color = TEXT_COLOR
            if frame.highlight_collapsed == i:
                name_color = CELL_JUST_COLLAPSED
            elif i in frame.highlight_removed:
                name_color = CELL_JUST_REMOVED

            draw.text((margin_x, y + 5), ns["name"], fill=name_color, font=font)
            draw.text((margin_x, y + 28), ns["type"], fill=TEXT_DIM, font=font_sm)

            # 후보 바
            total = ns["total"]
            alive = ns["num_candidates"]
            bar_w = bar_max_w

            # 전체 바 배경
            draw.rectangle([bar_x, y + 5, bar_x + bar_w, y + row_h - 5],
                            fill=CELL_DEAD, outline=BORDER)

            if ns["is_collapsed"]:
                collapsed_w = max(6, int(bar_w / total))
                color = CELL_JUST_COLLAPSED if frame.highlight_collapsed == i else CELL_COLLAPSED
                draw.rectangle([bar_x + 2, y + 7, bar_x + collapsed_w + 2, y + row_h - 7],
                                fill=color)
                cs = ns["collapsed_state"]
                st = f"{cs.tile_m}x{cs.tile_n} {cs.layout.name}"
                draw.text((bar_x + bar_w + 8, y + 16), st,
                           fill=color, font=font_sm)
            else:
                alive_w = max(2, int(bar_w * alive / total))
                color = CELL_JUST_REMOVED if i in frame.highlight_removed else CELL_ALIVE
                draw.rectangle([bar_x + 2, y + 7, bar_x + alive_w, y + row_h - 7],
                                fill=color)
                pct = alive / total * 100
                draw.text((bar_x + bar_w + 8, y + 16),
                           f"{alive}/{total} ({pct:.0f}%)",
                           fill=TEXT_DIM, font=font_sm)

        # 하단 진행 표시
        step_y = height - 40
        draw.rectangle([0, step_y, width, height], fill=STEP_BG)
        progress = (fi + 1) / len(frames)
        prog_w = int((width - 40) * progress)
        draw.rectangle([20, step_y + 8, 20 + prog_w, step_y + 12],
                        fill=CELL_COLLAPSED)
        draw.rectangle([20 + prog_w, step_y + 8, width - 20, step_y + 12],
                        fill=CELL_DEAD)
        draw.text((20, step_y + 18),
                   f"Step {fi + 1}/{len(frames)}",
                   fill=TEXT_DIM, font=font_sm)

        images.append(img)

    # 마지막 프레임 오래 보여주기
    durations = [duration_ms] * len(images)
    if durations:
        durations[0] = duration_ms * 2    # 첫 프레임
        durations[-1] = duration_ms * 3   # 마지막 프레임

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    print(f"GIF saved: {output_path} ({len(images)} frames)")


def main():
    spec = HardwareSpec.from_json(
        Path(__file__).resolve().parent.parent / "specs" / "stress_gpu.json"
    )

    nodes = [
        LayerNode("QKV_Proj", LayerType.LINEAR,
                  {"M": 2048, "N": 192, "K": 512},
                  generate_default_candidates()),
        LayerNode("QK_MatMul", LayerType.LINEAR,
                  {"M": 2048, "N": 2048, "K": 64},
                  generate_default_candidates()),
        LayerNode("Softmax", LayerType.SOFTMAX,
                  {"M": 2048, "N": 2048},
                  generate_default_candidates()),
        LayerNode("AV_MatMul", LayerType.LINEAR,
                  {"M": 2048, "N": 64, "K": 2048},
                  generate_default_candidates()),
        LayerNode("Out_Proj", LayerType.LINEAR,
                  {"M": 2048, "N": 512, "K": 64},
                  generate_default_candidates()),
        LayerNode("LayerNorm", LayerType.LAYERNORM,
                  {"M": 2048, "N": 512},
                  generate_default_candidates()),
    ]

    print("Capturing collapse frames...")
    frames = capture_collapse_frames(spec, nodes)
    print(f"Captured {len(frames)} frames")

    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    output_path = str(out_dir / "collapse_animation.gif")

    generate_gif(frames, output_path, width=880, duration_ms=1200)


if __name__ == "__main__":
    main()
