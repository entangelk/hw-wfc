"""HW-WFC 시각화 모듈.

스케줄링 과정과 결과를 시각화한다.
matplotlib 필수. 출력은 PNG/PDF 파일.
"""

from __future__ import annotations

import math
from pathlib import Path

from .state import LayerNode, LayerType, HWState
from .constraint import HardwareSpec
from .cost_model import (
    roofline_score, layout_affinity, cache_efficiency, compute_score,
)
from .scheduler import ScheduleResult


def _parse_steps(result: ScheduleResult) -> dict:
    """ScheduleResult.steps를 파싱하여 시각화용 데이터를 추출."""
    data = {
        "init_space": 0,
        "after_hard": 0,
        "propagate_events": [],  # (src, dst, removed)
        "collapse_events": [],   # (node_name, state_str)
        "backtrack_events": [],  # (node_name,)
    }
    for step in result.steps:
        if "[Init]" in step:
            parts = step.split("탐색 공간:")
            if len(parts) > 1:
                data["init_space"] = int(parts[1].strip().split()[0])
        elif "[Hard Constraint]" in step:
            parts = step.split("남은 탐색 공간:")
            if len(parts) > 1:
                data["after_hard"] = int(parts[1].strip().split()[0])
        elif "[Propagate]" in step:
            # [Propagate] src → dst: N개 후보 제거
            try:
                inner = step.split("[Propagate]")[1].strip()
                src_dst, rest = inner.split(":")
                src, dst = src_dst.split("→")
                removed = int(rest.strip().split("개")[0])
                data["propagate_events"].append((src.strip(), dst.strip(), removed))
            except (ValueError, IndexError):
                pass
        elif "[Collapse]" in step or "[Auto-Collapse]" in step:
            try:
                tag = "[Collapse]" if "[Collapse]" in step else "[Auto-Collapse]"
                inner = step.split(tag)[1].strip()
                name, state_str = inner.split("→", 1)
                data["collapse_events"].append((name.strip(), state_str.strip()))
            except (ValueError, IndexError):
                pass
        elif "[Backtrack]" in step:
            try:
                inner = step.split("[Backtrack]")[1].strip()
                name = inner.split()[0]
                data["backtrack_events"].append((name,))
            except (ValueError, IndexError):
                pass
    return data


# ─── 1. 탐색 공간 축소 퍼널 ────────────────────────────────


def plot_funnel(result: ScheduleResult, save_path: str | Path | None = None):
    """탐색 공간 축소 퍼널 차트."""
    import matplotlib.pyplot as plt

    data = _parse_steps(result)
    n_layers = len(result.nodes)

    # 단계별 남은 후보 수 계산
    stages = ["Initial", "Hard\nConstraint"]
    values = [data["init_space"], data["after_hard"]]

    # 전파 + 붕괴로 인한 축소
    total_propagated = sum(r for _, _, r in data["propagate_events"])
    after_propagate = data["after_hard"] - total_propagated
    if total_propagated > 0:
        stages.append("Propagation")
        values.append(max(after_propagate, n_layers))

    stages.append("Final")
    values.append(result.final_search_space)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#22c55e"][:len(stages)]
    bars = ax.barh(range(len(stages)), values, color=colors, height=0.6, edgecolor="white")

    for i, (bar, v) in enumerate(zip(bars, values)):
        pct = v / values[0] * 100 if values[0] > 0 else 0
        ax.text(bar.get_width() + values[0] * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate States", fontsize=11)
    ax.set_title("HW-WFC Search Space Reduction", fontsize=13, fontweight="bold")
    ax.set_xlim(0, values[0] * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ─── 2. 점수 분해 바 차트 ──────────────────────────────────


def plot_score_breakdown(
    nodes: list[LayerNode],
    spec: HardwareSpec,
    save_path: str | Path | None = None,
):
    """각 레이어의 최종 선택에 대한 점수 분해 스택 바 차트."""
    import matplotlib.pyplot as plt
    import numpy as np

    collapsed = [(n, n.collapsed_state) for n in nodes if n.is_collapsed]
    if not collapsed:
        return

    names = []
    roofline_vals = []
    affinity_vals = []
    cache_vals = []

    for node, state in collapsed:
        names.append(node.name)
        roofline_vals.append(roofline_score(node, state, spec))
        affinity_vals.append(layout_affinity(node, state))
        cache_vals.append(cache_efficiency(node, state, spec))

    x = np.arange(len(names))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))

    b1 = ax.bar(x, roofline_vals, width, label="Roofline", color="#3b82f6")
    b2 = ax.bar(x, affinity_vals, width, bottom=roofline_vals, label="Layout Affinity", color="#f97316")
    bottom2 = [r + a for r, a in zip(roofline_vals, affinity_vals)]
    b3 = ax.bar(x, cache_vals, width, bottom=bottom2, label="Cache Efficiency", color="#22c55e")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Score Breakdown per Layer", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 타일 크기 annotation
    for i, (node, state) in enumerate(collapsed):
        total = roofline_vals[i] + affinity_vals[i] + max(cache_vals[i], 0)
        ax.text(i, total + 0.02, f"{state.tile_m}x{state.tile_n}",
                ha="center", va="bottom", fontsize=7, color="#666")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ─── 3. DAG 그래프 + 선택 결과 ─────────────────────────────


def plot_dag(
    nodes: list[LayerNode],
    save_path: str | Path | None = None,
):
    """DAG 그래프에 선택 결과를 오버레이."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 1.5), 6))

    # 레이어 타입별 색상
    type_colors = {
        LayerType.LINEAR: "#3b82f6",
        LayerType.CONV2D: "#8b5cf6",
        LayerType.DEPTHWISE_CONV: "#a855f7",
        LayerType.RELU: "#22c55e",
        LayerType.SOFTMAX: "#ef4444",
        LayerType.LAYERNORM: "#f97316",
    }

    # 노드 위치 계산 (간단한 좌→우 배치)
    positions = {}
    for i, node in enumerate(nodes):
        positions[node.name] = (i * 2.0, 0)

    # 엣지 그리기
    for node in nodes:
        x0, y0 = positions[node.name]
        for neighbor_name in node.neighbors:
            if neighbor_name in positions:
                x1, y1 = positions[neighbor_name]
                # 순방향만 그리기 (중복 방지)
                if x1 > x0:
                    dy = 0.4 if abs(x1 - x0) > 2.5 else 0  # skip connection은 위로 굽힘
                    ax.annotate(
                        "", xy=(x1 - 0.4, y1 + dy), xytext=(x0 + 0.4, y0 + dy),
                        arrowprops=dict(
                            arrowstyle="->", color="#94a3b8", lw=1.5,
                            connectionstyle=f"arc3,rad={0.3 if dy else 0}",
                        ),
                    )

    # 노드 그리기
    for node in nodes:
        x, y = positions[node.name]
        color = type_colors.get(node.layer_type, "#6b7280")
        state = node.collapsed_state

        # 노드 박스
        box = mpatches.FancyBboxPatch(
            (x - 0.4, y - 0.35), 0.8, 0.7,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.85,
            edgecolor="white", linewidth=2,
        )
        ax.add_patch(box)

        # 노드 이름
        ax.text(x, y + 0.1, node.name, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

        # 타일 선택
        if state:
            tile_str = f"{state.tile_m}x{state.tile_n}"
            layout_str = state.layout.name[:3]
            ax.text(x, y - 0.12, f"{tile_str} {layout_str}",
                    ha="center", va="center", fontsize=6, color="white", alpha=0.9)

    # 범례
    legend_patches = [
        mpatches.Patch(color=c, label=t.name)
        for t, c in type_colors.items()
        if any(n.layer_type == t for n in nodes)
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

    ax.set_xlim(-1, len(nodes) * 2)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("HW-WFC Schedule Result (DAG)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ─── 4. GPU 스펙별 비교 테이블 ──────────────────────────────


def plot_spec_comparison(
    results: list[tuple[str, ScheduleResult, HardwareSpec]],
    save_path: str | Path | None = None,
):
    """여러 GPU 스펙의 스케줄링 결과를 비교하는 테이블+바 차트."""
    import matplotlib.pyplot as plt
    import numpy as np

    spec_names = [name for name, _, _ in results]
    times = []
    unique_counts = []
    scores = []

    for _, result, spec in results:
        times.append(0)  # 외부에서 측정 필요
        states = set()
        total_score = 0
        for n in result.nodes:
            if n.collapsed_state:
                states.add(n.collapsed_state)
                total_score += compute_score(n, n.collapsed_state, spec)
        unique_counts.append(len(states))
        scores.append(total_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unique states
    colors = ["#3b82f6", "#8b5cf6", "#ef4444", "#22c55e"][:len(spec_names)]
    axes[0].bar(spec_names, unique_counts, color=colors, width=0.5)
    axes[0].set_ylabel("Unique States")
    axes[0].set_title("Tile Diversity per GPU Spec")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Total score
    axes[1].bar(spec_names, scores, color=colors, width=0.5)
    axes[1].set_ylabel("Total Score")
    axes[1].set_title("Schedule Quality per GPU Spec")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
