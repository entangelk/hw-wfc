"""HW-WFC 붕괴(Collapse) 엔진.

섀넌 엔트로피 기반으로 가장 확실한 노드부터 붕괴시키고,
모순 발생 시 백트래킹한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .state import HWState, LayerNode
from .constraint import HardwareSpec, derive_transition_weight
from .cost_model import score_all_candidates, scores_to_probabilities


def shannon_entropy(probabilities: list[tuple[HWState, float]]) -> float:
    """섀넌 엔트로피 계산. 값이 작을수록 확실함(붕괴 우선순위 높음)."""
    entropy = 0.0
    for _, p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_node_entropy(node: LayerNode, spec: HardwareSpec) -> float:
    """노드의 현재 엔트로피 계산."""
    if node.is_collapsed:
        return 0.0
    if not node.candidates:
        return float("inf")  # 모순 — 후보 없음

    scored = score_all_candidates(node, spec)
    probs = scores_to_probabilities(scored)
    return shannon_entropy(probs)


def select_best_state(
    node: LayerNode,
    spec: HardwareSpec,
    all_nodes: list[LayerNode] | None = None,
    transition_weight: float = 0.1,
) -> HWState | None:
    """노드의 후보 중 최고 점수 상태를 선택.

    all_nodes가 제공되면 이미 붕괴된 인접 노드와의 전환 비용을 반영한다.
    """
    if not node.candidates:
        return None
    scored = score_all_candidates(node, spec)
    if not scored:
        return None

    # 인접 붕괴 노드가 없으면 독립 점수로 선택
    if all_nodes is None:
        return scored[0][0]

    from .constraint import total_transition_penalty

    # 노드 이름 → 노드 매핑
    node_map = {n.name: n for n in all_nodes}

    # neighbors 필드가 설정되어 있으면 사용, 아니면 리스트 순서로 추론
    neighbor_names = node.neighbors
    if not neighbor_names:
        idx = next((i for i, n in enumerate(all_nodes) if n.name == node.name), None)
        if idx is None:
            return scored[0][0]
        neighbor_names = []
        if idx > 0:
            neighbor_names.append(all_nodes[idx - 1].name)
        if idx < len(all_nodes) - 1:
            neighbor_names.append(all_nodes[idx + 1].name)

    # 붕괴된 인접 노드의 상태 수집
    neighbors = []
    for name in neighbor_names:
        n = node_map.get(name)
        if n and n.is_collapsed:
            neighbors.append(n.collapsed_state)

    if not neighbors:
        return scored[0][0]

    # 전환 비용을 반영한 점수 재계산
    best_state = None
    best_adjusted = -float("inf")
    for state, base_score in scored:
        penalty = sum(total_transition_penalty(state, ns) for ns in neighbors)
        adjusted = base_score - penalty * transition_weight
        if adjusted > best_adjusted:
            best_adjusted = adjusted
            best_state = state

    return best_state


@dataclass
class CollapseSnapshot:
    """백트래킹을 위한 스냅샷."""
    node_name: str
    collapsed_state: HWState
    # 붕괴 전 각 노드의 후보군 백업
    candidates_backup: dict[str, list[HWState]] = field(default_factory=dict)


class CollapseEngine:
    """엔트로피 기반 붕괴 + 백트래킹 엔진."""

    def __init__(self, max_backtracks: int = 10, transition_weight: float | None = None):
        self.max_backtracks = max_backtracks
        self.transition_weight = transition_weight
        self.history: list[CollapseSnapshot] = []
        self.backtrack_count = 0

    def find_lowest_entropy_node(
        self, nodes: list[LayerNode], spec: HardwareSpec
    ) -> LayerNode | None:
        """미붕괴 노드 중 엔트로피가 가장 낮은 노드를 반환.

        후보가 0개인 노드(모순)도 반환하여 백트래킹을 유도한다.
        """
        best_node = None
        best_entropy = float("inf")

        for node in nodes:
            if node.is_collapsed:
                continue
            if not node.candidates:
                # 모순 발생 — 즉시 반환하여 백트래킹 유도
                return node
            entropy = compute_node_entropy(node, spec)
            if entropy < best_entropy:
                best_entropy = entropy
                best_node = node

        return best_node

    def find_bottleneck_node(self, nodes: list[LayerNode]) -> LayerNode | None:
        """FLOPs가 가장 큰 미붕괴 노드를 반환 (병목 우선 붕괴)."""
        best = None
        best_flops = -1
        for node in nodes:
            if node.is_collapsed:
                continue
            f = node.flops()
            if f > best_flops:
                best_flops = f
                best = node
        return best

    def save_snapshot(
        self, node: LayerNode, state: HWState, all_nodes: list[LayerNode]
    ) -> None:
        """붕괴 직전 상태를 스냅샷으로 저장."""
        backup = {n.name: list(n.candidates) for n in all_nodes}
        self.history.append(CollapseSnapshot(
            node_name=node.name,
            collapsed_state=state,
            candidates_backup=backup,
        ))

    def backtrack(self, all_nodes: list[LayerNode]) -> bool:
        """가장 최근 붕괴를 취소하고 해당 상태를 후보에서 제외.

        Returns: 백트래킹 성공 여부.
        """
        if not self.history:
            return False
        if self.backtrack_count >= self.max_backtracks:
            return False

        snapshot = self.history.pop()
        self.backtrack_count += 1

        # 모든 노드의 후보군을 스냅샷 시점으로 복원
        for node in all_nodes:
            if node.name in snapshot.candidates_backup:
                node.candidates = snapshot.candidates_backup[node.name]
                node.collapsed_state = None

        # 실패한 상태를 후보에서 제거
        for node in all_nodes:
            if node.name == snapshot.node_name:
                node.remove_candidate(snapshot.collapsed_state)
                break

        return True

    def collapse_node(
        self, node: LayerNode, spec: HardwareSpec, all_nodes: list[LayerNode]
    ) -> bool:
        """노드를 최고 점수 상태로 붕괴 (인접 전환 비용 반영). 실패 시 False."""
        weight = self.transition_weight if self.transition_weight is not None else derive_transition_weight(spec)
        state = select_best_state(node, spec, all_nodes, transition_weight=weight)
        if state is None:
            return False

        self.save_snapshot(node, state, all_nodes)
        node.collapse_to(state)
        return True
