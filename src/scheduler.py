"""HW-WFC 스케줄러.

전체 파이프라인을 조율한다:
  1. 하드웨어 스펙 로드 → 초기 상태 생성
  2. Hard Constraint 적용 (탐색 공간 축소)
  3. 병목 우선 붕괴 (Bottleneck-First Collapse)
  4. 양방향 제약 전파 (Bidirectional Propagation)
  5. 나머지 노드 엔트로피 기반 붕괴
  6. 모순 발생 시 백트래킹
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .state import LayerNode, generate_default_candidates
from .constraint import HardwareSpec, apply_hard_constraints, propagate_constraints
from .collapse import CollapseEngine


def ensure_neighbors(nodes: list[LayerNode]) -> None:
    """neighbors가 설정되지 않은 노드에 대해 리스트 순서로 자동 추론.

    이미 neighbors가 설정된 노드는 건드리지 않는다 (DAG 구성 유지).
    """
    has_any_neighbors = any(n.neighbors for n in nodes)
    if has_any_neighbors:
        # DAG가 명시적으로 구성된 경우: neighbors 없는 노드만 보완하지 않음
        # (의도적으로 고립된 노드일 수 있으므로)
        return

    # 모든 노드에 neighbors가 없으면 리스트 순서로 1D 체인 구성
    for i, node in enumerate(nodes):
        if i > 0:
            node.neighbors.append(nodes[i - 1].name)
        if i < len(nodes) - 1:
            node.neighbors.append(nodes[i + 1].name)


@dataclass
class ScheduleResult:
    """스케줄링 결과."""
    success: bool
    nodes: list[LayerNode]
    steps: list[str] = field(default_factory=list)
    initial_search_space: int = 0
    final_search_space: int = 0
    backtracks_used: int = 0
    final_threshold: float = 0.0


class HWWFCScheduler:
    """HW-WFC 메인 스케줄러."""

    def __init__(
        self,
        spec: HardwareSpec,
        penalty_threshold: float | str = 1.5,
        max_backtracks: int = 10,
        transition_weight: float | None = None,
    ):
        self.spec = spec
        self.adaptive = (penalty_threshold == "auto")
        self.penalty_threshold = 1.5 if self.adaptive else penalty_threshold
        self.engine = CollapseEngine(
            max_backtracks=max_backtracks,
            transition_weight=transition_weight,
        )

    def schedule(self, nodes: list[LayerNode]) -> ScheduleResult:
        """전체 스케줄링 실행."""
        result = ScheduleResult(success=False, nodes=nodes)

        # ── Step 0: 그래프 연결 보장 + 초기 상태 카운트 ──
        ensure_neighbors(nodes)
        self._node_map = {n.name: n for n in nodes}
        result.initial_search_space = sum(n.num_candidates for n in nodes)
        result.steps.append(
            f"[Init] 총 {len(nodes)}개 레이어, "
            f"초기 탐색 공간: {result.initial_search_space} states"
        )

        # ── Step 1: Hard Constraints 적용 ──
        total_pruned = 0
        for node in nodes:
            pruned = apply_hard_constraints(node, self.spec)
            total_pruned += pruned

        after_hard = sum(n.num_candidates for n in nodes)
        result.steps.append(
            f"[Hard Constraint] {total_pruned}개 상태 제거 → "
            f"남은 탐색 공간: {after_hard} states"
        )

        # 모순 체크
        for node in nodes:
            if not node.candidates:
                result.steps.append(f"[Error] {node.name}: 모든 후보 제거됨 (모순)")
                return result

        # ── Step 2: 병목 우선 붕괴 ──
        bottleneck = self.engine.find_bottleneck_node(nodes)
        if bottleneck is None:
            result.steps.append("[Done] 모든 노드 이미 붕괴됨")
            result.success = True
            return result

        result.steps.append(
            f"[Bottleneck] {bottleneck.name} 선택 "
            f"(FLOPs: {bottleneck.flops():,})"
        )

        if not self.engine.collapse_node(bottleneck, self.spec, nodes):
            result.steps.append(f"[Error] {bottleneck.name} 붕괴 실패")
            result.backtracks_used = self.engine.backtrack_count
            return result

        result.steps.append(
            f"[Collapse] {bottleneck.name} → {bottleneck.collapsed_state}"
        )

        # ── Step 3: 양방향 제약 전파 ──
        removed, target = self._propagate_from(bottleneck, nodes, result)

        # ── Step 3b: 적응적 threshold 보정 (adaptive 모드) ──
        if self.adaptive and target > 0:
            self._adapt_threshold(removed, target, result)

        # ── Step 4: 나머지 노드 순차 붕괴 (엔트로피 순) ──
        while True:
            next_node = self.engine.find_lowest_entropy_node(nodes, self.spec)
            if next_node is None:
                break  # 전부 붕괴됨

            if not next_node.candidates:
                # 모순 → 백트래킹
                if self.engine.backtrack(nodes):
                    result.steps.append(
                        f"[Backtrack] {next_node.name} 모순 발생, 되돌림"
                    )
                    continue
                else:
                    result.steps.append(
                        f"[Error] {next_node.name} 백트래킹 한도 초과"
                    )
                    result.backtracks_used = self.engine.backtrack_count
                    return result

            if not self.engine.collapse_node(next_node, self.spec, nodes):
                if self.engine.backtrack(nodes):
                    result.steps.append(
                        f"[Backtrack] {next_node.name} 붕괴 실패, 되돌림"
                    )
                    continue
                else:
                    result.steps.append(f"[Error] {next_node.name} 붕괴 불가")
                    result.backtracks_used = self.engine.backtrack_count
                    return result

            result.steps.append(
                f"[Collapse] {next_node.name} → {next_node.collapsed_state}"
            )

            # 붕괴 후 제약 전파
            self._propagate_from(next_node, nodes, result)

        # ── 완료 ──
        result.success = all(n.is_collapsed for n in nodes)
        result.final_search_space = sum(n.num_candidates for n in nodes)
        result.backtracks_used = self.engine.backtrack_count
        result.final_threshold = self.penalty_threshold

        status = "성공" if result.success else "실패"
        result.steps.append(
            f"[Result] {status} | "
            f"탐색 공간: {result.initial_search_space} → {result.final_search_space} | "
            f"백트래킹: {result.backtracks_used}회"
        )
        return result

    def _adapt_threshold(
        self,
        removed: int,
        target: int,
        result: ScheduleResult,
    ) -> None:
        """첫 전파의 제거율을 기반으로 이후 전파에 사용할 threshold를 보정.

        목표 제거율 범위: 20~60%.
        - < 20%: 전파가 약함 → threshold를 제거율에 비례해서 축소 (강화)
        - > 60%: 전파가 과함 → threshold를 제거율에 비례해서 확대 (완화)
        - 20~60%: 변경 없음
        """
        rate = removed / target if target > 0 else 0.0

        if 0.20 <= rate <= 0.60:
            result.steps.append(
                f"[Adaptive] 제거율 {rate:.0%} (목표 범위 내) → "
                f"threshold={self.penalty_threshold:.2f} 유지"
            )
            return

        old_threshold = self.penalty_threshold
        if rate < 0.20:
            # 제거율이 낮을수록 더 큰 폭으로 축소
            # rate=0% → ×0.6, rate=10% → ×0.8, rate=20% → ×1.0
            factor = 0.6 + 2.0 * rate
            self.penalty_threshold *= factor
        else:
            # 제거율이 높을수록 더 큰 폭으로 확대
            # rate=60% → ×1.0, rate=80% → ×1.4, rate=100% → ×1.8
            factor = 1.0 + 2.0 * (rate - 0.60)
            self.penalty_threshold *= factor

        result.steps.append(
            f"[Adaptive] 제거율 {rate:.0%} → "
            f"threshold {old_threshold:.2f} → {self.penalty_threshold:.2f}"
        )

    def _propagate_from(
        self,
        source: LayerNode,
        all_nodes: list[LayerNode],
        result: ScheduleResult,
    ) -> tuple[int, int]:
        """붕괴된 노드에서 BFS로 인접 노드에 제약 전파.

        그래프 구조를 따라 전파하며, visited set으로 cycle을 방지한다.

        Returns: (제거된 후보 수, 전파 대상이었던 총 후보 수)
        """
        node_map = self._node_map
        visited: set[str] = {source.name}
        queue: deque[str] = deque()
        total_removed = 0
        total_target = 0

        # source의 이웃을 큐에 추가
        for neighbor_name in source.neighbors:
            if neighbor_name not in visited:
                queue.append(neighbor_name)

        while queue:
            current_name = queue.popleft()
            if current_name in visited:
                continue
            visited.add(current_name)

            current = node_map.get(current_name)
            if current is None or current.is_collapsed:
                # 이미 붕괴된 노드는 건너뛰되, 그 너머로 전파 계속
                if current and current.is_collapsed:
                    for next_name in current.neighbors:
                        if next_name not in visited:
                            queue.append(next_name)
                continue

            # 현재 노드의 이웃 중 붕괴된 노드에서 전파
            before = len(current.candidates)
            total_target += before
            for adj_name in current.neighbors:
                adj = node_map.get(adj_name)
                if adj and adj.is_collapsed:
                    removed = propagate_constraints(
                        adj, current, self.penalty_threshold
                    )
                    if removed > 0:
                        total_removed += removed
                        result.steps.append(
                            f"[Propagate] {adj.name} → {current.name}: "
                            f"{removed}개 후보 제거"
                        )

            # 후보가 1개만 남으면 자동 붕괴
            if len(current.candidates) == 1 and not current.is_collapsed:
                current.collapse_to(current.candidates[0])
                result.steps.append(
                    f"[Auto-Collapse] {current.name} → {current.collapsed_state}"
                )
                # 자동 붕괴 시 그 이웃도 전파 대상에 추가
                for next_name in current.neighbors:
                    if next_name not in visited:
                        queue.append(next_name)

        return total_removed, total_target
