"""HW-WFC 상태 정의 모듈.

각 연산 레이어(노드)가 가질 수 있는 하드웨어 구현 상태(State)의
중첩(Superposition)을 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MemoryLayout(Enum):
    ROW_MAJOR = auto()
    COL_MAJOR = auto()
    BLOCK_TILED = auto()


class ComputeLocation(Enum):
    SRAM = auto()
    DRAM = auto()


@dataclass(frozen=True)
class HWState:
    """단일 하드웨어 구현 후보."""
    tile_m: int
    tile_n: int
    layout: MemoryLayout
    location: ComputeLocation = ComputeLocation.SRAM

    @property
    def tile_elements(self) -> int:
        return self.tile_m * self.tile_n

    def memory_bytes(self, dtype_bytes: int = 4) -> int:
        """이 타일 구성이 차지하는 메모리(바이트)."""
        return self.tile_elements * dtype_bytes

    def __repr__(self) -> str:
        return (f"HWState(tile={self.tile_m}x{self.tile_n}, "
                f"{self.layout.name}, {self.location.name})")


class LayerType(Enum):
    LINEAR = auto()   # MatMul / GEMM
    CONV2D = auto()
    DEPTHWISE_CONV = auto()
    RELU = auto()
    SOFTMAX = auto()
    LAYERNORM = auto()


@dataclass
class LayerNode:
    """모델 그래프의 단일 레이어 노드.

    초기에는 모든 가능한 HWState가 중첩(superposition)되어 있다.
    제약 전파를 통해 후보가 줄어들고, 최종적으로 하나의 상태로 붕괴(collapse)된다.
    """
    name: str
    layer_type: LayerType
    # 연산 차원 (M, N, K for MatMul; H, W, C for Conv 등)
    dims: dict[str, int] = field(default_factory=dict)
    # 현재 남아있는 상태 후보군
    candidates: list[HWState] = field(default_factory=list)
    # 붕괴 확정된 상태 (None이면 아직 미확정)
    collapsed_state: HWState | None = None
    # DAG 그래프의 인접 노드 이름 목록 (미설정 시 리스트 순서로 자동 추론)
    neighbors: list[str] = field(default_factory=list)

    @property
    def is_collapsed(self) -> bool:
        return self.collapsed_state is not None

    @property
    def num_candidates(self) -> int:
        return 1 if self.is_collapsed else len(self.candidates)

    def flops(self) -> int:
        """레이어의 대략적인 FLOPs 추정."""
        if self.layer_type == LayerType.LINEAR:
            m = self.dims.get("M", 1)
            n = self.dims.get("N", 1)
            k = self.dims.get("K", 1)
            return 2 * m * n * k
        if self.layer_type == LayerType.CONV2D:
            h = self.dims.get("H", 1)
            w = self.dims.get("W", 1)
            c_in = self.dims.get("C_in", 1)
            c_out = self.dims.get("C_out", 1)
            kh = self.dims.get("KH", 3)
            kw = self.dims.get("KW", 3)
            return 2 * h * w * c_in * c_out * kh * kw
        if self.layer_type == LayerType.DEPTHWISE_CONV:
            h = self.dims.get("H", 1)
            w = self.dims.get("W", 1)
            c = self.dims.get("C", 1)
            kh = self.dims.get("KH", 3)
            kw = self.dims.get("KW", 3)
            return 2 * h * w * c * kh * kw
        # Activation 계열은 element-wise
        elements = 1
        for v in self.dims.values():
            elements *= v
        return elements

    def collapse_to(self, state: HWState) -> None:
        """특정 상태로 강제 붕괴."""
        self.collapsed_state = state
        self.candidates = [state]

    def remove_candidate(self, state: HWState) -> bool:
        """후보에서 특정 상태를 제거. 제거 성공 시 True."""
        if state in self.candidates:
            self.candidates.remove(state)
            return True
        return False

    def __repr__(self) -> str:
        status = f"COLLAPSED={self.collapsed_state}" if self.is_collapsed else f"{len(self.candidates)} candidates"
        return f"LayerNode({self.name}, {self.layer_type.name}, {status})"


def generate_default_candidates(
    tile_sizes: list[tuple[int, int]] | None = None,
    layouts: list[MemoryLayout] | None = None,
    locations: list[ComputeLocation] | None = None,
) -> list[HWState]:
    """기본 상태 후보군 생성 (모든 조합의 카르테시안 곱)."""
    if tile_sizes is None:
        tile_sizes = [
            (8, 8), (16, 16), (32, 32), (64, 64), (128, 128),
            # 직사각 타일: 레이어 shape에 따라 다른 최적해 유도
            (32, 64), (64, 32), (32, 128), (128, 32), (64, 128), (128, 64),
        ]
    if layouts is None:
        layouts = list(MemoryLayout)
    if locations is None:
        locations = list(ComputeLocation)

    candidates = []
    for tm, tn in tile_sizes:
        for layout in layouts:
            for loc in locations:
                candidates.append(HWState(
                    tile_m=tm, tile_n=tn,
                    layout=layout, location=loc,
                ))
    return candidates
