"""HW-WFC 모델 정의 모듈.

실제 모델 아키텍처의 레이어 구성을 하드코딩한다.
모델 가중치는 필요 없음 — 그래프 구조(layer_type, dims)만 정의.
"""

from __future__ import annotations

from .state import LayerNode, LayerType, HWState, MemoryLayout, ComputeLocation


def generate_large_candidates(
    include_large: bool = True,
) -> list[HWState]:
    """실제 GPU 스케일에 맞는 타일 후보 생성.

    A100/H100에서는 SRAM이 192~256KB이므로 큰 타일도 사용 가능.
    """
    tile_sizes = [
        (32, 32), (64, 64), (128, 128), (256, 256),
        (32, 64), (64, 32), (64, 128), (128, 64),
        (128, 256), (256, 128),
    ]
    if include_large:
        tile_sizes.extend([(256, 512), (512, 256), (512, 512)])

    layouts = list(MemoryLayout)
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


def gpt2_small(candidates: list[HWState] | None = None) -> list[LayerNode]:
    """GPT-2 Small (124M parameters).

    12 Transformer blocks, each:
      QKV_Proj → QK_MatMul → Softmax → AV_MatMul → Out_Proj → LayerNorm
    Total: 72 layers.

    Config: d_model=768, n_heads=12, d_head=64, seq_len=1024
    """
    if candidates is None:
        candidates = generate_large_candidates()

    d_model = 768
    d_head = 64
    seq_len = 1024
    n_blocks = 12

    nodes: list[LayerNode] = []
    for b in range(n_blocks):
        prefix = f"B{b}"
        block = [
            LayerNode(
                name=f"{prefix}_QKV_Proj",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": 3 * d_model, "K": d_model},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_QK_MatMul",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": seq_len, "K": d_head},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_Softmax",
                layer_type=LayerType.SOFTMAX,
                dims={"M": seq_len, "N": seq_len},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_AV_MatMul",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": d_head, "K": seq_len},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_Out_Proj",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": d_model, "K": d_model},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_LayerNorm",
                layer_type=LayerType.LAYERNORM,
                dims={"M": seq_len, "N": d_model},
                candidates=list(candidates),
            ),
        ]
        nodes.extend(block)

    return nodes


def bert_base(candidates: list[HWState] | None = None) -> list[LayerNode]:
    """BERT-Base (110M parameters).

    12 Transformer blocks, each:
      QKV_Proj → QK_MatMul → Softmax → AV_MatMul → Out_Proj → LayerNorm → FFN_Up
    Total: 84 layers.

    Config: d_model=768, n_heads=12, d_head=64, seq_len=512, d_ff=3072
    """
    if candidates is None:
        candidates = generate_large_candidates()

    d_model = 768
    d_head = 64
    d_ff = 3072
    seq_len = 512
    n_blocks = 12

    nodes: list[LayerNode] = []
    for b in range(n_blocks):
        prefix = f"B{b}"
        block = [
            LayerNode(
                name=f"{prefix}_QKV_Proj",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": 3 * d_model, "K": d_model},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_QK_MatMul",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": seq_len, "K": d_head},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_Softmax",
                layer_type=LayerType.SOFTMAX,
                dims={"M": seq_len, "N": seq_len},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_AV_MatMul",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": d_head, "K": seq_len},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_Out_Proj",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": d_model, "K": d_model},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_LayerNorm",
                layer_type=LayerType.LAYERNORM,
                dims={"M": seq_len, "N": d_model},
                candidates=list(candidates),
            ),
            LayerNode(
                name=f"{prefix}_FFN_Up",
                layer_type=LayerType.LINEAR,
                dims={"M": seq_len, "N": d_ff, "K": d_model},
                candidates=list(candidates),
            ),
        ]
        nodes.extend(block)

    return nodes
