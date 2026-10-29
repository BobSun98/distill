"""LangFlow 的线性低秩瓶颈及平衡 SVD 因子；不改变原有激活/残差。"""

import math

import torch
from torch import nn

# 覆盖 DDiTBlock 的全部二维权重；LayerNorm 与 bias 是向量，不做 SVD。
TARGETS = ("attn_qkv", "attn_out", "mlp.0", "mlp.2", "adaLN_modulation")


def retained_rank(shape, ratio):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("rank ratio 必须在 (0, 1] 内")
    return max(1, math.floor(ratio * min(shape)))


def balanced_factors(u, singular, vh, rank):
    if not 1 <= rank <= len(singular):
        raise ValueError("低秩维度超出 SVD 分解范围")
    root = singular[:rank].sqrt()
    # W[out,in] ≈ up[out,r] @ down[r,in]；对称分配奇异值，避免一个因子尺度过大。
    return root[:, None] * vh[:rank], u[:, :rank] * root[None, :]


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError("LowRankLinear 的 rank 越界")
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, inputs):
        # 两个因子之间没有新激活；FFN 的原 GELU 仍位于两组线性映射之间。
        return self.up(self.down(inputs))


def apply_low_rank(model, settings):
    """加载 checkpoint 时先按 config.json 建立因子结构，再严格加载其权重。"""
    for path, rank in settings["ranks"].items():
        parent_path, name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        original = parent.get_submodule(name)
        if not isinstance(original, nn.Linear):
            raise ValueError(f"低秩目标必须是原始 Linear: {path}")
        parent._modules[name] = LowRankLinear(original.in_features, original.out_features,
                                              rank, original.bias is not None)


def target_linears(teacher, targets=TARGETS):
    for index, block in enumerate(teacher.backbone.blocks):
        for suffix in targets:
            layer = block.get_submodule(suffix)
            if not isinstance(layer, nn.Linear):
                raise ValueError("SVD 起点需要原始 dense teacher")
            yield f"backbone.blocks.{index}.{suffix}", layer
