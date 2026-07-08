import torch
import torch.nn as nn
from typing import List


class ADD(nn.Module):
    """极简加法融合模块（直接加法，无需任何调整）"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 输入验证（确保只有两个分支）
        if len(x) != 2:
            raise ValueError(f"需要2个输入特征图，但得到{len(x)}个")

        feat1, feat2 = x[0], x[1]

        # 检查空间尺寸对齐
        if feat1.shape[2:] != feat2.shape[2:]:
            raise ValueError(f"空间尺寸未对齐: {feat1.shape[2:]} vs {feat2.shape[2:]}")

        # 检查通道数对齐
        if feat1.shape[1] != feat2.shape[1]:
            raise ValueError(f"通道数未对齐: {feat1.shape[1]} vs {feat2.shape[1]}")

        # 直接相加
        fused = feat1 + feat2

        return fused