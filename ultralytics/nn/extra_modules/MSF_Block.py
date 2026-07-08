import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EfficientChannelAttention(nn.Module):
    """动态适应输入通道的轻量通道注意力"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 动态计算中间维度（至少保留4通道）
        reduced_dim = max(4, channel // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入通道校验
        if x.size(1) != self.channel:
            raise ValueError(f"输入通道数{x.size(1)}与预期{self.channel}不匹配")

        avg_out = self.mlp(self.avg_pool(x))
        return x * self.sigmoid(avg_out)


class DepthwiseSpatialAttention(nn.Module):
    """深度可分离的空间注意力"""

    def __init__(self):
        super().__init__()
        # 深度可分离卷积减少计算量
        self.conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False),
            nn.Conv2d(2, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class MultiScaleFusion(nn.Module):
    """优化后的多尺度融合模块（固定ATSC流程）"""

    def __init__(self, num_scales=3, out_channels=256):
        super().__init__()
        # 参数校验
        if num_scales not in [2, 3]:
            raise ValueError("num_scales必须为2或3")

        self.num_scales = int(num_scales)
        self.out_channels = int(out_channels)
        self._initialized = False

    def _initialize(self, x):
        # 获取输入特征信息
        self.in_channels = [f.size(1) for f in x]
        self.input_sizes = [f.shape[2:] for f in x]

        # 确定目标尺寸（中间尺度）
        mid_idx = 1 if self.num_scales == 3 else 0
        self.target_size = self.input_sizes[mid_idx]

        # 初始化注意力模块
        self.attentions = nn.ModuleList([
            EfficientChannelAttention(self.in_channels[0]) if i == 0 else  # 小尺度
            DepthwiseSpatialAttention() if i == self.num_scales - 1 else  # 大尺度
            nn.Identity()  # 中尺度
            for i in range(self.num_scales)
        ])

        # 构建空间处理器
        self.spatial_ops = nn.ModuleList()
        for i, size in enumerate(self.input_sizes):
            if size[0] > self.target_size[0]:  # 下采样
                scale = size[0] // self.target_size[0]
                op = nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.in_channels[i],
                              kernel_size=scale + 1, stride=scale,
                              padding=scale // 2, groups=self.in_channels[i]),
                    nn.BatchNorm2d(self.in_channels[i])
                )
            elif size[0] < self.target_size[0]:  # 上采样
                op = nn.Upsample(size=self.target_size, mode='bilinear', align_corners=False)
            else:  # 尺寸相同
                op = nn.Identity()
            self.spatial_ops.append(op)

        # 构建通道调整器（分组卷积优化）
        self.channel_ops = nn.ModuleList()
        for in_c in self.in_channels:
            groups = max(1, in_c // 8)  # 自适应分组数
            self.channel_ops.append(
                nn.Sequential(
                    nn.Conv2d(in_c, self.out_channels, 1, groups=groups, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 轻量融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(self.out_channels * self.num_scales, self.out_channels,
                      kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self._initialized = True

    def forward(self, x):
        # 输入验证
        if len(x) != self.num_scales:
            raise ValueError(f"需要{self.num_scales}个输入特征图，但得到{len(x)}个")

        if not self._initialized:
            self._initialize(x)

        processed = []
        for i, feat in enumerate(x):
            # 阶段1：注意力
            feat = self.attentions[i](feat)

            # 阶段2：空间处理
            feat = self.spatial_ops[i](feat)

            # 阶段3：通道调整
            feat = self.channel_ops[i](feat)

            # 阶段4：尺寸强制对齐
            if feat.shape[-2:] != self.target_size:
                feat = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)

            processed.append(feat)

        return self.fusion(torch.cat(processed, dim=1))


