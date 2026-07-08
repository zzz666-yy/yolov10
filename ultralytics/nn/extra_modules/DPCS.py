import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)


class DualPoolingChannelSelector(nn.Module):
    def __init__(self, in_channels, reduction_ratio=0.5, groups=4):
        """
        保留双池化选择的轻量化模块
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道选择比例 (0-1)
            groups: 分组卷积的分组数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2  # 输出通道减半
        self.groups = min(groups, in_channels // 4)  # 动态分组
        self.k = max(1, int(in_channels * reduction_ratio))  # 至少选1通道

        # 轻量化的双池化通道选择
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 深度可分离卷积组（带Shuffle）
        self.dw_conv_selected = nn.Sequential(
            nn.Conv2d(self.k * 2, self.k * 2, 1, groups=self.groups, bias=False),
            ChannelShuffle(self.groups),
            nn.Conv2d(self.k * 2, self.k, 1, bias=False)  # 压缩到k
        )

        self.dw_conv_discarded = nn.Sequential(
            nn.Conv2d(in_channels - self.k * 2, in_channels - self.k * 2, 1,
                      groups=max(1, (in_channels - self.k * 2) // 4), bias=False),
            ChannelShuffle(max(1, (in_channels - self.k * 2) // 4)),
            nn.Conv2d(in_channels - self.k * 2, self.k // 2, 1, bias=False)  # 压缩到k/2
        )

        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(int(self.k * 1.5), int(self.k * 1.5), 1,
                      groups=self.groups, bias=False),
            ChannelShuffle(self.groups),
            nn.Conv2d(int(self.k * 1.5), self.out_channels, 1, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 双池化通道选择
        y_max = self.max_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        y_avg = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]

        # 2. 获取两种选择结果
        _, idx_max = torch.topk(y_max.abs(), k=self.k, dim=1)
        _, idx_avg = torch.topk(y_avg.abs(), k=self.k, dim=1)

        # 3. 创建选择掩码
        mask_max = torch.zeros(B, C, device=x.device)
        mask_avg = torch.zeros(B, C, device=x.device)
        mask_max.scatter_(1, idx_max, 1.0)
        mask_avg.scatter_(1, idx_avg, 1.0)

        # 4. 分割特征图
        selected_max = x * mask_max.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
        selected_avg = x * mask_avg.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
        discarded = x * (1 - torch.max(mask_max, mask_avg)).unsqueeze(-1).unsqueeze(-1)

        # 5. 深度可分离处理
        selected = torch.cat([selected_max, selected_avg], dim=1)  # [B, 2k, H, W]
        selected = self.dw_conv_selected(selected)  # [B, k, H, W]

        discarded = self.dw_conv_discarded(discarded)  # [B, k//2, H, W]

        # 6. 最终融合
        out = torch.cat([selected, discarded], dim=1)  # [B, 1.5k, H, W]
        return self.fusion(out)  # [B, C//2, H, W]


# 测试代码
if __name__ == "__main__":
    from thop import profile

    # 测试不同输入
    test_cases = [
        (2, 8, 16, 16),  # 小通道
        (4, 64, 32, 32),  # 常规
        (4, 256, 40, 40)  # 大通道
    ]

    for case in test_cases:
        B, C, H, W = case
        x = torch.randn(B, C, H, W)
        model = DualPoolingChannelSelector(in_channels=C)

        try:
            out = model(x)
            assert out.shape == (B, C // 2, H, W)
            print(f"测试通过: input {x.shape} -> output {out.shape}")

            flops = sum(p.numel() * H * W for p in model.parameters())
            print(f"FLOPs: {flops / 1e6:.1f}M | Params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
        except Exception as e:
            print(f"测试失败: {case} | {str(e)}")