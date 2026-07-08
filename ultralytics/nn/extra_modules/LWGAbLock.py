import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from math import gcd
from typing import Optional


class ChannelShuffle(nn.Module):
    """增强分组卷积间的信息交流"""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        groups = min(self.groups, C)  # 实际分组数不超过通道数
        channels_per_group = C // groups
        # 维度重排: [B, C, H, W] -> [B, groups, C//groups, H, W] -> [B, C//groups, groups, H, W]
        x = x.view(B, groups, channels_per_group, H, W).permute(0, 2, 1, 3, 4)
        return x.contiguous().view(B, C, H, W)


class LWGA(nn.Module):
    """轻量级全局注意力模块（完全兼容YOLOv10）"""

    def __init__(self, c1: int, c2: Optional[int] = None, stage: int = 1):
        """
        Args:
            c1: 输入通道数（必须能被4整除）
            c2: 输出通道数（None时等于c1）
            stage: 阶段标识（1=浅层,2=中层,3=深层）
        """
        super().__init__()

        # === 1. 参数验证 ===
        if c1 % 4 != 0:
            raise ValueError(f"输入通道数{c1}必须能被4整除（当前dim_split={c1 // 4}）")

        # === 2. 核心参数 ===
        self.c2 = c1 if c2 is None else c2
        self.stage = stage
        self.dim_split = c1 // 4  # 四等分通道

        # === 3. 自适应超参数 ===
        self.groups = self._calculate_groups(c1)
        self.mlp_ratio = 1.0 + 0.5 * stage
        self.drop_path_rate = min(0.1, 0.05 * stage)

        # === 4. 网络层定义 ===
        # 输入特征融合（关键修改：使用groups=1确保兼容性）
        self.prefusion = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 1, groups=1),  # 修改为普通卷积
            ChannelShuffle(1),  # 保持通道混洗
            nn.GELU(),
            nn.Conv2d(c1 // 2, c1, 1)
        )

        # 点注意力分支
        self.pa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim_split, max(4, self.dim_split // 8), 1),
            nn.GELU(),
            nn.Conv2d(max(4, self.dim_split // 8), self.dim_split, 1),
            nn.Sigmoid()
        )

        # 阶段自适应注意力
        self._build_stage_specific_layers()

        # 动态MLP（使用计算得到的分组数）
        hidden_dim = int(c1 * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden_dim, 1, groups=self.groups),
            ChannelShuffle(self.groups),
            nn.GELU(),
            nn.Conv2d(hidden_dim, c1, 1, groups=1)  # 最后层使用groups=1
        )

        # 输出处理
        self.proj = nn.Conv2d(c1, self.c2, 1) if c1 != self.c2 else nn.Identity()
        self.norm = nn.GroupNorm(self.groups, c1)
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0 else nn.Identity()

    def _calculate_groups(self, channels):
        """计算合法的最大分组数（能整除且不超过4）"""
        for g in range(min(4, channels), 0, -1):
            if channels % g == 0:
                return g
        return 1  # 保底值

    def _build_stage_specific_layers(self):
        """构建阶段相关层"""
        if self.stage == 1:  # 浅层-局部注意力
            self.ga = nn.Sequential(
                nn.Conv2d(self.dim_split, self.dim_split, 3,
                          padding=1, groups=self.groups),
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        elif self.stage == 2:  # 中层-区域注意力
            self.ga = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(self.dim_split, self.dim_split, 1),
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        else:  # 深层-全局注意力
            self.ga = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.dim_split, self.dim_split, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # 1. 保存原始尺寸
        orig_size = x.shape[2:]

        # 2. 输入特征融合
        x = self.prefusion(x)
        shortcut = x

        # 3. 四分支处理
        x1, x2, x3, x4 = torch.split(x, [self.dim_split] * 4, dim=1)

        # 4. 多尺度注意力
        x1 = x1 * F.interpolate(self.pa(x1), orig_size, mode='nearest')
        x4 = F.interpolate(self.ga(x4), orig_size, mode='nearest')

        # 5. 残差连接
        x = shortcut + self.drop_path(
            self.mlp(self.norm(torch.cat([x1, x2, x3, x4], dim=1))))

        return self.proj(x)


# 测试代码
if __name__ == "__main__":
    def run_tests():
        """兼容性测试"""
        test_cases = [
            (256, 256, 1),  # 浅层-通道不变
            (512, 256, 2),  # 中层-通道缩减
            (1024, 1024, 3)  # 深层-通道不变
        ]

        for c1, c2, stage in test_cases:
            try:
                print(f"\n测试输入: c1={c1}, c2={c2}, stage={stage}")
                x = torch.randn(2, c1, 20, 20)
                model = LWGA(c1, c2, stage)
                out = model(x)

                assert out.shape == (2, c2, 20, 20), \
                    f"形状不匹配: 输入{x.shape} 输出{out.shape}"
                print(f"✅ 测试通过 | 分组数: {model.groups} | 输出形状: {out.shape}")
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")


    run_tests()