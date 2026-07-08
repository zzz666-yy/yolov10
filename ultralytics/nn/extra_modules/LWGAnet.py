import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from typing import Optional


class SharedGroupConv(nn.Module):
    """权重共享的分组卷积"""

    def __init__(self, in_channels, out_channels=None, kernel_size=3, groups=4):
        super().__init__()
        out_channels = out_channels or in_channels

        # 确保通道数能被groups整除
        assert in_channels % groups == 0, f"输入通道{in_channels}必须能被groups={groups}整除"
        assert out_channels % groups == 0, f"输出通道{out_channels}必须能被groups={groups}整除"

        self.groups = groups
        self.conv = nn.Conv2d(in_channels // self.groups,
                              out_channels // self.groups,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.norm = nn.GroupNorm(self.groups, out_channels)

        # 权重初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = torch.cat([self.conv(x[:, i]) for i in range(self.groups)], dim=1)
        return self.norm(x.view(B, -1, H, W))


class UltraLightLWGA(nn.Module):
    """终极轻量版LWGA"""

    def __init__(self,
                 c1: int,
                 c2: Optional[int] = None,
                 stage: int = 1,
                 mlp_ratio: float = 1.2,
                 drop_path: float = 0.1,
                 groups: int = 4,
                 att_ratio: float = 0.25):
        super().__init__()
        # 参数校验
        assert c1 % groups == 0, f"输入通道{c1}必须能被groups={groups}整除"
        assert 0 < att_ratio < 0.5, "注意力比例必须在0到0.5之间"

        self.c2 = c2 if c2 is not None else c1
        self.stage = stage
        self.groups = groups

        # 确保att_dim能被groups整除
        self.att_dim = max(groups, int(c1 * att_ratio))
        self.att_dim = self.att_dim - (self.att_dim % groups)

        # 1. 共享特征提取层
        self.base_conv = SharedGroupConv(c1, groups=groups)

        # 2. 动态通道分配
        self.channel_split = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, self.att_dim, 1),
                nn.GroupNorm(groups, self.att_dim)
            ) for _ in range(4)
        ])

        # 3. 轻量注意力机制
        self.pa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.att_dim, max(groups, self.att_dim // 8), 1),
            nn.ReLU(),
            nn.Conv2d(max(groups, self.att_dim // 8), self.att_dim, 1),
            nn.Hardsigmoid()
        )

        # 4. 阶段自适应处理
        if stage == 3:  # 高层特征
            self.ga = nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Conv2d(self.att_dim, self.att_dim, 1),
                nn.Sigmoid()
            )
        else:  # 中低层特征
            self.ga = SharedGroupConv(self.att_dim, kernel_size=5, groups=groups)

        # 5. 极致轻量MLP
        hidden_dim = max(groups, int(c1 * mlp_ratio))
        hidden_dim = hidden_dim - (hidden_dim % groups)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden_dim, 1, groups=groups),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, c1, 1, groups=groups)
        )

        # 6. 投影层
        if c1 != self.c2:
            # 确保中间通道数能被groups整除
            proj_mid = max(groups, (c1 + self.c2) // 2)
            proj_mid = proj_mid - (proj_mid % groups)
            self.proj = nn.Sequential(
                nn.Conv2d(c1, proj_mid, 1, groups=groups),
                nn.Conv2d(proj_mid, self.c2, 1)
            )
        else:
            self.proj = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 7. 完全重构的蒸馏头
        if stage == 3:
            # 动态计算中间通道数，确保与输入通道匹配
            mid_channels = max(groups, c1 // 16)
            mid_channels = mid_channels - (mid_channels % groups)

            self.distill = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, mid_channels, 1),
                nn.ReLU(),
                nn.Conv2d(mid_channels, c1, 1)  # 输出通道与输入相同
            )
            # 初始化最后一层卷积的权重为接近零的值
            nn.init.constant_(self.distill[-1].weight, 1e-3)
            nn.init.constant_(self.distill[-1].bias, 0)
        else:
            self.distill = nn.Identity()

    def forward(self, x):
        # 保存原始尺寸
        orig_size = x.shape[2:]

        # 1. 共享特征提取
        x_base = self.base_conv(x)

        # 2. 多分支处理
        branches = []
        for i, conv in enumerate(self.channel_split):
            branch = conv(x_base)
            if i == 0:  # 点注意力分支
                branch = branch * self.pa(branch)
            elif i == 3:  # 全局注意力分支
                branch = self.ga(branch)
                branch = F.interpolate(branch, orig_size, mode='nearest')
            branches.append(branch)

        # 3. 特征融合
        try:
            x_out = torch.cat(branches, dim=1)
        except RuntimeError:
            target_size = branches[0].shape[2:]
            x_out = torch.cat([F.interpolate(b, target_size, mode='nearest')
                               for b in branches], dim=1)

        # 4. 残差连接
        x = x + self.drop_path(self.mlp(x_out))

        # 5. 投影输出
        x = self.proj(x)

        # 6. 高层特征蒸馏
        if self.stage == 3:
            # 直接相加，因为蒸馏头输出通道已经与输入匹配
            return x + 0.1 * self.distill(x)
        return x


# 增强测试代码
if __name__ == "__main__":
    def test_module(c1, stage, groups=4):
        print(f"\nTesting c1={c1}, stage={stage}, groups={groups}")
        x = torch.randn(2, c1, 32, 32)
        model = UltraLightLWGA(c1, stage=stage, groups=groups)

        # 测试蒸馏头
        if stage == 3:
            distill_out = model.distill(x)
            assert distill_out.shape == x.shape, f"蒸馏头形状不匹配: {distill_out.shape} != {x.shape}"

        out = model(x)
        assert out.shape == (2, c1, 32, 32), f"输出形状不匹配: {out.shape}"
        print(f"测试通过: c1={c1}, stage={stage}, groups={groups}")


    # 测试不同配置
    test_cases = [
        (256, 3, 4),  # 原始出错配置
        (320, 3, 4),  # 原始出错配置
        (64, 1, 4),
        (128, 2, 4),
        (512, 3, 8),
        (48, 1, 4),
        (96, 2, 8)
    ]

    for c1, stage, groups in test_cases:
        test_module(c1, stage, groups)

    print("\n所有测试用例均通过!")