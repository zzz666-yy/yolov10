import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PartialConv3(nn.Module):
    """轻量版部分卷积，默认实现split_cat模式"""

    def __init__(self, dim, n_div=4, fw_type='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.conv = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if fw_type == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        # 通道分组处理
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), 1)


class LightweightMS(nn.Module):
    """轻量级多尺度模块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 分支1: 3x3深度可分离卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 分支2: 空洞分组卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2, groups=max(in_ch // 2, 1), bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 分支3: 全局上下文
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 融合层
        self.fuse = nn.Conv2d(3 * out_ch, in_ch, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = F.interpolate(self.branch3(x), size=x.shape[2:], mode='nearest')
        return self.fuse(torch.cat([b1, b2, b3], dim=1))


class LiteAttention(nn.Module):
    """轻量级混合注意力"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1, groups=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        return x * ca * sa


class DynamicFusion(nn.Module):
    """动态门控融合"""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 4, 8), 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        g = self.gate(x1 + x2)  # [B,2,1,1]
        return g[:, 0:1] * x1 + g[:, 1:2] * x2


class ShiftMLP(nn.Module):
    """位移操作的轻量MLP"""

    def __init__(self, dim):
        super().__init__()
        self.shift_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1, bias=False)
        )

    def forward(self, x):
        x_shift = torch.roll(x, shifts=(1, 1), dims=(2, 3))  # 零FLOPs位移
        x = self.shift_conv(x + x_shift)
        return self.channel_mixer(x)


class Lite_Faster_Block(nn.Module):
    """完整轻量化改进版"""

    def __init__(self, inc, dim, n_div=8, mlp_ratio=2, drop_path=0.1):
        super().__init__()
        # 通道调整
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(inc, dim, 1, bias=False),
                nn.BatchNorm2d(dim)
            )

        # 多尺度模块
        self.ms_block = LightweightMS(dim, dim // 4)

        # 空间混合
        self.pconv = PartialConv3(dim, n_div)

        # 注意力
        self.attn = LiteAttention(dim)

        # 动态融合
        self.fusion = DynamicFusion(dim)

        # MLP
        self.mlp = ShiftMLP(dim)

        # 正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 通道调整
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)

        shortcut = x

        # 多尺度分支
        ms_feat = self.ms_block(x)

        # 主路径
        x_main = self.pconv(x)
        x_main = self.attn(x_main)

        # 动态融合
        x = self.fusion(ms_feat, x_main)

        # 残差连接
        x = shortcut + self.drop_path(self.mlp(x))

        return x