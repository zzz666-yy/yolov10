
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.extra_modules import MANet
from ultralytics.nn.modules import Conv, Bottleneck, DWConv
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class Pzconv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c = int(dim * 0.5)
        self.conv1 = nn.Conv2d(
            self.c, self.c, 3,
            1, 1, groups=self.c
        )
        self.conv_first = Conv(dim, self.c, k=1, s=1)
        self.conv_final = Conv(self.c, dim, k=1, s=1)
        self.conv2 = nn.Conv2d(
            self.c, self.c, 5,
            1, 2, groups=self.c
        )

    def forward(self, x):
        x1 = self.conv_first(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv_final(x3)
        x5 = x4 + x  # 残差连接
        return x5

class MANet_Pzconv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            Pzconv(dim_hid),  # 替换原来的 DWConv
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))


class C2f_MANet(nn.Module):
    """CSP Bottleneck with MANet-enhanced feature fusion"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, p=1, kernel_size=3):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 初始通道扩展
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 最终输出卷积

        # 用MANet替换原始Bottleneck列表
        self.manet = MANet(
            c1=self.c,  # 输入通道（来自chunk后的半通道）
            c2=self.c,  # 输出保持相同通道
            n=n,  # 保持相同的Bottleneck数量
            shortcut=shortcut,  # 继承shortcut参数
            p=p,  # MANet的隐藏层扩展系数
            kernel_size=kernel_size,  # DWConv的核大小
            g=g,  # 分组卷积参数
            e=1.0  # 内部通道保持1:1
        )

    def forward(self, x):
        # 1. 初始通道扩展
        y = self.cv1(x)

        # 2. 通道分半处理
        y1, y2 = y.chunk(2, 1)  # 各得到self.c通道

        # 3. 用MANet处理第二分支
        y2_processed = self.manet(y2)

        # 4. 合并结果
        y = [y1, y2_processed]
        return self.cv2(torch.cat(y, 1))


class PzSPPF(nn.Module):
    """融合Pzconv和SPPF优势的复合模块"""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        # Pzconv部分
        self.pz_conv1 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1)
        self.pz_conv2 = Conv(c1, c1, 1, 1)
        self.pz_conv3 = nn.Conv2d(c1, c1, 5, 1, 2, groups=c1)

        # SPPF部分
        c_ = c1 // 2
        self.sppf_conv1 = Conv(c1 * 2, c_, 1, 1)  # 输入来自Pzconv和原始特征
        self.sppf_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.sppf_conv2 = Conv(c_ * 4, c2, 1, 1)

        # 最终融合
        self.final_conv = Conv(c1 + c2, c2, 1, 1)  # 融合局部和全局特征

    def forward(self, x):
        # Pzconv分支
        x1 = self.pz_conv1(x)
        x2 = self.pz_conv2(x1)
        x3 = self.pz_conv3(x2)
        pz_out = x3 + x  # 残差连接

        # SPPF分支
        sppf_input = torch.cat([x, pz_out], 1)  # 结合原始特征和Pzconv输出
        y = [self.sppf_conv1(sppf_input)]
        y.extend(self.sppf_pool(y[-1]) for _ in range(3))
        sppf_out = self.sppf_conv2(torch.cat(y, 1))

        # 特征融合
        combined = torch.cat([pz_out, sppf_out], 1)
        return self.final_conv(combined)




