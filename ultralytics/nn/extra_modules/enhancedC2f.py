import torch
import torch.nn as nn
from torch.nn import init


class Conv(nn.Module):
    """Standard convolution with optional activation"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2  # auto-pad
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depth-wise convolution"""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.dwconv = Conv(c1, c2, k, s, g=c1, act=act)

    def forward(self, x):
        return self.dwconv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck with optional shortcut"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)  # 使用k[0]
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 使用k[1]
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class EnhancedC2f(nn.Module):
    """Enhanced C2f module combining advantages of MANet and C2f"""

    def __init__(self, c1, c2, n=1, shortcut=False, p=0.5, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.enhance = nn.Sequential(
            Conv(2 * self.c, int(p * 2 * self.c), 1, 1),
            DWConv(int(p * 2 * self.c), int(p * 2 * self.c), kernel_size, 1),
            Conv(int(p * 2 * self.c), self.c, 1, 1)
        )

        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

        self._initialize_weights()

    def forward(self, x):
        y = self.cv1(x)
        y0, y1 = y.chunk(2, 1)
        y2 = self.enhance(y)

        y = [y0, y1, y2]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)