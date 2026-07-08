import torch
import torch.nn as nn
import math
from torch.nn import Conv2d

from ultralytics.nn.extra_modules import SimAM
from ultralytics.nn.modules import Conv



class Mix(nn.Module):
    def __init__(self):
        super(Mix, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


class AFGCAttention(nn.Module):
    # Adaptive Fine-Grained Channel Attention
    def __init__(self, channel, b=1, gamma=2):
        super(AFGCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input * out


class HybridAttention(nn.Module):
    """
    Position-wise Spatial Attention module with AFGC Attention in the a branch.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
        simam (AFGCAttention): SimAM for the a branch.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention modules, and feed-forward network."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv2d(2 * self.c, c1, 1)

        self.attn = AFGCAttention(self.c)
        self.ffn = nn.Sequential(Conv2d(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))
        self.simam = SimAM(self.c)  # SimAM for the a branch

    def forward(self, x):
        """
        Forward pass of the modified PSA module with AFGC Attention in the a branch.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        a = self.simam(a)  # Apply AFGC Attention to the a branch
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
