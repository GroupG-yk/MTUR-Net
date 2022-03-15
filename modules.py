
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import numpy as np

class MTUR(nn.Module):
    def __init__(self, in_channels):
        super(MTUR, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)



    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)

		# [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        ### appearance relation map
        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

		# [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        Ra = F.softmax(torch.bmm(theta, phi), 2)


        ### MT relation map
        depth1 = F.interpolate(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        depth2 = F.interpolate(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))

        # n, (h / 4) * (w / 4), (h / 8) * (w / 8)
        depth1_expand = depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        depth2_expand = depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))

        Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))



        Rd = F.softmax(Rd, 2)


        S = F.softmax(Ra * Rd, 2)


        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)


class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


class DepthWiseDilatedResidualBlock(nn.Module):
    def __init__(self, reduced_channels, channels, dilation):
        super(DepthWiseDilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(

		    # pw
		    nn.Conv2d(channels, channels * 2, 1, 1, 0, 1, bias=False),
			nn.ReLU6(inplace=True),
		    # dw
		    nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False),
		    nn.ReLU6(inplace=True),
		    # pw-linear
		    nn.Conv2d(channels*2, channels, 1, 1, 0, 1, 1, bias=False)
        )

        self.conv1 = nn.Sequential(

			nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
					  bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(channels, channels, 1, 1, 0, 1, 1, bias=False)
		)


    def forward(self, x):
        res = self.conv1(self.conv0(x))
        return res + x


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation ):
        super(DilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        return x + conv1














