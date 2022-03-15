import torch
import torch.nn.functional as F
#from torchsummary import summary
from torch import nn

from modules import DilatedResidualBlock, NLB, MTUR, DepthWiseDilatedResidualBlock








class basic(nn.Module):
    def __init__(self, num_features=64):
        super(basic, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride = 2 ,padding=1), nn.ReLU(),
			nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.tail = nn.Sequential(
            #nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        self.f_process = nn.Sequential(
            nn.ConvTranspose2d(num_features, 64 , kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=64, num_channels=64),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            # nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        f = self.head(x)
        f = self.body(f)

        f = self.f_process(f)
        x = self.output(f)
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x







