import torch
import torch.nn.functional as F
#from torchsummary import summary
from torch import nn

from modules import DilatedResidualBlock, NLB, MTUR, DepthWiseDilatedResidualBlock




class MTURNet_YKCCR(nn.Module):
    def __init__(self, num_features=64):
        super(MTURNet_YKCCR, self).__init__()
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

        ############################################ MT prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv1_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )

        self.f_process = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=64, num_channels=64),
            nn.SELU(inplace=True),
        )

        self.depth_process = nn.Sequential(
            nn.Conv2d(65, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )



        ############################################ underwater enhanced network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body1 = nn.Sequential(
            DilatedResidualBlock(num_features, 1),)
        self.body2 = nn.Sequential(
            DilatedResidualBlock(num_features, 2))
        self.body4 = nn.Sequential(
            DilatedResidualBlock(num_features, 4))
        self.body8 = nn.Sequential(
            DilatedResidualBlock(num_features, 8))

        self.mturb = MTUR(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        self.c3_31 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_32 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_33 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_34 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_35 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_36 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_37 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_38 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.c3_39 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.c3_310 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

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
        ################################## MT prediction network

        d_f2 = self.conv2(f)  # torch.Size([8, 64, 32, 32])

        d_f10 = self.conv10(d_f2)  # torch.Size([8, 64, 64, 64])

        depth_pred = self.depth_pred(d_f10)  # torch.Size([8, 1, 128, 128])

        ################################## underwater enhanced network


        f = self.body1(f)  # torch.Size([8, 64, 64, 64])
        c = self.c3_31(d_f10)
        f = self.body1(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_32(d_f10)
        f = self.body2(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_33(d_f10)
        f = self.body2(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_34(d_f10)
        f = self.body4(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_35(d_f10)
        f = self.body8(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_36(d_f10)
        f = self.body4(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_37(d_f10)
        f = self.body2(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_38(d_f10)
        f = self.body2(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_39(d_f10)
        f = self.body1(f+c)  # torch.Size([8, 64, 64, 64])
        c = self.c3_310(d_f10)
        f = self.body1(f+c)  # torch.Size([8, 64, 64, 64])
        f = self.f_process(f)  # f_process(f) torch.Size([8, 64, 128, 128])
        f = torch.cat((f, depth_pred.detach()), 1)  # torch.Size([8, 65, 128, 128])
        f = self.depth_process(f)  # torch.Size([8, 128, 128, 128])
        x = self.output(f)

        x = (x * self.std + self.mean).clamp(min=0, max=1)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print('x', x.shape)
        # input()
        if self.training:
            return x, depth_pred

        return x

