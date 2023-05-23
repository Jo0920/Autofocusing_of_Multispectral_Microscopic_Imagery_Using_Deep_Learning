import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
from torchsummary import summary
from blitz.modules import BayesianConv2d, BayesianConvTranspose2d
from blitz.utils import variational_estimator

cfg = get_cfg()

##############################
#           U-NET 2D
##############################

class UNetDown_2d_bayesian(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown_2d_bayesian, self).__init__()
        layers = [BayesianConv2d(in_size, out_size, kernel_size=(4, 4), stride=2, padding=1, bias=False).to('cuda')]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp_2d_bayesian(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp_2d_bayesian, self).__init__()
        layers = [
            BayesianConvTranspose2d(in_size, out_size, kernel_size=(4, 4), stride=2, padding=1, bias=False).to('cuda'),
            # nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=True).to('cuda'),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


@variational_estimator
class GeneratorUNet_2d_bayesian(nn.Module):
    def __init__(self, in_channels=3 * cfg.spectrum_num, out_channels=3 * cfg.spectrum_num):
        super(GeneratorUNet_2d_bayesian, self).__init__()

        self.down1 = UNetDown_2d_bayesian(in_channels, 64, normalize=False)
        self.down2 = UNetDown_2d_bayesian(64, 128)
        self.down3 = UNetDown_2d_bayesian(128, 256)
        self.down4 = UNetDown_2d_bayesian(256, 512)#, dropout=0.5)
        self.down5 = UNetDown_2d_bayesian(512, 512)#, dropout=0.5)
        self.down6 = UNetDown_2d_bayesian(512, 512)#, dropout=0.5)
        self.down7 = UNetDown_2d_bayesian(512, 512)#, dropout=0.5)
        self.down8 = UNetDown_2d_bayesian(512, 512, normalize=False)#, dropout=0.5)

        self.up1 = UNetUp_2d_bayesian(512, 512)#, dropout=0.5)
        self.up2 = UNetUp_2d_bayesian(1024, 512)#, dropout=0.5)
        self.up3 = UNetUp_2d_bayesian(1024, 512)#, dropout=0.5)
        self.up4 = UNetUp_2d_bayesian(1024, 512)#, dropout=0.5)
        self.up5 = UNetUp_2d_bayesian(1024, 256)
        self.up6 = UNetUp_2d_bayesian(512, 128)
        self.up7 = UNetUp_2d_bayesian(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            BayesianConv2d(128, out_channels, (4, 4), padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

