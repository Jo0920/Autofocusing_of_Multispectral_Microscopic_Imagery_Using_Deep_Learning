import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
from torchsummary import summary

cfg = get_cfg()

'''only deterministic models here in this file'''

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET 2D
##############################
class UNetDown_2d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown_2d, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp_2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp_2d, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
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


class GeneratorUNet_2d(nn.Module):
    def __init__(self, in_channels=3 * cfg.spectrum_num, out_channels=3 * cfg.spectrum_num, droput=0.5):
        super(GeneratorUNet_2d, self).__init__()

        self.down1 = UNetDown_2d(in_channels, 64, normalize=False)
        self.down2 = UNetDown_2d(64, 128)
        self.down3 = UNetDown_2d(128, 256)
        self.down4 = UNetDown_2d(256, 512, dropout=droput)
        self.down5 = UNetDown_2d(512, 512, dropout=droput)
        self.down6 = UNetDown_2d(512, 512, dropout=droput)
        self.down7 = UNetDown_2d(512, 512, dropout=droput)
        self.down8 = UNetDown_2d(512, 512, normalize=False, dropout=droput)

        self.up1 = UNetUp_2d(512, 512, dropout=droput)
        self.up2 = UNetUp_2d(1024, 512, dropout=droput)
        self.up3 = UNetUp_2d(1024, 512, dropout=droput)
        self.up4 = UNetUp_2d(1024, 512, dropout=droput)
        self.up5 = UNetUp_2d(1024, 256)
        self.up6 = UNetUp_2d(512, 128)
        self.up7 = UNetUp_2d(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
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


##############################
#        Discriminator 2D
##############################
class Discriminator_2d(nn.Module):
    def __init__(self, in_channels=3 * cfg.spectrum_num):
        super(Discriminator_2d, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False), #3
            *discriminator_block(64, 128), #3*2
            *discriminator_block(128, 256), #3*4
            *discriminator_block(256, 512, stride=1), #3*8
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False) #3*8
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

#################################
#        Discriminator 2D cycle
#################################
class Discriminator_2d_cycle(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator_2d_cycle, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        return self.model(img_A)

##############################
#           U-NET 3D
##############################

class UNetDown_3d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_ms=1, stride_ms=1, padding_ms=0):
        super(UNetDown_3d, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, kernel_size=(kernel_ms, 4, 4), stride=(stride_ms, 2, 2),
                            padding=(padding_ms, 1, 1),
                            bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp_3d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, zeroPad=False, kernel_ms=1, stride_ms=1, padding_ms=0):
        super(UNetUp_3d, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size=(kernel_ms, 4, 4), stride=(stride_ms, 2, 2),
                               padding=(padding_ms, 1, 1),
                               bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if zeroPad:
            layers.append(nn.Conv3d(out_size, out_size, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                                    padding=(1, 0, 0),
                                    bias=False))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet_3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet_3d, self).__init__()

        self.down1 = UNetDown_3d(in_channels, 64, kernel_ms=2, stride_ms=2, normalize=False)
        self.down2 = UNetDown_3d(64, 128, kernel_ms=2, stride_ms=2)
        self.down3 = UNetDown_3d(128, 256, kernel_ms=2, stride_ms=2)
        self.down4 = UNetDown_3d(256, 512, kernel_ms=2, stride_ms=2, dropout=0.5)
        self.down5 = UNetDown_3d(512, 512, dropout=0.5)
        self.down6 = UNetDown_3d(512, 512, dropout=0.5)
        self.down7 = UNetDown_3d(512, 512, dropout=0.5)
        self.down8 = UNetDown_3d(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp_3d(512, 512, dropout=0.5)
        self.up2 = UNetUp_3d(1024, 512, dropout=0.5)
        self.up3 = UNetUp_3d(1024, 512, dropout=0.5)
        self.up4 = UNetUp_3d(1024, 512, dropout=0.5)
        self.up5 = UNetUp_3d(1024, 256, kernel_ms=2, stride_ms=2)
        self.up6 = UNetUp_3d(512, 128, kernel_ms=2, stride_ms=2)
        self.up7 = UNetUp_3d(256, 64, zeroPad=True, kernel_ms=2, stride_ms=2)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv3d(128, out_channels, (1, 4, 4), padding=(0, 1, 1)),
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


##############################
#        Discriminator 3D
##############################

class Discriminator_3d(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_3d, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, kernel_ms=1, stride_ms=1, padding_ms=0):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=(kernel_ms, 4, 4), stride=(stride_ms, 2, 2),
                                padding=(padding_ms, 1, 1))]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, kernel_ms=3, stride_ms=3),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv3d(512, 1, kernel_size=(1, 4, 4), padding=(0, 1, 1), bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


