"""
in this file:
SSIM  MS-SSIM-L1 PSNR MSE
"""
import math
import warnings
from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.2,
                 compensation=200.0,
                 cuda_dev=0,
                 channel_num=3):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        self.channel_num = channel_num
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((self.channel_num * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # ch10,ch20...chn0,...,ch1m, ch2m...chnm
            for channel in range(self.channel_num):
                g_masks[self.channel_num * idx + channel, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.ger(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        
        mux = F.conv2d(x, self.g_masks, groups=self.channel_num, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=self.channel_num, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=self.channel_num, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=self.channel_num, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=self.channel_num, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 5*channel_num, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        # lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        lM = 1
        for channel in range(1, self.channel_num+1):
            lM *= l[:, -channel, :, :]

        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, channel_num, H, W]
        # average l1 loss in n channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel_num, length=self.channel_num),
                               groups=self.channel_num, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


class ssim_psnr_mse(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(ssim_psnr_mse, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    @staticmethod
    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = self._create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    @staticmethod
    def psnr_mse(img1, img2, std, mean):
        """
        :param img1: tensor
        :param img2: tensor
        :param std: tensor like [0.5,0.5,0.5]
        :param mean: tensor like [0.5,0.5,0.5]
        :return:
        """
        img1 = img1.squeeze().detach().cpu()
        img2 = img2.squeeze().detach().cpu()
        for t, m, s in zip(img1, mean, std):
            t.mul_(s.item()).add_(m.item())
        for t, m, s in zip(img2, mean, std):
            t.mul_(s.item()).add_(m.item())
        img1 = transforms.ToPILImage()(img1).convert('RGB')
        img2 = transforms.ToPILImage()(img2).convert('RGB')
        img1 = np.array(img1)
        img2 = np.array(img2)
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1.0e-10:
            return mse, 100
        PIXEL_MAX = 255
        return mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    @staticmethod
    def _create_window(window_size, channel):

        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

