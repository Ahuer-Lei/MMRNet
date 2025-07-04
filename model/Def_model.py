from .layers import *
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
import yaml
import numpy as np
from model.Eva_model import Evaluator

class DoubleConv(nn.Sequential):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        if mid_channel is None:
            mid_channel = out_channel
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel,out_channel)
        )


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x // 2, diff_y//2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


def normMask(mask, strength=0.5):
    """
    :return: to attention more region
    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask/(max_value*strength)
    mask = torch.clamp(mask, 0, 1)

    return mask

class UNet(nn.Module):
    def __init__(self, in_channel: int = 2, num_class: int = 2, bilinear: bool = True, base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.num_class = num_class
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channel, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)

        factor = 2 if bilinear else 1

        self.down4 = Down(base_c*8, base_c*16//factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.process = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, x):

        x1 = self.in_conv(x)  # [4,64,256,256]
        x2 = self.down1(x1)   # [4,128,128,128]
        x3 = self.down2(x2)   # [4,256,64,64]
        x4 = self.down3(x3)   # [4,512,32,32]
        x5 = self.down4(x4)   # [4,512,16,16]
        x = self.up1(x5, x4)  # [4,256,32,32]
        x = self.up2(x, x3)   # [4,128,64,64]
        x = self.up3(x, x2)   # [4,64,128,128]
        x = self.up4(x, x1)   # [4,64,256,256]

        return x



class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.un = UNet()
        self.MT = Evaluator(2, 1, 2)


        self.process = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        )


    def forward(self, nir, opt):
        x = torch.cat([nir, opt], dim=1)
        Pre_MD = self.MT(x)
        fake_opt = nir + 2*Pre_MD
        fake_opt = torch.clamp(fake_opt,-1,1)
        u = torch.cat([fake_opt, opt], dim=1)
        u = self.un(u)
        u = self.process(u)

        return u, Pre_MD, fake_opt