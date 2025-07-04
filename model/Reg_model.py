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


# ResNet18/34的残差结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, block_num):
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.conv1 = nn.Conv2d(2, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        self.conv_last = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, down_sample=down_sample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

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

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.para_reg = resnet34()

        self.MT = Evaluator(2, 1, 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(512, 6)
       
    def forward(self, sar, opt):
        x = torch.cat([sar, opt], dim=1)
        Pre_MD = self.MT(x)
        fake_opt = sar + 2*Pre_MD
        fake_opt = torch.clamp(fake_opt,-1,1)

        input = torch.cat([fake_opt, opt], dim=1)
        out = self.para_reg(input)

        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        pre_tp = self.fc(out)
        return pre_tp, Pre_MD, fake_opt