import torch
import torch.nn as nn

import resnet
import gated_spatial_conv as gsc


class BaseStream(nn.Module):
    def __init__(self):
        super(BaseStream, self).__init__()

        self.layer1 = torch.nn.Conv2d(2, 2, 3)
        self.layer2 = torch.nn.Conv2d(2, 2, 3)
        self.layer3 = torch.nn.Conv2d(2, 2, 3)
        self.layer4 = torch.nn.Conv2d(2, 2, 3)
        self.layer5 = torch.nn.Conv2d(2, 2, 3)
        self.layer6 = torch.nn.Conv2d(2, 2, 3)
        self.layer7 = torch.nn.Conv2d(2, 2, 3)

    def get_layers(self):
        return self.layer1, self.layer3, self.layer5, self.layer7


class GSCNN:
    def __init__(self):
        super(GSCNN, self).__init__()
        # Define base stream.
        self.base = BaseStream()
        self.base_layer_1 = nn.Conv2d(2, 3, 1)
        self.base_layer_2 = nn.Conv2d(2, 3, 1)
        self.base_layer_3 = nn.Conv2d(2, 3, 1)
        self.base_layer_4 = nn.Conv2d(2, 3, 1)
        self.base_layer_5 = nn.Conv2d(2, 3, 1)
        self.base_layer_6 = nn.Conv2d(2, 3, 1)
        self.base_layer_7 = nn.Conv2d(2, 3, 1)

        # Get layers info from base stream (4 layers)
        self.bl1 = self.base_layer_1
        self.bl2 = self.base_layer_3
        self.bl3 = self.base_layer_5
        self.bl4 = self.base_layer_7

        # Create Resnet blocks
        self.res1 = resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        # Create Gated Convolutional layers (Attentions)
        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        # ASPP module.
        self.aspp = 1

    def forward(self, x):
        # Forward pass in base stream.

        bl1 = self.base_layer_1(x)
        bl2 = self.base_layer_2()

        return x


class ParallelStream(nn.Module):
    def __init__(self):
        super(ParallelStream, self).__init__()

    def forward(self, x):