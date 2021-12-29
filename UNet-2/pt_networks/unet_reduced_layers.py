import torch
import torch.nn as nn
from load_data import create_data_loaders
import time as time
import torchvision
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# https://amaarora.github.io/2020/09/13/unet.html
# https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

train_path = "datasets-oxpet/train"
validation_path = "datasets-oxpet/val"
test_path = "datasets-oxpet/test"

train_loader, validation_loader, test_loader = create_data_loaders(train_path, validation_path, test_path, batch_size=4)


class Block(nn.Module):
    def __init__(self, ch_input, ch_output):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(ch_input, ch_output, 3)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(ch_output, ch_output, 3)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, input):
        output = self.block(input)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        #  def __init__(self, chs=(3,64,17,34,68,136)):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()

        block_list = []
        for ch_idx in range(len(channels) - 1):
            block = Block(channels[ch_idx], channels[ch_idx + 1])
            block_list.append(block)

        self.encoder_blocks.extend(block_list)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):

        filters = []

        for blk in self.encoder_blocks:
            input = blk(input)
            filters.append(input)
            input = self.maxpool1(input)

        return filters


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()

        self.upsampling = nn.ModuleList()
        self.channels = channels

        up_list = []
        for idx in range(len(channels) - 1):
            block = nn.ConvTranspose2d(channels[idx], channels[idx + 1], kernel_size=2, stride=2)
            up_list.append(block)

        self.upsampling.extend(up_list)

        self.decoder_blocks = nn.ModuleList()

        dec_list = []
        for dec_idx in range(len(channels) - 1):
            dec_block = Block(channels[dec_idx], channels[dec_idx + 1])
            dec_list.append(dec_block)

        self.decoder_blocks.extend(dec_list)

    def forward(self, input, enc_channels):
        for idx in range(len(self.channels) - 1):
            input = self.upsampling[idx](input)
            enc_chan = self.crop(enc_channels[idx], input)
            input = torch.cat([input, enc_chan], dim=1)
            input = self.decoder_blocks[idx](input)
        return input

    def crop(self, enc_chans, input):
        height = input.shape[2]
        width = input.shape[3]
        enc_chans = torchvision.transforms.CenterCrop([height, width])(enc_chans)
        return enc_chans


class UNet(nn.Module):
    # def __init__(self, enc_chans=(3,64,128,256,512,1024), dec_chans=(1024, 512, 256, 128, 64), num_class=2, retain_dim=True, output_size=(256,256)):
    # def __init__(self, enc_chans=(3,64,128), dec_chans=(128, 64), num_class=2, retain_dim=True, output_size=(256,256)):
    def __init__(self, enc_chans=(3, 56, 112), dec_chans=(112, 56), num_class=2, retain_dim=True,
                 output_size=(256, 256)):

        super().__init__()

        self.encoder = Encoder(enc_chans)
        self.decoder = Decoder(dec_chans)
        self.head = nn.Conv2d(dec_chans[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = output_size

    def forward(self, x):
        enc_chans = self.encoder(x)
        out = self.decoder(enc_chans[::-1][0], enc_chans[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

