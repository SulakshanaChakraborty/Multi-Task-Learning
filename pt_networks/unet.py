import torch
import torch.nn as nn
from load_data import create_data_loaders
import time
import torchvision
import torch.nn.functional as F
import numpy as np 
from collections import OrderedDict



class Block(nn.Module):
    def __init__(self, ch_input, ch_output):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(ch_input, ch_output, 3)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2',nn.Conv2d(ch_output, ch_output, 3)),
            ('relu2',nn.ReLU(inplace=True))
        ]))
    
    def forward(self,input):
        output = self.block(input)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=(3,64,128,256,512,1024)):
    #  def __init__(self, chs=(3,64,17,34,68,136)):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()

        block_list = []
        for ch_idx in range(len(channels)-1):
             block = Block(channels[ch_idx],channels[ch_idx+1])
             block_list.append(block)

        self.encoder_blocks.extend(block_list)

        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self,input):

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
        for idx in range(len(channels)-1):
            block = nn.ConvTranspose2d(channels[idx],channels[idx+1],kernel_size=2,stride=2)
            up_list.append(block)
        
        self.upsampling.extend(up_list)

        self.decoder_blocks = nn.ModuleList()

        dec_list = []
        for dec_idx in range(len(channels)-1):
            dec_block = Block(channels[dec_idx], channels[dec_idx+1])
            dec_list.append(dec_block)
        
        self.decoder_blocks.extend(dec_list)
        
    def forward(self, input, enc_channels):
        for idx in range(len(self.channels)-1):
            input = self.upsampling[idx](input)
            enc_chan = self.crop(enc_channels[idx], input)
            input = torch.cat([input, enc_chan], dim=1)
            input = self.decoder_blocks[idx](input)
        return input
    
    def crop(self, enc_chans, input):
        height = input.shape[2]
        width = input.shape[3]
        enc_chans   = torchvision.transforms.CenterCrop([height, width])(enc_chans)
        return enc_chans

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=2, retain_dim=True, out_sz=(256,256)): 

        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz


    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        labels = torch.Tensor([0.8, 0.2]).repeat(x.size()[0], 1)
        bboxes = torch.Tensor([1, 2,3 ,4]).repeat(x.size()[0], 1)
        return labels, bboxes, out