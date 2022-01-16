import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

class SegNet(nn.Module):
    """A class for the segnet model with attention and denoising."""
    def __init__(self):
        super().__init__()
        channels_encoder = [3, 64, 128, 256, 512, 512] # 0th element: number of input chanels
        channels_decoder = [64,64,128,256,512,512] 
        # define first layer of encoder with 5 encoder blocks
        self.encoder = nn.ModuleList([nn.ModuleList([self.bn_conv_relu(channels_encoder[i], channels_encoder[i+1])]) for i in range(5)])
        self.decoder = nn.ModuleList([nn.ModuleList([self.bn_conv_relu(channels_decoder[i+1], channels_decoder[i])]) for i in range(5)])

        # define intermediate layes of each block
        for i in range(5):
            if i <2:
             self.encoder[i].append(self.bn_conv_relu(channels_encoder[i + 1], channels_encoder[i + 1]))
             self.decoder[i].append(self.bn_conv_relu(channels_decoder[i], channels_decoder[i]))

            else:
                self.encoder[i].append(nn.Sequential(self.bn_conv_relu(channels_encoder[i + 1], channels_encoder[i + 1]),
                                                         self.bn_conv_relu(channels_encoder[i + 1], channels_encoder[i + 1])))
                self.decoder[i].append(nn.Sequential(self.bn_conv_relu(channels_encoder[i], channels_encoder[i]),
                                                     self.bn_conv_relu(channels_encoder[i], channels_encoder[i])))

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # define attention for each task 
        # initialize first layer
        self.encoder_attention = nn.ModuleList([nn.ModuleList([self.attnt_layer([channels_encoder[1], channels_encoder[1], channels_encoder[1]])]) for _ in range(4)])
        self.decoder_attention = nn.ModuleList([nn.ModuleList([self.attnt_layer([2*channels_decoder[0], channels_decoder[0], channels_decoder[0]])]) for _ in range(2)])

        for i in range(4): #no of attention pipelines (task) in encoder
            for j in range(1,5): 
                self.encoder_attention[i].append(self.attnt_layer([2 * channels_encoder[j+1], channels_encoder[j+1], channels_encoder[j+1]]))
        
        for i in range(2): #no of attention pipelines (task) in decoder
            for j in range(1,5): 
                self.decoder_attention[i].append(self.attnt_layer([channels_decoder[j]+ channels_decoder[j+1], channels_decoder[j], channels_decoder[j]]))

        
        #define shared attention features (f)
        self.encoder_attnt_shared = nn.ModuleList([self.bn_conv_relu(channels_encoder[i], channels_encoder[i+1]) for i in range(1,5)])
        self.encoder_attnt_shared.append(self.bn_conv_relu(channels_encoder[-1], channels_encoder[-1])) 

        self.decoder_attnt_shared = nn.ModuleList([self.bn_conv_relu(channels_decoder[i+1], channels_decoder[i]) for i in range(0,4)])
        self.decoder_attnt_shared.append(self.bn_conv_relu(channels_decoder[-1], channels_decoder[-1]))
        
        self.flat=nn.Flatten()
        # number of task 4
        self.denoising = self.bn_conv_relu(64,3) # denoising task
        self.linear_class= nn.Linear(512*8*8,2)  # binary classification task
        self.linear_bb= nn.Linear(512*8*8,4) # bounding box prediction task
        self.target_seg =  self.bn_conv_relu(64,2) # denoising task
        #self.decoder = Decoder() # animal segmentaion task
    
    # clean the code? 
    def vgg_pretrained(self,vgg16):
        """A function for vgg weights of the pretrained model.
        
        Args:
        vgg16: model.
        """
        layers = list(vgg16.features.children()) #Getting all features of vgg 
        vgg_layers = []
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                vgg_layers.append(layer)

        encoder_layers = []
        for layers in self.encoder:
            for l in layers:
                for laye in l:
                    if isinstance(laye,Iterable):
                        for h in laye:
                            if isinstance(h, nn.Conv2d):
                                encoder_layers.append(h)
                    else:
                        if isinstance(laye, nn.Conv2d): 
                            encoder_layers.append(laye)

        for layer1, layer2 in zip(vgg_layers, encoder_layers):

            layer2.weight.data = layer1.weight.data
            layer2.bias.data = layer1.bias.data
            
    def bn_conv_relu(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):
        """A function creating an attention block consisting of convolution, batch normalisation and relu.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            padding (int, optional): Padding for the convolution stage. Default is 1.
        
        Returns:
            attnt_block (pytorch object): An attention block made up of three smaller layers.
        
        """

        layer=[]
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.BatchNorm2d(out_ch))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def attnt_layer(self, channel):
        """A function used to create an attention block from individual layers including convolutions, batch normalisation, relu and sigmoid.
        Returns:
            attnt_block (pytorch object): A structure made up of smaller individual layers.
        """
        attnt_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return attnt_block

    def forward(self,x):

        # define arrays for saving intermediate values 
        #during foward pass
        encoder_arr = np.empty((5,2),dtype=object) # store (u) & (p) from shared network
        decoder_arr = np.empty((5,2),dtype=object) # store (u) & (p) from shared network

        maxpool_arr,upsampling_arr,indices_arr = (np.empty((5),dtype=object) for _ in range(3))
        #,x_shapes_arr
      
        # durring attention
        attnt_encoder_arr = np.empty((4,5,3),dtype=object) # 4 tasks, 5 blocks, 3 input feature channels
        attnt_decoder_arr = np.empty((2,5,3),dtype=object) # 2 tasks, 5 blocks, 3 input feature channels

        # foward pass through global shared network
        # encoder
        for i in range(5):
            if i == 0:
                encoder_arr[i][0] = self.encoder[i][0](x)
            else:
                encoder_arr[i][0] = self.encoder[i][0](maxpool_arr[i - 1])

            encoder_arr[i][1] = self.encoder[i][1](encoder_arr[i][0])             
            maxpool_arr[i], indices_arr[i] = self.down_sampling(encoder_arr[i][1])
            #x_shapes_arr[i] = maxpool_arr[i].size() # storing indices and shape for upsamling 

        #decoder
        for i in range(5):
            if i == 0:
                upsampling_arr[i] = self.up_sampling(maxpool_arr[-1], indices_arr[-i - 1])
            else:
                upsampling_arr[i] = self.up_sampling(decoder_arr[i - 1][-1], indices_arr[-i - 1])
            
            decoder_arr[i][0] = self.decoder[-i - 1][0](upsampling_arr[i])
            decoder_arr[i][1] = self.decoder[-i - 1][1](decoder_arr[i][0])

        # foward pass through attention masks
        #encoder
        for i in range(4):
            for j in range(5):
                if j == 0:
                    attnt_encoder_arr[i][j][0] = self.encoder_attention[i][j](encoder_arr[j][0])
                else: # concatenate shared and task specific features from prev layer (g)
                    attnt_encoder_arr[i][j][0] = self.encoder_attention[i][j](torch.cat((encoder_arr[j][0], attnt_encoder_arr[i][j - 1][2]), dim=1))
                    
                attnt_encoder_arr[i][j][1] = (attnt_encoder_arr[i][j][0]) * encoder_arr[j][1]
                attnt_encoder_arr[i][j][2] = self.encoder_attnt_shared[j](attnt_encoder_arr[i][j][1])
                attnt_encoder_arr[i][j][2] = F.max_pool2d(attnt_encoder_arr[i][j][2], kernel_size=2, stride=2)
        
        #decoder
        for i in range(2):
            for j in range(5):
                if j == 0:
                    attnt_decoder_arr[i][j][0] = F.interpolate(attnt_encoder_arr[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                else: # concatenate shared and task specific features from prev layer (g)
                    attnt_decoder_arr[i][j][0] =  F.interpolate(attnt_decoder_arr[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)

                attnt_decoder_arr[i][j][0] = self.decoder_attnt_shared[-j - 1](attnt_decoder_arr[i][j][0])
                attnt_decoder_arr[i][j][1] = self.decoder_attention[i][-j - 1](torch.cat((upsampling_arr[j], attnt_decoder_arr[i][j][0]), dim=1))
                attnt_decoder_arr[i][j][2] = (attnt_decoder_arr[i][j][1]) * decoder_arr[j][-1]

        # final tasks
        target_seg_pred = self.target_seg(attnt_decoder_arr[0][-1][-1])
        aux_pred_denoising = self.denoising(attnt_decoder_arr[1][-1][-1])

        aux_pred_c = self.linear_class(self.flat(attnt_encoder_arr[2][-1][-1]))
        aux_pred_bb = self.linear_bb(self.flat(attnt_encoder_arr[3][-1][-1]))

        return aux_pred_c,aux_pred_bb, target_seg_pred,aux_pred_denoising
        
