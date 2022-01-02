import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

class SegNet(nn.Module):
    def __init__(self,noisy= True):
        super().__init__()
        channels = [3, 64, 128, 256, 512, 512] # 0th element: number of input chanels

        # define encoder with 5 encoder blocks
        self.encoder = nn.ModuleList([nn.ModuleList([self.bn_conv_relu(channels[i], channels[i+1])]) for i in range(5)])
        for i in range(5):
            if i <2:
             self.encoder[i].append(self.bn_conv_relu(channels[i + 1], channels[i + 1]))
            else:
                self.encoder[i].append(nn.Sequential(self.bn_conv_relu(channels[i + 1], channels[i + 1]),
                                                         self.bn_conv_relu(channels[i + 1], channels[i + 1])))

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # define attention for each task 
        # initialize first layer
        self.attention = nn.ModuleList([ nn.ModuleList([self.attnt_layer([channels[1], channels[1], channels[1]])]) for _ in range(3)])
        for i in range(3): #no of tasks
            for j in range(1,5): 
                self.attention[i].append(self.attnt_layer([2 * channels[j+1], channels[j+1], channels[j+1]]))
        
        #define shared attention features (f)
        self.attnt_shared = nn.ModuleList([self.bn_conv_relu(channels[i], channels[i+1]) for i in range(1,5)])
        self.attnt_shared.append(self.bn_conv_relu(channels[-1], channels[-1]))
        
        self.flat=nn.Flatten()
        #print(self.attention)
        self.linear_class=nn.Linear(512*8*8,2)  # binary classification task
        self.linear_bb= nn.Linear(512*8*8,4) # bounding box prediction task
        self.decoder = Decoder(noisy= True) # animal segmentaion task
        self.noisy = noisy
    
    def vgg_pretrained(self,vgg16):
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

        print("encoder_layers len",len(encoder_layers))
        print("vgg_layers len",len(vgg_layers))

        for layer1, layer2 in zip(vgg_layers, encoder_layers):
            print("############")
            print("layer_vgg:",layer1)
            print("layer_encoder:",layer2)
            print("############")
            

            layer2.weight.data = layer1.weight.data
            layer2.bias.data = layer1.bias.data
            
    def bn_conv_relu(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):

        layer=[]
        layer.append(nn.BatchNorm2d(in_ch))
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def attnt_layer(self, channel):
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
        maxpool_arr,indices_arr,x_shapes_arr = (np.empty((5),dtype=object) for _ in range(3))
      
        # durring attention
        attnt_arr = np.empty((3,5,3),dtype=object) # 3 tasks, 5 blocks, 3 input feature channels
        
        # foward pass through global shared network
        for i in range(5):
            if i == 0:
                encoder_arr[i][0] = self.encoder[i][0](x)
            else:
                encoder_arr[i][0] = self.encoder[i][0](maxpool_arr[i - 1])

            encoder_arr[i][1] = self.encoder[i][1](encoder_arr[i][0])             
            maxpool_arr[i], indices_arr[i] = self.down_sampling(encoder_arr[i][1])
            x_shapes_arr[i] = maxpool_arr[i].size() # storing indices and shape for upsamling 

        # foward pass through attention masks
        for i in range(3):
            for j in range(5):
                if j == 0:
                    attnt_arr[i][j][0] = self.attention[i][j](encoder_arr[j][0])
                else: # concatenate shared and task specific features from prev layer (g)
                    attnt_arr[i][j][0] = self.attention[i][j](torch.cat((encoder_arr[j][0], attnt_arr[i][j - 1][2]), dim=1))
                    
                attnt_arr[i][j][1] = (attnt_arr[i][j][0]) * encoder_arr[j][1]
                attnt_arr[i][j][2] = self.attnt_shared[j](attnt_arr[i][j][1])
                attnt_arr[i][j][2] = F.max_pool2d(attnt_arr[i][j][2], kernel_size=2, stride=2)
        
        predictions = self.decoder(attnt_arr[0][-1][-1],indices_arr,x_shapes_arr)
        flat_c = self.flat(attnt_arr[1][-1][-1])
        flat_bb = self.flat(attnt_arr[2][-1][-1])
        aux_pred_c = self.linear_class(flat_c)
        aux_pred_bb = self.linear_bb(flat_bb)
        if len(predictions)>0: 
            target_seg,target_denoise = predictions
            preds= (aux_pred_c,aux_pred_bb,target_seg,target_denoise)

        else: preds= (aux_pred_c,aux_pred_bb,predictions)

        return preds
        

class Decoder(nn.Module):

    def __init__(self,noisy):

        super().__init__()
        self.layer_52_t=self.bn_conv_relu(512,512)
        self.layer_51_t=self.bn_conv_relu(512,512)
        self.layer_50_t=self.bn_conv_relu(512,512)
        self.noisy = noisy
        
        self.layer_42_t=self.bn_conv_relu(512,512)
        self.layer_41_t=self.bn_conv_relu(512,512)
        self.layer_40_t=self.bn_conv_relu(512,256)

        self.layer_32_t=self.bn_conv_relu(256,256)
        self.layer_31_t=self.bn_conv_relu(256,256)
        self.layer_30_t=self.bn_conv_relu(256,128)

        self.layer_21_t=self.bn_conv_relu(128,128)
        self.layer_20_t=self.bn_conv_relu(128,64)

    
        self.layer_11_t=self.bn_conv_relu(64,64)
        self.layer_10_t=self.bn_conv_relu(64,2) 
        if self.noisy == True:
            self.layer_10_denoising=self.bn_conv_relu(64,3) 

        self.upsample = nn.MaxUnpool2d(2, stride=2)


    def bn_conv_relu(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):

        layer=[]
        layer.append(nn.BatchNorm2d(in_ch))
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def forward(self,x,indices,x_shapes):
        i1,i2,i3,i4,i5 = indices
        x1,x2,x3,x4,x5 = x_shapes
                    
        x=self.upsample(x,i5,output_size=x4)
        x=self.layer_52_t(x)
        x=self.layer_51_t(x)
        x=self.layer_50_t(x)


        x=self.upsample(x,i4,output_size=x3)
        x=self.layer_42_t(x)
        x=self.layer_41_t(x)
        x=self.layer_40_t(x)

        x=self.upsample(x,i3,output_size=x2)
        x=self.layer_32_t(x)
        x=self.layer_31_t(x)
        x=self.layer_30_t(x)

        x=self.upsample(x,i2,output_size=x1)
        x=self.layer_21_t(x)
        x=self.layer_20_t(x)
        
        x=self.upsample(x,i1)
        x=self.layer_11_t(x)
        x_seg=self.layer_10_t(x)
        if self.noisy == True:
            x_denoise=self.layer_10_denoising(x)
            predictions = (x_seg,x_denoise)
        else: predictions = x_seg

        return predictions

segnet = SegNet()
total_param = sum(p.numel() for p in segnet.parameters() if p.requires_grad)
print(segnet) 
print("Total number of parameters: ",total_param)

        



   







        


