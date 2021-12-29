import torch
import torch.nn as nn
from load_data import create_data_loaders
import time
import torchvision
import torch.nn.functional as F
import numpy as np 
from collections import OrderedDict

#https://amaarora.github.io/2020/09/13/unet.html
#https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

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
    def __init__(self, enc_chans=(3,64,128,256,512,1024), dec_chans=(1024, 512, 256, 128, 64), num_class=2, retain_dim=True, output_size=(256,256)): 
    # def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(68,68)):

        super().__init__()
        # self.unet = nn.Sequential(OrderedDict([
        #     ('Encoder', Encoder(enc_chans)),
        #     ('Decoder', Decoder(dec_chans)),
        #     ('Convolution1', nn.Conv2d(dec_chans[-1], num_class, 1))
        # ]))

        # self.unet.add_module('retain_dimension',retain_dim)
        # self.unet.add_module('output_size', output_size)

        self.encoder     = Encoder(enc_chans)
        self.decoder     = Decoder(dec_chans)
        self.head        = nn.Conv2d(dec_chans[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = output_size

    def forward(self, x):
        enc_chans = self.encoder(x)
        out      = self.decoder(enc_chans[::-1][0], enc_chans[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
    
    # def forward(self, input):
    #     input = self.unet(input)
    #     return unet 

        # def forward(self, x):
        #     enc_ftrs = self.encoder(x)
        #     out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        #     out      = self.head(out)
        #     if self.retain_dim:
        #         out = F.interpolate(out, self.out_sz)
        #     return out

# initialize our UNet model
unet = UNet()
# initialize loss function and optimizer
# loss = nn.BCEWithLogitsLoss()
criteria  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

for epoch in range(1):

    time_epoch = time.time()
    train_loss = []
    train_accuracy = []
    for i, batch_data in enumerate(train_loader, 1):
        inputs, labels = batch_data

        mask = torch.squeeze(labels['mask'])
        mask = mask.to(torch.long)
        binary = torch.squeeze(labels['classification'])
        binary = binary.to(torch.long)
        bbox = labels['bbox']

        print(binary.size(), "binary shape")

        optimizer.zero_grad()
        outputs= unet(inputs)
        classes = outputs[0]
        # classes = outputs.index_select(0)

        print(outputs.size(), "outputs shape")
        print(classes.size(), "class shape") 
        

        # todo: update the loss_criterion in the loss computation.
        print(mask.size(),'mask size')
        loss_seg = criteria(outputs, mask)
        # loss_class = cri_class(classes, classes)
        # loss = loss_seg + loss_class
        loss = loss_seg 
        print(loss)

        # todo: make the weight of losses a hyper-parameter

        # pred_ax=np.argmax(classes.detach().numpy(),axis=1)
        # train_accuracy.append(np.sum((classes.detach().numpy()==pred_ax).astype(int))/len(binary))
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        # todo: include validation loader in training loop (metrics)

    time_epoch_vl = time.time()
    print('----------------------------------------------------------------------------------')
    print(f"Epoch: {epoch + 1} Time taken : {round(time_epoch_vl - time_epoch, 3)} seconds")
    print("-----------------------Training Metrics-------------------------------------------")
    print("Loss: ", round(np.mean(train_loss), 3))