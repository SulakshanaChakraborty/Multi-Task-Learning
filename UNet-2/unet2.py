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

# class Block(nn.Module):
#     def __init__(self, ch_input, ch_output):
#         super().__init__()
#         self.conv1 = nn.Conv2d(ch_input, ch_output, 3)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(ch_output, ch_output, 3)
    
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.conv1(x))))

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


# class Encoder(nn.Module):
#     def __init__(self, chs=(3,64,128,256,512,1024)):
#     #  def __init__(self, chs=(3,64,17,34,68,136)):

#         super().__init__()
#         self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
#         self.pool       = nn.MaxPool2d(2)
    
#     def forward(self, x):
#         ftrs = []
#         for block in self.enc_blocks:
#             x = block(x)
#             ftrs.append(x)
#             x = self.pool(x)
#         return ftrs

class Encoder(nn.Module):
    def __init__(self, channels=(3,64,128,256,512,1024)):
    #  def __init__(self, chs=(3,64,17,34,68,136)):
        super().__init__()
        self.encoder_blocks = nn.Sequential()

        for ch_idx in range(len(channels)-1):
            self.encoder_blocks.add_module('EncoderBlock',Block(channels[ch_idx],channels[ch_idx+1]))
        
        self.encoder_blocks.add_module('maxpool1',nn.MaxPool2d(kernel_size=2))

    def forward(self,input):
        filters = []

        for blk in self.encoder_blocks:
            output = blk(input)
            filters.append(output)
            output = self.maxpool1(output)
            return filters



# class Decoder(nn.Module):
#     def __init__(self, chs=(1024, 512, 256, 128, 64)):
#         super().__init__()
#         self.chs         = chs
#         self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
#         self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
#     def forward(self, x, encoder_features):
#         for i in range(len(self.chs)-1):
#             x        = self.upconvs[i](x)
#             enc_ftrs = self.crop(encoder_features[i], x)
#             x        = torch.cat([x, enc_ftrs], dim=1)
#             x        = self.dec_blocks[i](x)
#         return x
    
#     def crop(self, enc_ftrs, x):
#         _, _, H, W = x.shape
#         enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#         return enc_ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs




class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=2, retain_dim=True, out_sz=(256,256)): 
    # def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(68,68)):

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
        return out

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