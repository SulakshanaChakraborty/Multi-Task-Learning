
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer_0 = self.conv2d_layer(3,64) #CHANGE 3 to in channels
        self.layer_1=self.conv2d_layer(64,64)
        self.layer_2=self.conv2d_layer(64,128)
       # self.layer_3=self.conv2d_layer(128,128)
        #self.layer_4=self.conv2d_layer(128,256)
        #self.layer_5=self.conv2d_layer(256,256)


    def conv2d_layer(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):

        layer=[]
        layer.append(nn.BatchNorm2d(in_ch))
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def forward(self,x):

      #  print(x.shape)
        x=self.layer_0(x)
      #  print(x.shape)
        x=self.layer_1(x)
      #  print(x.shape)
        x=self.layer_2(x)
        #print(x.shape)
     #   x=self.layer_3(x)
        #print(x.shape)
     #   x=self.layer_4(x)
        #print(x.shape)
      #  x=self.layer_5(x)

        return x

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

      #  self.layer_0 = self.conv2d_layer_T(256,256)
       # self.layer_1=self.conv2d_layer_T(256,128)
        #self.layer_2=self.conv2d_layer_T(128,128)
        self.layer_3=self.conv2d_layer_T(128,64)
        self.layer_4=self.conv2d_layer_T(64,64)
        self.layer_5=self.conv2d_layer_T(64,2) #CHANGE 2 to outchanels


    def conv2d_layer_T(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):

        layer=[]
        layer.append(nn.BatchNorm2d(in_ch))
        layer.append(nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def forward(self,x):

      #  print(x.shape)
      #  x=self.layer_0(x)
       # print(x.shape)
       # x=self.layer_1(x)
        #print(x.shape)
       # x=self.layer_2(x)
        #print(x.shape)
        x=self.layer_3(x)
        #print(x.shape)
        x=self.layer_4(x)
        #print(x.shape)
        x=self.layer_5(x)
        #print(x.shape)

        return x

class Segnet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder= Encoder()
        self.flat=nn.Flatten()
        self.linear_c_0=nn.Linear(128*64*64,64)
        self.linear_c_1=nn.Linear(64,2)

        self.linear_b_0=nn.Linear(128*64*64,64)
        self.linear_b_1=nn.Linear(64,4)
        self.decoder= Decoder()

 


    def forward(self,x):

        enc=self.encoder(x)
        #print(enc.size(),"encsize")
        flat=self.flat(enc)
       # print(flat.size(),"flatsize")

        c_0=F.relu(self.linear_c_0(flat))
        c_= (self.linear_c_1(c_0))
        b_0=F.relu(self.linear_b_0(flat))
        b_=F.relu(self.linear_b_1(b_0))
        dec=self.decoder(enc)

        return c_,b_,dec

class AttentionBlock(nn.Module):

    def __init__(self,in_channel,inter_channel,out_channel,features,in_channel_3,out_channel_3):

        """
        @params:
        features: Shared features (From the 3rd or so conv) to be multiplied element-wise
        in_channel: Number of in_channels for 1st 1x1 conv
        inter_channel: Intermediate channels for 2nd 1x1 conv
        out_channels: Out channels for 2nd 1x1 conv
        
        in_channel_3: In channels for 3x3 conv
        out_channel_3: Out channels for 3x3 conv


        
        """

        super().__init__()

        self.attention_weights= self.attention(in_channel,inter_channel,out_channel)
        self.features=features
        self.conv_layer=self.conv2d_layer(in_channel_3,out_channel_3)

    def conv2d_layer(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):

        layer=[]
        layer.append(nn.BatchNorm2d(in_ch))
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)


    def attention_(self,in_channel,inter_channel,out_channel):

        layer=[]
         
        layer.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0))
        layer.append(nn.BatchNorm2d(inter_channel))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=in_channel, out_channels=inter_channel, kernel_size=1, padding=0))
        layer.append(nn.BatchNorm2d(out_channel))
        layer.append(nn.Sigmoid())
        
        return nn.Sequential(*layer)

    def forward(self,x):

        x=self.attention_weights(x)
        x=x*self.features
        x=self.conv_layer(x)





    






        



   







        


