import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        filter = [64, 128, 256, 512, 512]

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))  

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        linear_class_layers = [nn.Linear(512*256*256,64),nn.Linear(64,2)]
        self.linear_class=nn.Sequential(*linear_class_layers)
        linear_bb_layers = [nn.Linear(512*256*256,64),nn.Linear(64,4)]
        self.linear_bb=nn.Sequential(*linear_bb_layers)
        self.decoder = Decoder()


    def foward(self,x):
        g_encoder, g_maxpool,indices = ([0] * 5 for _ in range(3))
        for i in range(5):
            g_encoder[i] = [0] * 2 

        # define attention list for tasks
        atten_encoder = [0] * 3
        for i in range(2):
            atten_encoder[i] = [0] * 5 
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j] = [0] * 3
        
        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
        
        target_pred = self.decoder(atten_encoder[0][-1][-1],indices)
        aux_pred_c = self.linear_class(atten_encoder[1][-1][-1])
        aux_pred_bb = self.linear_bb(atten_encoder[2][-1][-1])
        

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.layer_0 = self.conv2d_layer_T(512,512)
        
        self.layer_1 = self.conv2d_layer_T(256,512)

        self.layer_2=self.conv2d_layer_T(256,128)
        self.layer_3=self.conv2d_layer_T(128,64)
        self.layer_4=self.conv2d_layer_T(64,64)
        self.layer_5=self.conv2d_layer_T(64,2) #CHANGE 2 to outchanels


    def conv2d_layer_T(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):
        conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_ch),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def forward(self,x):

      #  print(x.shape)
      #  x=self.layer_0(x)
       # print(x.shape)
       # x=self.layer_1(x)
        #print(x.shape)
        x=self.layer_2(x)
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
        self.linear_c_0=nn.Linear(128*256*256,64)
        self.linear_c_1=nn.Linear(64,2)

        self.linear_b_0=nn.Linear(128*256*256,64)
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





    






        



   







        


