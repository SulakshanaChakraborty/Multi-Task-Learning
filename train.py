
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from network import Encoder, Segnet
from dataloader import H5ImageLoader



model=Segnet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATA_PATH='data/train'
t=H5ImageLoader(DATA_PATH+'/images.h5',DATA_PATH+'/masks.h5',DATA_PATH+'/bboxes.h5',DATA_PATH+'/binary.h5',transform=transform) #All data paths 
trainloader = DataLoader(t, batch_size=16, shuffle=True)



for epoch in range(1):  

        loss_ar=[]
        accuracy=[]

        for i, data in enumerate(trainloader, 0):
          
            inputs, labels = data

            inputs=inputs
            labels=labels
                
            outputs = model(inputs)
               
        

