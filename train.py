import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time

from network import Encoder, Segnet
from dataloader import H5ImageLoader


device = 'cpu'
print(device," running ")

model=Segnet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

cri_seg = torch.nn.CrossEntropyLoss()
cri_class=torch.nn.CrossEntropyLoss()



transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATA_PATH='data/train'
t=H5ImageLoader(DATA_PATH+'/images.h5',DATA_PATH+'/masks.h5',DATA_PATH+'/bboxes.h5',DATA_PATH+'/binary.h5',transform=transform) #All data paths 
trainloader = DataLoader(t, batch_size=4, shuffle=True)



for epoch in range(10):  
        
        time_epoch=time.time()
        train_loss=[]
        train_accuracy=[]

        for i, data in enumerate(trainloader, 0):
          
            inputs, labels = data

            inputs=inputs.to(device)
            mask=torch.squeeze(labels['mask'].to(device))
            mask=mask.to(torch.long)
            binary=torch.squeeze(labels['classification'].to(device))
            binary=binary.to(torch.long)
            bbox=labels['bbox'].to(device)

            print(binary.size(),"class shape")
            
       
            optimizer.zero_grad()         
            classes,boxes,segmask= model(inputs)

            print(classes.size(),"class shape")

            loss_seg = cri_seg(segmask,mask)
            loss_class=cri_class(classes,classes)
            loss = loss_seg + loss_class

            #pred_ax=np.argmax(classes.detach().numpy(),axis=1)
            #train_accuracy.append(np.sum((classes.detach().numpy()==pred_ax).astype(int))/len(binary))    
            train_loss.append( loss.item())     
            
            loss.backward()
            optimizer.step()

        time_epoch_vl=time.time() 
        print('----------------------------------------------------------------------------------')
        print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
        print("-----------------------Training Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(train_loss),3))

            
               
        

