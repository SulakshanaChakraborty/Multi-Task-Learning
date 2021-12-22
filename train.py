import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time

from network import Encoder, Segnet
from dataloader import H5ImageLoader

def iou_pytorch(outputs,labels):
  
    outputs = outputs.squeeze(1)  
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    eps=1e-4
    

    iou = (intersection + eps) / (union + eps)  # eps for numerical stability
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean().item()  # Or thresholded.mean() if you are interested in average across the batch


device = 'cuda'
print(device," running ")

model=Segnet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

cri_seg = torch.nn.CrossEntropyLoss()
cri_class=torch.nn.CrossEntropyLoss()



transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64,64))])

DATA_PATH='data/train'
t=H5ImageLoader(DATA_PATH+'/images.h5',DATA_PATH+'/masks.h5',DATA_PATH+'/bboxes.h5',DATA_PATH+'/binary.h5',transform=transform) #All data paths 
trainloader = DataLoader(t, batch_size=64, shuffle=True)

alpha=0.9 #Hyperparameter for loss


for epoch in range(10):  
        
        time_epoch=time.time()
        train_loss=[]
        train_accuracy=[]

        train_iou=[]

        for i, data in enumerate(trainloader, 0):
          
            inputs, labels = data

            inputs=inputs.to(device)
            mask=torch.squeeze(labels['mask'].to(device))
            mask=mask.to(torch.long)
            binary=torch.squeeze(labels['classification'].to(device))
            binary=binary.to(torch.long)
            bbox=labels['bbox'].to(device)

            optimizer.zero_grad()         
            classes,boxes,segmask= model(inputs)

            loss_seg = cri_seg(segmask,mask)
            print(loss_seg,"segmentation")
            loss_class=cri_class(classes,binary)
            print(loss_class,"class")
            loss = loss_seg + alpha*loss_class

            

            target_segmentation = torch.argmax(segmask, 1)

            print(iou_pytorch(target_segmentation,mask),"iou")

            train_iou.append(iou_pytorch(target_segmentation,mask))

          #  acc_l =  (torch.argmax(classes,dim = 1) == binary).type(torch.uint8)
           # train_accuracy.append(torch.sum(acc_l)/len(labels))

            pred_ax=np.argmax(classes.detach().cpu().numpy(),axis=1)
            train_accuracy.append(np.sum((binary.detach().cpu().numpy()==pred_ax).astype(int))/len(binary))    
            train_loss.append( loss.item())     
            
            loss.backward()
            optimizer.step()

        time_epoch_vl=time.time() 
        print('----------------------------------------------------------------------------------')
        print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
        print("-----------------------Training Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(train_loss),3),"Train Accu: ",round(np.mean(train_accuracy),3))
        print("IOU: ",round(np.mean(train_iou),3))


            
               
        

