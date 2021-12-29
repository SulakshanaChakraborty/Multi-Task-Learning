import sys
import torch
import numpy as np
import time
from metrics import eval_metrics
import pt_networks.segnet
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
model_name = 'Segnet-Baseline-loss-3task'
log_name='segnetbaseafter30/'
date=datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
writer = SummaryWriter('logs/{}{}'.format(log_name,date))

def train_model(model_type, train_loader, validation_loader, model, optimizer, loss_criterion, epochs, device):
    
    best_val_accuracy=0
    best_val_iou=0

    k=4

    train_bbox_loss=np.zeros((epochs,))
    train_segmentation_loss=np.zeros((epochs,))
    train_label_loss=np.zeros((epochs,))


    for epoch in range(epochs):

        time_epoch = time.time()
        train_loss = []
        train_accuracy = []
        train_iou=[]

        train_bbox_loss=[]
        train_segmentation_loss=[]
        train_label_loss=[]

        val_loss=[]
        val_accuracy=[]
        val_iou=[]
        val_bbox_loss=[]
        val_segmentation_loss=[]
        val_label_loss=[]
        
        for i, batch_data in enumerate(train_loader, 1):
     
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)
            bbox=bbox.float()

            optimizer.zero_grad()
            classes, boxes, segmask = model(inputs)

        
            loss,labels_loss,segmentation_loss,bboxes_loss=loss_criterion(input_labels=classes, input_segmentations=segmask, \
                input_bboxes=boxes, target_labels=binary, target_segmentations=mask,
                target_bboxes=bbox)

            print("loss",loss.dtype)
            
        
            pred_ax=np.argmax(classes.detach().cpu().numpy(),axis=1)
            train_accuracy.append(np.sum((binary.detach().cpu().numpy()==pred_ax).astype(int))/len(binary))    
            train_loss.append(loss.item())

            print(train_accuracy[i-1],"minibatch acc")

            train_label_loss.append(labels_loss.data.item())
            train_segmentation_loss.append(segmentation_loss.data.item())


            train_bbox_loss.append(bboxes_loss.data.item())
            target_segmentation = torch.argmax(segmask, 1)
            iou=(eval_metrics(mask.cpu(),target_segmentation.cpu(),2))
            train_iou.append(iou.item())

            loss.backward()
            optimizer.step()
        for i, batch_data in enumerate(validation_loader, 1):

         with torch.no_grad():
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)
            bbox=bbox.float()            
            classes, boxes, segmask = model(inputs)

            loss,labels_loss,segmentation_loss,bboxes_loss=loss_criterion(input_labels=classes, input_segmentations=segmask, \
                input_bboxes=boxes, target_labels=binary, target_segmentations=mask,
                target_bboxes=bbox)
        

            pred_ax=np.argmax(classes.detach().cpu().numpy(),axis=1)
            val_accuracy.append(np.sum((binary.detach().cpu().numpy()==pred_ax).astype(int))/len(binary))    
            val_loss.append(loss.item())  

            val_label_loss.append(labels_loss.data.item())
            val_segmentation_loss.append(segmentation_loss.data.item()) 


            target_segmentation = torch.argmax(segmask, 1)

            iou=(eval_metrics(mask.cpu(),target_segmentation.cpu(),2))
           # print(round(iou.item(),3),"iou")
            val_iou.append(iou.item())
            val_bbox_loss.append(bboxes_loss.data.item())  
            

        time_epoch_vl=time.time() 
        print('----------------------------------------------------------------------------------')
        print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
        print("-----------------------Training Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(train_loss),3),"Train Accu: ",round(np.mean(train_accuracy),3))
        print("IOU: ",round(np.mean(train_iou),3))
        print("BBOX-loss: ",round(np.mean(train_bbox_loss),3))
        print("Segmnetaiton-loss",round(np.mean(train_segmentation_loss),3))
        print("Label-loss",round(np.mean(train_label_loss),3))

        print("-----------------------Validation Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(val_loss),3),"Val Accu: ",round(np.mean(val_accuracy),3))
        print("IOU: ",round(np.mean(val_iou),3))
        print("BBOX-loss: ",round(np.mean(val_bbox_loss),3))
        print("Segmnetaiton-loss",round(np.mean(val_segmentation_loss),3))
        print("Label-loss",round(np.mean(val_label_loss),3))

        writer.add_scalar('Train-Epoch-Accuracy',round(np.mean(train_accuracy),3), epoch)
        writer.add_scalar('Train-Epoch-IOU',round(np.mean(train_iou),3), epoch)
        writer.add_scalar('Train-Epoch-BBOX',round(np.mean(train_bbox_loss),3), epoch)
        writer.add_scalar('Train-Epoch-Seg-loss',round(np.mean(train_segmentation_loss),3), epoch)
        writer.add_scalar('Train-Epoch-label-loss',round(np.mean(train_label_loss),3), epoch)


        writer.add_scalar('Val-Epoch-Accuracy',round(np.mean(val_accuracy),3), epoch)
        writer.add_scalar('Val-Epoch-IOU',round(np.mean(val_iou),3), epoch)
        writer.add_scalar('Val-Epoch-BBOX',round(np.mean(val_bbox_loss),3), epoch)
        writer.add_scalar('Val-Epoch-Seg-loss',round(np.mean(val_segmentation_loss),3), epoch)
        writer.add_scalar('Val-Epoch-label-loss',round(np.mean(val_label_loss),3), epoch)
    
        # if round(np.mean(val_iou),3) > best_val_iou and round(np.mean(val_accuracy),3) > best_val_accuracy:

        #  best_val_iou=round(np.mean(val_iou),3)
        #  best_val_accuracy=round(np.mean(val_accuracy),3)
        #  torch.save(model.state_dict(), 'MTL-base-Segnet-classfication-1.pt')
      
