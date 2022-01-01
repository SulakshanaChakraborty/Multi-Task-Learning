import sys
import torch
import numpy as np
import time
from metrics import eval_metrics, jaccard_index
from sklearn.metrics import jaccard_score,f1_score
import pt_networks.segnet
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

log_name='SegnetAttention3taskPretrainedVGGPooled/'
date='tuning_learning_rate_1e-03'
writer = SummaryWriter('logs/{}{}'.format(log_name,date))

def train_model(model_type, train_loader, validation_loader, model, optimizer, loss_criterion, epochs, device,soft_adapt = False):
    
    best_val_accuracy=0
    best_val_iou=0

    k=4

    train_loss =np.zeros((epochs,))
  
    train_segmentation_loss =np.zeros((epochs,))

    train_bbox_loss =np.zeros((epochs,))

    def soft_adapt(epoch,train_segmentation_loss,train_bbox_loss,train_label_loss,beta = -0.01,normalized = True):
        idx = epoch
        sk = np.zeros((3,2))
        norm_sk = np.zeros(3) # normalized rate of change 
        # rate of change 
        sk[0][0]= train_label_loss[idx] - train_label_loss[idx -1]
        sk[1][0] = train_segmentation_loss[idx] - train_segmentation_loss[idx -1]
        sk[2][0] = train_bbox_loss[idx] - train_bbox_loss[idx -1]
        # current task losses
        sk[0][1] = train_label_loss[idx] 
        sk[1][1] = train_segmentation_loss[idx]
        sk[2][1] = train_bbox_loss[idx]
        alpha = np.zeros(3)
        e = 1e-8
        print("sk rate:",sk[:,0])
        print("sk current loss:",sk[:,1])
        if normalized:
            for i in range(3):
                norm_sk[i] = sk[i][0]/(np.sum(np.abs(sk[:,0]))+e)
            x = norm_sk
        else:
            x = sk[:,0]
        print("normalized loss:",norm_sk)

        for i in range(3):
            alpha[i] = np.exp(beta*(x[i] - np.max(x) ))/(np.sum(np.exp(beta*(x - np.max(x) ))) + e)
        print("alpha 0:",alpha)
        # weighted loss
        for i in range(3):
            alpha[i] = sk[i][1] * alpha[i] / (np.sum( sk[i][1] * alpha) + e)

        print("alpha final:",alpha)

        return alpha

    alpha = torch.ones(3)

    for epoch in range(epochs):

        time_epoch = time.time()
        train_loss = []
        train_accuracy = []
        train_iou=[]
        train_jaca=[]
        train_f1_arr=[]

        train_bbox_loss=[]
        train_segmentation_loss=[]
        train_label_loss=[]

        val_loss=[]
        val_accuracy=[]
        val_iou=[]
        val_bbox_loss=[]
        val_segmentation_loss=[]
        val_label_loss=[]
        val_jaca=[]
        val_f1_arr=[]
       
        
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
            loss,labels_loss,segmentation_loss,bboxes_loss= loss_criterion(input_labels=classes, input_segmentations=segmask, \
                input_bboxes=boxes, target_labels=binary, target_segmentations=mask,
                target_bboxes=bbox)
            
        
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
            print(train_iou[i-1],"iou")

            mask_array=np.array(mask.cpu()).ravel()
            predicted_array=target_segmentation.cpu().ravel()
            print(jaccard_score(mask_array,predicted_array,average='weighted'),'skjac')
            print(f1_score(mask_array,predicted_array),"skf1")

            train_jac=jaccard_score(mask_array,predicted_array,average='weighted')
            train_f1=f1_score(mask_array,predicted_array)

            train_jaca.append(train_jac)
            train_f1_arr.append(train_f1)




          #  loss = alpha[0]*segmentation_loss + alpha[1]*bboxes_loss*0.0001 + alpha[2]*labels_loss
            train_loss.append(loss.item())
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
            print(round(iou.item(),3),"iou")

            val_jac=jaccard_score(mask_array,predicted_array,average='weighted')
            val_f1=f1_score(mask_array,predicted_array)

            val_jaca.append(val_jac)
            val_f1_arr.append(val_f1)
            

            
            val_iou.append(iou.item())
            val_bbox_loss.append(bboxes_loss.data.item())  
            

        time_epoch_vl=time.time()       

      
        

        # epoch_segmentation_loss.append(np.mean(train_segmentation_loss))
        # epoch_bbox_loss.append(0.0001*np.mean(train_bbox_loss))
        # epoch_label_loss.append(np.mean(train_label_loss))
        #dynamicly updating weights of loss
        # if epoch >0:
        #     alpha = soft_adapt(epoch,epoch_segmentation_loss,epoch_bbox_loss,epoch_label_loss)
        #     print("alpha:",alpha)


        print('----------------------------------------------------------------------------------')
        print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
        print("-----------------------Training Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(train_loss),3),"Train Accu: ",round(np.mean(train_accuracy),3))
        print("IOU: ",round(np.mean(train_iou),3))
        print("BBOX-loss: ",round(np.mean(train_bbox_loss),3))
        print("Segmnetaiton-loss",round(np.mean(train_segmentation_loss),3))
        print("Label-loss",round(np.mean(train_label_loss),3))
        print("Jac",round(np.mean(train_jaca),3))
        print("F1s",round(np.mean(train_f1_arr),3))

        print("-----------------------Validation Metrics-------------------------------------------")
        print("Loss: ",round(np.mean(val_loss),3),"Val Accu: ",round(np.mean(val_accuracy),3))
        print("IOU: ",round(np.mean(val_iou),3))
        print("BBOX-loss: ",round(np.mean(val_bbox_loss),3))
        print("Segmnetaiton-loss",round(np.mean(val_segmentation_loss),3))
        print("Label-loss",round(np.mean(val_label_loss),3))
        print("Jac",round(np.mean(val_jaca),3))
        print("F1s",round(np.mean(val_f1_arr),3))

        writer.add_scalar('Train-Epoch-Accuracy',round(np.mean(train_accuracy),3), epoch)
        writer.add_scalar('Train-Epoch-IOU',round(np.mean(train_iou),3), epoch)
        writer.add_scalar('Train-Epoch-BBOX',round(np.mean(train_bbox_loss),3), epoch)
        writer.add_scalar('Train-Epoch-Seg-loss',round(np.mean(train_segmentation_loss),3), epoch)
        writer.add_scalar('Train-Epoch-label-loss',round(np.mean(train_label_loss),3), epoch)
        writer.add_scalar('Train-Epoch-JAC',round(np.mean(train_jaca),3), epoch)
        writer.add_scalar('Train-Epoch-f1',round(np.mean(train_f1_arr),3), epoch)

        writer.add_scalar('Val-Epoch-Accuracy',round(np.mean(val_accuracy),3), epoch)
        writer.add_scalar('Val-Epoch-IOU',round(np.mean(val_iou),3), epoch)
        writer.add_scalar('Val-Epoch-BBOX',round(np.mean(val_bbox_loss),3), epoch)
        writer.add_scalar('Val-Epoch-Seg-loss',round(np.mean(val_segmentation_loss),3), epoch)
        writer.add_scalar('Val-Epoch-label-loss',round(np.mean(val_label_loss),3), epoch)
        writer.add_scalar('val-Epoch-JAC',round(np.mean(val_jaca),3), epoch)
        writer.add_scalar('va,-Epoch-f1',round(np.mean(val_f1_arr),3), epoch)
   
    
        # if round(np.mean(val_iou),3) > best_val_iou: #and round(np.mean(val_accuracy),3) > best_val_accuracy:

        #  best_val_iou=round(np.mean(val_iou),3)
     #    best_val_accuracy=round(np.mean(val_accuracy),3)
        torch.save(model.state_dict(), 'Segnet3taskPretrainedFixedmetric20020007.pt')
      
