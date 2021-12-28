import torch
import torch.nn as nn


class BaselineLoss(nn.Module):
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True):
        super(BaselineLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes

        ######################
        # Define weights
        ######################
        device='cpu'
        self.weights = torch.tensor([0.5,1],requires_grad=True).to(device)

     

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss() # todo: update loss

    def forward(self, input_labels, input_segmentations, input_bboxes, target_labels, target_segmentations,
                target_bboxes):

        # Loss for labels.
        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = 0

       # Loss for segmentations.
        if self.flag_segmentations:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = 0

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = 0

    #    loss = torch.cat([labels_loss, segmentations_loss, bboxes_loss])
    #    loss = torch.stack([labels_loss, segmentations_loss])

        loss = labels_loss + segmentations_loss + 0.001*bboxes_loss



        # print(loss,"total loss")

        return loss,labels_loss,segmentations_loss,bboxes_loss

class SoftAdaptLoss(nn.Module):
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True):
        super().__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes
        

        ######################
        # Define weights
        ######################
        device='cuda'
        
        self.grad= torch.zeros((3,)).to(device)
        self.history=torch.zeros((3,2)).to(device)
        self.counter=1
        self.n=torch.zeros((3,)).to(device)

        

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss() # todo: update loss

    def forward(self, input_labels, input_segmentations, input_bboxes, target_labels, target_segmentations,
                target_bboxes):

        # Loss for labels.
        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = 0

        # Loss for segmentations.
        if self.flag_labels:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = 0

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = 0

    
        if self.counter%2==0:
            k=1
        else:
            k=0

        
        self.history[0][k]=labels_loss.data.item()
        self.history[2][k]=bboxes_loss.data.item()
        self.history[1][k]=segmentations_loss.data.item()

        self.counter+=1

        if self.counter>2:

            self.n[0]= self.history[0][1]-self.history[0][0]
            self.n[1]= self.history[1][1]-self.history[1][0]
            self.n[2]= self.history[2][1]-self.history[2][0]

        
        beta=0.01
        
        a = labels_loss.data.item() * torch.exp(beta * (self.n[0] - torch.max(self.n)))
        b = segmentations_loss.data.item() * torch.exp(beta * (self.n[1] - torch.max(self.n)))
        c = 0.001*bboxes_loss.data.item() * torch.exp(beta * (self.n[2] - torch.max(self.n)))
        denom = a+b+c
        
        eps=1e-8
        self.grad[0]= a/(a+b+c+eps)
        self.grad[1]= b/(a+b+c+eps)
        self.grad[2]= c/(a+b+c+eps)

        print(self.grad,"weights")
        

        loss= self.grad[0]*labels_loss +  self.grad[1]* segmentations_loss + self.grad[2]*bboxes_loss*0.001


        return loss,self.grad[0]*labels_loss,self.grad[1]* segmentations_loss ,self.grad[2]*bboxes_loss
