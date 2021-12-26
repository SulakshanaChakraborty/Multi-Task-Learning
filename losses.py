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
        self.weights = torch.ones((2,), requires_grad=True).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = torch.nn.MSELoss()  # todo: update loss

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

        # # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes).float()
        else:
            bboxes_loss = 0

       # loss = torch.cat([labels_loss, segmentations_loss, bboxes_loss])
        loss =  0.9*labels_loss+1*segmentations_loss+0.001*bboxes_loss.float()

        print(f"labels_loss: {labels_loss:.8f}, segmentations_loss: {segmentations_loss:.8f}, bboxes_loss: {bboxes_loss:.8f}")
        #print(self.weights,"weights")

        return loss.float()
