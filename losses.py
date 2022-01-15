import torch
import torch.nn as nn


class BaselineLoss(nn.Module):
    """Class for computation of the loss for the baseline network.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True):
        super(BaselineLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes

        ######################
        # Define weights
        ######################

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()  # todo: update loss

    def forward(self, input_labels, input_segmentations, input_bboxes, target_labels, target_segmentations,
                target_bboxes):

        # Loss for labels.
        device = 'cuda'
        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for segmentations.
        if self.flag_segmentations:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = torch.zeros(1, requires_grad=True).to(device)

        #    loss = torch.cat([labels_loss, segmentations_loss, bboxes_loss])
        #    loss = torch.stack([labels_loss, segmentations_loss])

        loss = labels_loss + 20 * segmentations_loss + 0.00007 * bboxes_loss * 2

        # print(loss,"total loss")

        return loss, labels_loss, segmentations_loss, bboxes_loss


class OpencvFilterLoss(nn.Module):
    """Class for the computation of the loss for the canny filter.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True, flag_filters=True):
        super(OpencvFilterLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes
        self.flag_filters = flag_filters

        ######################
        # Define weights
        ######################
        device = 'cpu'
        self.weights = torch.tensor([0.5, 1], requires_grad=True).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()  # todo: update loss
        self.filters_criterion = torch.nn.L1Loss()

    def forward(self, input_labels, input_segmentations, input_bboxes, input_filters, target_labels,
                target_segmentations,
                target_bboxes, target_filters):

        # Loss for labels.
        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = torch.zeros(1, requires_grad=True)

        # Loss for segmentations.
        if self.flag_segmentations:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = torch.zeros(1, requires_grad=True)

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = torch.zeros(1, requires_grad=True)

        # Loss for opencv filter
        if self.flag_filters:
            filters_loss = self.filters_criterion(input_filters, target_filters)
        else:
            filters_loss = torch.zeros(1, requires_grad=True)

        labels_weight = 0.1
        segmentation_weight = 0.7
        bboxes_weights = 0.1 * 0.0001
        filters_weight = 0.1

        loss = labels_weight * labels_loss + \
               segmentation_weight * segmentations_loss + \
               bboxes_weights * bboxes_loss + \
               filters_weight * filters_loss

        return loss, labels_loss, segmentations_loss, bboxes_loss, filters_loss


class ColorLoss(nn.Module):
    """Class for the computation of the loss for the colorisation auxillary task.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True, flag_color=True):
        super(ColorLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes
        self.flag_color = flag_color

        ######################
        # Define weights
        ######################
        device = 'cuda'
        self.weights = torch.tensor([0.5, 1], requires_grad=True).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()  # todo: update loss
        self.ab_criterion = nn.L1Loss()

    def forward(self, input_labels, input_segmentations, input_bboxes, input_img, target_img, target_labels,
                target_segmentations,
                target_bboxes):

        # Loss for labels.
        device = 'cuda'
        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for segmentations.
        if self.flag_labels:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = torch.zeros(1, requires_grad=True).to(device)

        if self.flag_color:

            ab_loss = self.ab_criterion(input_img, target_img)

        else:
            ab_loss = torch.zeros(1, requires_grad=True).to(device)

        loss = 1 * labels_loss + 20 * segmentations_loss + 0.00007 * bboxes_loss * 2 + ab_loss

        return loss, labels_loss, segmentations_loss, bboxes_loss, ab_loss


class SoftAdaptLoss(nn.Module):
    """Class for the calculation of loss to allow for updating weights epoch wise.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True):
        super().__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes

        ######################
        # Define weights
        ######################
        device = 'cuda'

        self.grad = torch.zeros((3,)).to(device)
        self.counter = 1
        self.n = torch.ones((3,)).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()  # todo: update loss

    def forward(self, input_labels, input_segmentations, input_bboxes, target_labels, target_segmentations,
                target_bboxes, epoch):

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

        if self.counter % 2 == 0:
            k = 1
        else:
            k = 0

        self.counter += 1

        if self.counter > 2:
            self.n[0] = self.history[0][1] - self.history[0][0]
            self.n[1] = self.history[1][1] - self.history[1][0]
            self.n[2] = self.history[2][1] - self.history[2][0]

        beta = 0.01

        a = labels_loss.data.item() * torch.exp(beta * (self.n[0] - torch.max(self.n)))
        b = segmentations_loss.data.item() * torch.exp(beta * (self.n[1] - torch.max(self.n)))
        c = 0.001 * bboxes_loss.data.item() * torch.exp(beta * (self.n[2] - torch.max(self.n)))
        denom = a + b + c

        eps = 1e-8
        self.grad[0] = a / (a + b + c + eps)
        self.grad[1] = b / (a + b + c + eps)
        self.grad[2] = c / (a + b + c + eps)

        print(self.grad, "weights")

        loss = self.grad[0] * labels_loss + self.grad[1] * segmentations_loss + self.grad[2] * bboxes_loss * 0.001

        return loss, self.grad[0] * labels_loss, self.grad[1] * segmentations_loss, self.grad[2] * bboxes_loss


class DenoisingLoss(nn.Module):
    """Class for the computing the loss of the denoising model.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True, flag_denoise=True):
        super(DenoisingLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes
        self.flag_denoise = flag_denoise

        ######################
        # Define weights
        ######################
        device = 'cpu'
        self.weights = torch.tensor([0.5, 1], requires_grad=True).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()
        self.denoising_criterion = nn.MSELoss()

    def forward(self, input_labels, input_segmentations, input_bboxes, input_denoise, target_labels,
                target_segmentations,
                target_bboxes, target_denoise):

        device = 'cuda'

        # Loss for labels.
        if self.flag_denoise:
            denoise_loss = self.denoising_criterion(input_denoise, target_denoise)
        else:
            denoise_loss = torch.zeros(1, requires_grad=True).to(device)

        if self.flag_labels:
            labels_loss = self.labels_criterion(input_labels, target_labels)
        else:
            labels_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for segmentations.
        if self.flag_segmentations:
            segmentations_loss = self.segmentations_criterion(input_segmentations, target_segmentations)
        else:
            segmentations_loss = torch.zeros(1, requires_grad=True).to(device)

        # Loss for bounding boxes.
        if self.flag_bboxes:
            bboxes_loss = self.bboxes_criterion(input_bboxes, target_bboxes)
        else:
            bboxes_loss = torch.zeros(1, requires_grad=True).to(device)

        #    loss = torch.cat([labels_loss, segmentations_loss, bboxes_loss])
        #    loss = torch.stack([labels_loss, segmentations_loss])

        loss = 1 * labels_loss + 20 * segmentations_loss + 2 * 0.00007 * bboxes_loss + 1 * denoise_loss
        # print(loss,"total loss")
        return loss, labels_loss, segmentations_loss, bboxes_loss, denoise_loss


class GeometricLoss(nn.Module):
    """Class used for calculation of geometric loss.
    """
    def __init__(self, flag_labels=True, flag_segmentations=True, flag_bboxes=True, device='cpu'):
        super(GeometricLoss, self).__init__()
        self.flag_labels = flag_labels
        self.flag_segmentations = flag_segmentations
        self.flag_bboxes = flag_bboxes

        ######################
        # Define weights
        ######################
        # self.weights = torch.tensor([1, 1, 0], requires_grad=True).to(device)
        self.weights = torch.tensor([1, 1, 0.001]).to(device)

        ######################
        # Defines losses
        ######################
        # Labels loss
        self.labels_criterion = torch.nn.CrossEntropyLoss()
        self.segmentations_criterion = torch.nn.CrossEntropyLoss()
        self.bboxes_criterion = nn.MSELoss()  # todo: update loss

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

        # Compute total loss.
        # loss = torch.stack([labels_loss, segmentations_loss, bboxes_loss])
        # loss = torch.matmul(loss, self.weights)

        # Compute total loss.
        multiplication = self.weights[0] * labels_loss * self.weights[1] * segmentations_loss * self.weights[
            2] * bboxes_loss
        n_elements = 3
        loss = torch.pow(multiplication, 1 / n_elements)

        return loss, labels_loss, segmentations_loss, bboxes_loss
