import sys
import torch
import pt_networks
import torch.optim as optim
import losses
import pt_networks.segnet
import pt_networks.SegNet_Attnt
import torchvision.models as models


def get_model(model_type,device):

    
    if model_type == 'baseline':
        model = pt_networks.segnet.Segnet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        #model.load_state_dict(torch.load('Segnet3task3layer.pt'))
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.BaselineLoss(False, True,False)

    elif model_type == 'mlt_attention':
        model = pt_networks.SegNet_Attnt.SegNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = losses.BaselineLoss(True, True, True)  # todo: update
    elif model_type == 'mlt_hard':

        model, optimizer, loss_fn = 1, 2, 3  # todo: update
    elif model_type == 'mlt_gscnn':
        model, optimizer, loss_fn = pt_networks.GSCNN(), 2, 3  # todo: update
    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model_path):
    model = 1  # todo: update
    return model


def save_model(model, model_path):
    # todo: save model
    pass
