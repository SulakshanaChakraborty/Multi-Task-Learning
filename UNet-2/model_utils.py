import sys
import torch.nn as nn
import pt_networks
import torch.optim as optim
import losses
import pt_networks.segnet
import pt_networks.SegNet_Attnt_reformat
import pt_networks.SegNet_Attnt
import pt_networks.GSCNN
import pt_networks.unet_reduced_layers

def get_model(model_type,device):

   
    if model_type == 'baseline':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # todo: update
        loss_fn = losses.BaselineLoss(True, True, False)
    elif model_type == 'mlt_attention':
        model = pt_networks.SegNet_Attnt.SegNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)
        loss_fn = losses.BaselineLoss(True, True, True)  # todo: update
    elif model_type == 'mlt_hard':

        model, optimizer, loss_fn = 1, 2, 3  # todo: update
    elif model_type == 'mlt_gscnn':
        model, optimizer, loss_fn = pt_networks.GSCNN(), 2, 3  # todo: update
    elif model_type == 'unet_reduced_layers':
        model = pt_networks.unet_reduced_layers.UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)
        #loss_fn = losses.BaselineLoss(True, True, True)  # todo: update
        loss_fn = nn.CrossEntropyLoss()
    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model_path):
    model = 1  # todo: update
    return model


def save_model(model, model_path):
    # todo: save model
    pass
