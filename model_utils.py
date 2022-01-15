import sys
import torch
import torch.optim as optim
import losses
import pt_networks.segnet
import pt_networks.SegNet_Attnt
import torchvision.models as models
import pt_networks.SegNet_attnt_canny
import pt_networks.SegNet_attnt_color
import pt_networks.Segnet_attnt_denoising
import pt_networks.segnet_color
import pt_networks.SegNet_attnt_canny
import pt_networks.SegNet_canny


def get_model(model_type, device='cpu'):
   
    """A function used to initialise and define the model that will be used for training. 
    Depending on the selected model an appropriate loss function is assigned to the model type used.
    Adam optimiser is utilised for each of the models. 

    Args:
        model_type (str): Name of the model defined in the cw2_main.py.
        device (str, optional): The device that should be used to train the chosen model. Defaults to 'cpu'.

    Returns:
        model: The network after initialisation.
        optimizer: The pytorch optimiser (Adam).
        loss_fn: The loss function for the respective model/network.
    """
    if model_type == 'Segnet-1task-untrained':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        loss_fn = losses.BaselineLoss(flag_labels = False, flag_segmentations= True, flag_bboxes = False,device=device)

    elif model_type == 'Segnet-1task':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16) 
        loss_fn = losses.BaselineLoss(flag_labels = False, flag_segmentations= True, flag_bboxes = False,device=device)
    
    elif model_type == 'MTL-Segnet-untrained':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4) 

        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True,device=device)
    
    elif model_type == 'MTL-Segnet':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True,device=device)
    
    elif model_type == 'MTL-Attention':
        model = pt_networks.SegNet_Attnt.SegNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True,device=device)


    elif model_type == 'MTL-Attention-with-colorization':
        model = pt_networks.SegNet_attnt_color.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.ColorLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_color=True,device=device)
    elif model_type == 'MTL-Attention-with-denoising':
        model = pt_networks.Segnet_attnt_denoising.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.DenoisingLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_denoise=True,device=device)

    elif model_type == 'MTL-Attention-with-canny':
        model = pt_networks.SegNet_attnt_canny.SegNetFilters().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True, flag_filters=True,device=device)

    elif model_type == 'MTL-Attention-without-bbox':
        model = pt_networks.SegNet_Attnt.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = False,device=device)

    elif model_type == 'MTL-Attention-without-classification':
        model = pt_networks.SegNet_Attnt.SegNet().to(device)
       
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.BaselineLoss(flag_labels=False, flag_segmentations=True, flag_bboxes=True,device=device)

    elif model_type == 'MTL-segnet-with-canny':
        model = pt_networks.SegNet_canny.SegnetOpencv().to(device)
    
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_filters=True,device=device)
                                          
    elif model_type == 'MTL-segnet-with-colorization':
        model = pt_networks.segnet_color.Segnet().to(device)
    
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.ColorLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_color=True,device=device)

    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model

