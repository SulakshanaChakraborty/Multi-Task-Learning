import sys
import torch
import torch.optim as optim
import losses
import pt_networks.segnet
import pt_networks.SegNet_attnt
import torchvision.models as models
import pt_networks.SegNet_attnt_canny
import pt_networks.SegNet_attnt_color
import pt_networks.SegNet_attnt_denoising
import pt_networks.segnet_color
import pt_networks.SegNet_attnt_canny
import pt_networks.SegNet_canny


def get_model(model_type, device='cpu', load_pre_trained_weights=False):
    if model_type == 'Segnet-1task-untrained':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        loss_fn = losses.BaselineLoss(flag_labels = False, flag_segmentations= True, flag_bboxes = False)

    if model_type == 'Segnet-1task':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6) 
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16) 
        loss_fn = losses.BaselineLoss(flag_labels = False, flag_segmentations= True, flag_bboxes = False)
    
    elif model_type == 'MTL-Segnet-untrained':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4) 

        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True)
    
    elif model_type == 'MTL-Segnet':
        model = pt_networks.segnet.Segnet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True)
    
    elif model_type == 'MTL-Attention':
        model = pt_networks.SegNet_attnt.SegNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = True)


    elif model_type == 'MLT-Attention-with-colourization':
        model = pt_networks.SegNet_attnt_color.SegNet.to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.ColorLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_color=True)
    elif model_type == 'MTL-Attention-with-denoising':
        model = pt_networks.SegNet_attnt_denoising.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.DenoisingLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_denoise=True)

    elif model_type == 'MTL-Attention-with-canny':
        model = pt_networks.SegNet_attnt_canny.SegNetFilters().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True, flag_filters=True)

    elif model_type == 'MTL-Attention-without-bbox':
        model = pt_networks.SegNet_attnt.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.BaselineLoss(flag_labels = True, flag_segmentations= True, flag_bboxes = False)

    elif model_type == 'MTL-Attention-without-classification':
        model = pt_networks.SegNet_attnt.SegNet().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=False, flag_segmentations=True, flag_bboxes=True)

    elif model_type == 'MTL-segnet-with-canny':
        model = pt_networks.SegNet_canny.SegnetOpencv().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_filters=True)
                                          
    elif model_type == 'MTL-segnet-with-colourization':
        model = pt_networks.segnet_color.Segnet().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.ColorLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_color=True)

    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model


def save_model(model, model_path):
    # todo: save model
    pass
