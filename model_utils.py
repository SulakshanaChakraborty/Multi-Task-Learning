import sys
import torch
import pt_networks
import torch.optim as optim
import losses
import pt_networks.segnet
import pt_networks.SegNet_Attnt
import pt_networks.SegNet_Attnt_reformat
import torchvision.models as models
import pt_networks.SegNet_Attention_Filters
import pt_networks.segnet_color
import pt_networks.Segnet_attnt_denoising
import pt_networks.SegNet_Attnt_reformat_color
import pt_networks.SegNet_attnt_canny
import pt_networks.SegNet_canny


def get_model(model_type, device='cpu', load_pre_trained_weights=False):
    if model_type == 'baseline':

        model = pt_networks.segnet.Segnet().to(device)
        # vgg16 = models.vgg16(pretrained=True).to(device)
        # model.vgg16_init(vgg16)
        if load_pre_trained_weights:
            model.load_state_dict(torch.load('Segnet3task3layer.pt'))
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.BaselineLoss(False, True, False)
    elif model_type == 'attention_opencv_filter':
        model = pt_networks.SegNet_Attention_Filters.SegNetFilters().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_filters=True)
    elif model_type == 'color_segnet':
        model = pt_networks.segnet_color.Segnet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.ColorLoss(True, True, True, True)

    elif model_type == 'color_attention':
        model = pt_networks.SegNet_Attnt_reformat_color.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.ColorLoss(True, True, True, True)

    elif model_type == 'denoising_attention':
        model = pt_networks.Segnet_attnt_denoising.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # todo: update
        loss_fn = losses.DenoisingLoss(True, True, True, True)

    elif model_type == 'opencv_filter':
        model = pt_networks.SegNet_canny.SegnetOpencv().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg16_init(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_filters=True)

    elif model_type == 'attention_opencv_filter':

        model = pt_networks.SegNet_attnt_canny.SegNetFilters().to(device)
        if load_pre_trained_weights:
            vgg16 = models.vgg16(pretrained=True).to(device)
            model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=5e-6)  # todo: update
        loss_fn = losses.OpencvFilterLoss(flag_labels=True, flag_segmentations=True, flag_bboxes=True,
                                          flag_filters=True)

    elif model_type == 'mlt_attention':
        model = pt_networks.seg.SegNet().to(device)
        vgg16 = models.vgg16(pretrained=True).to(device)
        model.vgg_pretrained(vgg16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = losses.BaselineLoss(False, True, True)  # todo: update

    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model


def save_model(model, model_path):
    # todo: save model
    pass
