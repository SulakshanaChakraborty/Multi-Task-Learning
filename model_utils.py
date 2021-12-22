import sys
import networks
import torch.optim as optim





def get_model(model_type):
    if model_type == 'baseline':
        model = networks.segnet.Segnet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # todo: update
        loss
    if model_type == 'mlt_hard':
        model, optimizer, loss_fn = 1, 2, 3  # todo: update
    if model_type == 'mlt_attention':
        model, optimizer, loss_fn = 1, 2, 3  # todo: update
    if model_type == 'mlt_gscnn':
        model, optimizer, loss_fn = networks.GSCNN(), 2, 3  # todo: update
    else:
        sys.exit(f'Model Type: {model_type} is not implemented.')

    return model, optimizer, loss_fn


def load_model(model_path):
    model = 1  # todo: update
    return model


def save_model(model, model_path):
    # todo: save model
    pass
