import torch


def baseline_losses():

 cri_seg = torch.nn.CrossEntropyLoss()
 cri_class=torch.nn.CrossEntropyLoss()

 return cri_seg,cri_class