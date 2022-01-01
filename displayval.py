from torch.utils.data import dataloader
import displaying
import load_data
import model_utils
import train_model
import torch
import test_model
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image

train_path = 'data/train/'
validation_path = 'data/val/'
test_path = 'data/test/'
batch_size = 5
device='cpu'
 
train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,noisy = False
                                                                                 )

writer = SummaryWriter('runs/display_data_validation')
data=iter(train_loader)

images,labels=data.next()


img_grid = torchvision.utils.make_grid(images)
im_1=torchvision.utils.make_grid(images/2*255+.5*255,nrow=4,padding=2) #Making grid for montage of 
                                                                           #Cutout Images
im_=Image.fromarray(im_1.permute(1,2,0).numpy().astype('uint8'))
im_.save("val_data.png") 

print(labels['classification'])

sum_train=0
sum_val=0
sum_test=0
# for i, batch_data in enumerate(train_loader, 1):
     
#             inputs, labels = batch_data
#             inputs = inputs.to(device)
#             binary = torch.squeeze(labels['classification'].to(device))
         

#             sum_train+=torch.sum(binary)
#             print(sum_train,"sum_t")
# for i, batch_data in enumerate(validation_loader, 1):
     
#             inputs, labels = batch_data
#             inputs = inputs.to(device)
#             mask = torch.squeeze(labels['mask'].to(device))
#             mask = mask.to(torch.long)
#             binary = torch.squeeze(labels['classification'].to(device))

#             sum_val+=torch.sum(binary)
#             print(sum_train,"val_t")
# for i, batch_data in enumerate(test_loader, 1):
     
#             inputs, labels = batch_data
#             inputs = inputs.to(device)
#             binary = torch.squeeze(labels['classification'].to(device))
#             sum_test+=torch.sum(binary)
#             print(sum_test,"val_t")

            

print(sum_train,"t")
print(sum_val,"v")
print(sum_test,"test") 
