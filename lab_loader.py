import h5py
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pathlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def rgb2lab(img_path,destination_path):
    """A function used to convert the rgb colour space images to the 
    lab colour space, where l is the lightness channel, a is the red green-channel and 
    b is the blue-yellow channel.

    Args:
        img_path (path): file path of the images to be converted.
        destination_path (path): file of the where converted images should be transfered.
    """

    hf=h5py.File(os.path.join(destination_path,'Labimages.h5'), 'w')
    
    imgs = h5py.File(img_path, 'r')

    dataset_list = list(imgs.keys())[0]

    numpy_array=np.array(imgs[dataset_list]).astype(np.uint8)


    hf.create_dataset("Lab_img",
                 shape=numpy_array.shape)
               
        
    for idx,img in enumerate(numpy_array):

        hf['Lab_img'][idx,...] = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)


    hf.close()

class H5LabImageLoader(Dataset):
    """
    Dataloader containing __len__ & __getitem__ as per
    Pytorch dataloader specifications
    """

    def __init__(self, img_path_lab,mask_file, bbox_file, classification_file, transform):
        """
        @params:
    
        img_lab_path(string): Path for Lab images
        mask_file(string): Path for corresponding masks
        bbox_file(string): Path for bounding boxes
        classification_file(string): Path for classes (0 or 1)
        transform(callable): Transform to be applied to the images ## ONLY TRAINING IMAGES, MASKS? ( I DONT THINK SO)
        """
       


        self.mask_h5 = h5py.File(mask_file, 'r')
        self.bbox_h5 = h5py.File(bbox_file, 'r')
        self.classifcation_h5 = h5py.File(classification_file, 'r')



        self.mask_list = list(self.mask_h5.keys())[0]
        self.bbox_list = list(self.bbox_h5.keys())[0]
        self.classification_list = list(self.classifcation_h5.keys())[0]

        self.lab = h5py.File(img_path_lab, 'r')
        self.lab_list=list(self.lab.keys())[0]
        

        self.transform = transform

    def __len__(self):
       return self.lab[list(self.lab.keys())[0]].shape[0]
       #return 10

    def __getitem__(self, idx):

    
        mask = self.mask_h5[self.mask_list][idx]
        bbox = self.bbox_h5[self.bbox_list][idx]
        classification = self.classifcation_h5[self.classification_list][idx]

        lab_image=self.lab[self.lab_list][idx]


      
        if self.transform:
        #    # mask_transform = transforms.Compose(
        #      [transforms.ToTensor(),
        #       transforms.Resize((64,64))]) 
            lab_image = self.transform(lab_image).to(
                torch.float32)  # float32 for pytorch compatibility (weights initialized to the same)
            
        L=lab_image[0,:,:]
        L=L[None,:]
        ab=lab_image[1:3,:,:]
        

            #mask= mask_transform(mask)

        

        return L, {'mask': mask, 'bbox': bbox, 'classification': classification,'ab':ab}

def create_data_loaders(train_path, validation_path, test_path, batch_size=16):
    """A function for creating pytorch training, validation and testing dataloader objects.

    Args:
        train_path (path): The path for the training data.
        validation_path (path): The path for the validation data.
        test_path (path): The path for the testing data.
        batch_size (int, optional): Size of the batch. Defaults to 16.
        noisy (bool, optional): Boolean for addition of noise. Defaults to False.

    Returns:
        train_loader, val_loader, test_loader are the torch.DataLoader() objects for their respective datasets;
        training, validation and testing sets.
    """
    # Train data
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
        ])
    train_loader = build_data_loader(data_path=train_path, pt_transforms=train_transform, batch_size=batch_size)
    # Validation data
    validation_transform = train_transform
    validation_loader = build_data_loader(data_path=validation_path, pt_transforms=validation_transform,
                                          batch_size=batch_size)
    # Test data
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
         ])
    test_loader = build_data_loader(data_path=test_path, pt_transforms=test_transform, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def build_data_loader(data_path, pt_transforms, batch_size=16):
    """A function used for creating a single data loader.

    Args:
        data_path (path): Path used to access the data.
        pt_transforms (pytorch object): Pytorch transforms to be applied onto the dataset, contained
        as a list inside a pytorch transforms object. 
        batch_size (int, optional): Size of the batch. Defaults to 16.
        noisy (bool, optional): Boolean for addition of noise. Defaults to False.

    Returns:
        data_loader (pytorch object): DataLoader object for a specific dataset.
    """
    # Define paths
    images_filepath = pathlib.Path(data_path + '/Labimages.h5')
    masks_filepath = pathlib.Path(data_path + '/masks.h5')
    bboxes_filepath = pathlib.Path(data_path + '/bboxes.h5')
    labels_filepath = pathlib.Path(data_path + '/binary.h5')

    # Create loader
    image_loader = H5LabImageLoader(img_path_lab=images_filepath, mask_file=masks_filepath, bbox_file=bboxes_filepath,
                                 classification_file=labels_filepath, transform=pt_transforms)  # All data paths
    # Create pytorch loader
    data_loader = DataLoader(image_loader, batch_size=batch_size, shuffle=True)

    return data_loader


