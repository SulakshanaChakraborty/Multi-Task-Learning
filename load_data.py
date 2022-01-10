import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pathlib


def create_data_loaders(train_path, validation_path, test_path, batch_size=16):
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
    # Define paths
    images_filepath = pathlib.Path(data_path + '/images.h5')
    masks_filepath = pathlib.Path(data_path + '/masks.h5')
    bboxes_filepath = pathlib.Path(data_path + '/bboxes.h5')
    labels_filepath = pathlib.Path(data_path + '/binary.h5')

    # Create loader
    image_loader = H5ImageLoader(img_file=images_filepath, mask_file=masks_filepath, bbox_file=bboxes_filepath,
                                 classification_file=labels_filepath, transform=pt_transforms)  # All data paths
    # Create pytorch loader
    data_loader = DataLoader(image_loader, batch_size=batch_size, shuffle=True)

    return data_loader


def take_random_samples(data_loader, n_samples):
    # todo: implement random sampler from data.
    images = 1
    labels = 2
    segmentations = 3
    bboxes = 4
    return images, labels, segmentations, bboxes


class H5ImageLoader(Dataset):
    """
    Dataloader containing __len__ & __getitem__ as per
    Pytorch dataloader specifications

    """

    def __init__(self, img_file, mask_file, bbox_file, classification_file, transform):
        """

        @params:
        img_file(string): Path for images
        mask_file(string): Path for corresponding masks
        bbox_file(string): Path for bounding boxes
        classification_file(string): Path for classes (0 or 1)

        transform(callable): Transform to be applied to the images ## ONLY TRAINING IMAGES, MASKS? ( I DONT THINK SO)
        """

        self.img_h5 = h5py.File(img_file, 'r')
        self.mask_h5 = h5py.File(mask_file, 'r')
        self.bbox_h5 = h5py.File(bbox_file, 'r')
        self.classifcation_h5 = h5py.File(classification_file, 'r')

        self.dataset_list = list(self.img_h5.keys())[0]
        self.mask_list = list(self.mask_h5.keys())[0]
        self.bbox_list = list(self.bbox_h5.keys())[0]
        self.classification_list = list(self.classifcation_h5.keys())[0]

        self.transform = transform

    def __len__(self):
      return self.img_h5[list(self.img_h5.keys())[0]].shape[0]

        #return 10

    def __getitem__(self, idx):
        image = self.img_h5[self.dataset_list][idx]
        mask = self.mask_h5[self.mask_list][idx]
        bbox = self.bbox_h5[self.bbox_list][idx]
        classification = self.classifcation_h5[self.classification_list][idx]

        if self.transform:
            image = self.transform(image).to(
                torch.float32)  # float32 for pytorch compatibility (weights initialized to the same)

        # mask= mask_transform(mask)

        return image, {'mask': mask, 'bbox': bbox, 'classification': classification}
