
import h5py
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision

class H5ImageLoader(Dataset):

    """
    Dataloader containing __len__ & __getitem__ as per 
    Pytorch dataloader specifications

    """

    def __init__(self,img_file,mask_file,bbox_file,classification_file,transform=None):
        
        """

        @params:
        img_file(string): Path for images
        mask_file(string): Path for corresponding masks
        bbox_file(string): Path for bounding boxes 
        classification_file(string): Path for classes (0 or 1)
        
        transform(callable): Transform to be applied to the images ## ONLY TRAINING IMAGES, MASKS? ( I DONT THINK SO)
        """

  
        self.img_h5 = h5py.File(img_file,'r')
        self.mask_h5=h5py.File(mask_file,'r')
        self.bbox_h5=h5py.File(bbox_file,'r')
        self.classifcation_h5=h5py.File(classification_file,'r')
  
        self.dataset_list = list(self.img_h5.keys())[0]
        self.mask_list = list(self.mask_h5.keys())[0]
        self.bbox_list = list(self.bbox_h5.keys())[0]
        self.classification_list = list(self.classifcation_h5.keys())[0]

        self.transform=transform


    def __len__(self):
      return self.img_h5[list(self.img_h5.keys())[0]].shape[0]

    def __getitem__(self,idx):
      image=self.img_h5[self.dataset_list][idx]
      mask=self.mask_h5[self.mask_list][idx]
      bbox=self.bbox_h5[self.bbox_list][idx]
      classification=self.classifcation_h5[self.classification_list][idx]

      if self.transform:
        image=self.transform(image)

      return image,{'mask':mask, 'bbox':bbox, 'classification':classification}

if __name__=='__main__':
 DATA_PATH='data/train'
 t=H5ImageLoader(DATA_PATH+'/images.h5',DATA_PATH+'/masks.h5',DATA_PATH+'/bboxes.h5',DATA_PATH+'/binary.h5') #All data paths 
 dataloader = DataLoader(t, batch_size=8, shuffle=True)

### Showing examples of labels

 fig, ax = plt.subplots()
 train_features, trains = next(iter(dataloader))
 #print("Feature batch shape: ", train_features.size())
 print("Labels batch shape: ",trains['mask'].size())
 img = train_features[0]
 label = torch.squeeze(trains['mask'][0])
 #plt.imshow(img.to(torch.int32))
 X,Y,W,H =trains['bbox'][0]
 box = patches.Rectangle((X, Y), W, H, linewidth=1, edgecolor='b',facecolor='none') #To create the bounding box

 ax.imshow(img.to(torch.int32))
 ax.add_patch(box)
 print(f"The class is trains{trains['classification'][0]}")
 plt.show()