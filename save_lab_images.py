import h5py
import os
import numpy as np
import cv2

def rgb2lab(img_path,destination_path):

    hf=h5py.File(os.path.join(destination_path,'Labimages.h5'), 'w')
    
    imgs = h5py.File(img_path, 'r')

    dataset_list = list(imgs.keys())[0]

    numpy_array=np.array(imgs[dataset_list]).astype(np.uint8)


    hf.create_dataset("Lab_img",
                 shape=numpy_array.shape)
               
        
    for idx,img in enumerate(numpy_array):

        hf['Lab_img'][idx,...] = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)


    hf.close()

if __name__=="__main__":
    rgb2lab('data/train/images.h5','data/train')
    rgb2lab('data/test/images.h5','data/test')
    rgb2lab('data/val/images.h5','data/val')

