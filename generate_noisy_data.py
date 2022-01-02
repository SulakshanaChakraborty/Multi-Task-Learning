import pathlib
import numpy as np
import h5py
from matplotlib import pyplot as plt
def add_noise(data_path, std_high=20, mean_high=10):
        # convert h5 to numpy
        images_filepath = pathlib.Path(data_path + '/images.h5')
        images = h5py.File(images_filepath, 'r')

        key = list(images.keys())[0]
        img = images[key]     

        std_arr = np.random.uniform(low = 0,high =std_high,size = (img.shape[0],1,1,1))
        mean_arr = np.random.uniform(low = 0,high =mean_high,size = (img.shape[0],1,1,1))

        noise_normal = np.random.normal(size = img.shape)
        noise_random_normal = noise_normal*std_arr + mean_arr
        noisy_img = img + noise_random_normal
        images_noisy_filepath = pathlib.Path(data_path + '/noisy_data.h5')
        noisy_img_clipped = np.clip(noisy_img, 0, 255) 
        h5_noisy = h5py.File(images_noisy_filepath, 'w')
        h5_noisy.create_dataset("noisy_data", data=noisy_img_clipped)
        h5_noisy.close()
        return images_noisy_filepath



