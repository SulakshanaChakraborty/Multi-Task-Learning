import sys
import cv2
import h5py
import numpy as np


def apply_filter_to_h5(input_path, output_path, cv_filter):
    # Read original h5 file.
    h5_images = h5py.File(input_path, mode='r')
    key = list(h5_images.keys())[0]
    images = np.array(h5_images[key]).astype(np.uint8)

    # Create new h5 file.
    h5_ds = h5py.File(output_path, mode='w')
    h5_ds.create_dataset(cv_filter, shape=images.shape[:-1])

    # Apply opencv filter to each image.
    for i, img in enumerate(images):
        print(f'{cv_filter} -- > Image: {i}')
        if cv_filter == 'canny_filter':
            mod_img = cv2.Canny(img, 100, 200)
        elif cv_filter == 'harris_filter':
            mod_img = img
        else:
            sys.exit(f'Please choose a valid filter: (1) "canny_filter" or "harris_filter"')

        h5_ds[cv_filter][i, ...] = mod_img

    h5_ds.close()

    print(f'H5 File with {cv_filter} dataset has been created in {output_path}')


if __name__ == '__main__':

    for data_set in ['train', 'val', 'test']:
        # Define input filepath
        opencv_filter = 'canny_filter'
        input_filepath = f'data/{data_set}/images.h5'
        # Define output filepath
        output_filepath = f'data/{data_set}/{opencv_filter}.h5'
        # Create dataset with images after filter.
        apply_filter_to_h5(input_path=input_filepath, output_path=output_filepath, cv_filter=opencv_filter)
