import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imageio

train_folder = 'data/train'
val_folder = 'data/val'
test_folder = 'data/test'

integer_masks_dir = 'integer_masks'


def generate_integer_bitmaps(rgb_bitmap):
    nx, ny = rgb_bitmap.shape[0], rgb_bitmap.shape[1]
    integer_bitmap = np.zeros((nx, ny))
    integer_bitmap[rgb_bitmap[:, :, 0] == 1] = 1
    integer_bitmap[rgb_bitmap[:, :, 1] == 1] = 2
    integer_bitmap[rgb_bitmap[:, :, 2] == 1] = 3
    integer_bitmap.astype(np.uint8)
    return integer_bitmap


for folder in [train_folder, val_folder, test_folder]:

    full_integer_masks_dir = os.path.join(folder, integer_masks_dir)
    if os.path.isdir(full_integer_masks_dir):
        shutil.rmtree(full_integer_masks_dir)

    os.mkdir(full_integer_masks_dir)

    raw_masks_dir = os.path.join(folder, 'masks')
    masks_name = os.listdir(raw_masks_dir)

    for mask_name in tqdm(masks_name):
        if mask_name.endswith('.png'):
            image_array = mpimg.imread(os.path.join(raw_masks_dir, mask_name))
            integer_bitmap = generate_integer_bitmaps(image_array)
            integer_mask_name = os.path.join(folder, integer_masks_dir, mask_name)
            imageio.imsave(integer_mask_name, integer_bitmap.astype(np.uint8))
