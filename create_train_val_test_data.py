import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
import imageio


raw_image_path = 'data_raw/Training_dataset/Images'
raw_mask_path = 'data_raw/Training_dataset/Masks/all'

train_folder = 'data/train'
val_folder = 'data/val'
test_folder = 'data/test'

val_ratio = 0.3
test_ratio = 0.2

if os.path.isdir('data'):
    shutil.rmtree('data')

os.mkdir('data')
os.mkdir(train_folder)
os.mkdir(val_folder)
os.mkdir(test_folder)

os.mkdir(train_folder+'/masks')
os.mkdir(val_folder+'/masks')
os.mkdir(test_folder+'/masks')

os.mkdir(train_folder+'/images')
os.mkdir(val_folder+'/images')
os.mkdir(test_folder+'/images')

list_of_images = os.listdir(raw_image_path)
nbr_images = int(len(list_of_images))
list_of_images = np.random.choice(list_of_images, replace=False, size=nbr_images)

nbr_val = int(val_ratio*nbr_images)
nbr_test = int(test_ratio*nbr_images)
nbr_train = nbr_images-nbr_val-nbr_test

train_images = list_of_images[:nbr_train]
print('%i train images' % train_images.size)
val_images = list_of_images[nbr_train: (nbr_train + nbr_val)]
print('%i val images' % val_images.size)
test_images = list_of_images[(nbr_train + nbr_val):]
print('%i test images' % test_images.size)

for image_name in tqdm(train_images):
    shutil.copy(raw_image_path + '/' + image_name, train_folder + '/images/' + image_name)

    image_nm = train_folder + '/images/' + image_name
    image_array = mpimg.imread(image_nm)
    if image_array.shape != (1024, 1024, 3):
        A = np.zeros((1024, 1024, 3))

        A[:image_array.shape[0], :image_array.shape[1], :] = image_array

        assert np.alltrue(A[:image_array.shape[0], :image_array.shape[1], :] == image_array[:image_array.shape[0], :image_array.shape[1], :])

        imageio.imsave(image_nm, A.astype(np.uint8))

    image_name = image_name.replace('.jpg', '.png')
    shutil.copy(raw_mask_path + '/' + image_name, train_folder + '/masks/' + image_name)

for image_name in tqdm(val_images):
    shutil.copy(raw_image_path + '/' + image_name, val_folder + '/images/' + image_name)

    image_nm = val_folder + '/images/' + image_name
    image_array = mpimg.imread(image_nm)
    if image_array.shape != (1024, 1024, 3):
        A = np.zeros((1024, 1024, 3))

        A[:image_array.shape[0], :image_array.shape[1], :] = image_array

        assert np.alltrue(A[:image_array.shape[0], :image_array.shape[1], :] == image_array[:image_array.shape[0], :image_array.shape[1], :])

        imageio.imsave(image_nm, A.astype(np.uint8))

    image_name = image_name.replace('.jpg', '.png')
    shutil.copy(raw_mask_path + '/' + image_name, val_folder + '/masks/' + image_name)

for image_name in tqdm(test_images):
    shutil.copy(raw_image_path + '/' + image_name, test_folder + '/images/' + image_name)

    image_nm = test_folder + '/images/' + image_name
    image_array = mpimg.imread(image_nm)
    if image_array.shape != (1024, 1024, 3):
        A = np.zeros((1024, 1024, 3))

        A[:image_array.shape[0], :image_array.shape[1], :] = image_array

        assert np.alltrue(A[:image_array.shape[0], :image_array.shape[1], :] == image_array[:image_array.shape[0], :image_array.shape[1], :])

        imageio.imsave(image_nm, A.astype(np.uint8))

    image_name = image_name.replace('.jpg', '.png')
    shutil.copy(raw_mask_path + '/' + image_name, test_folder + '/masks/' + image_name)


