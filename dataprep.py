from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch
import PIL
import json

from torchvision import transforms


class GLOBHEDataset(Dataset):
    def __init__(self, base_path, dataset_type, transform=None):
        import os
        self.image_names = os.listdir(f'{base_path}/{dataset_type}/images')
        self.image_paths = [f'{base_path}/{dataset_type}/images/{image_name}' for image_name in self.image_names]
        self.bitmap_paths = [path.replace('images', 'integer_masks').replace('.jpg', '.png') for path in self.image_paths]
        # self.percentage_paths = [path.replace('images', 'percentages').replace('.jpg', '.json') for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
#         with open(self.percentage_paths[idx], 'r') as f:
#             perc = json.load(f)

        sample = {
            'image': Image.open(self.image_paths[idx]),
            'bitmap': Image.open(self.bitmap_paths[idx]),
#             'percentage': perc
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']
        w, h = bitmap.size
        w_new, h_new = self.output_size
        new_corner_x = np.random.randint(0, 1+w-w_new)
        new_corner_y = np.random.randint(0, 1+h-h_new)

        cropped_bitmap = bitmap.crop((new_corner_x, new_corner_y, new_corner_x+w_new, new_corner_y+h_new))
        cropped_image = image.crop((new_corner_x, new_corner_y, new_corner_x+w_new, new_corner_y+h_new))
        sample['image'] = cropped_image
        sample['bitmap'] = cropped_bitmap
        return sample


class RandomFlip:
    def __call__(self, sample):

        image = sample['image']
        bitmap = sample['bitmap']

        r = np.random.rand()
        if r < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            bitmap = bitmap.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = image
        sample['bitmap'] = bitmap
        return sample


class RandomRotate:
    def __init__(self):
        self.possible_rotations = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, None]

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']
        rotation = np.random.choice(self.possible_rotations)
        if rotation is not None:
            image = image.transpose(rotation)
            bitmap = bitmap.transpose(rotation)

        sample['image'] = image
        sample['bitmap'] = bitmap
        return sample


class ToTensor:
    def __init__(self):
        from torchvision import transforms
        self._built_in_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']

        return {
            'image': self._built_in_to_tensor(image),
            'bitmap': (255*self._built_in_to_tensor(bitmap)).long().squeeze(),
#             'percentage': sample['percentage']
        }

class RescalePretrained:
    def __call__(self, sample):
        # should be called after 2 tensor
        image = sample['image']
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            image[i] = (image[i]-mean[i])/std[i]

        sample['image'] = image
        return sample


class Resize:
    def __init__(self, output_size):
        from torchvision import transforms

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self._built_in_resize = transforms.Resize(self.output_size, interpolation=0)

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']

        img = self._built_in_resize(image)
        bmp = self._built_in_resize(bitmap)

        return {'image': img, 'bitmap': bmp, 'percentage': sample['percentage']}


def split_data_to_train_val_test(raw_base_path, new_base_path, val_ratio=0.3, test_ratio=0.2):
    import os
    import shutil
    import numpy as np
    from tqdm import tqdm
    import matplotlib.image as mpimg
    import imageio

    print(f'Copying data from "{raw_base_path}" to "{new_base_path}"')

    # Considition format
    raw_image_path = raw_base_path + '/Images'
    raw_mask_path = raw_base_path + '/Masks/all'
    raw_perc_path = raw_base_path + '/Percentages'

    split_names = ['train', 'val', 'test']

    new_train_val_test_folders = [new_base_path + '/' + tmp for tmp in split_names]

    # if data already exists, remove
    if os.path.isdir(new_base_path):
        shutil.rmtree(new_base_path)

    # create folders
    os.mkdir(new_base_path)
    folders_to_create = new_train_val_test_folders + \
                       [f + '/masks' for f in new_train_val_test_folders] + \
                       [f + '/images' for f in new_train_val_test_folders] + \
                       [f + '/percentages' for f in new_train_val_test_folders] + \
                       [f + '/integer_masks' for f in new_train_val_test_folders]

    for folder in folders_to_create:
        os.mkdir(folder)

    list_of_images = os.listdir(raw_image_path)
    nbr_images = int(len(list_of_images))

    # shuffle images
    list_of_images = np.random.choice(list_of_images, replace=False, size=nbr_images)

    # split limits
    nbr_val = int(val_ratio*nbr_images)
    nbr_test = int(test_ratio*nbr_images)
    nbr_train = nbr_images - nbr_val - nbr_test

    # split the data
    train_images = list_of_images[:nbr_train]
    val_images = list_of_images[nbr_train: (nbr_train + nbr_val)]
    test_images = list_of_images[(nbr_train + nbr_val):]

    print('%i train images' % train_images.size)
    print('%i val images' % val_images.size)
    print('%i test images' % test_images.size)

    image_splits = [train_images, val_images, test_images]

    for idx, split_name in enumerate(split_names):
        print(f'\nCopying {split_name} data')

        for image_name in tqdm(image_splits[idx]):

            # copy image
            new_image_path = new_train_val_test_folders[idx] + '/images/' + image_name
            shutil.copy(raw_image_path + '/' + image_name, new_image_path)

            # images may be of different size, pad it out
            image_array = mpimg.imread(new_image_path)
            if image_array.shape != (1024, 1024, 3):
                img = np.zeros((1024, 1024, 3))

                img[:image_array.shape[0], :image_array.shape[1], :] = image_array

                assert np.alltrue(
                    img[:image_array.shape[0], :image_array.shape[1], :] ==
                    image_array[:image_array.shape[0], :image_array.shape[1], :])

                imageio.imsave(new_image_path, img.astype(np.uint8))

            # copy masks
            mask_name = image_name.replace('.jpg', '.png')
            new_mask_path = new_train_val_test_folders[idx] + '/masks/' + mask_name
            shutil.copy(raw_mask_path + '/' + mask_name, new_mask_path)

            # copy percentages
            perc_name = image_name.replace('.jpg', '.json')
            new_perc_path = new_train_val_test_folders[idx] + '/percentages/' + perc_name
            shutil.copy(raw_perc_path + '/' + perc_name, new_perc_path)

            # also create integer bitmaps from the masks
            image_array = mpimg.imread(new_mask_path)
            integer_bitmap = generate_integer_bitmaps(image_array)

            new_integer_mask_path = new_train_val_test_folders[idx] + '/integer_masks/' + mask_name
            imageio.imsave(new_integer_mask_path, integer_bitmap.astype(np.uint8))


def generate_integer_bitmaps(rgb_bitmap):
    import numpy as np

    nx, ny = rgb_bitmap.shape[0], rgb_bitmap.shape[1]
    integer_bitmap = np.zeros((nx, ny))
    integer_bitmap[rgb_bitmap[:, :, 0] == 1] = 1
    integer_bitmap[rgb_bitmap[:, :, 1] == 1] = 2
    integer_bitmap[rgb_bitmap[:, :, 2] == 1] = 3
    integer_bitmap.astype(np.uint8)

    return integer_bitmap


def get_data_loaders(params):

    # Transforms
    GLOBHE_transforms_train = transforms.Compose([
        RandomCrop(params['image_size']['train']),
        RandomFlip(),
        RandomRotate(),
        ToTensor()
    ])

    GLOBHE_transforms_val = transforms.Compose([ToTensor()])

    train_dataset = GLOBHEDataset('data', 'train', transform=GLOBHE_transforms_train)
    test_dataset = GLOBHEDataset('data', 'test', transform=GLOBHE_transforms_val)
    val_dataset = GLOBHEDataset('data', 'val', transform=GLOBHE_transforms_val)

    train_loader, test_loader, val_loader = None, None, None
    if train_dataset.__len__() > 0:
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'][params['model_name']]['train'],
                                  shuffle=True, num_workers=params['nbr_cpu'])

    if test_dataset.__len__() > 0:
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'][params['model_name']]['test'],
                                 shuffle=True, num_workers=params['nbr_cpu'])

    if val_dataset.__len__() > 0:
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'][params['model_name']]['val'],
                                shuffle=True, num_workers=params['nbr_cpu'])

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


if __name__ == '__main__':

    # TODO: Remove this?
    # split_data_to_train_val_test(raw_base_path='data_raw/Training_dataset', new_base_path='data', val_ratio=0.3, test_ratio=0.2)

    train_dataset = GLOBHEDataset('data', 'train')
    sample = train_dataset[8]

    import matplotlib.pyplot as plt

    bitmap_array = np.array(sample['bitmap'])
    class_perc = np.zeros(4)
    for i in range(4):
        class_perc[i] = np.sum(bitmap_array == i) / np.prod(bitmap_array.shape)
    print(class_perc*100)
    image_array = np.array(sample['image'])
    print(sample['percentage'])

    plt.subplot(1, 2, 1)
    plt.imshow(bitmap_array)
    plt.colorbar()
    plt.clim(0, 3)
    plt.subplot(1, 2, 2)
    plt.imshow(image_array)
    plt.colorbar()
    plt.clim(0, 3)
    plt.show()
    """
    plt.figure()
    rc = RandomCrop((512, 512))
    sample = rc(sample)
    plt.subplot(1,2,1)
    plt.imshow(np.array(sample['bitmap']))
    plt.colorbar()
    plt.clim(0, 3)
    plt.subplot(1,2,2)
    plt.imshow(np.array(sample['image']))
    plt.colorbar()
    plt.clim(0, 3)
    plt.show()"""


