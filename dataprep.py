from torch.utils.data import Dataset


class GLOBHEDataset(Dataset):
    def __init__(self, base_path, dataset_type, transform=None):
        import os
        self.image_names = os.listdir(f'{base_path}/{dataset_type}/images')
        self.image_paths = [f'{base_path}/{dataset_type}/images/{image_name}' for image_name in self.image_names]
        self.bitmap_paths = [path.replace('images', 'integer_masks').replace('.jpg', '.png') for path in self.image_paths]
        self.percentage_paths = [path.replace('images', 'percentages').replace('.jpg', '.json') for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        from PIL import Image
        import json

        with open(self.percentage_paths[idx], 'r') as f:
            perc = json.load(f)

        sample = {
            'image': Image.open(self.image_paths[idx]),
            'bitmap': Image.open(self.bitmap_paths[idx]),
            'percentage': perc
        }

        if self.transform is not None:
            sample = self.transform(sample)

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
            'percentage': sample['percentage']
        }


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

if __name__ == '__main__':
    split_data_to_train_val_test(raw_base_path='data_raw/Training_dataset', new_base_path='data', val_ratio=0.3, test_ratio=0)