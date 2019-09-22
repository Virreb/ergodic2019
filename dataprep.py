import torch
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import numpy as np
import imageio


class GLOBHEDataset(Dataset):
    def __init__(self, dataset_type, transform=None):
        import os
        self.image_names = os.listdir(f'data/{dataset_type}/images')
        self.image_paths = [f'data/{dataset_type}/images/{image_name}' for image_name in self.image_names]
        self.bitmap_paths = [path.replace('images', 'integer_masks').replace('.jpg', '.png') for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        sample = {
            'image': Image.open(self.image_paths[idx]),
            'bitmap': Image.open(self.bitmap_paths[idx]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __init__(self):
        self._built_in_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']

        return {'image': self._built_in_to_tensor(image),
                'bitmap': (255*self._built_in_to_tensor(bitmap)).long().squeeze()}


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self._built_in_resize = transforms.Resize(self.output_size, interpolation=0)

    def __call__(self, sample):
        image = sample['image']
        bitmap = sample['bitmap']

        img = self._built_in_resize(image)
        bmp = self._built_in_resize(bitmap)

        return {'image': img, 'bitmap': bmp}
