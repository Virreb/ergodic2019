from torch.utils.data import Dataset


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
        import imageio
        sample = {
            'image': imageio.imread(self.image_paths[idx]),
            'bitmap': imageio.imread(self.bitmap_paths[idx]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


