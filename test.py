from plot import show_image

# show_image(base='data/raw', name='coxs_1_02_02')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataprep import GLOBHEDataset
import os

output_size = (256, 256)

nbr_cpu = os.cpu_count() - 2
print('GPU available:', torch.cuda.is_available())
print('#CPU:', nbr_cpu)

# Transforms
GLOBHE_transforms = transforms.Compose(
    [
        transforms.Resize(output_size),
        transforms.ToTensor()
    ]
)

train_dataset = GLOBHEDataset('train', transform=GLOBHE_transforms)
test_dataset = GLOBHEDataset('test', transform=GLOBHE_transforms)
# val_dataset = GLOBHEDataset('val', transform=GLOBHE_transforms)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=nbr_cpu)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=nbr_cpu)
# val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True, num_workers=nbr_cpu)

