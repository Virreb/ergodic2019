from plot import show_image

# show_image(base='data/raw', name='coxs_1_02_02')

import torch
from torch.utils.data import DataLoader
from dataprep import GLOBHEDataset
import os

print('GPU available:', torch.cuda.is_available())
train_dataset = GLOBHEDataset('train')
test_dataset = GLOBHEDataset('test')
val_dataset = GLOBHEDataset('val')

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=os.cpu_count()-2)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=os.cpu_count()-2)

