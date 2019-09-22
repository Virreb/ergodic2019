from plot import show_image

# show_image(base='data/raw', name='coxs_1_02_02')

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataprep import GLOBHEDataset, ToTensor, Resize
import os
from UnetModel import UNet
import numpy as np
import torch.nn.functional as F
import plot
import shutil

if os.path.exists('models') is False:
    os.mkdir('models')

if os.path.exists('models/trained') is False:
    os.mkdir('models/trained')

# if os.path.exists('runs') is True:
#     shutil.rmtree('runs')


output_size = (256, 256)
# output_size = (512, 512)
# output_size = (1024, 1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nbr_cpu = os.cpu_count() - 2
batch_size = 8
num_epochs = 5
learning_rate = 0.0005
model_name = 'test_net.pth'

print('GPU available:', torch.cuda.is_available())
print('Number of CPUs:', nbr_cpu)
print('Batch size:', batch_size)

# Transforms
GLOBHE_transforms = transforms.Compose(
    [
        Resize(output_size),
        ToTensor()
    ]
)

train_dataset = GLOBHEDataset('train', transform=GLOBHE_transforms)
test_dataset = GLOBHEDataset('test', transform=GLOBHE_transforms)
val_dataset = GLOBHEDataset('val', transform=GLOBHE_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)

net = UNet(3, 4).float()
net = net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# init tensorboard
writer = SummaryWriter()

for epoch in range(num_epochs):
    epoch_loss = []
    for bath_index, sample in enumerate(train_loader):
        image_input = sample['image'].to(device)
        bitmap = sample['bitmap'].to(device)
        # out_arr = net(image_input)

        optimizer.zero_grad()
        network_output = net(image_input)

        loss = criterion(network_output, bitmap)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data.item())

    writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)
    net.eval()
    eval_loss = []
    with torch.no_grad():
        for bath_index, sample in enumerate(val_loader):
            image_input = sample['image'].to(device)
            bitmap = sample['bitmap'].to(device)

            network_output = net(image_input)
            loss = criterion(network_output, bitmap)
            eval_loss.append(loss.data.item())

        writer.add_scalar('Loss/val', np.mean(eval_loss), epoch)

    net.train()

    _, output_for_plot = F.softmax(network_output, dim=1).max(1)
    print(f'{epoch}/{num_epochs}')

    # print image to tensorboard
    fig = plot.get_images(original=image_input, mask=bitmap, predicted=output_for_plot)
    writer.add_figure('Images', fig, epoch)

    writer.flush()

    torch.save(net.state_dict(), f'models/trained/{model_name}')


