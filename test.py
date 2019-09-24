
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataprep import GLOBHEDataset, ToTensor, Resize, RandomCrop, RandomFlip, RandomRotate  # split_data_to_train_val_test
import os
from UnetModel import UNet
import numpy as np
import torch.nn.functional as F
import plot
import tqdm

# TODO: Create better structure for model pipeline
# start tensorboard with tensorboard --logdir='runs'


def get_percentages_from_output(output):
    """
    Work in progress :)

    :param output:
    :return:
    """

    output_size = output.size()
    print(output_size)
    total = output_size[1]*output_size[2]
    keys = ['building_percentage', 'water_percentage', 'road_percentage']
    perc_dict = {}
    perc_list = []

    # Loop over all results in the batch and sum the error? mean?
    for idx in range(output_size[0]):
        percentage = np.sum(output[idx, :, :]) / total * 100
        perc_dict[keys[idx]] = percentage
        perc_list.append(percentage)

    return perc_dict, perc_list


def get_percentage_error(predicted, real):
    """
    Testing :)

    :param predicted:
    :param real:
    :return:
    """
    prec_error = {}
    for key in predicted.keys():
        prec_error[key] = predicted[key] - real[key]

    total_error = sum(prec_error.values())

    return prec_error, total_error


# split_data_to_train_val_test(raw_base_path='data_raw/Training_dataset', new_base_path='data', val_ratio=0.3, test_ratio=0.2)
# exit(0)

if os.path.exists('models') is False:
    os.mkdir('models')

if os.path.exists('models/trained') is False:
    os.mkdir('models/trained')

# if os.path.exists('runs') is True:
#     shutil.rmtree('runs')


image_size = (128, 128)
# output_size = (512, 512)
# output_size = (1024, 1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nbr_cpu = os.cpu_count() - 2
batch_size = 2
num_epochs = 2
learning_rate = 0.0005
model_name = 'test_net.pth'

print('GPU available:', torch.cuda.is_available())
print('Number of CPUs:', nbr_cpu)
print('Batch size:', batch_size)

# Transforms
GLOBHE_transforms_train = transforms.Compose([
        RandomCrop(image_size),
        RandomFlip(),
        RandomRotate(),
        ToTensor()
    ])

GLOBHE_transforms_val = transforms.Compose([ToTensor()])

train_dataset = GLOBHEDataset('data', 'train', transform=GLOBHE_transforms_train)
#test_dataset = GLOBHEDataset('data', 'test', transform=GLOBHE_transforms)
val_dataset = GLOBHEDataset('data', 'val', transform=GLOBHE_transforms_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nbr_cpu)

model = UNet(3, 4).float()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# init tensorboard
writer = SummaryWriter()

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    train_loss = []
    train_perc_error = []

    print(f'Training')
    for batch in train_loader:
        image_input = batch['image'].to(device)
        bitmap = batch['bitmap'].to(device)

        optimizer.zero_grad()
        output, class_fraction = model(image_input)

        loss = criterion(output, bitmap)
        loss.backward()
        optimizer.step()

        # get percentages
        _, output_integer = F.softmax(output, dim=1).max(1)
        # predicted_perc, _ = get_percentages_from_output(output_integer)
        # perc_error, total_perc_error = get_percentage_error(predicted_perc, batch['percentage'])
        # train_perc_error.append(total_perc_error)

        train_loss.append(loss.data.item())

    # writer.add_scalar('Perc error/train', np.mean(train_perc_error), epoch)
    writer.add_scalar('Loss/train', np.mean(train_loss), epoch)

    print(f'Evaluating')
    eval_loss = []
    eval_perc_error = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            image_input = batch['image'].to(device)
            bitmap = batch['bitmap'].to(device)

            output, class_fraction = model(image_input)
            loss = criterion(output, bitmap)
            eval_loss.append(loss.data.item())

            # get percentages
            _, output_integer = F.softmax(output, dim=1).max(1)
            # predicted_perc, _ = get_percentages_from_output(output_integer)
            # perc_error, total_perc_error = get_percentage_error(predicted_perc, batch['percentage'])
            # eval_perc_error.append(total_perc_error)

        # writer.add_scalar('Perc error/val', np.mean(eval_perc_error), epoch)
        writer.add_scalar('Loss/val', np.mean(eval_loss), epoch)

    model.train()

    # print image to tensorboard
    fig = plot.get_images(original=image_input, mask=bitmap, predicted=output_integer)
    writer.add_figure(f'Epoch {epoch+1}', fig, epoch)

    writer.flush()

    torch.save(model.state_dict(), f'models/trained/{model_name}')

