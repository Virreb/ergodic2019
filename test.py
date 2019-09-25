
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataprep import GLOBHEDataset, ToTensor, Resize, RandomCrop, RandomFlip, RandomRotate
import os, json
from UnetModel import UNet
import numpy as np
import torch.nn.functional as F
import plot
import datetime

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = f'unet_{datetime.datetime.today().strftime("%Y-%m-%d_%H%M")}.pth'

params = {
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'nbr_cpu': os.cpu_count() - 1,
    'device': device,
    'model_name': model_name,
    'image_size': {
        'train': (256, 256),
        'val': (1024, 1024),
        'test': (1024, 1024)
    },
    'batch_size': {
        'train': 8,
        'val': 2,
        'test': 2,
    },
}

print('GPU available:', torch.cuda.is_available(), ' \nNumber of CPUs:', params['nbr_cpu'])

# Transforms
GLOBHE_transforms_train = transforms.Compose([
        RandomCrop(params['image_size']['train']),
        RandomFlip(),
        RandomRotate(),
        ToTensor()
    ])

GLOBHE_transforms_val = transforms.Compose([ToTensor()])

train_dataset = GLOBHEDataset('data', 'train', transform=GLOBHE_transforms_train)
# test_dataset = GLOBHEDataset('data', 'test', transform=GLOBHE_transforms)
val_dataset = GLOBHEDataset('data', 'val', transform=GLOBHE_transforms_val)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size']['train'], shuffle=True, num_workers=params['nbr_cpu'])
# test_loader = DataLoader(test_dataset, batch_size=params['batch_size']['test'], shuffle=True, num_workers=params['nbr_cpu'])
val_loader = DataLoader(val_dataset, batch_size=params['batch_size']['val'], shuffle=True, num_workers=params['nbr_cpu'])

model = UNet(3, 4).float()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

# init tensorboard
writer = SummaryWriter()

best_val_loss = 1000000
for epoch in range(params['num_epochs']):
    print(f'\nEpoch {epoch+1}/{params["num_epochs"]}')
    train_loss = []
    # train_perc_error = []

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
    val_loss = []
    # val_perc_error = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            image_input = batch['image'].to(device)
            bitmap = batch['bitmap'].to(device)

            output, class_fraction = model(image_input)
            loss = criterion(output, bitmap)
            val_loss.append(loss.data.item())

            # get percentages
            _, output_integer = F.softmax(output, dim=1).max(1)
            # predicted_perc, _ = get_percentages_from_output(output_integer)
            # perc_error, total_perc_error = get_percentage_error(predicted_perc, batch['percentage'])
            # eval_perc_error.append(total_perc_error)

    # writer.add_scalar('Perc error/val', np.mean(eval_perc_error), epoch)
    epoch_val_loss = np.mean(val_loss)
    writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)

    model.train()

    # print image to tensorboard
    fig = plot.get_images(original=image_input, mask=bitmap, predicted=output_integer)
    writer.add_figure(f'Epoch {epoch+1}', fig, epoch)

    writer.flush()

    if epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), f'models/trained/{model_name}')
        print('New best model! Current loss:', epoch_val_loss)
        best_val_loss = epoch_val_loss
        # TODO: also save the params? Should include the transform names? Maybe also the loss?

    if epoch_val_loss > epoch_val_loss:
        # if this happens X epochs in a row, abort?
        print('Overfitting?')

print(f'All done! Best validation loss was {best_val_loss}. Saved to file {model_name}.')
