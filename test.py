
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataprep import GLOBHEDataset, ToTensor, RandomCrop, RandomFlip, RandomRotate
import os
from UnetModel import UNet
import numpy as np
import torch.nn.functional as F
import plot
import datetime
from help_functions import calculate_segmentation_percentages, calculate_segmentation_percentage_error

# TODO: Create better structure for model pipeline
# start tensorboard with tensorboard --logdir='runs'

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

params_victor = {
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

params_isak = {
    'learning_rate': 0.0002,
    'num_epochs': 100,
    'nbr_cpu': os.cpu_count() - 4,
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

params = params_victor

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


def ratio_loss(class_fractions, bitmaps):
    bitmap_fraction = torch.zeros(size=class_fraction.shape).to(device)
    for j in range(4):
        bitmap_fraction[:, j] = torch.sum(torch.sum(bitmaps==j, dim=2), dim=1).float() / (bitmaps.shape[1]*bitmaps.shape[2])

    loss = torch.mean((class_fractions - bitmap_fraction)**2)
    return loss


best_val_loss = 1000000
for epoch in range(params['num_epochs']):
    print(f'\nEpoch {epoch+1}/{params["num_epochs"]}')
    train_loss = []
    val_percentage_error = []

    print(f'Training')
    for batch in train_loader:
        image_input = batch['image'].to(device)
        bitmap = batch['bitmap'].to(device)

        optimizer.zero_grad()
        output, class_fraction = model(image_input)     # TODO: Use percentage as keyword from segmentation and fractions as keyword for raw prediction?

        loss = criterion(output, bitmap)
        loss2 = ratio_loss(class_fraction, bitmap)

        loss2.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        # TODO: Create loss function with predicted class fractions aswel?

        output_soft = F.softmax(output, dim=1)
        # _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?

        segmentation_percentages = calculate_segmentation_percentages(output_soft)
        batch_seg_perc_error, total_batch_seg_perc_error = \
            calculate_segmentation_percentage_error(segmentation_percentages, batch['percentage'])

        val_percentage_error.append(batch_seg_perc_error)
        train_loss.append(loss.data.item() + loss2.item())

    # add perc error for every class in plot
    for class_name in batch_seg_perc_error.keys():
        writer.add_scalar(f'Percentage_error_train/{class_name}',
                          np.mean([b[class_name] for b in val_percentage_error]),
                          epoch)

    writer.add_scalar('Loss/train', np.mean(train_loss), epoch)

    print(f'Evaluating')
    val_loss = []
    val_percentage_error = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            image_input = batch['image'].to(device)
            bitmap = batch['bitmap'].to(device)

            output, class_fraction = model(image_input)
            loss = criterion(output, bitmap)
            val_loss.append(loss.data.item())

            output_soft = F.softmax(output, dim=1)
            _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?

            segmentation_percentages = calculate_segmentation_percentages(output_soft)
            batch_seg_perc_error, total_batch_seg_perc_error = \
                calculate_segmentation_percentage_error(segmentation_percentages, batch['percentage'])

            val_percentage_error.append(batch_seg_perc_error)
            val_loss.append(loss.data.item())

    # add perc error for every class in plot
    for class_name in batch_seg_perc_error.keys():
        writer.add_scalar(f'Percentage_error_val/{class_name}',
                          np.mean([b[class_name] for b in val_percentage_error]),
                          epoch)

    epoch_val_loss = np.mean(val_loss)
    writer.add_scalar('Loss/val', epoch_val_loss, epoch)

    # TODO: update plots to show every class
    # print image to tensorboard
    fig = plot.get_images(original=image_input, mask=bitmap, predicted=output_integer)
    writer.add_figure(f'Epoch {epoch+1}', fig, epoch)
    writer.flush()

    model.train()

    if epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), f'models/trained/{model_name}')
        print('New best model! Current loss:', epoch_val_loss)
        best_val_loss = epoch_val_loss
        # TODO: also save the params? Should include the transform names? Maybe also the loss?

    if epoch_val_loss > epoch_val_loss:
        # if this happens X epochs in a row, abort?
        print('Overfitting?')

print(f'All done! Best validation loss was {best_val_loss}. Saved to file {model_name}.')
