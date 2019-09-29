
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from config import device, params_victor, params_isak
from dataprep import GLOBHEDataset, ToTensor, RandomCrop, RandomFlip, RandomRotate
import os, shutil
from UnetModel import UNet
import numpy as np
import torch.nn.functional as F
import plot
import datetime
from help_functions import calculate_segmentation_percentages, calculate_segmentation_percentage_error, \
    correct_mask_bitmaps_for_crop, ratio_loss_function

# TODO: Create better structure for model pipeline
# start tensorboard with tensorboard --logdir='runs'
# watch -n 0.5 nvidia-smi

# split_data_to_train_val_test(raw_base_path='data_raw/Training_dataset', new_base_path='data', val_ratio=0.3, test_ratio=0.2)
# exit(0)

if os.path.exists('models') is False:
    os.mkdir('models')

if os.path.exists('models/trained') is False:
    os.mkdir('models/trained')

if os.path.exists('runs') is True:
    shutil.rmtree('runs')

model_name = f'unet_{datetime.datetime.today().strftime("%Y-%m-%d_%H%M")}.pth'

params = params_victor
params['model_name'] = model_name

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
test_dataset = GLOBHEDataset('data', 'test', transform=GLOBHE_transforms_val)
val_dataset = GLOBHEDataset('data', 'val', transform=GLOBHE_transforms_val)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size']['train'], shuffle=True, num_workers=params['nbr_cpu'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size']['test'], shuffle=True, num_workers=params['nbr_cpu'])
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
    train_loss_segment = []
    train_loss_percentage = []
    val_percentage_error = []

    print(f'Training')
    for batch in train_loader:
        image_input = batch['image'].to(device)
        bitmap = batch['bitmap'].to(device)

        optimizer.zero_grad()
        output, class_fraction = model(image_input)

        segment_loss = criterion(output, bitmap)
        ratio_loss = ratio_loss_function(class_fraction, bitmap)

        ratio_loss.backward(retain_graph=True)
        segment_loss.backward()
        optimizer.step()

        output_soft = F.softmax(output, dim=1)
        # _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?

        # Calculate percentages from image segmentation
        segmentation_percentages = calculate_segmentation_percentages(output_soft)

        # Correct bitmaps to crop to get percentages to evaluate to
        corrected_bitmaps = correct_mask_bitmaps_for_crop(bitmap)

        # calulate percentage error
        batch_seg_perc_error, total_batch_seg_perc_error = \
            calculate_segmentation_percentage_error(segmentation_percentages, corrected_bitmaps)

        # save values for evaluating
        val_percentage_error.append(batch_seg_perc_error)
        train_loss_segment.append(segment_loss.data.item())
        train_loss_percentage.append(ratio_loss.item())
        train_loss.append(segment_loss.data.item() + ratio_loss.item())

    # add perc error for every class in plot
    for class_name in batch_seg_perc_error.keys():
        writer.add_scalar(f'Percentage_error_train/{class_name}',
                          np.mean([b[class_name] for b in val_percentage_error]),
                          epoch)

    writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
    writer.add_scalar('Loss/train_segment', np.mean(train_loss_segment), epoch)
    writer.add_scalar('Loss/train_percentage', np.mean(train_loss_percentage), epoch)

    print(f'Evaluating')
    val_loss = []
    val_loss_segment = []
    val_loss_percentage = []
    val_percentage_error = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            image_input = batch['image'].to(device)
            bitmap = batch['bitmap'].to(device)

            output, class_fraction = model(image_input)
            segment_loss = criterion(output, bitmap)
            ratio_loss = ratio_loss_function(class_fraction, bitmap)

            val_loss.append(segment_loss.data.item() + ratio_loss.item())
            val_loss_segment.append(segment_loss.data.item())
            val_loss_percentage.append(ratio_loss.item())

            output_soft = F.softmax(output, dim=1)
            _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?

            segmentation_percentages = calculate_segmentation_percentages(output_soft)
            batch_seg_perc_error, total_batch_seg_perc_error = \
                calculate_segmentation_percentage_error(segmentation_percentages, batch['percentage'])

            val_percentage_error.append(batch_seg_perc_error)

    epoch_val_loss = np.mean(val_loss)
    # add perc error for every class in plot
    for class_name in batch_seg_perc_error.keys():
        writer.add_scalar(f'Percentage_error_val/{class_name}',
                          np.mean([b[class_name] for b in val_percentage_error]),
                          epoch)

    writer.add_scalar('Loss/val', np.mean(val_loss), epoch)
    writer.add_scalar('Loss/val_segment', np.mean(val_loss_segment), epoch)
    writer.add_scalar('Loss/val_percentage', np.mean(val_loss_percentage), epoch)

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

    if epoch_val_loss > best_val_loss:
        # if this happens X epochs in a row, abort?
        print('Overfitting?')

print(f'Training done! Best validation loss was {best_val_loss}. Saved to file {model_name}.')

print(f'Testing :O')
test_loss = []
test_loss_segment = []
test_loss_percentage = []
test_percentage_error = []

model.eval()

with torch.no_grad():
    for batch in test_loader:
        image_input = batch['image'].to(device)
        bitmap = batch['bitmap'].to(device)

        output, class_fraction = model(image_input)
        segment_loss = criterion(output, bitmap)
        ratio_loss = ratio_loss_function(class_fraction, bitmap)

        test_loss.append(segment_loss.data.item() + ratio_loss.item())
        test_loss_segment.append(segment_loss.data.item())
        test_loss_percentage.append(ratio_loss.item())

        output_soft = F.softmax(output, dim=1)
        # _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?

        segmentation_percentages = calculate_segmentation_percentages(output_soft)
        batch_seg_perc_error, total_batch_seg_perc_error = \
            calculate_segmentation_percentage_error(segmentation_percentages, batch['percentage'])

        test_percentage_error.append(batch_seg_perc_error)


print(f'Test loss was {np.mean(test_loss)}')
for class_name in batch_seg_perc_error.keys():
    print(f'Percentage error for {class_name}', np.mean([b[class_name] for b in test_percentage_error]))
