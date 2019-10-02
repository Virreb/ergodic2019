

def train_model(model, params, writer):
    from dataprep import get_data_loaders
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    import torch.nn.functional as F
    import plot
    from help_functions import calculate_segmentation_percentages, calculate_segmentation_percentage_error, \
        correct_mask_bitmaps_for_crop, ratio_loss_function
    from config import device
    import time

    best_val_loss = 1000000

    # initiate
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning']['rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=params['learning']['patience'], factor=params['learning']['decay'])
    data_loaders = get_data_loaders(params)
    model = model.to(device)

    print(f'Starting to train model: {params["model_name"]}')
    print('GPU available:', torch.cuda.is_available(), ' \nNumber of CPUs:', params['nbr_cpu'])

    training_start_time = time.time()
    for epoch in range(params['num_epochs']):
        print(f'\nEpoch {epoch+1}/{params["num_epochs"]}')
        epoch_start_time = time.time()

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            batch_loss = []
            batch_loss_segment = []
            batch_loss_percentage = []
            batch_percentage_error = []

            print('Phase:', phase)
            for batch in data_loaders[phase]:
                image_input = batch['image'].to(device)
                bitmap = batch['bitmap'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output, class_fraction = model(image_input)

                    segment_loss = criterion(output, bitmap)
                    ratio_loss = ratio_loss_function(class_fraction, bitmap)

                    if phase == 'train':
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
                batch_percentage_error.append(batch_seg_perc_error)
                batch_loss_segment.append(segment_loss.data.item())
                batch_loss_percentage.append(ratio_loss.item())
                batch_loss.append(segment_loss.data.item() + ratio_loss.item())

            # add perc error for every class in plot
            for class_name in batch_seg_perc_error.keys():
                writer.add_scalar(f'Percentage_error/{phase}/{class_name}',
                                  np.mean([b[class_name] for b in batch_percentage_error]),
                                  epoch)

            writer.add_scalar(f'Loss/{phase}', np.mean(batch_loss), epoch)
            writer.add_scalar(f'Loss/{phase}_segment', np.mean(batch_loss_segment), epoch)
            writer.add_scalar(f'Loss/{phase}_percentage', np.mean(batch_loss_percentage), epoch)

            if phase == 'val':
                epoch_val_loss = np.mean(batch_loss)
                print('Validation loss:', np.round(epoch_val_loss, 2))

                scheduler.step(epoch_val_loss)

                fig = plot.draw_class_bitmaps(mask=bitmap.cpu().numpy()[0],
                                              prediction=output_soft.cpu().numpy()[0],
                                              image=image_input.cpu().numpy()[0])
                writer.add_figure('CompareClasses', fig, epoch)

                if epoch_val_loss < best_val_loss:
                    print('This is the best model so far! Saving it!')
                    torch.save(model.state_dict(), params["path_to_model"])
                    best_val_loss = epoch_val_loss
                    # TODO: also save the params? Should include the transform names? Maybe also the loss?

                if epoch_val_loss > best_val_loss:
                    # if this happens X epochs in a row, abort?
                    print('Overfitting?')

            writer.flush()
        print(f'Epoch done, took {round((time.time() - training_start_time))/60} min')

    print(f'Training done! Best validation loss was {best_val_loss}. Saved to file {params["path_to_model"]}.'
          f'Took {round((time.time() - training_start_time)/60, 2)} min')


# TODO: Remove this?
# print(f'Testing :D')
# test_loss = []
# test_loss_segment = []
# test_loss_percentage = []
# test_percentage_error = []
#
# model.eval()
# with torch.no_grad():
#     for batch in test_loader:
#         image_input = batch['image'].to(device)
#         bitmap = batch['bitmap'].to(device)
#
#         output, class_fraction = model(image_input)
#         segment_loss = criterion(output, bitmap)
#         ratio_loss = ratio_loss_function(class_fraction, bitmap)
#
#         test_loss.append(segment_loss.data.item() + ratio_loss.item())
#         test_loss_segment.append(segment_loss.data.item())
#         test_loss_percentage.append(ratio_loss.item())
#
#         output_soft = F.softmax(output, dim=1)
#         # _, output_integer = output_soft.max(1)    # TODO: Evaluate nbr correct pixels instead?
#
#         segmentation_percentages = calculate_segmentation_percentages(output_soft)
#         batch_seg_perc_error, total_batch_seg_perc_error = \
#             calculate_segmentation_percentage_error(segmentation_percentages, batch['percentage'])
#
#         test_percentage_error.append(batch_seg_perc_error)
#
#
# print(f'Test loss was {np.mean(test_loss)}')
# for class_name in batch_seg_perc_error.keys():
#     print(f'Percentage error for {class_name}', np.mean([b[class_name] for b in test_percentage_error]))
