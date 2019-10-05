

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
    best_percentage_error = {}

    model = model.to(device)
    class_weights = torch.tensor(params['class_weights']).to(device)

    # initiate
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning']['rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=params['learning']['patience'], factor=params['learning']['decay'])
    data_loaders = get_data_loaders(params)

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
                    best_percentage_error = {class_name: np.mean([b[class_name] for b in batch_percentage_error])
                                             for class_name in batch_percentage_error[0].keys()}

                if epoch_val_loss > best_val_loss:
                    # if this happens X epochs in a row, abort?
                    print('Overfitting?')

            writer.flush()
        print(f'Epoch done, took {round((time.time() - training_start_time)/60, 2)} min')

    run_time = round((time.time() - training_start_time)/60, 2)
    print(f'Training done! Best validation loss was {best_val_loss}. Saved to file {params["path_to_model"]}.'
          f'Took {run_time} min')

    return {
        'val_loss': best_val_loss,
        'percentage_error': best_percentage_error,
        'run_time': run_time
    }


def create_jobs_to_run(prefix='A', remake=False):
    from config import params_victor, jobs_spec_file_path
    import pickle, os
    from UnetModel import UNet

    if os.path.exists(jobs_spec_file_path) and remake is False:
        with open(jobs_spec_file_path, 'rb') as f:
            return pickle.load(f)

    learning_rates = [0.1, 0.2]
    class_weights = [
        [1, 1, 1, 1], [1, 2.7, 1.6, 3.5], [1, 2.0, 1.6, 5.5]
    ]
    models = [
        (UNet(3, 4).float(), 'UNet')
    ]

    idx = 0
    all_jobs = []
    for model in models:
        for lr in learning_rates:
            for cw in class_weights:
                job = params_victor.copy(deep=True)

                job['id'] = idx
                job['model'] = model[0]
                job['model_name'] = f'{prefix}{idx}_{model[1]}'
                job['learning']['rate'] = lr
                job['class_weights'] = cw
                job['status'] = None

                all_jobs.append(job)
                idx += 1

    with open(jobs_spec_file_path, 'wb') as f:
        pickle.dump(all_jobs, f)

    return all_jobs


def execute_parameter_sweep(writer):
    import pickle, json, time
    from config import jobs_spec_file_path
    import datetime

    sweep_start_time = datetime.datetime.today().strftime("%Y-%m-%d_%H%M")
    sweep_start = time.time()

    with open(jobs_spec_file_path, 'rb') as f:
        all_jobs = pickle.load(f)

    best_job_val_loss = 10000

    for idx, job in enumerate(all_jobs):
        if job['status'] is None:

            # Add date to model name
            job['model_name'] += f'_{sweep_start_time}'
            job['path_to_model'] = f'models/trained/{job["model_name"]}.pth'
            job['path_to_spec_file'] = f'models/trained/{job["model_name"]}.json'
            model = job['model']

            job['result'] = train_model(model, job, writer)
            job['status'] = 'done'
            all_jobs[idx] = job

            with open(job['path_to_spec_file'], 'w') as f:
                json.dump(job, f)

            with open(jobs_spec_file_path, 'wb') as f:
                pickle.dump(all_jobs, f)

            if job['result']['val_loss'] < best_job_val_loss:
                best_model_name = job['model_name']

    print(f'Sweep done!\nBest model was {best_model_name} after {round((time.time() - sweep_start)/60, 2)}')
