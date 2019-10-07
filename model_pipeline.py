

def train_model(job, writer, verbose=True):
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

    best_val_loss = 10000000000
    best_percentage_error = {}

    model = job['model'].float().to(device)
    class_weights = torch.tensor(job['class_weights']).float().to(device)

    # initiate
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=job['learning']['rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=job['learning']['patience'], factor=job['learning']['decay'])
    data_loaders = get_data_loaders(job)

    training_start_time = time.time()
    for epoch in range(job['num_epochs']):

        if verbose:
            print(f'\n\tEpoch {epoch+1}/{job["num_epochs"]}')

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

            if verbose:
                print('\tPhase:', phase)

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

                if verbose:
                    print('\tValidation loss:', np.round(epoch_val_loss, 2))

                scheduler.step(epoch_val_loss)

                fig = plot.draw_class_bitmaps(mask=bitmap.cpu().numpy()[0],
                                              prediction=output_soft.cpu().numpy()[0],
                                              image=image_input.cpu().numpy()[0])
                writer.add_figure('CompareClasses', fig, epoch)

                if epoch_val_loss < best_val_loss:
                    if verbose:
                        print('\tThis is the best model so far! Saving it!')
                    # torch.save(model.state_dict(), job["path_to_model"])
                    best_val_loss = epoch_val_loss
                    best_model_state = model.state_dict()
                    best_percentage_error = {class_name: np.mean([b[class_name] for b in batch_percentage_error])
                                             for class_name in batch_percentage_error[0].keys()}

                if epoch_val_loss > best_val_loss:
                    # if this happens X epochs in a row, abort?
                    if verbose:
                        print('\tOverfitting?')

            writer.flush()
        if verbose:
            print(f'\tEpoch done, took {round((time.time() - epoch_start_time)/60, 2)} min')

    run_time = round((time.time() - training_start_time)/60, 2)
    print(f'\tTraining done! Best validation loss was {round(best_val_loss, 2)} and took {run_time} min')

    return {
        'val_loss': best_val_loss,
        'percentage_error': best_percentage_error,
        'model_state': best_model_state,
        'run_time': run_time
    }


def create_jobs_to_run(sweep_name, base_params, models, learning_rates, class_weights, force_remake=False):
    import pickle, os

    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'

    if os.path.exists(jobs_spec_file_path) and force_remake is False:
        with open(jobs_spec_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        idx = 0
        all_jobs = {}
        for model in models:
            for lr in learning_rates:
                for cw in class_weights:
                    job = base_params.copy()

                    job['id'] = idx
                    job['model'] = model[0]
                    job['model_name'] = model[1]
                    job['learning']['rate'] = lr
                    job['class_weights'] = cw
                    job['status'] = None

                    all_jobs[idx] = job
                    idx += 1

        with open(jobs_spec_file_path, 'wb') as f:
            pickle.dump(all_jobs, f)

        return all_jobs


def execute_jobs(sweep_name, writer):
    import pickle, time
    from main import get_score_from_api
    from config import device

    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'
    sweep_start = time.time()

    print(f'Starting sweeping jobs called "{sweep_name}"')
    print('On device:', device)

    with open(jobs_spec_file_path, 'rb') as f:
        all_jobs = pickle.load(f)

    best_job_val_loss = 100000000
    jobs_to_run = [a for a in all_jobs.keys() if all_jobs[a]['status'] is None]
    for idx, job_id in enumerate(jobs_to_run):
        job = all_jobs[job_id]
        print(f'\nStarting job_id:{job_id} with model:{job["model_name"]}. \t {idx+1}/{len(jobs_to_run)}')

        # train model
        job['result'] = train_model(job, writer, verbose=False)

        # get result from API
        print('\tTesting against API')
        job['total_score'] = get_score_from_api(job, verbose=False)
        print('\tTest score:', job['total_score'])

        job['status'] = 'done'
        all_jobs[job_id] = job

        # update all jobs list if want to restart midway
        with open(jobs_spec_file_path, 'wb') as f:
            pickle.dump(all_jobs, f)

        if job['result']['val_loss'] < best_job_val_loss:
            best_model_name = job['model_name']
            best_job_id = job['id']

    print(f'Sweep done!\nBest model for {sweep_name} was {best_model_name} in job {best_job_id} '
          f'after {round((time.time() - sweep_start)/60, 2)}min')
