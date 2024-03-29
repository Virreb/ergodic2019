

def train_percentage_model(job, writer, verbose=True):
    from dataprep import get_data_loaders
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    from help_functions import calculate_segmentation_percentage_error, \
        correct_mask_bitmaps_for_crop, ratio_loss_function
    from config import device, CLASS_ORDER
    import time

    best_percentage_loss = 10000000000
    best_percentage_error = {}

    model = job['model'].float().to(device)

    # initiate
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

            batch_loss_percentage = []
            batch_percentage_error = []

            if verbose:
                print('\tPhase:', phase)

            for batch in data_loaders[phase]:
                image_input = batch['image'].to(device)
                bitmap = batch['bitmap'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    class_fraction = model(image_input)
                    ratio_loss = ratio_loss_function(class_fraction, bitmap)

                    if phase == 'train':
                        ratio_loss.backward(retain_graph=True)
                        optimizer.step()

                # Correct bitmaps to crop to get percentages to evaluate to
                corrected_bitmaps = correct_mask_bitmaps_for_crop(bitmap)

                class_fraction_list = list()
                for i in range(class_fraction.shape[0]):
                    tmp_dict = {}
                    for j, class_name in enumerate(CLASS_ORDER):
                        tmp_dict[class_name] = class_fraction[i, j].cpu().item()
                    class_fraction_list.append(tmp_dict)

                # calulate percentage error
                batch_seg_perc_error, total_batch_seg_perc_error = \
                    calculate_segmentation_percentage_error(class_fraction_list, corrected_bitmaps)

                # save values for evaluating
                batch_percentage_error.append(batch_seg_perc_error)
                batch_loss_percentage.append(ratio_loss.item())

            # add perc error for every class in plot
            for class_name in batch_seg_perc_error.keys():
                writer.add_scalar(f'Percentage_error/{phase}/{class_name}',
                                  np.mean([b[class_name] for b in batch_percentage_error]),
                                  epoch)

            writer.add_scalar(f'Loss/{phase}_percentage', np.mean(batch_loss_percentage), epoch)

            if phase == 'val':
                epoch_val_loss = np.mean(batch_loss_percentage)

                if verbose:
                    print('\tValidation loss:', np.round(epoch_val_loss, 2))

                scheduler.step(epoch_val_loss)

                if epoch_val_loss < best_percentage_loss:
                    if verbose:
                        print('\tThis is the best model so far! Saving it!')
                    torch.save(model.state_dict(),
                               f'models/trained/{job["sweep_name"]}_{job["model_name"]}_{job["id"]}.pth')
                    best_percentage_loss = epoch_val_loss
                    best_model_state = model.state_dict()
                    best_percentage_error = {class_name: np.mean([b[class_name] for b in batch_percentage_error])
                                             for class_name in batch_percentage_error[0].keys()}

                if epoch_val_loss > best_percentage_loss:
                    # if this happens X epochs in a row, abort?
                    if verbose:
                        print('\tOverfitting?')

            writer.flush()
        if verbose:
            print(f'\tEpoch done, took {round((time.time() - epoch_start_time)/60, 2)} min')

    run_time = round((time.time() - training_start_time)/60, 2)
    print(f'\tTraining done! Best validation loss was {round(best_percentage_loss, 2)} and took {run_time} min')

    return {
        'val_loss': best_percentage_loss,
        'percentage_error': best_percentage_error,
        'model_state': best_model_state,
        'run_time': run_time
    }


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
    from main import get_score_from_api

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
                    ratio_loss = ratio_loss_function(class_fraction, bitmap)*job['perc_loss_weight']

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
                    torch.save(model.state_dict(),
                               f'models/trained/{job["sweep_name"]}_{job["model_name"]}_{job["id"]}.pth')
                    best_val_loss = epoch_val_loss
                    best_model_state = model.state_dict()
                    best_percentage_error = {class_name: np.mean([b[class_name] for b in batch_percentage_error])
                                             for class_name in batch_percentage_error[0].keys()}

                if epoch_val_loss > best_val_loss:
                    # if this happens X epochs in a row, abort?
                    if verbose:
                        print('\tOverfitting?')

            writer.flush()

        # get score from API
        get_score_from_api(model=model, model_name=job['model_name'])

        if verbose:
            print(f'\tEpoch done, took {round((time.time() - epoch_start_time)/60, 2)} min')

    run_time = round((time.time() - training_start_time)/60, 2)
    print(f'\tTraining done! Best validation loss was {round(best_val_loss, 2)} and took {run_time} min')

    return {
        'val_loss': best_val_loss,
        'percentage_error': best_percentage_error,
        'model_state': best_model_state,
        'last_model_state': model.state_dict(),
        'run_time': run_time
    }


def create_jobs_to_run(sweep_name, base_params, models, learning_rates, class_weights, perc_loss_weights,
                       force_remake=False):
    import pickle, os

    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'

    if os.path.exists(jobs_spec_file_path) and force_remake is False:
        with open(jobs_spec_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        idx = 0
        all_jobs = {}
        for model in models:
            model_name = model[1]

            if model_name.startswith('perc'):   # doesnt use percentage weight
                tmp_perc_loss_weights = [1]
            else:
                tmp_perc_loss_weights = perc_loss_weights

            for lr in learning_rates:
                for cw in class_weights:
                    for plw in tmp_perc_loss_weights:
                        job = base_params.copy()

                        job['id'] = idx
                        job['sweep_name'] = sweep_name
                        job['model'] = model[0]
                        job['model_name'] = model_name
                        job['learning']['rate'] = lr
                        job['class_weights'] = cw
                        job['perc_loss_weight'] = plw
                        job['status'] = None

                        all_jobs[idx] = job
                        idx += 1

        with open(jobs_spec_file_path, 'wb') as f:
            pickle.dump(all_jobs, f)

        return all_jobs


def execute_jobs(sweep_name):
    import pickle, time, datetime
    from main import get_score_from_api
    from config import device
    from torch.utils.tensorboard import SummaryWriter
    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'
    sweep_start = time.time()

    print(f'Starting sweeping jobs called "{sweep_name}"')
    print('On device:', device)

    with open(jobs_spec_file_path, 'rb') as f:
        all_jobs = pickle.load(f)

    best_job_val_loss = 100000000
    best_model_name, best_job_id = None, None
    jobs_to_run = [a for a in all_jobs.keys() if all_jobs[a]['status'] is None]
    for idx, job_id in enumerate(jobs_to_run):
        job = all_jobs[job_id]
        print(f'\nStarting job_id {job_id} with model "{job["model_name"]}" \t {idx+1}/{len(jobs_to_run)}')

        current_time = datetime.datetime.today().strftime('%d-%b-%H-%M')
        log_name = f'runs/{sweep_name}_{job_id}-{job["model_name"]}_{current_time}'
        writer = SummaryWriter(log_dir=log_name)

        # train model
        if job['model_name'].startswith('perc'):
            job['result'] = train_percentage_model(job, writer, verbose=True)
        else:
            job['result'] = train_model(job, writer, verbose=True)

        # update all jobs list if want to restart midway
        with open(jobs_spec_file_path, 'wb') as f:
            pickle.dump(all_jobs, f)

        # get result from API
        print('\tTesting against API')
        job['result']['total_score'] = get_score_from_api(job, verbose=False)
        print('\tTest score:', job['result']['total_score'])

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


def print_sweep_overview(sweep_name):
    import pickle, time

    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'

    print(f'Loading sweeping jobs called "{sweep_name}"')

    with open(jobs_spec_file_path, 'rb') as f:
        all_jobs = pickle.load(f)

    for job_id in all_jobs:
        job = all_jobs[job_id]
        print(f'Job {job_id} with model {job["model_name"]} got a val loss of {job["result"]["val_loss"]} and ')
    #          f'a test score of {job["result"]["total_score"]}')


def load_job_from_sweep(sweep_name, idx):
    import pickle, time

    jobs_spec_file_path = f'jobs/{sweep_name}.pkl'
    with open(jobs_spec_file_path, 'rb') as f:
        all_jobs = pickle.load(f)

    return all_jobs[idx]

if __name__ == '__main__':
    print_sweep_overview('google_tuesday')
    job = load_job_from_sweep('google_tuesday', 0)
    print(job.keys())
    
