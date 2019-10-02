import os
from dataprep import split_data_to_train_val_test

# split_data_to_train_val_test(raw_base_path='data_raw/Training_dataset',
#                              new_base_path='data', val_ratio=0.3, test_ratio=0.0)

if os.path.exists('models') is False:
    os.mkdir('models')

if os.path.exists('models/trained') is False:
    os.mkdir('models/trained')

# if os.path.exists('runs') is True:
#     shutil.rmtree('runs')

if os.path.exists('credentials.json') is False:
    print('credentials file missing')
