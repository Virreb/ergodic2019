import os

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params_victor = {
    'learning_rate': 0.0002,
    'num_epochs': 5,
    'nbr_cpu': os.cpu_count() - 1,
    'device': device,
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
    'num_epochs': 4,
    'nbr_cpu': os.cpu_count() - 4,
    'device': device,
    'image_size': {
        'train': (256, 256),
        'val': (1024, 1024),
        'test': (1024, 1024)
    },
    'batch_size': {
        'train': 2,
        'val': 2,
        'test': 2,
    },
}
