import os
import torch

CLASS_ORDER = ['nothing', 'water', 'building', 'road']
jobs_spec_file_path = 'jobs_spec_file.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params_victor = {
    'learning': {
        'rate': 0.1,
        'patience': 2,
        'decay': 0.2
    },
    'num_epochs': 150,
    'nbr_cpu': 14,
    'device': device,
    'image_size': {
        'train': (512, 512),
        'val': (1024, 1024),
        'test': (1024, 1024)
    },
    'batch_size': {
        'UNet': {
            'train': 4,
            'val': 2,
            'test': 2,
        },
        'GCN': {
            'train': 32,
            'val': 12,
            'test': 12,
        },
    }
}
params_isak = {
    'learning': {
        'rate': 0.1,
        'patience': 2,
        'decay': 0.2
    },
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
