from config import params_isak, params_victor, device
from torch.utils.tensorboard import SummaryWriter
from UnetModel import UNet
import model_pipeline
from GCN import GCN
from Deeplab import DeeplabFork
from perc_model import get_resnet_101

# start tensorboard with tensorboard --logdir='runs'
# watch -n 0.5 nvidia-smi

# TODO: Utilize regularization?
# TODO: Create model that only outputs percentage
# TODO: Test one model per class
# TODO: Implement logic to load sweep and get best model and write stats

# init tensorboard
writer = SummaryWriter()

base_params = {
    'learning': {
        'rate': 0.1,
        'patience': 2,
        'decay': 0.2
    },
    'num_epochs': 60,
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
        'perc_resnet101': {
            'train': 32,
            'val': 12,
            'test': 12,
        },
        'deeplab': {
            'train': 24,
            'val': 8,
            'test': 8,
        },
    }
}
# base_params = params_isak

# init models to sweep
model_gcn = GCN(4)
for tmp_layer in [model_gcn.layer0, model_gcn.layer1, model_gcn.layer2, model_gcn.layer3, model_gcn.layer4]:
    for param in tmp_layer.parameters():
        param.requires_grad = False

deeplab = DeeplabFork()

models = [
#     (UNet(3, 4), 'UNet'),
#    (model_gcn, 'GCN'),
    (deeplab, 'deeplab'),
    # (get_resnet_101(4), 'perc_resnet101'),
]

# set parameters to sweep
learning_rates = [0.1, 0.2]
class_weights = [
    # [1, 1, 1, 1]
    [1, 1, 1, 1], [1, 7.3**0.25, 2.5**0.25, 12.3**0.75]
]

sweep_name = 'google_tuesday'
model_pipeline.create_jobs_to_run(sweep_name, base_params=base_params, models=models,
                                  learning_rates=learning_rates, class_weights=class_weights,
                                  force_remake=True)
model_pipeline.execute_jobs(sweep_name, writer)
