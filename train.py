import datetime
from math import sqrt
from config import params_isak, params_victor
from torch.utils.tensorboard import SummaryWriter
from UnetModel import UNet
import model_pipeline
from GCN import GCN

# start tensorboard with tensorboard --logdir='runs'
# watch -n 0.5 nvidia-smi

# TODO: Test transfer learning
# TODO: Utilize regularization'
# TODO: Create function that calls the API for new pictures and calculates errors? As a test.
# TODO: Framework for testing parameter comb.
# TODO: Merge model and params selection. Create list of dicts to run over? Save every dict to disk also?


# choose model
model = UNet(3, 4).float()
model_name = f'UNet'
# model_name = f'unet_{datetime.datetime.today().strftime("%Y-%m-%d_%H%M")}.pth'

model_2 = GCN(4)
for tmp_layer in [model_2.layer0, model_2.layer1, model_2.layer2, model_2.layer3, model_2.layer4]:
    for param in tmp_layer.parameters():
        param.requires_grad = False

model_2_name = f'GCN'
# model_2_name = f'gcn_{datetime.datetime.today().strftime("%Y-%m-%d_%H%M")}.pth'

# get parameters''
params = params_victor
params['model_name'] = model_2_name
params['path_to_model'] = f'models/trained/{model_2_name}.pth'
params['class_weights'] = [1, 7.3**0.25, 2.5**0.25, 12.3**0.25]

# init tensorboard
writer = SummaryWriter()

# train model
model_pipeline.train_model(model_2, params, writer)

