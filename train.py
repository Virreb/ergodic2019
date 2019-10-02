import datetime
from config import params_isak, params_victor
from torch.utils.tensorboard import SummaryWriter
from UnetModel import UNet
import model_pipeline

# start tensorboard with tensorboard --logdir='runs'
# watch -n 0.5 nvidia-smi

# TODO: Test transfer learning
# TODO: Difference learning weights for different classes
# TODO: Utilize regularization
# TODO: Create function that calls the API for new pictures and calculates errors? As a test.
# TODO: Framework for testing parameter comb.
# TODO: Merge model and params selection. Create list of dicts to run over? Save every dict to disk also?


# choose model
model = UNet(3, 4).float()
model_name = f'unet_{datetime.datetime.today().strftime("%Y-%m-%d_%H%M")}.pth'

# get parameters
params = params_victor
params['model_name'] = model_name
params['path_to_model'] = f'models/trained/{model_name}'
class_weights = [1, 7.3, 2.5, 12.3]

# init tensorboard
writer = SummaryWriter()

# train model
model_pipeline.train_model(model, params, writer)

