# Nicola Dinsdale 2020
# Look at the features being used for segmentation
########################################################################################################################
from models.age_predictor import Regressor, Encoder
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from utils import Args
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 200
args.batch_size = 16
args.diff_model_flag = False
args.alpha = 1
args.patience = 50
args.learning_rate = 1e-4

LOAD_PATH_ENCODER = 'age_pred_encoder_checkpoint'
LOAD_PATH_REGRESSOR =  'age_pred_regressor_checkpoint'

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda Available', flush=True)

########################################################################################################################
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook
########################################################################################################################
im_size = (128, 128, 32)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print('Data shape: ', X_test.shape, flush=True)
if args.channels_first:
    X_test = np.transpose(X_test, (0, 4, 1, 2, 3))
    print('Data shape: ', X_test.shape)

o_test_dataset = numpy_dataset(X_test, y_test)
o_test_dataloader = DataLoader(o_test_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

# Load the models
encoder = Encoder()
regressor = Regressor()


if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()

# Make everything parallelisable
encoder = nn.DataParallel(encoder)
regressor = nn.DataParallel(regressor)

if LOAD_PATH_ENCODER:
    print('Loading Weights')
    encoder_dict = encoder.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_ENCODER)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
    encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))

if LOAD_PATH_REGRESSOR:
    regressor_dict = regressor.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
    regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))

print(encoder.module.feature[22])
encoder.module.feature[22].register_forward_hook(get_activation('f_relu_4_1'))

features_store = []


for data in o_test_dataloader:
    images, target, domain_target = data

    images, target = images.cuda(), target.cuda()
    images, target = Variable(images), Variable(target)

    features = encoder(images)

    act = activation['f_relu_4_1'].squeeze()
    print(act.size())

    act = act.detach().cpu().numpy()

    features_store.append(act)

features_store = np.array(features_store)

np.save('encoder_features_relu_4_1.npy', features_store)