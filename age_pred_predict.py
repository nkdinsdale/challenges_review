# Nicola Dinsdale 2020
# Normal predict file
########################################################################################################################
from models.age_predictor import Regressor, Encoder
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from utils import Args, EarlyStopping_unlearning, load_pretrained_model
from losses.confusion_loss import confusion_loss
from losses.DANN_loss import DANN_loss
import torch.optim as optim
from train_utils import unlearn_predict
from torch.autograd import Variable


import sys

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 200
args.batch_size = 1
args.diff_model_flag = False
args.alpha = 1
args.patience = 50
args.learning_rate = 1e-4

LOAD_PATH_ENCODER = '/gpfs0/well/win/users/ndinsdale/unlearning/weights/normal_model_encoder_split_checkpoint_whitehall_oasis_biobank'
LOAD_PATH_REGRESSOR = '/gpfs0/well/win/users/ndinsdale/unlearning/weights/normal_model_regressor_split_checkpoint_whitehall_oasis_biobank'

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda Available', flush=True)

########################################################################################################################
########################################################################################################################
im_size = (128, 128, 32)

X_test_biobank = np.random.rand(1, 1, 128, 128, 32)
X_test_biobank = X_test_biobank / np.percentile(X_test_biobank, 99)
X_test_biobank = X_test_biobank - np.mean(X_test_biobank)
y_test_biobank = np.array(0).reshape(-1, 1)
print('Creating datasets and dataloaders')
b_test_dataset = numpy_dataset(X_test_biobank, y_test_biobank)
#o_test_dataset = numpy_dataset(X_test_oasis, y_test_oasis)

b_test_dataloader = DataLoader(b_test_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
#o_test_dataloader = DataLoader(o_test_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

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
    regressor_dict = regressor.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))

    encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))
    regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))

criterion = nn.MSELoss()
criterion.cuda()

optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-5)

models = [encoder, regressor]

print('Predict Biobank')

age_results = []
age_true = []
embeddings = []
encoder.eval()
regressor.eval()

with torch.no_grad():
    for data, target in b_test_dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if list(data.size())[0] == args.batch_size:
            age_true.append(target.detach().cpu().numpy())
            features = encoder(data)
            embeddings.append(features.detach().cpu().numpy())
            age_preds = regressor(features)
            print(age_preds)
            age_results.append(age_preds.detach().cpu().numpy())
age_results = np.array(age_results)
age_results_cp = np.copy(age_results)
del age_results
age_true = np.array(age_true)
age_true_cp = np.copy(age_true)
del age_true
embeddings = np.array(embeddings)
embeddings_cp = np.copy(embeddings)
del embeddings

'''
np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/biobank_age_preds', age_results_cp)
np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/biobank_age_true', age_true_cp)
np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/biobank_embeddings', embeddings_cp)

print('Predict Oasis')

age_results = []
age_true = []
embeddings = []
encoder.eval()
regressor.eval()

with torch.no_grad():
    for data, target in o_test_dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if list(data.size())[0] == args.batch_size:
            age_true.append(target.detach().cpu().numpy())
            features = encoder(data)
            embeddings.append(features.detach().cpu().numpy())
            age_preds = regressor(features)
            age_results.append(age_preds.detach().cpu().numpy())
age_results = np.array(age_results)
age_results_cp = np.copy(age_results)
del age_results
age_true = np.array(age_true)
age_true_cp = np.copy(age_true)
del age_true
embeddings = np.array(embeddings)
embeddings_cp = np.copy(embeddings)
del embeddings

np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/oasis_age_preds', age_results_cp)
np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/oasis_age_true', age_true_cp)
np.save('/gpfs0/well/win/users/ndinsdale/unlearning/perceptual_loss/oasis_embeddings', embeddings_cp)

'''

