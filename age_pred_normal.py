# Nicola Dinsdale 2020
# Script to train the original model
########################################################################################################################
from models.age_predictor import DomainPredictor, Regressor, Encoder
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_split_models
import torch.optim as optim
from train_utils_abide_segmentation import train_normal, val_normal
import sys
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 150
args.batch_size = 16
args.diff_model_flag = False
args.alpha = 100
args.patience = 25

cuda = torch.cuda.is_available()

LOAD_PATH_ENCODER = None
LOAD_PATH_REGRESSOR = None

PATH_ENCODER = 'age_pred_encoder'
CHK_PATH_ENCODER = 'age_pred_encoder_checkpoint'
PATH_REGRESSOR = 'age_pred_regressor'
CHK_PATH_REGRESSOR = 'age_pred_regressor_checkpoint'
LOSS_PATH = 'age_pred_losses'

########################################################################################################################
im_size = (128, 128, 32)

# Load in the data
X = np.load('X_train.npy')
y = np.load('y_train.npy').reshape(-1, 1).astype(float)

print('Data shape: ', X.shape, flush=True)

if args.channels_first:
    X = np.transpose(X, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Data shape: ', X.shape)

X, y = shuffle(X, y, random_state=0)
proportion = int(args.train_val_prop * len(X))
X_train = X[:proportion, :, :, :, :]
X_val = X[proportion:, :, :, :, :]
y_train = y[:proportion]
y_val = y[proportion:]

print('Data splits')
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

print('Creating datasets and dataloaders')
train_dataset = numpy_dataset(X_train, y_train)
val_dataset = numpy_dataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

# Load the model
encoder = Encoder()
regressor = Regressor()

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()

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


criterion = nn.MSELoss()
criterion.cuda()

optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=args.learning_rate)

# Initalise the early stopping
early_stopping = EarlyStopping_split_models(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

models = [encoder, regressor]

for epoch in range(epoch_reached, args.epochs+1):
    print('Epoch ', epoch, '/', args.epochs, flush=True)
    loss, _ = train_normal(args, models, train_dataloader, optimizer, criterion, epoch)

    val_loss, _ = val_normal(args, models, val_dataloader, criterion)
    loss_store.append([loss, val_loss])
    np.save(LOSS_PATH, np.array(loss_store))

    # Decide whether the model should stop training or not
    early_stopping(val_loss, models, epoch, optimizer, loss, [CHK_PATH_ENCODER, CHK_PATH_REGRESSOR])

    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')

    if epoch == args.epochs:
        print('Finished Training', flush=True)
        print('Saving the model', flush=True)

        # Save the model in such a way that we can continue training later
        torch.save(encoder.state_dict(), PATH_ENCODER)
        torch.save(regressor.state_dict(), PATH_REGRESSOR)

        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache
