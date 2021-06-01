# Nicola Dinsdale 2020
# ABIDE 3D Segmentation for reviewers without unlearning
########################################################################################################################
from models.unet_model import UNet, Segmenter
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_split_models
from losses.dice_loss import dice_loss

import torch.optim as optim
from train_utils_abide_segmentation import val_normal, train_normal
import sys
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 300
args.batch_size = 4
args.diff_model_flag = False
args.alpha = 50
args.patience = 25

cuda = torch.cuda.is_available()

LOAD_PATH_UNET = None
LOAD_PATH_SEGMENTER = None

PATH_UNET = 'UM_unet'
CHK_PATH_UNET = 'UM_unet_checkpoint'
PATH_SEGMENTER = 'UM_segmenter'
CHK_PATH_SEGMENTER = 'UM_segmenter_checkpoint'

LOSS_PATH = 'UM_losses'

########################################################################################################################
im_size = (128, 128, 128)
pth = 'ABIDE_segmentation'
X = np.zeros((1, 128, 128, 128, 1))
y = np.zeros((1, 128, 128, 128, 4))

sites = np.array(['um'])

for site in sites:
    x_data = np.load(pth + '/X_' + site + '_seg_train.npy').astype(float).reshape(-1, 128, 128, 128, 1)
    x_data = np.reshape(x_data, (-1, 128, 128, 128, 1))
    x_data = x_data / np.percentile(x_data, 99)
    x_data = x_data - np.mean(x_data)
    y_data = np.load(pth + '/y_' + site + '_seg_train.npy').astype(int).reshape(-1, 128, 128, 128, 1)


    print(np.unique(y_data))
    y_store = np.zeros((y_data.shape[0], 128, 128, 128, 4))

    print(y_store.shape)

    y_store[:, :, :, :, 0][y_data[:, :, :, :, 0] == 0] = 1
    y_store[:, :, :, :, 1][y_data[:, :, :, :, 0] == 1] = 1
    y_store[:, :, :, :, 2][y_data[:, :, :, :, 0] == 2] = 1
    y_store[:, :, :, :, 3][y_data[:, :, :, :, 0] == 3] = 1
    y_data = y_store
    print(np.unique(y_data))

    print(site)
    print(len(x_data))

    X = np.append(X, x_data, axis=0)
    y = np.append(y, y_data, axis=0)

    print(X.shape, y.shape)

if args.channels_first:
    X = np.transpose(X, (0, 4, 1, 2, 3))
    y = np.transpose(y, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Data shape: ', X.shape)
    print('Labels shape: ', y.shape)

X, y = shuffle(X, y, random_state=0)      #Same seed everytime
proportion = int(args.train_val_prop * len(X))
X_train = X[:proportion]
X_val = X[proportion:]
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
unet = UNet()
segmenter = Segmenter()

if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()

# Make everything parallelisable
unet = nn.DataParallel(unet)
segmenter = nn.DataParallel(segmenter)

if LOAD_PATH_UNET:
    print('Loading Weights')
    encoder_dict = unet.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_UNET)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
    unet.load_state_dict(torch.load(LOAD_PATH_UNET))

if LOAD_PATH_SEGMENTER:
    regressor_dict = segmenter.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_SEGMENTER)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
    segmenter.load_state_dict(torch.load(LOAD_PATH_SEGMENTER))

criterion = dice_loss()
if cuda:
    criterion = criterion.cuda()

optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)

# Initalise the early stopping
early_stopping = EarlyStopping_split_models(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

models = [unet, segmenter]

for epoch in range(epoch_reached, args.epochs+1):
    print('Epoch ', epoch, '/', args.epochs, flush=True)
    loss, _ = train_normal(args, models, train_dataloader, optimizer, criterion, epoch)

    val_loss, _ = val_normal(args, models, val_dataloader, criterion)
    loss_store.append([loss, val_loss])
    np.save(LOSS_PATH, np.array(loss_store))

    # Decide whether the model should stop training or not
    early_stopping(val_loss, models, epoch, optimizer, loss, [CHK_PATH_UNET, CHK_PATH_SEGMENTER])

    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')

    if epoch == args.epochs:
        print('Finished Training', flush=True)
        print('Saving the model', flush=True)

        # Save the model in such a way that we can continue training later
        torch.save(unet.state_dict(), PATH_UNET)
        torch.save(segmenter.state_dict(), PATH_SEGMENTER)

        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache
