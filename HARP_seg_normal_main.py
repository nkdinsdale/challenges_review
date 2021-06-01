# Nicola Dinsdale 2020
# HARP segmentation to explore pruning with manual labels
########################################################################################################################
from models.unet_model_normal_dropout import UNet, Segmenter
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from collections import OrderedDict
from losses.dice_loss import dice_loss
from torch.autograd import Variable

import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_split_models_pruning_unet
import torch.optim as optim
import sys

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 1000
args.batch_size = 32
args.diff_model_flag = False
args.alpha = 100
args.patience = 100
args.learning_rate = 1e-3

cuda = torch.cuda.is_available()

LOAD_PATH_UNET = None
LOAD_PATH_SEGMENTER = None

PATH_UNET = 'harp_unet_dropout'
CHK_PATH_UNET = 'harp_unet_dropout_checkpoint'
PATH_SEGMENTER = 'harp_segmenter_dropout'
CHK_PATH_SEGMENTER = 'harp_segmenter_dropout_checkpoint'

LOSS_PATH = 'harp_losses'

########################################################################################################################
def update_dict(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        name = k[7:]  # Remove module.
        new_state_dict[name] = v
    return new_state_dict

def train_normal(args, models, train_loader, optimizer, criterion, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models

    total_loss = 0

    encoder.train()
    regressor.train()

    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        if list(data.size())[0] == args.batch_size:
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            x = regressor(features)
            loss = criterion(x, target)
            loss.backward()
            optimizer.step()

            total_loss += loss

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()), flush=True)
            del loss
            del features

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy, flush=True))

    return av_loss_copy, np.NaN

def val_normal(args, models, val_loader, criterion):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models

    encoder.eval()
    regressor.eval()

    total_loss = 0

    batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            batches += 1
            features = encoder(data)
            x = regressor(features)

            loss = criterion(x, target)

            total_loss += loss

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())
    del av_loss

    print('Validation set: Average Domain loss: {:.4f}\n'.format(av_loss_copy, flush=True))

    return av_loss_copy, np.NaN

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        print(m)
        torch.nn.init.xavier_uniform(m.weight.data)


########################################################################################################################
im_size = (64, 64, 64)
pth = 'HARP'

X = np.load(pth + '/X_harp.npy').astype(float)
X = np.reshape(X, (-1, 64, 64, 64, 1))
X = X / np.percentile(X, 99)
X = X - np.mean(X)

y = np.load(pth + '/y_harp.npy').astype(int).reshape(-1, 64, 64, 64, 1)
print(np.unique(y))
y_store = np.zeros((y.shape[0], 64, 64, 64, 2))
y_store[:, :, :, :, 0][y[:, :, :, :, 0] == 0] = 1
y_store[:, :, :, :, 1][y[:, :, :, :, 0] == 1] = 1
y = y_store

if args.channels_first:
    X = np.transpose(X, (0, 4, 1, 2, 3))
    y = np.transpose(y, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Data shape: ', X.shape)
    print('Labels shape: ', y.shape)


X, y = shuffle(X, y, random_state=0)  # Same seed everytime

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
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

unet = UNet(init_features=int(4))
segmenter = Segmenter(out_channels=2, init_features=int(4))
unet.apply(weights_init)
segmenter.apply(weights_init)

if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()

criterion = dice_loss()
criterion.cuda()

optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)

# Initalise the early stopping
early_stopping = EarlyStopping_split_models_pruning_unet(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

models = [unet, segmenter]

for epoch in range(epoch_reached, args.epochs + 1):
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

        torch.save(unet, PATH_UNET)
        torch.save(segmenter, PATH_SEGMENTER)

        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache










