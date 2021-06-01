# Nicola Dinsdale 2020
# Predict file for the 3d segmentation with the ABIDE data
########################################################################################################################
from models.unet_model import UNet, Segmenter
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from utils import Args
from losses.dice_loss import dice_loss
from torch.autograd import Variable
########################################################################################################################
args = Args()
args.channels_first = True
args.epochs = 1000
args.batch_size = 1
args.diff_model_flag = False
args.alpha = 100
args.patience = 25
args.learning_rate = 1e-3

cuda = torch.cuda.is_available()
LOAD_PATH_UNET = 'UM_unet_checkpoint'
LOAD_PATH_SEGMENTER = 'UM_segmenter_checkpoint'
########################################################################################################################
def update_dict(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        name = k[7:]    # Remove module.
        new_state_dict[name] = v
    return new_state_dict

def predict(args, models, test_loader):
    cuda = torch.cuda.is_available()

    seg_pred = []
    seg_true = []
    input = []

    [unet, segmenter] = models
    unet.eval()
    segmenter.eval()

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if list(data.size())[0] == args.batch_size:

                input.append(data.detach().cpu().numpy())
                seg_true.append(target.detach().cpu().numpy())

                features = unet(data)
                seg = segmenter(features)
                seg_pred.append(seg.detach().cpu().numpy())


    input = np.array(input)
    seg_pred = np.array(seg_pred)
    seg_pred_cp = np.copy(seg_pred)
    del seg_pred
    seg_true = np.array(seg_true)
    seg_true_cp = np.copy(seg_true)

    return seg_pred_cp, seg_true_cp, input

def dice3d(ground_truth, prediction):
    # Calculate the 3D dice coefficient of the ground truth and the prediction
    ground_truth = ground_truth > 0.5  # Binarize volume
    prediction = prediction > 0.5  # Binarize volume
    epsilon = 1e-5  # Small value to prevent dividing by zero
    true_positive = np.sum(np.multiply(ground_truth, prediction))
    false_positive = np.sum(np.multiply(ground_truth == 0, prediction))
    false_negative = np.sum(np.multiply(ground_truth, prediction == 0))
    dice3d_coeff = 2*true_positive / \
        (2*true_positive + false_positive + false_negative + epsilon)
    return dice3d_coeff

########################################################################################################################
im_size = (128, 128, 128)

pth = 'ABIDE_segmentation/'
X_um = np.load(pth+'X_UM_seg_test.npy').astype(float).reshape(-1, 128, 128, 128 ,1)
print(X_um.shape)
X_um = X_um / np.percentile(X_um, 99)
X_um = X_um - np.mean(X_um)

y_um = np.load(pth+'y_UM_seg_test.npy').astype(int).reshape(-1, 128, 128, 128 ,1)
print(y_um.shape)
y_store = np.zeros((y_um.shape[0], 128, 128, 128, 4))
y_store[:, :, :, :, 0][y_um[:, :, :, :, 0] == 0] = 1
y_store[:, :, :, :, 1][y_um[:, :, :, :, 0] == 1] = 1
y_store[:, :, :, :, 2][y_um[:, :, :, :, 0] == 2] = 1
y_store[:, :, :, :, 3][y_um[:, :, :, :, 0] == 3] = 1
y_um = y_store

if args.channels_first:
    print('CHANNELS FIRST')
    X_um = np.transpose(X_um, (0, 4, 1, 2, 3))
    y_um = np.transpose(y_um, (0, 4, 1, 2, 3))

um_test_dataset = numpy_dataset(X_um, y_um)
um_test_dataloader = DataLoader(um_test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

# Load the model
encoder = UNet()
regressor = Segmenter()

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()

# Make everything parallelisable
encoder = nn.DataParallel(encoder)
regressor = nn.DataParallel(regressor)

if LOAD_PATH_UNET:
    print('Loading Weights')
    encoder_dict = encoder.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_UNET)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
    encoder.load_state_dict(torch.load(LOAD_PATH_UNET))

if LOAD_PATH_SEGMENTER:
    regressor_dict = regressor.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_SEGMENTER)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
    regressor.load_state_dict(torch.load(LOAD_PATH_SEGMENTER))

criteron = dice_loss()
criteron.cuda()

models = [encoder, regressor]
pred, true, input = predict(args, models, um_test_dataloader)
