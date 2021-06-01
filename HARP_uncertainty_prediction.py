# Nicola Dinsdale 2021
# Predict on the HarP data with uncertainty
########################################################################################################################
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
from losses.dice_loss import dice_loss
from torch.autograd import Variable
import numpy as np
from utils import Args
import torch.optim as optim

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 1000
args.batch_size = 1
args.diff_model_flag = False
args.alpha = 100
args.patience = 25
args.learning_rate = 1e-3

cuda = torch.cuda.is_available()

LOAD_PATH_UNET = 'harp_unet_dropout_checkpoint'
LOAD_PATH_SEGMENTER = 'harp_segmenter_dropout_checkpoint'

SAVE_PATH = 'harp_uncertainty_results'
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

    [unet, segmenter] = models
    unet.eval()
    segmenter.eval()
    enable_dropout(unet)

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if list(data.size())[0] == args.batch_size:

                seg_true.append(target.detach().cpu().numpy())

                features = unet(data)
                seg = segmenter(features)
                seg_pred.append(seg.detach().cpu().numpy())


    seg_pred = np.array(seg_pred)
    seg_pred_cp = np.copy(seg_pred)
    del seg_pred
    seg_true = np.array(seg_true)
    seg_true_cp = np.copy(seg_true)

    return seg_pred_cp, seg_true_cp

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

def enable_dropout(m):
  for m in unet.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

########################################################################################################################
im_size = (64, 64, 64)
pth = 'HARP'
X = np.load(pth + '/X_harp_test.npy').astype(float)
X = np.reshape(X, (-1, 64, 64, 64, 1))
X = X / np.percentile(X, 99)
X = X - np.mean(X)

np.save(SAVE_PATH + '_raw', X)

y = np.load(pth + '/y_harp_test.npy').astype(int).reshape(-1, 64, 64, 64, 1)
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

print('Data splits')
print(X.shape, y.shape)

print('Creating datasets and dataloaders')
test_dataset = numpy_dataset(X, y)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

unet = torch.load(LOAD_PATH_UNET, map_location=lambda storage, loc: storage)
segmenter = torch.load(LOAD_PATH_SEGMENTER, map_location=lambda storage, loc: storage)

if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()

criterion = dice_loss()
criterion.cuda()

optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)
models = [unet, segmenter]

seg_true = None
preds = []
iterations = 1000
for i in range(0, iterations):
    seg_pred, seg_true = predict(args, models, test_dataloader)

    seg_pred = np.argmax(seg_pred.squeeze().reshape(-1, 2, 64, 64, 64), axis=1)
    seg_true = np.argmax(seg_true.squeeze().reshape(-1, 2, 64, 64, 64), axis=1)
    preds.append(seg_pred)
    print(np.array(preds).shape)

preds = np.array(preds)
var = np.var(preds, axis=0)
mean = np.mean(preds, axis=0)

np.save(SAVE_PATH + '_var', var)
np.save(SAVE_PATH + '_mean', mean)

np.save(SAVE_PATH + '_true', seg_true)