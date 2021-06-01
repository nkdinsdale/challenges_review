# Nicola Dinsdale 2020
# Functions for training and validating the model
########################################################################################################################
# Import dependencies
import numpy as np
import torch
from torch.autograd import Variable
########################################################################################################################

def train_normal(args, models, train_loader, optimizer, criterion, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models

    total_loss = 0

    encoder.train()
    regressor.train()

    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        if list(data.size())[0] == args.batch_size :
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
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)
            del loss
            del features

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))

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
            target = target.type(torch.LongTensor)

            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                x = regressor(features)

                loss = criterion(x, target)

                total_loss  += loss

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())
    del av_loss

    print('Validation set: Average Domain loss: {:.4f}\n'.format(av_loss_copy,  flush=True))

    return av_loss_copy, np.NaN
