
import torch
import numpy as np
import pickle as pkl
import sys
import os

from functools import partial
import scipy.stats

from modeling.losses import *
from modeling.models.bethge import BethgeModel
from modeling.models.utils import num_params
from modeling.train_utils import array_to_dataloader, simple_train_loop, train_loop_with_scheduler

# general things
DATASET = 'both'
CORR_THRESHOLD = 0.7

# get and setup the data
downsample = 4
num_samples = 49000


def random_permute(set_x, set_y):
    p = np.random.permutation(num_samples)
    x_new = np.copy(set_x)
    y_new = np.copy(set_y)
    for i in range(num_samples):
        x_new[i] = set_x[p[i]]
        y_new[i] = set_y[p[i]]
    return x_new, y_new

site = 'm2s1'

train_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_'+site+'.npy')
val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_'+site+'.npy')
train_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/trainRsp_'+site+'.npy')
val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_'+site+'.npy')

if site == 'm2s1' or site == 'm2s2':
    slicea = list(range(5000))
    sliceb = list(range(10000,train_x.shape[0]))
    slice = slicea + sliceb
    train_x = train_x[slice]
    train_y = train_y[slice]

train_x =  np.transpose(train_x, (0, 3, 1, 2))
val_x =  np.transpose(val_x, (0, 3, 1, 2))
# set up network/training params
channels = 256
num_layers = 9
input_size = 50
first_k = 9
later_k = 3
pool_size = 2
factorized = True
num_maps = 1
output_size = val_y.shape[1]


lr = 1e-3
scale = 8e-4
smooth = 3e-6

# run a few models, not just one
num_seeds = 1
for sd in range(num_seeds):
    # for saving
    key = f'c{channels}_l{num_layers}_i{input_size}_o{output_size}_fk{first_k}_lk{later_k}_p{pool_size}_f{factorized}_n{num_maps}__lr{lr}_sc{scale}_sm{smooth}___sd{sd}'

    num_epochs = 50
    print_every = 2


    # define functions for use in the training loop
    def validation_loss(p, y, n):
        return corr_loss(p, y, corr_portion=1, mae_portion=1)


    def maskcnn_loss(p, y, n, scale=8e-5, smooth=3e-6):
        mse = corr_loss(p, y, corr_portion=1, mae_portion=1)

        readout_sparsity = 0
        for i in range(len(n.fc[0].bank)):
            spatial_map_flat = n.fc[0].bank[i].weight_spatial.view(
                n.fc[0].bank[i].weight_spatial.size(0), -1)
            feature_map_flat = n.fc[0].bank[i].weight_feature.view(
                n.fc[0].bank[i].weight_feature.size(0), -1)

            readout_sparsity += scale * torch.mean(
                torch.sum(torch.abs(spatial_map_flat), 1) *
                torch.sum(torch.abs(feature_map_flat), 1))

        readout_sparsity /= len(n.fc[0].bank)

        kern_smoothness = maskcnn_loss_v1_kernel_smoothness(
            [n.conv[0][0]], [smooth], torch.device('cuda'))

        return mse + readout_sparsity + kern_smoothness


    # lock in the hyperparams outside of the training loop
    maskcnn_loss = partial(maskcnn_loss, scale=scale, smooth=smooth)


    def stopper(train_losses, val_losses):
        patience = 10
        if len(val_losses) >= max(patience, 100):
            last_few = val_losses[-patience:]
            diffs = np.diff(last_few)

            if all(diffs >= 0) or last_few[-1] > 3.0:
                return False

        return True


    num_laps = 1
    for i in range(num_laps):
        cur_x = train_x[:round(num_samples*((i+1)/num_laps))]
        cur_y = train_y[:round(num_samples*((i+1)/num_laps))]
        print('length of set:' + str(len(cur_x)))
        # set up model and training parameters
        net = BethgeModel(channels=256, num_layers=9, input_size=input_size,
                          output_size=output_size, first_k=first_k, later_k=later_k,
                          input_channels=1, pool_size=pool_size, factorized=True,
                          num_maps=num_maps).cuda()

        train_loader = array_to_dataloader(cur_x, cur_y, batch_size=300, shuffle=True)
        val_loader = array_to_dataloader(val_x, val_y, batch_size=300)

        print(f'model has {num_params(net)} params')
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # now train, for three stages (learning rates)
        trained, first_losses, first_accs = train_loop_with_scheduler(train_loader, val_loader, net,
                                                                      optimizer, maskcnn_loss, validation_loss,
                                                                      num_epochs, print_every,
                                                                      stop_criterion=stopper, device='cuda',
                                                                      save_location= site + "_" + str(i) + "_model_mae_corr_version_"+ str(sd), loss_require_net=True)

