### 041320 -- transfer learning on saved resnet34 activations
### following the approach in the Cadena 2019 paper

import os
import numpy as np
import torch
import torch.nn as nn

from functools import partial
from scipy.stats import pearsonr
from skimage.util import pad
from analysis.data_utils import get_all_neural_data, spike_counts, \
        sigma_noise, trial_average, DATA_DIR
from modeling.data_utils import train_val_test_split
from modeling.models.transfer_learner import TransferLearner
from modeling.losses import poisson_loss
from modeling.train_utils import array_to_dataloader, simple_train_loop
from modeling import LOG_DIR, SAVE_DIR

# set some constants
DATASET = 'both'
CORR_THRESHOLD = 0.7
LOG_FILE = open(LOG_DIR + f'{__file__}'[:-3] + '.txt', 'w')

# get the neural data (constant over all blocks)
neural = get_all_neural_data(corr_threshold=CORR_THRESHOLD,
        elecs=False)
neural = spike_counts(neural, start=540, end=640)
noises = sigma_noise(neural)
neural = trial_average(neural)

train_idx, val_idx, test_idx = train_val_test_split(total_size=5850,
        train_size=3800, val_size=1000, deterministic=True)

train_y = neural[train_idx]
val_y = neural[val_idx]
test_y = neural[test_idx]

# set up training params
# params found from the Cadena paper, probably a reasonable
# start, but should maybe be tuned further
sparse_weight = 0.1
smooth_weight = 1.0
group_weight = 0.01

lr = 3e-7
num_epochs = 30 # tends to reach the minimum by then
print_every = 2
log_file = open(LOG_DIR + f'{__file__}'[:-3] + '.txt', 'w')

# now start looping over the blocks
for block in range(16):
    # load in the data
    tang_fname = DATA_DIR + f'tang/resnet34/block{block}.npy'
    googim_fname = DATA_DIR + f'google-imagenet/resnet34/block{block}.npy'

    tang_acts = np.load(tang_fname)
    googim_acts = np.load(googim_fname)

    # googim images are 251x251, leading to slightly different act sizes
    # but padding one pixel won't throw off RFs much
    # since it's just gray aperture there
    if tang_acts.shape[3] < googim_acts.shape[3]:
        tang_acts = pad(tang_acts, ((0, 0), (0, 0), (0, 1), (0, 1)),
                mode='edge')
    elif tang_acts.shape[3] > googim_acts.shape[3]:
        googim_acts = pad(googim_acts, ((0, 0), (0, 0), (0, 1), (0, 1)),
                mode='edge')

    print(tang_acts.shape)
    print(googim_acts.shape)
    inputs = np.concatenate([tang_acts, googim_acts],
            axis=0)
    # normalize to zero mean, unit variance for each channel
    # to avoid unfair sparseness penalties
    inputs = (inputs - np.expand_dims(inputs.mean((0,2,3)), (0,2,3))) / \
        np.expand_dims(inputs.std((0,2,3)), (0,2,3))
    # and split
    train_x = inputs[train_idx]
    val_x = inputs[val_idx]
    test_x = inputs[test_idx]
    # then put into dataloader form
    train_loader = array_to_dataloader(train_x, train_y, batch_size=200)
    val_loader = array_to_dataloader(val_x, val_y, batch_size=250)

    # define model
    model = TransferLearner(inputs[0].shape, train_y.shape[1])
    # and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define loss (based on model params)
    def transfer_loss(x, y, model):
        base_loss = poisson_loss(x, y)

        weight = model.weight
        c, h, w, n = weight.shape

        sparse_loss = sparse_weight * weight.abs().sum((1,2,3)).mean()
        
        # laplacian
        smooth_kernel = torch.tensor([[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0]]]])
        smooth_kernel = torch.cat(n * [smooth_kernel], dim=0)
        smooth_loss = smooth_weight * (nn.functional.conv2d(
                weight.permute(0, 3, 1, 2), smooth_kernel, groups=n) ** 2).sum(0).sqrt().mean()

        group_loss = group_weight * (weight ** 2).sum((1, 2)).sqrt().sum(0).mean()

        return base_loss + sparse_loss + smooth_loss + group_loss 

    # and a validation loss to work with the loop
    def n_poisson_loss(p, y, n):
        return poisson_loss(p, y)

    # define early stopping method
    def stopper(train_losses, val_losses):
        patience = 10
        if len(val_losses) >= patience:
            last_few = val_losses[-patience:]
            diffs = np.diff(last_few)

            if all(diffs >= 0) or last_few[-1] > 3.0 or last_few[-1] > last_few[0]:
                return False

        return True


    # now train
    trained, train_losses, val_losses = simple_train_loop(train_loader, val_loader,
            model, optimizer, transfer_loss, n_poisson_loss, num_epochs, print_every,
            stop_criterion=stopper, device='cpu', log_file=log_file)

    # compute some performance metrics
    trained.eval()
    with torch.no_grad():
        #train_x = torch.tensor(train_x).float()
        #train_preds = trained(train_x).numpy()

        val_x = torch.tensor(val_x).float()
        val_preds = trained(val_x).numpy()

        test_x = torch.tensor(test_x).float()
        test_preds = trained(test_x).numpy()

    #train_corrs = [pearsonr(train_preds[:,i].flatten(), train_y[:,i].flatten())[0]
            #for i in range(train_y.shape[1])]
    val_corrs = [pearsonr(val_preds[:,i].flatten(), val_y[:,i].flatten())[0]
            for i in range(val_y.shape[1])]
    test_corrs = [pearsonr(test_preds[:,i].flatten(), test_y[:,i].flatten())[0]
            for i in range(test_y.shape[1])]

    #print(np.mean(train_corrs))
    print(np.mean(val_corrs))
    print(np.mean(test_corrs))

    # and save the trained model
    save_dir = SAVE_DIR + f'transfer_learning_resnet/block{block}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(trained, save_dir + 'model.pt')
    #np.save(save_dir + 'train_preds', train_preds)
    #np.save(save_dir + 'val_preds', val_preds)
    np.save(save_dir + 'test_preds', test_preds)
    np.savez(save_dir + 'loss_curves', train_losses, val_losses)
