### assorted loss functions
import torch
from audtorch.metrics.functional import pearsonr
import torch.nn as nn

def poisson_loss(preds, real):
    """
    Poisson loss, appropriate for neural firing rates.
    Average loss for a batch (assumed to be the first dimension).
    """
    eps = 1e-6 # for numerical stability
    return torch.mean(preds - real * torch.log(preds + eps))

def corr_loss(out,y):
    mae = torch.nn.L1Loss()
    loss = torch.mean(-pearsonr(out, y, batch_first=False)) + 0.1 * mae(out, y)
    return loss

def mse_w(preds,real):
    criterion = torch.nn.MSELoss(reduction='none')
    y_exp = torch.exp(real)
    loss = criterion(preds, real)
    loss_w = torch.mean(loss * y_exp)
    return loss_w

def fev_explained(preds, real, sigma_noise=0):
    """
    Fraction of explainable variance explained, for each
    of a set of neurons (along dimension 1).
    If sigma_noise is 0, then it is equivalent to fraction
    of variance explained (e.g. R^2).
    """
    rss = torch.mean((preds - real) ** 2, dim=0)
    var_y = torch.var(real, dim=0)
    return 1 - ((rss - sigma_noise) / (var_y - sigma_noise))

# regularization losses, adapted from Yimeng's code
def maskcnn_loss_v1_kernel_smoothness(module_list,
                                      scale_list, device):
    kernel = torch.tensor(
        [[[[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]]],
        dtype=torch.float32).to(device)

    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()

        w_this = w_this.view(c_out * c_in, 1, h, w)
        w_this_conved = torch.nn.functional.conv2d(w_this, kernel, padding=1).view(c_out, c_in, -1)
        w_this = w_this.view(c_out, c_in, -1)

        sum_to_add = s * torch.sum(
            torch.sum(w_this_conved ** 2, -1) / torch.sum(w_this ** 2, -1))
        sum_list.append(sum_to_add)
    return sum(sum_list)

def maskcnn_loss_v1_kernel_group_sparsity(module_list,
                                          scale_list):
    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()
        w_this = w_this.view(c_out, c_in, h * w)
        sum_to_add = s * torch.sum(torch.sqrt(torch.sum(w_this ** 2, -1)))
        sum_list.append(sum_to_add)
    return sum(sum_list)
