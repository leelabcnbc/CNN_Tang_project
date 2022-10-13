import numpy as np
import torch
from torch.utils import data
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def array_to_dataloader(x, y, batch_size=128, shuffle=False):
    """
    Takes x and y as data and labels; makes dataloaders of them.
    """
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x), 
            torch.FloatTensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)

    return dataloader

def simple_train_loop(train_loader, val_loader, network, optimizer,
        criterion, val_criterion, num_epochs, print_every=2,
        stop_criterion=None, device='cpu', log_file=sys.stdout,
        clipping=False):
    """
    Basic scaffolding for training.
    """
    network = network.to(device)
    losses = []
    accs = []

    bestloss = 200

    for e in tqdm(range(num_epochs)):

        train_losses = []
        network = network.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            preds = network(x)
            loss = criterion(preds, y, network)
            noreg_loss = val_criterion(preds, y, network)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(noreg_loss.item())
        losses.append(np.mean(train_losses))

        val_losses = []
        with torch.no_grad():
            network = network.eval()
            for i, (x, y) in enumerate(val_loader):
                x = x.float().to(device)
                y = y.float().to(device)
                preds = network(x)
                loss = val_criterion(preds, y, network)

                val_losses.append(loss.item())
        avg_loss = np.mean(val_losses)
        accs.append(avg_loss)
        if avg_loss < bestloss:
            torch.save(network.state_dict(), "shared_core_256_9_model")
            bestloss = avg_loss

        if e % print_every == 1: # don't check the first epoch
            print(f'epoch {e} : train loss is {float(losses[-1])}',
                    file=log_file)
            print(f'epoch {e} : val loss is   {float(accs[-1])}',
                    file=log_file)
            log_file.flush()

        # early stopping
        if stop_criterion is not None:
            continue_flag = stop_criterion(losses, accs)
        else:
            continue_flag = True

        if not continue_flag:
            break

    return network, losses, accs

def train_loop_with_scheduler(train_loader, val_loader, network, optimizer,
        criterion, val_criterion, num_epochs, print_every=1,
        stop_criterion=None, device='cpu', log_file=sys.stdout,
        save_location = "", loss_require_net = False):
    """
    Basic scaffolding for training.
    """
    network = network.to(device)
    losses = []
    accs = []

    bestloss = 200
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=- 1)
    for e in tqdm(range(num_epochs)):
        train_losses = []
        network = network.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            preds = network(x)
            if loss_require_net:
                loss = criterion(preds, y, network)
                noreg_loss = val_criterion(preds, y, network)
            else:
                loss = criterion(preds, y)
                noreg_loss = val_criterion(preds, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(noreg_loss.item())
        losses.append(np.mean(train_losses))
        scheduler.step()
        val_losses = []
        with torch.no_grad():
            network = network.eval()
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                preds = network(x)
                if loss_require_net:
                    loss = val_criterion(preds, y, network)
                else:
                    loss = val_criterion(preds, y)

                val_losses.append(loss.item())
        avg_loss = np.mean(val_losses)
        accs.append(avg_loss)
        if avg_loss < bestloss:
            torch.save(network.state_dict(), save_location)
            bestloss = avg_loss

        if e % print_every == 0:
            print(f'epoch {e} : train loss is {float(losses[-1])}',
                    file=log_file)
            print(f'epoch {e} : val loss is   {float(accs[-1])}',
                    file=log_file)
            log_file.flush()

        # early stopping
        if stop_criterion is not None:
            continue_flag = stop_criterion(losses, accs)
        else:
            continue_flag = True

        if not continue_flag:
            break

    return network, losses, accs
