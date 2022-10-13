from PyTorch_VAE.models import VanillaVAE
import numpy as np
from tqdm import tqdm
from modeling.train_utils import array_to_dataloader
import torch
from torch.nn import functional as F

model = VanillaVAE(1, latent_dim=128, hidden_dims=None)
all_train_x = []
all_val_x = []
sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2','m3s1']
sites = ['m2s1']
for site in sites:
    all_train_x.append(
        np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_' + site + '.npy'))
    all_val_x.append(np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy'))
train_x = np.concatenate(all_train_x)
val_x = np.concatenate(all_val_x)

train_x = np.transpose(train_x, (0, 3, 1, 2))
val_x = np.transpose(val_x, (0, 3, 1, 2))
train_set = torch.utils.data.TensorDataset(torch.FloatTensor(train_x))
train_loader = torch.utils.data.DataLoader(train_set, 2048, shuffle=True)
val_set = torch.utils.data.TensorDataset(torch.FloatTensor(train_x))
val_loader = torch.utils.data.DataLoader(val_set, 2048, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

device = 'cuda'

network = model.to(device)
losses = []
accs = []

bestloss = 200
num_epochs = 100
criterion = model.loss_function
for e in tqdm(range(num_epochs)):

    train_losses = []
    network = network.train()
    for i, x in enumerate(train_loader):
        x = x[0].float().to(device)
        [recon, input, mu, log_var] = network(x)
        loss = criterion(recon, input, mu, log_var, M_N=0.00025)
        noreg_loss = loss['Reconstruction_Loss']

        optimizer.zero_grad()
        loss_value = loss["loss"]
        loss_value.backward()
        optimizer.step()

        train_losses.append(noreg_loss.item())
    losses.append(np.mean(train_losses))

    val_losses = []
    with torch.no_grad():
        network = network.eval()
        for i, x in enumerate(val_loader):
            x = x[0].float().to(device)
            [recon, input, mu, log_var] = network(x)
            loss = criterion(recon, input, mu, log_var, M_N=0.00025)
            loss = loss['Reconstruction_Loss']

            val_losses.append(loss.item())
    avg_loss = np.mean(val_losses)
    accs.append(avg_loss)
    if avg_loss < bestloss:
        torch.save(network.state_dict(), "Image_vae_m2s1")
        bestloss = avg_loss

    print(f'epoch {e} : train loss is {float(losses[-1])}')
    print(f'epoch {e} : val loss is   {float(accs[-1])}')
