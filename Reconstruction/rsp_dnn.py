import numpy as np
from tqdm import tqdm
from modeling.train_utils import array_to_dataloader

from PyTorch_VAE.models import VanillaVAE
from modeling.train_utils import array_to_dataloader
import torch
from torch.nn import functional as F
from reconstruct_translator import rsp_translator
vae = VanillaVAE(1, latent_dim=128, hidden_dims=None)
vae.load_state_dict(torch.load("Image_vae_m2s1"))
vae = vae.eval()


site = 'm2s1'
train_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_'+site+'.npy')
val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_'+site+'.npy')
train_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/trainRsp_'+site+'.npy')
val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_'+site+'.npy')
model = rsp_translator(299, 128)
train_x = np.transpose(train_x, (0, 3, 1, 2))
val_x = np.transpose(val_x, (0, 3, 1, 2))
train_loader = array_to_dataloader(train_x, train_y, batch_size=2048, shuffle=True)
val_loader = array_to_dataloader(val_x, val_y, batch_size=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = F.mse_loss
device = 'cuda'

network = model.to(device)
vae = vae.to(device)
losses = []
accs = []

bestloss = 200
num_epochs = 100
for e in tqdm(range(num_epochs)):

    train_losses = []
    network = network.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.float().to(device)
        y = y.float().to(device)
        preds = network(y)
        recon = vae.decode(preds)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    losses.append(np.mean(train_losses))

    val_losses = []
    with torch.no_grad():
        network = network.eval()
        for i, (x, y) in enumerate(val_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            preds = network(y)
            recon = vae.decode(preds)
            loss = criterion(recon, x)

            val_losses.append(loss.item())
    avg_loss = np.mean(val_losses)
    accs.append(avg_loss)
    if avg_loss < bestloss:
        torch.save(network.state_dict(), "rsp_translator")
        bestloss = avg_loss

    print(f'epoch {e} : train loss is {float(losses[-1])}')
    print(f'epoch {e} : val loss is   {float(accs[-1])}')