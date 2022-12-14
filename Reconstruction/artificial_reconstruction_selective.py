import torchvision
import torchvision.transforms as transforms
import torch
from modeling.models.bethge import BethgeModel
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from modeling.train_utils import array_to_dataloader
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt
from analysis.Stat_utils import get_site_corr
device = 'cuda'

class reconstruct_CNN(nn.Module):
    def __init__(self, num_neuron):
        super().__init__()
        modules = []

        hidden_dims = [16, 64, 128, 64, 16]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               output_padding=2,
                               dilation=5),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self.layers = nn.Sequential(*modules)
        self.linear_input = nn.Linear(num_neuron, hidden_dims[0] * 4)

    def forward(self, x):
        x = self.linear_input(x)
        x = x.view(-1, 16, 2, 2)
        x = self.layers(x)
        x = self.final_layer(x)
        return x


class selected_rsp_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rsps, cifarset:Dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cifarset = cifarset
        self.rsps = rsps

    def __len__(self):
        return len(self.rsps)

    def __getitem__(self, idx):
        img, _ = self.cifarset.__getitem__(idx)
        rsp = torch.FloatTensor(self.rsps[idx])
        return img, rsp

if __name__ == '__main__':
    criterion = nn.functional.mse_loss
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(50),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Grayscale()]
    )

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    train_rsp = np.load('279_selected_rsp_cifar10_16.npy')
    test_rsp = np.load('279_selected_rsp_cifar10_test_16.npy')

    trainset_rsp = selected_rsp_dataset(train_rsp,trainset)
    testset_rsp = selected_rsp_dataset(test_rsp, testset)

    testloader = torch.utils.data.DataLoader(trainset_rsp, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(testset_rsp, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    prednet = reconstruct_CNN(train_rsp.shape[1])
    prednet.to(device)
    optimizer = torch.optim.Adam(prednet.parameters(), lr=0.005)
    losses = []
    accs = []

    bestloss = 200
    num_epochs = 100
    for e in tqdm(range(num_epochs)):
        train_losses = []
        prednet = prednet.train()
        for i, (x, y) in enumerate(trainloader):
            x = x.float().to(device)
            y = y.float().to(device)
            recon = prednet(y)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        losses.append(np.mean(train_losses))

        val_losses = []
        with torch.no_grad():
            prednet = prednet.eval()
            for i, (x, y) in enumerate(testloader):
                x = x.float().to(device)
                y = y.float().to(device)
                recon = prednet(y)
                loss = criterion(recon, x)

                val_losses.append(loss.item())
        avg_loss = np.mean(val_losses)
        accs.append(avg_loss)
        if avg_loss < bestloss:
            torch.save(prednet.state_dict(), "artificial_recon_model_selective_279_16")
            bestloss = avg_loss

        print(f'epoch {e} : train loss is {float(losses[-1])}')
        print(f'epoch {e} : val loss is   {float(accs[-1])}')