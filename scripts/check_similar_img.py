from train_torch import ImageDataset
from train_torch import net_one_neuron
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

nb_validation_samples = 1000
y_all_train = np.load('../Rsp.npy')
y_all_val = np.load('../valRsp.npy')
batch_size = 1
num_neurons = 299
x_train = np.load('../train_x.npy')
x_val = np.load('../val_x.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])

net.to(device)
Imageset = ImageDataset(x_train, y_all_train)
loader = DataLoader(Imageset, batch_size=batch_size, shuffle=True)
valset = ImageDataset(x_val, y_all_val)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

net.load_state_dict(torch.load('model_test_corr_mae'))

with torch.no_grad():
    net.eval()
    prediction = []
    actual = []

    (target_x,target_y) = valset.__getitem__(50)

    target_x = target_x.to(device)
    target_x = torch.reshape(target_x,(1,1,50,50))
    target_p = np.stack([subnet(target_x).cpu().numpy() for subnet in net])

    all_dist = []

    for batch_num, (x, y) in enumerate(tqdm(val_loader)):

        x, y = x.to(device), y.to(device)
        outputs = np.stack([subnet(x).cpu().numpy() for subnet in net])
        dist = np.linalg.norm(outputs - target_p)
        all_dist.append((batch_num,dist))

    sorted_dist = sorted(all_dist, key=lambda x: x[1])

    for i in range(10):
        (x, _) = valset.__getitem__(sorted_dist[i][0])
        img = np.reshape(x.numpy(), (50, 50))
        plt.imsave(fname=f"similars/{i}.jpg", arr=img)