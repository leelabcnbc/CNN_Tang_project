#%%
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.models.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader
from scipy.stats import spearmanr


nb_validation_samples = 1000
nb_training_samples = 49000
val_y = np.load('../data/Processed_Tang_data/valRsp.npy')
val_x = np.load('../data/Processed_Tang_data/val_x.npy')
train_x = np.load('../data/Processed_Tang_data/train_x.npy')
train_y = np.load('../data/Processed_Tang_data/Rsp.npy')
batch_size = 2048
num_neurons = 299

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channels = 256
num_layers = 9
input_size = 50
output_size = 299
first_k = 9
later_k = 3
pool_size = 2
factorized = True

num_maps = 1

net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                  output_size=output_size, first_k=first_k, later_k=later_k,
                  input_channels=1, pool_size=pool_size, factorized=True,
                  num_maps=num_maps).cuda()

net.to(device)
net.load_state_dict(torch.load('../saved_models/shared_core_256_9_model'))

fc_layer = net.fc[0].bank[0]
#%%
