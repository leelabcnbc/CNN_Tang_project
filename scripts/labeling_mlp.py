
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.train_utils import train_loop_with_scheduler

PS_labels = np.load('../data/Processed_pattern_stimuli/labels.npy')

def simple_index (x):
    if x == 3:
        return 0
    else:
        return 1
def cut_ps_img (x):
    new_img = np.full((1,50,50), 0.54)
    new_img[:, 13:37, 13:37] = x[:, 13:37, 13:37]
    return new_img
PS_labels = [simple_index(x) for x in PS_labels]

PS_labels_all = np.concatenate((PS_labels, PS_labels, PS_labels, PS_labels), 0)

PS_set1 = np.load('../data/Processed_pattern_stimuli/crop_100_resize_1.npy')
PS_set2 = np.load('../data/Processed_pattern_stimuli/crop_100_resize_2.npy')
PS_set3 = np.load('../data/Processed_pattern_stimuli/crop_100_resize_3.npy')
PS_set4 = np.load('../data/Processed_pattern_stimuli/crop_100_resize_4.npy')
PS_imgs_all = np.concatenate((PS_set1, PS_set2, PS_set3, PS_set4), 0)


all_PS_sets = [PS_imgs_all,PS_set1, PS_set2, PS_set3, PS_set4]

all_PS_sets = [np.array([cut_ps_img(x) for x in set]) for set in all_PS_sets]

all_PS_labels = [PS_labels_all, PS_labels, PS_labels, PS_labels, PS_labels]

all_PS_sets = [np.reshape(p, (p.shape[0],1,50, 50)) for p in all_PS_sets]

all_PS_sets = [torch.tensor(p, dtype=torch.float) for p in all_PS_sets]
all_PS_labels = [torch.tensor(p, dtype=torch.long) for p in all_PS_labels]


num_out = 2

class label_model_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(43320, num_out)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.Linear(x)
        return x

batchsize = 100
for set_id, (set, labels) in enumerate(zip(all_PS_sets,all_PS_labels)):
    p = torch.randperm(set.shape[0])

    labels = labels[p]
    set = set[p]

    set_len = set.shape[0]
    test_len = round(set_len / 20)
    train_len = set_len-test_len

    train_set = set[:train_len]
    test_set = set[train_len:]

    train_labels = labels[:train_len]
    test_labels = labels[train_len:]

    dataset = TensorDataset(train_set, train_labels)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    test_dataset = TensorDataset(test_set,test_labels )
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = label_model_CNN()
    model = model.to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    network, losses, accs = train_loop_with_scheduler(loader, test_loader, model, optimizer,
            criterion, criterion, num_epochs, device=device,
            save_location = 'A:/school/College_Junior/research/CNN_Tang_project/saved_models/PS_labeling_model_CNN_CV_cut' + str(set_id))