import torch.nn as nn
class rsp_translator(nn.Module):
    def __init__(self, num_neuron, num_latent):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_neuron, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(512, num_latent)

    def forward(self, x):
        x = self.layers(x)
        x = self.Linear(x)
        return x