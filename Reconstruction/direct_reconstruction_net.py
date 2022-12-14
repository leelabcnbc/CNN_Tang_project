import torch.nn as nn
class reconstruct_CNN(nn.Module):
    def __init__(self, num_neuron):
        super().__init__()
        modules = []

        hidden_dims = [256, 512, 1024, 512, 256, 1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding= 1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.layers = nn.Sequential(*modules)
        self.linear_input = nn.Linear(num_neuron, hidden_dims[0] * 4)

    def forward(self, x):
        x = self.linear_input(x)
        x = x.view(-1, 256, 2, 2)
        x = self.layers(x)
        return x