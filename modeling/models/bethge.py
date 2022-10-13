import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FactorizedLinear(nn.Module):
    def __init__(self, in_chan, map_size, out_size):
        super().__init__()
        self.in_chan = in_chan
        self.map_size = map_size
        self.out_size = out_size

        # simplified from Yimeng's code
        self.weight_spatial = nn.Parameter(
                torch.empty(self.out_size, self.map_size, self.map_size))
        self.weight_feature = nn.Parameter(
                torch.empty(self.out_size, self.in_chan))
        self.bias = nn.Parameter(
                torch.empty(self.out_size))

        self.reset_parameters()

    def reset_parameters(self):
        # taken from nn.Linear and nn._ConvNd
        nn.init.kaiming_uniform_(self.weight_spatial, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_feature, a=np.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_feature)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        n, c, h, w = x.shape

        # also taken/simplified from Yimeng's code
        spatial_view = self.weight_spatial.view(
                self.out_size, 1, self.map_size, self.map_size)
        feature_view = self.weight_feature.view(
                self.out_size, self.in_chan, 1, 1)

        weight_view = (spatial_view * feature_view).view(
                self.out_size, self.in_chan * self.map_size ** 2)

        x_view = x.view(n, c * h * w)

        return F.linear(x_view, weight_view, self.bias)


class multiFactorizedLinear(nn.Module):
    '''
    Combine multiple factorized linear layers, allowing
    multiple different feature weightings at different
    locations.
    '''
    def __init__(self, in_chan, map_size, out_size,
            num_maps=1):
        '''
        parameters are same as a normal linear layer
        '''
        super().__init__()
        self.in_chan = in_chan
        self.map_size = map_size
        self.out_size = out_size
        self.num_maps = num_maps

        self.bank = nn.ModuleList([FactorizedLinear(in_chan, map_size, out_size) for i in range(num_maps)])

    def forward(self, x):
        outs = [lin(x) for lin in self.bank]
        #print(outs)
        out = torch.mean(torch.stack(outs), 0)

        return out

class BethgeModel(nn.Module):
    """
    Predict a whole group of neurons' responses.
    Hopefully comparable to Yimeng's maskcnn.
    """
    def __init__(self, channels, num_layers, input_size, output_size,
            first_k=9, later_k=3, input_channels=3, pool_size=2,
            factorized=False, num_maps=1, final_nonlin=True):
        super().__init__()

        self.factorized = factorized
        self.final_nonlin = final_nonlin

        self.norm = nn.BatchNorm2d(num_features=input_channels,
                affine=True, track_running_stats=True)

        kernels = [later_k] * (num_layers - 1)
        self.conv = []
        self.conv.append(nn.Sequential(*[
            nn.Conv2d(input_channels, channels, first_k,
                bias=False),
            nn.Softplus(),
            nn.BatchNorm2d(channels, affine=True, track_running_stats=True),
            ]))

        for k in kernels:
            self.conv.append(nn.Sequential(*[
                nn.Conv2d(channels, channels, later_k, padding = later_k // 2,
                    bias=False),
                nn.Softplus(),
                nn.BatchNorm2d(channels, affine=True, track_running_stats=True),
                ]))
        self.conv = nn.Sequential(*self.conv)

        self.pool = nn.AvgPool2d(pool_size, pool_size)

        size_by_now = (input_size - first_k + 1) // pool_size
        if not factorized:
            self.fc = nn.Linear(size_by_now ** 2 * channels, output_size)
        else:
            self.fc = multiFactorizedLinear(channels, size_by_now, output_size, num_maps)
        if final_nonlin:
            self.fc = nn.Sequential(*[
                self.fc,
                nn.Softplus(),
                ])
        else:
            self.fc = nn.Sequential(*[
                self.fc,
                ])

        self.initialize_std()

    def initialize_std(self):
        '''
        Initialize all parameteres in a standard fashion.
        '''
        for c in self.conv:
            for l in c:
                #print(type(l))
                if type(l) is nn.Conv2d:
                    nn.init.normal_(l.weight, std=0.01)
                #elif type(l) is DeformableConv:
                    #print('deformable conv initialized')
                    #nn.init.normal_(l.conv.weight, std=0.01)
                    #nn.init.normal_(l.offset.weight, std=0.01)
                elif type(l) is nn.BatchNorm2d:
                    nn.init.zeros_(l.bias)
                    nn.init.constant_(l.weight, 1.0)

        if type(self.fc[0]) is multiFactorizedLinear:
            for b in self.fc[0].bank:
                nn.init.normal_(b.weight_spatial, std=0.01)
                #TODO: initialization is zero
                #nn.init.zeros_(b.weight_spatial)
                nn.init.normal_(b.weight_feature, std=0.01)
                nn.init.zeros_(b.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.pool(x)
        if not self.factorized:
            x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class StdRecModel(nn.Module):
    '''
    Bethge model with a vanilla recurrent layer stuck on the end.
    '''
    def __init__(self, channels, num_layers, input_size, output_size,
            time_steps, first_k=9, later_k=3, input_channels=3, pool_size=2,
            factorized=False, num_maps=1, rc_weights=None, deformable=False):
        super().__init__()

        self.time_steps = time_steps
        self.hidden_size = output_size

        self.feed_forward = BethgeModel(channels=channels, num_layers=num_layers,
                input_size = input_size, output_size = output_size, first_k = first_k,
                later_k = later_k, input_channels = input_channels, pool_size = pool_size,
                factorized=factorized, num_maps = num_maps, deformable = deformable)

        #self.recurrent = nn.RNNCell(output_size, self.hidden_size, bias=True, #?
                #nonlinearity = 'tanh') # would be nicer to have softplus... maybe add it?
        self.recurrent = nn.Parameter(
                torch.empty(output_size, output_size))

        self.rec_act = nn.Softplus()

        self.initialize_std()

        if rc_weights is not None:
            self.recurrent.data = rc_weights.data

    def initialize_std(self):
        #nn.init.normal_(self.recurrent.weight_hh, std=0.005)
        #nn.init.normal_(self.recurrent.weight_ih, std=0.005)
        #nn.init.zeros_(self.recurrent.bias_hh)
        #nn.init.zeros_(self.recurrent.bias_ih)
        nn.init.normal_(self.recurrent, std=0.01)


    def forward(self, x):
        x = self.feed_forward(x)

        for _ in range(self.time_steps):
            x = x + self.rec_act(nn.functional.linear(x, self.recurrent))

        return x

class SkipModel(nn.Module):
    """
    Predict a whole group of neurons' responses.
    Hopefully comparable to Yimeng's maskcnn.
    """
    def __init__(self, channels, num_layers, input_size, output_size,
            first_k=9, later_k=3, input_channels=3, pool_size=2,
            factorized=False, num_maps=1, deformable=False):
        super().__init__()

        self.factorized = factorized

        self.norm = nn.InstanceNorm2d(num_features=input_channels, affine=False)

        kernels = [later_k] * (num_layers - 1)
        self.conv = []
        self.conv.append(nn.Sequential(*[
            nn.Conv2d(input_channels, channels, first_k,
                bias=False),
            nn.Softplus(),
            nn.InstanceNorm2d(channels, affine=False),
            ]))

        for k in kernels:
            #if deformable:
                #self.conv.append(nn.Sequential(*[
                    #DeformableConv(channels, channels, later_k, padding = later_k // 2,
                        #bias=False),
                    #nn.Softplus(),
                    #nn.BatchNorm2d(channels, affine=True, track_running_stats=True),
                    #]))
            self.conv.append(nn.Sequential(*[
                nn.Conv2d(channels, channels, later_k, padding = later_k // 2,
                    bias=False),
                nn.Softplus(),
                nn.InstanceNorm2d(channels, affine=False),
                ]))
        self.conv = nn.ModuleList(*self.conv)

        self.pool = nn.AvgPool2d(pool_size, pool_size)

        size_by_now = (input_size - first_k + 1) - 2 * (len(self.conv))
        if not factorized:
            self.fc = nn.Linear(size_by_now ** 2 * channels, output_size)
        else:
            self.fc = multiFactorizedLinear(channels, size_by_now, output_size, num_maps)
        self.fc = nn.Sequential(*[
            self.fc,
            nn.Softplus(),
            ])

        self.initialize_std()

    def initialize_std(self):
        '''
        Initialize all parameteres in a standard fashion.
        '''
        for c in self.conv:
            for l in c:
                print(type(l))
                if type(l) is nn.Conv2d:
                    nn.init.normal_(l.weight, std=0.01)
                elif type(l) is nn.BatchNorm2d:
                    nn.init.zeros_(l.bias)
                    nn.init.constant_(l.weight, 1.0)

        if type(self.fc[0]) is multiFactorizedLinear:
            for b in self.fc[0].bank:
                nn.init.normal_(b.weight_spatial, std=0.01)
                nn.init.normal_(b.weight_feature, std=0.01)
                nn.init.zeros_(b.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.pool(x)
        if not self.factorized:
            x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class KBethgeModel(nn.Module):
    """
    Bethge model with Kriegeskorte-style convolutional
    recurrence in all layers but the first.
    """
    def __init__(self, channels, num_layers, input_size, output_size,
            iterations,
            first_k=9, later_k=3, rec_k=3, input_channels=3, pool_size=2,
            factorized=False, num_maps=1, final_nonlin=True):
        super().__init__()

        self.factorized = factorized
        self.final_nonlin = final_nonlin
        self.iterations = iterations
        self.output_size = output_size

        self.norm = nn.BatchNorm2d(num_features=input_channels,
                affine=True, track_running_stats=True)

        kernels = [later_k] * (num_layers - 1)
        self.first_layer = nn.Sequential(*[
            nn.Conv2d(input_channels, channels, first_k,
                bias=False),
            nn.Softplus(),
            nn.BatchNorm2d(channels, affine=True, track_running_stats=True),
            ])

        self.conv = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.int_nonlin = nn.Softplus()
        self.r_conv = nn.ModuleList()
        for k in kernels:
            self.conv.append(nn.Conv2d(channels, channels, later_k, 
                padding = later_k // 2, bias=False))
            self.r_conv.append(nn.Conv2d(channels, channels, rec_k,
                padding = rec_k // 2, bias=False))
            self.norms.append(nn.BatchNorm2d(channels, affine=True, 
                track_running_stats=True))

        self.pool = nn.MaxPool2d(pool_size, pool_size)

        size_by_now = (input_size - first_k + 1) // pool_size
        if not factorized:
            self.fc = nn.Linear(size_by_now ** 2 * channels, output_size)
        else:
            self.fc = multiFactorizedLinear(channels, size_by_now, output_size, num_maps)
        if final_nonlin:
            self.fc = nn.Sequential(*[
                self.fc,
                nn.Softplus(),
                ])
        else:
            self.fc = nn.Sequential(*[
                self.fc,
                ])

        self.initialize_std()

    def initialize_std(self):
        '''
        Initialize all parameteres in a standard fashion.
        '''
        for l in self.conv:
            nn.init.normal_(l.weight, std=0.01)
        for l in self.r_conv:
            nn.init.normal_(l.weight, std=0.01)

        for l in self.norms:
            nn.init.zeros_(l.bias)
            nn.init.constant_(l.weight, 1.0)

        if type(self.fc[0]) is multiFactorizedLinear:
            for b in self.fc[0].bank:
                nn.init.normal_(b.weight_spatial, std=0.01)
                nn.init.normal_(b.weight_feature, std=0.01)
                nn.init.zeros_(b.bias)

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_size, self.iterations), device=x.device)
        x = self.norm(x)
        x = self.first_layer(x)

        int_reps = []
        for i in range(self.iterations):
            this_x = x
            for j, (conv, norm, r_conv) in enumerate(zip(self.conv, 
                    self.norms, self.r_conv)):

                if i > 0:
                    this_x = norm(self.int_nonlin(r_conv(int_reps[j]) +
                            conv(this_x)))
                    int_reps[j] = this_x
                else:
                    this_x = norm(self.int_nonlin(conv(this_x)))
                    int_reps.append(this_x)

            this_x = self.pool(this_x)
            if not self.factorized:
                this_x = this_x.view(this_x.shape[0], -1)
            out[:, :, i] = self.fc(this_x)

        return out
