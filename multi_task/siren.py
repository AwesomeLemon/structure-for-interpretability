# taken from https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py

import torch
import math
from torch import nn
import torch.nn.functional as F

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 30., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.net = nn.Sequential(*layers)
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 3,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)
#
# coor = torch.randn(1, 5, 2)
# print(net(coor)) # (1, 3) <- rgb value