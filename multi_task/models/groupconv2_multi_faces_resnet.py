# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import math

def aggregate(gate, D, I, K, sort=False):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=False, b=32):
        print(b)
        super(DGConv2d, self).__init__()
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))
        self.K = int(math.log2(in_channels))
        eps = 1e-8
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort
        self.b = b
        self.num_groups = 2 ** (torch.sum(1 - ((self.gate.data - 0).sign() + 1) / 2.))

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        self.num_groups = 2 ** (torch.sum(1 - self.gate.data)) # number of groups == 2 ** number of times 'identity' was used (and not 'ones'); each identity increases number of groups twofold.
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org # Straight-Through Estimator
        U, gate = aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        if False:
            U_regularizer = 2 ** (self.K  + torch.sum(self.gate))
        else:
            U_regularizer = torch.sum(U)
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)

        channel_num = self.in_channels # This is the standard convolution complexity according to paper.
        assert self.in_channels == self.out_channels
        # o=sum(Layer_i**2)/b, where b is 32.
        complexity_term = channel_num ** 2 / self.b
        return x, U_regularizer, complexity_term


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return DGConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, bias=False, dilation=dilation)

class BasicBlockG(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockG, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups // 2
        print(width)
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=3, stride=stride, padding=1, bias=False)#conv3x3(in_planes, width, stride, dilation)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, 1, dilation)
        # self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        U_regularizer_sum = 0
        complexity_term_sum = 0
        if isinstance(x, tuple):
            x, U_regularizer_sum, complexity_term_sum = x[0], x[1], x[2]
        identity = x

        if False:
            out, u_reg_cur = self.conv1(x)
            U_regularizer_sum += u_reg_cur
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        if True:
            out, u_reg_cur, complexity_cur = self.conv2(out)
            U_regularizer_sum += u_reg_cur
            complexity_term_sum += complexity_cur
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out)
        return out, U_regularizer_sum, complexity_term_sum


class G_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 groups=1, width_per_group=64):
        super(G_ResNet, self).__init__()
        self.in_planes = 64

        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=None))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, u_reg1, complexity1 = self.layer1(out)

        out, u_reg2, complexity2 = self.layer2(out)

        out, u_reg3, complexity3 = self.layer3(out)

        out, u_reg4, complexity4 = self.layer4(out)

        out = F.avg_pool2d(out, 8)

        out = out.view(out.size(0), -1)

        return out, u_reg1 + u_reg2 + u_reg3 + u_reg4, complexity1 + complexity2 + complexity3 + complexity4


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 2)
    
    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask