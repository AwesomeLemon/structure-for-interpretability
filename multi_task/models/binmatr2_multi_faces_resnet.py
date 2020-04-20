# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")


def kronecker(A, B):
    # haha, this code is so elegant & so unreadable at the same time
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))

class BinarizeBySample(Function):
    @staticmethod
    def forward(ctx, connectivity):
        sampled = connectivity > torch.rand(connectivity.size()).to(device)
        res = torch.where(sampled, torch.ones_like(connectivity), torch.zeros_like(connectivity))
        ctx.save_for_backward(sampled)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sampled, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[~sampled] = 0
        return grad_input

class BinarizeByThreshold(Function):
    @staticmethod
    def forward(ctx, connectivity):
        sampled = connectivity > torch.ones(connectivity.size()).to(device) * 0.5
        res = torch.where(sampled, torch.ones_like(connectivity), torch.zeros_like(connectivity))
        ctx.save_for_backward(sampled)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sampled, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[~sampled] = 0
        return grad_input


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, connectivity, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.connectivity = [connectivity]
        '''
        the above line is a hack, because I want: 
        (1) for 'connectivity' to be a separate parameter with separate lr
        (2) for it to be of type Parameter (seems necessary for backprop)
        
        but having self.smth=Parameter leads to that parameter being registered and sharing common lr.
        Thus the hack of putting it into a list
        
        '''
        #TODO: I set this only for graph visualization, change back ASAP
        # self.fake_connectivity = connectivity
        filter_to_chunks_factors = (out_channels // connectivity.size(0), in_channels // connectivity.size(1))
        self.chunks_to_filters_for_kronecker = torch.ones(filter_to_chunks_factors).to(device)
        self.ordinary_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ones_like_conn = torch.ones_like(self.connectivity[0]).to(device)
        self.zeros_like_conn = torch.zeros_like(self.connectivity[0]).to(device)

    def forward(self, x):
        binarized_connectivity = BinarizeByThreshold.apply(self.connectivity[0])

        # now upsample matrix from (chunks_out, chunks_in) to (out_channels, in_channels)
        # using the kronecker product with matrix of ones
        cur_connectivity = kronecker(binarized_connectivity, self.chunks_to_filters_for_kronecker)

        masked_weight = self.ordinary_conv.weight * cur_connectivity.view(self.out_channels, self.in_channels, 1, 1)
        x = F.conv2d(x, masked_weight, self.ordinary_conv.bias, self.ordinary_conv.stride, self.ordinary_conv.padding,
                     self.ordinary_conv.dilation)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, connectivity=None):
        super(BasicBlock, self).__init__()
        if connectivity is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = MaskedConv2d(in_planes, planes, connectivity=connectivity, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            #TODO: I don't know if in case of BinarizeBySample having sampling done in 2 places independently helps or hinders.
            if connectivity is None:
                conv_to_use = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            else:
                conv_to_use = MaskedConv2d(in_planes, self.expansion * planes, connectivity=connectivity, kernel_size=1, stride=stride, bias=False)
            self.shortcut = nn.Sequential(
                conv_to_use,
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BinMatrResNet(nn.Module):
    def __init__(self, block, num_blocks, num_chunks, width_mul, if_fully_connected):
        super(BinMatrResNet, self).__init__()
        self.if_fully_connected = if_fully_connected
        self.in_planes = 64
        self._create_connectivity_parameters(num_chunks)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64 * width_mul, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width_mul, num_blocks[1], stride=2)
        # layer2 is separated into 8 blocks after this, at the start of layer3
        self.layer3 = self._make_layer(block, 256 * width_mul, num_blocks[2],
                                       stride=2, connectivity=self.connectivities[0])
        # layer3 is separated into 8 blocks after this, at the start of layer4
        self.layer4 = self._make_layer(block, 512 * width_mul, num_blocks[3],
                                       stride=2, connectivity=self.connectivities[1])
        # layer4 is separated into 8 blocks during forward pass, when we get 40 outputs
        # it is there where connectivities[-1] is used


    def _make_layer(self, block, planes, num_blocks, stride, connectivity=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, connectivity))
            self.in_planes = planes * block.expansion

            connectivity = None # not more than the first layer in the block should work with connectivity
        return nn.Sequential(*layers)

    def _create_connectivity_parameters(self, num_chunks, tasks_num=40):
        self.connectivities = []
        for i in range(len(num_chunks) - 1):
            connectivity_shape = (num_chunks[i + 1], num_chunks[i])
            if self.if_fully_connected:
                cur_conn = torch.ones(connectivity_shape, requires_grad=False).to(device)
            else:
                # cur_conn = torch.nn.Parameter(torch.rand(connectivity_shape, requires_grad=True).to(device) )#* 2 - 1)
                cur_conn = torch.nn.Parameter(torch.ones(connectivity_shape, requires_grad=True).to(device) * 0.75)#* 2 - 1)
                # self.register_parameter(f'chunk_connectivity_{i}', cur_conn)
            self.connectivities.append(cur_conn)

        connectivity_shape = (tasks_num, num_chunks[-1])
        if self.if_fully_connected:
            task_conn = torch.ones(connectivity_shape, requires_grad=False).to(device)
        else:
            # task_conn = torch.nn.Parameter(torch.rand(connectivity_shape, requires_grad=True).to(device) )#* 2 - 1)
            task_conn = torch.nn.Parameter(torch.ones(connectivity_shape, requires_grad=True).to(device) * 0.75)#* 2 - 1)
            # self.register_parameter(f'chunk_connectivity_{total_blocks_num-1}', task_conn)
            # TODO: I set this only for graph visualization, change back ASAP
            # self.fake_task_conn = task_conn
        self.connectivities.append(task_conn)

    def forward(self, x):  # , ignored_filters_per_layer):

        if True:
            if not self.if_fully_connected:
                max_val = 1.0
                min_val = 0.5
                with torch.no_grad():
                    for connectivity in self.connectivities:
                        # connectivity[connectivity <= min_val] += 0.0003
                        connectivity[connectivity <= min_val] = min_val
                        connectivity[connectivity > max_val] = max_val
                        # connectivity.data.div_(torch.sum(connectivity.data, dim=1, keepdim=True))
        else:
            if not self.if_fully_connected:
                with torch.no_grad():
                    connectivity = self.connectivities[0]
                    connectivity[connectivity <= 0.5] = 0.5
                    connectivity[connectivity > 0.6] = 0.6
                    connectivity = self.connectivities[1]
                    connectivity[connectivity <= 0.5] = 0.5
                    connectivity[connectivity > 0.6] = 0.6

                    connectivity = self.connectivities[2]
                    connectivity[connectivity <= 0.5] = 0.5
                    connectivity[connectivity > 1.0] = 1.0

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)[:, :, None]

        outs = []
        connectivity = self.connectivities[-1]
        filter_to_chunks_factor = (out.size(1) // connectivity.size(1), 1)
        chunks_to_filters_for_kronecker = torch.ones(filter_to_chunks_factor).to(device)

        binarized_connectivity = BinarizeByThreshold.apply(connectivity)

        for i in range(40):
            cur_connectivity = kronecker(binarized_connectivity[i][:, None], chunks_to_filters_for_kronecker)
            outs.append((out * cur_connectivity).squeeze(-1))

        return outs


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask
