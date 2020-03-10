# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LearnableGroupsResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(LearnableGroupsResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_channels = 512
        self.groups = 8
        self.out_channels = 40  # number of tasks
        # this is connectivity after last convolution: connecting it to the 40 FC heads
        self.in_channel_group_connectivity = nn.Parameter(
            torch.rand((self.in_channels, self.groups)).cuda(),
            requires_grad=True)
        self.out_channel_group_connectivity = nn.Parameter(
            torch.rand((self.out_channels, self.groups)).cuda(),
            requires_grad=True)
        # self.in_channel_group_connectivity = nn.Parameter(
        #     torch.rand((self.in_channels, self.groups)).cuda() * 0.1 + 0.7,
        #     requires_grad=True)
        # self.out_channel_group_connectivity = nn.Parameter(
        #     torch.rand((self.out_channels, self.groups)).cuda() * 0.1 + 0.7,
        #     requires_grad=True)
        self.register_parameter('in_channels_group_connectivity', self.in_channel_group_connectivity)
        self.register_parameter('group_out_channel_connectivity', self.out_channel_group_connectivity)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 8)

        out = out.view(out.size(0), -1)

        if False:
            in_connect_softmaxed = torch.softmax(self.in_channel_group_connectivity, dim=1)
            in_connect_binarized = Binarize.apply(in_connect_softmaxed)

            out_connect_softmaxed = torch.softmax(self.out_channel_group_connectivity, dim=1)
            out_connect_binarized = Binarize.apply(out_connect_softmaxed)
        else:
            in_connect_softmaxed = torch.softmax(self.in_channel_group_connectivity, dim=1)
            in_connect_binarized = Binarize.apply(in_connect_softmaxed)

            out_connect_softmaxed = torch.softmax(self.out_channel_group_connectivity, dim=1)
            out_connect_binarized = Binarize.apply(out_connect_softmaxed)

        connectivity = torch.mm(in_connect_binarized,
                                torch.t(out_connect_binarized))
        connectivity.clamp(0, 1)

        if False:
            out_repeated = out[:, :, None].repeat(1, 1,
                                                  self.out_channel)  # batch_size, actual output, number of times to repeat
            out = out_repeated * connectivity
        else:
            outs = []
            for i in range(40):
                outs.append(out * connectivity[:, i])
        # out = out * connectivity[:, 0]

        return outs


class Binarize(Function):
    @staticmethod
    def forward(ctx, connectivity):
        if False:
            connectivity_softmaxed_maxed = torch.max(connectivity_softmaxed, dim=1, keepdim=True)[0]
            connectivity_softmaxed_normalized = connectivity_softmaxed / connectivity_softmaxed_maxed  # make biggest probability value equal to 1, instead of whatever its probability is
            connectivity_softmaxed_maxed_position = connectivity_softmaxed == connectivity_softmaxed_maxed
            res = torch.where(connectivity_softmaxed_maxed_position, connectivity_softmaxed_normalized,
                              torch.zeros_like(connectivity_softmaxed_normalized))
        else:
            # connectivity = connectivity.clamp_min(0)
            # connectivity = torch.softmax(connectivity, dim=1)
            # connectivity = connectivity / torch.sum(connectivity, dim=1, keepdim=True)
            num_samples = 4 if connectivity.size()[0] == 40 else 1
            sampled = torch.multinomial(connectivity, num_samples=num_samples, replacement=False)
            res = torch.zeros_like(connectivity).scatter(1, sampled, torch.ones_like(connectivity))
            if connectivity.size()[0] == 40:
                print(sampled[0])
        ctx.save_for_backward(sampled)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sampled, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.zeros_like(grad_input).scatter(1, sampled, grad_input)
        return grad_input


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out
