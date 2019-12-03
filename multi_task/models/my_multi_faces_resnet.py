# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetSeparated(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetSeparated, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, if_separate=False, separate_chunks_num=None)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, if_separate=False, separate_chunks_num=4)
        self.num_automl_blocks2 = 1
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, if_separate=False, separate_chunks_num=4)
        self.layer3 = nn.ModuleList()
        self.num_automl_blocks3 = 1
        for i in range(self.num_automl_blocks3):
            cur = self._make_layer(block, 256 // self.num_automl_blocks3, num_blocks[2], stride=2, if_separate=False, separate_chunks_num=4) # should increase dim only after this whole layer is done
            self.in_planes = 128
            self.layer3.append(cur)

        self.in_planes = 256
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, if_separate=False, separate_chunks_num=4)
        self.layer4 = nn.ModuleList()
        self.num_automl_blocks4 = 8
        for i in range(self.num_automl_blocks4):
            cur = self._make_layer(block, 512 // self.num_automl_blocks4, num_blocks[3], stride=2, if_separate=False, separate_chunks_num=4)
            self.in_planes = 256
            self.layer4.append(cur)


        self.lin_coeffs_id_zero = []
        # chunk_strengths_0 = torch.nn.Parameter(- torch.rand((4, 4), requires_grad=True).cuda()) # number of chunks before, number of chunks after, id+zero
        # self.lin_coeffs_id_zero.append(chunk_strengths_0)
        #
        # chunk_strengths_1 = torch.nn.Parameter(- torch.rand((4, 8), requires_grad=True).cuda())
        # self.lin_coeffs_id_zero.append(chunk_strengths_1)
        #
        # chunk_strengths_2 = torch.nn.Parameter(- torch.rand((8, 40), requires_grad=True).cuda())
        # self.lin_coeffs_id_zero.append(chunk_strengths_2)

        chunk_strengths_0 = torch.nn.Parameter(0.4 * torch.ones((self.num_automl_blocks2, self.num_automl_blocks3), requires_grad=True).cuda()) # number of chunks before, number of chunks after, id+zero
        self.lin_coeffs_id_zero.append(chunk_strengths_0)

        chunk_strengths_1 = torch.nn.Parameter(0.4 * torch.ones((self.num_automl_blocks3, self.num_automl_blocks4), requires_grad=True).cuda())
        self.lin_coeffs_id_zero.append(chunk_strengths_1)

        chunk_strengths_2 = torch.nn.Parameter(0.4 * torch.ones((self.num_automl_blocks4, 40), requires_grad=True).cuda())
        self.lin_coeffs_id_zero.append(chunk_strengths_2)

        if_optimize_strenghts_separately = True
        if not if_optimize_strenghts_separately:
            self.register_parameter('chunk_strength_0', chunk_strengths_0)
            self.register_parameter('chunk_strength_1', chunk_strengths_1)
            self.register_parameter('chunk_strength_2', chunk_strengths_2)

    def _make_layer(self, block, planes, num_blocks, stride, if_separate, separate_chunks_num):
        print('layer')
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            #TODO: for my if_separate=True this is wrong if block.expansion != 1
            print(self.in_planes)
            if not if_separate:
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers).cuda()

    #TODO: I don't want to mess with the original 'mask' parameter, although it seems pretty useless
    def forward(self, x, mask):
        sigmoid_normalization = 1.#250.

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        out = self.layer2(out)

        curs = out.split(128 // self.num_automl_blocks2, dim=1)
        outs = []
        sigmoid_internal_multiple = 1.
        for j in range(self.num_automl_blocks3):  # o == number of chunks in the next layer
            cur_outs = []
            for i, cur in enumerate(curs):
                cur_outs.append(
                    cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[0][i, j]) / sigmoid_normalization)
            cur_outs_concat = torch.cat(cur_outs, 1)
            outs.append(cur_outs_concat)

        # for i in range(4):
        #     print(f'!! {i}')
        #     print(outs[i].shape)
        #     self.layer3[i](outs[i])

        # Now just feed each of these 4 outs into one of the 4 next chunks:
        curs = [self.layer3[i](outs[i]) for i in range(self.num_automl_blocks3)]

        # out = self.layer3(out)

        # curs = out.split(256 // 4, dim=1)
        outs = []
        for j in range(self.num_automl_blocks4):  # o == number of chunks in the next layer
            cur_outs = []
            for i, cur in enumerate(curs):
                cur_outs.append(
                    cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[1][i, j]) / sigmoid_normalization)
            cur_outs_concat = torch.cat(cur_outs, 1)
            outs.append(cur_outs_concat)

        curs = [self.layer4[i](outs[i]) for i in range(self.num_automl_blocks4)]

        # curs = list(out.split(512 // 4, dim=1))
        outs = []
        for j in range(40):  # o == number of chunks in the next layer
            cur_outs = []
            for i, cur in enumerate(curs):
                cur_outs.append(
                    cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[2][i, j]) / sigmoid_normalization)#+ cur * self.lin_coeffs_id_zero[2][i, j, 1] * 0)
            cur_outs_concat = torch.cat(cur_outs, 1)
            outs.append(cur_outs_concat)

        outs = [F.avg_pool2d(out, 4) for out in outs]

        outs = [out.view(out.size(0), -1) for out in outs]

        return outs, mask


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(2048, 2)
    
    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask