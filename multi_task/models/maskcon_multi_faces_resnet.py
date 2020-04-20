# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, mid_planes=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        # TODO: DANGER!!! proper version:
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = self.conv2(out)
        # out += self.shortcut(x)
        # out = self.bn2(out)
        # above is the improper version

        out = F.relu(out, inplace=True)
        return out

class Bottleneck(nn.Module):
    expansion = 1 #!!!!!

    def __init__(self, in_planes, out_planes, stride=1, mid_planes=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.expansion*out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class Binarize(Function):
    @staticmethod
    def forward(ctx, connectivity, num_samples):
        sampled = torch.multinomial(connectivity, num_samples=num_samples, replacement=False)
        res = torch.zeros_like(connectivity).scatter(1, sampled, torch.ones_like(connectivity))
        ctx.save_for_backward(sampled)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        sampled, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.zeros_like(grad_input).scatter(1, sampled, grad_input)
        return grad_input, None

class MaskConResNet(nn.Module):
    def __init__(self, block, num_blocks, num_chunks, width_mul, K=2):
        super(MaskConResNet, self).__init__()
        self.affine_everywhere = True
        self.width_mul = width_mul
        self.K = K

        #similar to wideresnets, don't change 'conv1'
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(16)

        # self.layer1 = self._make_layer(block, 64, 8, num_blocks[0], stride=1)

        self.layer1 = nn.ModuleList()
        #TODO: here I assume the last number of blocks is actually the first one (because in this model there's no layer4, while at the same time layer1 must be separated into branches, unlike what I had before
        self.num_automl_blocks1 = num_chunks[-1]
        self.in_planes = 16
        for i in range(self.num_automl_blocks2):
            cur = self._make_layer(block, 64, 8, num_blocks[0], stride=1)
            # cur = self._make_layer(block, 128 * self.width_mul, 128 * self.width_mul // (self.num_automl_blocks2 * 2), num_blocks[1], stride=2)
            self.in_planes = 16
            self.layer2.append(cur)

        self.layer2 = nn.ModuleList()
        self.num_automl_blocks2 = num_chunks[0]
        self.in_planes = 64 * block.expansion
        for i in range(self.num_automl_blocks2):
            cur = self._make_layer(block, 128 * self.width_mul, 16, num_blocks[1], stride=2)
            # cur = self._make_layer(block, 128 * self.width_mul, 128 * self.width_mul // (self.num_automl_blocks2 * 2), num_blocks[1], stride=2)
            self.in_planes = 64 * block.expansion
            self.layer2.append(cur)

        self.layer3 = nn.ModuleList()
        self.num_automl_blocks3 = num_chunks[1]
        self.in_planes = 128 * self.width_mul * block.expansion
        for i in range(self.num_automl_blocks3):
            cur = self._make_layer(block, 256 * self.width_mul, 32, num_blocks[2], stride=2)
            # cur = self._make_layer(block, 256 * self.width_mul, 256 * self.width_mul // (self.num_automl_blocks3 * 2), num_blocks[2], stride=2)
            # should increase dim only after this whole layer is done
            self.in_planes = 128 * self.width_mul * block.expansion
            self.layer3.append(cur)

        self.in_planes = 256 * self.width_mul * block.expansion
        # self.layer4 = nn.ModuleList()
        self.num_automl_blocks4 = num_chunks[2]  # 16
        # for i in range(self.num_automl_blocks4):
        #     cur = self._make_layer(block, 512, 4, num_blocks[3], stride=2)
        #     # cur = self._make_layer(block, 512, 512 // (self.num_automl_blocks4 * 2), num_blocks[3], stride=2)
        #     self.in_planes = 256 * self.width_mul * block.expansion
        #     self.layer4.append(cur)

        self.lin_coeffs_id_zero = []

        chunk_strengths_0 = torch.nn.Parameter(torch.rand((self.num_automl_blocks2, self.num_automl_blocks3),
                                                                requires_grad=True).to(device))  # number of chunks before, number of chunks after, id+zero
        self.lin_coeffs_id_zero.append(chunk_strengths_0)


        chunk_strengths_1 = torch.nn.Parameter(torch.rand((self.num_automl_blocks3, self.num_automl_blocks4), requires_grad=True).to(device))
        self.lin_coeffs_id_zero.append(chunk_strengths_1)

        if False:
            chunk_strengths_2 = torch.nn.Parameter(
                torch.rand((self.num_automl_blocks4, 40), requires_grad=True).to(device))
        else:
            chunk_strengths_2 = torch.nn.Parameter(
                torch.rand((self.num_automl_blocks3, 40), requires_grad=True).to(device))

        self.lin_coeffs_id_zero.append(chunk_strengths_2)

        if_optimize_strenghts_separately = False
        if not if_optimize_strenghts_separately:
            self.register_parameter('chunk_strength_0', chunk_strengths_0)
            self.register_parameter('chunk_strength_1', chunk_strengths_1)
            self.register_parameter('chunk_strength_2', chunk_strengths_2)

    def _make_layer(self, block, out_planes, mid_planes, num_blocks, stride):
        # print('layer')
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            cur_block = block(self.in_planes, out_planes, stride, mid_planes)
            layers.append(cur_block)
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers).to(device)

    # TODO: I don't want to mess with the original 'mask' parameter, although it seems pretty useless
    def forward(self, x, mask):
        def binarize(lin_coeffs_id_zero):
            lin_coeffs_id_zero = torch.clamp(lin_coeffs_id_zero, 0, 1)
            normalized = lin_coeffs_id_zero / torch.sum(lin_coeffs_id_zero, dim=1, keepdim=True)
            return Binarize.apply(normalized, self.K)

        out = F.relu(self.bn1(self.conv1(x)),
                     inplace=True)
        out = self.layer1(out)

        # curs = out.split(128 * self.width_mul // self.num_automl_blocks2, dim=1)
        curs = [self.layer2[i](out) for i in range(self.num_automl_blocks2)]
        # curs = torch.stack(curs, dim=1)

        outs = []
        mask_binarized_cur = binarize(self.lin_coeffs_id_zero[0])
        for j in range(self.num_automl_blocks3):
            for smth in range(self.num_automl_blocks2):
                if smth == 0:
                    cur_outs = curs[smth] * mask_binarized_cur[smth, j]
                else:
                    cur_outs += curs[smth] * mask_binarized_cur[smth, j]
            # tmp = curs * mask_binarized_cur[:, j][None, :, None, None, None]
            # cur_outs = torch.sum(tmp, dim=1)
            outs.append(cur_outs)

        curs = [self.layer3[i](outs[i]) for i in range(self.num_automl_blocks3)]
        curs = torch.stack(curs, dim=1)

        outs = []
        mask_binarized_cur = binarize(self.lin_coeffs_id_zero[2])
        for j in range(40):
            cur_outs = torch.sum(curs * mask_binarized_cur[:, j][None, :, None, None, None], dim=1)
            outs.append(cur_outs)

        # outs = []
        # mask_binarized_cur = binarize(self.lin_coeffs_id_zero[1])
        # for j in range(self.num_automl_blocks4):
        #     cur_outs = torch.sum(curs * mask_binarized_cur[:, j][None, :, None, None, None], dim=1)
        #     outs.append(cur_outs)
        #
        # curs = [self.layer4[i](outs[i]) for i in range(self.num_automl_blocks4)]
        # curs = torch.stack(curs, dim=1)
        #
        # outs = []
        # mask_binarized_cur = binarize(self.lin_coeffs_id_zero[2])
        # for j in range(40):  # o == number of chunks in the next layer
        #     cur_outs = torch.sum(curs * mask_binarized_cur[:, j][None, :, None, None, None], dim=1)
        #     outs.append(cur_outs)

        outs = [F.avg_pool2d(out, 8) for out in outs]

        outs = [out.view(out.size(0), -1) for out in outs]

        return outs, mask

class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(1024, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask