# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")


class Bottleneck(nn.Module):
    expansion = 1  # !!!!!

    def __init__(self, in_planes, out_planes, stride=1, mid_planes=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.expansion * out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x), inplace=True))
        out = self.bn2(F.relu(self.conv2(out), inplace=True))
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
    def __init__(self, block, num_blocks, num_chunks, width_mul, if_fully_connected=False, K=4):
        super(MaskConResNet, self).__init__()

        self.affine_everywhere = True
        self.width_mul = width_mul
        self.K = K
        self.if_fully_connected = if_fully_connected
        print(f'self.if_fully_connected = {if_fully_connected}')
        # similar to wideresnets, don't change 'conv1'

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # .to(device)
        self.bn1 = nn.BatchNorm2d(16)
        # todo: add maxpooling as in maskconnect model for MiniImageNet, where they also work with 64x64 images
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        # self.conv1_2_ala_maxpool = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0, bias=False)

        # self.layer1 = self._make_layer(block, 64, 8, num_blocks[0], stride=1)

        self.layer1 = []
        # TODO: here I assume the last number of blocks is actually the first one (because in this model there's no layer4, while at the same time layer1 must be separated into branches, unlike what I had before
        self.num_automl_blocks1 = num_chunks[-1]
        self.in_planes = 16
        for i in range(self.num_automl_blocks1):
            cur = self._make_layer_depthwise(block, 64, 4, num_blocks[0], stride=1)
            # cur = self._make_layer(block, 128 * self.width_mul, 128 * self.width_mul // (self.num_automl_blocks2 * 2), num_blocks[1], stride=2)
            self.in_planes = 16
            self.layer1.append(cur)

        self.layer2 = []
        self.num_automl_blocks2 = num_chunks[0]
        self.in_planes = 64 * block.expansion
        for i in range(self.num_automl_blocks2):
            cur = self._make_layer_depthwise(block, 128 * self.width_mul, 8, num_blocks[1], stride=2)
            # cur = self._make_layer(block, 128 * self.width_mul, 128 * self.width_mul // (self.num_automl_blocks2 * 2), num_blocks[1], stride=2)
            self.in_planes = 64 * block.expansion
            self.layer2.append(cur)

        self.layer3 = []
        self.num_automl_blocks3 = num_chunks[1]
        self.in_planes = 128 * self.width_mul * block.expansion
        for i in range(self.num_automl_blocks3):
            cur = self._make_layer_depthwise(block, 256 * self.width_mul, 16, num_blocks[2], stride=2)
            # cur = self._make_layer(block, 256 * self.width_mul, 256 * self.width_mul // (self.num_automl_blocks3 * 2), num_blocks[2], stride=2)
            # should increase dim only after this whole layer is done
            self.in_planes = 128 * self.width_mul * block.expansion
            self.layer3.append(cur)

        self.layers = self._transform_layers_to_parallel([self.layer1, self.layer2, self.layer3])
        assert num_chunks[0] == num_chunks[1]
        assert num_chunks[1] == num_chunks[2]
        self._create_connectivity_parameters(num_chunks[0], len(self.layers))

    def _make_layer_depthwise(self, block, out_planes, mid_planes, num_blocks, stride):
        '''
        3 layers: 1x1, 3x3, 1x1
        only one of the 8 chunks
        '''
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            cur_block = block(self.in_planes, out_planes, stride, mid_planes)
            layers.append(cur_block)
            self.in_planes = out_planes * block.expansion
        return layers

    def _transform_layers_to_parallel(self, list_of_layers_lists):
        parallel_layers = torch.nn.ModuleList()
        for layer in list_of_layers_lists:
            # layer is 8 chunks of 3 blocks
            # => input is 8-lengthed list of 3-lengthed lists
            # output is 3-lengthed list of 8-lengthed lists
            blocks_num = len(layer[0])
            chunks_num = len(layer)

            for block_depth in range(blocks_num):
                cur_parallel_layer_cur_depth = torch.nn.ModuleList()
                for cur_chunk_num in range(chunks_num):
                    cur_parallel_layer_cur_depth.append(layer[cur_chunk_num][block_depth])
                parallel_layers.append(cur_parallel_layer_cur_depth)

        return parallel_layers

    def _create_connectivity_parameters(self, chunks_num, total_blocks_num, tasks_num=40):
        self.connectivities = []
        for i in range(total_blocks_num - 1):
            if self.if_fully_connected:
                cur_conn = torch.ones((chunks_num, chunks_num), requires_grad=False).to(device)
            else:
                cur_conn = torch.nn.Parameter(torch.rand((chunks_num, chunks_num), requires_grad=True).to(device))
                # self.register_parameter(f'chunk_connectivity_{i}', cur_conn)
            self.connectivities.append(cur_conn)

        if False:
            if self.if_fully_connected:
                task_conn = torch.ones((tasks_num, chunks_num), requires_grad=False).to(device)
            else:
                task_conn = torch.nn.Parameter(torch.rand((tasks_num, chunks_num), requires_grad=True).to(device))
                # self.register_parameter(f'chunk_connectivity_{total_blocks_num-1}', task_conn)
            self.connectivities.append(task_conn)

    # TODO: I don't want to mess with the original 'mask' parameter, although it seems pretty useless
    def forward(self, x, mask):
        def binarize(lin_coeffs_id_zero, branches_to_sample_num):
            # lin_coeffs_id_zero = torch.clamp(lin_coeffs_id_zero, 1e-10, 1) #multinomial doesn't like exact 0 probability

            # normalized = lin_coeffs_id_zero / torch.sum(lin_coeffs_id_zero, dim=1, keepdim=True)
            # lin_coeffs_id_zero.data.div_(torch.sum(lin_coeffs_id_zero.data.clone(), dim=1, keepdim=True))
            return Binarize.apply(lin_coeffs_id_zero, branches_to_sample_num)

        def sum_list_of_tensors(list_of_tensors):
            if False:
                res = list_of_tensors[0]
                for i in range(1, len(list_of_tensors)):
                    res += list_of_tensors[i]
            else:
                res = torch.stack(list_of_tensors, dim=1)
                res = torch.sum(res,dim=1)
            return res

        # out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.maxpool1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        # del x
        if not self.if_fully_connected:
            with torch.no_grad():
                for connectivity in self.connectivities:
                    connectivity[connectivity <= 0] = 1e-10
                    connectivity[connectivity > 1] = 1
                    connectivity.data.div_(torch.sum(connectivity.data, dim=1, keepdim=True))

        if_use_list_comprehension = False  # if True, then faster, but uses more memory
        for i, layer in enumerate(self.layers):
            # if if_use_list_comprehension:
            #     if i == 0:
            #         curs = [block(out) for block in layer]
            #     else:
            #         curs = [block(out) for block, out in zip(layer, outs)]
            # else:
            curs = []
            if i == 0:
                for block in layer:
                    curs.append(block(out))
            else:
                for block, out in zip(layer, outs):
                    curs.append(block(out))

            if True:
                if i == len(self.layers) - 1:
                    curs = torch.stack(curs, dim=1)
                    out = torch.sum(curs, dim=1)
                    break

            if not self.if_fully_connected:
                mask_binarized_cur = binarize(self.connectivities[i], self.K)
            else:
                mask_binarized_cur = self.connectivities[i]
            if if_use_list_comprehension:# or i % 2 == 0:
                curs = torch.stack(curs, dim=1)
                outs = [torch.sum(curs * mask_binarized_cur[j][None, :, None, None, None], dim=1)
                        for j in
                        range(self.connectivities[i].size(0))]
            else:
                outs = []
                for j in range(self.connectivities[i].size(0)):
                    cur_inner = []
                    for k in range(self.connectivities[i].size(1)):
                        cur_inner.append(curs[k] * mask_binarized_cur[j, k])
                    outs.append(sum_list_of_tensors(cur_inner))
        if False:
            if if_use_list_comprehension:
                outs = [F.avg_pool2d(out, (8, 4)).view(out.size(0), -1) for out in outs]
            else:
                new_outs = []
                for out in outs:
                    new_outs.append(F.avg_pool2d(out, (8, 4)).view(out.size(0), -1))
                outs = new_outs
        else:
            outs = F.avg_pool2d(out, (8, 4)).view(out.size(0), -1)

        return outs, mask


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask
