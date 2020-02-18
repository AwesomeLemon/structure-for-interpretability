# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, affine=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.bn2_affine = affine

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),
                                  inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        #above is the improper version

        if self.bn2_affine:
            out = F.relu(out,
                                  inplace=True)
            #otherwise, I'll apply ReLu myself outside after multiplying by my coefficients
        return out

class PlainNet(nn.Module):
    def __init__(self, block, num_blocks, num_chunks):
        super(PlainNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, if_separate=False, separate_chunks_num=None, enforce_affine=True)
        #TODO: stride was 2 for 64x64, changed to 1 for 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, if_separate=False, separate_chunks_num=4)
        self.num_automl_blocks2 = num_chunks[0]#16
        #WRONG: we don't need true batch norms, we need 1 gamma and 1 beta per block pair
        # self.layer2_batch_norms = nn.ModuleList()
        # for i in range(self.num_automl_blocks2):
        #     self.layer2_batch_norms.append(nn.BatchNorm2d(128 // self.num_automl_blocks2))
        self.layer3 = nn.ModuleList()
        # self.layer3_batch_norms = nn.ModuleList()
        self.num_automl_blocks3 = num_chunks[1]#16
        for i in range(self.num_automl_blocks3):
            cur = self._make_layer(block, 256 // self.num_automl_blocks3, num_blocks[2], stride=2, if_separate=False, separate_chunks_num=4) # should increase dim only after this whole layer is done
            self.in_planes = 128
            self.layer3.append(cur)
            # self.layer3_batch_norms.append(nn.BatchNorm2d(256 // self.num_automl_blocks3))

        self.in_planes = 256
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, if_separate=False, separate_chunks_num=4)
        self.layer4 = nn.ModuleList()
        # self.layer4_batch_norms = nn.ModuleList()
        self.num_automl_blocks4 = num_chunks[2]#16
        for i in range(self.num_automl_blocks4):
            cur = self._make_layer(block, 512 // self.num_automl_blocks4, num_blocks[3], stride=2, if_separate=False, separate_chunks_num=4)
            self.in_planes = 256
            self.layer4.append(cur)
            # self.layer4_batch_norms.append(nn.BatchNorm2d(512 // self.num_automl_blocks4))


        self.lin_coeffs_id_zero = []
        self.bn_biases = []
        # chunk_strengths_0 = torch.nn.Parameter(- torch.rand((4, 4), requires_grad=True).cuda()) # number of chunks before, number of chunks after, id+zero
        # self.lin_coeffs_id_zero.append(chunk_strengths_0)
        #
        # chunk_strengths_1 = torch.nn.Parameter(- torch.rand((4, 8), requires_grad=True).cuda())
        # self.lin_coeffs_id_zero.append(chunk_strengths_1)
        #
        # chunk_strengths_2 = torch.nn.Parameter(- torch.rand((8, 40), requires_grad=True).cuda())
        # self.lin_coeffs_id_zero.append(chunk_strengths_2)

        chunk_strengths_0 = torch.nn.Parameter(0.1 * torch.ones((self.num_automl_blocks2, self.num_automl_blocks3), requires_grad=True).cuda()) # number of chunks before, number of chunks after, id+zero
        # chunk_strengths_0 = torch.nn.Parameter(torch.ones((self.num_automl_blocks2, self.num_automl_blocks3), requires_grad=True).cuda()) # number of chunks before, number of chunks after, id+zero
        self.lin_coeffs_id_zero.append(chunk_strengths_0)
        bn_bias_0 = torch.nn.Parameter(torch.zeros((self.num_automl_blocks2, self.num_automl_blocks3), requires_grad=True).cuda())
        self.bn_biases.append(bn_bias_0)

        chunk_strengths_1 = torch.nn.Parameter(0.1 * torch.ones((self.num_automl_blocks3, self.num_automl_blocks4), requires_grad=True).cuda())
        # chunk_strengths_1 = torch.nn.Parameter(torch.ones((self.num_automl_blocks3, self.num_automl_blocks4), requires_grad=True).cuda())
        self.lin_coeffs_id_zero.append(chunk_strengths_1)
        bn_bias_1 = torch.nn.Parameter(
            torch.zeros((self.num_automl_blocks3, self.num_automl_blocks4), requires_grad=True).cuda())
        self.bn_biases.append(bn_bias_1)

        chunk_strengths_2 = torch.nn.Parameter(0.1 * torch.ones((self.num_automl_blocks4, 40), requires_grad=True).cuda())
        # chunk_strengths_2 = torch.nn.Parameter(torch.ones((self.num_automl_blocks4, 40), requires_grad=True).cuda())
        self.lin_coeffs_id_zero.append(chunk_strengths_2)
        bn_bias_2 = torch.nn.Parameter(
            torch.zeros((self.num_automl_blocks4, 40), requires_grad=True).cuda())
        self.bn_biases.append(bn_bias_2)

        if_optimize_strenghts_separately = False
        if not if_optimize_strenghts_separately:
            self.register_parameter('chunk_strength_0', chunk_strengths_0)
            self.register_parameter('chunk_strength_1', chunk_strengths_1)
            self.register_parameter('chunk_strength_2', chunk_strengths_2)

            self.register_parameter('bn_bias_0', bn_bias_0)
            self.register_parameter('bn_bias_1', bn_bias_1)
            self.register_parameter('bn_bias_2', bn_bias_2)

    def _make_layer(self, block, planes, num_blocks, stride, if_separate, separate_chunks_num, enforce_affine=False):
        # print('layer')
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            if i == (len(strides) - 1):
                cur_block = block(self.in_planes, planes, stride, False or enforce_affine)
            else:
                cur_block = block(self.in_planes, planes, stride, True)
            layers.append(cur_block)
            #TODO: for my if_separate=True this is wrong if block.expansion != 1
            # print(self.in_planes)
            if not if_separate:
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers).cuda()

    #TODO: I don't want to mess with the original 'mask' parameter, although it seems pretty useless
    def forward(self, x, mask):
        '''
        previous version input shape: blocks, batch_size, filters, W, H
        actual input shape: batch_size, blocks, filters, W, H
        output shape: batch_size, (blocks * filters), W, H
        '''
        def concat_tensor_of_blocks_across_blocks(tensor_of_blocks):
            new_size = list(tensor_of_blocks.size())
            return tensor_of_blocks.view(new_size[0], -1, new_size[-2], new_size[-1])
            # 0-th dimension is number of blocks. we need to concatenate them across filter dimension, i.e. the 2-nd dimension:
            # new_size[2] *= new_size[0]
            # new_size = new_size[1:]
            # return tensor_of_blocks.view(new_size)

        sigmoid_normalization = 1.#250.

        out = F.relu(self.bn1(self.conv1(x)),
                                  inplace=True)
        out = self.layer1(out)

        out = self.layer2(out)

        if_new_implementation = True
        curs = out.split(128 // self.num_automl_blocks2, dim=1)
        if if_new_implementation:
            curs = torch.stack(curs, dim=1)
        outs = []
        sigmoid_internal_multiple = 1.
        for j in range(self.num_automl_blocks3):  # o == number of chunks in the next layer
            if not if_new_implementation:
                cur_outs = []
                for i, cur in enumerate(curs):
                    # cur_outs.append(
                    #     cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[0][i, j]) / sigmoid_normalization)
                    cur_outs.append(
                        F.relu(cur * self.lin_coeffs_id_zero[0][i, j] + self.bn_biases[0][i, j]))
                cur_outs_concat = torch.cat(cur_outs, 1)
            else:
                cur_outs = F.relu(curs * self.lin_coeffs_id_zero[0][:, j][None, :, None, None, None]   + self.bn_biases[0][:, j][None, :, None, None, None],
                                  inplace=True)
                cur_outs_concat = concat_tensor_of_blocks_across_blocks(cur_outs)
            outs.append(cur_outs_concat)

        # for i in range(4):
        #     print(f'!! {i}')
        #     print(outs[i].shape)
        #     self.layer3[i](outs[i])
        # Now just feed each of these 4 outs into one of the 4 next chunks:
        curs = [self.layer3[i](outs[i]) for i in range(self.num_automl_blocks3)]
        if if_new_implementation:
            # print(curs[0].size())
            # curs = torch.stack(curs, dim=0)
            curs = torch.stack(curs, dim=1)

        # out = self.layer3(out)

        # curs = out.split(256 // 4, dim=1)
        outs = []
        for j in range(self.num_automl_blocks4):  # o == number of chunks in the next layer
            if not if_new_implementation:
                cur_outs = []
                for i, cur in enumerate(curs):
                    # cur_outs.append(
                    #     cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[1][i, j]) / sigmoid_normalization)
                    cur_outs.append(
                        F.relu(cur * self.lin_coeffs_id_zero[1][i, j]+ self.bn_biases[1][i, j]))
                cur_outs_concat = torch.cat(cur_outs, 1)
            if if_new_implementation:
                cur_outs = F.relu(curs * self.lin_coeffs_id_zero[1][:, j][None, :, None, None, None] + self.bn_biases[1][:, j][None, :, None, None, None],
                                  inplace=True)
                cur_outs_concat = concat_tensor_of_blocks_across_blocks(cur_outs)
                # cur_outs_concat = torch.cat(cur_outs, 0)
            outs.append(cur_outs_concat)

        curs = [self.layer4[i](outs[i]) for i in range(self.num_automl_blocks4)]
        if if_new_implementation:
            # curs = torch.stack(curs, dim=0)
            curs = torch.stack(curs, dim=1)

        # curs = list(out.split(512 // 4, dim=1))
        outs = []
        for j in range(40):  # o == number of chunks in the next layer
            if not if_new_implementation:
                cur_outs = []
                for i, cur in enumerate(curs):
                    # cur_outs.append(
                    #     cur * torch.sigmoid(sigmoid_internal_multiple * self.lin_coeffs_id_zero[2][i, j]) / sigmoid_normalization)#+ cur * self.lin_coeffs_id_zero[2][i, j, 1] * 0)
                    cur_outs.append(
                        F.relu(cur * self.lin_coeffs_id_zero[2][i, j]+ self.bn_biases[2][i, j]))
                cur_outs_concat = torch.cat(cur_outs, 1)
            else:
                #TODO: remove clamp_max
                # cur_outs = F.relu(curs * torch.clamp_max(self.lin_coeffs_id_zero[2][:, j][None, :, None, None, None], 0.2) + self.bn_biases[2][:, j][None, :, None, None, None],
                cur_outs = F.relu(curs * self.lin_coeffs_id_zero[2][:, j][None, :, None, None, None] + self.bn_biases[2][:, j][None, :, None, None, None],
                                  inplace=True)
                cur_outs_concat = concat_tensor_of_blocks_across_blocks(cur_outs)
            outs.append(cur_outs_concat)

            '''
            unit test replacement: 
            checking whether calculation of cur_outs isn't screwed up by dimensionality extensions
            (using strange non-default values to be sure)
            curs[1, 255, 4, 7, 7]
            curs[1, 255, 4, 7, 7] * self.lin_coeffs_id_zero[2][1, 0] + self.bn_biases[2][1, 0]
            cur_outs[1, 255, 4, 7, 7] 
            '''

        outs = [F.avg_pool2d(out, 8) for out in outs]

        outs = [out.view(out.size(0), -1) for out in outs]

        return outs, mask


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask