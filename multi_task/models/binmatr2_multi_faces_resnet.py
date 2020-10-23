# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

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
                 padding=0, bias=True):
        super(MaskedConv2d, self).__init__()
        self.connectivity = [connectivity]
        '''
        the above line is a hack, because I want: 
        (1) for 'connectivity' to be a separate parameter with separate lr
        (2) for it to be of type Parameter (seems necessary for backprop)
        
        but having self.smth=Parameter leads to that parameter being registered and sharing common lr.
        Thus the hack of putting it into a list
        
        '''
        # TODO: I set this only for graph visualization, change back ASAP
        # self.fake_connectivity = connectivity
        filter_to_chunks_factors = (out_channels // connectivity.size(0), in_channels // connectivity.size(1))
        if filter_to_chunks_factors == (1, 1):  # i.e. 1 block == 1 filter & therefore kronecker is unnecessary
            self.chunks_to_filters_for_kronecker = None
        else:
            self.chunks_to_filters_for_kronecker = torch.ones(filter_to_chunks_factors).to(device)
        self.ordinary_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ones_like_conn = torch.ones_like(self.connectivity[0]).to(device)
        self.zeros_like_conn = torch.zeros_like(self.connectivity[0]).to(device)

    def forward(self, x):
        if True:
            binarized_connectivity = BinarizeByThreshold.apply(self.connectivity[0])

            # now upsample matrix from (chunks_out, chunks_in) to (out_channels, in_channels)
            # using the kronecker product with matrix of ones
            if self.chunks_to_filters_for_kronecker is None:
                cur_connectivity = binarized_connectivity
            else:
                cur_connectivity = kronecker(binarized_connectivity, self.chunks_to_filters_for_kronecker)

            masked_weight = self.ordinary_conv.weight * cur_connectivity.view(self.out_channels, self.in_channels, 1, 1)
            # assert torch.all(torch.eq(masked_weight, self.ordinary_conv.weight))
            out = F.conv2d(x, masked_weight, self.ordinary_conv.bias, self.ordinary_conv.stride,
                           self.ordinary_conv.padding,
                           self.ordinary_conv.dilation)
            # assert torch.all(torch.eq(out, self.ordinary_conv(x)))
        else:
            out = self.ordinary_conv(x)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, connectivity=(None, None), if_enable_bias=False):
        super(BasicBlock, self).__init__()
        connectivity1, connectivity2 = connectivity
        if connectivity1 is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=if_enable_bias)
        else:
            self.conv1 = MaskedConv2d(in_planes, planes, connectivity=connectivity1, kernel_size=3, stride=stride,
                                      padding=1, bias=if_enable_bias)

        self.bn1 = nn.BatchNorm2d(planes)
        if connectivity2 is None:
            if False:
                print('ATTENZIONE: using eye connectivity for conv2')
                connectivity2 = torch.eye(planes, requires_grad=False).to(device);
                self.conv2 = MaskedConv2d(planes, planes, connectivity=connectivity2, kernel_size=3, stride=1,
                                          padding=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=if_enable_bias)
        else:
            self.conv2 = MaskedConv2d(planes, planes, connectivity=connectivity2, kernel_size=3, stride=1, padding=1,
                                      bias=if_enable_bias)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # TODO: I don't know if in case of BinarizeBySample having sampling done in 2 places independently helps or hinders.
            if connectivity1 is None:
                conv_to_use = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                                        bias=if_enable_bias)
            else:
                conv_to_use = MaskedConv2d(in_planes, self.expansion * planes, connectivity=connectivity1,
                                           kernel_size=1, stride=stride, bias=if_enable_bias)
            self.shortcut = nn.Sequential(
                conv_to_use,
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if False:
            out = F.relu(out)
        else:
            out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if False:
            out = F.relu(out)
        else:
            out = self.relu2(out)
        return out


class BasicBlockNoSkip(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, connectivity=(None, None), if_enable_bias=False):
        super(BasicBlockNoSkip, self).__init__()
        connectivity1, connectivity2 = connectivity
        if connectivity1 is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=if_enable_bias)
        else:
            self.conv1 = MaskedConv2d(in_planes, planes, connectivity=connectivity1, kernel_size=3, stride=stride,
                                      padding=1, bias=if_enable_bias)

        self.bn1 = nn.BatchNorm2d(planes)
        if connectivity2 is None:
            if False:
                print('ATTENZIONE: using eye connectivity for conv2')
                connectivity2 = torch.eye(planes, requires_grad=False).to(device);
                self.conv2 = MaskedConv2d(planes, planes, connectivity=connectivity2, kernel_size=3, stride=1,
                                          padding=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=if_enable_bias)
        else:
            self.conv2 = MaskedConv2d(planes, planes, connectivity=connectivity2, kernel_size=3, stride=1, padding=1,
                                      bias=if_enable_bias)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class BasicBlockMock(BasicBlock):
    expansion = 1

    class SparseIdentityShortcut(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.mask = mask

        def forward(self, input):
            return input * self.mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # mask is over channels

    def __init__(self, in_planes, planes, stride=1, connectivity=(None, None, None), if_enable_bias=False):
        super().__init__(in_planes, planes, stride, connectivity[:-1], if_enable_bias)

        if (len(list(self.shortcut.children())) == 0):  # is identity shortcut
            if connectivity[-1] is not None:
                self.shortcut = BasicBlockMock.SparseIdentityShortcut(connectivity[-1])
            else:
                if not ((connectivity[0] is None) and (connectivity[1] is None)):
                    raise ValueError('If identity shortcut, either all 3 conns are not-None, or all 3 are None. '
                                     'This was violated.')

def are_activations_for_each_input_equal(activations):
    return torch.eq(activations.min(dim=0)[0], activations.max(dim=0)[0]).all()

class BasicBlockAdditivesCreator(BasicBlock):
    '''
    this class is for converting constant inputs to constant additives
    '''
    expansion = 1
    id = 0
    additives_dict = {}

    def forward(self, x):
        if_projection_shortcut = False
        shortcut_modules = list(self.shortcut.children())
        if (len(shortcut_modules) > 0) and (type(shortcut_modules[0]) is MaskedConv2d):
            if_projection_shortcut = True
        if_identity_shortcut = not if_projection_shortcut
        pre_conv1 = x
        additives_dict_cur = {'conv1':{}, 'shortcut':{}, 'shortcut_id':{}, 'conv2':{}}

        const_features_input_idx = []
        for i in range(self.conv1.in_channels):
            cur_feature = pre_conv1[:, i]
            if are_activations_for_each_input_equal(cur_feature):
                const_features_input_idx.append(i)
        if if_identity_shortcut:  # don't need the processing per outgoing connection in the 'for' below
            for i in const_features_input_idx:
                if i not in additives_dict_cur['shortcut_id']:
                    additives_dict_cur['shortcut_id'][i] = x[0:1, i:i+1, :, :]
                else:
                    print('wtf')
                if not (torch.isclose(x[0, i, 4, 4], x[17, i, 5, 3], rtol=1e-4)):
                    print('!!!', i)
        if type(self.conv1) is MaskedConv2d:
            if self.conv1.ordinary_conv.bias is not None:
                conv1_bias_stored = self.conv1.ordinary_conv.bias.detach().clone()
                self.conv1.ordinary_conv.bias *= 0
                if if_projection_shortcut:
                    shortcut_bias_stored = shortcut_modules[0].ordinary_conv.bias.detach().clone()
                    shortcut_modules[0].ordinary_conv.bias *= 0
            for i in const_features_input_idx:
                connected_to_i_idx = torch.where(self.conv1.connectivity[0][:, i] > 0.5)[0]

                temp = self.conv1.connectivity[0].detach().clone()
                self.conv1.connectivity[0] *= 0.0
                self.conv1.connectivity[0][connected_to_i_idx, i] += 1.0

                post_conv1_only_cur_input = self.conv1(pre_conv1)
                if if_projection_shortcut:
                    post_shortcut_only_cur_input = shortcut_modules[0](x)
                for j in connected_to_i_idx:

                    if j not in additives_dict_cur['conv1']:
                        additives_dict_cur['conv1'][j] = post_conv1_only_cur_input[0:1, j:j+1, :, :]
                    else:
                        additives_dict_cur['conv1'][j] += post_conv1_only_cur_input[0:1, j:j + 1, :, :]

                    # a crude sanity check (comparison with other arbitrary sample & not-border index)
                    if not (torch.isclose(post_conv1_only_cur_input[0, j, 4, 4], post_conv1_only_cur_input[17, j, 5, 3],
                                          rtol=1e-4)):
                        print('!!!', i)

                    if if_projection_shortcut:
                        if j not in additives_dict_cur['shortcut']:
                            additives_dict_cur['shortcut'][j] = post_shortcut_only_cur_input[0:1, j:j + 1, :, :]
                        else:
                            additives_dict_cur['shortcut'][j] += post_shortcut_only_cur_input[0:1, j:j + 1, :, :]
                        if not (torch.isclose(post_shortcut_only_cur_input[0, j, 4, 4], post_shortcut_only_cur_input[17, j, 5, 3],
                                      rtol=1e-4)):
                            print('!!!', i)

                self.conv1.connectivity[0].data = temp.data
                self.conv1.connectivity[0][:, i] *= 0.0

            if self.conv1.ordinary_conv.bias is not None:
                self.conv1.ordinary_conv.bias.data = conv1_bias_stored
                if if_projection_shortcut:
                    shortcut_modules[0].ordinary_conv.bias.data = shortcut_bias_stored
            post_conv1 = self.conv1(pre_conv1)
            for key, value in additives_dict_cur['conv1'].items():
                post_conv1[:, key:key+1, :, :] += additives_dict_cur['conv1'][key]
            post_conv1_bn = F.relu(self.bn1(post_conv1))

        if not(type(self.conv1) is MaskedConv2d):
            post_conv1 = self.conv1(pre_conv1)
            post_conv1_bn = F.relu(self.bn1(post_conv1))

        if type(self.conv2) is MaskedConv2d:
            if self.conv2.ordinary_conv.bias is not None:
                conv2_bias_stored = self.conv2.ordinary_conv.bias.detach().clone()
                self.conv2.ordinary_conv.bias *= 0
            const_features_conv1_idx = []
            for i in range(self.conv1.out_channels):
                cur_feature = post_conv1_bn[:, i]
                if are_activations_for_each_input_equal(cur_feature):
                    const_features_conv1_idx.append(i)

            for i in const_features_conv1_idx:
                connected_to_i_idx = torch.where(self.conv2.connectivity[0][:, i] > 0.5)[0]

                temp = self.conv2.connectivity[0].detach().clone()
                self.conv2.connectivity[0] *= 0.0
                self.conv2.connectivity[0][connected_to_i_idx, i] += 1.0

                post_conv2_only_cur_input = self.conv2(post_conv1_bn)
                for j in connected_to_i_idx:
                    if j not in additives_dict_cur['conv2']:
                        additives_dict_cur['conv2'][j] = post_conv2_only_cur_input[0:1, j:j+1, :, :]
                    else:
                        additives_dict_cur['conv2'][j] += post_conv2_only_cur_input[0:1, j:j + 1, :, :]

                    # a crude sanity check (comparison with other arbitrary sample & not-border index)
                    if not (torch.isclose(post_conv2_only_cur_input[0, j, 4, 4], post_conv2_only_cur_input[17, j, 5, 3],
                                          rtol=1e-4)):
                        print('!!!', i)

                self.conv2.connectivity[0].data = temp.data
                self.conv2.connectivity[0][:, i] *= 0.0

            if self.conv2.ordinary_conv.bias is not None:
                self.conv2.ordinary_conv.bias.data = conv2_bias_stored
            post_conv2 = self.conv2(post_conv1_bn)
            for key, value in additives_dict_cur['conv2'].items():
                post_conv2[:, key:key+1, :, :] += additives_dict_cur['conv2'][key]
            post_conv2_bn = self.bn2(post_conv2)

        if not (type(self.conv2) is MaskedConv2d):
            post_conv2 = self.conv2(post_conv1_bn)
            post_conv2_bn = self.bn2(post_conv2)

        out = post_conv2_bn

        if if_projection_shortcut:
            shortcut_out = shortcut_modules[0](x)
            for key, value in additives_dict_cur['shortcut'].items():
                shortcut_out[:, key:key+1, :, :] += additives_dict_cur['shortcut'][key]
            shortcut_out = shortcut_modules[1](shortcut_out)
            out += shortcut_out
        elif if_identity_shortcut:
            for key, value in additives_dict_cur['shortcut_id'].items():
                out[:, key:key+1, :, :] += additives_dict_cur['shortcut_id'][key]
            mask = torch.Tensor([0 if i in additives_dict_cur['shortcut_id'].keys() else 1 for i in range(self.conv1.in_channels)]).cuda()
            print(torch.where(1 - mask)[0])
            out += self.shortcut(x) * mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = F.relu(out)

        BasicBlockAdditivesCreator.additives_dict[BasicBlockAdditivesCreator.id] = additives_dict_cur
        BasicBlockAdditivesCreator.id += 1
        print('Danger! Assume only 1 batch when creating additives')
        return out


class BasicBlockMockAdditivesUser(BasicBlock):
    expansion = 1
    id = 0
    additives_dict = {}

    def __init__(self, in_planes, planes, stride=1, connectivity=(None, None, None), if_enable_bias=False):
        if len(connectivity) == 3:
            super().__init__(in_planes, planes, stride, connectivity[:-1], if_enable_bias)
        else:
            print('Less connectivities than expected')
            super().__init__(in_planes, planes, stride, connectivity, if_enable_bias)

        self.has_projection_shortcut = True
        if (len(list(self.shortcut.children())) == 0):  # is identity shortcut
            self.has_projection_shortcut = False
            if len(connectivity) == 3:
                if connectivity[-1] is not None:
                    self.shortcut = BasicBlockMock.SparseIdentityShortcut(connectivity[-1])
                else:
                    if not ((connectivity[0] is None) and (connectivity[1] is None)):
                        raise ValueError('If identity shortcut, either all 3 conns are not-None, or all 3 are None. '
                                         'This was violated.')

        self.id = BasicBlockMockAdditivesUser.id
        BasicBlockMockAdditivesUser.id += 1

    def forward(self, x):
        if_projection_shortcut = self.has_projection_shortcut
        shortcut_modules = list(self.shortcut.children())
        if_identity_shortcut = not if_projection_shortcut
        pre_conv1 = x
        additives_dict_cur = BasicBlockMockAdditivesUser.additives_dict[self.id]

        if type(self.conv1) is MaskedConv2d:
            post_conv1 = self.conv1(pre_conv1)
            for key, value in additives_dict_cur['conv1'].items():
                post_conv1[:, key:key + 1, :, :] += additives_dict_cur['conv1'][key]
            post_conv1_bn = F.relu(self.bn1(post_conv1))
        else:
            post_conv1 = self.conv1(pre_conv1)
            post_conv1_bn = F.relu(self.bn1(post_conv1))

        if type(self.conv2) is MaskedConv2d:
            post_conv2 = self.conv2(post_conv1_bn)
            for key, value in additives_dict_cur['conv2'].items():
                post_conv2[:, key:key + 1, :, :] += additives_dict_cur['conv2'][key]
            post_conv2_bn = self.bn2(post_conv2)
        else:
            post_conv2 = self.conv2(post_conv1_bn)
            post_conv2_bn = self.bn2(post_conv2)

        out = post_conv2_bn

        if if_projection_shortcut:
            shortcut_out = shortcut_modules[0](x)
            for key, value in additives_dict_cur['shortcut'].items():
                shortcut_out[:, key:key + 1, :, :] += additives_dict_cur['shortcut'][key]
            shortcut_out = shortcut_modules[1](shortcut_out)
            out += shortcut_out
        elif if_identity_shortcut:
            for key, value in additives_dict_cur['shortcut_id'].items():
                out[:, key:key + 1, :, :] += additives_dict_cur['shortcut_id'][key]
            mask = torch.Tensor([0 if i in additives_dict_cur['shortcut_id'].keys() else 1 for i in range(self.conv1.in_channels)]).cuda()
            out += self.shortcut(x) * mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = F.relu(out)
        return out


class BasicBlockAvgAdditivesCreator(BasicBlock):
    '''
    this class is for converting non-constant inputs to constant (average) additives
    current limitations:
        - shortcuts aren't handled properly
        - connections to task heads aren't handled properly
    '''
    expansion = 1
    id = 0
    additives_dict = {}
    used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()

    @staticmethod
    def get_indices_of_used_neurons(layer_ind):
        return np.array([int(x[x.find('_') + 1:]) for x in BasicBlockAvgAdditivesCreator.used_neurons[layer_ind]])

    def forward(self, x):
        assert BasicBlockAvgAdditivesCreator.id < 16, 'only a single forward pass should be made'
        if_projection_shortcut = False
        shortcut_modules = list(self.shortcut.children())
        if (len(shortcut_modules) > 0) and (type(shortcut_modules[0]) is MaskedConv2d):
            if_projection_shortcut = True
        if_identity_shortcut = not if_projection_shortcut
        pre_conv1 = x
        additives_dict_cur = {'conv1':{}, 'shortcut':{}, 'shortcut_id':{}, 'conv2':{}}

        if not(type(self.conv1) is MaskedConv2d): #the 0-th block
            post_conv1 = self.conv1(pre_conv1)
            post_conv1_bn = F.relu(self.bn1(post_conv1))
        else:
            corresponding_layer_ind_conv1 = (BasicBlockAvgAdditivesCreator.id - 1) * 2 #0-th block is without connectivities
            corresponding_layer_ind_conv2 = corresponding_layer_ind_conv1 + 1
            relevant_conv1_input_idx = set(BasicBlockAvgAdditivesCreator.get_indices_of_used_neurons(corresponding_layer_ind_conv1))
            relevant_conv2_input_idx = set(BasicBlockAvgAdditivesCreator.get_indices_of_used_neurons(corresponding_layer_ind_conv2))
                #conv2_input is other name for conv1_output
            if if_identity_shortcut:  # don't need the processing per outgoing connection in the 'for' below
                for i in relevant_conv1_input_idx:
                    additives_dict_cur['shortcut_id'][i] = x[:, i:i+1, :, :].mean(dim=0, keepdim=True)
            if self.conv1.ordinary_conv.bias is not None:
                conv1_bias_stored = self.conv1.ordinary_conv.bias.detach().clone()
                self.conv1.ordinary_conv.bias *= 0
                if if_projection_shortcut:
                    shortcut_bias_stored = shortcut_modules[0].ordinary_conv.bias.detach().clone()
                    shortcut_modules[0].ordinary_conv.bias *= 0
            for i in relevant_conv1_input_idx:
                connected_to_i_idx = torch.where(self.conv1.connectivity[0][:, i] > 0.5)[0].cpu().numpy()
                connected_to_i_idx = set(connected_to_i_idx).intersection(relevant_conv2_input_idx)
                if_only_shortcut_conn_exists = len(connected_to_i_idx) == 0#this means that only shortcut connections exists

                temp = self.conv1.connectivity[0].detach().clone()
                self.conv1.connectivity[0] *= 0.0
                if not if_only_shortcut_conn_exists:
                    connected_to_i_idx = np.array(list(connected_to_i_idx))
                    self.conv1.connectivity[0][connected_to_i_idx, i] += 1.0

                post_conv1_only_cur_input = self.conv1(pre_conv1)
                if if_projection_shortcut:
                    post_shortcut_only_cur_input = shortcut_modules[0](x)
                    additives_dict_cur['shortcut'][i] = {}

                additives_dict_cur['conv1'][i] = {}

                if not if_only_shortcut_conn_exists:
                    for j in connected_to_i_idx:
                        additives_dict_cur['conv1'][i][j] = post_conv1_only_cur_input[:, j:j+1, :, :].mean(axis=0, keepdim=True)

                if if_projection_shortcut:
                    corresponding_layer_ind_nextlayer = corresponding_layer_ind_conv2 + 1
                    relevant_nextlayer_idx = set(
                        BasicBlockAvgAdditivesCreator.get_indices_of_used_neurons(corresponding_layer_ind_nextlayer))
                    for j in relevant_nextlayer_idx:
                        additives_dict_cur['shortcut'][i][j] = post_shortcut_only_cur_input[:, j:j + 1, :, :].mean(axis=0, keepdim=True)

                self.conv1.connectivity[0].data = temp.data

            if self.conv1.ordinary_conv.bias is not None:
                self.conv1.ordinary_conv.bias.data = conv1_bias_stored
                if if_projection_shortcut:
                    shortcut_modules[0].ordinary_conv.bias.data = shortcut_bias_stored
            post_conv1 = self.conv1(pre_conv1)
            post_conv1_bn = F.relu(self.bn1(post_conv1))

        if not (type(self.conv2) is MaskedConv2d):
            post_conv2 = self.conv2(post_conv1_bn)
            post_conv2_bn = self.bn2(post_conv2)
        else:
            if self.conv2.ordinary_conv.bias is not None:
                conv2_bias_stored = self.conv2.ordinary_conv.bias.detach().clone()
                self.conv2.ordinary_conv.bias *= 0

            corresponding_layer_ind_nextlayer = corresponding_layer_ind_conv2 + 1
            relevant_nextlayer_idx = set(BasicBlockAvgAdditivesCreator.get_indices_of_used_neurons(corresponding_layer_ind_nextlayer))

            for i in relevant_conv2_input_idx:
                connected_to_i_idx = torch.where(self.conv2.connectivity[0][:, i] > 0.5)[0].cpu().numpy()
                connected_to_i_idx = set(connected_to_i_idx).intersection(relevant_nextlayer_idx)
                if len(connected_to_i_idx) == 0:#this means that only shortcut connections exists => implement later
                    continue
                connected_to_i_idx = np.array(list(connected_to_i_idx))

                temp = self.conv2.connectivity[0].detach().clone()
                self.conv2.connectivity[0] *= 0.0
                self.conv2.connectivity[0][connected_to_i_idx, i] += 1.0

                post_conv2_only_cur_input = self.conv2(post_conv1_bn)
                additives_dict_cur['conv2'][i] = {}
                for j in connected_to_i_idx:
                    additives_dict_cur['conv2'][i][j] = post_conv2_only_cur_input[:, j:j+1, :, :].mean(dim=0, keepdim=True)

                self.conv2.connectivity[0].data = temp.data

            if self.conv2.ordinary_conv.bias is not None:
                self.conv2.ordinary_conv.bias.data = conv2_bias_stored
            post_conv2 = self.conv2(post_conv1_bn)
            post_conv2_bn = self.bn2(post_conv2)

        out = post_conv2_bn

        if if_projection_shortcut:
            shortcut_out = shortcut_modules[0](x)
            shortcut_out = shortcut_modules[1](shortcut_out)
            out += shortcut_out
        elif if_identity_shortcut:
            out += self.shortcut(x)
        out = F.relu(out)

        BasicBlockAvgAdditivesCreator.additives_dict[BasicBlockAvgAdditivesCreator.id] = additives_dict_cur
        BasicBlockAvgAdditivesCreator.id += 1
        return out



class BasicBlockAvgAdditivesUser(BasicBlock):
    '''
    this class is for converting non-constant inputs to constant (average) additives
    current limitations:
        - shortcuts aren't handled properly
        - connections to task heads aren't handled properly
    '''
    expansion = 1
    id = 0
    additives_dict = {}
    connections_to_remove = defaultdict(list) # layer -> [(from1, to1), (from2, to2),..]

    def __init__(self, in_planes, planes, stride=1, connectivity=(None, None), if_enable_bias=False):
        super().__init__(in_planes, planes, stride, connectivity, if_enable_bias)
        self.id = self.__class__.id
        self.__class__.id += 1

    def forward(self, x):
        if_projection_shortcut = False
        shortcut_modules = list(self.shortcut.children())
        if (len(shortcut_modules) > 0) and (type(shortcut_modules[0]) is MaskedConv2d):
            if_projection_shortcut = True
        if_identity_shortcut = not if_projection_shortcut
        pre_conv1 = x
        additives_dict_cur = self.__class__.additives_dict[self.id]

        if not(type(self.conv1) is MaskedConv2d): #the 0-th block
            post_conv1 = self.conv1(pre_conv1)
            post_conv1_bn = F.relu(self.bn1(post_conv1))
        else:
            corresponding_layer_ind_conv1 = (self.id - 1) * 2 #0-th block is without connectivities
            conns_to_remove_cur = self.__class__.connections_to_remove[corresponding_layer_ind_conv1]

            with torch.no_grad():
                for (from_idx, to_idx) in conns_to_remove_cur:
                    self.conv1.connectivity[0][to_idx, from_idx] *= 0.0
            post_conv1 = self.conv1(pre_conv1)
            for (from_idx, to_idx) in conns_to_remove_cur:
                post_conv1[:, to_idx, :, :] += additives_dict_cur['conv1'][from_idx][to_idx][0]#.max() / 3
                    #the last index 0 above is because the stored shape is [1,stuff], and LHS is just [stuff]
            post_conv1_bn = F.relu(self.bn1(post_conv1))

        if not (type(self.conv2) is MaskedConv2d):
            post_conv2 = self.conv2(post_conv1_bn)
            post_conv2_bn = self.bn2(post_conv2)
        else:
            corresponding_layer_ind_conv2 = corresponding_layer_ind_conv1 + 1
            conns_to_remove_cur = self.__class__.connections_to_remove[corresponding_layer_ind_conv2]

            with torch.no_grad():
                for (from_idx, to_idx) in conns_to_remove_cur:
                    self.conv2.connectivity[0][to_idx, from_idx] = 0.0
            post_conv2 = self.conv2(post_conv1_bn)
            for (from_idx, to_idx) in conns_to_remove_cur:
                post_conv2[:, to_idx, :, :] += additives_dict_cur['conv2'][from_idx][to_idx][0]#.max() / 3
            post_conv2_bn = self.bn2(post_conv2)

        out = post_conv2_bn

        if not (type(self.conv1) is MaskedConv2d):
            out += self.shortcut(x)
        else:
            if if_projection_shortcut:
                conns_to_remove_shortcut = self.__class__.connections_to_remove['shortcut'][corresponding_layer_ind_conv1]
                shortcut_out = shortcut_modules[0](x)
                shortcut_out = shortcut_modules[1](shortcut_out)
                for (from_idx, to_idx) in conns_to_remove_shortcut:
                    shortcut_out[:, to_idx, :, :] = additives_dict_cur['shortcut'][from_idx][to_idx]
                out += shortcut_out
            elif if_identity_shortcut:
                conns_to_remove_shortcut = self.__class__.connections_to_remove['shortcut'][corresponding_layer_ind_conv1]
                shortcut_out = self.shortcut(x)
                for (from_idx, to_idx) in conns_to_remove_shortcut:
                    assert from_idx == to_idx
                    shortcut_out[:, to_idx, :, :] = additives_dict_cur['shortcut_id'][to_idx]
                out += shortcut_out

        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, connectivity=None):
        super(Bottleneck, self).__init__()
        print('ATTENTION! This block wasn"t updated')
        if connectivity is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = MaskedConv2d(in_planes, planes, connectivity=connectivity, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # TODO: I don't know if in case of BinarizeBySample having sampling done in 2 places independently helps or hinders.
            if connectivity is None:
                conv_to_use = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            else:
                conv_to_use = MaskedConv2d(in_planes, self.expansion * planes, connectivity=connectivity, kernel_size=1,
                                           stride=stride, bias=False)
            self.shortcut = nn.Sequential(
                conv_to_use,
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BinMatrResNet(nn.Module):
    def __init__(self, block, num_blocks, num_chunks, width_mul, if_fully_connected,
                 dataset_type='celeba', num_tasks=40, input_size='default', auxillary_connectivities_for_id_shortcut=None,
                 if_enable_bias=False, replace_constants_last_layer_mode=None, replace_with_avgs_last_layer_mode=None):
        super(BinMatrResNet, self).__init__()
        self.block = [block]
        self.if_fully_connected = if_fully_connected
        self.in_planes = 64
        self.num_tasks = num_tasks
        self._create_connectivity_parameters(num_chunks)
        self.dataset_type = dataset_type
        self.input_size = input_size
        if_restore_aux = auxillary_connectivities_for_id_shortcut is not None
        self.replace_constants_last_layer_mode = replace_constants_last_layer_mode
        self.replace_with_avgs_last_layer_mode = replace_with_avgs_last_layer_mode
        assert (replace_constants_last_layer_mode is None) or (replace_with_avgs_last_layer_mode is None)

        stride = 1
        if self.dataset_type == 'imagenette':
            stride = 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stride, padding=1, bias=if_enable_bias)
        self.bn1 = nn.BatchNorm2d(64)
        if len(num_chunks) == 3:
            if if_restore_aux:
                raise NotImplementedError(f'Loading of auxillary connctivities not implemented for {num_chunks} chunks')
            self.layer1 = self._make_layer(block, 64 * width_mul, num_blocks[0], 1, [(None, None)] * num_blocks[0], if_enable_bias)
            self.layer2 = self._make_layer(block, 128 * width_mul, num_blocks[1], 2, [(None, None)] * num_blocks[1], if_enable_bias)
            # layer2 is separated into 8 blocks after this, at the start of layer3
            self.layer3 = self._make_layer(block, 256 * width_mul, num_blocks[2], 2,
                                           [(self.connectivities[0], None)] + [(None, None)] * (num_blocks[2] - 1), if_enable_bias)
            # layer3 is separated into 8 blocks after this, at the start of layer4
            self.layer4 = self._make_layer(block, 512 * width_mul, num_blocks[3], 2,
                                           [(self.connectivities[1], None)] + [(None, None)] * (num_blocks[3] - 1), if_enable_bias)
            # layer4 is separated into 8 blocks during forward pass, when we get 40 outputs
            # it is there where connectivities[-1] is used
        elif len(num_chunks) == 8:
            if if_restore_aux:
                raise NotImplementedError(f'Loading of auxillary connctivities not implemented for {num_chunks} chunks')

            n_used_conns = 0
            self.layer1 = self._make_layer(block, 64 * width_mul, num_blocks[0], 1,
                                           [None] + list(
                                               self.connectivities[n_used_conns:n_used_conns + num_blocks[0] - 1]))
            n_used_conns += num_blocks[0] - 1
            self.layer2 = self._make_layer(block, 128 * width_mul, num_blocks[1], 2,
                                           self.connectivities[n_used_conns:n_used_conns + num_blocks[1]])
            n_used_conns += num_blocks[1]
            self.layer3 = self._make_layer(block, 256 * width_mul, num_blocks[2], 2,
                                           self.connectivities[n_used_conns:n_used_conns + num_blocks[2]])
            n_used_conns += num_blocks[2]
            self.layer4 = self._make_layer(block, 512 * width_mul, num_blocks[3], 2,
                                           self.connectivities[n_used_conns:n_used_conns + num_blocks[3]])
            n_used_conns += num_blocks[3]
        elif len(num_chunks) == 15:
            n_used_conns = 0

            conn_range = self.connectivities[n_used_conns:n_used_conns + (num_blocks[0] - 1) * 2]
            assert len(
                conn_range) % 2 == 0  # they should always come in pairs, and if they don't the last element is silently discarded, which is bad
            if if_restore_aux:
                aux_range = auxillary_connectivities_for_id_shortcut[
                            n_used_conns:n_used_conns + (num_blocks[0] - 1) * 2]
                all_conns = [(None, None, None)] + list(zip(conn_range[::2], conn_range[1::2], aux_range[::2]))
            else:
                all_conns = [(None, None)] + list(zip(conn_range[::2], conn_range[1::2]))
            self.layer1 = self._make_layer(block, int(64 * width_mul), num_blocks[0], 1, all_conns, if_enable_bias)
            n_used_conns += (num_blocks[0] - 1) * 2
            print(f'n_used_conns = {n_used_conns}')

            conn_range = self.connectivities[n_used_conns:n_used_conns + num_blocks[1] * 2]
            assert len(conn_range) % 2 == 0
            if if_restore_aux:
                aux_range = auxillary_connectivities_for_id_shortcut[n_used_conns:n_used_conns + num_blocks[1] * 2]
                all_conns = list(zip(conn_range[::2], conn_range[1::2], aux_range[::2]))
            else:
                all_conns = list(zip(conn_range[::2], conn_range[1::2]))
            self.layer2 = self._make_layer(block, int(128 * width_mul), num_blocks[1], 2, all_conns, if_enable_bias)
            n_used_conns += num_blocks[1] * 2
            print(f'n_used_conns = {n_used_conns}')

            conn_range = self.connectivities[n_used_conns:n_used_conns + num_blocks[2] * 2]
            assert len(conn_range) % 2 == 0
            if if_restore_aux:
                aux_range = auxillary_connectivities_for_id_shortcut[n_used_conns:n_used_conns + num_blocks[2] * 2]
                all_conns = list(zip(conn_range[::2], conn_range[1::2], aux_range[::2]))
            else:
                all_conns = list(zip(conn_range[::2], conn_range[1::2]))
            self.layer3 = self._make_layer(block, int(256 * width_mul), num_blocks[2], 2, all_conns, if_enable_bias)
            n_used_conns += num_blocks[2] * 2
            print(f'n_used_conns = {n_used_conns}')

            conn_range = self.connectivities[n_used_conns:n_used_conns + num_blocks[3] * 2]
            assert len(conn_range) % 2 == 0
            if if_restore_aux:
                aux_range = auxillary_connectivities_for_id_shortcut[n_used_conns:n_used_conns + num_blocks[3] * 2]
                all_conns = list(zip(conn_range[::2], conn_range[1::2], aux_range[::2]))
            else:
                all_conns = list(zip(conn_range[::2], conn_range[1::2]))
            self.layer4 = self._make_layer(block, int(512 * width_mul), num_blocks[3], 2, all_conns, if_enable_bias)
            n_used_conns += num_blocks[3] * 2
            print(f'n_used_conns = {n_used_conns}')
        else:
            raise ValueError(f'Unexpected number of chunks: {num_chunks}')

    def _make_layer(self, block, planes, num_blocks, stride, connectivity_per_block, if_enable_bias):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, connectivity_per_block[i], if_enable_bias))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _create_connectivity_parameters(self, num_chunks):
        self.connectivities = []
        for i in range(len(num_chunks) - 1):
            connectivity_shape = (num_chunks[i + 1], num_chunks[i])
            if self.if_fully_connected:
                cur_conn = torch.ones(connectivity_shape, requires_grad=False).to(device)
            else:
                # cur_conn = torch.nn.Parameter(torch.rand(connectivity_shape, requires_grad=True).to(device) )#* 2 - 1)
                cur_conn = torch.nn.Parameter(
                    torch.ones(connectivity_shape, requires_grad=True).to(device) * 0.75)  # * 2 - 1)
                # self.register_parameter(f'chunk_connectivity_{i}', cur_conn)
            self.connectivities.append(cur_conn)

        connectivity_shape = (self.num_tasks, num_chunks[-1])
        if self.if_fully_connected:
            if True:
                task_conn = torch.ones(connectivity_shape, requires_grad=False).to(device)
            else:
                # random connectivity: one connection per task
                task_conn = torch.zeros(connectivity_shape, requires_grad=False).to(device)
                for i in range(tasks_num):
                    sampled_conn = np.random.randint(num_chunks[-1])
                    task_conn[i, sampled_conn] = 1
        else:
            # task_conn = torch.nn.Parameter(torch.rand(connectivity_shape, requires_grad=True).to(device) )#* 2 - 1)
            task_conn = torch.nn.Parameter(
                torch.ones(connectivity_shape, requires_grad=True).to(device) * 0.75)  # * 2 - 1)
            # self.register_parameter(f'chunk_connectivity_{total_blocks_num-1}', task_conn)
            # TODO: I set this only for graph visualization, change back ASAP
            # self.fake_task_conn = task_conn
        self.connectivities.append(task_conn)

        if True:
            self.connectivity_comeback_multipliers = []
            for conn in self.connectivities:
                self.connectivity_comeback_multipliers.append(torch.ones_like(conn).to(device))

    def forward(self, x):  # , ignored_filters_per_layer):
        if True:
            if not self.if_fully_connected:
                max_val = 1.0
                min_val = 0.5
                with torch.no_grad():
                    for connectivity, conn_comeback_mul in zip(self.connectivities, self.connectivity_comeback_multipliers):
                        if False:
                            connectivity[connectivity <= min_val] = 0.0
                        else:
                            if self.training: # because we don't wanna change connections during validation
                                idx = connectivity <= min_val
                                additive = (0.005 - 0.01 * (min_val - connectivity).abs()[idx]) * 0.05
                                cur_comeback_mul = conn_comeback_mul[idx]
                                connectivity[idx] += additive * cur_comeback_mul
                                conn_comeback_mul[idx] *= 0.999
                        if False:
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
        if self.dataset_type == 'cifar':
            global_average_pooling_size = 4
        elif self.dataset_type == 'imagenette':
            global_average_pooling_size = 14 # 224x224
        elif self.dataset_type == 'celeba':
            # (24, 21) for 192x168 images, 16 for 128x128, 8 for 64x64
            if self.input_size == 'default':
                global_average_pooling_size = 8
            elif self.input_size == 'bigimg':
                global_average_pooling_size = 16
            elif self.input_size == 'biggerimg':
                global_average_pooling_size = (24, 21)
        else:
            raise NotImplementedError()
        out = F.avg_pool2d(out, global_average_pooling_size)
        out = out.view(out.size(0), -1)[:, :, None]

        outs = []
        connectivity = self.connectivities[-1]
        filter_to_chunks_factor = (out.size(1) // connectivity.size(1), 1)
        chunks_to_filters_for_kronecker = torch.ones(filter_to_chunks_factor).to(device)

        if (self.replace_constants_last_layer_mode == 'store'):
            assert filter_to_chunks_factor == (1, 1) #this makes sense only when 1 block == 1 filter
            const_features_idx = []
            for i in range(out.size(1)):
                cur_feature = out[:, i]
                if are_activations_for_each_input_equal(cur_feature):
                    const_features_idx.append(i)
            self.last_layer_additives = torch.zeros(self.num_tasks, out.size(1)).to(device)
            for i in const_features_idx:
                self.last_layer_additives[torch.where(connectivity[:, i] > 0.5)[0], i] = out[0, i]
                #crude self-check
                assert out[0, i] == out[17, i]
                connectivity[:, i] *= 0.0
        elif self.replace_with_avgs_last_layer_mode == 'store':
            assert filter_to_chunks_factor == (1, 1) #this makes sense only when 1 block == 1 filter
            self.last_layer_additives = out.mean(dim=0).squeeze()

        binarized_connectivity = BinarizeByThreshold.apply(connectivity)

        for i in range(self.num_tasks):
            if filter_to_chunks_factor == (1, 1):
                cur_connectivity = binarized_connectivity[i][:, None]
            else:
                cur_connectivity = kronecker(binarized_connectivity[i][:, None], chunks_to_filters_for_kronecker)

            if (self.replace_with_avgs_last_layer_mode == 'restore'):
                conns_to_remove_all = self.block[0].connections_to_remove['label']
                conns_to_remove_cur = []
                for neuron_from, neuron_to in conns_to_remove_all:
                    if neuron_to == i:
                        cur_connectivity[neuron_from] = 0.0
                        conns_to_remove_cur.append(neuron_from)
                conns_to_remove_cur = np.array(conns_to_remove_cur)

            out_cur = (out * cur_connectivity).squeeze(-1)

            if (self.replace_with_avgs_last_layer_mode == 'restore'):
                if len(conns_to_remove_cur) > 0:
                    out_cur[:, conns_to_remove_cur] += self.last_layer_additives[conns_to_remove_cur].unsqueeze(0).repeat(out_cur.size(0), 1)
            # if i in [0,2,5,15,28]:
            #     print("AAAAAAAAAAAAAAAAAAAAAAAAA, DOING SOMETHING REALLY STUPID")
            #     out_cur[:, 148] = 0.8824
            if (self.replace_constants_last_layer_mode == 'store') or (self.replace_constants_last_layer_mode == 'restore'):
                out_cur += self.last_layer_additives[i].unsqueeze(0).repeat(out_cur.size(0), 1)

            outs.append(out_cur)
        return outs


class FaceAttributeDecoder(nn.Module):
    def __init__(self, dim=512):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(dim, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask


class FaceAttributeDecoderCifar10(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoderCifar10, self).__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask

class FaceAttributeDecoderCifar10SingleTask(nn.Module):
    def __init__(self, dim=512):
        super(FaceAttributeDecoderCifar10SingleTask, self).__init__()
        self.linear = nn.Linear(dim, 10)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask


class FaceAttributeDecoderImagenetteSingleTask(nn.Module):
    def __init__(self, dim=512):
        super(FaceAttributeDecoderImagenetteSingleTask, self).__init__()
        self.linear = nn.Linear(dim, 10)

    def forward(self, x, mask):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out, mask

    # for lr finder:
    # def forward(self, x):
    #     x = self.linear(x[0])
    #     out = F.log_softmax(x, dim=1)
    #     return out