# Adapted from: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
# and then adapted from multiobjective optimization repo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import models.resnet_mit as resnet
device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")

try:
    from models.binmatr2_multi_faces_resnet import BinarizeByThreshold, kronecker
except:
    from multi_task.models.binmatr2_multi_faces_resnet import BinarizeByThreshold, kronecker


def get_segmentation_encoder():
    orig_resnet = resnet.__dict__['resnet50'](pretrained=True)
    return ResnetDilated(orig_resnet, dilate_scale=8)
    

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class ResNetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, if_resnet50=False):
        super(ResNetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2)) #todo: for some reason spatial dimensions of output are not influenced by these parameters
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))


        #Danger: my conv1 & bn1 are for smaller images => different params
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #my version of resnet for smaller images, ie without maxpool
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.if_fully_connected = orig_resnet.if_fully_connected
        self.connectivities = orig_resnet.connectivities
        self.if_cifar = orig_resnet.if_cifar
        self.num_tasks = orig_resnet.num_tasks

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if 'MaskedConv2d' in classname:
            m = m.ordinary_conv
        if 'Conv' in classname:
            print(m)
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            print(m)
            print()

    def forward(self, x):
        if not self.if_fully_connected:
            max_val = 1.0
            min_val = 0.5
            with torch.no_grad():
                for connectivity in self.connectivities:
                    # connectivity[connectivity <= min_val] += 0.0003
                    connectivity[connectivity <= min_val] = min_val
                    connectivity[connectivity > max_val] = max_val
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        outs = []

        connectivity = self.connectivities[-1]
        filter_to_chunks_factor = (x.size(1) // connectivity.size(1), 1) # x is [batch_size, filters_size, W, H]
        chunks_to_filters_for_kronecker = torch.ones(filter_to_chunks_factor).to(device)

        binarized_connectivity = BinarizeByThreshold.apply(connectivity)

        for i in range(self.num_tasks):
            cur_connectivity = kronecker(binarized_connectivity[i][:, None], chunks_to_filters_for_kronecker)
            outs.append((x * cur_connectivity.unsqueeze(0).unsqueeze(-1)).squeeze(-1))
        return outs

# pyramid pooling, bilinear upsample
class SegmentationDecoder(nn.Module):
    def __init__(self, num_class=21, fc_dim=2048, pool_scales=(1, 2, 3, 6), task_type='C'):
        super(SegmentationDecoder, self).__init__()

        self.task_type = task_type

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, mask):
        # conv5 = conv_out[-1]
        conv5 = conv_out

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.task_type == 'C':
            x = nn.functional.log_softmax(x, dim=1)
        return x, mask

