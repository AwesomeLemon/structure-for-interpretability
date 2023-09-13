import operator

import io

import json
import os
import pickle
from collections import defaultdict

import random
import torchvision
from functools import partial
from pathlib import Path

import PIL
import numpy as np
import skimage
import sklearn.cluster
import sklearn.manifold
import sklearn.mixture
import sklearn.preprocessing
import torch
from PIL import Image, ImageFont
from PIL import ImageDraw
from matplotlib import pyplot as plt
from sortedcontainers import SortedDict
from torch.nn.functional import softmax
from torch.utils import data
import pandas as pd
import scipy.stats
from sklearn.metrics import balanced_accuracy_score, accuracy_score


try:
    from multi_task import datasets
    from multi_task.gan.attgan.data import CustomDataset
    # from multi_task.gan.change_attributes import AttributeChanger
    from multi_task.load_model import load_trained_model, eval_trained_model
    from multi_task.loaders.celeba_loader import CELEBA
    from multi_task.util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict
    from multi_task.util.util import *

    from multi_task.models.binmatr2_multi_faces_resnet import BasicBlockAvgAdditivesUser
except:
    import datasets
    from gan.attgan.data import CustomDataset
    # from gan.change_attributes import AttributeChanger
    from load_model import load_trained_model
    from loaders.celeba_loader import CELEBA
    from util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict
    from util.util import *

    from models.binmatr2_multi_faces_resnet import BasicBlockAvgAdditivesUser
from efficientnet_pytorch import EfficientNet

import glob
from shutil import copyfile, copy
import math
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExplorativeBasicBlock(torchvision.models.resnet.BasicBlock):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer = None) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        delattr(self, 'relu')
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.relu2 = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out  


class ModelExplorer:
    def __init__(self, save_model_path, param_file, model_to_use='my'):
        if model_to_use == 'my':
            with open(param_file) as json_params:
                params = json.load(json_params)
            if 'input_size' not in params:
                params['input_size'] = 'default'
            if params['input_size'] == 'default':
                im_size = (64, 64)
                config_path = 'configs.json'
            elif params['input_size'] == 'bigimg':
                im_size = (128, 128)
                config_path = 'configs_big_img.json'
            elif params['input_size'] == 'biggerimg':
                im_size = (192, 168)
                config_path = 'configs_bigger_img.json'
            with open(config_path) as config_params:
                configs = json.load(config_params)
            removed_conns = defaultdict(list)
            removed_conns['shortcut'] = defaultdict(list)
            if_replace_by_zeroes = False  # True
            if False:
                print('Some connections were disabled')
                # removed_conns[5] = [(5, 23), (5, 25), (5, 58)]
                # removed_conns[8] = [(137, 143)]
                # removed_conns[9] = [(142, 188), (216, 188)]
                removed_conns[10] = [(188, 104),
                                     # (86, 104)
                                     ]
                # removed_conns['label'] = [(481, 3),(481, 7)]
                # removed_conns[9] = [#(216, 181)
                #                     (142, 188),
                #                     (216, 188),
                #                     (187, 86),
                #                     (224, 86)
                #                     ]
                # removed_conns[10] = [(181, 279)]
                # removed_conns[11] = [(28, 421)] + [(331, 392)]
                # removed_conns[12] = [(406, 204)]
                # removed_conns['label'] = [  # (356, 9),
                #     # (204, 9),
                #     (126, 9),
                #     (187, 9),
                #     (123, 9),
                #     # (134, 9),
                #     #  (400, 9),
                #     #  (383, 9)
                # ]
                removed_conns['shortcut'][8] = [(86, 86)]
                # removed_conns['shortcut'][10] = [#(193, 125)
                #                                   (118, 125)
                #                                  ]

            if False:
                save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_avgadditives' + '.pkl'
                try:
                    trained_model = load_trained_model(param_file, save_model_path, if_additives_user=True,
                                                       if_store_avg_activations_for_disabling=True,
                                                       conns_to_remove_dict=removed_conns,
                                                       replace_with_avgs_last_layer_mode='restore',
                                                       if_replace_by_zeroes=if_replace_by_zeroes)
                except:
                    print('assume problem where cifar networks ignored the enable_bias parameter')
                    BasicBlockAvgAdditivesUser.id = 0 #need to reset
                    trained_model = load_trained_model(param_file, save_model_path, if_additives_user=True,
                                                       if_store_avg_activations_for_disabling=True,
                                                       conns_to_remove_dict=removed_conns,
                                                       replace_with_avgs_last_layer_mode='restore',
                                                       if_actively_disable_bias=True,
                                                       if_replace_by_zeroes=if_replace_by_zeroes)
            else:
                try:
                    trained_model = load_trained_model(param_file, save_model_path)
                except:
                    print('assume problem where cifar networks ignored the enable_bias parameter')
                    trained_model = load_trained_model(param_file, save_model_path, if_actively_disable_bias=True)

            model = trained_model
            self.params = params
            self.configs = configs
            use_my_model = True
        else:
            im_size = (224, 224)
            if model_to_use == 'resnet18':
                # trained_model = torchvision.models.__dict__['resnet18'](pretrained=True).to(device)
                trained_model = torchvision.models.resnet._resnet('resnet18', ExplorativeBasicBlock, [2, 2, 2, 2], pretrained=True, progress=True).to(device)
                # trained_model = torchvision.models.__dict__['resnet34'](pretrained=True).to(device)
                # trained_model = models.__dict__['vgg19_bn'](pretrained=True).to(device)
            if model_to_use == 'mobilenet':
                trained_model = torchvision.models.__dict__['mobilenet_v2'](pretrained=True).to(device)
            if model_to_use == 'efficientnet':
                trained_model = EfficientNet.from_pretrained('efficientnet-b3').to(device)
            model = trained_model
            self.params = None
            self.configs = None
            use_my_model = False

        self.model = model
        self.use_my_model = use_my_model
        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        if self.use_my_model:
            for m in self.model:
                model[m].zero_grad()
                model[m].eval()

            self.feature_extractor = self.model['rep']
            self.prediction_head = self.model['all'].linear
        else:
            self.model.eval()
            self.model.zero_grad()
            self.feature_extractor = self.model
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)