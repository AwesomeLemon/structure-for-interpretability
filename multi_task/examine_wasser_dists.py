#   lipstick only
from collections import defaultdict

import json
import torchvision
import torch
import pandas as pd
import numpy as np

from multi_task.load_model import load_trained_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = 'imagenet_weights'#'baseline'#'wearing_lipstick'#
if experiment == 'wearing_lipstick':
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/19_43_on_September_21/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._120_model.pkl'
    param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_lipstickonly_weightedce.json'
    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if_pretrained_imagenet = False
elif experiment == 'baseline':
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_51_on_May_21/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_15_model.pkl'
    param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc.json'
    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if_pretrained_imagenet = False
elif experiment == 'imagenet_weights':
    if_pretrained_imagenet = True

if not if_pretrained_imagenet:
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
    try:
        model = load_trained_model(param_file, save_model_path)
    except:
        print('assume problem where cifar networks ignored the enable_bias parameter')
        model = load_trained_model(param_file, save_model_path, if_actively_disable_bias=True)
else:
    im_size = (224, 224)
    model = torchvision.models.__dict__['resnet18'](pretrained=True).to(device)

# w = model['all'].linear.weight.cpu().detach().numpy()
if experiment == 'wearing_lipstick':
    wasser_dists = np.load('wasser_dists/wasser_dist_attr_hist_lipstickonly_v2_14.npy', allow_pickle=True).item()
    wasser_dists = np.array(pd.DataFrame(wasser_dists))
    for i in range(40):
        wd_cur = wasser_dists[i]
        print(i, wd_cur[wd_cur > 0].mean(), wd_cur[wd_cur > 0].std())
elif experiment == 'imagenet_weights':
    wasser_dists = np.load('wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test_14.npy', allow_pickle=True).item()
    wasser_dists = np.array(pd.DataFrame(wasser_dists))
    w = model.fc.weight.cpu().detach().numpy()
    corrs = []
    for neuron in range(512):
        w_cur = w[:, neuron]
        dist_cur = wasser_dists[:, neuron]
        corrs.append(np.corrcoef(w_cur, dist_cur)[0, 1]) #scipy.stats.spearmanr(w_cur, dist_cur)[0]
    corrs = np.array(corrs)
    print(np.mean(corrs), np.std(corrs))
elif experiment == 'baseline':
    wasser_dists = np.load('wasser_dists/wasser_dist_attr_hist_bettercifar10single_14.npy', allow_pickle=True).item()
    wasser_dists = np.array(pd.DataFrame(wasser_dists))

print(1)