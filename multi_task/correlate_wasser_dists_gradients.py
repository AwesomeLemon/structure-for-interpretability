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
from util.util import *
import pandas as pd
import scipy.stats
from sklearn.metrics import balanced_accuracy_score, accuracy_score


try:
    from multi_task import datasets
    from multi_task.gan.attgan.data import CustomDataset
    from multi_task.gan.change_attributes import AttributeChanger
    from multi_task.load_model import load_trained_model, eval_trained_model
    from multi_task.loaders.celeba_loader import CELEBA
    from multi_task.util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict
    from multi_task.model_explorer import ModelExplorer
except:
    import datasets
    from gan.attgan.data import CustomDataset
    # from gan.change_attributes import AttributeChanger
    from load_model import load_trained_model
    from loaders.celeba_loader import CELEBA
    from util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict
    from model_explorer import ModelExplorer

from models.binmatr2_multi_faces_resnet import BasicBlockAvgAdditivesUser

import glob
from shutil import copyfile, copy
import math
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradWassersteinCorrelator(ModelExplorer):
    def __init__(self, save_model_path, param_file, model_to_use='my'):
        super().__init__(save_model_path, param_file, model_to_use)

    def calc_gradients_wrt_output_whole_network_all_tasks(self, loader, out_path, if_pretrained_imagenet=False,
                                                          layers = layers_bn_afterrelu,
                                                          neuron_nums=[64, 64, 64, 128, 128, 128, 128, 256, 256, 256,
                                                                       256, 512, 512, 512, 512],
                                                          if_rename_layers=True
                                                          ):
        print("Warning! Assume that loader returns in i-th batch only instances of i-th class")
        # for model in self.model.values():
        #     model.zero_grad()
        #     model.eval()

        # target_layer_names = [layer.replace('_', '.') for layer in layers]#layers_bn_afterrelu] #+ ['feature_extractor']
        target_layer_names = [layer.replace('_', '.') if if_rename_layers else layer for layer in layers]
        # neuron_nums = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        def save_activation(activations, name, mod, inp, out):
            if name in target_layer_names:
                if_out_is_list = type(out) == list #backbone output
                if if_out_is_list:
                    out = out[0] #single-head cifar
                # print(out.shape)
                out.requires_grad_(True)
                # if 'bn1' in name:
                #     out = F.relu(out)
                out.retain_grad()
                activations[name] = out
                if if_out_is_list:
                    out = [out]
                return out

        activations = {}
        hooks = []
        for name, m in self.feature_extractor.named_modules():
            if name in target_layer_names:
                hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))
        hooks.append(self.feature_extractor.register_forward_hook(partial(save_activation, activations, 'feature_extractor')))

        layer_names_for_pd = []
        neuron_indices_for_pd = []
        mean_grads_for_pd = defaultdict(list)
        if_already_saved_layer_names_and_neuron_indices = False
        n_classes = 10
        if if_pretrained_imagenet:
            n_classes = 1000
        iter_loader = iter(loader)
        for cond_idx in range(n_classes):
            print(cond_idx)
            batch = next(iter_loader)
            cur_grads = defaultdict(lambda : defaultdict(list)) # layer -> neuron -> grads from every batch (i.e. 1 scalar per batch)
            ims, labels = batch
            if False:
                mask = (labels == cond_idx)#(labels != cond_idx)#np.array([True] * len(labels))#
                print(labels)
                ims_masked = ims[mask,...]
                ims_masked = ims_masked.cuda()
            else:
                ims_masked = ims.cuda()
                print(labels)
            out = self.feature_extractor(ims_masked)
            if not if_pretrained_imagenet:
                #single-headed
                y = out[0]
                out_cond = self.model['all'].linear(y)
                out_cond[:, cond_idx].sum().backward()
            else:
                out[:, cond_idx].sum().backward()

            for layer_name in target_layer_names:
                print(layer_name)
                layer_grad = activations[layer_name].grad.detach().cpu()
                n_neurons = neuron_nums[target_layer_names.index(layer_name)]
                # print(layer_grad.shape[1], n_neurons)
                for target_neuron in range(n_neurons):
                    cur_grad = layer_grad[:, target_neuron]
                    try:
                        cur_grad = cur_grad.mean(axis=(-1, -2))
                    except:
                        pass
                    # cur_grad = np.sign(cur_grad)
                    # cur_grad[cur_grad < 0] = 0
                    cur_grad = cur_grad.mean().item()
                    cur_grads[layer_name][target_neuron].append(cur_grad)
                    if not if_already_saved_layer_names_and_neuron_indices:
                        layer_names_for_pd.append(layer_name)
                        neuron_indices_for_pd.append(target_neuron)

                activations[layer_name].grad.zero_()

            if_already_saved_layer_names_and_neuron_indices = True # is set after the first batch of the first cond_idx

            for layer_name in target_layer_names:
                n_neurons = neuron_nums[target_layer_names.index(layer_name)]
                for target_neuron in range(n_neurons):
                    grad_meaned = np.mean(cur_grads[layer_name][target_neuron])
                    mean_grads_for_pd[cond_idx].append(grad_meaned)

        for hook in hooks:
            hook.remove()

        data = []
        for i in range(len(neuron_indices_for_pd)):
            data.append(
                [layer_names_for_pd[i], neuron_indices_for_pd[i]] + [mg[i] for mg in mean_grads_for_pd.values()])
        df = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] + list(range(n_classes)))
        df.to_pickle(out_path)

        return df

    def correlate_grads_with_wass_dists_per_neuron(self, df_grads_path, out_path, wass_dists_path_prefix,
                                                   if_replace_wass_dists_with_noise=False, target_indices=None,
                                                   layers=layers_bn_afterrelu, neuron_nums=[64, 64, 64, 128, 128, 128,
                                                                        128, 256, 256, 256, 256, 512, 512, 512, 512],
                                                   if_rename_layers=True, layer_ind_additive=0,
                                                   if_only_for_negative_mwds=False):
        # target_layer_names = [layer.replace('_', '.') for layer in layers]
        target_layer_names = np.array([layer.replace('_', '.') if if_rename_layers else layer for layer in layers])

        df_grads = pd.read_pickle(df_grads_path)
        data_corrs = []
        if if_replace_wass_dists_with_noise:
            wasser_dists_np = np.random.laplace(0, 0.05, (10, 512))
        for i, layer_name in enumerate(target_layer_names):
            i += layer_ind_additive
            print(layer_name)
            if target_indices is not None:
                if i not in target_indices:
                    continue
            if not if_replace_wass_dists_with_noise:
                wasser_dists = np.load(f'{wass_dists_path_prefix}_{i}.npy', allow_pickle=True).item()
                wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
                # wasser_dists_np[wasser_dists_np > 0] = 0
            # else:
            #     if True:
            #         wasser_dists_np = np.random.laplace(0, 0.05, (10, neuron_nums[i]))
            #     else:
            #         wasser_dists = np.load(f'{wass_dists_path_prefix}_{i}.npy', allow_pickle=True).item()
            #         wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
            #         def shuffle_along_axis(a, axis):
            #             idx = np.random.rand(*a.shape).argsort(axis=axis)
            #             return np.take_along_axis(a, idx, axis=axis)
            #         wasser_dists_np = shuffle_along_axis(wasser_dists_np, axis=1)
            for neuron in range(neuron_nums[i]):
                print(neuron)
                if neuron >= wasser_dists_np.shape[1]:
                    print('Something strange is afoot', neuron, wasser_dists_np.shape[1])
                    data_corrs.append([layer_name, neuron, 0])
                    break
                # if (np.abs(wasser_dists_np[:, neuron]) >= 0.09).sum() < 3:
                #     continue
                cur_grads = np.array(df_grads.loc[(df_grads['layer_name'] == layer_name)
                             & (df_grads['neuron_idx'] == neuron)].drop(['layer_name', 'neuron_idx'], axis=1))
                mwds_neuron = wasser_dists_np[:, neuron]
                cur_grads = cur_grads.squeeze()
                if if_only_for_negative_mwds:
                    idx = mwds_neuron < 0
                    mwds_neuron = mwds_neuron[idx]
                    cur_grads = cur_grads[idx]
                cur_corr = np.corrcoef(mwds_neuron, cur_grads)[0, 1]
                # cur_corr = scipy.stats.spearmanr(wasser_dists_np[:, neuron], cur_grads.squeeze())[0]
                data_corrs.append([layer_name, neuron, cur_corr])
        df_corr = pd.DataFrame(data_corrs, columns=['layer_name', 'neuron_idx', 'corr'])
        df_corr.to_pickle(out_path)

    def correlate_grads_with_wass_dists_per_task(self, df_grads_path, out_path, wass_dists_path_prefix):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn]
        df_grads = pd.read_pickle(df_grads_path)
        data_corrs = []
        for i, layer_name in enumerate(target_layer_names):
            wasser_dists = np.load(f'{wass_dists_path_prefix}_{i}.npy', allow_pickle=True).item()
            wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
            cur_grads = np.array(df_grads.loc[(df_grads['layer_name'] == layer_name)].drop(['layer_name', 'neuron_idx'], axis=1))
            for task in range(10):
                idx = wasser_dists_np[task] < -0.05#np.array([True] * wasser_dists_np[task].shape[0])#
                cur_corr = np.corrcoef(wasser_dists_np[task, idx], cur_grads[idx, task])[0, 1]
                # cur_corr = scipy.stats.spearmanr(wasser_dists_np[:, neuron], cur_grads.squeeze())[0]
                data_corrs.append([layer_name, task, cur_corr])
        df_corr = pd.DataFrame(data_corrs, columns=['layer_name', 'task_idx', 'corr'])
        df_corr.to_pickle(out_path)

    def plot_correlation_of_grads_with_wass_dists(self, df_corr_path, layers,
                                                  df_corr_paths_early=None, early_layers=None, if_rename_layers=True,
                                                  if_show=True):
        df_corr = pd.read_pickle(df_corr_path)
        # target_layer_names = [layer.replace('_', '.') for layer in layers]
        target_layer_names = np.array([layer.replace('_', '.') if if_rename_layers else layer for layer in layers])
        avg_corrs = []
        for layer_name in target_layer_names:
            cur_corrs = np.array(df_corr.loc[(df_corr['layer_name'] == layer_name)].drop(
                ['layer_name', 'neuron_idx'], axis=1))
            print(cur_corrs.shape)
            avg_corrs.append(np.nanmean(cur_corrs))
        if df_corr_paths_early is not None:
            df_corr = pd.read_pickle(df_corr_paths_early)
            target_layer_names = [layer.replace('_', '.') for layer in early_layers]
            avg_corrs_early = []
            for layer_name in target_layer_names:
                cur_corrs = np.array(df_corr.loc[(df_corr['layer_name'] == layer_name)].drop(
                    ['layer_name', 'neuron_idx'], axis=1))
                print(cur_corrs.shape)
                avg_corrs_early.append(np.nanmean(cur_corrs))
            avg_corrs = avg_corrs_early + avg_corrs

        x = np.arange(len(avg_corrs))
        # plt.figure(figsize=(8*1.2, 6*1.2))
        plt.plot(x, avg_corrs, '-o', label=df_corr_path.replace('.pkl', ''))
        # plt.xticks(x)
        print(avg_corrs)
        plt.ylim(0, 1)
        plt.xlabel('Layer index')
        plt.ylabel('Avg. correlation')
        if if_show:
            plt.legend()
            plt.show()

    def plot_correlation_of_grads_with_wass_dists_many_runs(self, df_corr_paths, layers,
                                                            df_corr_paths_early=None, early_layers=None):
        df_corrs = [pd.read_pickle(df_corr_path) for df_corr_path in df_corr_paths]
        target_layer_names = [layer.replace('_', '.') for layer in layers]
        avg_corrs_per_run = np.zeros((len(df_corrs), len(target_layer_names)))
        for i, df_corr in enumerate(df_corrs):
            for j, layer_name in enumerate(target_layer_names):
                cur_corrs = np.array(df_corr.loc[(df_corr['layer_name'] == layer_name)].drop(
                    ['layer_name', 'neuron_idx'], axis=1))
                avg_corrs_per_run[i, j] = np.mean(cur_corrs)
        means = np.mean(avg_corrs_per_run, axis=0)
        stds = np.std(avg_corrs_per_run, axis=0)

        if df_corr_paths_early is not None:
            df_corrs = [pd.read_pickle(df_corr_path) for df_corr_path in df_corr_paths_early]
            target_layer_names = [layer.replace('_', '.') for layer in early_layers]
            avg_corrs_per_run = np.zeros((len(df_corrs), len(early_layers)))
            for i, df_corr in enumerate(df_corrs):
                for j, layer_name in enumerate(target_layer_names):
                    cur_corrs = np.array(df_corr.loc[(df_corr['layer_name'] == layer_name)].drop(
                        ['layer_name', 'neuron_idx'], axis=1))
                    avg_corrs_per_run[i, j] = np.nanmean(cur_corrs)
            means_early = np.mean(avg_corrs_per_run, axis=0)
            stds_early = np.std(avg_corrs_per_run, axis=0)
            means = np.append(means_early, means)
            stds = np.append(stds_early, stds)

        x = np.arange(len(means))
        # plt.figure(figsize=(8*1.2, 6*1.2))
        plt.plot(x, means, '-o')
        plt.fill_between(x, means-stds, means+stds, facecolor='orange')
        plt.ylim(0, 1)
        # plt.title('Average correlation of W.dists & gradients\n averaged over 5 runs\n ± 1 std')
        # plt.xticks(x)
        plt.xlabel('Layer index')
        plt.ylabel('Avg. correlation')
        plt.show()

    def plot_grads_and_wass_dists(self, df_grads_path, out_path, wass_dists_path_prefix,
                                                   if_replace_wass_dists_with_noise=False, target_indices=None,
                                                   layers=layers_bn_afterrelu, neuron_nums=[64, 64, 64, 128, 128, 128,
                                                                        128, 256, 256, 256, 256, 512, 512, 512, 512],
                                                   if_rename_layers=True, layer_ind_additive=0,
                                                   if_only_for_negative_mwds=False):
        # target_layer_names = [layer.replace('_', '.') for layer in layers]
        target_layer_names = np.array([layer.replace('_', '.') if if_rename_layers else layer for layer in layers])

        df_grads = pd.read_pickle(df_grads_path)
        data_corrs = []
        if if_replace_wass_dists_with_noise:
            wasser_dists_np = np.random.laplace(0, 0.05, (10, 512))
        for i, layer_name in enumerate(target_layer_names):
            i += layer_ind_additive
            print(layer_name)
            if target_indices is not None:
                if i not in target_indices:
                    continue
            if not if_replace_wass_dists_with_noise:
                wasser_dists = np.load(f'{wass_dists_path_prefix}_{i}.npy', allow_pickle=True).item()
                wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
                # wasser_dists_np[wasser_dists_np > 0] = 0
            # else:
            #     if True:
            #         wasser_dists_np = np.random.laplace(0, 0.05, (10, neuron_nums[i]))
            #     else:
            #         wasser_dists = np.load(f'{wass_dists_path_prefix}_{i}.npy', allow_pickle=True).item()
            #         wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
            #         def shuffle_along_axis(a, axis):
            #             idx = np.random.rand(*a.shape).argsort(axis=axis)
            #             return np.take_along_axis(a, idx, axis=axis)
            #         wasser_dists_np = shuffle_along_axis(wasser_dists_np, axis=1)
            plt.figure(figsize=(10, 10))
            mwds_neuron_all = []
            cur_grads_all = []
            for neuron in range(neuron_nums[i]):
                print(neuron)
                if neuron >= wasser_dists_np.shape[1]:
                    print('Something strange is afoot', neuron, wasser_dists_np.shape[1])
                    data_corrs.append([layer_name, neuron, 0])
                    break
                # if (np.abs(wasser_dists_np[:, neuron]) >= 0.09).sum() < 3:
                #     continue
                cur_grads = np.array(df_grads.loc[(df_grads['layer_name'] == layer_name)
                             & (df_grads['neuron_idx'] == neuron)].drop(['layer_name', 'neuron_idx'], axis=1))
                mwds_neuron = wasser_dists_np[:, neuron]
                cur_grads = cur_grads.squeeze()
                if if_only_for_negative_mwds:
                    idx = mwds_neuron < 0
                    mwds_neuron = mwds_neuron[idx]
                    cur_grads = cur_grads[idx]
                # plt.scatter(mwds_neuron, cur_grads, marker='.', alpha=0.5)
                mwds_neuron_all.append(mwds_neuron)
                cur_grads_all.append(cur_grads)
                # if neuron > 100:
                #     break
            mwds_neuron_all = np.concatenate(mwds_neuron_all)
            print(mwds_neuron_all.shape)
            cur_grads_all = np.concatenate(cur_grads_all)
            print(cur_grads_all.shape)
            plt.scatter(mwds_neuron_all, cur_grads_all, marker='.', alpha=0.5)
            plt.xlabel('MWDs')
            plt.ylabel('Gradients')
            # plt.title(f'Neurons with negative MWDs, layer {i}, corr={np.corrcoef(mwds_neuron_all, cur_grads_all)[0, 1]:.2f}')
            plt.title(f'Neurons with all MWDs, layer {i}, corr={np.corrcoef(mwds_neuron_all, cur_grads_all)[0, 1]:.2f}')
            # plt.xlim(-0.01, 0.01)
            plt.show()

if __name__ == '__main__':
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
    # param_file = 'old_params/sample_all.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
    # param_file = 'params/bigger_reg_4_4_4.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_42_on_April_17/optimizer=SGD|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=True|use_pretrained_17_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgd001_pretrain_fc.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
    # fully-connected:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_51_on_May_21/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_15_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0_20_model.pkl'
    # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_nocondecay_comeback.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall2e-6_comeback_rescaled.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_6_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled.json'
    # sparse celeba
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_39_on_September_06/optimizer=SGD_Adam|batch_size=256|lr=0.005|connectivities_lr=0.001|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_120_model.pkl'
    # param_file = 'params/binmatr2_cifarfashionmnist_filterwise_sgdadam005+001_bias_condecayall2e-6.json'
    # hat only:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_56_on_September_09/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._67_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_wearinghatonly_weightedce.json'
    # bangs only:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_23_on_September_10/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._120_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_bangsonly_weightedce.json'
    # multi-head cifar:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_18_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4.json'
    #   single-head cifar
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    #   no bias cifar:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_55_on_September_18/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_120_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1_nobias_fc_batch128_weightdecay3e-4.json'
    #   very sparse cifar [single head]:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_28_on_September_21/optimizer=SGD_Adam|batch_size=128|lr=0.1|connectivities_lr=0.001|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_deca_145_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgdadam1+001bias_batch128_weightdecay1e-4_condecayall2e-6_inc_singletask.json'
    #   lipstick only
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/19_43_on_September_21/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._120_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_lipstickonly_weightedce.json'
    #   cifar +1.2
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_46_on_September_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_120_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    #   single-head cifar 2
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_10_on_September_25/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    #   single-head noskip cifar, and then 4 more
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_31_on_October_01/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_240_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_10_on_December_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_120_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_15_on_December_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_120_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_11_on_December_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_120_model.pkl'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_15_on_December_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_120_model.pkl'
    param_file = 'params/binmatr2_cifar_sgd1bias_fc_noskip_batch128_weightdecay3e-4_singletask.json'
    #   fine-tuned with fixed connections very sparse cifar [single head] ; AKA cifar2
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_58_on_October_02/optimizer=SGD|batch_size=128|lr=0.001|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd001bias_finetune_batch128_singletask.json'
    #  single-head cifar 3
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_25_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd001bias_finetune_batch128_singletask.json'
    #  single-head cifar 4
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_56_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd001bias_finetune_batch128_singletask.json'
    #  single-head cifar 5
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_34_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd001bias_finetune_batch128_singletask.json'
    # imagenette
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_19_on_October_13/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_imagenette_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    # cifar 6vsAll
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_01_on_October_29/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_6vsAll.json'
    # cifar quarter-width
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_05_on_November_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0.0003|con_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_quarterwidth.json'
    # cifar quarter-width layer4narrow
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_56_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_32|_32|_32|_32]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0.0003|connect_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_quarterwidth_layer4narrow.json'
    # cifar eights-width layer4narrow
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_15_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[8|_8|_8|_16|_16|_16|_16|_32|_32|_32|_32|_16|_16|_16|_16]|architecture=binmatr2_resnet18|width_mul=0.125|weight_decay=0.0003|connectiv_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_eightswidth_layer4narrow.json'
    # cifar eights-width layer4narrowER
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_17_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[8|_8|_8|_16|_16|_16|_16|_32|_32|_32|_32|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=0.125|weight_decay=0.0003|connectivitie_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_eightswidth_layer4narrower.json'
    # cifar eights-width layer4narrowER yet - 2 neurons
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/19_39_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[8|_8|_8|_16|_16|_16|_16|_32|_32|_32|_32|_2|_2|_2|_2]|architecture=binmatr2_resnet18|width_mul=0.125|weight_decay=0.0003|connectivitie_216_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_eightswidth_layer4narrower2.json'
    # cifar full-width, but 1 neuron in layer41
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_22_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_1|_1|_1|_1]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0003|connec_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_layer4narrower1.json'

    model_to_use = 'resnet18'
    if_pretrained_imagenet = model_to_use != 'my'
    gwc = GradWassersteinCorrelator(save_model_path, param_file, model_to_use)
    params, configs = gwc.params, gwc.configs

    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
        with open(param_file) as json_params:
            params = json.load(json_params)
        with open('configs.json') as config_params:
            configs = json.load(config_params)
        # params['dataset'] = 'imagenet_val'
        params['dataset'] = 'imagenet_test'
        params['batch_size'] = 12#256#320
    print(model_name_short)
    plt.rcParams.update({'font.size': 17})
    # params['batch_size'] = 50/2#50/4#1000/4#26707#11246#1251#10001#
    # params['dataset'] = 'broden_val'
    if False:
        _, val_loader, tst_loader = datasets.get_dataset(params, configs)
        loader = tst_loader#val_loader#

    # moniker = 'cifar_noskip_whole5'
    # chunks = [64, 64] + [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    # gwc.calc_gradients_wrt_output_whole_network_all_tasks(loader, f'grads_test_{moniker}.pkl',
    #                                                      if_pretrained_imagenet, all_layers_bn_afterrelu,
    #                                                       chunks)
    # moniker = 'efficientnet_b3'
    # chunks = [40, 40, 40, 24, 24, 24, 24, 144, 144, 32, 576, 576, 96, 576, 576, 96, 576, 576, 136,
    #           1392, 1392, 232, 1392, 1392, 384, 2304, 2304, 384, 1536]
    # ls = efficientnet_layers_mod
    # ch = chunks
    # gwc.calc_gradients_wrt_output_whole_network_all_tasks(loader, f'grads_test_{moniker}.pkl',
    #                                                      if_pretrained_imagenet, ls,
    #                                                       ch, if_rename_layers=False)
    # # # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_pretrained_imagenet_afterrelu_test.pkl', if_pretrained_imagenet)
    # # # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_pretrained_imagenet_afterrelu_test_early.pkl',
    # # #                                                      if_pretrained_imagenet, layers=early_layers_bn_afterrelu)
    # if True:
    #     gwc.correlate_grads_with_wass_dists_per_neuron(f'grads_test_{moniker}.pkl',
    #                                                   f'corr_grads_test_{moniker}.pkl',
    #                                                    f'wasser_dists/wasser_dist_attr_hist_{moniker}',
    #                                                   if_replace_wass_dists_with_noise=False, layers=ls,
    #                                                    neuron_nums=chunks, if_rename_layers=False)
    #     # ac.correlate_grads_with_wass_dists_per_neuron('grads_pretrained_imagenet_afterrelu_test.pkl',
    #     #                                               f'corr_grads_imagenet_test.pkl',
    #     #                                               'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test',
    #     #                                               if_replace_wass_dists_with_noise=False)
    #     # ac.correlate_grads_with_wass_dists_per_neuron('grads_pretrained_imagenet_afterrelu_test_early.pkl',
    #     #                                               f'corr_grads_imagenet_test_early.pkl',
    #     #                       'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test_early_and_last',
    #     #                                       if_replace_wass_dists_with_noise=False, layers=early_layers_bn_afterrelu)
    # else:
    #     ac.correlate_grads_with_wass_dists_per_task(f'grads_{model_name_short}.pkl', f'corr_task_grads_{model_name_short}.pkl',
    #                                                   'wasser_dist_attr_hist_bettercifar10single')
    # gwc.plot_correlation_of_grads_with_wass_dists(f'corr_grads_test_{moniker}.pkl', ls, if_rename_layers=False)

    # gwc.plot_correlation_of_grads_with_wass_dists(f'corr_grads_imagenet_test.pkl', layers_bn_afterrelu, if_rename_layers=True)
    # gwc.correlate_grads_with_wass_dists_per_neuron('grads_pretrained_imagenet_afterrelu_test.pkl',
    #                                           f'corr_grads_imagenet_test_negonly.pkl',
    #                                           'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test',
    #                                           if_replace_wass_dists_with_noise=False, if_only_for_negative_mwds=True)
    # gwc.plot_correlation_of_grads_with_wass_dists(f'corr_grads_imagenet_test.pkl', layers_bn_afterrelu, if_rename_layers=True, if_show=False)
    # gwc.plot_correlation_of_grads_with_wass_dists(f'corr_grads_imagenet_test_negonly.pkl', layers_bn_afterrelu, if_rename_layers=True)

    gwc.plot_grads_and_wass_dists('grads_pretrained_imagenet_afterrelu_test.pkl',
                                                   f'corr_grads_imagenet_test_negonly.pkl',
                                                   'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test',
                                                   if_replace_wass_dists_with_noise=False,
                                                   if_only_for_negative_mwds=False,
                                  layers=[layers_bn_afterrelu[9]], layer_ind_additive=9)
    exit()
    # gwc.plot_correlation_of_grads_with_wass_dists(f'corr_grads_imagenet_test.pkl',
    #                                              layers_bn_afterrelu,
    #                                              'corr_grads_imagenet_test_early.pkl',
    #                                              early_layers_bn_afterrelu)
    # ac.plot_correlation_of_grads_with_wass_dists('corr_grads_imagenet_test_early.pkl', layers=early_layers_bn_afterrelu)
    # exit()
    # ac.plot_correlation_of_grads_with_wass_dists_many_runs([
    #     f'corr_grads_{save_model_path[37:53] + "..." + save_model_path[-12:-10]}.pkl' for save_model_path in [
    #         r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
    #         r'/mnt/raid/data/chebykin/saved_models/12_10_on_September_25/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
    #         r'/mnt/raid/data/chebykin/saved_models/11_25_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
    #         r'/mnt/raid/data/chebykin/saved_models/12_56_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
    #         r'/mnt/raid/data/chebykin/saved_models/14_34_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
    #     ]
    # ])
    model_paths = [
            r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
            r'/mnt/raid/data/chebykin/saved_models/12_10_on_September_25/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
            r'/mnt/raid/data/chebykin/saved_models/11_25_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
            r'/mnt/raid/data/chebykin/saved_models/12_56_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
            r'/mnt/raid/data/chebykin/saved_models/14_34_on_October_08/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl',
        ]
    gwc.plot_correlation_of_grads_with_wass_dists_many_runs(
        [f'corr_grads_test_{save_model_path[37:53] + "..." + save_model_path[-12:-10]}.pkl' for save_model_path in model_paths],
        layers_bn_afterrelu,
        [f'corr_grads_test_early_{save_model_path[37:53] + "..." + save_model_path[-12:-10]}.pkl' for save_model_path in model_paths],
        early_layers_bn_afterrelu
    )
    # gwc.plot_correlation_of_grads_with_wass_dists_many_runs(
    #     [f'corr_grads_test_cifar_noskip_whole{i}.pkl' for i in range(1, 5 + 1)],
    #     all_layers_bn_afterrelu
    # )
    exit()
    #
    # grads_per_neuron = {}
    # layer = 12
    # neurons = [0, 1, 2, 3, 4]
    # wasser_dists = np.load(f'wasser_dist_attr_hist_bettercifar10single_{layer}.npy', allow_pickle=True).item()
    # wasser_dists_np = np.array(pd.DataFrame(wasser_dists))
    # for neuron in neurons:
    #     print(neuron)
    #     w_cur = ac.model['all'].linear.weight.cpu().detach().numpy()[:, neuron]
    #     cur_grads = []
    #     for i in range(10):
    #         grads = ac.calc_gradients_wrt_output(loader, layer, neuron, i)
    #         print(f'{cifar10_dict[i]}:\t{grads:.4f}\t{w_cur[i]:.2f}'.expandtabs(15))
    #         cur_grads.append(grads)
    #     grads_per_neuron[neuron] = np.array(cur_grads)
    # grads_per_neuron_np = np.array(pd.DataFrame(grads_per_neuron))
    # for neuron in neurons:
    #     print(np.corrcoef(wasser_dists_np[:, neuron], grads_per_neuron_np[:, neuron])[0, 1])
    # exit()