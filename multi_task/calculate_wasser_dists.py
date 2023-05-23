from audioop import reverse
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
plt.style.use('seaborn-paper')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wasserstein_divergence(u_values, v_values, u_weights=None, v_weights=None):
        # Adapted from scipy.stats.wasserstein_distance and _cdf_distance
        u_values, u_weights = scipy.stats.stats._validate_distribution(u_values, u_weights)
        v_values, v_weights = scipy.stats.stats._validate_distribution(v_values, v_weights)

        u_sorter = np.argsort(u_values)
        v_sorter = np.argsort(v_values)

        all_values = np.concatenate((u_values, v_values))
        all_values.sort(kind='mergesort')

        # Compute the differences between pairs of successive values of u and v.
        deltas = np.diff(all_values)

        # Get the respective positions of the values of u and v among the values of
        # both distributions.
        u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
        v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

        # Calculate the CDFs of u and v using their weights, if specified.
        if u_weights is None:
            u_cdf = u_cdf_indices / u_values.size
        else:
            u_sorted_cumweights = np.concatenate(([0], np.cumsum(u_weights[u_sorter])))
            u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

        if v_weights is None:
            v_cdf = v_cdf_indices / v_values.size
        else:
            v_sorted_cumweights = np.concatenate(([0],
                                                np.cumsum(v_weights[v_sorter])))
            v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

        # Compute the value of the integral based on the CDFs.
        # Removes the np.abs to compute a divergence.
        return np.sum(np.multiply(u_cdf - v_cdf, deltas))


class WassersteinCalculator(ModelExplorer):
    def __init__(self, save_model_path, param_file, model_to_use='my'):
        super().__init__(save_model_path, param_file, model_to_use)

    def find_highest_activating_images(self, loader, if_average_spatial=True, dataset_type='celeba', target_layer_indices=None,
                                       save_path='img_paths_most_activating_sorted_dict_afterrelu_fcmodel.npy', used_neurons=None,
                                       if_sort_by_path=False, if_dont_save=False, if_left_lim_zero=True, layer_list=layers_bn,
                                       if_save_argmax_labels=False, cond_neurons=None, if_rename_layers=True):
        target_layer_names = np.array([layer.replace('_', '.') if if_rename_layers else layer for layer in layer_list] + ['label'])
        if_find_for_label = False
        if target_layer_indices is None:
            # target_layer_indices = list(range(len(target_layer_names)))
            target_layer_indices = list(used_neurons.keys())
        target_layer_names = target_layer_names[target_layer_indices]
        target_layer_names = list(target_layer_names)
        if 'label' in target_layer_names:
            target_layer_names.remove('label')
            if_find_for_label = True
        if_find_for_label = True

        if used_neurons is None:
            raise NotImplementedError()
            used_neurons_loaded = np.load('actually_good_nodes.npy', allow_pickle=True).item()
            used_neurons = {}
            for layer_idx, neurons in used_neurons_loaded.items():
                used_neurons[layer_idx] = np.array([int(x[x.find('_') + 1:]) for x in neurons])

        if dataset_type in ['celeba', 'broden_val']:
            im_paths_to_labels_dict = None
        else:
            cnt = 0
            im_paths_to_labels_dict = {}

        def save_activation(activations, name, mod, inp, out):
            if name in target_layer_names:
                out = out.detach()
                if ('bn1' in name) and if_left_lim_zero:
                    out = F.relu(out)
                if ('_bn0' in name or '_bn1' in name):
                    out = out * torch.sigmoid(out) # swish
                if ('relu2' in name or 'depthwise_conv' in name or 'project_conv' in name) and not if_left_lim_zero:
                    out = inp[0]
                if if_average_spatial:
                    cur_activations = out.mean(dim=(-1, -2)).cpu().numpy()
                else:
                    cur_activations = out.cpu().numpy()
                if name in activations:
                    cur_activations = np.append(activations[name], cur_activations, axis=0)
                activations[name] = cur_activations

        activations = {}
        hooks = []
        sorted_dict_per_layer_per_neuron = defaultdict(lambda: defaultdict(SortedDict))
        for name, m in self.feature_extractor.named_modules():
            print(name)
            if name in target_layer_names:
                hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                if i % 10 == 0:
                    print(i)

                val_images = batch_val[0].cuda()
                if dataset_type in ['celeba', 'broden_val']:
                    im_paths = batch_val[-1]
                else:
                    root_path = f'/mnt/raid/data/chebykin/{dataset_type}/my_imgs'
                    Path(root_path).mkdir(exist_ok=True)
                    im_paths = []
                    if_saved_images = False

                outs = self.feature_extractor(val_images)
                if dataset_type not in ['celeba', 'broden_val']:
                    recreated_images = recreate_image_nonceleba_batch(val_images, dataset_type=dataset_type)
                for layer_idx, layer in zip(target_layer_indices, target_layer_names):
                    used_neurons_cur = used_neurons[layer_idx]
                    if len(used_neurons_cur) == 0:
                        continue
                    acts = activations[layer]  # shape (batch_size, n_neurons)
                    for j in range(acts.shape[0]):
                        if dataset_type in ['celeba', 'broden_val']:
                            cur_path = im_paths[j]
                        else:
                            if not if_saved_images:
                                cur_path = root_path + f'/{cnt}.jpg'
                                cnt += 1
                                cur_img = recreated_images[j]
                                # save_image(cur_img, cur_path)
                                im_paths.append(cur_path)
                                im_paths_to_labels_dict[cur_path] = batch_val[1][j].item()
                            else:
                                cur_path = im_paths[j]
                        if used_neurons_cur == 'all':
                            used_neurons_cur_ = range(acts.shape[1])
                        else:
                            used_neurons_cur_ = used_neurons_cur
                        for k in used_neurons_cur_:
                            if not if_sort_by_path:
                                sorted_dict_per_layer_per_neuron[layer][k][acts[j][k]] = cur_path
                            else:
                                sorted_dict_per_layer_per_neuron[layer][k][cur_path] = acts[j][k]
                    if_saved_images = True # relevant only for the first iteration: False->True, afterwards always True

                activations.clear()

                if if_find_for_label:
                    if self.use_my_model:
                        print('cond_neurons are ignored!')
                        task_cnt = 0
                        for key in self.model.keys():
                            if key == 'rep': #only interested in tasks
                                continue
                            out_t, _ = self.model[key](outs[task_cnt], None)
                            n_classes = out_t.shape[1]
                            task_cnt += 1

                            out_t = torch.exp(out_t)
                            if if_save_argmax_labels:
                                out_t = torch.argmax(out_t, dim=1)
                            else:
                                diff = out_t[:, 1] - out_t[:, 0]
                            for j in range(out_t.shape[0]):
                                if key != 'all': # multi-task network
                                    if not if_save_argmax_labels:
                                        cur_value = diff[j].item()
                                    else:
                                        cur_value = int(out_t[j] == 1)
                                    if not if_sort_by_path:
                                        sorted_dict_per_layer_per_neuron['label'][int(key)][cur_value] = im_paths[j]
                                    else:
                                        sorted_dict_per_layer_per_neuron['label'][int(key)][im_paths[j]] = cur_value

                                else: # single task network, e.g. a 10-class softmax
                                    for k in range(n_classes):
                                        if not if_save_argmax_labels:
                                            cur_value = out_t[j, k].item()
                                        else:
                                            cur_value = int(out_t[j] == k)
                                        if not if_sort_by_path:
                                            sorted_dict_per_layer_per_neuron['label'][k][cur_value] = im_paths[j]
                                        else:
                                            sorted_dict_per_layer_per_neuron['label'][k][im_paths[j]] = cur_value
                    else:
                        outs_softmaxed = torch.softmax(outs, dim=1)
                        if if_save_argmax_labels:
                            outs_softmaxed_argmaxed = torch.argmax(outs_softmaxed, dim=1)
                        class_indices = cond_neurons#[0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
                        for j in range(outs.shape[0]):
                            cur_path = im_paths[j]
                            for k in class_indices:
                                if not if_save_argmax_labels:
                                    cur_value = outs_softmaxed[j, k].item()
                                else:
                                    cur_value = int(outs_softmaxed_argmaxed[j] == k)
                                if not if_sort_by_path:
                                    sorted_dict_per_layer_per_neuron['label'][k][cur_value] = cur_path
                                else:
                                    sorted_dict_per_layer_per_neuron['label'][k][cur_path] = cur_value


                print(len(im_paths))

        for hook in hooks:
            hook.remove()

        if not if_dont_save:
            if im_paths_to_labels_dict is not None:
                print(len(im_paths_to_labels_dict.items()))
                np.save(f'path_to_label_dict_{dataset_type}_val.npy', im_paths_to_labels_dict)
            if save_path is not None:
                np.save(save_path, dict(sorted_dict_per_layer_per_neuron))

        return sorted_dict_per_layer_per_neuron


    def compute_attr_hist_for_neuron_pandas(self, target_layer_idx, target_neurons,
                                            cond_layer_idx, cond_neurons,
                                            if_cond_labels=False, df=None,
                                            out_dir='attr_hist_for_neuron_fc_all', used_neurons=None, dataset_type='celeba',
                                            if_calc_wasserstein=False, offset=(0, 0), if_show=True, if_left_lim_zero=True,
                                            layer_list=layers_bn, if_plot=True, cond_neuron_lambda=lambda x: [x], if_rename_layers=True,
                                            use_wasserstein_divergence=False):
        # when if_show==False, hists are plotted and saved, but not shown
        # when if_plot==False, nothing is plotted, only wasserstein distances are calculated
        # target_layer_names = [layer.replace('_', '.') for layer in layer_list] + ['label']
        target_layer_names = np.array([layer.replace('_', '.') if if_rename_layers else layer for layer in layer_list] + ['label'])
        if df is None:
            df = pd.read_pickle('sparse_afterrelu.pkl')
        Path(out_dir).mkdir(exist_ok=True)

        n_cond_indices = len(cond_neurons)
        rows_num = math.floor(math.sqrt(n_cond_indices))
        while n_cond_indices % rows_num != 0:
            rows_num += 1
        cols_num = int(math.ceil(n_cond_indices / rows_num))
        if target_neurons == 'all':
            if used_neurons is None:
                used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()
                target_neurons = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[target_layer_idx]])
            else:
                target_neurons = used_neurons[target_layer_idx]
            print(target_neurons)

        if if_calc_wasserstein:
            wasserstein_dists_dict = defaultdict(dict)

        if offset == 'argmax':
            offset = (1e-5, 1 - 1e-5)

        df_cond = df
        if if_cond_labels:
            if dataset_type == 'celeba':
                path_to_label_dict = np.load('path_to_label_dict_celeba_val.npy', allow_pickle=True).item()
                df_cond = df.copy()
                df_cond = df_cond.drop(df_cond.index[df_cond.layer_name == 'label']) #remove label predictions, replace with true labels
                path_label_items = list(path_to_label_dict.items())
                path_label_items.sort(key=operator.itemgetter(0))
                per_label_dict = defaultdict(list)
                for path, labels in path_label_items:
                    for i in range(40):
                        per_label_dict[i].append(labels[i])
                data = []
                for i in range(40):
                    data.append(['label', i, *per_label_dict[i]])
                    # df_cond.loc['label', i
                df_cond_addendum = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] + list(list(zip(*path_label_items))[0]))
                df_cond = df_cond.append(df_cond_addendum)
            elif dataset_type == 'broden_val': #there's a problem with names, but I think both broden and broden_val are actuall broden_train
                # df_cond = pd.read_pickle('path_to_label_df_broden_train.pkl')
                df_cond = pd.read_pickle('path_to_label_df_broden_train_all.pkl') # includes colour and material
            # elif dataset_type == 'broden_val':
            #     df_cond = pd.read_pickle('path_to_label_df_broden_val.pkl')
            elif dataset_type in ['cifar_6vsAll', 'cifar']:
                # need normal cifar dict because need all 10 labels
                path_to_label_dict = np.load('path_to_label_dict_cifar_val.npy', allow_pickle=True).item()
                df_cond = df.copy()
                df_cond = df_cond.drop(df_cond.index[df_cond.layer_name == 'label'])  # remove label predictions, replace with true labels
                path_label_items = list(path_to_label_dict.items())
                path_label_items.sort(key=operator.itemgetter(0))
                per_label_dict = defaultdict(list)
                for path, label in path_label_items:
                    for i in range(10):
                        per_label_dict[i].append(1 if label == i else 0)
                data = []
                for i in range(10):
                    data.append(['label', i, *per_label_dict[i]])
                    # df_cond.loc['label', i
                modify_path_lambda = lambda path:path
                if dataset_type == 'cifar_6vsAll':
                    modify_path_lambda = lambda path:path.replace('cifar', 'cifar_6vsAll')
                df_cond_addendum = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] +
                                  [modify_path_lambda(path) for path in list(zip(*path_label_items))[0]])
                df_cond = df_cond.append(df_cond_addendum)

            else:
                raise NotImplementedError()
        df_cond = df_cond.drop(df_cond[df_cond['layer_name'] != target_layer_names[cond_layer_idx]].index)
        df_cond_np = np.array(df_cond)[:, 2:].astype('int')

        # if dataset_type == 'broden_val':
        #     print(df.columns)
        #     clmns = [i.item() for i in df.columns[2:]]
        #     df.columns = clmns

        selected_paths1_cache = {}
        selected_paths2_cache = {}

        if isinstance(target_neurons, str) and target_neurons == 'all':
            target_neurons_ = np.unique(df['neuron_idx'])
        else:
            target_neurons_ = target_neurons

        for target_neuron in target_neurons_:
            print(target_neuron)
            if if_plot:
                _, axs = plt.subplots(rows_num, cols_num, squeeze=False, figsize=(10 * 1.5, 14 * 1.5))
                axs = axs.flatten()

            selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                                   (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'], axis=1))

            if len(selected_values_list) == 0:
                continue

            bin_size = 0.02
            selected_values_max = selected_values_list.max()
            selected_values_min = selected_values_list.min()
            diff = selected_values_max - selected_values_min
            if diff == 0.0:
                print(f'Useless neuron {target_layer_idx}_{target_neuron}')
                continue
            while diff / bin_size < 30:
                bin_size /= 2
                print(f'New bin size: {bin_size}')
            while diff / bin_size > 80:
                bin_size *= 2
                print(f'New bin size: {bin_size}')
            xlim_right = selected_values_max + 0.01
            if if_left_lim_zero:
                xlim_left = 0.0
            else:
                xlim_left = selected_values_min - 0.01

            for cond_i, idx in enumerate(cond_neurons):
                if if_plot:
                    title = str(idx)
                    if cond_layer_idx == -1:
                        if dataset_type=='celeba':
                            title = celeba_dict[idx]
                        elif dataset_type=='cifar':
                            # title = cifarfashion_dict[idx]
                            title = cifar10_dict[idx]
                        elif dataset_type=='cifar_6vsAll':
                            # title = cifarfashion_dict[idx]
                            title = cifar10_dict[idx]
                        elif dataset_type=='imagenette':
                            if idx > 9:
                                dict_from_full_imagenet_idx = dict(zip([0, 217, 482, 491, 497, 566, 569, 571, 574, 701],range(10)))
                                title = imagenette_dict[dict_from_full_imagenet_idx[idx]]
                            else:
                                title = imagenette_dict[idx]
                        elif dataset_type == 'imagenet_val':
                            title = imagenet_dict[idx]
                        else:
                            raise NotImplementedError()

                        title = 'Class: ' + title.capitalize()

                if idx not in selected_paths1_cache:
                    corresponding_indices = cond_neuron_lambda(idx)
                    final_mask = np.array([False]*selected_values_list.shape[1])[None, :] #shape is number of images
                    for ind in corresponding_indices:
                        # cond_mask1 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                        #                 (df_cond['neuron_idx'] == ind)].drop(['layer_name', 'neuron_idx'], axis=1) <= offset[0]
                        cond_mask1 = df_cond_np[ind] <= offset[0]
                        final_mask += np.array(cond_mask1)
                    selected_paths1_cache[idx] = np.array(final_mask, dtype=bool)
                cond_mask1 = selected_paths1_cache[idx]
                selected_values_list1 = selected_values_list[cond_mask1] # why is this the not predicted class? smaller than small offset selects those samples which don't have this class (value 0 in one-hot)
                if len(selected_values_list1) == 0:
                    print('continue1', idx)
                    continue

                # xlim_right = 12
                # bin_size = 0.08
                if if_plot:
                    proper_hist(selected_values_list1, bin_size=bin_size, ax=axs[cond_i], xlim_left=xlim_left,
                            xlim_right=xlim_right,
                            density=True, label='Out-of-class')

                if idx not in selected_paths2_cache:
                    corresponding_indices = cond_neuron_lambda(idx)
                    final_mask = np.array([False] * selected_values_list.shape[1])[None, :] # shape is number of images
                    for ind in corresponding_indices:
                        # cond_mask2 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                        #                          (df_cond['neuron_idx'] == ind)].drop(['layer_name', 'neuron_idx'],
                        #                                                               axis=1) > offset[1]
                        cond_mask2 = df_cond_np[ind] > offset[1]
                        final_mask += np.array(cond_mask2)
                    selected_paths2_cache[idx] = np.array(final_mask, dtype=bool)
                cond_mask2 = selected_paths2_cache[idx]
                selected_values_list2 = selected_values_list[cond_mask2] # why is this the not predicted class? larger than close to 1 offset selects those samples which have this class (value 1 in one-hot)
                if len(selected_values_list2) == 0:
                    print('continue2', idx)
                    continue

                if if_calc_wasserstein:
                    # wd = scipy.stats.wasserstein_distance(selected_values_list1, selected_values_list2)
                    # title += '\n'
                    # title += f'{wd:.2f}'
                    if not use_wasserstein_divergence:
                        wd_normed = scipy.stats.wasserstein_distance(
                            (selected_values_list1 - selected_values_min) / (selected_values_max - selected_values_min),
                            (selected_values_list2 - selected_values_min) / (selected_values_max - selected_values_min))

                        # this means: 2 to the right of 1 -> positive, 2 to the left of 1 -> negative
                        wd_normed *= np.sign(selected_values_list2.mean() - selected_values_list1.mean())
                        if if_plot:
                            title += f'\nMWD: {wd_normed:.2f}'
                        wasserstein_dists_dict[target_neuron][idx] = wd_normed
                    else:
                        # wd_normed = wasserstein_divergence(
                            # (selected_values_list1 - selected_values_min) / (selected_values_max - selected_values_min),
                            # (selected_values_list2 - selected_values_min) / (selected_values_max - selected_values_min))
                        
                        wd = wasserstein_divergence(
                            selected_values_list1,
                            selected_values_list2)

                        # the sign should match automatically
                        if if_plot:
                            title += f'\nShift: {wd:.2f}'
                        wasserstein_dists_dict[target_neuron][idx] = wd


                if if_plot:
                    proper_hist(selected_values_list2, bin_size=bin_size, ax=axs[cond_i], xlim_left=xlim_left,
                            xlim_right=xlim_right,
                            alpha=0.75, title=title, density=True, label='In-class')
                    
                    fontsize=16
                    axs[cond_i].set_xlabel('Mean activation', fontsize=fontsize)
                    axs[cond_i].set_ylabel('Density', fontsize=fontsize)
                    # axs[cond_i].set_frame_on(False)
                    axs[cond_i].get_yaxis().set_ticks([])
                    # axs[cond_i].get_yaxis().set_visible(False)
                    axs[cond_i].tick_params(labelsize=14)
                    handles, labels = axs[cond_i].get_legend_handles_labels()
                    axs[cond_i].legend(handles[::-1], labels[::-1], fontsize=fontsize-2)



            if if_plot:
                plot_name = f'{target_layer_idx}_{target_neuron}'
                title_name = f'{target_layer_idx}/{target_layer_names[target_layer_idx]}_{target_neuron}'
                plt.suptitle(title_name)
                plt.tight_layout()
                wdiv_str = "_shift" if use_wasserstein_divergence else ""
                plt.savefig(f'{out_dir}/hist_{plot_name}{wdiv_str}.png', format='png')#, bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{out_dir}/hist_{plot_name}{wdiv_str}.svg', format='svg')#, bbox_inches='tight', pad_inches=0)
                if if_show:
                    plt.show()
                plt.close()

        if if_calc_wasserstein:
            wname = "shift" if use_wasserstein_divergence else "dist"
            np.save(f'wasser_dists/wasser_{wname}_' + out_dir + '_' + str(target_layer_idx) + '.npy', wasserstein_dists_dict, allow_pickle=True)

    def wasserstein_barplot(self, suffix, target_neurons_dict, classes_dict, n_show=10, if_show=False, out_path_suffix=''):
        n_best = n_show // 2
        out_path = 'wd_barplot_' + suffix + out_path_suffix
        Path(out_path).mkdir(exist_ok=True)
        for layer_idx, neurons in target_neurons_dict.items():
            wasser_dists_cur = np.load(f'wasser_dists/wasser_dist_{suffix}_{layer_idx}.npy', allow_pickle=True).item()
            for neuron in neurons:
                print(neuron)
                class_wd_pairs = [(classes_dict[idx], wd) for (idx, wd) in wasser_dists_cur[neuron].items()]
                class_wd_pairs.sort(key=operator.itemgetter(1))
                class_wd_pairs_best = class_wd_pairs[:n_best] + class_wd_pairs[-n_best:]
                classes, wds = zip(*class_wd_pairs_best)

                # width = 1 / n_show - 0.05
                colors = ['r'] * n_best + ['g'] * n_best
                fig, ax = plt.subplots(figsize=(5 * 1.5, 7 * 1.5), nrows=2, gridspec_kw={'height_ratios': [5, 3]})
                title = f'{layer_idx}_{neuron}'
                plt.suptitle(title)
                labels = classes
                x = np.arange(n_show)
                ax[0].barh(x, wds, color=colors)#, width)
                ax[0].set_yticks(x)
                ax[0].set_yticklabels(labels)
                ax[0].set_xlim(-0.5, 0.5)

                proper_hist(np.array(list(zip(*class_wd_pairs))[1]),ax=ax[1], xlim_left=-0.5, xlim_right=0.5, bin_size=0.04)
                ax[1].set_yscale('log')
                ax[1].set_ylim(top=400)
                # ax[1].hist(list(zip(*class_wd_pairs))[1])
                # ax[1].set_xlim(-0.5, 0.5)
                plt.savefig(f'{out_path}/{title}.png', format='png', bbox_inches='tight', pad_inches=0)
                if if_show:
                    plt.show()
                plt.close()

    def compute_attr_hist_for_neuron_pandas_wrapper(self, loader, df_val_images_activations_path, target_dict,
                                                    out_dir_path, if_cond_label=False,
                                                    sorted_dict_path=None,
                                                    used_neurons=None, dataset_type='celeba', if_calc_wasserstein=False,
                                                    offset=(0, 0), if_show=True, if_force_recalculate=False, if_dont_save=False,
                                                    if_left_lim_zero=True, layer_list=layers_bn, if_plot=True, if_rename_layers=True, use_wasserstein_divergence=False):
        '''

        :param target_dict: 'all_network' OR {target_layer_idx -> [neuron_indices] OR 'all'}
        :param if_cond_label: whether separate into 2 histograms by true labels (in contrast to model predictions)
        '''
        if target_dict == 'all_network':
            target_dict = {i: 'all' for i in reversed(range(15))}
        elif target_dict == 'all_network_with_early':
            target_dict = {i: 'all' for i in reversed(range(17))}
        elif target_dict == 'all_network_efficientnet':
            target_dict = {i: 'all' for i in reversed(range(29))}
        if used_neurons is not None:
            if used_neurons != 'use_target':
                if not os.path.exists(used_neurons):
                    if used_neurons == 'resnet_full':
                        chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
                    elif used_neurons == 'resnet_quarter':
                        chunks = [16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]
                        # chunks = [16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32]
                        # chunks = [8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 16, 16, 16, 16]
                        # chunks = [8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 8, 8, 8, 8]
                        # chunks = [8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 2, 2, 2, 2]
                    elif used_neurons == 'resnet_early':
                        chunks = [64, 64]
                    elif used_neurons == 'resnet_full_with_early':
                        chunks = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
                    elif used_neurons == 'vgg_few':
                        chunks = [512, 512]
                    elif used_neurons == 'mobilenet_few':
                        chunks = [160, 160, 320]
                    elif used_neurons == 'efficientnet':
                        # chunks = [40, 40, 40, 24, 24, 24, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 48,
                        #           288, 288, 48, 288, 288, 48, 288, 288, 96, 576, 576, 96, 576, 576, 96, 576, 576, 96,
                        #           576, 576, 96, 576, 576, 136, 816, 816, 136, 816, 816, 136, 816, 816, 136, 816, 816,
                        #           136, 816, 816, 232, 1392, 1392, 232, 1392, 1392, 232, 1392, 1392, 232, 1392, 1392,
                        #           232, 1392, 1392, 232, 1392, 1392, 384, 2304, 2304, 384, 1536]
                        chunks = [40, 40, 40, 24, 24, 24, 24, 144, 144, 32, 576, 576, 96, 576, 576, 96, 576, 576, 136,
                                  1392, 1392, 232, 1392, 1392, 384, 2304, 2304, 384, 1536]

                    used_neurons = {}
                    for idx, ch in enumerate(chunks):
                        used_neurons[idx] = np.array(range(chunks[idx]))
                else:
                    used_neurons_loaded = np.load(used_neurons, allow_pickle=True).item()
                    used_neurons = {}
                    for layer_idx, neurons in used_neurons_loaded.items():
                        used_neurons[layer_idx] = np.array([int(x[x.find('_') + 1:]) for x in neurons])
            else:
                used_neurons = defaultdict(list)
                for k, v in target_dict.items():
                    used_neurons[k] = v

        if dataset_type == 'celeba':
            cond_neurons = list(range(40))
        elif 'imagenet' in dataset_type:
            cond_neurons = list(range(1000))
        elif 'cifar' in dataset_type:
            cond_neurons = list(range(10))
        else:
            cond_neurons = None
            for l in reversed(list(self.model.named_modules())):
                if l[0] == 'fc':
                    cond_neurons = l[1].out_features
                    break
            assert cond_neurons != None, "Can't infer number of output neurons = classes."
            # cond_neurons = list(range(20))
            # cond_neurons = list(range(10))
            # cond_neurons = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
            # cond_neurons = list(range(1000))#[0, 217, 482, 491, 497, 566, 569, 571, 574, 701] + list(range(1, 10)) + list(range(901, 911))
            # cond_neurons = list(range(301))
            # cond_neurons = list(range(1197))
            # cond_neurons = list(range(10))

        if not os.path.exists(df_val_images_activations_path) or if_force_recalculate:
            if not os.path.exists(sorted_dict_path) or if_force_recalculate:
                sorted_dict_per_layer_per_neuron = self.find_highest_activating_images(loader, dataset_type=dataset_type,
                                                                                     save_path=None, #note that I decided not to save the dict. df should be enough
                                                                                     used_neurons=used_neurons,
                                                                                     if_sort_by_path=True,
                                                                                     if_dont_save=if_dont_save,
                                                                                     if_left_lim_zero=if_left_lim_zero,
                                                                                     layer_list=layer_list,
                                                                                     if_save_argmax_labels=offset == 'argmax',
                                                                                     cond_neurons=cond_neurons,
                                                                                     if_rename_layers=if_rename_layers)
                print('Sorted dict calculated')
            else:
                sorted_dict_per_layer_per_neuron = np.load(sorted_dict_path, allow_pickle=True).item()
            df = self.convert_sorted_dict_per_layer_per_neuron_to_dataframe(sorted_dict_per_layer_per_neuron,
                                                                          out_path=df_val_images_activations_path)
            print('Converted to df')
        else:
            df = pd.read_pickle(os.path.abspath(df_val_images_activations_path))


        for layer_idx, targets in target_dict.items():
            self.compute_attr_hist_for_neuron_pandas(layer_idx, targets, -1, cond_neurons, if_cond_label, df,
                                                   out_dir_path, used_neurons, dataset_type, if_calc_wasserstein,
                                                   offset, if_show, if_left_lim_zero, layer_list, if_plot,
                                                   lambda x: [x],
                                                   #cond_neuron_lambda=lambda x: hypernym_idx_to_imagenet_idx[x],
                                                   if_rename_layers,
                                                   use_wasserstein_divergence
                                                  )

    def convert_sorted_dict_per_layer_per_neuron_to_dataframe(self, sorted_dict_per_layer_per_neuron, out_path=None):
        data = []
        for layer_name, layer_dict in sorted_dict_per_layer_per_neuron.items():
            for neuron_idx, neuron_dict in layer_dict.items():
                nditems = list(neuron_dict.items())
                nditems.sort(key=operator.itemgetter(0))
                data.append([layer_name, neuron_idx] + [value for (path, value) in nditems])#list(list(zip(*nditems))[1]))
        df = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] + list(list(zip(*nditems))[0]))
        if out_path is not None:
            df.to_pickle(out_path)
        return df

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
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
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
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_15_on_December_22/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_120_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_noskip_batch128_weightdecay3e-4_singletask.json'
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


    model_to_use = 'my'
    if_pretrained_imagenet = model_to_use != 'my'
    wc = WassersteinCalculator(save_model_path, param_file, model_to_use)
    params, configs = wc.params, wc.configs

    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
        with open(param_file) as json_params:
            params = json.load(json_params)
        with open('configs.json') as config_params:
            configs = json.load(config_params)
        params['dataset'] = 'imagenet_val'
        # params['dataset'] = 'imagenet_test'
        params['batch_size'] = 64#128#256#320
    print(model_name_short)

    plt.rcParams.update({'font.size': 18})
    # params['batch_size'] = 50/2#50/4#1000/4#26707#11246#1251#10001#
    # params['dataset'] = 'broden_val'

    params['dataset'] = 'cifar10'

    _, val_loader, tst_loader = datasets.get_dataset(params, configs)
    loader = val_loader#tst_loader#

    # store_path_and_label_df_broden(loader, 'path_to_label_df_broden_train.pkl')
    # store_path_and_label_df_broden(loader, 'path_to_label_df_broden_train_all.pkl') #includes color and material
    # exit()
    # conf_matr1 = ac.eval_with_neuron_replaced(params, loader, 8, 0, 0)
    # prec1, rec1 = ac.calc_precision_recall_from_confusion_matr(conf_matr1)
    # conf_matr2 = ac.eval_with_neuron_replaced(params, loader, 8, 74, 255)
    # prec2, rec2 = ac.calc_precision_recall_from_confusion_matr(conf_matr2)
    # print(conf_matr1)
    # print(conf_matr2)
    # [print(f'{i}: {p:.6f}, {r:.6f}') for i, (p, r) in enumerate(zip(list(prec2 - prec1), list(rec2 - rec1)))]
    # exit()
    # ac.store_highest_activating_images('highest_activating_ims_bn_nonsparse',
    #                                    df_path='nonsparse_afterrelu.pkl')
    # ac.store_highest_activating_images('highest_activating_ims_bn_cifar10single_sparse2',
    #                                    df_path='sparse2_afterrelu_cifar.pkl')
    # ac.store_highest_activating_images('highest_activating_ims_bn_cifar10single_rgb',
    #                                    df_path='bettercifar10single_nonsparse_afterrelu.pkl', n_save=20)
    # ac.store_highest_activating_images('highest_activating_ims_imagenet_afterrelu_many',
    #                                    df_path='pretrained_imagenet_afterrelu.pkl', n_save=64, layer_list=layers_bn_afterrelu)
    # ac.store_highest_activating_images('highest_activating_ims_imagenet_afterrelu_20_again',
    #                                    df_path='pretrained_imagenet_afterrelu.pkl', n_save=20,
    #                                    layer_list=layers_bn)#layers_bn_afterrelu)
    # ac.store_highest_activating_images('highest_activating_ims_cifar_layer4narrower1_afterrelu',
    #                                    df_path='cifar_layer4narrower1_lbl_afterrelu.pkl', n_save='all', layer_list=layers_bn_afterrelu)
    # ac.assess_binary_separability_1vsAll_whole_network('bettercifar10single_nonsparse_afterrelu.pkl', 'df_label_cifar.pkl',
    #                                                    layers_bn, 10, postfix='_cifar')
    # convert_imagenet_path_to_label_dict_to_df()
    # ac.assess_binary_separability_1vsAll_whole_network('pretrained_imagenet_afterrelu.pkl', 'df_label_imagenet_val.pkl',
    #                                                    layers_bn_afterrelu, 1000, postfix='_imagenet_val')
    # ac.plot_overall_hists('pretrained_imagenet_afterrelu.pkl', 'overall_hists_imagenet_afterrelu', layers_bn)
    # exit()
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single_noskip_nonsparse_afterrelu.pkl',
    #                                                {10:'all', 9:'all', 8:'all', 7:'all', 6:'all', 5:'all', 4:'all', 3:'all', 2:'all', 1:'all', 0:'all'},
    #                                                'attr_hist_bettercifar10single_noskip', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single_noskip.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_afterrelu_cifar.pkl', 'all_network',
    #                                                'attr_hist_sparsecifar', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_sparsecifar.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse2_afterrelu_cifar.pkl', 'all_network',
    #                                                'attr_hist_sparse2cifar', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_sparse2cifar.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse2_afterrelu_cifar_TEMP2.pkl', {12: [125]},
    #                                                'TEMP2', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                dataset_type='cifar',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_sparse2cifar_TEMP2.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=True,
    #                                                if_force_recalculate=True, if_dont_save=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'prerelu_imagenette_TEMP.pkl', {14:[112]},
    #                                                'TEMP', if_cond_label=False,
    #                                                used_neurons='use_target',
    #                                                dataset_type='imagenette',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_prerelu_imagenette_TEMP.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False,
    #                                                if_force_recalculate=True, if_dont_save=True, if_left_lim_zero=False,
    #                                                layer_list=layers_bn_prerelu)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'pretrained_imagenet_broden3_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_pretrained_imagenet_broden3_afterrelu', if_cond_label=True,
    #                                                used_neurons='resnet_full',
    #                                                dataset_type='broden_val',
    #                                    sorted_dict_path='img_paths_most_activating_sorted_dict_paths_pretrained_imagenet_broden3_afterrelu.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=layers_bn_afterrelu, if_plot=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'pretrained_imagenet_afterrelu_l8.pkl', {8:list(range(256))},
    #                                                'attr_hist_pretrained_imagenet_afterrelu_l8', if_cond_label=False,
    #                                                used_neurons='use_target',
    #                                                dataset_type='imagenet_val',
    #                                    sorted_dict_path='img_paths_most_activating_sorted_dict_paths_pretrained_imagenet_afterrelu_l8.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=layers_bn, if_plot=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'pretrained_imagenet_afterrelu_test_early_and_last.pkl',
    #                                                {0:'all', 1:'all'},
    #                                                'attr_hist_pretrained_imagenet_afterrelu_test_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='imagenet_test',
    #             sorted_dict_path='img_paths_most_activating_sorted_dict_paths_pretrained_imagenet_afterrelu_test_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=early_layers_and_last_bn_afterrelu, if_plot=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'cifar10_6vsAll_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_cifar10_6vsAll_afterrelu', if_cond_label=True,
    #                                                used_neurons='resnet_full',
    #                                                dataset_type='cifar_6vsAll',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_cifar10_6vsAll.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False,
    #                                                if_force_recalculate=False, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=layers_bn_afterrelu, if_plot=True)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'cifar_eightswidth_layer4narrower2_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_cifar_eightswidth_layer4narrower2_afterrelu', if_cond_label=False,
    #                                                used_neurons='resnet_quarter',
    #                                                dataset_type='cifar',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_cifar_eightswidth_layer4narrower2.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=layers_bn_afterrelu, if_plot=True)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'cifar_layer4narrower1_lbl_afterrelu.pkl',
    #                                                {14:[0], 13:[0], 12:[0], 11:[0]},
    #                                                'attr_hist_cifar_layer4narrower1_lbl_afterrelu', if_cond_label=True,
    #                                                used_neurons='use_target',
    #                                                dataset_type='cifar',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_cifar_layer4narrower1_lbl.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_dont_save=False, if_left_lim_zero=True,
    #                                                layer_list=layers_bn_afterrelu, if_plot=True)
    # moniker = 'cifar_noskip_whole5'
    # wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, f'{moniker}.pkl', 'all_network_with_early',
    #                                                f'attr_hist_{moniker}', if_cond_label=False,
    #                                                used_neurons='resnet_full_with_early',
    #                                                dataset_type='cifar',
    #                                                sorted_dict_path=f'sorted_dict_{moniker}.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_left_lim_zero=True,
    #                                                layer_list=all_layers_bn_afterrelu, if_plot=False)
    # moniker = 'vgg_few'
    # wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, f'{moniker}.pkl', {0:'all', 1:'all'},
    #                                                f'attr_hist_{moniker}', if_cond_label=False,
    #                                                used_neurons='vgg_few',
    #                                                dataset_type='imagenet_test',
    #                                                sorted_dict_path=f'sorted_dict_{moniker}.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_left_lim_zero=True,
    #                                                layer_list=vgg_layers_few, if_plot=False)
    # moniker = 'mobilenet_few'
    # wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, f'{moniker}.pkl', {0:'all', 1:'all', 2:'all'},
    #                                                f'attr_hist_{moniker}', if_cond_label=False,
    #                                                used_neurons='mobilenet_few',
    #                                                dataset_type='imagenet_test',
    #                                                sorted_dict_path=f'sorted_dict_{moniker}.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_left_lim_zero=True,
    #                                                layer_list=mobilenet_layers_few, if_plot=False)
    # moniker = 'efficientnet_b3'
    # wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, f'{moniker}.pkl', 'all_network_efficientnet',
    #                                                f'attr_hist_{moniker}', if_cond_label=False,
    #                                                used_neurons='efficientnet',
    #                                                dataset_type='imagenet_test',
    #                                                sorted_dict_path=f'sorted_dict_{moniker}.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=False, if_left_lim_zero=False,
    #                                                layer_list=efficientnet_layers, if_plot=False, if_rename_layers=False)
    moniker = 'cifar_for_paper'
    wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, f'{moniker}.pkl', {5:[103], 14:[393]},
                                                   f'attr_hist_{moniker}', if_cond_label=False,
                                                   used_neurons='use_target',
                                                   dataset_type='cifar',
                                                   sorted_dict_path=f'sorted_dict_{moniker}.npy',
                                                   if_calc_wasserstein=True, offset='argmax', if_show=True,
                                                   if_force_recalculate=False, if_left_lim_zero=True,
                                                   layer_list=layers_bn_afterrelu, if_plot=True, if_rename_layers=True,
                                                   use_wasserstein_divergence=True)

    # chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    # ac.wasserstein_barplot('attr_hist_pretrained_imagenet_prerelu', dict(zip(range(15), [range(c) for c in chunks])),
    #                        imagenet_dict, n_show=20, out_path_suffix='')
    # ac.wasserstein_barplot('attr_hist_pretrained_imagenet_broden3_afterrelu', dict(zip(range(15), [range(c) for c in chunks])),
    #                        broden_categories_list, n_show=20, out_path_suffix='')
    # wc.wasserstein_barplot(f'attr_hist_{moniker}', {0:range(160), 1:range(160), 2:range(320)},
    #                        imagenet_dict, n_show=20, out_path_suffix='')
    # wc.wasserstein_barplot(f'attr_hist_{moniker}', {26:range(2304)},
    #                        imagenet_dict, n_show=20, out_path_suffix='')

    # ac.store_layer_activations_many(loader, [14], if_average_spatial=True, if_store_labels=True, out_path_postfix='_pretrained_imagenet')
    # ac.store_layer_activations_many(loader, list(range(15)), out_path_postfix='_bettercifar10single',
    #                                 layer_list=layers_bn_afterrelu, if_store_labels=False,
    #                                 if_store_layer_names=True, if_average_spatial=False, if_save_separately=False)
    # chunks = [16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32]
    # chunks = [8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 16, 16, 16, 16]
    # ac.store_layer_activations_many(loader, list(range(15)), out_path_postfix='cifar_eightswidth_layer4narrow',
    #                                 layer_list=layers_bn_afterrelu, if_store_labels=False,
    #                                 if_store_layer_names=True, if_average_spatial=False, if_save_separately=False)
    # ac.plot_hists_of_spatial_activations(path='activations_on_validation_preserved_spatial_cifar_eightswidth_layer4narrow.npy',
    #                                      out_path='hist_spatial_cifar_eightswidth_layer4narrow', layer_list=layers_bn_afterrelu,
    #                                      chunks=chunks)
    # ac.plot_hists_of_spatial_activations_no_save(loader, list(range(15)),
    #                                      out_path='hist_spatial_cifar_bettercifar10single_zeros', layer_list=layers_bn_afterrelu)
    # ac.plot_hists_of_spatial_activations_no_save(loader, list(range(15)),
    #                                              out_path='hist_spatial_imagenet_r34_afterrelu',
    #                                              layer_list=layers_bn_afterrelu, chunks=chunks)
    # ac.store_highest_activating_patches(loader, [0], out_path='patches_bettercifar10single',
    #                                     base_path='/mnt/raid/data/chebykin/cifar/my_imgs',
    #                                     image_size=32)
    # ac.store_highest_activating_patches(loader, [10], out_path='patches_imagenet',
    #                                     base_path='/mnt/raid/data/chebykin/imagenet_val/my_imgs',
    #                                     image_size=224)
    # exit()

    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'nonsparse_afterrelu_nobiascifar.pkl', {14:'all'},
    #                                                'attr_hist_nobiascifar', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_nobiascifar.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5))
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_afterrelu_cifar.pkl', {14:'all'},
    #                                                'attr_hist_sparsecifar', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_sparsecifar.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'lipstickonly_nonsparse_afterrelu_v2.pkl', {14:'all'},
    #                                                'attr_hist_lipstickonly_v2', if_cond_label=True,
    #                                                used_neurons='resnet_quarter',
    #                                                dataset_type='celeba',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_lipstickonly_v2.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False, if_force_recalculate=True,
    #                                                if_left_lim_zero=True, layer_list=layers_bn_afterrelu
    #                                                )
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'hatonly_nonsparse_afterrelu.pkl', {14:'all'},
    #                                                'attr_hist_hatonly', if_cond_label=True,
    #                                                used_neurons='resnet_quarter',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_hatonly.npy',
    #                                                if_calc_wasserstein=True, offset=(0.0, 0.0), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'nonsparse_afterrelu_add1_2cifar.pkl', {14:'all'},
    #                                                'attr_hist_add1_2cifar', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_add1_2cifar.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    
    path_prefix_mnt = Path("/mnt/raid/data/chebykin/pycharm_project_AA/")
    # wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, path_prefix_mnt/'bettercifar10single_nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_bettercifar10single', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                sorted_dict_path=path_prefix_mnt/'img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5),
    #                                                dataset_type='cifar', if_show=False, use_wasserstein_divergence=True)

    wc.compute_attr_hist_for_neuron_pandas_wrapper(loader, path_prefix_mnt/'pretrained_imagenet_afterrelu.pkl', 'all_network', 'attr_hist_pretrained_imagenet_afterrelu', if_cond_label=False, used_neurons='resnet_full', dataset_type='imagenet_val', sorted_dict_path=path_prefix_mnt/'img_paths_most_activating_sorted_dict_paths_pretrained_imagenet_afterrelu.npy', if_plot=False,if_calc_wasserstein=True, offset='argmax', if_show=False, use_wasserstein_divergence=True)
    
    exit()

    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single2_nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_bettercifar10single2', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single2.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)

    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single3_nonsparse_afterrelu.pkl',
    #                                                'all_network',
    #                                                'attr_hist_bettercifar10single3', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single3.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single4_nonsparse_afterrelu.pkl',
    #                                                'all_network',
    #                                                'attr_hist_bettercifar10single4', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single4.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single5_nonsparse_afterrelu.pkl',
    #                                                'all_network',
    #                                                'attr_hist_bettercifar10single5', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single5.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'imagenette_nonsparse_afterrelu.pkl',
    #                                                'all_network',
    #                                                'attr_hist_imagenette', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                dataset_type='imagenette',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_imagenette.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False, if_left_lim_zero=True)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'imagenette_neg_nonsparse_afterrelu.pkl',
    #                                                'all_network',
    #                                                'attr_hist_imagenette_neg', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_imagenette_neg.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False, if_left_lim_zero=False,
    #                                                layer_list=layers_bn_prerelu)

    # ac.convert_label_dict_to_dataframe(np.load('path_to_label_dict_cifar_val.npy', allow_pickle=True).item(),
    #                                    'df_label_cifar.pkl', if_nonceleba=True)
    # ac.assess_ifelse_predictor('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 7, 0.5)
    # ac.store_highest_activating_images('highest_activating_ims_imagenette_neg',
    #                                    df_path='imagenette_neg_nonsparse_afterrelu.pkl', individual_size=224,
    #                                    layer_list=layers_bn_prerelu)
    # exit()

    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 7, 8)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 0, 9)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 61, 7, 8)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 61, 0, 9)

    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 1, 7)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 1, 9)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 61, 1, 7)
    # ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 61, 1, 9)
    # exit()
    # ac.convert_label_dict_to_dataframe(np.load('path_to_label_dict_celeba_val.npy', allow_pickle=True).item(), 'df_label_celeba.pkl')
    # ac.assess_ifelse_predictor('sparse_afterrelu.pkl', 'df_label_celeba.pkl', 14, 421, 33, 0.65)
    # exit()
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist', if_cond_label=True,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu.npy',
    #                                                if_calc_wasserstein=True, offset=(0.0, 0.0))
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_contrast', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu.npy',
    #                                                if_calc_wasserstein=True, offset=(-0.3, 0.3))
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_celeba_afterrelu.pkl', {14:[187]},
    #                                                'attr_hist_sparse_celeba', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                dataset_type='celeba',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_sparse_celeba_afterrelu.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=True,
    #                                                if_force_recalculate=True,
    #                                                if_left_lim_zero=True, layer_list=layers_bn_afterrelu)

    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'cifarfashion_sparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_cifarshion', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy', if_nonceleba=True,
    #                         sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_cifarfashion.npy')


    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_07_on_September_10/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._120_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_smilingonly_weightedce.json'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
    # _, loader, _ = datasets.get_dataset(params, configs)
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'smilingonly_nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_smilingonly', if_cond_label=True,
    #                                                used_neurons='resnet_quarter',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_smilingonly.npy')
    #
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
    # _, loader, _ = datasets.get_dataset(params, configs)
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'sparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist', if_cond_label=False,
    #                                                used_neurons=f'actually_good_nodes_{model_name_short}.npy',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path=None)

    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_51_on_May_21/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_15_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc.json'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
    # _, loader, _ = datasets.get_dataset(params, configs)
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_fc', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path=None)
    #
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bangsonly_nonsparse_afterrelu2.pkl', {14:'all'},
    #                                                'attr_hist_bangsonly2', if_cond_label=True,
    #                                                used_neurons='resnet_quarter',
    #                                                dataset_type='celeba',
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bangsonly2.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False,
    #                                                if_force_recalculate=True,
    #                                                if_left_lim_zero=True, layer_list=layers_bn_afterrelu
    #                                                )

    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_32_on_September_14/optimizer=SGD|batch_size=256|lr=0.005|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0_66_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd005bias_fc.json'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
    # _, loader, _ = datasets.get_dataset(params, configs)
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'cifar10_nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_cifar10', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                if_nonceleba=True,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_cifar10.npy')

    # sorted_dict_per_layer_per_neuron = ac.find_highest_activating_images(loader,
    #                                              save_path='img_paths_most_activating_sorted_dict_paths_afterrelu.npy',
    #                                   if_sort_by_path=True)

    # if True:
    #     sorted_dict_per_layer_per_neuron = ac.find_highest_activating_images(loader, if_nonceleba=False,
    #                                          save_path='img_paths_most_activating_sorted_dict_paths_afterrelu_hatonly.npy',
    #                                                                          used_neurons=used_neurons,
    #                                                                          if_sort_by_path=True
    #                                                                          )
    # else:
    #     sorted_dict_per_layer_per_neuron = np.load('img_paths_most_activating_sorted_dict_afterrely_hatonly.npy',
    #                                            allow_pickle=True).item()

    # ac.compute_attr_hist_for_neuron(11, [279], 15, list(range(40)), lambda v, p, i: v < 0, False,
    #                                 sorted_dict_per_layer_per_neuron)
    # ac.compute_attr_hist_for_neuron(13, [204], 15, list(range(40)), lambda v, p, i: v < 0)
    # ac.compute_attr_hist_for_neuron(12, [421, 392, 406, 261], 15, list(range(40)), lambda v, p, i: bool(random.getrandbits(1)))
    #
    # if False:
    #     df = ac.convert_sorted_dict_per_layer_per_neuron_to_dataframe(sorted_dict_per_layer_per_neuron, out_path='hatonly_nonsparse_afterrelu.pkl')
    # else:
    #     df = pd.read_pickle('/mnt/raid/data/chebykin/pycharm_project_AA/hatonly_nonsparse_afterrelu.pkl')
    # print(df)
    # ac.compute_attr_hist_for_neuron_pandas(14, 'all', 15, list(range(40)), lambda v, p, i: v < 0, True, df=df,
    #                                 out_dir='attr_hist_for_neuron_hatonly_normed', used_neurons=used_neurons)
    # # ac.compute_attr_hist_for_neuron(14, list(range(1, 64)), 15, list(range(40)), lambda v, p, i: v < 0)
    # # ac.compute_attr_hist_for_neuron(14, list(range(1, 64)), 15, list(range(40)), if_cond_labels=True)
    # # sorted_dict_per_layer_per_neuron = np.load('img_paths_most_activating_sorted_dict_afterrelu_fcmodel.npy',
    # #                                            allow_pickle=True).item()
    exit()
    # for i in range(15):
    #     # ac.compute_attr_hist_for_neuron(i, 'all', 15, list(range(40)), lambda v, p, i: v < 0, False,
    #     #                             sorted_dict_per_layer_per_neuron)
    #     ac.compute_attr_hist_for_neuron_pandas(i, 'all', 15, list(range(40)), lambda v, p, i: v < 0, False, df=df,
    #                                 out_dir='attr_hist_for_neuron_fc_normed', used_neurons=used_neurons)
    # # ac.compute_attr_hist_for_neuron(14, 'all', 15, list(range(40)), lambda v, p, i: v < 0, True,
    # #                                 used_neurons=used_neurons, out_dir='attr_hist_for_neuron_hatonly')
    # exit()