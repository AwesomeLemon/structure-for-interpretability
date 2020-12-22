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
except:
    import datasets
    from gan.attgan.data import CustomDataset
    # from gan.change_attributes import AttributeChanger
    from load_model import load_trained_model
    from loaders.celeba_loader import CELEBA
    from util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict

from models.binmatr2_multi_faces_resnet import BasicBlockAvgAdditivesUser

import glob
from shutil import copyfile, copy
import math
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivityCollector():
    def __init__(self, model, im_size, use_my_model=True):
        self.model = model
        self.use_my_model = use_my_model
        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        if self.use_my_model:
            for m in self.model:
                model[m].zero_grad()
                model[m].eval()

            self.feature_extractor = self.model['rep']
        else:
            self.model.eval()
            self.model.zero_grad()
            self.feature_extractor = self.model
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)

    def img_from_np_to_torch(self, img):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.use_my_model:
            if True:
                # when running on normal CelebA images
                img -= np.array([73.15835921, 82.90891754, 72.39239876])
                std = [1, 1, 1]
            else:
                # when running on AM images
                img -= np.array([0.38302392, 0.42581415, 0.50640459]) * 255  #
                std = [0.2903, 0.2909, 0.3114]
        else:
            img -= np.array([0.485, 0.456, 0.406]) * 255
            std = [0.229, 0.224, 0.225]

        img = skimage.transform.resize(img, (self.size_0, self.size_1), order=3)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # HWC -> CWH
        img = img.transpose(2, 0, 1)

        for channel, _ in enumerate(img):
            img[channel] /= std[channel]

        img = torch.from_numpy(img).float()
        img.unsqueeze_(0)
        return img  # .requires_grad_(True)

    def img_from_torch_to_int_np(self, im_as_var):
        recreated_im = np.copy(im_as_var.cpu().data.numpy()[0])

        if not self.use_my_model or True:
            std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
            for channel, _ in enumerate(recreated_im):
                recreated_im[channel] *= std[channel]

        recreated_im = recreated_im.transpose((1, 2, 0))

        recreated_im *= 255.0
        if self.use_my_model:
            recreated_im += np.array([73.15835921, 82.90891754, 72.39239876])
        else:
            recreated_im += np.array([0.485, 0.456, 0.406]) * 255

        recreated_im[recreated_im > 255.0] = 255.0
        recreated_im[recreated_im < 0.0] = 0.0

        recreated_im = np.round(recreated_im)
        recreated_im = np.uint8(recreated_im)

        return recreated_im

    def get_activations_single_image(self, input_img_path):
        img = np.asarray(PIL.Image.open(input_img_path))
        img = self.img_from_np_to_torch(img).to(device)

        # img = self.img_from_torch_to_int_np(img)
        # im_path = 'test.jpg'
        # save_image(img, im_path)

        def save_activation(activations, name, mod, inp, out):
            if type(out) == list:
                assert len(out) == 40
                for i, out_cur in enumerate(out):
                    activations[f'features_task_{i}'] = out_cur[0].cpu()
            else:
                # why "[0]": this is batch size, which is always == 1 here.
                activations[name] = out[0].cpu()

        activations = {}

        hooks = []

        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        if False:
            self.feature_extractor(img, None)[0]
        else:
            self.feature_extractor(img)

        # a problem that I ignore for now is that for each layer there exist not only 'layer1.1' as final output, but also simply 'layer1'

        for hook in hooks:
            hook.remove()

        return activations

    def store_layer_activations_many(self, loader, target_layer_indices, if_average_spatial=True, out_path_postfix='',
                                     if_save_separately=True, if_store_labels=False, if_store_layer_names=False,
                                     layer_list=layers, if_store_activations=True):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if if_average_spatial:
                cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
            else:
                cur_activations = out.detach().cpu().numpy()
            # old version:
            # if name in activations:
            #     cur_activations = np.append(activations[name], cur_activations, axis=0)
            # activations[name] = cur_activations
            # new version:
            if name not in activations:
                activations[name] = [cur_activations]
            else:
                activations[name].append(cur_activations)

        hooks = []
        for name, m in self.feature_extractor.named_modules():
            if name in target_layer_names:
                hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        labels = []
        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                print(i)

                if if_store_labels:
                    labels += list(np.array(batch_val[-1]))

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)

        for hook in hooks:
            hook.remove()

        if if_store_activations:
            if if_save_separately:
                if if_store_layer_names:
                    raise NotImplementedError()
                for i, idx in enumerate(target_layer_indices):
                    if if_average_spatial:
                        filename = f'activations_on_validation_{idx}{out_path_postfix}.npy'
                    else:
                        filename = f'activations_on_validation_preserved_spatial_{idx}{out_path_postfix}.npy'
                    with open(filename, 'wb') as f:
                        pickle.dump(np.vstack(activations[target_layer_names[i]]), f, protocol=4)
            else:
                if not if_store_layer_names:
                    filename = 'representatives_kmeans14_50_alllayers.npy'
                    np.save(filename, activations)
                else:
                    filename = f'activations_on_validation_preserved_spatial_{out_path_postfix}.npy'
                    np.save(filename, (target_layer_names, activations), allow_pickle=True)
        if if_store_labels:
            np.save(f'labels{out_path_postfix}.npy', labels)

        return activations

    def find_highest_activating_images(self, loader, if_average_spatial=True, dataset_type='celeba', target_layer_indices=None,
                                       save_path='img_paths_most_activating_sorted_dict_afterrelu_fcmodel.npy', used_neurons=None,
                                       if_sort_by_path=False, if_dont_save=False, if_left_lim_zero=True, layer_list=layers_bn,
                                       if_save_argmax_labels=False, cond_neurons=None):
        target_layer_names = np.array([layer.replace('_', '.') for layer in layer_list] + ['label'])
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
                if ('relu2' in name) and not if_left_lim_zero:
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
                                save_image(cur_img, cur_path)
                                im_paths.append(cur_path)
                                im_paths_to_labels_dict[cur_path] = batch_val[1][j].item()
                            else:
                                cur_path = im_paths[j]
                        for k in used_neurons_cur:
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

    def store_highest_activating_images(self, out_path, sorted_dict_path=None, df_path=None, n_save=16,
                                        if_plot_hist=False,individual_size=None, layer_list=layers_bn):
        assert (sorted_dict_path is None) or (df_path is None), 'choose one'
        if_used_df = df_path is not None
        target_layer_names = [layer.replace('_', '.') for layer in layer_list] #+ ['label']
        layers_local_var = layer_list #+ ['label']
        if not if_used_df:
            sorted_dict_per_layer_per_neuron = defaultdict(lambda: dict(), np.load(sorted_dict_path, allow_pickle=True).item())
        else:
            df = pd.read_pickle(df_path)
        # out_path = 'highest_activating_ims_bn_hist_cifarfshmnst'
        Path(out_path).mkdir(exist_ok=True)
        # fontPath = "/usr/share/fonts/dejavu-lgc/DejaVuLGCSansCondensed-Bold.ttf"
        fontPath = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
        font = ImageFont.truetype(fontPath, 28)

        def plot_most_activating_ims(values_and_paths, if_plot_hist=False):
            ims = []
            for (value, path) in values_and_paths:
                if True:
                    # path = str.replace(path, 'cifar10/my_imgs2', 'imagenette/my_imgs')
                    path = str.replace(path, 'cifar10/my_imgs2', 'cifar/my_imgs')
                im = Image.open(path)
                b, g, r = im.split()
                im = Image.merge("RGB", (r, g, b))
                value_str = f'{value:.6f}'#str(value)[:7]
                # if (i == 12) and (neuron == 400):
                #     print(value)
                # if layers_local_var[i] == 'label':
                #     value_str = str(value.item())[:7]
                # ImageDraw.Draw(im).text((0, 0), value_str, (255, 150, 100), font=font)
                if individual_size is not None:
                    im = im.resize((individual_size, individual_size), Image.ANTIALIAS)
                ims.append(im)
            if if_plot_hist:
                plt.tight_layout()
                # plt.figure(figsize=(1.6, 2.2))
                proper_hist(np.array([k.item() for k in sorted_dict.keys()]), bin_size=0.01)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                im = Image.open(buf)
                im = im.resize((ims[0].size[0], ims[0].size[1]), Image.ANTIALIAS)
                ims.append(im)
            new_im = images_list_to_grid_image(ims)
            new_im.save(cur_folder + '/' + f'{neuron}.jpg')
            if if_plot_hist:
                buf.close()
                plt.close()

        for i in range(len(layers_local_var)):
            # if layers_local_var[i] != 'label':
            #      continue
            cur_folder = out_path + '/' + layers_local_var[i]
            Path(cur_folder).mkdir(exist_ok=True)

            if not if_used_df:
                for neuron, sorted_dict in sorted_dict_per_layer_per_neuron[target_layer_names[i]].items():
                    sorted_dict_list = list(sorted_dict.items())
                    if n_save == 'all':
                        values_and_paths = list(reversed(sorted_dict_list))
                    else:
                        values_and_paths = list(reversed(sorted_dict_list[-n_save // 2:])) + sorted_dict_list[:n_save // 2 - int(if_plot_hist)]
                    # values_and_paths = sorted_dict_list[:n_save]
                    plot_most_activating_ims(values_and_paths, if_plot_hist)
            else:
                for row in df.loc[(df['layer_name'] == target_layer_names[i])].itertuples():
                    neuron = row[2]
                    values_and_paths_all = list(sorted(zip(row[3:], df.columns[2:])))
                    if n_save == 'all':
                        values_and_paths = list(reversed(values_and_paths_all))
                    else:
                        values_and_paths = list(reversed(values_and_paths_all[-n_save // 2:])) + \
                                       values_and_paths_all[:n_save // 2 - int(if_plot_hist)]

                    plot_most_activating_ims(values_and_paths, if_plot_hist)

    def compute_attr_hist_for_neuron(self, target_layer_idx, target_neurons,
                                     cond_layer_idx, cond_neurons, cond_predicate=None,
                                     if_cond_labels=False, sorted_dict_per_layer_per_neuron=None,
                                     out_dir='attr_hist_for_neuron_fc_all', used_neurons=None):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn] + ['label']
        if sorted_dict_per_layer_per_neuron is None:
            sorted_dict_per_layer_per_neuron = np.load('img_paths_most_activating_sorted_dict_afterrely_hatonly.npy',#'img_paths_most_activating_sorted_dict_afterrelu.npy',
                                                           allow_pickle=True).item()
        Path(out_dir).mkdir(exist_ok=True)

        n_cond_indices = len(cond_neurons)
        rows_num = math.floor(math.sqrt(n_cond_indices))
        cols_num = int(math.ceil(n_cond_indices / rows_num))
        if target_neurons == 'all':
            if used_neurons is None:
                used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()
                target_neurons = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[target_layer_idx]])
            else:
                target_neurons = used_neurons[target_layer_idx]

        if if_cond_labels:
            path_to_label_dict = np.load('path_to_label_dict_celeba_val.npy', allow_pickle=True).item()
            cond_predicate = lambda _, path, idx: path_to_label_dict[path][idx] == 0

        sorted_dict_cond_layer = sorted_dict_per_layer_per_neuron[target_layer_names[cond_layer_idx]]
        if False:
            print("WARNING! Applying a hack specific to wearinghatonly")
            for i in range(40):
                sorted_dict_cond_layer[i] = sorted_dict_cond_layer[35]
        selected_paths1_cache = {}
        selected_paths2_cache = {}
        for target_neuron in target_neurons:
            print(target_neuron)
            cols_rows_ratio = cols_num / rows_num
            _, axs = plt.subplots(rows_num, cols_num, squeeze=False, figsize=(14 * cols_rows_ratio, 14 / cols_rows_ratio))
            axs = axs.flatten()

            sorted_dict = sorted_dict_per_layer_per_neuron[target_layer_names[target_layer_idx]][target_neuron]
            sorted_dict_items = sorted_dict.items()

            for cond_i, idx in enumerate(cond_neurons):
                title = str(idx)
                if cond_layer_idx == 15:
                    title = celeba_dict[idx]
                bin_size = 0.01
                sorted_dict_cond_items = sorted_dict_cond_layer[idx].items()

                if idx not in selected_paths1_cache:
                    selected_paths1_cache[idx] = set([path for (value, path) in sorted_dict_cond_items if cond_predicate(value, path, idx)])
                selected_paths1 = selected_paths1_cache[idx]
                selected_values_list = np.array([value for (value, path) in sorted_dict_items if path in selected_paths1])
                if len(selected_values_list) == 0:
                    continue

                diff = selected_values_list.max() - selected_values_list.min()
                if diff == 0.0:
                    continue
                while diff / bin_size < 10:
                    bin_size /= (diff / bin_size)
                    print(f'New bin size: {bin_size}')
                while diff / bin_size > 100:
                    bin_size *= 2
                    print(f'New bin size: {bin_size}')

                proper_hist(selected_values_list, bin_size=bin_size, ax=axs[cond_i], xlim_left=0, density=True)

                if idx not in selected_paths2_cache:
                    selected_paths2_cache[idx] = set([path for (value, path) in sorted_dict_cond_items if not cond_predicate(value, path, idx)])
                selected_paths2 = selected_paths2_cache[idx]
                selected_values_list = np.array([value for (value, path) in sorted_dict_items if path in selected_paths2])
                if len(selected_values_list) == 0:
                    continue
                proper_hist(selected_values_list, bin_size=bin_size, ax=axs[cond_i], alpha=0.75, title=title, xlim_left=0, density=True)

            plot_name = f'{target_layer_idx}_{target_neuron}'
            plt.suptitle(plot_name)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/hist_{plot_name}.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.show()


    def compute_attr_hist_for_neuron_pandas(self, target_layer_idx, target_neurons,
                                     cond_layer_idx, cond_neurons,
                                     if_cond_labels=False, df=None,
                                     out_dir='attr_hist_for_neuron_fc_all', used_neurons=None, dataset_type='celeba',
                                     if_calc_wasserstein=False, offset=(0, 0), if_show=True, if_left_lim_zero=True,
                                            layer_list=layers_bn, if_plot=True, cond_neuron_lambda=lambda x:[x]):
        # when if_show==False, hists are plotted and saved, but not shown
        # when if_plot==False, nothing is plotted, only wasserstein distances are calculated
        target_layer_names = [layer.replace('_', '.') for layer in layer_list] + ['label']
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

        # if dataset_type == 'broden_val':
        #     print(df.columns)
        #     clmns = [i.item() for i in df.columns[2:]]
        #     df.columns = clmns

        selected_paths1_cache = {}
        selected_paths2_cache = {}
        for target_neuron in target_neurons:
            print(target_neuron)
            if if_plot:
                _, axs = plt.subplots(rows_num, cols_num, squeeze=False, figsize=(10 * 1.5, 14 * 1.5))
                axs = axs.flatten()

            selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                                   (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'], axis=1))

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
                            dict_from_full_imagenet_idx = dict(zip([0, 217, 482, 491, 497, 566, 569, 571, 574, 701], range(10)))
                            title = imagenette_dict[dict_from_full_imagenet_idx[idx]]
                        else:
                            raise NotImplementedError()

                if idx not in selected_paths1_cache:
                    corresponding_indices = cond_neuron_lambda(idx)
                    final_mask = np.array([False]*selected_values_list.shape[1])[None, :] #shape is number of images
                    for ind in corresponding_indices:
                        cond_mask1 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                                        (df_cond['neuron_idx'] == ind)].drop(['layer_name', 'neuron_idx'], axis=1) <= offset[0]
                        final_mask += np.array(cond_mask1)
                    selected_paths1_cache[idx] = np.array(final_mask, dtype=bool)
                cond_mask1 = selected_paths1_cache[idx]
                selected_values_list1 = selected_values_list[cond_mask1]
                if len(selected_values_list1) == 0:
                    print('continue1', idx)
                    continue

                # xlim_right = 12
                # bin_size = 0.08
                if if_plot:
                    proper_hist(selected_values_list1, bin_size=bin_size, ax=axs[cond_i], xlim_left=xlim_left,
                            xlim_right=xlim_right,
                            density=True)

                if idx not in selected_paths2_cache:
                    corresponding_indices = cond_neuron_lambda(idx)
                    final_mask = np.array([False] * selected_values_list.shape[1])[None, :] # shape is number of images
                    for ind in corresponding_indices:
                        cond_mask2 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                                                 (df_cond['neuron_idx'] == ind)].drop(['layer_name', 'neuron_idx'],
                                                                                      axis=1) > offset[1]
                        final_mask += np.array(cond_mask2)
                    selected_paths2_cache[idx] = np.array(final_mask, dtype=bool)
                cond_mask2 = selected_paths2_cache[idx]
                selected_values_list2 = selected_values_list[cond_mask2]
                if len(selected_values_list2) == 0:
                    print('continue2', idx)
                    continue

                if if_calc_wasserstein:
                    # wd = scipy.stats.wasserstein_distance(selected_values_list1, selected_values_list2)
                    # title += '\n'
                    # title += f'{wd:.2f}'
                    wd_normed = scipy.stats.wasserstein_distance(
                        (selected_values_list1 - selected_values_min) / (selected_values_max - selected_values_min),
                        (selected_values_list2 - selected_values_min) / (selected_values_max - selected_values_min))
                    wd_normed *= np.sign(selected_values_list2.mean() - selected_values_list1.mean())
                    if if_plot:
                        title += f'\n{wd_normed:.2f}'
                    wasserstein_dists_dict[target_neuron][idx] = wd_normed


                if if_plot:
                    proper_hist(selected_values_list2, bin_size=bin_size, ax=axs[cond_i], xlim_left=xlim_left,
                            xlim_right=xlim_right,
                            alpha=0.75, title=title, density=True)



            if if_plot:
                plot_name = f'{target_layer_idx}_{target_neuron}'
                plt.suptitle(plot_name)
                plt.tight_layout()
                plt.savefig(f'{out_dir}/hist_{plot_name}.png', format='png', bbox_inches='tight', pad_inches=0)
                if if_show:
                    plt.show()
                plt.close()

        if if_calc_wasserstein:
            np.save('wasser_dists/wasser_dist_' + out_dir + '_' + str(target_layer_idx) + '.npy', wasserstein_dists_dict, allow_pickle=True)

    def wasserstein_barplot(self, suffix, target_neurons_dict, classes_dict, n_show=10, if_show=False, out_path_suffix=''):
        n_best = n_show // 2
        out_path = 'wd_barplot_' + suffix + out_path_suffix
        Path(out_path).mkdir(exist_ok=True)
        for layer_idx, neurons in target_neurons_dict.items():
            wasser_dists_cur = np.load(f'wasser_dists/wasser_dist_{suffix}_{layer_idx}.npy', allow_pickle=True).item()
            for neuron in neurons:
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

    def plot_overall_hists(self, df_path, out_dir, layer_list):
        target_layer_names = [layer.replace('_', '.') for layer in layer_list]
        Path(out_dir).mkdir(exist_ok=True)
        chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        df = pd.read_pickle(df_path)
        for target_layer_idx, target_layer_name in enumerate(target_layer_names):
            for target_neuron in range(chunks[target_layer_idx]):
                print(target_neuron)
                plot_name = f'{target_layer_idx}_{target_neuron}'
                selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                               (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                         axis=1)).flatten()
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
                proper_hist(selected_values_list, plot_name, bin_size=bin_size, xlim_left=0)
                plt.savefig(f'{out_dir}/{plot_name}.png', format='png', bbox_inches='tight', pad_inches=0)
                plt.close()

    def convert_label_dict_to_dataframe(self, path_to_label_dict, out_path, if_nonceleba=False):
        path_label_items = list(path_to_label_dict.items())
        path_label_items.sort(key=operator.itemgetter(0))

        if not if_nonceleba:
            n_labels = len(path_label_items[0][1])
        else:
            n_labels = 10
        per_label_dict = defaultdict(list)
        for path, labels in path_label_items:
            for i in range(n_labels):
                if not if_nonceleba:
                    per_label_dict[i].append(labels[i])
                else:
                    label = int(labels == i)
                    per_label_dict[i].append(label)
        data = []
        for i in range(n_labels):
            data.append(['label', i, *per_label_dict[i]])
        df_labels = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] + list(list(zip(*path_label_items))[0]))
        if out_path is not None:
            df_labels.to_pickle(out_path)
        return df_labels


    def assess_ifelse_predictor(self, df_path, df_labels_path, target_layer_idx, target_neuron, cond_idx, decision_boundary):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn] + ['label']
        df = pd.read_pickle(df_path)
        selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                               (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                         axis=1))
        predicted = selected_values_list > decision_boundary

        df_labels = pd.read_pickle(df_labels_path)
        actual = np.array(df_labels.loc[(df_labels['layer_name'] == 'label') & (df_labels['neuron_idx'] == cond_idx)].drop(
            ['layer_name', 'neuron_idx'], axis=1))
        print(balanced_accuracy_score(actual[0], predicted[0].astype(int)))


    def assess_binary_separability(self, df_path, df_labels_path, target_layer_idx, target_neuron, cond_idx1, cond_idx2):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn] + ['label']
        df = pd.read_pickle(df_path)
        target_neuron_activations = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                               (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                         axis=1))[0]

        df_labels = pd.read_pickle(df_labels_path)
        labels1 = np.array(df_labels.loc[(df_labels['layer_name'] == 'label') & (df_labels['neuron_idx'] == cond_idx1)].drop(
            ['layer_name', 'neuron_idx'], axis=1))[0]
        labels2 = np.array(df_labels.loc[(df_labels['layer_name'] == 'label') & (df_labels['neuron_idx'] == cond_idx2)].drop(
            ['layer_name', 'neuron_idx'], axis=1))[0]
        train_pos = target_neuron_activations[labels1 == 1]
        lbl_pos = np.array([1] * len(train_pos))
        train_neg = target_neuron_activations[labels2 == 1]
        lbl_neg = np.array([-1] * len(train_neg))
        train = np.hstack((train_pos, train_neg))
        lbl = np.hstack((lbl_pos, lbl_neg))
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate
        clf = LogisticRegression(random_state=0, penalty='none')
        cv_results = cross_validate(clf, train.reshape(-1, 1), lbl, cv=5)
        print(f'{target_layer_idx}_{target_neuron}: {cifar10_dict[cond_idx1]} vs {cifar10_dict[cond_idx2]}: ',
              cv_results['test_score'])

    def assess_binary_separability_1vsAll_whole_network(self, df_path, df_labels_path, layer_list, n_classes, postfix):
        target_layer_names = [layer.replace('_', '.') for layer in layer_list]
        df = pd.read_pickle(df_path)
        df_labels = pd.read_pickle(df_labels_path)
        chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        res_data = []
        for i, target_layer_name in enumerate(target_layer_names):
            for target_neuron in range(chunks[i]):
                print(i, target_neuron)
                target_neuron_activations = np.array(df.loc[(df['layer_name'] == target_layer_name) &
                                                       (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                                 axis=1))[0]
                for class_ind in range(n_classes):
                    if class_ind % 50 == 0:
                        print(class_ind)
                    labels1 = np.array(df_labels.loc[(df_labels['layer_name'] == 'label') & (df_labels['neuron_idx'] == class_ind)]
                                       .drop(['layer_name', 'neuron_idx'], axis=1))[0]
                    train_pos = target_neuron_activations[labels1 == 1]
                    lbl_pos = np.array([1] * len(train_pos))
                    train_neg = target_neuron_activations[labels1 != 1]
                    lbl_neg = np.array([-1] * len(train_neg))
                    train = np.hstack((train_pos, train_neg))
                    lbl = np.hstack((lbl_pos, lbl_neg))

                    clf = LogisticRegression(random_state=0, penalty='l2', class_weight="balanced")
                    cv_results = cross_validate(clf, train.reshape(-1, 1), lbl, cv=5, scoring='balanced_accuracy')
                    res_data.append([target_layer_name, target_neuron, class_ind, np.mean(cv_results['test_score'])])
                    # print(i, target_neuron, class_ind, np.mean(cv_results['test_score']))
            df_res = pd.DataFrame(res_data, columns=['layer_name', 'neuron_idx', 'class_idx', 'mean_balanced_acc'])
            df_res.to_csv('binary_separability' + postfix + f'_{i}.csv')


    def compute_attr_hist_for_neuron_pandas_wrapper(self, loader, df_val_images_activations_path, target_dict,
                                                    out_dir_path, if_cond_label=False,
                                                    sorted_dict_path=None,
                                                    used_neurons=None, dataset_type='celeba', if_calc_wasserstein=False,
                                                    offset=(0, 0), if_show=True, if_force_recalculate=False, if_dont_save=False,
                                                    if_left_lim_zero=True, layer_list=layers_bn, if_plot=True):
        '''

        :param target_dict: 'all_network' OR {target_layer_idx -> [neuron_indices] OR 'all'}
        :param if_cond_label: whether separate into 2 histograms by true labels (in contrast to model predictions)
        '''
        if target_dict == 'all_network':
            target_dict = {}
            for i in reversed(range(15)):
            # for i in reversed(range(12)):
                target_dict[i] = 'all'
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
        else:
            # cond_neurons = list(range(20))
            # cond_neurons = list(range(10))
            # cond_neurons = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
            # cond_neurons = list(range(1000))#[0, 217, 482, 491, 497, 566, 569, 571, 574, 701] + list(range(1, 10)) + list(range(901, 911))
            # cond_neurons = list(range(301))
            # cond_neurons = list(range(1197))
            cond_neurons = list(range(10))

        if not os.path.exists(df_val_images_activations_path) or if_force_recalculate:
            if not os.path.exists(sorted_dict_path) or if_force_recalculate:
                sorted_dict_per_layer_per_neuron = ac.find_highest_activating_images(loader, dataset_type=dataset_type,
                                                                                     save_path=None, #note that I decided not to save the dict. df should be enough
                                                                                     used_neurons=used_neurons,
                                                                                     if_sort_by_path=True,
                                                                                     if_dont_save=if_dont_save,
                                                                                     if_left_lim_zero=if_left_lim_zero,
                                                                                     layer_list=layer_list,
                                                                                     if_save_argmax_labels=offset == 'argmax',
                                                                                     cond_neurons=cond_neurons)
                print('Sorted dict calculated')
            else:
                sorted_dict_per_layer_per_neuron = np.load(sorted_dict_path, allow_pickle=True).item()
            df = ac.convert_sorted_dict_per_layer_per_neuron_to_dataframe(sorted_dict_per_layer_per_neuron,
                                                                          out_path=df_val_images_activations_path)
            print('Converted to df')
        else:
            df = pd.read_pickle(os.path.abspath(df_val_images_activations_path))


        for layer_idx, targets in target_dict.items():
            ac.compute_attr_hist_for_neuron_pandas(layer_idx, targets, -1, cond_neurons, if_cond_label, df,
                                                   out_dir_path, used_neurons, dataset_type, if_calc_wasserstein,
                                                   offset, if_show, if_left_lim_zero, layer_list, if_plot
                                                   #,cond_neuron_lambda=lambda x: hypernym_idx_to_imagenet_idx[x]
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

    def cluster_stored_layer_activations(self, target_layer_indices, if_average_spatial=True):
        used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()
        representatives = {}
        for target_layer_idx in target_layer_indices:
            print(target_layer_idx)
            target_layer_name = layers[target_layer_idx].replace('_', '.')
            if if_average_spatial:
                filename = f'activations_on_validation_{target_layer_idx}.npy'
            else:
                filename = f'activations_on_validation_preserved_spatial_{target_layer_idx}.npy'
            # activations = np.load(filename, allow_pickle=True).item()
            with open(filename, 'rb') as f:
                activations = pickle.load(f)
            used_neurons_cur = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[target_layer_idx]])
            print(target_layer_name)
            if type(activations) == dict:
                x = activations[target_layer_name]
            else:
                x = activations
            x = x[:, used_neurons_cur]
            x_shape = x.shape
            x = x.reshape((x.shape[0], -1))
            # scaler = sklearn.preprocessing.StandardScaler()
            # x = scaler.fit_transform(x)

            # clustering = skcl.OPTICS(n_jobs=8, max_eps=5.0).fit(x)
            # clustering = sklearn_extra.cluster.KMedoids(n_clusters=300, metric='cosine', init='k-medoids++', max_iter=2000).fit(x)
            # clustering = sklearn.cluster.DBSCAN(n_jobs=8, eps=0.15, metric='cosine', min_samples=10).fit(x)
            clustering = sklearn.cluster.KMeans(50, max_iter=1500).fit(x)

            # labels, counts = np.unique(clustering.labels_, return_counts=True)
            # gm = sklearn.mixture.GaussianMixture(n_components=10, verbose=1).fit(x)
            # x_mdsed = sklearn.manifold.MDS(n_components=2, n_jobs=8).fit_transform(x)
            # plt.scatter(x_mdsed[:, 0], x_mdsed[:, 1])
            # plt.show()
            # print(clustering.labels_)
            representatives_cur = clustering.cluster_centers_
            representatives[target_layer_name] = representatives_cur.reshape(
                (representatives_cur.shape[0], x_shape[1], x_shape[2], x_shape[3]))
        np.save('representatives_kmeans14_50.npy', representatives)

    def get_images_open_mouth_nonblond(self, configs):
        _, loader, _ = datasets.get_dataset(params, configs)
        res_images = []
        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                if i % 10 == 0:
                    print(i)
                val_images = batch_val[0]
                label_9 = batch_val[9 + 1].detach().cpu().numpy()
                label_21 = batch_val[21 + 1].detach().cpu().numpy()

                label_9_false = label_9 == 0
                label_21_true = label_21 == 1
                both = np.logical_and(label_9_false, label_21_true)
                if len(np.where(both)[0]) > 0:
                    cur_images = val_images[both].clone().detach()
                    res_images.append(cur_images)

                break

        return res_images

    def compare_AM_images(self, class1: int, class2: int):
        activations1 = self.get_activations_single_image(f'generated_best/{class1}.jpg')
        activations2 = self.get_activations_single_image(f'generated_best/{class2}.jpg')

        # for now compare only layer4
        # shape is CHW torch.Size([512, 24, 21])
        activations1 = activations1['layer4']
        activations2 = activations2['layer4']

        strengths1 = []
        strengths2 = []

        assert activations1.size(0) == activations2.size(0)
        for i in range(activations1.size(0)):
            strengths1.append(torch.norm(activations1[i], 1).item())
            strengths2.append(torch.norm(activations2[i], 1).item())

        strengths1 = np.array(strengths1)
        strengths2 = np.array(strengths2)

        plt.hist(strengths1 - strengths2)
        plt.show()

        plt.plot(strengths1, 'o')
        plt.plot(strengths2, 'o')
        plt.show()

        print(strengths1 - strengths2)
        print(len(np.arange(512)[np.abs(strengths1 - strengths2) > 400.0]))

    def get_output_probs_single_image(self, input_img_path):
        img = np.asarray(PIL.Image.open(input_img_path))
        img = self.img_from_np_to_torch(img).to(device)

        out = self.feature_extractor(img)

        all_tasks = [str(task) for task in range(40)]
        probs = np.ones((40)) * -17
        for i, t in enumerate(all_tasks):
            out_t_val, _ = self.model[t](out[i], None)
            # WHY SOFTMAX?
            probs[i] = torch.nn.functional.softmax(out_t_val, dim=1)[0][1].item()

        return probs

    def get_output_probs_many_images_stupid(self, img_paths):
        probs = []
        for i, path in enumerate(img_paths):
            probs.append(self.get_output_probs_single_image(path))
            if i % 100 == 0:
                print(i)
        return probs

    def get_output_probs_many_images(self, img_paths, if_use_celeba_loader=False):
        if not if_use_celeba_loader:
            # labels aren't needed here => placeholders
            dataset = CustomDataset(img_paths, [-17] * len(img_paths),
                                    lambda img: self.img_from_np_to_torch(np.asarray(img))[0],
                                    lambda x: -17)
        else:
            dataset = CELEBA(root=configs['celeba']['path'], is_transform=True, split='custom',
                             img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                             augmentations=None, custom_img_paths=img_paths)
        batch_size = 250#1450  #
        dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False, drop_last=False)
        probs = np.ones((len(img_paths), 40)) * -17
        all_tasks = [str(task) for task in range(40)]

        with torch.no_grad():
            # for batch_idx, (img, _) in enumerate(dataloader):
            for batch_idx, img_and_lbls in enumerate(dataloader):
                img = img_and_lbls[0]
                print(batch_idx)
                out = self.feature_extractor(img.to(device))
                for task_idx, task_name in enumerate(all_tasks):
                    out_t_val, _ = self.model[task_name](out[task_idx], None)
                    exped = torch.exp(out_t_val).cpu().numpy()
                    probs[batch_size * batch_idx:batch_size * (batch_idx + 1), task_idx] = exped[:, 1]
                    # for img_idx in range(img.shape[0]):
                    #     probs[batch_size * batch_idx + img_idx, task_idx] = exped[img_idx][1].item()

                # print(out[8].detach().cpu().mean(axis=0)[[172, 400, 383]])
                # print(out[9].detach().cpu().mean(axis=0)[[356, 204, 126, 187, 123, 134, 400, 383]])
                # print(out[11].detach().cpu().mean(axis=0)[[164, 400, 383]])
        return probs

    def get_feature_distribution_per_class(self, target: int, n_images: int, folder_suffix):
        strengths_sum = None
        for i in range(n_images):
            activations = self.get_activations_single_image(f'generated_10_{folder_suffix}/{target}_{i}.jpg')

            # for now compare only layer4
            # shape is CHW torch.Size([512, 24, 21])
            activations_last = activations['layer4']

            strengths = []
            for i in range(activations_last.size(0)):
                strengths.append(torch.norm(activations_last[i], 1).item())

            strengths = np.array(strengths)
            if strengths_sum is None:
                strengths_sum = strengths
            else:
                strengths_sum += strengths

        return strengths_sum / float(n_images)

    def get_target_probs_per_class(self, target: int, n_images: int, folder_suffix):
        probs_sum = None
        for i in range(n_images):
            probs = self.get_output_probs_single_image(f'generated_separate_{folder_suffix}/{target}_{i}.jpg')
            probs = np.array(probs)
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs

        return probs_sum / float(n_images)

    def get_feature_distribution_many(self, targets, n_images, folder_suffix):
        n_features = 512
        res = np.ones((len(targets), n_features))
        targets = [int(target) for target in targets]
        for target in targets:
            print(target)
            res[target] = self.get_feature_distribution_per_class(target, n_images, folder_suffix)
        return res

    def get_target_probs_many(self, targets, n_images, folder_suffix):
        n_tasks = 40
        res = np.ones((len(targets), n_tasks))
        targets = [int(target) for target in targets]
        for i, target in enumerate(targets):
            print(target)
            res[i] = self.get_target_probs_per_class(target, n_images, folder_suffix)
        return res

    def visualize_feature_distribution(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald',
                       5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry',
                       11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses',
                       16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male',
                       21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face',
                       26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns',
                       31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat',
                       36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        feature_distr = self.get_feature_distribution_many(targets, n_images, folder_suffix)
        for i in range(512):
            # feature_distr[:, i][feature_distr[:, i] < np.percentile(feature_distr[:, i], 75)] = 0
            feature_distr[:, i] -= feature_distr[:, i].mean()
            feature_distr[:, i] /= feature_distr[:, i].std()
        f = plt.figure(figsize=(19.20 * 1.55, 10.80 * 1.5))
        plt.tight_layout()
        # plt.matshow(feature_distr, fignum=f.number, vmin=-1, vmax=1, cmap='rainbow')
        # plt.imshow(feature_distr, aspect='auto')
        plt.pcolormesh(feature_distr, figure=f)  # ,cmap='rainbow')#,vmax=1200)#, edgecolors='k', linewidth=1)
        plt.xticks(range(512), [])
        # plt.yticks(range(len(targets)), [celeba_dict[target] for target in targets], fontsize=14)
        ax = plt.gca()
        ax.set_yticks(np.arange(.5, len(targets), 1))
        ax.set_yticklabels([celeba_dict[target] for target in targets])
        # ax.set_yticks(np.arange(-.5, len(targets), 1), minor=True)
        ax.set_aspect('auto')
        ax.get_xaxis().set_visible(False)
        cb = plt.colorbar(fraction=0.03, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        plt.savefig(f'features_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=1200)
        plt.show()

    def visualize_feature_histograms_per_task(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald',
                       5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry',
                       11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses',
                       16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male',
                       21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face',
                       26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns',
                       31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat',
                       36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        feature_distr = self.get_feature_distribution_many(targets, n_images, folder_suffix)
        for i in range(512):
            # feature_distr[:, i][feature_distr[:, i] < np.percentile(feature_distr[:, i], 75)] = 0
            feature_distr[:, i] -= feature_distr[:, i].mean()
            feature_distr[:, i] /= feature_distr[:, i].std()

        f = plt.figure(figsize=(19.20 * 1.55, 10.80 * 1.5))
        ax = f.subplots(nrows=5, ncols=8)

        for task in range(40):
            row = task // 8
            col = task - row * 8
            ax[row, col].hist(feature_distr[task, :], range=(-3, 3), bins=15)
            ax[row, col].set_ylim((0, 175))
            ax[row, col].set_title(celeba_dict[task])

        f.subplots_adjust(hspace=0.4)

        plt.savefig(f'feature_hists_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=1200)

    def visualize_probs_distribution(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald',
                       5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry',
                       11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses',
                       16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male',
                       21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face',
                       26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns',
                       31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat',
                       36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        probs = self.get_target_probs_many(targets, n_images, folder_suffix)
        f = plt.figure(figsize=(10.80 * 1.5, 10.80 * 1.5))
        plt.tight_layout()
        plt.pcolormesh(probs, figure=f, vmin=0, vmax=1)  # , edgecolors='k', linewidth=1)
        ax = plt.gca()
        ax.set_yticks(np.arange(.5, len(targets), 1))
        ax.set_yticklabels([celeba_dict[target] for target in targets])
        all = range(40)
        ax.set_xticks(np.arange(.5, len(all), 1))
        ax.set_xticklabels([celeba_dict[i] for i in all], rotation=90)
        cb = plt.colorbar(fraction=0.03, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        plt.savefig(f'probs_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.show()

    def store_images_based_on_label(self, decide_if_store, n_to_store, store_path):
        local_path = "/mnt/raid/data/chebykin/celeba"
        celeba_dataset = CELEBA(local_path, is_transform=False, augmentations=None, split='val')
        n_images = len(celeba_dataset.files[celeba_dataset.split])

        img_paths = []
        labels = []
        for i in range(n_images):
            img_path = celeba_dataset.files[celeba_dataset.split][i].rstrip()
            label = celeba_dataset.labels[celeba_dataset.split][i]

            if decide_if_store(label):
                img_paths.append(img_path)
                labels.append(label)

            if len(labels) == n_to_store:
                break

        if len(labels) < n_to_store:
            print(f'Warning: {n_to_store} images were requested, but only {len(labels)} were found.')
        np.save(store_path, (img_paths, labels))

    def calc_gradients_wrt_output(self, loader, target_layer_idx, target_neuron, cond_idx):
        for model in self.model.values():
            model.zero_grad()
            model.eval()
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad_(True)
        if target_layer_idx != 14:
            target_layer_name = layers_bn[target_layer_idx].replace('_', '.')
        else:
            target_layer_name = 'feature_extractor'

        def save_activation(activations, name, mod, inp, out):
            if name == target_layer_name:
                if type(out) == list:
                    #backbone output
                    out = out[0] #single-head cifar
                # print(out.shape)
                out.requires_grad_(True)
                activations['target'] = out

        activations = {}
        hooks = []
        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))
        hooks.append(self.feature_extractor.register_forward_hook(partial(save_activation, activations, 'feature_extractor')))

        grads = []
        for batch in loader:
            ims, labels = batch
            # preds = self.model['all'].linear(self.feature_extractor(ims.cuda())[0])[:, cond_idx].detach().cpu().numpy()
            # best_preds = np.argpartition(preds,
            #                              -5)[-5:]
            #                              # -25)[:-25]
            # best_preds_mask = np.array([False] * len(labels))
            # best_preds_mask[best_preds] = True
            mask = (labels == cond_idx) #* torch.tensor(best_preds_mask)
            ims_masked = ims[mask,...]
            ims_masked = ims_masked.cuda()
            out = self.feature_extractor(ims_masked)
            #single-headed
            y = out[0]
            # y.requires_grad_(True)
            # target = y
            # target = activations['target']
            out_cond = self.model['all'].linear(y)
            out_cond[:, cond_idx].sum().backward()

            # res_cur = activations['target'].grad[:, target_neuron].sum(axis=(-1, -2)).mean().item()
            cur_grad = activations['target'].grad[:, target_neuron].detach().cpu()
            cur_activation = activations['target'][:, target_neuron].detach().cpu()
            # print(cur_grad.shape, cur_activation.shape)
            # res_cur = cur_grad #* cur_activation
            # cur_grad_reshaped = cur_grad.reshape(cur_grad.shape[0], -1)
            res_cur = cur_grad #* cur_grad
            # print(res_cur.shape)
            try:
                res_cur = res_cur.mean(axis=(-1, -2))
            except:
                pass
            # res_cur = res_cur.sign()
            # res_cur[res_cur == -1] = 0

            # offset = .000001
            # print((res_cur.abs() <= offset).sum().item(), (res_cur.abs() > offset).sum().item())
            # res_cur[res_cur < -offset] = -1
            # res_cur[res_cur > offset] = 1
            # res_cur[(res_cur.abs() <= offset)] = 0

            # print(cur_grad[:5])
            # print(cur_activation[:5])
            # print(res_cur[0])
            res_cur = res_cur.mean().item()
            grads.append(res_cur)
            activations['target'].grad.zero_()
        grads = np.mean(grads)#torch.cat(grads)
        for hook in hooks:
            hook.remove()
        return grads#.detach().cpu().numpy()


    def calc_gradients_wrt_output_whole_network_all_tasks(self, loader, out_path, if_pretrained_imagenet=False,
                                                          layers = layers_bn_afterrelu):
        print("Warning! Assume that loader returns in i-th batch only instances of i-th class")
        # for model in self.model.values():
        #     model.zero_grad()
        #     model.eval()

        target_layer_names = [layer.replace('_', '.') for layer in layers]#layers_bn_afterrelu] #+ ['feature_extractor']
        neuron_nums = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
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
                                                   layers=layers_bn_afterrelu):
        target_layer_names = [layer.replace('_', '.') for layer in layers]
        neuron_nums = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        df_grads = pd.read_pickle(df_grads_path)
        data_corrs = []
        if if_replace_wass_dists_with_noise:
            wasser_dists_np = np.random.laplace(0, 0.05, (10, 512))
        for i, layer_name in enumerate(target_layer_names):
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
                cur_corr = np.corrcoef(wasser_dists_np[:, neuron], cur_grads.squeeze())[0, 1]
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
                                                  df_corr_paths_early=None, early_layers=None):
        df_corr = pd.read_pickle(df_corr_path)
        target_layer_names = [layer.replace('_', '.') for layer in layers]
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
        plt.figure(figsize=(8*1.2, 6*1.2))
        plt.plot(x, avg_corrs)
        plt.xticks(x)
        print(avg_corrs)
        plt.ylim(0, 1)
        plt.xlabel('Layer index')
        plt.ylabel('Avg. correlation')
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
        plt.figure(figsize=(8*1.2, 6*1.2))
        plt.plot(x, means)
        plt.fill_between(x, means-stds, means+stds, facecolor='orange')
        plt.ylim(0, 1)
        # plt.title('Average correlation of W.dists & gradients\n averaged over 5 runs\n ± 1 std')
        plt.xticks(x)
        plt.xlabel('Layer index')
        plt.ylabel('Avg. correlation')
        plt.show()


    @staticmethod
    def create_activity_collector(save_model_path, param_file, if_pretrained_imagenet=False):
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
            return ActivityCollector(trained_model, im_size), params, configs
        else:
            im_size = (224, 224)
            trained_model = torchvision.models.__dict__['resnet18'](pretrained=True).to(device)
            # trained_model = torchvision.models.__dict__['resnet34'](pretrained=True).to(device)
            return ActivityCollector(trained_model, im_size, use_my_model=False), None, None



    def eval_with_neuron_replaced(self, params, loader,
                                  target_layer, neuron_replaced, neuron_replacing):
        tasks = params['tasks']
        all_tasks = configs[params['dataset']]['all_tasks']
        target_layer_name = layers[target_layer].replace('_', '.')

        def replace_activation(name, mod, inp, out):
            if name == target_layer_name:
                out[:, neuron_replaced, :, :] = out[:, neuron_replacing, :, :]
                return out

        hooks = []
        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(replace_activation, name)))
        y_pred = []
        y_true = []

        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                val_images = batch_val[0].cuda()
                labels_val = get_relevant_labels_from_batch(batch_val, all_tasks, tasks, params, device)
                val_reps = self.model['rep'](val_images)
                val_rep = val_reps[0]
                #assume single-task cifar
                out_t_val, _ = self.model['all'](val_rep, None)
                y_pred_cur = out_t_val.data.max(1, keepdim=True)[1]
                y_pred_cur = list(y_pred_cur.detach().cpu().numpy().squeeze())
                y_pred += y_pred_cur
                y_true += list(labels_val['all'].detach().cpu().numpy())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        print(balanced_accuracy_score(y_true, y_pred))
        conf_matr = confusion_matrix(y_true, y_pred)

        for hook in hooks:
            hook.remove()
        return conf_matr

    @staticmethod
    def calc_precision_recall_from_confusion_matr(conf_matr):
        return conf_matr.diagonal() / conf_matr.sum(axis=1), conf_matr.diagonal() / conf_matr.sum(axis=0)

    def plot_hists_of_spatial_activations(self, path='activations_on_validation_preserved_spatial__bettercifar10single.npy',
                                         out_path='hist_spatial_bettercifar10single', layer_list=layers_bn_afterrelu,
                                         chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]):
        Path(out_path).mkdir(exist_ok=True)
        saved_layers, acts = np.load(path, allow_pickle=True)
        target_layer_names = [layer.replace('_', '.') for layer in layer_list]
        assert saved_layers == target_layer_names

        for layer_idx, acts_for_layer in enumerate(acts.values()):
            acts_for_layer_stacked = np.vstack(acts_for_layer)
            for neuron_idx in range(chunks[layer_idx]):
                print(neuron_idx)
                proper_hist(acts_for_layer_stacked[:, neuron_idx], title=f'{layer_idx}_{neuron_idx}', if_determine_bin_size=True)
                plt.savefig(f'{out_path}/{layer_idx}_{neuron_idx}.png', format='png', bbox_inches='tight', pad_inches=0)
                plt.close()

    def plot_hists_of_spatial_activations_no_save(self, loader, target_layer_indices,
                                         out_path='hist_spatial_bettercifar10single', layer_list=layers_bn_afterrelu,
                                         chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        Path(out_path).mkdir(exist_ok=True)
        zero_ratios_data = []

        for layer_idx, layer_name in enumerate(target_layer_names):
            activation_dict = self.store_layer_activations_many(loader, [layer_idx], if_average_spatial=False, if_store_activations=False,
                                              layer_list=layer_list)
            activations = np.vstack(activation_dict[layer_name])
            n_total_feature_map_values = len(activations[:, 0].flatten())
            for neuron_idx in range(chunks[layer_idx]):
                print(neuron_idx)
                n_zeros = np.sum(activations[:, neuron_idx].flatten() == 0)
                ratio_zeros = n_zeros / n_total_feature_map_values
                proper_hist(activations[:, neuron_idx], title=f'{layer_idx}_{neuron_idx}\n{ratio_zeros :.2f}',
                            if_determine_bin_size=True)
                plt.savefig(f'{out_path}/{layer_idx}_{neuron_idx}.png', format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                zero_ratios_data.append([layer_name, neuron_idx, ratio_zeros])
            del activation_dict
            del activations

        df_zeros = pd.DataFrame(zero_ratios_data, columns=['layer_name', 'neuron_idx', 'ratio_zeros'])
        df_zeros.to_pickle(out_path + '_ratio_zeros_df.pkl')


    def store_highest_activating_patches(self, loader, target_layer_indices,
                                         out_path='patches', layer_list=layers_bn_afterrelu,
                                         chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512], n_save=64,
                                         base_path='/mnt/raid/data/chebykin/imagenet_val/my_imgs', image_size = 32):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        Path(out_path).mkdir(exist_ok=True)
        n_save_half = n_save // 2

        def indices_to_patch_list(idx, base_path, patch_size, image_to_feature_ratio, anchors):
            ims = []
            for i, im_idx in enumerate(idx):
                path = base_path + f'/{im_idx}.jpg'
                im = Image.open(path)
                im2arr = np.array(im) # h, w, c
                # from bgr to rgb :
                im2arr = im2arr[:, :, ::-1]

                x, y = anchors[i]
                x *= image_to_feature_ratio
                y *= image_to_feature_ratio

                patch_size_half = patch_size // 2

                patch = im2arr[max(int(np.floor(x - patch_size_half)), 0): min(int(np.ceil(x+patch_size_half)), image_size),
                        max(int(np.floor(y - patch_size_half)), 0): min(int(np.ceil(y+patch_size_half)), image_size)]

                padded = np.zeros((patch_size, patch_size, 3))
                padded_rgba = np.dstack((padded, np.zeros((patch_size, patch_size), dtype=np.uint8) + 255))
                padded_rgba[:patch.shape[0], :patch.shape[1], :3] = patch
                padded_rgba[patch.shape[0]:, :, 3] = 0
                padded_rgba[:, patch.shape[1]:, 3] = 0
                arr2im = Image.fromarray(np.uint8(padded_rgba))
                ims.append(arr2im)
            return ims

        def vstack_fast(list_of_np_arrays):
            total_samples = np.sum([a.shape[0] for a in list_of_np_arrays])
            res = np.zeros((total_samples, list_of_np_arrays[0].shape[1], list_of_np_arrays[0].shape[2], list_of_np_arrays[0].shape[3]))
            cur = 0
            for ar in list_of_np_arrays:
                for i in range(ar.shape[0]):
                    res[cur] = ar[i]
                    cur += 1
            return res

        for layer_idx, layer_name in zip(target_layer_indices, target_layer_names):
            patch_size = 100
            activation_dict = self.store_layer_activations_many(loader, [layer_idx], if_average_spatial=False, if_store_activations=False,
                                              layer_list=layer_list)
            activations = vstack_fast(activation_dict[layer_name])
            image_to_feature_ratio = image_size / activations.shape[2]
            for neuron_idx in range(chunks[layer_idx]):
                print(neuron_idx)
                cur_activations = activations[:, neuron_idx]

                # cur_activations[cur_activations == 0] = 0.5

                print(cur_activations.shape)

                if True:
                    cur_activations_mins = cur_activations.min(axis=(-1, -2))
                    # cur_activations_mins = cur_activations.mean(axis=(-1, -2))
                    idx_min = np.argpartition(cur_activations_mins, n_save_half)[:n_save_half]
                    print(cur_activations_mins[idx_min])
                    anchors_min = [np.unravel_index(cur_activations[i].argmin(), cur_activations[i].shape) for i in idx_min]
                else:
                    idx_min = np.argpartition(cur_activations[:, 2, 2], n_save_half)[:n_save_half]

                    anchors_min = [(2, 2) for _ in idx_min]
                # print([cur_activations[idx_min[i], am[0], am[1]] for i, am in enumerate(anchors_min)])

                if True:
                    cur_activations_maxs = cur_activations.max(axis=(-1, -2))
                    # cur_activations_maxs = cur_activations.mean(axis=(-1, -2))
                    idx_max = np.argpartition(cur_activations_maxs, -n_save_half)[-n_save_half:]
                    print(cur_activations_maxs[idx_max])
                    anchors_max = [np.unravel_index(cur_activations[i].argmax(), cur_activations[i].shape) for i in idx_max]
                else:
                    idx_max = np.argpartition(cur_activations[:, 2, 2], -n_save_half)[-n_save_half:]
                    anchors_max = [(2, 2) for _ in idx_max]

                ims_min = indices_to_patch_list(idx_min, base_path, patch_size, image_to_feature_ratio, anchors_min)
                ims_max = indices_to_patch_list(idx_max, base_path, patch_size, image_to_feature_ratio, anchors_max)

                # print([cur_activations[idx_max[i], am[0], am[1]] for i, am in enumerate(anchors_max)])

                new_im = images_list_to_grid_image(ims_max + ims_min, if_rgba=True, if_draw_line=True)
                new_im.save(out_path + '/' + f'{layer_idx}_{neuron_idx}.png')
                # break

            del activation_dict
            del activations

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
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_51_on_May_21/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_15_model.pkl'
    param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc.json'
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
    #   single-head noskip cifar
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_31_on_October_01/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18_noskip|width_mul=1|weight_deca_240_model.pkl'
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


    if_pretrained_imagenet = False
    ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file, if_pretrained_imagenet)

    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
        with open(param_file) as json_params:
            params = json.load(json_params)
        with open('configs.json') as config_params:
            configs = json.load(config_params)
        # params['dataset'] = 'imagenet_val'
        params['dataset'] = 'imagenet_test'
        params['batch_size'] = 256#320
    print(model_name_short)
    # ac.compare_AM_images(31, 12)
    # ac.visualize_feature_distribution(range(40), 10, 'binmatr')
    # ac.visualize_feature_histograms_per_task(range(40), 10, 'binmatr')
    # ac.visualize_probs_distribution(range(40), 10, model_name_short)
    # for i in range(4):
    # print(ac.get_output_probs(f'big_diverse_generated_separate_12_18_on_June_24...46/label/33_{i}.jpg')[23])
    # print(celeba_dict[i], ac.get_output_probs_single_image(f'big_diverse_generated_separate_12_18_on_June_24...46/label/{i}_0.jpg')[9])
    # print(ac.get_output_probs_single_image(f'big_diverse_generated_separate_12_18_on_June_24...46/label/21_{i}.jpg')[9])
    # print(celeba_dict[i], ac.get_output_probs_single_image(f'generated_archive/generated_separate_12_18_on_June_24...46/label/{i}_0.jpg')[11])
    # print(ac.get_output_probs(f'generated_separate/23_{i}.jpg')[23])

    # ac.store_layer_activations_many(configs, range(1, 5), False)
    # ac.store_layer_activations_many(configs, range(5, 14), False)
    # if True:
    #     _, loader, _ = datasets.get_dataset(params, configs)
    # else:
    #     img_paths = glob.glob('/mnt/raid/data/chebykin/pycharm_project_AA/generated_separate_reps/*')
    #     dataset = CustomDataset(img_paths, [-17] * len(img_paths),
    #                             lambda img: ac.img_from_np_to_torch(np.asarray(img))[0], lambda x: -17)
    #     loader = data.DataLoader(dataset, batch_size=100, num_workers=1, shuffle=False, drop_last=False)
    # ac.store_layer_activations_many(loader, range(15), False, out_path_postfix='', if_save_separately=False)
    # ac.cluster_stored_layer_activations([14], False)
    # ac.cluster_stored_layer_activations(list(range(15)), False)
    # store_path_to_label_dict(loader, 'path_to_label_dict_celeba_val.npy')

    # ac.store_highest_activating_images()
    plt.rcParams.update({'font.size': 16})
    # params['batch_size'] = 50/2#50/4#1000/4#26707#11246#1251#10001#
    # params['dataset'] = 'broden_val'
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
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single_nonsparse_afterrelu_early_and_last.pkl', {0:'all', 1:'all'},
    #                                                'attr_hist_bettercifar10single_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='cifar',
    #                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=early_layers_and_last_bn_afterrelu)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single2_nonsparse_afterrelu_early_and_last.pkl', {0:'all', 1:'all'},
    #                                                'attr_hist_bettercifar10single2_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='cifar',
    #                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single2_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=early_layers_and_last_bn_afterrelu)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single3_nonsparse_afterrelu_early_and_last.pkl', {0:'all', 1:'all'},
    #                                                'attr_hist_bettercifar10single3_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='cifar',
    #                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single3_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=early_layers_and_last_bn_afterrelu)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single4_nonsparse_afterrelu_early_and_last.pkl', {0:'all', 1:'all'},
    #                                                'attr_hist_bettercifar10single4_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='cifar',
    #                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single4_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=early_layers_and_last_bn_afterrelu)
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bettercifar10single5_nonsparse_afterrelu_early_and_last.pkl', {0:'all', 1:'all'},
    #                                                'attr_hist_bettercifar10single5_early_and_last', if_cond_label=False,
    #                                                used_neurons='resnet_early',
    #                                                dataset_type='cifar',
    #                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bettercifar10single5_early_and_last.npy',
    #                                                if_calc_wasserstein=True, offset='argmax', if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=early_layers_and_last_bn_afterrelu)
    ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'celeba_fc_v3.pkl', {13:'all'},
                                                   'attr_hist_celeba_fc_v3', if_cond_label=False,
                                                   used_neurons='resnet_full',
                                                   dataset_type='celeba',
                                                    sorted_dict_path='img_paths_most_activating_sorted_dict_paths_celeba_fc_v3.npy',
                                                   if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False,
                                                   if_force_recalculate=True, if_left_lim_zero=True,
                                                    layer_list=layers_bn_afterrelu)
    # chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    # ac.wasserstein_barplot('attr_hist_pretrained_imagenet_prerelu', dict(zip(range(15), [range(c) for c in chunks])),
    #                        imagenet_dict, n_show=20, out_path_suffix='')
    # ac.wasserstein_barplot('attr_hist_pretrained_imagenet_broden3_afterrelu', dict(zip(range(15), [range(c) for c in chunks])),
    #                        broden_categories_list, n_show=20, out_path_suffix='')

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
    exit()
    # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, f'grads_test_early_{model_name_short}.pkl',
    #                                                      if_pretrained_imagenet, early_layers_bn_afterrelu)
    # # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_pretrained_imagenet_afterrelu_test.pkl', if_pretrained_imagenet)
    # # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_pretrained_imagenet_afterrelu_test_early.pkl',
    # #                                                      if_pretrained_imagenet, layers=early_layers_bn_afterrelu)
    # if True:
    #     ac.correlate_grads_with_wass_dists_per_neuron(f'grads_test_early_{model_name_short}.pkl',
    #                                                   f'corr_grads_test_early_{model_name_short}.pkl',
    #                                        'wasser_dists/wasser_dist_attr_hist_bettercifar10single5_early_and_last',
    #                                                   if_replace_wass_dists_with_noise=False, layers=early_layers_bn_afterrelu)
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
    # exit()
    # ac.plot_correlation_of_grads_with_wass_dists(f'corr_grads_test_{model_name_short}.pkl')
    # ac.plot_correlation_of_grads_with_wass_dists(f'corr_grads_imagenet_test.pkl',
    #                                              layers_bn_afterrelu,
    #                                              'corr_grads_imagenet_test_early.pkl',
    #                                              early_layers_bn_afterrelu)
    # ac.plot_correlation_of_grads_with_wass_dists('corr_grads_imagenet_test_early.pkl', layers=early_layers_bn_afterrelu)

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
    ac.plot_correlation_of_grads_with_wass_dists_many_runs(
        [f'corr_grads_test_{save_model_path[37:53] + "..." + save_model_path[-12:-10]}.pkl' for save_model_path in model_paths],
        layers_bn_afterrelu,
        [f'corr_grads_test_early_{save_model_path[37:53] + "..." + save_model_path[-12:-10]}.pkl' for save_model_path in model_paths],
        early_layers_bn_afterrelu
    )
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
    ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'lipstickonly_nonsparse_afterrelu_v2.pkl', {14:'all'},
                                                   'attr_hist_lipstickonly_v2', if_cond_label=True,
                                                   used_neurons='resnet_quarter',
                                                   dataset_type='celeba',
                                                   sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_lipstickonly_v2.npy',
                                                   if_calc_wasserstein=True, offset='argmax', if_show=False, if_force_recalculate=True,
                                                   if_left_lim_zero=True, layer_list=layers_bn_afterrelu
                                                   )
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

    # ims = ac.get_images_open_mouth_nonblond(configs)

    experiment_name = 'black_hair_closed_mouth'
    # experiment_name = 'black_hair_open_mouth'
    # experiment_name = 'black_hair_open_mouth_female'
    # experiment_name = 'black_hair_nonsmiling'
    # experiment_name = 'black_hair_smiling'

    # experiment_name = 'blond_hair_closed_mouth'
    # experiment_name = 'blond_hair_open_mouth'
    # experiment_name = 'blond_hair_nonsmiling'
    # experiment_name = 'blond_hair_smiling'

    # experiment_name = 'brown_hair_nonsmiling'
    # experiment_name = 'brown_hair_smiling'

    # experiment_name = 'bald_smile_opened_mouth'

    # ac.store_images_based_on_label(lambda label: (label[task_ind_from_task_name('brownhair')] == 1) and
    #                                              (label[task_ind_from_task_name('smiling')] == 1),
    #                                1700, f'{experiment_name}.npy')

    img_paths, labels = np.load(f'{experiment_name}.npy', allow_pickle=True)
    img_paths = img_paths[:1450]
    labels = labels[:1450]
    # img_paths = np.array(sorted(glob.glob(f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1_reversed/*')))
    # img_paths = np.array([f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan6/{i}.jpg' for i in range(1450)])
    # dataset = CustomDataset(img_paths, [-17] * len(img_paths), lambda img: ac.img_from_np_to_torch(np.asarray(img))[0],lambda x: -17)
    # loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
    # for idx, (img, lbl) in enumerate(loader):
    #     torchvision.utils.save_image(img[0].flip(dims=[0]), os.path.join(experiment_name, f'{idx}.jpg'))#, normalize=True, range=(-1., 1.))
    #     # Image.fromarray(img[0].permute(1, 2, 0).numpy(), "RGB").save(os.path.join(experiment_name, f'{idx}.jpg'))

    # gan_path = 'gans/384_shortcut1_inject0_none_hq/checkpoint/weights.149.pth'
    # postfix = '_attgan'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/pretrained_relgan/generator519.h5'
    # postfix = '_relgan0'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model/generator1400.h5' #this is the one that works
    # postfix = '_relgan1' + '_makehairblacker'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model2/generator600.h5'
    # postfix = '_relgan2'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_40attr/generator186.h5'
    # postfix = '_relgan3A'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_40attr_cont/generator7.h5'
    # postfix = '_relgan3'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_18attr_cont/generator145.h5'
    # postfix = '_relgan5' + '_openmouthandsmile'
    gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_18attr2/generator568.h5' #543, 522
    postfix = '_relgan6' + '_reversed' # + '_donothing'

    if False:
        # attribute_changer = AttributeChanger('AttGAN', gan_path, 'gans/384_shortcut1_inject0_none_hq/setting.txt',
        #                                      'close_mouth', experiment_name)
        # attribute_changer.generate(img_paths, labels)
        attribute_changer = AttributeChanger('RelGAN', gan_path,
                                             out_path=experiment_name + postfix)
        # '/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1_reversed2')
        attribute_changer.generate(img_paths, labels, kwargs={
            'Mouth_Slightly_Open':1})  # need to pass labels just to peacify CustomDataset; don't really need them
        # attribute_changer.generate(img_paths, labels, kwargs={'Black_Hair':-1, 'Blond_Hair':1, 'Eyeglasses':1}) # need to pass labels just to peacify CustomDataset; don't really need them
        exit() # can't properly free memory, and Torch crashes in the next lines => can just as well exit myself

    if True:
        n_pics_total = len(img_paths)
        n_pics_to_use = n_pics_total  #min(n_pics_total, 1450)#
        # idx = np.random.choice(n_pics_total, size=n_pics_to_use, replace=False)
        probs_before = np.array(ac.get_output_probs_many_images(img_paths, True))
        pb = probs_before.mean(axis=0)
        print(pb[8])
        # probs_after = np.array(
        #     ac.get_output_probs_many_images([f'./{experiment_name}{postfix}/{i}.jpg' for i in range(n_pics_to_use)],
        #                                     True))
        probs_after = np.array(ac.get_output_probs_many_images(np.array(sorted(glob.glob(f'./{experiment_name}{postfix}/*'))), True))
        # probs_after = np.array(ac.get_output_probs_many_images(np.array([
        #     f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan6_reversed/{i}.jpg' for i in range(n_pics_to_use)]), True))
        # print(probs_before.std(axis=0))
        # print(probs_after.std(axis=0))
        pa = probs_after.mean(axis=0)
        print(pa[8])
        exit()
        print((probs_after - probs_before).mean(axis=0)[11])

        width = 0.35
        fig, ax = plt.subplots()
        labels = list(celeba_dict.values())
        x = np.arange(len(labels))
        rects1 = ax.bar(x - width / 2, pb, width, label='before')
        rects2 = ax.bar(x + width / 2, pa, width, label='after')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical')
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()

        diff = (probs_after - probs_before).mean(axis=0)
        fig, ax = plt.subplots()
        labels = list(celeba_dict.values())
        x = np.arange(len(labels))
        rects1 = ax.bar(x, diff, label='diff')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical')
        ax.set_ylim([-1, 1])
        ax.legend()
        plt.show()

        if True:
            _, axs = plt.subplots(5, 8, figsize=(20, 12))
            for i in range(40):
                data = probs_after[:, i] - probs_before[:, i]
                proper_hist(data, celeba_dict[i], axs.flatten()[i], xlim_left=-1, xlim_right=1)
            plt.show()

        # black_hair_diff = (probs_after[:, 8] - probs_before[:, 8])
        # if False:
        #     np.save('black_hair_diff.npy', black_hair_diff)
        # else:
        #     black_hair_diff_nodisabledconns = np.load('black_hair_diff.npy')
        #     idx = np.argsort(black_hair_diff - black_hair_diff_nodisabledconns)
        #     plt.figure(figsize=(15, 5))
        #     # plt.scatter(range(len(black_hair_diff_nodisabledconns)), black_hair_diff_nodisabledconns[idx])
        #     # plt.scatter(range(len(black_hair_diff)), black_hair_diff[idx])
        #     plt.scatter(range(len(black_hair_diff_nodisabledconns)), black_hair_diff[idx] - black_hair_diff_nodisabledconns[idx])
        #     # plt.ylim(-1, 1)
        #     plt.show()

        # np.save(f'probs_beforegan_cnsdsbld_{model_name_short}', pb)
        # np.save(f'probs_aftergan_cnsdsbld_{model_name_short}', pa)
    else:
        pbs = []
        pas = []
        diffs = []
        models = ['12_18_on_June_24...46', '12_25_on_April_3...23', '23_06_on_April_2...27']
        for cur_name in models:
            pb = np.load(f'probs_beforegan_{cur_name}.npy')
            pbs.append(pb)
            pa = np.load(f'probs_aftergan_{cur_name}.npy')
            pas.append(pa)
            diffs.append(pa - pb)

        n_models = len(models)
        width = 1 / n_models - 0.05
        fig, ax = plt.subplots(figsize=(6, 10))
        labels = list(celeba_dict.values())
        x = np.arange(len(labels))
        evenly_spaced_nums = np.arange(-1.5, 1.5 + 0.0001, 3 / (n_models - 1))
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = ["C3", "C2", "C4"]
        for i, diff in enumerate(diffs):
            ax.barh(x + evenly_spaced_nums[i] * width / 2, diff, width, label=models[i],
                    color=colors[i])  # cycle[(i+1)*2])
        ax.set_yticks(x)
        ax.set_yticklabels(labels)  # , rotation='vertical')
        plt.legend()
        plt.show()
'''
interesting_indices = np.hstack((np.argpartition(data, -2)[-2:], np.argpartition(data, 2)[:2]))
for i in interesting_indices:
    path = img_paths[i]
    copy(path, '/mnt/antares_raid/home/awesomelemon/')
    copy(f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1/{i}.jpg', '/mnt/antares_raid/home/awesomelemon/')  
'''
'''
l_i = 14
neuron_i = 494
wass_dists = []
for cond_i in range(40):
# cond_i = task_ind_from_task_name('smiling')
    selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[l_i]) &
         (df['neuron_idx'] == neuron_i)].drop(['layer_name', 'neuron_idx'], axis=1))
    cond_mask1 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
            (df_cond['neuron_idx'] == cond_i)].drop(['layer_name', 'neuron_idx'], axis=1) <= 0
    selected_values_list1 = selected_values_list[cond_mask1]
    cond_mask2 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
            (df_cond['neuron_idx'] == cond_i)].drop(['layer_name', 'neuron_idx'], axis=1) > 0
    selected_values_list2 = selected_values_list[cond_mask2]
    wass_dists.append((celeba_dict[cond_i], scipy.stats.wasserstein_distance(selected_values_list1, selected_values_list2)))
print(wass_dists)

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
xs, ys = ecdf(selected_values_list1)
xs, ys = ecdf(selected_values_list2)
plt.plot(xs, ys)
'''
'''
np.arange(512)[np.sum(wd_np_sign[np.arange(10) != 4] == np.sign(wd_np[np.arange(10) != 4, :][:, 507])[:, None], axis=0) == 9]
'''