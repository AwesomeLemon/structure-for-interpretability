import operator

import io

import json
import os
import pickle
from collections import defaultdict

import random
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

try:
    from multi_task import datasets
    from multi_task.gan.attgan.data import CustomDataset
    from multi_task.gan.change_attributes import AttributeChanger
    from multi_task.load_model import load_trained_model
    from multi_task.loaders.celeba_loader import CELEBA
except:
    import datasets
    from gan.attgan.data import CustomDataset
    # from gan.change_attributes import AttributeChanger
    from load_model import load_trained_model
    from loaders.celeba_loader import CELEBA
import glob
from shutil import copyfile, copy
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivityCollector():
    def __init__(self, model, im_size):
        self.model = model
        self.use_my_model = True
        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        for m in self.model:
            model[m].eval()

        self.feature_extractor = self.model['rep']
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
                                     if_save_separately=True):
        target_layer_names = [layers[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if name in target_layer_names:
                if if_average_spatial:
                    cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
                else:
                    cur_activations = out.detach().cpu().numpy()
                if name in activations:
                    cur_activations = np.append(activations[name], cur_activations, axis=0)
                activations[name] = cur_activations

        hooks = []
        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                if i % 10 == 0:
                    print(i)

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)

        for hook in hooks:
            hook.remove()

        if if_save_separately:
            for i, idx in enumerate(target_layer_indices):
                if if_average_spatial:
                    filename = f'activations_on_validation_{idx}{out_path_postfix}.npy'
                else:
                    filename = f'activations_on_validation_preserved_spatial_{idx}{out_path_postfix}.npy'
                with open(filename, 'wb') as f:
                    pickle.dump(activations[target_layer_names[i]], f, protocol=4)
        else:
            filename = 'representatives_kmeans14_50_alllayers.npy'
            np.save(filename, activations)

        return activations

    def find_highest_activating_images(self, loader, if_average_spatial=True, if_nonceleba=False, target_layer_indices=None,
                                       save_path='img_paths_most_activating_sorted_dict_afterrelu_fcmodel.npy', used_neurons=None,
                                       if_sort_by_path=False):
        target_layer_names = np.array([layer.replace('_', '.') for layer in layers_bn] + ['label'])
        if_find_for_label = False
        if target_layer_indices is None:
            target_layer_indices = list(range(len(target_layer_names)))
        target_layer_names = target_layer_names[target_layer_indices]
        target_layer_names = list(target_layer_names)
        if 'label' in target_layer_names:
            target_layer_names.remove('label')
            if_find_for_label = True

        if used_neurons is None:
            used_neurons_loaded = np.load('actually_good_nodes.npy', allow_pickle=True).item()
            used_neurons = {}
            for layer_idx, neurons in used_neurons_loaded.items():
                used_neurons[layer_idx] = np.array([int(x[x.find('_') + 1:]) for x in neurons])

        if if_nonceleba:
            cnt = 0

        def save_activation(activations, name, mod, inp, out):
            if name in target_layer_names:
                out = out.detach()
                if 'bn1' in name:
                    out = F.relu(out)
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
                if not if_nonceleba:
                    im_paths = batch_val[-1]
                else:
                    root_path = '/mnt/raid/data/chebykin/cifar10/my_imgs'
                    im_paths = []
                outs = self.feature_extractor(val_images)
                if if_nonceleba:
                    recreated_images = recreate_image_cifarfashionmnist_batch(val_images)
                for layer_idx, layer in zip(target_layer_indices, target_layer_names):
                    acts = activations[layer]  # shape (batch_size, n_neurons)
                    used_neurons_cur = used_neurons[layer_idx]
                    for j in range(acts.shape[0]):
                        if not if_nonceleba:
                            cur_path = im_paths[j]
                        else:
                            if layer_idx == 0:
                                cur_path = root_path + f'/{cnt}.jpg'
                                cnt += 1
                                cur_img = recreated_images[j]
                                save_image(cur_img, cur_path)
                                im_paths.append(cur_path)
                            else:
                                cur_path = im_paths[j]
                        for k in used_neurons_cur:
                            if not if_sort_by_path:
                                sorted_dict_per_layer_per_neuron[layer][k][acts[j][k]] = cur_path
                            else:
                                sorted_dict_per_layer_per_neuron[layer][k][cur_path] = acts[j][k]

                activations.clear()

                if if_find_for_label:
                    task_cnt = 0
                    for key in self.model.keys():
                        if key == 'rep': #only interested in tasks
                            continue
                        out_t, _ = self.model[key](outs[task_cnt], None)
                        task_cnt += 1
                        out_t = torch.exp(out_t)
                        diff = out_t[:, 1] - out_t[:, 0]
                        for j in range(out_t.shape[0]):
                            if not if_sort_by_path:
                                sorted_dict_per_layer_per_neuron['label'][int(key)][diff[j]] = im_paths[j]
                            else:
                                sorted_dict_per_layer_per_neuron['label'][int(key)][im_paths[j]] = diff[j].item()


                print(len(im_paths))

        for hook in hooks:
            hook.remove()

        np.save(save_path, dict(sorted_dict_per_layer_per_neuron))
        return sorted_dict_per_layer_per_neuron

    def store_highest_activating_images(self):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn] + ['label']
        layers_local_var = layers_bn + ['label']
        sorted_dict_per_layer_per_neuron = defaultdict(lambda: dict(),
                                                       np.load(
                                                           'img_paths_most_activating_sorted_dict_afterrelu_cifarfshmnst.npy',
                                                           allow_pickle=True).item())
        n_save = 16
        folder_out = 'highest_activating_ims_bn_hist_cifarfshmnst'
        Path(folder_out).mkdir(exist_ok=True)
        # fontPath = "/usr/share/fonts/dejavu-lgc/DejaVuLGCSansCondensed-Bold.ttf"
        fontPath = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
        font = ImageFont.truetype(fontPath, 28)
        for i in range(len(layers_local_var)):
            # if layers_local_var[i] != 'label':
            #      continue
            cur_folder = folder_out + '/' + layers_local_var[i]
            Path(cur_folder).mkdir(exist_ok=True)
            for neuron, sorted_dict in sorted_dict_per_layer_per_neuron[target_layer_names[i]].items():
                sorted_dict_list = list(sorted_dict.items())
                values_and_paths = list(reversed(sorted_dict_list[-n_save // 2:])) + sorted_dict_list[:n_save // 2 - 1]
                # values_and_paths = sorted_dict_list[:n_save]
                ims = []
                for (value, path) in values_and_paths:
                    im = Image.open(path)
                    value_str = str(value)[:7]
                    # if (i == 12) and (neuron == 400):
                    #     print(value)
                    if layers_local_var[i] == 'label':
                        value_str = str(value.item())[:7]
                    # ImageDraw.Draw(im).text((0, 0), value_str, (255, 150, 100), font=font)
                    ims.append(im)

                plt.tight_layout()
                plt.figure(figsize=(1.6, 2.2))
                proper_hist(np.array([k.item() for k in sorted_dict.keys()]), bin_size=0.01)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                im = Image.open(buf)
                im = im.resize((ims[0].size[0], ims[0].size[1]), Image.ANTIALIAS)
                ims.append(im)

                new_im = images_list_to_grid_image(ims)
                new_im.save(cur_folder + '/' + f'{neuron}.jpg')
                buf.close()
                plt.close()



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
                                     out_dir='attr_hist_for_neuron_fc_all', used_neurons=None, if_nonceleba=False):
        target_layer_names = [layer.replace('_', '.') for layer in layers_bn] + ['label']
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

        df_cond = df
        if if_cond_labels:
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

        selected_paths1_cache = {}
        selected_paths2_cache = {}
        for target_neuron in target_neurons:
            print(target_neuron)
            cols_rows_ratio = cols_num / rows_num
            _, axs = plt.subplots(rows_num, cols_num, squeeze=False, figsize=(9, 14))
            axs = axs.flatten()

            selected_values_list = np.array(df.loc[(df['layer_name'] == target_layer_names[target_layer_idx]) &
                                                   (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'], axis=1))

            bin_size = 0.02
            diff = selected_values_list.max() - selected_values_list.min()
            if diff == 0.0:
                print(f'Useless neuron {target_layer_idx}_{target_neuron}')
                continue
            while diff / bin_size < 15:
                bin_size /= 2
                print(f'New bin size: {bin_size}')
            while diff / bin_size > 80:
                bin_size *= 2
                print(f'New bin size: {bin_size}')
            xlim_right = selected_values_list.max() + 0.01

            for cond_i, idx in enumerate(cond_neurons):
                title = str(idx)
                if cond_layer_idx == 15:
                    if not if_nonceleba:
                        title = celeba_dict[idx]
                    else:
                        # title = cifarfashion_dict[idx]
                        title = cifar10_dict[idx]

                if idx not in selected_paths1_cache:
                    cond_mask1 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                                        (df_cond['neuron_idx'] == idx)].drop(['layer_name', 'neuron_idx'], axis=1) <= 0
                    selected_paths1_cache[idx] = np.array(cond_mask1)
                cond_mask1 = selected_paths1_cache[idx]
                selected_values_list1 = selected_values_list[cond_mask1]
                if len(selected_values_list1) == 0:
                    print('continue1', idx)
                    continue

                proper_hist(selected_values_list1, bin_size=bin_size, ax=axs[cond_i], xlim_left=0, xlim_right=xlim_right,
                            density=True)

                if idx not in selected_paths2_cache:
                    cond_mask2 = df_cond.loc[(df_cond['layer_name'] == target_layer_names[cond_layer_idx]) &
                                        (df_cond['neuron_idx'] == idx)].drop(['layer_name', 'neuron_idx'], axis=1) > 0
                    selected_paths2_cache[idx] = np.array(cond_mask2)
                cond_mask2 = selected_paths2_cache[idx]
                selected_values_list2 = selected_values_list[cond_mask2]
                if len(selected_values_list2) == 0:
                    print('continue2', idx)
                    continue
                proper_hist(selected_values_list2, bin_size=bin_size, ax=axs[cond_i], xlim_left=0, xlim_right=xlim_right,
                            alpha=0.75, title=title, density=True)

            plot_name = f'{target_layer_idx}_{target_neuron}'
            plt.suptitle(plot_name)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/hist_{plot_name}.png', format='png', bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

    def compute_attr_hist_for_neuron_pandas_wrapper(self, loader, df_val_images_activations_path, target_dict,
                                                    out_dir_path, if_cond_label=False,
                                                    sorted_dict_path=None,
                                                    used_neurons=None, if_nonceleba=False):
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
            if not os.path.exists(used_neurons):
                if used_neurons == 'resnet_full':
                    chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
                elif used_neurons == 'resnet_quarter':
                    chunks = [16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128]
                used_neurons = {}
                for idx, ch in enumerate(chunks):
                    used_neurons[idx] = np.array(range(chunks[idx]))
            else:
                used_neurons_loaded = np.load(used_neurons, allow_pickle=True).item()
                used_neurons = {}
                for layer_idx, neurons in used_neurons_loaded.items():
                    used_neurons[layer_idx] = np.array([int(x[x.find('_') + 1:]) for x in neurons])

        if not os.path.exists(df_val_images_activations_path):
            if not os.path.exists(sorted_dict_path):
                sorted_dict_per_layer_per_neuron = ac.find_highest_activating_images(loader, if_nonceleba=if_nonceleba,
                                                                                     save_path=sorted_dict_path,
                                                                                     used_neurons=used_neurons,
                                                                                     if_sort_by_path=True)
            else:
                sorted_dict_per_layer_per_neuron = np.load(sorted_dict_path, allow_pickle=True).item()
            df = ac.convert_sorted_dict_per_layer_per_neuron_to_dataframe(sorted_dict_per_layer_per_neuron,
                                                                          out_path=df_val_images_activations_path)
        else:
            df = pd.read_pickle(os.path.abspath(df_val_images_activations_path))

        if not if_nonceleba:
            cond_neurons = list(range(40))
        else:
            # cond_neurons = list(range(20))
            cond_neurons = list(range(10))

        for layer_idx, targets in target_dict.items():
            ac.compute_attr_hist_for_neuron_pandas(layer_idx, targets, 15, cond_neurons, if_cond_label, df,
                                                   out_dir_path, used_neurons, if_nonceleba)

    def convert_sorted_dict_per_layer_per_neuron_to_dataframe(self, sorted_dict_per_layer_per_neuron, out_path=None):
        data = []
        for layer_name, layer_dict in sorted_dict_per_layer_per_neuron.items():
            for neuron_idx, neuron_dict in layer_dict.items():
                nditems = list(neuron_dict.items())
                nditems.sort(key=operator.itemgetter(0))
                data.append([layer_name, neuron_idx] + list(list(zip(*nditems))[1]))
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
        batch_size = 1450  # 250
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
                ttt = 1
                # print(out[8].detach().cpu().mean(axis=0)[[172, 400, 383]])
                # print(out[9].detach().cpu().mean(axis=0)[[356, 204, 126, 187, 123, 134, 400, 383]])
                print(out[11].detach().cpu().mean(axis=0)[[164, 400, 383]])
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


    @staticmethod
    def create_activity_collector(save_model_path, param_file):
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
        if False:
            print('Some connections were disabled')
            # removed_conns[5] = [(5, 23), (5, 25), (5, 58)]
            # removed_conns[8] = [(137, 143)]
            # removed_conns[9] = [(142, 188), (216, 188)]
            # removed_conns[10] = [(188, 104),(86, 104)]
            # removed_conns['label'] = [(481, 3),(481, 7)]
            # removed_conns[9] = [(216, 181)] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # removed_conns[10] = [(181, 279)]
            # removed_conns[11] = [(28, 421)] + [(331, 392)] #didn't properly check (331, 392)
            removed_conns[12] = [(406, 204)]
            # removed_conns['label'] = [  # (356, 9),
            #     # (204, 9),
            #     (126, 9),
            #     (187, 9),
            #     (123, 9),
            #     # (134, 9),
            #     #  (400, 9),
            #     #  (383, 9)
            # ]
        if False:
            save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_avgadditives' + '.pkl'
            trained_model = load_trained_model(param_file, save_model_path, if_additives_user=True,
                                               if_store_avg_activations_for_disabling=True,
                                               conns_to_remove_dict=removed_conns,
                                               replace_with_avgs_last_layer_mode='restore')
        else:
            trained_model = load_trained_model(param_file, save_model_path)
        return ActivityCollector(trained_model, im_size), params, configs


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
    # my precious:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_39_on_September_06/optimizer=SGD_Adam|batch_size=256|lr=0.005|connectivities_lr=0.001|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_120_model.pkl'
    # param_file = 'params/binmatr2_cifarfashionmnist_filterwise_sgdadam005+001_bias_condecayall2e-6.json'
    # hat only:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_56_on_September_09/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._67_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_wearinghatonly_weightedce.json'
    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]

    ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
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

    _, loader, _ = datasets.get_dataset(params, configs)
    ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'nonsparse_afterrelu.pkl', 'all_network',
                                                   'attr_hist_fc', if_cond_label=False,
                                                   used_neurons='resnet_full',
                                                   if_nonceleba=False,
                                                   sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_fc.npy')

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
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_23_on_September_10/optimizer=SGD_Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0|chunks=[16|_16|_16|_32|_32|_32|_32|_64|_64|_64|_64|_128|_128|_128|_128]|architecture=binmatr2_resnet18|width_mul=0.25|weight_decay=0._120_model.pkl'
    # param_file = 'params/binmatr2_filterwise_adam0005_fc_quarterwidth_bangsonly_weightedce.json'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file)
    # _, loader, _ = datasets.get_dataset(params, configs)
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'bangsonly_nonsparse_afterrelu.pkl', 'all_network',
    #                                                'attr_hist_bangsonly', if_cond_label=True,
    #                                                used_neurons='resnet_quarter',
    #                                                if_nonceleba=False,
    #                                                sorted_dict_path='img_paths_most_activating_sorted_dict_paths_afterrelu_bangsonly.npy')

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
    # exit()
    # for i in range(15):
    #     # ac.compute_attr_hist_for_neuron(i, 'all', 15, list(range(40)), lambda v, p, i: v < 0, False,
    #     #                             sorted_dict_per_layer_per_neuron)
    #     ac.compute_attr_hist_for_neuron_pandas(i, 'all', 15, list(range(40)), lambda v, p, i: v < 0, False, df=df,
    #                                 out_dir='attr_hist_for_neuron_fc_normed', used_neurons=used_neurons)
    # # ac.compute_attr_hist_for_neuron(14, 'all', 15, list(range(40)), lambda v, p, i: v < 0, True,
    # #                                 used_neurons=used_neurons, out_dir='attr_hist_for_neuron_hatonly')
    exit()
    # ims = ac.get_images_open_mouth_nonblond(configs)

    # experiment_name = 'black_hair_closed_mouth'
    # experiment_name = 'black_hair_open_mouth'
    # experiment_name = 'black_hair_open_mouth_female'
    # experiment_name = 'black_hair_nonsmiling'
    # experiment_name = 'black_hair_smiling'

    # experiment_name = 'blond_hair_closed_mouth'
    # experiment_name = 'blond_hair_open_mouth'
    # experiment_name = 'blond_hair_nonsmiling'
    # experiment_name = 'blond_hair_smiling'

    # experiment_name = 'brown_hair_nonsmiling'
    experiment_name = 'brown_hair_smiling'

    # experiment_name = 'bald_smile_opened_mouth'

    # ac.store_images_based_on_label(lambda label: (label[task_ind_from_task_name('brownhair')] == 1) and
    #                                              (label[task_ind_from_task_name('smiling')] == 1),
    #                                1700, f'{experiment_name}.npy')

    img_paths, labels = np.load(f'{experiment_name}.npy', allow_pickle=True)
    # img_paths = img_paths[:1450]
    # labels = labels[:1450]
    # img_paths = np.array(sorted(glob.glob(f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1_reversed/*')))
    # dataset = CustomDataset(img_paths, [-17] * len(img_paths), lambda img: ac.img_from_np_to_torch(np.asarray(img))[0],lambda x: -17)
    # loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
    # for idx, (img, lbl) in enumerate(loader):
    #     torchvision.utils.save_image(img[0].flip(dims=[0]), os.path.join(experiment_name, f'{idx}.jpg'))#, normalize=True, range=(-1., 1.))
    #     # Image.fromarray(img[0].permute(1, 2, 0).numpy(), "RGB").save(os.path.join(experiment_name, f'{idx}.jpg'))

    # gan_path = 'gans/384_shortcut1_inject0_none_hq/checkpoint/weights.149.pth'
    # postfix = '_attgan'
    gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/pretrained_relgan/generator519.h5'
    postfix = '_relgan0'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model/generator1400.h5' #this is the one that works
    # postfix = '_relgan1'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model2/generator600.h5'
    # postfix = '_relgan2'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_40attr/generator186.h5'
    # postfix = '_relgan3A'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_40attr_cont/generator7.h5'
    # postfix = '_relgan3'
    # gan_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model_18attr_cont/generator145.h5'
    # postfix = '_relgan5'

    if False:
        # attribute_changer = AttributeChanger('AttGAN', gan_path, 'gans/384_shortcut1_inject0_none_hq/setting.txt',
        #                                      'close_mouth', experiment_name)
        # attribute_changer.generate(img_paths, labels)
        attribute_changer = AttributeChanger('RelGAN', gan_path,
                                             out_path=experiment_name + postfix)  # '/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1_reversed2')
        attribute_changer.generate(img_paths, labels, kwargs={
            'Smiling': -1})  # need to pass labels just to peacify CustomDataset; don't really need them
        # attribute_changer.generate(img_paths, labels, kwargs={'Black_Hair':-1, 'Blond_Hair':1, 'Eyeglasses':1}) # need to pass labels just to peacify CustomDataset; don't really need them
        exit()

    if True:
        n_pics_total = len(img_paths)
        n_pics_to_use = n_pics_total  # min(n_pics_total, 1450)
        # idx = np.random.choice(n_pics_total, size=n_pics_to_use, replace=False)
        probs_before = np.array(ac.get_output_probs_many_images(img_paths, True))
        probs_after = np.array(
            ac.get_output_probs_many_images([f'./{experiment_name}{postfix}/{i}.jpg' for i in range(n_pics_to_use)],
                                            True))
        # probs_after = np.array(ac.get_output_probs_many_images(np.array(sorted(glob.glob(f'./{experiment_name}{postfix}/*'))), True))
        # probs_after = np.array(ac.get_output_probs_many_images(np.array(sorted(glob.glob(f'/mnt/raid/data/chebykin/pycharm_project_AA/black_hair_closed_mouth_relgan1_reversed2/*')))[idx], True))
        # print(probs_before.std(axis=0))
        # print(probs_after.std(axis=0))
        pb = probs_before.mean(axis=0)
        pa = probs_after.mean(axis=0)
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

        if False:
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
    wass_dists.append((celeba_dict[cond_i], scipy.stats.energy_distance(selected_values_list1, selected_values_list2)))
print(wass_dists)

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
xs, ys = ecdf(selected_values_list1)
xs, ys = ecdf(selected_values_list2)
plt.plot(xs, ys)
'''