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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

try:
    from multi_task import datasets
    from multi_task.gan.attgan.data import CustomDataset
    from multi_task.gan.change_attributes import AttributeChanger
    from multi_task.load_model import load_trained_model, eval_trained_model
    from multi_task.loaders.celeba_loader import CELEBA
    from multi_task.util.dicts import imagenet_dict, broden_categories_list, hypernym_idx_to_imagenet_idx, hypernym_dict
    from multi_task.model_explorer import ModelExplorer

    from multi_task.models.binmatr2_multi_faces_resnet import BasicBlockAvgAdditivesUser
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

from correlate_wasser_dists_gradients import GradWassersteinCorrelator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivityCollector(ModelExplorer):
    def __init__(self, save_model_path, param_file, model_to_use='my'):
        super().__init__(save_model_path, param_file, model_to_use)

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
                                     layer_list=layers, if_store_activations=True, if_left_lim_zero=False, out_path_prefix=Path(''), save_logits=False):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        activations = defaultdict(list)

        def save_activation(activations, name, mod, inp, out):
            if ('relu2' in name) and not if_left_lim_zero:
                out = inp[0]
            if if_average_spatial:
                cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
            else:
                cur_activations = out.detach().cpu().numpy()
            # old version:
            # if name in activations:
            #     cur_activations = np.append(activations[name], cur_activations, axis=0)
            # activations[name] = cur_activations
            # new version:
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
                feat_out = self.feature_extractor(val_images)
                if save_logits and hasattr(self, "prediction_head"):
                    logits = self.prediction_head(feat_out[0])
                    activations['logits'].append(logits.detach().cpu().numpy())
                # if i == 10:
                #     break

        for hook in hooks:
            hook.remove()

        activations = dict(activations)

        if save_logits and hasattr(self, "prediction_head"):
            target_layer_names.append("logits")

        if if_store_activations:
            print('saving activations')
            if if_save_separately:
                if if_store_layer_names:
                    raise NotImplementedError()
                for i, idx in enumerate(target_layer_indices):
                    if if_average_spatial:
                        filename = f'{out_path_prefix}activations_on_validation_{idx}{out_path_postfix}.npy'
                    else:
                        filename = f'{out_path_prefix}activations_on_validation_preserved_spatial_{idx}{out_path_postfix}.npy'
                    # with open(filename, 'wb') as f:
                    #     pickle.dump(np.vstack(activations[target_layer_names[i]]), f, protocol=4)
                    # 03.10.2021: try to spare memory, save via numpy => it worked, but I uncommented the stuff above
                    #                           to avoid replication problems
                    np.save(filename, np.vstack(activations[target_layer_names[i]]))
            else:
                if not if_store_layer_names:
                    filename = 'representatives_kmeans14_50_alllayers.npy'
                    np.save(filename, activations)
                else:
                    filename = out_path_prefix/f'activations_on_validation_averaged_spatial{out_path_postfix}.npy'
                    # filename = f'{out_path_prefix}activations_on_validation_preserved_spatial_{out_path_postfix}.npy'

                    activations['target_layer_names'] = target_layer_names

                    np.save(filename, activations, allow_pickle=True)
        if if_store_labels:
            print('saving labels')
            np.save(out_path_prefix/f'labels{out_path_postfix}.npy', labels)

        return activations

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
                plt.savefig(buf, format='png', bbox_inches='tight')
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
                plt.savefig(f'{out_dir}/{plot_name}.png', format='png', bbox_inches='tight')
                plt.savefig(f'{out_dir}/{plot_name}.svg', format='svg', bbox_inches='tight')
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
        df_labels = df_labels.drop(df_labels[df_labels['layer_name'] != 'label'].index)
        # df_labels.loc[df_labels == 0] = -1 # negative labels are "-1" in sklearn, "0" in the df_labels
        df_labels.replace(0, -1)
        df_labels.loc[(df['layer_name'] == 'label') & (df['neuron_idx'] == 0)] = 0 # restore the number for the 0-th label
        df_labels_np = np.array(df_labels)[:, 2:].astype('int')

        print(df_labels_np.shape)
        chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        res_data = []
        for i, target_layer_name in enumerate(target_layer_names):
            # if i == 0:
            #     continue
            for target_neuron in range(chunks[i]):
                print(i, target_neuron)
                target_neuron_activations = np.array(df.loc[(df['layer_name'] == target_layer_name) &
                                                       (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                                 axis=1))[0].reshape(-1, 1)
                for class_ind in range(n_classes):
                    # if class_ind % 50 == 0:
                    #     print(class_ind)
                    # labels1 = np.array(df_labels.loc[(df_labels['layer_name'] == 'label') & (df_labels['neuron_idx'] == class_ind)]
                    #                    .drop(['layer_name', 'neuron_idx'], axis=1))[0]
                    # labels1 = np.array(df_labels.loc[(df_labels['neuron_idx'] == class_ind)])[0, 2:]
                    # train_pos = target_neuron_activations[labels1 == 1]
                    # lbl_pos = np.array([1] * len(train_pos))
                    # train_neg = target_neuron_activations[labels1 != 1]
                    # lbl_neg = np.array([-1] * len(train_neg))
                    # data = np.hstack((train_pos, train_neg)).reshape(-1, 1)
                    # lbl = np.hstack((lbl_pos, lbl_neg))
                    # labels1[labels1 == 0] = -1
                    data = target_neuron_activations
                    lbl = df_labels_np[class_ind]#labels1.astype('int')

                    clf = LogisticRegression(random_state=0, penalty='l2', class_weight="balanced")
                    if 'nofolds' not in postfix: # proper cross-validation takes too long on imagenet
                        cv_results = cross_validate(clf, data, lbl, cv=5, scoring='balanced_accuracy', n_jobs=-1)
                        blncd_acc = np.mean(cv_results['test_score']) # average over folds
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(data, lbl, test_size=0.2)
                        clf = clf.fit(X_train, y_train)
                        y_test_predicted = clf.predict(X_test)
                        blncd_acc = balanced_accuracy_score(y_test, y_test_predicted)
                    res_data.append([target_layer_name, target_neuron, class_ind, blncd_acc])
                    # print(i, target_neuron, class_ind, blncd_acc)
            df_res = pd.DataFrame(res_data, columns=['layer_name', 'neuron_idx', 'class_idx', 'mean_balanced_acc'])
            df_res.to_csv('binary_separability' + postfix + f'_{i}.csv')


    def save_diff_between_MWD_means(self, df_path, df_labels_path, layer_list, n_classes, postfix):
        target_layer_names = [layer.replace('_', '.') for layer in layer_list]
        df = pd.read_pickle(df_path)
        df_labels = pd.read_pickle(df_labels_path)
        df_labels = df_labels.drop(df_labels[df_labels['layer_name'] != 'label'].index)
        # df_labels.loc[df_labels == 0] = -1 # negative labels are "-1" in sklearn, "0" in the df_labels
        df_labels.replace(0, -1)
        df_labels.loc[(df['layer_name'] == 'label') & (df['neuron_idx'] == 0)] = 0 # restore the number for the 0-th label
        df_labels_np = np.array(df_labels)[:, 2:].astype('int')

        print(df_labels_np.shape)
        chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        res_data = []
        for i, target_layer_name in enumerate(target_layer_names):
            for target_neuron in range(chunks[i]):
                print(i, target_neuron)
                target_neuron_activations = np.array(df.loc[(df['layer_name'] == target_layer_name) &
                                                       (df['neuron_idx'] == target_neuron)].drop(['layer_name', 'neuron_idx'],
                                                                                                 axis=1))[0].reshape(-1, 1)
                for class_ind in range(n_classes):
                    lbl = df_labels_np[class_ind]#labels1.astype('int')
                    train_pos = target_neuron_activations[lbl == 1]
                    train_neg = target_neuron_activations[lbl != 1]
                    mean_diff = train_pos.mean() - train_neg.mean()
                    res_data.append([target_layer_name, target_neuron, class_ind, mean_diff])
                    # print(i, target_neuron, class_ind, blncd_acc)
            df_res = pd.DataFrame(res_data, columns=['layer_name', 'neuron_idx', 'class_idx', 'mean_balanced_acc'])
            df_res.to_csv('diff_MWD_means' + postfix + f'_{i}.csv')


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
        plt.savefig(f'features_{folder_suffix}.svg', format='svg', bbox_inches='tight', dpi=1200)
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

        plt.savefig(f'feature_hists_{folder_suffix}.svg', format='svg', bbox_inches='tight', dpi=1200)

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
        plt.savefig(f'probs_{folder_suffix}.svg', format='svg', bbox_inches='tight', dpi=200)
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
                plt.savefig(f'{out_path}/{layer_idx}_{neuron_idx}.png', format='png', bbox_inches='tight')
                plt.close()
                xy = None
                '''
                neuron_idx = 2#8
                fig = plt.figure(figsize=(4.5, 2.5))
                ax = fig.gca()
                percentile05 = np.percentile(acts_for_layer_stacked[:, neuron_idx], 0.5)
                percentile5 = np.percentile(acts_for_layer_stacked[:, neuron_idx], 5)
                proper_hist(acts_for_layer_stacked[:, neuron_idx], title=f'0.5 percentile = {percentile05:.2f}\n'
                                                                         f'5 percentile = {percentile5:.2f}',
                                            if_determine_bin_size=False, bin_size=0.02, xlim_right=3, ax=ax)
                plt.xlabel('Activation')
                plt.ylabel('Count')
                axes = plt.gca()
                axes.get_yaxis().set_ticks([])
                plt.savefig(f'{out_path}/{layer_idx}_{neuron_idx}.png', format='png', bbox_inches='tight')
                plt.show()
                '''

    def plot_hists_of_spatial_activations_no_save(self, loader, target_layer_indices,
                                         out_path='hist_spatial_bettercifar10single', layer_list=layers_bn_afterrelu,
                                         chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        Path(out_path).mkdir(exist_ok=True)
        zero_ratios_data = []

        for layer_idx, layer_name in enumerate(target_layer_names):
            activation_dict = self.store_layer_activations_many(loader, [layer_idx], if_average_spatial=False, if_store_activations=True,
                                              layer_list=layer_list)
            activations = np.vstack(activation_dict[layer_name])
            n_total_feature_map_values = len(activations[:, 0].flatten())
            for neuron_idx in range(chunks[layer_idx]):
                print(neuron_idx)
                n_zeros = np.sum(activations[:, neuron_idx].flatten() == 0)
                ratio_zeros = n_zeros / n_total_feature_map_values
                proper_hist(activations[:, neuron_idx], title=f'{layer_idx}_{neuron_idx}\n{ratio_zeros :.2f}',
                            if_determine_bin_size=True)
                plt.savefig(f'{out_path}/{layer_idx}_{neuron_idx}.png', format='png', bbox_inches='tight')
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
            patch_size = 32
            activation_dict = self.store_layer_activations_many(loader, [layer_idx],
                                                                if_average_spatial=False, if_store_activations=False,
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

    def accs_barplot(self, suffix, target_neurons_dict, classes_dict, layer_list, n_show=10, if_show=False, out_path_suffix=''):
        target_layer_names = [cur.replace('_', '.') for cur in layer_list]
        n_best = n_show // 2
        out_path = 'accs_barplot_' + suffix + out_path_suffix
        Path(out_path).mkdir(exist_ok=True)
        for layer_idx, neurons in target_neurons_dict.items():
            # wasser_dists_cur = np.load(f'wasser_dists/wasser_dist_{suffix}_{layer_idx}.npy', allow_pickle=True).item()
            accs_cur_df = pd.read_csv(f'binary_separability_imagenet_val_nofolds_{layer_idx}.csv')
            mean_diffs_cur_df = pd.read_csv(f'diff_MWD_means_imagenet_val_nofolds_{layer_idx}.csv')
            for neuron in neurons:
                accs = np.array(accs_cur_df.loc[(accs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                                (accs_cur_df['neuron_idx'] == neuron)].drop(['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                print(accs.shape)
                mean_diffs = np.array(mean_diffs_cur_df.loc[(mean_diffs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                                                (mean_diffs_cur_df['neuron_idx'] == neuron)].drop(
                    ['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                print(mean_diffs.shape)
                class_wd_pairs = [(classes_dict[i], accs[i] * (-1 if mean_diffs[i] < 0 else 1)) for i in range(1000)]
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
                ax[0].set_xlim(-1, 1)

                proper_hist(np.array(list(zip(*class_wd_pairs))[1]),ax=ax[1], xlim_left=-1, xlim_right=1, bin_size=0.04)
                ax[1].set_yscale('log')
                ax[1].set_ylim(top=400)
                # ax[1].hist(list(zip(*class_wd_pairs))[1])
                # ax[1].set_xlim(-0.5, 0.5)
                plt.savefig(f'{out_path}/{title}.png', format='png', bbox_inches='tight')
                if if_show:
                    plt.show()
                plt.close()


    def correlate_accs_with_wasser(self, suffix, target_neurons_dict, classes_dict, layer_list, binary_separability_suffix='imagenet_val_nofolds', n_show=10, if_show=False, out_path_suffix='', wasser_type='shift', prefix_path='', binary_separabilities_local=False):
        prefix_path = Path(prefix_path)

        target_layer_names = [cur.replace('_', '.') for cur in layer_list]
        n_best = n_show // 2
        res = []
        res_pos_only = []
        res_neg_only = []
        for layer_idx, neurons in target_neurons_dict.items():
            wasser_cur = np.load(f'wasser_dists/wasser_{wasser_type}_{suffix}_{layer_idx}.npy', allow_pickle=True).item()
            prefix_path_bin_sep = Path('') if binary_separabilities_local else prefix_path
            accs_cur_df = pd.read_csv(prefix_path_bin_sep/f'binary_separability_{binary_separability_suffix}_{layer_idx}.csv')
            # mean_diffs_cur_df = pd.read_csv(prefix_path/f'diff_MWD_means_imagenet_val_nofolds_{layer_idx}.csv')
            for neuron in neurons:
                accs = np.array(accs_cur_df.loc[(accs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                                (accs_cur_df['neuron_idx'] == neuron)].drop(['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                # mean_diffs = np.array(mean_diffs_cur_df.loc[(mean_diffs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                #                                 (mean_diffs_cur_df['neuron_idx'] == neuron)].drop(
                #     ['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                
                # accs_with_signs = [accs[i] * (-1 if mean_diffs[i] < 0 else 1) for i in range(len(accs))]
                # wds = list(wasser_cur[neuron].values())
                accs_with_signs = np.sign(list(wasser_cur[neuron].values())) * accs
                wds = np.array(list(wasser_cur[neuron].values()))
                
                res.append(np.corrcoef(accs_with_signs, wds)[0, 1])
                res_pos_only.append(np.corrcoef(accs_with_signs[wds >= 0], wds[wds >= 0])[0, 1])
                res_neg_only.append(np.corrcoef(accs_with_signs[wds < 0], wds[wds < 0])[0, 1])
            
            # plt.plot(cur_res, '.')
            # plt.show()
        print(f"Whole domain: {np.mean(res)} +- {np.std(res)}")
        print(f"Positive shifts: {np.mean(res_pos_only)} +- {np.std(res_pos_only)}")
        print(f"Negative shifts: {np.mean(res_neg_only)} +- {np.std(res_neg_only)}")

    def correlate_accs_with_wasser_per_layer(self, suffix, target_neurons_dict, classes_dict, layer_list, n_show=10, if_show=False, out_path_suffix='', wasser_type='shift', prefix_path=''):
        prefix_path = Path(prefix_path)

        target_layer_names = [cur.replace('_', '.') for cur in layer_list]
        n_best = n_show // 2
        res = []
        pvals = []
        for layer_idx, neurons in target_neurons_dict.items():
            wasser_cur = np.load(f'wasser_dists/wasser_{wasser_type}_{suffix}_{layer_idx}.npy', allow_pickle=True).item()
            accs_cur_df = pd.read_csv(prefix_path/f'binary_separability_imagenet_val_nofolds_{layer_idx}.csv')
            # mean_diffs_cur_df = pd.read_csv(prefix_path/f'diff_MWD_means_imagenet_val_nofolds_{layer_idx}.csv')

            all_accs = []
            all_wds = []

            for neuron in neurons:
                accs = np.array(accs_cur_df.loc[(accs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                                (accs_cur_df['neuron_idx'] == neuron)].drop(['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                # mean_diffs = np.array(mean_diffs_cur_df.loc[(mean_diffs_cur_df['layer_name'] == target_layer_names[layer_idx]) &
                #                                 (mean_diffs_cur_df['neuron_idx'] == neuron)].drop(
                #     ['layer_name', 'neuron_idx', 'class_idx'], axis=1))[:, 1]
                
                # accs_with_signs = [accs[i] * (-1 if mean_diffs[i] < 0 else 1) for i in range(len(accs))]
                # wds = list(wasser_cur[neuron].values())
                accs_with_signs = np.sign(list(wasser_cur[neuron].values())) * accs
                wds = list(wasser_cur[neuron].values())

                all_accs.append(accs_with_signs)
                all_wds.append(wds)
            
            res_ = scipy.stats.pearsonr(accs_with_signs, wds)
            res.append(res_[0])
            pvals.append(res_[1])
            # plt.plot(cur_res, '.')
            # plt.show()
        print(np.mean(res), np.std(res))

    def find_negative_neurons_for_classes(self, loader, target_layer_indices, if_average_spatial=False,
                                          if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        assert len(target_layer_names) == 1
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if ('relu2' in name) and not if_left_lim_zero:
                out = inp[0]
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

                if if_store_labels:
                    labels += list(np.array(batch_val[-1]))

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)
                # if i == 10:
                #     break

        for hook in hooks:
            hook.remove()

        labels = np.array(labels)

        acts = np.vstack(activations[target_layer_names[0]])
        assert len(acts.shape) == 4
        n_neurons = acts.shape[1]

        # Option A: avg is negative
        # this gives a lot of neurons
        # acts_avg = np.mean(acts, axis=(-1, -2))
        # acts_neg = acts_avg <= 0

        # Option B: threshold is 0.01 instead of 0.0
        # this stricter version doesn't
        # acts_neg_full = acts <= 0.01
        # acts_neg_sum = np.sum(acts_neg_full, axis=(-1, -2))
        # acts_neg = acts_neg_sum == acts.shape[-1] * acts.shape[-2] # all are True

        # Option C: threshold 0.0, but allow some positive values
        # fraction_positives_to_allow = 0.5
        # acts_neg_full = acts <= 0.0
        # acts_neg_sum = np.sum(acts_neg_full, axis=(-1, -2))
        # acts_neg = acts_neg_sum >= acts.shape[-1] * acts.shape[-2] * fraction_positives_to_allow
        #
        # for c in np.unique(labels):
        #     a = acts_neg[labels == c]
        #     res = np.all(a, axis=0)
        #     for i, r in enumerate(res):
        #         if r:
        #             print(f'{target_layer_names} ; class {c} ; neuron {i}')

        # Option C2: many fractions
        acts_neg_full = acts <= 0.0
        acts_neg_sum = np.sum(acts_neg_full, axis=(-1, -2))


        fractions_positives_to_allow = [0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
        frac_to_n_neurons = {}
        for frac in fractions_positives_to_allow:
            acts_neg = acts_neg_sum >= acts.shape[-1] * acts.shape[-2] * frac
            cur = 0
            for c in np.unique(labels):
                a = acts_neg[labels == c]
                if_neurons_are_negative_for_the_class = np.all(a, axis=0)
                cur += np.sum(if_neurons_are_negative_for_the_class)
            frac_to_n_neurons[frac] = cur

        return frac_to_n_neurons

    def find_negative_neurons_for_classes_no_aggregation(self, loader, target_layer_indices, if_average_spatial=False,
                                                         if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        assert len(target_layer_names) == 1
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if ('relu2' in name) and not if_left_lim_zero:
                out = inp[0]
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

                if if_store_labels:
                    labels += list(np.array(batch_val[-1]))

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)
                # if i == 10:
                #     break
                if i % 10 == 0:
                    print(i)

        for hook in hooks:
            hook.remove()

        labels = np.array(labels)

        acts = np.vstack(activations[target_layer_names[0]])
        assert len(acts.shape) == 4
        n_neurons = acts.shape[1]

        # Option C2: many fractions
        acts_neg_full = acts <= 0.0
        acts_neg_sum = np.sum(acts_neg_full, axis=(-1, -2))


        fractions_positives_to_allow = [0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
        frac_to_neuron_to_classes = defaultdict(dict)
        for frac in fractions_positives_to_allow:
            acts_neg = acts_neg_sum >= acts.shape[-1] * acts.shape[-2] * frac
            for c in np.unique(labels):
                a = acts_neg[labels == c]
                if_neurons_are_negative_for_the_class = np.all(a, axis=0)
                for neuron in np.arange(n_neurons)[if_neurons_are_negative_for_the_class]:
                    if neuron not in frac_to_neuron_to_classes[frac]:
                        frac_to_neuron_to_classes[frac][neuron] = [] # defaultdict of defaultdicts is hard to store
                    frac_to_neuron_to_classes[frac][neuron].append(c)
            if frac not in frac_to_neuron_to_classes:
                frac_to_neuron_to_classes[frac] = {}

        return frac_to_neuron_to_classes

    def find_negative_neurons_for_classes_no_aggregation_inspect_average(self, loader, target_layer_indices, if_average_spatial=True,
                                                         if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        assert len(target_layer_names) == 1
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if ('relu2' in name) and not if_left_lim_zero:
                out = inp[0]
            if if_average_spatial:
                cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
            else:
                cur_activations = out.detach().cpu().numpy()
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

                if if_store_labels:
                    labels += list(np.array(batch_val[-1]))

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)
                # if i == 10:
                #     break
                if i % 10 == 0:
                    print(i)

        for hook in hooks:
            hook.remove()

        labels = np.array(labels)

        acts = np.vstack(activations[target_layer_names[0]])
        assert len(acts.shape) == 2
        n_neurons = acts.shape[1]

        acts_avg_neg = acts < 0

        # acts_avg_neg = np.mean(acts, axis=(-1, -2)) < 0

        # Q: WHY THE FRACTIONS ARE NOT MONOTOLICALLY DECREASING FROM 80% ACTS BELOW ZERO TO 100% ACTS BELOW ZERO
        # A: this just counts neurons where exactly one class is negatively activating, therefore having a lower fractions as
        #    the criterion may lead to more classes and therefore less unique neurons

        fractions_positives_to_allow = [0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0]
        frac_to_neuron_to_classes = defaultdict(dict)
        for frac in fractions_positives_to_allow:
            for c in np.unique(labels):
                n_samples_for_class = np.sum(labels == c)
                acts_cur = acts_avg_neg[labels == c]
                idx_neg_neurons = np.sum(acts_cur, axis=0) >= frac * n_samples_for_class
                for neuron in np.arange(n_neurons)[idx_neg_neurons]:
                    if neuron not in frac_to_neuron_to_classes[frac]:
                        frac_to_neuron_to_classes[frac][neuron] = [] # defaultdict of defaultdicts is hard to store
                    frac_to_neuron_to_classes[frac][neuron].append(c)
            if frac not in frac_to_neuron_to_classes:
                frac_to_neuron_to_classes[frac] = {}

        return frac_to_neuron_to_classes


    def find_positive_neurons_for_classes_no_aggregation_inspect_average(self, loader, target_layer_indices, if_average_spatial=True,
                                                         if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        target_layer_names = [layer_list[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        assert len(target_layer_names) == 1
        activations = {}

        def save_activation(activations, name, mod, inp, out):
            if ('relu2' in name) and not if_left_lim_zero:
                out = inp[0]
            if if_average_spatial:
                cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
            else:
                cur_activations = out.detach().cpu().numpy()
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

                if if_store_labels:
                    labels += list(np.array(batch_val[-1]))

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)
                # if i == 10:
                #     break
                if i % 10 == 0:
                    print(i)

        for hook in hooks:
            hook.remove()

        labels = np.array(labels)

        acts = np.vstack(activations[target_layer_names[0]])
        assert len(acts.shape) == 2
        n_neurons = acts.shape[1]

        acts_avg_pos = acts > 0

        fractions_negatives_to_allow = [0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0]
        frac_to_neuron_to_classes = defaultdict(dict)
        for frac in fractions_negatives_to_allow:
            for c in np.unique(labels):
                n_samples_for_class = np.sum(labels == c)
                acts_cur = acts_avg_pos[labels == c]
                idx_pos_neurons = np.sum(acts_cur, axis=0) >= frac * n_samples_for_class
                for neuron in np.arange(n_neurons)[idx_pos_neurons]:
                    if neuron not in frac_to_neuron_to_classes[frac]:
                        frac_to_neuron_to_classes[frac][neuron] = [] # defaultdict of defaultdicts is hard to store
                    frac_to_neuron_to_classes[frac][neuron].append(c)
            if frac not in frac_to_neuron_to_classes:
                frac_to_neuron_to_classes[frac] = {}

        return frac_to_neuron_to_classes


    def find_negative_neurons_for_classes_all_layers(self, loader, if_average_spatial=False,
                                          if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        for i in range(len(layer_list)):
            print(f'Layer {i}')
            frac_to_n_neurons = self.find_negative_neurons_for_classes(loader, [i], layer_list=layer_list, if_average_spatial=if_average_spatial,
                                                   if_store_labels=if_store_labels, if_left_lim_zero=if_left_lim_zero)
            pickle.dump(frac_to_n_neurons, open(f'layer_{i}_frac_to_neurons.pkl', 'wb'))
            plt.bar(list(map(str, frac_to_n_neurons.keys())), frac_to_n_neurons.values(), width=0.4)
            plt.yscale('log')
            plt.xlabel('Percent negative values')
            plt.ylabel('Number of neuron+class combinations')
            plt.title(f'Layer {i}')
            plt.savefig(f'layer_{i}_frac_to_neurons.png', format='png', bbox_inches='tight')
            plt.show()

    def find_negative_neurons_for_classes_all_layers_no_aggregation(self, loader, if_average_spatial=False,
                                                                    if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        for i in range(len(layer_list)):
            print(f'Layer {i}')
            frac_to_neuron_to_classes = self.find_negative_neurons_for_classes_no_aggregation(loader, [i], layer_list=layer_list, if_average_spatial=if_average_spatial,
                                                                                              if_store_labels=if_store_labels, if_left_lim_zero=if_left_lim_zero)
            pickle.dump(frac_to_neuron_to_classes, open(f'layer_{i}_frac_to_neuron_to_classes.pkl', 'wb'))
            plt.bar(list(map(str, frac_to_neuron_to_classes.keys())), np.sum([len(v) for v in frac_to_neuron_to_classes.values()]), width=0.4)
            plt.yscale('log')
            plt.xlabel('Percent negative values')
            plt.ylabel('Number of neuron+class combinations')
            plt.title(f'Layer {i}')
            plt.savefig(f'layer_{i}_frac_to_neuron_to_classes.png', format='png', bbox_inches='tight')
            plt.show()

    def find_negative_neurons_for_classes_all_layers_no_aggregation_inspect_average(self, loader, if_average_spatial=False,
                                                                    if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        for i in range(len(layer_list)):
            print(f'Layer {i}')
            frac_to_neuron_to_classes = self.find_negative_neurons_for_classes_no_aggregation_inspect_average(loader, [i], layer_list=layer_list, if_average_spatial=if_average_spatial,
                                                                                              if_store_labels=if_store_labels, if_left_lim_zero=if_left_lim_zero)
            pickle.dump(frac_to_neuron_to_classes, open(f'layer_{i}_frac_to_neuron_to_classes_inspect_avg.pkl', 'wb'))
            plt.bar(list(map(str, frac_to_neuron_to_classes.keys())), np.sum([len(v) for v in frac_to_neuron_to_classes.values()]), width=0.4)
            plt.yscale('log')
            plt.xlabel('Percent negative values')
            plt.ylabel('Number of neuron+class combinations')
            plt.title(f'Layer {i}')
            plt.savefig(f'layer_{i}_frac_to_neuron_to_classes_inspect_avg.png', format='png', bbox_inches='tight')
            plt.show()

    def find_positive_neurons_for_classes_all_layers_no_aggregation_inspect_average(self, loader, if_average_spatial=False,
                                                                    if_store_labels=True, layer_list=layers_bn_prerelu, if_left_lim_zero=False):
        for i in range(len(layer_list)):
            print(f'Layer {i}')
            frac_to_neuron_to_classes = self.find_positive_neurons_for_classes_no_aggregation_inspect_average(loader, [i], layer_list=layer_list, if_average_spatial=if_average_spatial,
                                                                                              if_store_labels=if_store_labels, if_left_lim_zero=if_left_lim_zero)
            pickle.dump(frac_to_neuron_to_classes, open(f'layer_{i}_frac_to_neuron_to_classes_inspect_avg_positive.pkl', 'wb'))
            plt.bar(list(map(str, frac_to_neuron_to_classes.keys())), np.sum([len(v) for v in frac_to_neuron_to_classes.values()]), width=0.4)
            plt.yscale('log')
            plt.xlabel('Percent positive values')
            plt.ylabel('Number of neuron+class combinations')
            plt.title(f'Layer {i}')
            plt.savefig(f'layer_{i}_frac_to_neuron_to_classes_inspect_avg_positive.png', format='png', bbox_inches='tight')
            plt.show()


    def plot_n_negative_neurons_for_classes_for_all_layers(self, chunks, layer_list=layers_bn_prerelu):
        frac_to_n_neurons_per_layer = defaultdict(list)
        for i in range(len(layer_list)):
            d = pickle.load(open(f'layer_{i}_frac_to_neurons.pkl', 'rb'))
            for f, n_neurons in d.items():
                # frac_to_n_neurons_per_layer[f].append(n_neurons)
                frac_to_n_neurons_per_layer[f].append(n_neurons / (chunks[i] * 1000))

        layer_indices = list(range(len(layer_list)))


        prop_cycle_backup = plt.rcParams["axes.prop_cycle"]
        n_fracs = len(frac_to_n_neurons_per_layer.keys())
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_fracs)))
        plt.figure(figsize=(8, 5))
        for f, n_neurons_per_layer in frac_to_n_neurons_per_layer.items():
            plt.plot(layer_indices, n_neurons_per_layer, '-o', label=str(f))
        plt.yscale('symlog')
        plt.xticks(layer_indices)
        plt.xlabel('Layer index')
        plt.ylabel('Number of neuron+class combinations')
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(axis='y')
        plt.show()

        plt.rcParams["axes.prop_cycle"] = prop_cycle_backup

    def plot_n_negative_neurons_for_classes_for_all_layers_no_aggregation(self, chunks, layer_list=layers_bn_prerelu):
        frac_to_n_neurons_per_layer = defaultdict(list)
        for i in range(len(layer_list)):
            d = pickle.load(open(f'layer_{i}_frac_to_neuron_to_classes.pkl', 'rb'))
            for f, neurons_to_classes in d.items():
                # this line reproduces the old graph - to make sure that the number are consistent
                # frac_to_n_neurons_per_layer[f].append(np.sum([len(cl_list) for cl_list in neurons_to_classes.values()]) / (chunks[i] * 1000))
                # this is the new graph
                frac_to_n_neurons_per_layer[f].append(len(neurons_to_classes) / chunks[i])

        layer_indices = list(range(len(layer_list)))


        prop_cycle_backup = plt.rcParams["axes.prop_cycle"]
        n_fracs = len(frac_to_n_neurons_per_layer.keys())
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_fracs)))
        plt.figure(figsize=(8, 5))
        for f, n_neurons_per_layer in frac_to_n_neurons_per_layer.items():
            plt.plot(layer_indices, n_neurons_per_layer, '-o', label=str(f))
        # plt.yscale('symlog')
        plt.xticks(layer_indices)
        plt.xlabel('Layer index')
        plt.title('Proportion of neurons with >= 1 negative class')
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(axis='y')
        plt.savefig('ratio_neurons_coding_for_negative.png', bbox_inches='tight')
        plt.show()

        plt.rcParams["axes.prop_cycle"] = prop_cycle_backup

    def plot_n_neurons_for_classes_for_all_layers_no_aggregation_inspect_average(self, chunks, layer_list=layers_bn_prerelu, proportion=False, positive=False):
        frac_to_n_neurons_per_layer = defaultdict(list)
        for i in range(len(layer_list)):
            d = pickle.load(open(f'layer_{i}_frac_to_neuron_to_classes_inspect_avg{"_positive" if positive else ""}.pkl', 'rb'))
            for f, neurons_to_classes in d.items():
                # this line reproduces the old graph - to make sure that the number are consistent
                # frac_to_n_neurons_per_layer[f].append(np.sum([len(cl_list) for cl_list in neurons_to_classes.values()]) / (chunks[i] * 1000))
                # this is the new graph
                # frac_to_n_neurons_per_layer[f].append(len(neurons_to_classes) / chunks[i])
                if proportion:
                    frac_to_n_neurons_per_layer[f].append(len([neuron for neuron, class_list in neurons_to_classes.items() if len(class_list) <= 1]) / chunks[i])
                else:
                    frac_to_n_neurons_per_layer[f].append(len([neuron for neuron, class_list in neurons_to_classes.items() if len(class_list) <= 1]))

        layer_indices = list(range(len(layer_list)))


        prop_cycle_backup = plt.rcParams["axes.prop_cycle"]
        n_fracs = len(frac_to_n_neurons_per_layer.keys())
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_fracs)))
        plt.figure(figsize=(8, 5))
        for f, n_neurons_per_layer in frac_to_n_neurons_per_layer.items():
            plt.plot(layer_indices, n_neurons_per_layer, '-o', label=str(f))
        plt.xticks(layer_indices)
        plt.xlabel('Layer index')
        # plt.title('Proportion of neurons with >= 1 positive class')
        if proportion:
            plt.ylim((0.0, 0.3))
            plt.title(f'Proportion of neurons with >= 1 {"positive" if positive else "negative"} class, but <= 1')
        else:
            plt.yscale('symlog')
            # plt.yscale('log', nonpositive='mask')
            plt.ylim((0, 10e2))
            plt.title(f'Count of neurons with >= 1 {"positive" if positive else "negative"} class, but <= 1')
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(axis='y')
        plt.savefig(f'ratio_neurons_coding_for_{"positive" if positive else "negative"}_avg{"_proportion" if proportion else ""}.png', bbox_inches='tight')
        plt.savefig(f'ratio_neurons_coding_for_{"positive" if positive else "negative"}_avg{"_proportion" if proportion else ""}.svg', bbox_inches='tight')
        plt.show()

        plt.rcParams["axes.prop_cycle"] = prop_cycle_backup

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

    model_to_use = 'my' #'resnet18'
    if_pretrained_imagenet = model_to_use != 'my'
    # ac, params, configs = ActivityCollector.create_activity_collector(save_model_path, param_file, if_pretrained_imagenet)
    ac = ActivityCollector(save_model_path, param_file, model_to_use)
    gwc = GradWassersteinCorrelator(save_model_path, param_file, model_to_use)
    params, configs = ac.params, ac.configs

    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
        with open(param_file) as json_params:
            params = json.load(json_params)
        with open('configs.json') as config_params:
            configs = json.load(config_params)
        params['dataset'] = 'imagenet_val'
        # params['dataset'] = 'imagenet_test'
        params['batch_size'] = 256#320
    print(model_name_short)
    
    params['dataset'] = 'cifar10'
    # params['dataset'] = 'broden_val'

    params['batch_size'] = 128#50/2#50/4#1000/4#26707#11246#1251#10001#
    
    _, val_loader, tst_loader = datasets.get_dataset(params, configs)
    loader = val_loader#tst_loader#
    
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
    
    path_prefix_mnt = Path("/mnt/raid/data/chebykin/pycharm_project_AA/")
    
    # ac.assess_binary_separability_1vsAll_whole_network(path_prefix_mnt/'bettercifar10single_nonsparse_afterrelu.pkl', path_prefix_mnt/'df_label_cifar.pkl',
    #                                                    layers_bn, 10, postfix='_cifar')


    #TODO: adapt this to the regression discontinuity experiment
    # prerelu activations are not available for bettercifar10
    # i need layers_before_relu

    #DONE: check how relu2 handles the bias: there is no bias

    # layerinds = list(filter(lambda x: x % 2 == 0, range(15))) # those are just the relu2 layers, BUT i missed the first half of the blocks.. these are also before relu
    # ac.store_layer_activations_many(loader, layerinds, out_path_postfix='_bettercifar10single_prerelu',
    #                                 layer_list=layers_bn_prerelu, if_store_labels=True,
    #                                 if_store_layer_names=True, if_average_spatial=True, if_save_separately=False, out_path_prefix=Path('local_storage'), save_logits=True)
    
    # plotting in a ipynb

    # ac.store_layer_activations_many(loader, range(len(layers_bn_afterrelu)), out_path_postfix='_bettercifar10single_afterrelu',
    #                                 layer_list=layers_bn_afterrelu, if_store_labels=True,
    #                                 if_store_layer_names=True, if_average_spatial=True, if_save_separately=False, out_path_prefix=Path('local_storage'), save_logits=True)

    #TODO: skip this

    # gwc.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_val_bettercifar10single_afterrelu_all_samples.pkl',
    #                                                       if_pretrained_imagenet=False, layers=layers_bn_afterrelu,
    #                                                       neuron_nums=None, only_in_class_samples=False)

    # ac.assess_binary_separability_1vsAll_whole_network(path_prefix_mnt/'bettercifar10single_nonsparse_afterrelu.pkl', path_prefix_mnt/'df_label_cifar.pkl',
    #                                                    layers_bn, 10, postfix='_cifar')                                                      
    
    # TODO: UNTIL HERE
    
    # convert_imagenet_path_to_label_dict_to_df(path_prefix=path_prefix_mnt)
    
    
    # #TODO compute if there is time
    # ac.assess_binary_separability_1vsAll_whole_network(path_prefix_mnt/'pretrained_imagenet_afterrelu.pkl', path_prefix_mnt/'df_label_imagenet_val.pkl',
    #                                                    layers_bn, 1000, postfix='_imagenet_val_cv')
    
    # ac.save_diff_between_MWD_means('pretrained_imagenet_afterrelu.pkl', 'df_label_imagenet_val.pkl',
    #                                                    layers_bn, 1000, postfix='_imagenet_val_nofolds')
    
    chunks = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    # ac.accs_barplot('', dict(zip(range(15), [range(c) for c in chunks])),
    #                        imagenet_dict, layers_bn, n_show=20, out_path_suffix='')
    
    #DONE: correlate positive and negative separate

    # ac.correlate_accs_with_wasser('attr_hist_pretrained_imagenet_afterrelu_test', dict(zip(range(15), [range(c) for c in chunks])),
    #                        imagenet_dict, layers_bn, n_show=20, out_path_suffix='', prefix_path=path_prefix_mnt)

    #DONE: imagenet valid

    ac.correlate_accs_with_wasser('attr_hist_pretrained_imagenet_afterrelu', dict(zip(range(15), [range(c) for c in chunks])),
                           imagenet_dict, layers_bn, n_show=20, out_path_suffix='', prefix_path=path_prefix_mnt)
    
    #DONE: do this for cifar10

    # ac.correlate_accs_with_wasser('attr_hist_bettercifar10single', dict(zip(range(15), [range(c) for c in chunks])),
    #                        cifar10_dict, layers_bn, n_show=20, out_path_suffix='', binary_separability_suffix='cifar', prefix_path=path_prefix_mnt, binary_separabilities_local=True)

    # ac.store_layer_activations_many(loader, [0], if_average_spatial=False, if_store_labels=False, if_save_separately=False,
    #                                 out_path_postfix='pretrained_imagenet_afterrelu_again2', if_store_layer_names=True)
    # ac.store_layer_activations_many(loader, [0], if_average_spatial=False, if_store_labels=False, if_save_separately=False,
    #                                 out_path_postfix='pretrained_imagenet_prerelu_again2', if_store_layer_names=True,
    #                                 layer_list=layers_bn_prerelu, if_left_lim_zero=False)
    # ac.plot_hists_of_spatial_activations(path='activations_on_validation_preserved_spatial_pretrained_imagenet_afterrelu_again2.npy',
    #                                              out_path='hist_spatial_imagenet_afterrelu_again',
    #                                              layer_list=layers_bn_afterrelu[0:1], chunks=chunks)
    # ac.plot_hists_of_spatial_activations(path='activations_on_validation_preserved_spatial_pretrained_imagenet_prerelu_again2.npy',
    #                                              out_path='hist_spatial_imagenet_prerelu_again',
    #                                              layer_list=layers_bn_prerelu[0:1], chunks=chunks)
    # ac.plot_overall_hists('pretrained_imagenet_afterrelu.pkl', 'overall_hists_imagenet_afterrelu', layers_bn)

    ac.store_layer_activations_many(loader, [6, 10, 14], if_average_spatial=False, if_store_labels=True, if_store_activations=False, if_save_separately=True,
                                    out_path_postfix='_pretrained_imagenet', if_store_layer_names=False, out_path_prefix='stored_activations/')
    exit()
    
    # ac.find_negative_neurons_for_classes(loader, [10], if_average_spatial=False, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.find_negative_neurons_for_classes_all_layers(loader, if_average_spatial=False, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.plot_n_negative_neurons_for_classes_for_all_layers(chunks)
    # ac.find_negative_neurons_for_classes_no_aggregation(loader, [10], if_average_spatial=False, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.find_negative_neurons_for_classes_all_layers_no_aggregation(loader, if_average_spatial=False, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.plot_n_negative_neurons_for_classes_for_all_layers_no_aggregation(chunks)
    # ac.find_negative_neurons_for_classes_no_aggregation_inspect_average(loader, [10], if_average_spatial=True, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.find_negative_neurons_for_classes_all_layers_no_aggregation_inspect_average(loader, if_average_spatial=True, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.plot_n_neurons_for_classes_for_all_layers_no_aggregation_inspect_average(chunks, proportion=False, positive=False)

    # # ac.find_positive_neurons_for_classes_all_layers_no_aggregation_inspect_average(loader, if_average_spatial=True, if_store_labels=True, layer_list=layers_bn_prerelu)
    # ac.plot_n_neurons_for_classes_for_all_layers_no_aggregation_inspect_average(chunks, proportion=False, positive=True)

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
    # ac.compute_attr_hist_for_neuron_pandas_wrapper(loader, 'celeba_fc_v3.pkl', {13:'all'},
    #                                                'attr_hist_celeba_fc_v3', if_cond_label=False,
    #                                                used_neurons='resnet_full',
    #                                                dataset_type='celeba',
    #                                                 sorted_dict_path='img_paths_most_activating_sorted_dict_paths_celeba_fc_v3.npy',
    #                                                if_calc_wasserstein=True, offset=(0.5, 0.5), if_show=False,
    #                                                if_force_recalculate=True, if_left_lim_zero=True,
    #                                                 layer_list=layers_bn_afterrelu)
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
    ac.plot_hists_of_spatial_activations_no_save(loader, list(range(15)),
                                                 out_path='hist_spatial_imagenet_r34_afterrelu',
                                                 layer_list=layers_bn_afterrelu, chunks=chunks)
    # ac.store_highest_activating_patches(loader, [0], out_path='patches_bettercifar10single',
    #                                     base_path='/mnt/raid/data/chebykin/cifar/my_imgs',
    #                                     image_size=32)
    # ac.store_highest_activating_patches(loader, [0], out_path='patches_imagenet',
    #                                     base_path='/mnt/raid/data/chebykin/imagenet_val/my_imgs',
    #                                     image_size=224)
    # exit()
    # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, f'grads_test_early_{model_name_short}.pkl',
    #                                                      if_pretrained_imagenet, early_layers_bn_afterrelu)
    # ac.calc_gradients_wrt_output_whole_network_all_tasks(loader, 'grads_pretrained_imagenet_afterrelu_test.pkl', if_pretrained_imagenet)
    # exit()
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

    ac.assess_binary_separability('nonsparse_afterrelu_bettercifar.pkl', 'df_label_cifar.pkl', 14, 12, 7, 8)
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