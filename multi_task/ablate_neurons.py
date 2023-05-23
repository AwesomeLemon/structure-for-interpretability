import json
import pickle
import time

from pathlib import Path

import torch
import torchvision
from functools import partial
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sns
sns.set_context("paper")
# import scienceplots
# plt.style.use(['science','no-latex'])
from util.util import *
from scipy.integrate import simps
import scipy.stats

from sklearn.metrics import confusion_matrix
import pandas as pd
try:
    from load_model import eval_trained_model, load_trained_model
    import datasets
    from util.dicts import imagenet_dict
except:
    from multi_task.load_model import eval_trained_model, load_trained_model
    from multi_task import datasets
    from multi_task.util.dicts import imagenet_dict

# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# cudnn.enabled = True

imagenet_dict_reversed = dict(zip(imagenet_dict.values(), imagenet_dict.keys()))

if_pretrained_imagenet = True
# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

def calc_l1_norms_of_weights(save_model_path):
    if not if_pretrained_imagenet:
        model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
        state = torch.load(save_model_path)
        model_rep_state = state['model_rep']
    else:
        model_name_short = 'pretrained_imagenet'
        model = torchvision.models.__dict__['resnet18'](pretrained=True).cuda()
        model_rep_state = model.state_dict()
    layers_to_neurons_to_l1_norms = {}
    for k in model_rep_state.keys():
        if 'conv' not in k:
            continue
        cur_dict = {}
        for i in range(model_rep_state[k].shape[0]):
            cur_dict[i] = torch.norm(model_rep_state[k][i], 1).item()
        layers_to_neurons_to_l1_norms[k] = cur_dict
    np.save(f'l1_norms_{model_name_short}.npy', layers_to_neurons_to_l1_norms, allow_pickle=True)

def reload_model(save_model_path, param_file):
    if not if_pretrained_imagenet:
        try:
            trained_model = load_trained_model(param_file, save_model_path)
        except:
            print('assume problem where cifar networks ignored the enable_bias parameter')
            trained_model = load_trained_model(param_file, save_model_path,
                                               if_actively_disable_bias=True)
    else:
        trained_model = torchvision.models.__dict__['resnet18'](pretrained=True).cuda()
    return trained_model

def ablate_neurons(model, layer_ind, neurons_to_ablate):
    target_layer_name = layers_bn[layer_ind].replace('_', '.')

    # model['rep'].connectivities[layer_ind][:, neurons_to_ablate] = 0
    def save_activation(name, mod, inp, out):
        if name == target_layer_name:
            out[:, neurons_to_ablate, :, :] = 0
            return out

    hooks = []
    model_true = model
    if not if_pretrained_imagenet:
        model_true = model['rep']
    for name, m in model_true.named_modules():
        hooks.append(m.register_forward_hook(partial(save_activation, name)))
    return hooks


def calc_accuracies_after_ablation(param_file, model, layer_ind, neurons_sorted, step_size, n_steps, n_start):
    accs = []
    loader = None
    start_hooks = ablate_neurons(model, layer_ind, neurons_sorted[:n_start])
    for i in range(n_steps + 1):
        if n_start + i * step_size > len(neurons_sorted):
            print(i, n_start + i * step_size, len(neurons_sorted))
            break
        hooks = ablate_neurons(model, layer_ind, neurons_sorted[n_start:n_start + i * step_size])
        errror, loader = eval_trained_model(param_file, model, loader=loader, if_print=False, if_pretrained_imagenet=if_pretrained_imagenet)
        acc_cur = 1 - errror
        accs.append(acc_cur)
        for hook in hooks:
            hook.remove()
    print(accs)
    for hook in start_hooks:
        hook.remove()
    return np.array(accs)


def select_neurons_to_ablate_by_wasserstein_dist(data_per_neuron, selection_fun=lambda vals: np.sum(np.abs(vals))):
    # wasser_dists = np.load(wd_path, allow_pickle=True).item()
    # total_wd_per_neuron = [selection_fun(list(neuron_wds.values())) for neuron_wds in wasser_dists.values()]
    total_wd_per_neuron = [selection_fun(neuron_data) for neuron_data in data_per_neuron]
    idx = np.argsort(total_wd_per_neuron)[::-1]
    return np.array(idx)


def compare_ablation_strategies(save_model_path, param_file, step_size, n_steps, layer_ind, n_start,
                                selections_funs, folder_name):
    Path(f'ablations/{folder_name}').mkdir(exist_ok=True)
    model = reload_model(save_model_path, param_file)
    for (fun, fun_name, data_per_neuron) in selections_funs:
        print(fun_name)
        # model = reload_model(save_model_path, param_file)
        sorted_neurons = select_neurons_to_ablate_by_wasserstein_dist(data_per_neuron, selection_fun=fun)
        accs = calc_accuracies_after_ablation(param_file, model, layer_ind, sorted_neurons, step_size, n_steps, n_start)
        x_range = range(0, (n_steps + 1) * step_size, step_size)[:len(accs)]
        area = simps(accs, dx=step_size / x_range[-1])
        plt.plot(x_range, accs, label=fun_name + f'_{area:.2f}')
        np.save(f'ablations/{folder_name}/accs_{fun_name}_{layer_ind}.npy', (accs, area), allow_pickle=True)
    plt.legend()
    plt.savefig(f'ablations/{folder_name}/l_{layer_ind}.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_ablation_strategy_comparisons(folder_name, fun_names, layer_inds):
    for layer_ind in layer_inds:
        step_size = 5
        n_steps = 51
        if layer_ind > 11:
            step_size = 10
        for fun_name in fun_names:
            accs, area = np.load(f'ablations/{folder_name}/accs_{fun_name}_{layer_ind}.npy', allow_pickle=True)
            x_range = range(0, (n_steps + 1) * step_size, step_size)[:len(accs)]
            plt.plot(x_range, accs, label=fun_name + f'_{area:.2f}')
        plt.legend()
        plt.title(layer_ind)
        plt.savefig(f'ablations/{folder_name}/l_{layer_ind}_absmax_{"_".join(fun_names)}.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

def create_areas_array(folder_name, fun_names, layer_inds):
    areas = np.zeros((len(layer_inds), len(fun_names)))
    for layer_ind in layer_inds:
        for i, fun_name in enumerate(fun_names):
            _, area = np.load(f'ablations/{folder_name}/accs_{fun_name}_{layer_ind}.npy', allow_pickle=True)
            areas[layer_ind, i] = area
    return areas

def make_predictions(model, loaded_activations, layeri, if_disable, disable_ind=None):
    with torch.no_grad():
        acts = loaded_activations.clone()#.cuda()
        if if_disable:
            acts[:, disable_ind] = 0  # 10 * acts_avg[i]#

        y_pred = []
        n_batches = 40
        n_samples = acts.shape[0]
        batch_size = int(np.ceil(n_samples / n_batches))
        for i in range(n_batches):
            x = acts[i * batch_size:(i+1) * batch_size, :].cuda()
            if layeri == 6:
                x = model.layer3(x) # for layer 6
            if layeri <= 10:
                x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            x = model.fc(x)
            y_pred_cur = torch.argmax(x, dim=1).detach().cpu().numpy()
            y_pred.append(y_pred_cur)
        y_pred = np.hstack(y_pred)
        del acts
    return y_pred

def sort_neurons_for_class(wasser_dists_values, target_class):
    w_dists = np.array([w_dict[target_class] for w_dict in wasser_dists_values])
    idx = np.argsort(w_dists)
    return idx, w_dists


def plot_prec_recall_change(save_model_path, param_file,
                          # saved_activations_path='activations_on_validation_preserved_spatial_10_pretrained_imagenet.npy',
                          saved_activations_path='activations_on_validation_14_pretrained_imagenet.npy',
                            saved_labels_path='labels_pretrained_imagenet.npy', wd_path=None,
                            l1_norms=None):

    model = reload_model(save_model_path, param_file)
    print(saved_activations_path)
    # works for layers 10 & 14:
    # with open(saved_activations_path, 'rb') as f:
    #     loaded_activations = torch.tensor(pickle.load(f))
    # works for layer 6:
    loaded_activations = torch.tensor(np.load(saved_activations_path))

    labels = np.load(saved_labels_path)
    wasser_dists_values = list(np.load(wd_path, allow_pickle=True).item().values())
    # print('Attention! What I do to get last linear weights only works for the pretrained imagenet model')

    if 'pretrained_imagenet' in saved_activations_path:
        fc_weights = model.fc.weight.detach().cpu().numpy()

    layeri = int(saved_activations_path.split('_')[-3])

    y_pred = make_predictions(model, loaded_activations, layeri, if_disable=False)
    conf_matr = confusion_matrix(labels, y_pred)
    precision_before, recall_before = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)

    n_neurons_dict = {
        6: 128, 
        10: 256, 
        14: 512
    }

    testing = False

    if not testing:
        ablate_neurons_config = {
            256: [8, 16, 32, 64, 128, 192],
            512: [16, 32, 64, 128, 256, 384],
            128: [4, 8, 16, 32, 64, 96], 
        }
    else:
        ablate_neurons_config = {
            256: [8, 16, 32],
            512: [16, 32, 64],
            128: [4, 8, 16], 
        }

    n_neurons = n_neurons_dict[layeri]
    assert n_neurons == loaded_activations.size()[1]
    
    n_classes = 1000 if not testing else 5

    n_neurons_to_disable = ablate_neurons_config[n_neurons]

    save_postfix = f'layer{layeri}_disabled_{"_".join(str(x) for x in n_neurons_to_disable)}'
    if n_classes == 1000:
        save_postfix += '_imagenet'
    elif n_classes == 10:
        save_postfix += '_cifar10'
    elif n_classes == 5:
        save_postfix += '_testing'
    else:
        raise Exception(f"Unknown n_classes: {n_classes}")

    # def change_2d_array_into_dict(arr, neuron_sorting_fun_names):
    #         means_dict = {}
    #         for i, n in enumerate(neuron_sorting_fun_names):
    #             means_dict[n] = arr[i, :]
    #         return means_dict

    permuted_neurons = np.random.permutation(n_neurons)

    sorting_funs_list = [
        ('random', lambda a, b: permuted_neurons),
        ('sort-max', lambda wasser_dists_values, idx: np.array(sort_neurons_for_class(wasser_dists_values, idx)[0][::-1])),
        ('sort-min', lambda wasser_dists_values, idx: sort_neurons_for_class(wasser_dists_values, idx)[0])
    ]

    if layeri == 14:
        if 'pretrained_imagenet' in saved_activations_path:
            sorting_funs_list += [
                ('sort-readout-max', lambda _, idx: np.argsort(fc_weights[idx])[::-1]),
                ('sort-readout-min', lambda _, idx: np.argsort(fc_weights[idx])),
            ]
    else:
        sorting_funs_list += [
            ('sort-l1', lambda _, idx: np.argsort(l1_norms))# this is sort-l1 for non-last layers
        ]

    neuron_sorting_fun_names, neuron_sorting_funs = zip(*sorting_funs_list)

    try:
        precision_means_dict = np.load(f'prec_means_dict{save_postfix}.npy', allow_pickle=True).item()
        recall_means_dict = np.load(f'recall_means_dict{save_postfix}.npy', allow_pickle=True).item()
        precision_stds_dict = np.load(f'prec_stds_dict{save_postfix}.npy', allow_pickle=True).item()
        recall_stds_dict = np.load(f'recall_stds_dict{save_postfix}.npy', allow_pickle=True).item()
    except:
        precision_means_dict = {}
        recall_means_dict = {}
        precision_stds_dict = {}
        recall_stds_dict = {}
        
        
        # precision_means = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
        # recall_means = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
        # precision_stds = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
        # recall_stds = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
    
    for (sorting_fun_name, sorting_fun) in zip(neuron_sorting_fun_names, neuron_sorting_funs):
        if sorting_fun_name in precision_means_dict:
            continue

        precision_means_dict[sorting_fun_name] = np.zeros(len(n_neurons_to_disable))
        recall_means_dict[sorting_fun_name] = np.zeros(len(n_neurons_to_disable))
        precision_stds_dict[sorting_fun_name] = np.zeros(len(n_neurons_to_disable))
        recall_stds_dict[sorting_fun_name] = np.zeros(len(n_neurons_to_disable))

        for j, n_to_disable in enumerate(n_neurons_to_disable):
            print(f'{n_to_disable} disabled')
            p_diffs = []
            r_diffs = []
            old_cur_idx = None
            for class_idx in range(n_classes):
                print(class_idx)
                st = time.time()
                cur_idx = list(sorting_fun(wasser_dists_values, class_idx)[:n_to_disable])
                # x = loaded_activations.clone().cuda()
                # x[:, cur_idx] = 0
                # x = model.fc(x)
                # conf_matr = confusion_matrix(labels, torch.argmax(x, dim=1).detach().cpu().numpy())
                
                if (old_cur_idx != cur_idx):
                    # don't recompute if ablations doesn't change
                    y_pred = make_predictions(model, loaded_activations, layeri, if_disable=True, disable_ind=cur_idx)
                    conf_matr = confusion_matrix(labels, y_pred)
                    p, r = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
                    p = np.nan_to_num(p)
                
                old_cur_idx = cur_idx
                p_diff = p[class_idx] - precision_before[class_idx]
                r_diff = r[class_idx] - recall_before[class_idx]
                p_diffs.append(p_diff)
                r_diffs.append(r_diff)
                ed = time.time()
                print(f'time={ed - st}')

            precision_means_dict[sorting_fun_name][j] = np.mean(p_diffs)
            recall_means_dict[sorting_fun_name][j] = np.mean(r_diffs)
            precision_stds_dict[sorting_fun_name][j] = np.std(p_diffs)
            recall_stds_dict[sorting_fun_name][j] = np.std(r_diffs)


            # plt.rcParams.update({'font.size': 17})
            # proper_hist(np.array(p_diffs), bin_size=0.02)
            # plt.savefig('prec_change32_hist.png', format='png')
            # plt.show()

    np.save(f'prec_means_dict{save_postfix}.npy', precision_means_dict)
    np.save(f'recall_means_dict{save_postfix}.npy', recall_means_dict)
    np.save(f'prec_stds_dict{save_postfix}.npy', precision_stds_dict)
    np.save(f'recall_stds_dict{save_postfix}.npy', recall_stds_dict)

    info = {'neuron_sorting_fun_names':neuron_sorting_fun_names, 'n_neurons_to_disable':n_neurons_to_disable}
    np.save(f'info_{save_postfix}.npy', info)

    plt.title('Precision change')
    plt.xlabel('Units ablated')
    # plt.xscale('log')
    plt.xticks(n_neurons_to_disable)
    # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for n in neuron_sorting_fun_names:
        plt.errorbar(n_neurons_to_disable, precision_means_dict[n], yerr=precision_stds_dict[n],
                    label=n, capsize=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'prec_change_{save_postfix}.png', format='png')
    plt.savefig(f'prec_change_{save_postfix}.svg', format='svg')
    plt.show()
    plt.close()

    plt.title('Recall change')
    plt.xlabel('Units ablated')
    # plt.xscale('log')
    plt.xticks(n_neurons_to_disable)
    # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for n in neuron_sorting_fun_names:
        plt.errorbar(n_neurons_to_disable, recall_means_dict[n], yerr=recall_stds_dict[n],
                     label=n, capsize=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rec_change_{save_postfix}.png', format='png')
    plt.savefig(f'rec_change_{save_postfix}.svg', format='svg')
    plt.show()
    plt.close()


def load_and_plot_prec_and_recall_change(dataset_str):
    assert dataset_str in ["imagenet", "cifar10"], f"Unknown dataset_str: {dataset_str}"

    # save_postfixes = [f'layer10_disabled_{i}' for i in [8, 16, 32, 64, 128, 192]]
    save_postfixes = [f'layer6_disabled_{i}' for i in [4, 8, 16, 32, 64, 96]]
    prec_means_all, prec_stds_all, recall_means_all, recall_stds_all = [], [], [], []
    n_neurons_to_disable = []
    for save_postfix in save_postfixes:
        pm = np.load(f'prec_means_{save_postfix}.npy')
        ps = np.load(f'prec_stds_{save_postfix}.npy')
        rm = np.load(f'recall_means_{save_postfix}.npy')
        rs = np.load(f'recall_stds_{save_postfix}.npy')
        info = np.load(f'info_{save_postfix}.npy', allow_pickle=True).item()
        neuron_sorting_fun_names = info['neuron_sorting_fun_names']
        n_neurons_to_disable += info['n_neurons_to_disable']

        prec_means_all.append(pm)
        prec_stds_all.append(ps)
        recall_means_all.append(rm)
        recall_stds_all.append(rs)
    def f(x):
        return np.concatenate(x, axis=1)
    prec_means_all, prec_stds_all, recall_means_all, recall_stds_all = f(prec_means_all), f(prec_stds_all), f(recall_means_all), f(recall_stds_all)
    plt.title('Precision change')
    plt.xlabel('Units ablated')
    plt.xticks(n_neurons_to_disable)
    for i in range(prec_means_all.shape[0]):
        plt.errorbar(n_neurons_to_disable, prec_means_all[i], yerr=prec_stds_all[i],
                     label=neuron_sorting_fun_names[i], capsize=5)
    plt.legend()
    plt.savefig(f'prec_change_combined.png', format='png')
    plt.show()

    plt.title('Recall change')
    plt.xlabel('Units ablated')
    plt.xticks(n_neurons_to_disable)
    for i in range(recall_means_all.shape[0]):
        plt.errorbar(n_neurons_to_disable, recall_means_all[i], yerr=recall_stds_all[i],
                     label=neuron_sorting_fun_names[i], capsize=5)
    plt.legend()
    plt.savefig(f'recall_change_combined.png', format='png')
    plt.show()

def ablate_neurons_based_on_saved_activations(save_model_path, param_file,
                                              # saved_activations_path='activations_on_validation_preserved_spatial_10_pretrained_imagenet.npy',
                                              saved_activations_path='activations_on_validation_14_pretrained_imagenet.npy',
                                              saved_labels_path='labels_pretrained_imagenet.npy', wd_path=None):
    model = reload_model(save_model_path, param_file)
    with open(saved_activations_path, 'rb') as f:
        loaded_activations = torch.tensor(np.load(f))
    acts_avg = loaded_activations.mean(dim=0)

    labels = np.load(saved_labels_path)

    layeri = int(saved_activations_path.split('_')[-3])

    n_neurons_dict = {
        6: 128, 
        10: 256, 
        14: 512
    }

    # ablate_neurons_config = {
    #     256: [8, 16, 32, 64, 128, 192],
    #     512: [16, 32, 64, 128, 256, 384],
    #     128: [4, 8, 16, 32, 64, 96], 
    # }

    # this is for the statistics in the appendix
    ablate_neurons_config = {
        256: np.arange(256),
        512: np.arange(512),
        128: np.arange(128), 
    }

    n_neurons = n_neurons_dict[layeri]
    assert n_neurons == loaded_activations.size()[1], f"n_neurons {n_neurons} doesn't match size: {loaded_activations.size()[1]}"

    ablate_neurons = ablate_neurons_config[n_neurons]

    n_classes = 1000
    precisions_after_disabling = np.zeros((len(ablate_neurons), n_classes))
    recalls_after_disabling = np.zeros((len(ablate_neurons), n_classes))
    acc_diffs = np.zeros(len(ablate_neurons))

    with torch.no_grad():
        y_pred = make_predictions(model, loaded_activations, layeri, if_disable=False)
        conf_matr = confusion_matrix(labels, y_pred)
        precision_before, recall_before = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
        acc_before = conf_matr.diagonal().sum() / conf_matr.sum()

        perm = np.random.RandomState(1).permutation(n_neurons)

        for i, n in enumerate(ablate_neurons):
            print(i)
            # dind = perm[:n]
            dind = n
            y_pred = make_predictions(model, loaded_activations, layeri, if_disable=True, disable_ind=dind)
            conf_matr = confusion_matrix(labels, y_pred)
            precision, recall = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
            precisions_after_disabling[i] = precision
            recalls_after_disabling[i] = recall
            acc_diffs[i] = conf_matr.diagonal().sum() / conf_matr.sum() - acc_before

    # plt.plot(acc_diffs, 'o')
    # plt.ylim(-0.0015, 0.0015)
    # plt.show()

    # plt.plot((recalls_after_disabling - recall_before)[444], 'o')
    # plt.show()

    wasser_dists_values = list(np.load(wd_path, allow_pickle=True).item().values())
    wds = pd.DataFrame(wasser_dists_values).to_numpy().T

    w = model.fc.weight.detach().cpu().numpy()

    rng = np.arange(n_neurons) ; np.random.shuffle(rng)
    maxs = np.array([np.max(list(wd_neuron.values())) for wd_neuron in wasser_dists_values])
    idx = np.argsort(maxs)
    mins = np.array([np.abs(np.min(list(wd_neuron.values()))) for wd_neuron in wasser_dists_values])
    max_recall_drop = (recalls_after_disabling - recall_before).min(axis=-1)
    max_prec_drop = (precisions_after_disabling - precision_before).min(axis=-1)
    print(scipy.stats.pearsonr(maxs, max_recall_drop))
    print(scipy.stats.pearsonr(maxs, max_prec_drop))
    print(scipy.stats.pearsonr(mins, max_recall_drop))
    print(scipy.stats.pearsonr(mins, max_prec_drop))
    # l1_norms_dict = np.load(f'l1_norms_{model_name_short}.npy', allow_pickle=True).item()
    # l1s = l1_norms_dict['layer4.1.conv2.weight']
    # l1s = l1s.values()
    # l1s = np.array(list(l1s))
    # idx = np.argsort(l1s)

    # np.save('prec_rec_10.npy',
    #         (precision_before, recall_before, precisions_after_disabling, recalls_after_disabling, acc_diffs),
    #         allow_pickle=True)
    exit()
    accs_cumulative_ablation = np.zeros(n_neurons)
    with torch.no_grad():
        for i in range(0, n_neurons):
            print(i)
            x = loaded_activations.clone()
            x[:, idx[:i]] = 0#acts_avg[:i]#
            x = model.fc(x)
            y_pred = torch.argmax(x, dim=1).detach().cpu().numpy()
            conf_matr = confusion_matrix(labels, y_pred)
            accs_cumulative_ablation[i] = conf_matr.diagonal().sum() / conf_matr.sum()
    plt.plot(accs_cumulative_ablation)
    accs_cumulative_ablation = np.zeros(n_neurons)
    with torch.no_grad():
        for i in range(0, n_neurons):
            print(i)
            x = loaded_activations.clone()
            x[:, idx[:i]] = acts_avg[idx[:i]]#
            x = model.fc(x)
            y_pred = torch.argmax(x, dim=1).detach().cpu().numpy()
            conf_matr = confusion_matrix(labels, y_pred)
            accs_cumulative_ablation[i] = conf_matr.diagonal().sum() / conf_matr.sum()
    plt.plot(accs_cumulative_ablation)
    plt.show()

    imagenet_dict_reversed = dict(zip(imagenet_dict.values(), imagenet_dict.keys()))

    idx, wdists = sort_neurons_for_class(wasser_dists_values, 689)

    class_idx1 = 11
    idx, wdists = sort_neurons_for_class(wasser_dists_values, class_idx1)
    x = loaded_activations.clone()
    cur_idx = list(idx[:8])  # + list(idx[-5:])#idx[0]
    x[:, cur_idx] = 0  # acts_avg[cur_idx]
    x = model.fc(x)
    conf_matr = confusion_matrix(labels, torch.argmax(x, dim=1).detach().cpu().numpy())
    p, r = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
    class_idx2 = class_idx1
    print(f'{precision_before[class_idx2]:.3f} {p[class_idx2]:.3f} \n{recall_before[class_idx2]:.3f} {r[class_idx2]:.3f}')
    print(w[class_idx2, cur_idx])

    y = 1

if __name__ == '__main__':
    # load_and_plot_prec_and_recall_change()
    # exit()
    
    # #   single-head cifar
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    # model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
        save_model_path = 'dummy'
        param_file = 'dummy'
    
    try:
        l1_norms_dict = np.load(f'l1_norms_{model_name_short}.npy', allow_pickle=True).item()
    except:
        calc_l1_norms_of_weights(save_model_path)
        l1_norms_dict = np.load(f'l1_norms_{model_name_short}.npy', allow_pickle=True).item()
    
    usual_layer_idx_to_l1_norms_idx = { 0:'layer1.0.conv2.weight',
                                        1:'layer1.1.conv1.ordinary_conv.weight',
                                        2:'layer1.1.conv2.ordinary_conv.weight',
                                        3:'layer2.0.conv1.ordinary_conv.weight',
                                        4:'layer2.0.conv2.ordinary_conv.weight',
                                        5:'layer2.1.conv1.ordinary_conv.weight',
                                        6:'layer2.1.conv2.ordinary_conv.weight',
                                        7:'layer3.0.conv1.ordinary_conv.weight',
                                        8:'layer3.0.conv2.ordinary_conv.weight',
                                        9:'layer3.1.conv1.ordinary_conv.weight',
                                        10:'layer3.1.conv2.ordinary_conv.weight',
                                        11:'layer4.0.conv1.ordinary_conv.weight',
                                        12:'layer4.0.conv2.ordinary_conv.weight',
                                        13:'layer4.1.conv1.ordinary_conv.weight',
                                        14:'layer4.1.conv2.ordinary_conv.weight'}
    
    path_prefix = 'stored_activations/'
    for layeri in [14]: #[6, 10, 14]:
        # based_on = "_shift"
        based_on = "_dist"
        # ablate_neurons_based_on_saved_activations(save_model_path, param_file,
        #                 saved_activations_path=f'{path_prefix}activations_on_validation_preserved_spatial_{layeri}_pretrained_imagenet.npy',
        #                 wd_path=f'wasser_dists/wasser{based_on}_attr_hist_pretrained_imagenet_afterrelu_test_{layeri}.npy')
        # exit()
        
        plot_prec_recall_change(save_model_path, param_file,
                                # saved_activations_path=f'activations_on_validation_{layeri}_pretrained_imagenet.npy',
                                saved_activations_path=f'{path_prefix}activations_on_validation_preserved_spatial_{layeri}_pretrained_imagenet.npy',
                                wd_path=f'wasser_dists/wasser{based_on}_attr_hist_pretrained_imagenet_afterrelu_test_{layeri}.npy',
                                l1_norms=list(l1_norms_dict[usual_layer_idx_to_l1_norms_idx[layeri].replace('ordinary_conv.', '')].values()))
    exit()
    # plot_ablation_strategy_comparisons(model_name_short, ['sum', 'max', 'max+min', 'min'], range(15))
    # exit()
    # areas = create_areas_array(model_name_short, ['sum', 'max', 'max+min', 'min'], range(15))
    # exit()
    n_start = 0
    # layer_ind = 10
    for layer_ind in [14, 13, 12]:#range(12):#[7, 8, 9, 10]:#
        n_steps = 51
        step_size = 5
        if layer_ind <= 2:
            step_size = 3
            n_steps = 22
        if layer_ind >= 11:
            step_size = 10
        # wd_path = 'wasser_dists/wasser_dist_attr_hist_bettercifar10single_' + str(layer_ind) + '.npy'
        wd_path = 'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_' + str(layer_ind) + '.npy'
        plt.title(f'{layer_ind}')
        wasser_dists_values = np.load(wd_path, allow_pickle=True).item().values()
        compare_ablation_strategies(save_model_path, param_file, step_size, n_steps, layer_ind, n_start,
                [
                    # (lambda wd_neuron: np.sum(np.abs(list(wd_neuron.values()))), 'sum', wasser_dists_values),
                    # (lambda wd_neuron: np.max(list(wd_neuron.values())), 'max', wasser_dists_values),
                    # # lambda vals: np.max(vals) + (np.abs(np.min(vals)) if np.min(vals) < -0.1 else 0),
                    # (lambda wd_neuron: np.max(list(wd_neuron.values())) + np.abs(np.min(list(wd_neuron.values()))), 'max+min', wasser_dists_values),
                    # (lambda wd_neuron: np.abs(np.min(list(wd_neuron.values()))), 'min', wasser_dists_values),
                    # (lambda wd_neuron: np.random.rand(), 'rand', wasser_dists_values),
                    # (lambda l1_norm: l1_norm, 'l1', l1_norms_dict[usual_layer_idx_to_l1_norms_idx[layer_ind].replace('ordinary_conv.', '')].values()),
                    # (lambda wd_neuron: np.random.rand(), 'rand2', wasser_dists_values),
                    # (lambda wd_neuron: np.random.rand(), 'rand3', wasser_dists_values),
                    # (lambda wd_neuron: np.random.rand(), 'rand4', wasser_dists_values),
                    (lambda wd_neuron: np.max(np.abs(list(wd_neuron.values()))), 'absmax', wasser_dists_values),
                ], model_name_short)

    # if True:
    #     sorted_neurons = select_neurons_to_ablate_by_wasserstein_dist(wd_path, selection_fun=lambda vals:np.max(vals))
    # else:
    #     sorted_neurons = list(range(512))
    #     np.random.shuffle(sorted_neurons)
    # accs = calc_accuracies_after_ablation(param_file, trained_model, layer_ind, sorted_neurons, step_size, n_steps)
    # plt.plot(range(0, n_steps * step_size, step_size), accs)
    # plt.show()

'''
w = model.fc.weight.detach().cpu().numpy()

imagenet_dict_reversed = dict(zip(imagenet_dict.values(), imagenet_dict.keys()))

wasser_dists_values = np.load(wd_path, allow_pickle=True).item().values()
wasser_dists_values = list(np.load(wd_path, allow_pickle=True).item().values())

def sort_neurons_for_class(wasser_dists_values, target_class):
    w_dists = np.array([w_dict[target_class] for w_dict in wasser_dists_values])
    idx = np.argsort(w_dists)
    return idx, w_dists


idx, wdists = sort_neurons_for_class(wasser_dists_values, 689)
print(wdists[idx[:5]])

x = loaded_activations.clone()
x[:, idx[:16]] = 0
x = model.fc(x)
y_pred = torch.argmax(x, dim=1).detach().cpu().numpy()
conf_matr = confusion_matrix(labels, y_pred)
p, r = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
class_idx = 689
print(precision_before[class_idx], p[class_idx])
print(recall_before[class_idx], r[class_idx])
'''
