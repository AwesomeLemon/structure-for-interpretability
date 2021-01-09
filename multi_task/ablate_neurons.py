import json
import pickle

from pathlib import Path

import torch
import torchvision
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
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

def make_predictions(model, loaded_activations, if_disable, disable_ind=None):
    with torch.no_grad():
        acts = loaded_activations.clone().cuda()
        if if_disable:
            acts[:, disable_ind] = 0  # 10 * acts_avg[i]#

        y_pred = []
        n_batches = 10
        n_samples = acts.shape[0]
        batch_size = int(np.ceil(n_samples / n_batches))
        for i in range(n_batches):
            x = acts[i * batch_size:(i+1) * batch_size, :]
            # x = model.layer4(x)
            # x = model.avgpool(x)
            # x = torch.flatten(x, 1)

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
                            saved_labels_path='labels_pretrained_imagenet.npy', wd_path=None):

    model = reload_model(save_model_path, param_file)
    with open(saved_activations_path, 'rb') as f:
        loaded_activations = torch.tensor(pickle.load(f))
    labels = np.load(saved_labels_path)
    wasser_dists_values = list(np.load(wd_path, allow_pickle=True).item().values())

    y_pred = make_predictions(model, loaded_activations, if_disable=False)
    conf_matr = confusion_matrix(labels, y_pred)
    precision_before, recall_before = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)

    n_neurons = 512
    n_classes = 1000
    n_neurons_to_disable = [32]#[1, 4, 8, 16, 32, 64, 128]#[16, 32]#[8, 16]#
    neuron_sorting_funs = [lambda wasser_dists_values, idx: np.random.permutation(n_neurons),
                            lambda wasser_dists_values, idx: np.array(sort_neurons_for_class(wasser_dists_values, idx)[0][::-1]),
                            lambda wasser_dists_values, idx: sort_neurons_for_class(wasser_dists_values, idx)[0],
                           ]
    neuron_sorting_fun_names = ['random',
                                'sort-max',
                                'sort-min'
                                ]
    precision_means = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
    recall_means = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
    precision_stds = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
    recall_stds = np.zeros((len(neuron_sorting_funs), len(n_neurons_to_disable)))
    for i, sorting_fun in enumerate(neuron_sorting_funs):
        for j, n_to_disable in enumerate(n_neurons_to_disable):
            print(f'{n_to_disable} disabled')
            p_diffs = []
            r_diffs = []
            for class_idx in range(n_classes):
                print(class_idx)
                cur_idx = list(sorting_fun(wasser_dists_values, class_idx)[:n_to_disable])
                # x = loaded_activations.clone().cuda()
                # x[:, cur_idx] = 0
                # x = model.fc(x)
                # conf_matr = confusion_matrix(labels, torch.argmax(x, dim=1).detach().cpu().numpy())
                y_pred = make_predictions(model, loaded_activations, if_disable=True, disable_ind=cur_idx)
                conf_matr = confusion_matrix(labels, y_pred)
                p, r = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
                p = np.nan_to_num(p)
                p_diff = p[class_idx] - precision_before[class_idx]
                r_diff = r[class_idx] - recall_before[class_idx]
                p_diffs.append(p_diff)
                r_diffs.append(r_diff)

            precision_means[i, j] = np.mean(p_diffs)
            recall_means[i, j] = np.mean(r_diffs)
            precision_stds[i, j] = np.std(p_diffs)
            recall_stds[i, j] = np.std(r_diffs)

            # plt.rcParams.update({'font.size': 17})
            # proper_hist(np.array(p_diffs), bin_size=0.02)
            # plt.savefig('prec_change32_hist.png', format='png')
            # plt.show()

    plt.title('Precision change')
    plt.xlabel('Units ablated')
    # plt.xscale('log')
    plt.xticks(n_neurons_to_disable)
    # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for i in range(len(neuron_sorting_funs)):
        plt.errorbar(n_neurons_to_disable, precision_means[i], yerr=precision_stds[i],
                     label=neuron_sorting_fun_names[i], capsize=5)
    plt.legend()
    plt.savefig(f'prec_change.png', format='png')
    plt.show()

    plt.title('Recall change')
    plt.xlabel('Units ablated')
    # plt.xscale('log')
    plt.xticks(n_neurons_to_disable)
    # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for i in range(len(neuron_sorting_funs)):
        plt.errorbar(n_neurons_to_disable, recall_means[i], yerr=recall_stds[i],
                     label=neuron_sorting_fun_names[i], capsize=5)
    plt.legend()
    plt.savefig(f'rec_change.png', format='png')
    plt.show()
    np.save('prec_means2.npy', precision_means)
    np.save('recall_means2.npy', recall_means)
    np.save('prec_stds2.npy', precision_stds)
    np.save('recall_stds2.npy', recall_stds)


def ablate_neurons_based_on_saved_activations(save_model_path, param_file,
                                              # saved_activations_path='activations_on_validation_preserved_spatial_10_pretrained_imagenet.npy',
                                              saved_activations_path='activations_on_validation_14_pretrained_imagenet.npy',
                                              saved_labels_path='labels_pretrained_imagenet.npy', wd_path=None):
    model = reload_model(save_model_path, param_file)
    with open(saved_activations_path, 'rb') as f:
        loaded_activations = torch.tensor(pickle.load(f))
    acts_avg = loaded_activations.mean(dim=0)

    labels = np.load(saved_labels_path)

    n_neurons = 256#512#256#
    n_classes = 1000
    precisions_after_disabling = np.zeros((n_neurons, n_classes))
    recalls_after_disabling = np.zeros((n_neurons, n_classes))
    acc_diffs = np.zeros(n_neurons)

    with torch.no_grad():
        y_pred = make_predictions(model, loaded_activations, if_disable=False)
        conf_matr = confusion_matrix(labels, y_pred)
        precision_before, recall_before = conf_matr.diagonal() / conf_matr.sum(axis=0), conf_matr.diagonal() / conf_matr.sum(axis=1)
        acc_before = conf_matr.diagonal().sum() / conf_matr.sum()
        for i in range(n_neurons):
            print(i)
            y_pred = make_predictions(model, loaded_activations, if_disable=True, disable_ind=i)
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
    #   single-head cifar
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
    param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    if if_pretrained_imagenet:
        model_name_short = 'pretrained_imagenet'
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
    # calc_l1_norms_of_weights(save_model_path)
    # exit()
    # ablate_neurons_based_on_saved_activations(save_model_path, param_file,
    #                 saved_activations_path='activations_on_validation_preserved_spatial_10_pretrained_imagenet.npy',
    #                 wd_path='wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_prerelu_' + str(10) + '.npy')
    # exit()
    plot_prec_recall_change(save_model_path, param_file,
                            saved_activations_path='activations_on_validation_14_pretrained_imagenet.npy',
                            wd_path='wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_test_' + str(14) + '.npy')
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
