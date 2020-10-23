from pathlib import Path

import torch
import torchvision
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from util.util import *
from scipy.integrate import simps
from multi_task.load_model import eval_trained_model, load_trained_model

if_pretrained_imagenet = True
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

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
        # np.save(f'ablations/{folder_name}/accs_{fun_name}_{layer_ind}.npy', (accs, area), allow_pickle=True)
    plt.legend()
    # plt.savefig(f'ablations/{folder_name}/l_{layer_ind}.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.show()


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
    step_size = 100
    n_steps = 51
    n_start = 0
    # layer_ind = 10
    for layer_ind in [13]:
        # if layer_ind > 11:
        #     step_size = 10
        # wd_path = 'wasser_dists/wasser_dist_attr_hist_bettercifar10single_' + str(layer_ind) + '.npy'
        wd_path = 'wasser_dists/wasser_dist_attr_hist_pretrained_imagenet_afterrelu_' + str(layer_ind) + '.npy'
        plt.title(f'{layer_ind}')
        wasser_dists_values = np.load(wd_path, allow_pickle=True).item().values()
        compare_ablation_strategies(save_model_path, param_file, step_size, n_steps, layer_ind, n_start,
                [
                    (lambda wd_neuron: np.sum(np.abs(list(wd_neuron.values()))), 'sum', wasser_dists_values),
                    (lambda wd_neuron: np.max(list(wd_neuron.values())), 'max', wasser_dists_values),
                    # lambda vals: np.max(vals) + (np.abs(np.min(vals)) if np.min(vals) < -0.1 else 0),
                    (lambda wd_neuron: np.max(list(wd_neuron.values())) + np.abs(np.min(list(wd_neuron.values()))), 'max+min', wasser_dists_values),
                    (lambda wd_neuron: np.random.rand(), 'rand', wasser_dists_values),
                    (lambda wd_neuron: np.abs(np.min(list(wd_neuron.values()))), 'min', wasser_dists_values),
                    # (lambda l1_norm: l1_norm, 'l1', l1_norms_dict[usual_layer_idx_to_l1_norms_idx[layer_ind]].values()),
                ], model_name_short)

    # if True:
    #     sorted_neurons = select_neurons_to_ablate_by_wasserstein_dist(wd_path, selection_fun=lambda vals:np.max(vals))
    # else:
    #     sorted_neurons = list(range(512))
    #     np.random.shuffle(sorted_neurons)
    # accs = calc_accuracies_after_ablation(param_file, trained_model, layer_ind, sorted_neurons, step_size, n_steps)
    # plt.plot(range(0, n_steps * step_size, step_size), accs)
    # plt.show()
