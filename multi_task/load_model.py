import json
from collections import defaultdict

# import multi_task.datasets as datasets
# import multi_task.metrics as metrics
# import multi_task.model_selector_automl as model_selector_automl
import datasets as datasets
import metrics as metrics
import model_selector_automl as model_selector_automl
import numpy as np
import pandas
import torch
import torchvision
from torch.autograd import Variable
from util.util import celeba_dict
from matplotlib import pyplot as plt
from util.util import proper_hist
from util.util import get_relevant_labels_from_batch
import scipy.stats

device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")


def disable_conn(state, ind, src, dest):
    state['connectivities'][ind][dest, src] = 0


def load_trained_model(param_file, save_model_path, if_restore_connectivities=True,
                       if_visuzalization_conns=False, if_replace_useless_conns_with_bias=False,
                       if_enable_bias=False, if_replace_useless_conns_with_additives=False,
                       if_additives_user=False, replace_constants_last_layer_mode=None,
                       if_store_avg_activations_for_disabling=False, conns_to_remove_dict=None,
                       replace_with_avgs_last_layer_mode=None, if_actively_disable_bias=False,
                       if_replace_by_zeroes=False):
    # if_actively_disable_bias is needed because some cifar networks ignored the enable_bias parameter
    assert not (if_replace_useless_conns_with_additives and if_replace_useless_conns_with_bias
                and if_store_avg_activations_for_disabling)  # only 1 of those can be true

    with open(param_file) as json_params:
        params = json.load(json_params)

    if if_visuzalization_conns:
        viz_conns_state = torch.load('visualized_connectivities.pkl')
        auxillary_connectivities_for_id_shortcut = viz_conns_state['auxillary_connectivities_for_id_shortcut']

        params['this_is_graph_visualization_run'] = 'Yep'
        if True:
            params['auxillary_connectivities_for_id_shortcut'] = list(
                map(lambda x: torch.Tensor(x).cuda() if x is not None else None,
                    auxillary_connectivities_for_id_shortcut)) + [None, None, None]
        else:
            params['auxillary_connectivities_for_id_shortcut'] = [None] * 15

    if if_replace_useless_conns_with_bias:
        params['if_replace_useless_conns_with_bias'] = 'Yep'
    if if_replace_useless_conns_with_additives:
        params['if_replace_useless_conns_with_additives'] = 'Yep'
    if if_store_avg_activations_for_disabling:
        params['if_store_avg_activations_for_disabling'] = 'Yep'
    if if_additives_user:
        params['if_additives_user'] = 'Yep'

    if if_enable_bias:
        params['if_enable_bias'] = True
    if if_actively_disable_bias:
        params['if_enable_bias'] = False

    params['replace_constants_last_layer_mode'] = replace_constants_last_layer_mode
    params['replace_with_avgs_last_layer_mode'] = replace_with_avgs_last_layer_mode

    model = model_selector_automl.get_model(params)

    tasks = params['tasks']

    state = torch.load(save_model_path)

    model_rep_state = state['model_rep']
    # print('ACHTUNG! messing with model["rep"] state (because I"m fixing the shortcut connection not being a MaskConv2d')
    # model_rep_state['layer3.0.shortcut.0.ordinary_conv.weight'] = model_rep_state['layer3.0.shortcut.0.weight']
    # del model_rep_state['layer3.0.shortcut.0.weight']
    # model_rep_state['layer4.0.shortcut.0.ordinary_conv.weight'] = model_rep_state['layer4.0.shortcut.0.weight']
    # del model_rep_state['layer4.0.shortcut.0.weight']

    # state['connectivities'][-4][215, 71] = 0.
    # state['connectivities'][-4][215, 103] = 0.

    # state['connectivities'][-3][15, 79] = 0.
    # state['connectivities'][-3][15, 282] = 0.

    # state['connectivities'][-2][481, 214] = 0.
    # state['connectivities'][-2][481, 385] = 0.

    # state['connectivities'][-1][31, 57] = 0.
    # state['connectivities'][-1][31, 279] = 0. #responsible only for pale skin & smiling, but makes viz worse
    # state['connectivities'][-1][31, 126] = 0. # responsible for blond hair, overpowers smile
    # state['connectivities'][-1][31, 331] = 0. # responsible for something black, overpowers smile
    # state['connectivities'][-1][31, 261] = 0.

    # state['connectivities'][-2][148, 15] = 1.
    # state['connectivities'][-1][:, 148] = 0.
    # state['connectivities'][-1][12, 148] = 1.

    if False:
        print('Some connections are disabled')
        # black hair:
        # the 2 below are the most important ones
        # disable_conn(state, 9, 142, 188)
        # disable_conn(state, 10, 86, 104)

        # disable_conn(state, 1, 27, 50)
        # disable_conn(state, 1, 35, 50)
        # # state['connectivities'][2][28, 16] = 0
        # disable_conn(state, 2, 35, 76)
        # disable_conn(state, 3, 76, 96)
        # disable_conn(state, 3, 101, 96)
        # disable_conn(state, 3, 113, 96)
        # disable_conn(state, 6, 29, 160)
        # the one below by itself makes p(black_hair)=1 before&after
        # disable_conn(state, 6, 96, 160)
        # # state['connectivities'][6][248, 124] = 0
        # disable_conn(state, 7, 82, 53)
        # # state['connectivities'][7][106, 248] = 0
        # # if True:
        # #     state['connectivities'][9][188, 216] = 0
        # # else:
        # #     state['connectivities'][8][216, 149] = 0
        # #     state['connectivities'][8][216, 0] = 0
        # #     state['connectivities'][8][216, 203] = 0
        # #     state['connectivities'][8][216, 106] = 0
        # #     state['connectivities'][8][216, 31] = 0
        # #     state['connectivities'][8][216, 25] = 0

        if False:
            if False:
                disable_conn(state, 8, 23, 224)
            else:
                # disable_conn(state, 7, 241, 23)
                disable_conn(state, 7, 86, 23)

            if False:
                disable_conn(state, 9, 174, 86)
            else:
                disable_conn(state, 8, 149, 174)

        # disable_conn(state, 9, 187, 86)
        # state['connectivities'][8][86, :] = 0
        # state['connectivities'][9][86, :] = 0
        # state['connectivities'][9][86, 224] = 1
        '''
        1
        disabling 188->104: same mediocre drop in pb & pa
        disabling 160->104 or 86->104: huge drop; equal p(black) in pb & pa
        '''
        disable_conn(state, 10, 188, 104)
        # disable_conn(state, 10, 160, 104)
        # disable_conn(state, 10, 86, 104)

        # disable_conn(state, 9, 174, 86)
        # disable_conn(state, 9, 187, 86)
        # disable_conn(state, 9, 224, 86)

        # disable_conn(state, 13, 497, 172)
        #
        # state['connectivities'][14][8, :] = 0
        # state['connectivities'][14][8, 383] = 0
        # state['connectivities'][14][8, 172] = 0
        # state['connectivities'][14][8, 400] = 1

        # disable_conn(state, 14, 134, 9)
        # disable_conn(state, 14, 400, 9)
        # disable_conn(state, 14, 383, 9)

        # disable_conn(state, 10, 181, 279)
        # disable_conn(state, 10, 73, 279)
        # disable_conn(state, 10, 191, 279)

        # disable_conn(state, 12, 406, 204)
        # disable_conn(state, 10, 193, 125)
    # brown hair:
    # state['connectivities'][14][11, :] = 0
    # state['connectivities'][14][11, 383] = 1
    # blond hair:
    # # print(torch.where(state['connectivities'][7][83, :] > 0.5))
    # disable_conn(state, 1, 47, 37)
    # # disable_conn(state, 6, 42, 57)
    # # disable_conn(state, 6, 57, 57)
    # # disable_conn(state, 6, 105, 57)
    # disable_conn(state, 7, 195, 83)
    # # disable_conn(state, 7, 57, 83)
    # disable_conn(state, 10, 181, 490)
    # disable_conn(state, 11, 279, 123)
    # # disable_conn(state, 14, 123, 9)
    # # disable_conn(state, 14, 126, 9)
    # # disable_conn(state, 14, 187, 9)
    # # disable_conn(state, 14, 204, 9)
    # # disable_conn(state, 14, 356, 9)
    # state['connectivities'][14][9, :] = 0
    # state['connectivities'][14][9, 123] = 1

    if hasattr(model['rep'], 'connectivities') and if_restore_connectivities:
        # print('ACHTUNG! Trying out restoring connectivities')
        for i, conn in enumerate(state['connectivities']):
            if not if_visuzalization_conns:
                model['rep'].connectivities[i].data = state['connectivities'][i].data
            else:
                model['rep'].connectivities[i].data = torch.Tensor(viz_conns_state['connectivities'][i]).cuda().data

    # model['rep'].connectivities[0].data *= 0
    '''
    if mutliply val_rep by 0, get 0.32
    if multiply conn[-1] by 0, get 0.32
    if multiply conn[-3] by 0, get 0.25
    if multiply conn[0] by 0, get 0.25
    
    I think this is simply explained by batch norm. I've checked with conn[0]*=0: before batchnorm indeed 0, after - non-zero
    '''
    # model['rep'].connectivities[-1][3].data *= 0 # Bags under eyes - appears disconnected in my graph vizualization (i.e. all incoming connections
    #         are useless. Well, let's test this.
    # model['rep'].connectivities[-1][17].data *= 0 # same with gray hair
    '''
    both get worse (while all the others stay the same)
    Task = 3, acc = 0.8259928524689183   -------> Task = 3, acc = 0.2074294055468868
    Task = 17, acc = 0.9607892485025419  -------> Task = 17, acc = 0.9513263200281874
    '''

    # model['rep'].connectivities[-5][389, 126] = False
    '''
    are affected (per graph): ['5_o_Clock_Shadow', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Blurry', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mustache', 'Oval_Face', 'Pale_Skin', 'Receding_Hairline', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie']
    didn't change, although should've: Bags_Under_Eyes, Bald, Blurry, Eyeglasses, Mustache, Receding_Hairline, Wearing_Hat, Wearing_Necktie
    changed, although shouldn't have:
    improved (?!): Big_Lips
    '''
    # model['rep'].connectivities[-1][:, 173] = False # behaved exactly as expected
    # model['rep'].connectivities[-5][410, 121] = False # 9 things changed, all 9 should've. But Wearing_Hat also should've, but didn't
    # => disabling 14_ that leads to Wearing_Hat along the disabled path shouldn't change the hat's performance:
    # model['rep'].connectivities[-1][:, 138] = False # ... and it doesn't.

    # Next thing: disable everything for Hat, get performance => it turns out to be the same (i.e. connections don't matter)
    # but connections are far from 0.5:
    # print(model['rep'].connectivities[-1][35, :])

    # model['rep'].connectivities[-1][:, 435] = 0.

    if "layer1.1.conv1.ordinary_conv.weight_mask" in model_rep_state:
        # this is the pruned model that wasn't stored properly
        for k, v in list(model_rep_state.items()):
            if 'weight_mask' in k:
                mask = v
                param_name = k[:k.find('weight_mask')]
                weight_orig = model_rep_state[param_name + 'weight_orig']
                pruned_weight = mask * weight_orig
                model_rep_state[param_name + 'weight'] = pruned_weight
                del model_rep_state[k]
                del model_rep_state[param_name + 'weight_orig']

    model['rep'].load_state_dict(model_rep_state)
    for i in range(3):
        # apparently learnings scales & biases are saved automatically as part of the model state
        pass
        # model['rep'].lin_coeffs_id_zero[i] = state[f'learning_scales{i}']
        # model['rep'].bn_biases[i] = state[f'bn_bias{i}']

    if if_replace_useless_conns_with_additives:
        if 'additives' in state:
            model['rep'].block[0].additives_dict = state['additives']
    if if_store_avg_activations_for_disabling:
        if if_additives_user:
            model['rep'].block[0].additives_dict = state['additives']
            if conns_to_remove_dict is None:
                raise ValueError('conns_to_remove_dict not initialized')
            model['rep'].block[0].connections_to_remove = conns_to_remove_dict
            if if_replace_by_zeroes:
                model['rep'].block[0].replace_with_zeroes = True

    if replace_constants_last_layer_mode == 'restore':
        model['rep'].last_layer_additives = state['last_layer_additives']
    if replace_with_avgs_last_layer_mode == 'restore':
        model['rep'].last_layer_additives = state['last_layer_additives']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        cur_state = state[key_name]
        if "linear.weight_mask" in cur_state:
            # this is the pruned model that wasn't stored properly
            for k, v in list(cur_state.items()):
                if 'weight_mask' in k:
                    mask = v
                    param_name = k[:k.find('weight_mask')]
                    weight_orig = cur_state[param_name + 'weight_orig']
                    pruned_weight = mask * weight_orig
                    cur_state[param_name + 'weight'] = pruned_weight
                    del cur_state[k]
                    del cur_state[param_name + 'weight_orig']
        model[t].load_state_dict(cur_state)

    # print(model['35'].linear.weight)
    # np.where(model['rep'].connectivities[-1][35, :].cpu().detach().numpy() > 0.5)[0]
    # np.where(np.abs(model['35'].linear.weight.cpu().detach().numpy()[0]) > 0.1)[0]

    # model['12'].linear.weight[:, 148] = torch.tensor([-0.06, 0.05#-0.8863,  0.8848
    #                                                   ]).cuda()
    # model['12'].linear.bias.data = torch.tensor([ 0.0676, -0.0725]).cuda() # first number is no, second number is yes.
    # I.e. if first is 1000 we get 0 predictions, and when second is -1000 we get 0 predictions

    #model['21'].linear.weight[:, [295, 356, 363, 481, 126, 23]]
    #model['8'].linear.weight[:, [172, 400, 383]]
    #model['38'].linear.weight[:, [250, 503, 56, 103, 97, 204]]
    #x = model['1'].linear.weight[:, 6].cpu().detach() ; x[1] - x[0]
    #x = np.array([.24, -.07, -.1, -.09, -.11, -.12, -.08, -.06, .24, .13])
    #y = model['all'].linear.weight[:, 508].cpu().detach().numpy()
    #np.corrcoef(x, y), scipy.stats.spearmanr(x, y)
    '''
    w = model['all'].linear.weight.cpu().detach().numpy()
    wasser_dists = np.load('wasser_dist_attr_hist_bettercifar10single_14.npy', allow_pickle=True).item()
    corrs = []
    for neuron in range(512):
        w_cur = w[:, neuron]
        dist_cur = np.array(list(wasser_dists[neuron].values()))
        corrs.append(np.corrcoef(w_cur, dist_cur)[0, 1]) #scipy.stats.spearmanr(w_cur, dist_cur)[0]
    '''
    '''
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            m1 = wds[i] > 0.1
            m2 = wds[j] < -0.1
            m_res = np.logical_and(m1, m2)
            print(i, j, len(np.where(m_res)[0]))
    '''
    for m in model:
        model[m].to(device)

    return model


def eval_trained_model(param_file, model, if_store_val_pred=False, save_model_path=None, loader=None,
                       if_stop_after_single_batch=False, batch_size=None, if_print=True, if_pretrained_imagenet=False):
    with open(param_file) as json_params:
        params = json.load(json_params)

    params['metric_type'] = 'ACC_BLNCD'
    if batch_size is not None:
        params['batch_size'] = batch_size
    if 'input_size' not in params:
        params['input_size'] = 'default'
    if params['input_size'] == 'default':
        config_path = 'configs.json'
    elif params['input_size'] == 'bigimg':
        config_path = 'configs_big_img.json'
    elif params['input_size'] == 'biggerimg':
        config_path = 'configs_bigger_img.json'

    if if_pretrained_imagenet:
        params['dataset'] = 'imagenet_val'
        params['batch_size'] = 256 #* 2
        params['tasks'] = ['all']
        params['metric_type'] = 'ACC'
        # model = torch.nn.DataParallel(model)
        m_tmp = {} # model should be a dict
        m_tmp['rep'] = model
        model = m_tmp

    with open(config_path) as config_params:
        configs = json.load(config_params)

    metric = metrics.get_metrics(params)

    if loader is None:
        if not if_store_val_pred:
            tst_loader = datasets.get_test_dataset(params, configs)
            loader = tst_loader
        else:
            _, val_loader, _ = datasets.get_dataset(params, configs)
            loader = val_loader
            preds = [[] for _ in range(40)]
            model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]

    if True:
        tasks = params['tasks']
    else:
        tasks = [str(x) for x in range(40) if x != 12]

    if if_pretrained_imagenet:
        all_tasks = ['all']
    else:
        all_tasks = configs[params['dataset']]['all_tasks']

    for m in model:
        model[m].eval()

    if_train_default_resnet = 'vanilla' in params['architecture']

    with torch.no_grad():
        # print('ACHTUNG! Setting first connectivities to 0 (because they weren't saved)')
        # model['rep'].connectivities[0] *= 0
        # model['rep'].connectivities[1] *= 0
        for i, batch_val in enumerate(loader):
            if i % 10 == 0:
                if if_print:
                    print(i)
            if if_stop_after_single_batch and (i == 1):
                break

            val_images = batch_val[0].cuda()
            labels_val = get_relevant_labels_from_batch(batch_val, all_tasks, tasks, params, device)

            # val_reps, _ = model['rep'](val_images, None)
            val_reps = model['rep'](val_images)
            # for j, t in enumerate(tasks):
            for t in tasks:
                try:
                    j = int(t)
                except: # assume single-task cifar
                    j = 0
                    # print('j = 0')
                '''problem: suppose I trained 40 tasks, but wanna evaulate only 39. Then enumerate breaks stuff
                problem: suppose I trained some arbitrary tasks (e.g. [31, 36]). Then int() breaks stuff
                Choose your poison carefully.'''
                val_rep = val_reps if if_train_default_resnet or if_pretrained_imagenet else val_reps[j]
                if not if_pretrained_imagenet:
                    out_t_val, _ = model[t](val_rep, None)
                else:
                    out_t_val = val_reps
                # loss_t = loss_fn[t](out_t_val, labels_val[t])
                # tot_loss['all'] += loss_t.item()
                # tot_loss[t] += loss_t.item()
                metric[t].update(out_t_val, labels_val[t])
                if if_store_val_pred:
                    labels = out_t_val.data.max(1, keepdim=True)[1]
                    temp = list(labels.detach().cpu().numpy().squeeze())
                    preds[j] += temp
                # print(out_t_val)
                # print(labels_val[t])
                # print(metric[t].get_result())
    if if_store_val_pred:
        preds = np.array(preds).T
        preds[preds == 0] = -1
        if if_print:
            print(preds.shape)
        df = pandas.DataFrame({name: preds[:, i] for i, name in enumerate(celeba_dict.values())})
        df.to_csv(f'predicted_labels_celeba_{model_name_short}.csv', sep=' ')

    error_sums = defaultdict(lambda: 0)
    for t in tasks:
        metric_results = metric[t].get_result()

        for metric_key in metric_results:
            try:
                name = celeba_dict[int(t)]
            except:
                # assume single-head cifar
                name = t
            if if_print:
                print(f'({t}) {name}\t acc = {metric_results[metric_key]}'.expandtabs(30))
            error_sums[metric_key] += 1 - metric_results[metric_key]

        metric[t].reset()

    for metric_key in metric_results:
        error_sum = error_sums[metric_key]
        error_sum /= float(len(tasks))
        last_error = error_sum
        if if_print:
            print('error', metric_key, error_sum * 100)

    return last_error, loader

def convert_useless_connections_to_biases(param_file, save_model_path):
    model = load_trained_model(param_file, save_model_path, True, False, True)
    eval_trained_model(param_file, model)

    state = {'model_rep': model['rep'].state_dict(),
             'connectivities': model['rep'].connectivities}
    with open(param_file) as json_params:
        params = json.load(json_params)
    tasks = params['tasks']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        state[key_name] = model[t].state_dict()
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_biased' + '.pkl'
    torch.save(state, new_save_model_path)


def convert_useless_connections_to_additives(param_file, save_model_path):
    model = load_trained_model(param_file, save_model_path, True, False, False, False, True, False, 'store')
    eval_trained_model(param_file, model)

    state = {'model_rep': model['rep'].state_dict(),
             'connectivities': model['rep'].connectivities,
             'additives': model['rep'].block[0].additives_dict,
             'last_layer_additives': model['rep'].last_layer_additives}
    with open(param_file) as json_params:
        params = json.load(json_params)
    tasks = params['tasks']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        state[key_name] = model[t].state_dict()
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_additives' + '.pkl'
    torch.save(state, new_save_model_path)


def test_biased_net(param_file, save_model_path):
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_biased' + '.pkl'
    model = load_trained_model(param_file, new_save_model_path, True, True, False, True)
    eval_trained_model(param_file, model)


def test_additives_net(param_file, save_model_path):
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_additives' + '.pkl'
    model = load_trained_model(param_file, new_save_model_path, True, True, False, False, True, True, 'restore')
    eval_trained_model(param_file, model)


def store_averaged_activations_for_disabling(param_file, save_model_path, if_actively_disable_bias):
    with open(param_file) as json_params:
        params = json.load(json_params)
    if params['input_size'] == 'default':
        config_path = 'configs.json'
    elif params['input_size'] == 'bigimg':
        config_path = 'configs_big_img.json'
    elif params['input_size'] == 'biggerimg':
        config_path = 'configs_bigger_img.json'
    with open(config_path) as config_params:
        configs = json.load(config_params)

    model = load_trained_model(param_file, save_model_path, True, if_store_avg_activations_for_disabling=True,
                               replace_with_avgs_last_layer_mode='store', if_actively_disable_bias=if_actively_disable_bias)
    if True:
        params['batch_size'] = 1200
    loader = datasets.get_random_val_subset(params, configs)
    if loader is None:
        _, loader, _ = datasets.get_dataset(params, configs)
    eval_trained_model(param_file, model, loader=loader, if_stop_after_single_batch=True)

    state = {'model_rep': model['rep'].state_dict(),
             'connectivities': model['rep'].connectivities,
             'connectivity_comeback_multipliers': model['rep'].connectivity_comeback_multipliers,
             'additives': model['rep'].block[0].additives_dict,
             'last_layer_additives': model['rep'].last_layer_additives}

    tasks = params['tasks']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        state[key_name] = model[t].state_dict()
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_avgadditives' + '.pkl'
    torch.save(state, new_save_model_path)


def test_averagedadditives_net(param_file, save_model_path, if_actively_disable_bias):
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_avgadditives' + '.pkl'
    removed_conns = defaultdict(list)
    removed_conns['shortcut'] = defaultdict(list)
    # removed_conns[10] = [(86, 104)]
    # removed_conns['label'] = [(172, 8), (400, 8), (383, 8)]
    # removed_conns['shortcut'][2] = [(23, 102)]
    # removed_conns['shortcut'][10] = [(212, 383)]
    # removed_conns['shortcut'][12] = [(383, 383)]
    model = load_trained_model(param_file, new_save_model_path, True, if_additives_user=True,
                               if_store_avg_activations_for_disabling=True, replace_with_avgs_last_layer_mode='restore'
                               , conns_to_remove_dict=removed_conns, if_actively_disable_bias=if_actively_disable_bias)
    eval_trained_model(param_file, model)


if __name__ == '__main__':
    model_to_evaluate = 'learned-structure-l1-disable-mouth'#'learned-structure-l1-disable-black-hair'#'learned-structure-l1'#'baseline-pruned'#'learned-structure-no-l1'#'baseline'
    # THE TWO MODELS BELOW ARE MY REFERENCES AS "BIG" & "BIGGER" MODELS
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'

    if model_to_evaluate in ['baseline', 'learned-structure-no-l1', 'learned-structure-l1', 'baseline-pruned', 'only-hat', 'only-lipstick', 'cifar-net']:
        save_model_path = f'pretrained_models/{model_to_evaluate}.pkl'
        param_file = f'named_params/{model_to_evaluate}.json'
    elif model_to_evaluate in ['learned-structure-l1-disable-black-hair', 'learned-structure-l1-disable-mouth']:
        save_model_path = f'pretrained_models/learned-structure-l1-avg-additives.pkl'
        param_file = f'named_params/learned-structure-l1.json'
    else:
        raise NotImplementedError(f'Unknown model name: {model_to_evaluate}')

    # trained_model = torchvision.models.__dict__['resnet18'](pretrained=True).cuda()
    # eval_trained_model(param_file, trained_model, if_pretrained_imagenet=True)
    #figure out whether need to actively disable bias due to cifar networks that ignored the enable_bias parameter
    if_actively_disable_bias=False
    try:
        if model_to_evaluate in ['learned-structure-l1-disable-black-hair', 'learned-structure-l1-disable-mouth']:
            removed_conns = defaultdict(list)
            removed_conns['shortcut'] = defaultdict(list)
            if model_to_evaluate == 'learned-structure-l1-disable-mouth':
                removed_conns[10] = [(188, 104)]
                removed_conns['shortcut'][8] = [(86, 86)]
            elif model_to_evaluate == 'learned-structure-l1-disable-black-hair':
                removed_conns['shortcut'][12] = [(400, 400)]
            trained_model = load_trained_model(param_file, save_model_path, if_additives_user=True,
                                               if_store_avg_activations_for_disabling=True,
                                               conns_to_remove_dict=removed_conns)
        else:
            trained_model = load_trained_model(param_file, save_model_path)
    except:
        print('assume problem where cifar networks ignored the enable_bias parameter')
        if_actively_disable_bias = True
        trained_model = load_trained_model(param_file, save_model_path, if_actively_disable_bias=if_actively_disable_bias)
    eval_trained_model(param_file, trained_model)

    # convert_useless_connections_to_biases(param_file, save_model_path, config_name)
    # test_biased_net(param_file, save_model_path, config_name)
    # convert_useless_connections_to_additives(param_file, save_model_path)
    # test_additives_net(param_file, save_model_path)
    # store_averaged_activations_for_disabling(param_file, save_model_path, if_actively_disable_bias)
    # test_averagedadditives_net(param_file, save_model_path, if_actively_disable_bias)