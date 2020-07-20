import json
import torch

from torch.autograd import Variable

# import multi_task.datasets as datasets
# import multi_task.metrics as metrics
# import multi_task.model_selector_automl as model_selector_automl
import datasets as datasets
import metrics as metrics
import model_selector_automl as model_selector_automl
from util.util import celeba_dict
import pandas
import numpy as np


def disable_conn(state, ind, src, dest):
    state['connectivities'][ind][dest, src] = 0


def load_trained_model(param_file, save_model_path, if_restore_connectivities=True,
                       if_visuzalization_conns=False, if_replace_useless_conns_with_bias=False,
                       if_enable_bias=False, if_replace_useless_conns_with_additives=False,
                       if_additives_user=False, replace_constants_last_layer_mode=None):
    assert not (if_replace_useless_conns_with_additives and if_replace_useless_conns_with_bias) #only 1 of those can be true

    with open(param_file) as json_params:
        params = json.load(json_params)

    if if_visuzalization_conns:
        viz_conns_state = torch.load('visualized_connectivities.pkl')
        auxillary_connectivities_for_id_shortcut = viz_conns_state['auxillary_connectivities_for_id_shortcut']

        params['this_is_graph_visualization_run'] = 'Yep'
        if True:
            params['auxillary_connectivities_for_id_shortcut'] = list(
                map(lambda x: torch.Tensor(x).cuda() if x is not None else None, auxillary_connectivities_for_id_shortcut)) \
                                                             + [None, None, None]
        else:
            params['auxillary_connectivities_for_id_shortcut'] = [None] * 15

    if if_replace_useless_conns_with_bias:
        params['if_replace_useless_conns_with_bias'] = 'Yep'
    if if_replace_useless_conns_with_additives:
        params['if_replace_useless_conns_with_additives'] = 'Yep'
    if if_additives_user:
        params['if_additives_user'] = 'Yep'

    if if_enable_bias:
        params['if_enable_bias'] = True
    params['replace_constants_last_layer_mode'] = replace_constants_last_layer_mode

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
    #black hair:
        disable_conn(state, 1, 27, 50)
        disable_conn(state, 1, 35, 50)
        # state['connectivities'][2][28, 16] = 0
        disable_conn(state, 2, 35, 76)
        # disable_conn(state, 3, 76, 96)
        disable_conn(state, 3, 101, 96)
        disable_conn(state, 3, 113, 96)
        disable_conn(state, 6, 29, 160)
        # disable_conn(state, 6, 96, 160)
        # state['connectivities'][6][248, 124] = 0
        # state['connectivities'][7][106, 248] = 0
        # disable_conn(state, 9, 142, 188)
        # if True:
        #     state['connectivities'][9][188, 216] = 0
        # else:
        #     state['connectivities'][8][216, 149] = 0
        #     state['connectivities'][8][216, 0] = 0
        #     state['connectivities'][8][216, 203] = 0
        #     state['connectivities'][8][216, 106] = 0
        #     state['connectivities'][8][216, 31] = 0
        #     state['connectivities'][8][216, 25] = 0
        disable_conn(state, 10, 188, 104)
        disable_conn(state, 10, 86, 104)
        # disable_conn(state, 10, 160, 104)

        state['connectivities'][14][8, :] = 0
        # state['connectivities'][14][8, 383] = 1
        # state['connectivities'][14][8, 172] = 1
        state['connectivities'][14][8, 400] = 1
    #brown hair:
        # state['connectivities'][14][11, :] = 0
        # state['connectivities'][14][11, 383] = 1
    #blond hair:
        # state['connectivities'][14][9, 356] = 0

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

    model['rep'].load_state_dict(model_rep_state)
    for i in range(3):
        # apparently learnings scales & biases are saved automatically as part of the model state
        pass
        # model['rep'].lin_coeffs_id_zero[i] = state[f'learning_scales{i}']
        # model['rep'].bn_biases[i] = state[f'bn_bias{i}']

    if if_replace_useless_conns_with_additives:
        if 'additives' in state:
            model['rep'].block[0].additives_dict = state['additives']
    if replace_constants_last_layer_mode == 'restore':
        model['rep'].last_layer_additives = state['last_layer_additives']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        model[t].load_state_dict(state[key_name])

    # print(model['35'].linear.weight)
    # np.where(model['rep'].connectivities[-1][35, :].cpu().detach().numpy() > 0.5)[0]
    # np.where(np.abs(model['35'].linear.weight.cpu().detach().numpy()[0]) > 0.1)[0]

    # model['12'].linear.weight[:, 148] = torch.tensor([-0.06, 0.05#-0.8863,  0.8848
    #                                                   ]).cuda()
    # model['12'].linear.bias.data = torch.tensor([ 0.0676, -0.0725]).cuda() # first number is no, second number is yes.
        # I.e. if first is 1000 we get 0 predictions, and when second is -1000 we get 0 predictions
    return model

def eval_trained_model(param_file, model, if_store_val_pred=False, save_model_path=None):
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

    metric = metrics.get_metrics(params)

    if not if_store_val_pred:
        tst_loader = datasets.get_test_dataset(params, configs)
        loader = tst_loader
    else:
        _, val_loader, _ = datasets.get_dataset(params, configs)
        loader = val_loader
        preds = [[] for _ in range(40)]
        model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]

    tasks = params['tasks']
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
                print(i)

            def get_relevant_labels_from_batch(batch):
                labels = {}
                # Read all targets of all tasks
                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    if params['dataset'] == 'cifar10':
                        labels[t] = (batch[1] == int(t)).type(torch.LongTensor)
                    elif params['dataset'] == 'cifarfashionmnist':
                        labels[t] = (batch[1] == int(t)).type(torch.LongTensor)
                    else:
                        labels[t] = batch[i + 1]
                    labels[t] = labels[t].cuda()
                return labels

            val_images = Variable(batch_val[0].cuda())
            labels_val = get_relevant_labels_from_batch(batch_val)

            # val_reps, _ = model['rep'](val_images, None)
            val_reps = model['rep'](val_images)
            for i, t in enumerate(tasks):
                if if_train_default_resnet:
                    val_rep = val_reps
                else:
                    val_rep = val_reps[i]
                out_t_val, _ = model[t](val_rep, None)
                # loss_t = loss_fn[t](out_t_val, labels_val[t])
                # tot_loss['all'] += loss_t.item()
                # tot_loss[t] += loss_t.item()
                metric[t].update(out_t_val, labels_val[t])
                if if_store_val_pred:
                    labels = out_t_val.data.max(1, keepdim=True)[1]
                    temp = list(labels.detach().cpu().numpy().squeeze())
                    preds[i] += temp
                # print(out_t_val)
                # print(labels_val[t])
                # print(metric[t].get_result())
    if if_store_val_pred:
        preds = np.array(preds).T
        preds[preds == 0] = -1
        print(preds.shape)
        df = pandas.DataFrame({name:preds[:, i] for i, name in enumerate(celeba_dict.values())})
        df.to_csv(f'predicted_labels_celeba_{model_name_short}.csv', sep=' ')
    error_sum = 0
    for t in tasks:
        metric_results = metric[t].get_result()

        for metric_key in metric_results:
            print(f'({t}) {celeba_dict[int(t)]}\t acc = {metric_results[metric_key]}'.expandtabs(30))
            error_sum += 1 - metric_results[metric_key]

        metric[t].reset()

    error_sum /= float(len(tasks))
    print(error_sum * 100)


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
             'additives':model['rep'].block[0].additives_dict,
             'last_layer_additives':model['rep'].last_layer_additives}
    with open(param_file) as json_params:
        params = json.load(json_params)
    tasks = params['tasks']
    for t in tasks:
        key_name = 'model_{}'.format(t)
        state[key_name] = model[t].state_dict()
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_additives' + '.pkl'
    torch.save(state, new_save_model_path)

def test_biased_net(param_file, save_model_path, config_name):
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_biased' + '.pkl'
    model = load_trained_model(param_file, new_save_model_path, True, True, False, True)
    eval_trained_model(param_file, model)

def test_additives_net(param_file, save_model_path, config_name):
    new_save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_additives' + '.pkl'
    model = load_trained_model(param_file, new_save_model_path, True, True, False, False, True, True, 'restore')
    eval_trained_model(param_file, model)


if __name__ == '__main__':
    # save_model_path = r"/mnt/raid/data/chebykin/saved_models/first_model_epoch_100.pkl"
    # "optimizer=Adam|batch_size=170|lr=0.0005|dataset=celeba|normalization_type=loss+|algorithm=mgda|use_approximation=True|scales={'\''0'\'': 0.025, '\''1'\'': 0.025, '\''2'\'': 0.025, '\''3'\'': 0.025, '\''4'\'': 0.025, '\''5'\'': 0.025, '\''6_100_model.pkl"'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/model_25nov_epoch31.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_50_on_November_27/ep1.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    # param_file = 'sample_all_test.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
    # param_file = 'params/bigger_reg_4_4_4.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
    # param_file = 'old_params/sample_all.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/06_58_on_February_26/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|chunks=[1|_1|_16]|architecture=resnet18|width_mul=1|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximatio_5_model.pkl'
    # param_file = 'params/small_reg_1_1_16.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_20_on_April_15/optimizer=Adam|batch_size=256|lr=0.0005|connectivities_lr=0.0015|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|if_fully_connected=False|dataset=_58_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_condecay2e6_conlr0015.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_36_on_April_15/optimizer=SGD_Adam|batch_size=64|lr=0.1|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=False|dataset=ce_70_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgdadam01.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/02_09_on_April_17/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=False|use_pret_26_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain.json' # THIS gave 8.98
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_33_on_February_25/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0|architecture=resnet18_vanilla|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_5_model.pkl'
    # param_file = 'params/vanilla.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/18_37_on_April_25/optimizer=SGD_Adam|batch_size=256|lr=0.005|connectivities_lr=0.001|chunks=[8|_8|_32]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=3e-05|connectivities_l1_all=False|if_14_model.pkl'
    # param_file = 'params/binmatr2_cifar.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_42_on_April_17/optimizer=SGD|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=True|use_pretrained_17_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgd001_pretrain_fc.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_37_on_May_15/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_128|_128|_256|_256|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|co_100_model.pkl'
    # param_file = 'params/binmatr2_64_64_128_128_256_256_512_512_sgdadam0004_pretrain_condecayall2e-6_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_49_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004_pretrain_condecayall2e-6_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_condecayall2e-6.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_06_on_June_08/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[32|_32|_32|_32|_32|_32|_32|_32|_32|_32|_32|_32|_32|_32|_32]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|conn_45_model.pkl'
    # param_file = 'params/binmatr2_15_32s_sgdadam001+0005_pretrain_bias_nocondecay_comeback.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_55_on_June_13/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay_60_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_nobias_fc_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_30_on_June_18/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg_consontop_condecayall3e-5.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_57_on_June_19/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_76_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_41_on_June_21/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall2e-6_comeback_rescaled.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_120_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled.json'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_43_on_June_24/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_rescaled3_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/04_25_on_June_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_31_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_rescaled3_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_22_on_June_26/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall1e-6_nocomeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/19_15_on_June_28/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall1e-6_nocomeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_34_on_June_29/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_22_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall1e-6_nocomeback_rescaled2.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_23_on_July_01/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_10_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall1e-6_comeback_rescaled2.json'

    # config_name = 'configs.json'
    # config_name = 'configs_big_img.json'
    # config_name = 'configs_bigger_img.json'
    if False:
        # convert_useless_connections_to_biases(param_file, save_model_path, config_name)
        # test_biased_net(param_file, save_model_path, config_name)
        convert_useless_connections_to_additives(param_file, save_model_path)
        # test_additives_net(param_file, save_model_path)
    else:
        model = load_trained_model(param_file, save_model_path, if_restore_connectivities=True)
        eval_trained_model(param_file, model, True, save_model_path)