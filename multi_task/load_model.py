import json
import torch

from torch.autograd import Variable

import multi_task.datasets as datasets
import multi_task.metrics as metrics

import multi_task.model_selector_automl as model_selector_automl

if_train_default_resnet = not True

def load_trained_model(param_file, save_model_path):
    with open(param_file) as json_params:
        params = json.load(json_params)

    # train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    # loss_fn = losses.get_loss(params)

    model = model_selector_automl.get_model(params)

    tasks = params['tasks']

    state = torch.load(save_model_path)

    model['rep'].load_state_dict(state['model_rep'])
    for i in range(3):
        #apparently learnings scales & biases are saved automatically as part of the model state
        pass
        # model['rep'].lin_coeffs_id_zero[i] = state[f'learning_scales{i}']
        # model['rep'].bn_biases[i] = state[f'bn_bias{i}']

    for t in tasks:
        key_name = 'model_{}'.format(t)
        model[t].load_state_dict(state[key_name])

    return model

def eval_trained_model(param_file, model):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    metric = metrics.get_metrics(params)

    tst_loader = datasets.get_test_dataset(params, configs)

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']
    for m in model:
        model[m].eval()

    with torch.no_grad():
        for batch_val in tst_loader:
            val_images = Variable(batch_val[0].cuda())
            labels_val = {}

            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_val[i + 1]
                labels_val[t] = Variable(labels_val[t].cuda())

            val_reps, _ = model['rep'](val_images, None)
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
                # print(out_t_val)
                # print(labels_val[t])
                # print(metric[t].get_result())
    error_sum = 0
    for t in tasks:
        metric_results = metric[t].get_result()

        for metric_key in metric_results:
            print(f'Task = {t}, acc = {metric_results[metric_key]}')
            error_sum += 1 - metric_results[metric_key]

        metric[t].reset()

    error_sum /= float(len(tasks))
    print( error_sum * 100)

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
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/06_58_on_February_26/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|chunks=[1|_1|_16]|architecture=resnet18|width_mul=1|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximatio_5_model.pkl'
    param_file = 'params/small_reg_1_1_16.json'
    model = load_trained_model(param_file, save_model_path)
    eval_trained_model(param_file, model)
