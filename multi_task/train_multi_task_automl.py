import click
import os
import math
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import losses
import datasets
import metrics

import random
import re

import model_selector_automl
from shutil import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ignored_filters_per_task_per_layer(ignore_up_to, filter_nums=None, already_calculated_dict={}):
    # to avoid mutable default argument
    if filter_nums is None:
        filter_nums = [
            # 64, #ignore 64: we want at least 1 layer to be shared
            128, 256, 512]

    task_num = 2
    ignored_filter_idx = np.empty((task_num, 0)).tolist()
    for i, filter_num in enumerate(filter_nums):
        if i > ignore_up_to:
            if filter_num in already_calculated_dict:
                cur_ignored_filter_idx1 = already_calculated_dict[filter_num]
            else:
                cur_ignored_filter_idx1 = get_ignored_filters_per_layer(filter_num, filter_num // 2)
                already_calculated_dict[filter_num] = cur_ignored_filter_idx1

            cur_ignored_filter_idx2 = get_complimenting_filters(cur_ignored_filter_idx1, filter_num)
        else:
            cur_ignored_filter_idx1 = cur_ignored_filter_idx2 = []
        ignored_filter_idx[0].append(cur_ignored_filter_idx1)
        ignored_filter_idx[1].append(cur_ignored_filter_idx2)

    print(ignored_filter_idx)
    return ignored_filter_idx


# def get_ignored_filters_per_task(filter_num=512, filters_to_ignore=256):
#     ignored_filter_idx = random.sample(range(filter_num), filters_to_ignore)
#     return [ignored_filter_idx, list(set(range(filter_num)) - set(ignored_filter_idx))]

def get_ignored_filters_per_layer(filter_num=512, filters_to_ignore=256):
    ignored_filter_idx = random.sample(range(filter_num), filters_to_ignore)
    return ignored_filter_idx  # range(filter_num)


def get_complimenting_filters(ignored_filter_idx, filter_num):
    return list(set(range(filter_num)) - set(ignored_filter_idx))


@click.command()
@click.option('--param_file', default='sample_all.json', help='JSON parameters file')
def train_multi_task(param_file):
    # print(get_ignored_filters_per_layer_per_task())

    # ignored_filters_per_task = get_ignored_filters_per_task()
    # print(sorted(ignored_filters_per_task[0]))
    # print(sorted(ignored_filters_per_task[1]))
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    exp_identifier = []
    for (key, val) in params.items():
        if 'tasks' in key:
            continue
        exp_identifier += ['{}={}'.format(key, val)]

    exp_identifier = '|'.join(exp_identifier)
    params['exp_id'] = exp_identifier

    time_str = datetime.datetime.now().strftime("%H_%M_on_%B_%d")
    log_dir_name = '/mnt/antares_raid/home/awesomelemon/runs7/{}_{}'.format(params['exp_id'],
                                                                            time_str)
    log_dir_name = re.sub(r'\s+', '_', log_dir_name)
    log_dir_name = re.sub(r"'", '_', log_dir_name)
    log_dir_name = re.sub(r'"', '_', log_dir_name)
    log_dir_name = re.sub(r':', '_', log_dir_name)
    log_dir_name = re.sub(r',', '|', log_dir_name)
    print(log_dir_name)

    if len(log_dir_name) > 255:
        log_dir_name = '/mnt/antares_raid/home/awesomelemon/runs7/{}'.format(
            time_str)

    print(f'Log dir: {log_dir_name}')
    writer = SummaryWriter(log_dir=log_dir_name)

    train_loader, val_loader, train2_loader = datasets.get_dataset(params, configs)

    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector_automl.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    n_iter = 0

    filter_nums = [
        # 64, #ignore 64: we want at least 1 layer to be shared
        128, 256, 512]

    NUM_EPOCHS = 50

    if 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model_params, lr=params['lr'])
        # # right now, optimize all, may just as well pass "model['rep'].lin_coeffs_id_zero":
        # lst = [model['rep'].lin_coeffs_id_zero[-1], model['rep'].lin_coeffs_id_zero[-2],
        #        model['rep'].lin_coeffs_id_zero[-3]]
        # optimizer_val = torch.optim.Adam(lst, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.9)

    # right now, optimize all, may just as well pass "model['rep'].lin_coeffs_id_zero":
    lst = [model['rep'].lin_coeffs_id_zero[-1]
        # , model['rep'].lin_coeffs_id_zero[-2],
        #    model['rep'].lin_coeffs_id_zero[-3]
           ]
    optimizer_val = torch.optim.SGD(lst, lr=0.1, momentum=0.9)

    train2_loader_iter = iter(train2_loader)

    for epoch in range(NUM_EPOCHS):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch + 1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Halve the learning rate{}'.format(n_iter))
            for param_group in optimizer_val.param_groups:
                param_group['lr'] *= 0.5

        for m in model:
            model[m].train()

        for batch_idx, batch in enumerate(train_loader):

            print(n_iter)
            n_iter += 1

            def get_desired_labels_from_batch(batch):
                labels = {}
                # Read all targets of all tasks
                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels[t] = batch[i + 1]
                    labels[t] = Variable(labels[t].cuda())
                return labels

            if True:
                # 1. Gradient step for connectivity variables using one batch from the validation set:

                # get a random minibatch from the search queue with replacement
                batch_val = next(train2_loader_iter, None)

                if batch_val is None:
                    print('Recreating train2 loader')
                    train2_loader_iter = iter(train2_loader)
                    batch_val = next(train2_loader_iter)

                images_val = batch_val[0]
                images_val = Variable(images_val.cuda())
                labels_val = get_desired_labels_from_batch(batch_val)

                scale = {}
                for t in tasks:
                    scale[t] = float(params['scales'][t])

                loss = 0
                optimizer_val.zero_grad()
                reps, _ = model['rep'](images_val, None)
                for i, t in enumerate(tasks):
                    rep = reps[i]
                    out_t, _ = model[t](rep, None)
                    loss_t = loss_fn[t](out_t, labels_val[t])
                    if i > 0:
                        loss = loss + scale[t] * loss_t
                    else:
                        loss = scale[t] * loss_t
                loss.backward()
                optimizer_val.step()

                writer.add_scalar('training2_loss', loss.item(), n_iter)

            # 2. Gradient step for normal weights using one batch from the training set:

            # First member is always images
            images = batch[0]
            images = Variable(images.cuda())
            labels = get_desired_labels_from_batch(batch)

            loss_data = {}
            scale = {}
            for t in tasks:
                scale[t] = float(params['scales'][t])

            loss = 0
            optimizer.zero_grad()
            reps, _ = model['rep'](images, None)
            for i, t in enumerate(tasks):
                rep = reps[i]
                out_t, _ = model[t](rep, None)
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                if i > 0:
                    loss = loss + scale[t] * loss_t
                else:
                    loss = scale[t] * loss_t
            loss.backward()
            optimizer.step()

            writer.add_scalar('training_loss', loss.item(), n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

        # print('got to evaluating models')
        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss['all'] = 0.0
        met = {}
        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = Variable(batch_val[0].cuda())
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i + 1]
                    labels_val[t] = Variable(labels_val[t].cuda())

                val_reps, _ = model['rep'](val_images, None)
                for i, t in enumerate(tasks):
                    val_rep = val_reps[i]
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss['all'] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                    # print(out_t_val)
                    # print(labels_val[t])
                    # print(metric[t].get_result())
                num_val_batches += 1

        error_sum = 0
        for t in tasks:
            writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t] / num_val_batches, n_iter)
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                # pass
                writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
                error_sum += 1 - metric_results[metric_key]

            metric[t].reset()

        error_sum /= float(len(tasks))
        writer.add_scalar('average_error', error_sum * 100, n_iter)

        writer.add_scalar('validation_loss', tot_loss['all'] / num_val_batches / len(tasks), n_iter)

        # write scales to log

        sigmoid_normalization = 1.  # 250.
        sigmoid_internal_multiple = 1.
        for i in range(40):
            coeffs = list(
                torch.sigmoid(sigmoid_internal_multiple * model['rep'].lin_coeffs_id_zero[-1][:, i]).cpu().detach() / sigmoid_normalization)
            coeffs = {str(i): coeff for i, coeff in enumerate(coeffs)}
            writer.add_scalars(f'learning_scales_3_{i}_sigmoid', coeffs, n_iter)

        for i in range(model['rep'].num_automl_blocks4):
            coeffs = list(
                torch.sigmoid(sigmoid_internal_multiple * model['rep'].lin_coeffs_id_zero[-2][:, i]).cpu().detach() / sigmoid_normalization)
            coeffs = {str(i): coeff for i, coeff in enumerate(coeffs)}
            writer.add_scalars(f'learning_scales_2_{i}_sigmoid', coeffs, n_iter)

        for i in range(model['rep'].num_automl_blocks3):
            coeffs = list(
                torch.sigmoid(sigmoid_internal_multiple * model['rep'].lin_coeffs_id_zero[-3][:, i]).cpu().detach() / sigmoid_normalization)
            coeffs = {str(i): coeff for i, coeff in enumerate(coeffs)}
            writer.add_scalars(f'learning_scales_1_{i}_sigmoid', coeffs, n_iter)

        if epoch % 3 == 0:
            # Save after every 3 epoch
            state = {'epoch': epoch + 1,
                     'model_rep': model['rep'].state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'learning_scales0': model['rep'].lin_coeffs_id_zero[0].data,
                     'learning_scales1': model['rep'].lin_coeffs_id_zero[1].data,
                     'learning_scales2': model['rep'].lin_coeffs_id_zero[2].data,}
            for t in tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = model[t].state_dict()

            saved_models_prefix = '/mnt/raid/data/chebykin/saved_models/{}'.format(time_str)
            if not os.path.exists(saved_models_prefix):
                os.makedirs(saved_models_prefix)

            save_model_path = saved_models_prefix + "/{}_{}_model.pkl".format(params['exp_id'], epoch + 1)
            save_model_path = re.sub(r'\s+', '_', save_model_path)
            save_model_path = re.sub(r"'", '_', save_model_path)
            save_model_path = re.sub(r'"', '_', save_model_path)
            save_model_path = re.sub(r':', '_', save_model_path)
            save_model_path = re.sub(r',', '|', save_model_path)
            if len(save_model_path) > 255:
                save_model_path = saved_models_prefix + "/{}".format(params['exp_id'])[
                                                                            :200] + "_{}_model.pkl".format(epoch + 1)
                save_model_path = re.sub(r'\s+', '_', save_model_path)
                save_model_path = re.sub(r"'", '_', save_model_path)
                save_model_path = re.sub(r'"', '_', save_model_path)
                save_model_path = re.sub(r':', '_', save_model_path)
                save_model_path = re.sub(r',', '|', save_model_path)
            torch.save(state, save_model_path)

            if epoch == 0:
                #to properly restore model, we need source code for it
                copy('multi_task/models/my_multi_faces_resnet.py', saved_models_prefix)

        writer.flush()

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    writer.close()


if __name__ == '__main__':
    train_multi_task()
