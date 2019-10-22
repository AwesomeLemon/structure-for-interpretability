import sys
import torch
import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import types

from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100


def get_ignored_filters_per_task_per_layer(filter_nums=None):
    # to avoid mutable default argument
    if filter_nums is None:
        filter_nums = [
            # 64, #ignore 64: we want at least 1 layer to be shared
            128, 256, 512]

    task_num = 2
    ignored_filter_idx = np.empty((task_num, 0)).tolist()
    print(ignored_filter_idx)
    for filter_num in filter_nums:
        cur_ignored_filter_idx1 = get_ignored_filters_per_layer(filter_num, filter_num // 2)
        cur_ignored_filter_idx2 = get_complimenting_filters(cur_ignored_filter_idx1, filter_num)
        ignored_filter_idx[0].append(cur_ignored_filter_idx1)
        print(ignored_filter_idx)
        ignored_filter_idx[1].append(cur_ignored_filter_idx2)
    return ignored_filter_idx


# def get_ignored_filters_per_task(filter_num=512, filters_to_ignore=256):
#     ignored_filter_idx = random.sample(range(filter_num), filters_to_ignore)
#     return [ignored_filter_idx, list(set(range(filter_num)) - set(ignored_filter_idx))]

def get_ignored_filters_per_layer(filter_num=512, filters_to_ignore=256):
    # ignored_filter_idx = random.sample(range(filter_num), filters_to_ignore)
    return range(filter_num)#ignored_filter_idx

def get_complimenting_filters(ignored_filter_idx, filter_num):
    return list(set(range(filter_num)) - set(ignored_filter_idx))


@click.command()
@click.option('--param_file', default='sample.json', help='JSON parameters file')
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

    log_dir_name = '/mnt/raid/data/chebykin/runs/{}_{}'.format(params['exp_id'],
                                                               datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    if len(log_dir_name) > 255:
        log_dir_name = '/mnt/raid/data/chebykin/runs/{}'.format(datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y"))
    writer = SummaryWriter(log_dir=log_dir_name)

    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    n_iter = 0

    ignored_filters_per_task_per_layer = get_ignored_filters_per_task_per_layer()
    if 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch + 1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Half the learning rate{}'.format(n_iter))

        for m in model:
            model[m].train()

        for batch in train_loader:
            print(n_iter)
            n_iter += 1
            # First member is always images
            images = batch[0]
            images = Variable(images.cuda())

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i + 1]
                labels[t] = Variable(labels[t].cuda())

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}
            mask = None
            masks = {}
            if 'mgda' in params['algorithm']:
                raise NotImplementedError()
            else:
                for t in tasks:
                    masks[t] = None
                    scale[t] = float(params['scales'][t])

            # print(scale)
            # Scaled back-propagation
            optimizer.zero_grad()
            for i, t in enumerate(tasks):
                rep, _ = model['rep'](images, mask, ignored_filters_per_task_per_layer[i])
                out_t, _ = model[t](rep, masks[t])
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
                writer.add_scalar('scale_{}'.format(t), scale[t], n_iter)

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
                val_images = Variable(batch_val[0].cuda(), volatile=True)
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i + 1]
                    labels_val[t] = Variable(labels_val[t].cuda(), volatile=True)

                for i, t in enumerate(tasks):
                    val_rep, _ = model['rep'](val_images, None, ignored_filters_per_task_per_layer[i])
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss['all'] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                    # print(out_t_val)
                    # print(labels_val[t])
                    # print(metric[t].get_result())
                num_val_batches += 1

        for t in tasks:
            writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t] / num_val_batches, n_iter)
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                # pass
                writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
            metric[t].reset()
        writer.add_scalar('validation_loss', tot_loss['all'] / len(val_dst), n_iter)

        if epoch % 3 == 0:
            # Save after every 3 epoch
            state = {'epoch': epoch + 1,
                     'model_rep': model['rep'].state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            for t in tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = model[t].state_dict()

            save_model_path = "/mnt/raid/data/chebykin/saved_models/{}_{}_model.pkl".format(params['exp_id'], epoch + 1)
            if len(save_model_path) > 255:
                save_model_path = "/mnt/raid/data/chebykin/saved_models/" + "{}".format(params['exp_id'])[
                                                                            :200] + "_{}_model.pkl".format(epoch + 1)
            torch.save(state, save_model_path)

        end = timer()
        print('Epoch ended in {}s'.format(end - start))


if __name__ == '__main__':
    train_multi_task()
