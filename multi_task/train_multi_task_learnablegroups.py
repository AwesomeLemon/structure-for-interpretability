import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import losses
import datasets
import metrics

import random
import re

import model_selector_automl
from shutil import copy
import os
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


@click.command()
# @click.option('--param_file', default='old_params/sample_all.json', help='JSON parameters file')
@click.option('--param_file', default='params/learnable_groups.json', help='JSON parameters file')
def train_multi_task(param_file, overwrite_lr=None, overwrite_lambda_reg=None, overwrite_weight_decay=None):
    # with open('configs_smaller_img.json') as config_params:
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    def get_log_dir_name(params):
        exp_identifier = []
        for (key, val) in params.items():
            if 'tasks' in key or 'scales' in key:
                continue
            exp_identifier += ['{}={}'.format(key, val)]

        exp_identifier = '|'.join(exp_identifier)
        params['exp_id'] = exp_identifier

        if_debug = True
        run_dir_name = 'runs_debug' if if_debug else 'runsA'
        time_str = datetime.datetime.now().strftime("%H_%M_on_%B_%d")
        log_dir_name = '/mnt/antares_raid/home/awesomelemon/{}/{}'.format(run_dir_name, time_str)
        def print_proper_log_dir_name():
            log_dir_name_full = '/mnt/antares_raid/home/awesomelemon/{}/{}_{}'.format(run_dir_name, params['exp_id'], time_str)
            log_dir_name_full = re.sub(r'\s+', '_', log_dir_name_full)
            log_dir_name_full = re.sub(r"'", '_', log_dir_name_full)
            log_dir_name_full = re.sub(r'"', '_', log_dir_name_full)
            log_dir_name_full = re.sub(r':', '_', log_dir_name_full)
            log_dir_name_full = re.sub(r',', '|', log_dir_name_full)
            print(log_dir_name_full)

        print_proper_log_dir_name()
        #
        # if len(log_dir_name) > 255:
        #     log_dir_name = '/mnt/antares_raid/home/awesomelemon/{}/{}'.format(run_dir_name, time_str)

        return log_dir_name, time_str

    log_dir_name, time_str = get_log_dir_name(params)
    print(f'Log dir: {log_dir_name}')

    writer = SummaryWriter(log_dir=log_dir_name)

    train_loader, val_loader, train2_loader = datasets.get_dataset(params, configs)

    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']

    arc = params['architecture']
    if_train_default_resnet = 'vanilla' in arc
    model = model_selector_automl.get_model(params)
    # model = model_selector_plainnet.get_model(params)

    if_continue_training = False
    if if_continue_training:
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/18_10_on_December_06/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3_4_model.pkl'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_04_on_December_10/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|___5_model.pkl'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_10_on_December_11/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3_10_model.pkl'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/06_58_on_February_26/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|chunks=[1|_1|_16]|architecture=resnet18|width_mul=1|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximatio_4_model.pkl'
        state = torch.load(save_model_path)
        model['rep'].load_state_dict(state['model_rep'])
        for t in tasks:
            key_name = 'model_{}'.format(t)
            model[t].load_state_dict(state[key_name])

    model_params = []
    for m in model:
        model_params += model[m].parameters()
    #todo: remove freezing
    # for name, param in model['rep'].named_parameters():
    #     if 'chunk_strength' in name or 'bn_bias' in name:
    #         param.requires_grad = False

    print(f'Starting training with parameters \n \t{str(params)} \n')

    lr = params['lr']
    if overwrite_lr is not None:
        lr = overwrite_lr
    weight_decay = 0.0
    if overwrite_weight_decay is not None:
        weight_decay = overwrite_weight_decay
    lambda_reg = params['lambda_reg']
    if overwrite_lambda_reg is not None:
        lambda_reg = overwrite_lambda_reg

    if 'Adam' in params['optimizer']:
        # optimizer = torch.optim.Adam(model_params, lr=params['lr'])
        optimizer = torch.optim.AdamW(model_params, weight_decay=weight_decay, lr=lr)
        # # right now, optimize all, may just as well pass "model['rep'].lin_coeffs_id_zero":
        # lst = [model['rep'].lin_coeffs_id_zero[-1], model['rep'].lin_coeffs_id_zero[-2],
        #        model['rep'].lin_coeffs_id_zero[-3]]
        # optimizer_val = torch.optim.Adam(lst, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    if if_continue_training:
        optimizer.load_state_dict(state['optimizer_state'])

    error_sum_min = 1.0  # highest possible error on the scale from 0 to 1 is 1

    # train2_loader_iter = iter(train2_loader)
    NUM_EPOCHS = 50
    print(f'NUM_EPOCHS={NUM_EPOCHS}')
    n_iter = 0

    if_show_softmax = True
    if if_show_softmax:
        x = vutils.make_grid(torch.softmax(model['rep'].out_channel_group_connectivity, dim=1), padding=0)
        writer.add_image('Groups - tasks connectivity (softmax)', x, n_iter)
    else:
        x = vutils.make_grid(
            model['rep'].out_channel_group_connectivity / torch.sum(model['rep'].out_channel_group_connectivity, dim=1,
                                                                    keepdim=True), padding=0)
        writer.add_image('Groups - tasks connectivity (normalized)', x, n_iter)

    x = vutils.make_grid((model['rep'].out_channel_group_connectivity ==
                          torch.max(model['rep'].out_channel_group_connectivity, dim=1, keepdim=True)[0]).float(),
                         padding=0)
    writer.add_image('Groups - tasks connectivity (max)', x, n_iter)

    if if_show_softmax:
        x = vutils.make_grid(torch.softmax(model['rep'].in_channel_group_connectivity, dim=1), padding=0)
        writer.add_image('Groups - filters connectivity (softmax)', x, n_iter)
    else:
        x = vutils.make_grid(
            model['rep'].in_channel_group_connectivity / torch.sum(model['rep'].in_channel_group_connectivity,
                                                                   dim=1, keepdim=True), padding=0)
        writer.add_image('Groups - filters connectivity (normalized)', x, n_iter)

    x = vutils.make_grid((model['rep'].in_channel_group_connectivity ==
                          torch.max(model['rep'].in_channel_group_connectivity, dim=1, keepdim=True)[0]).float(),
                         padding=0)
    writer.add_image('Groups - filters connectivity (max)', x, n_iter)

    # x = vutils.make_grid(torch.softmax(model['rep'].out_channel_group_connectivity, dim=1), padding=0)
    # writer.add_image('Groups - tasks connectivity (softmax)', x, n_iter)
    #
    # x = vutils.make_grid((model['rep'].out_channel_group_connectivity ==
    #                       torch.max(model['rep'].out_channel_group_connectivity, dim=1, keepdim=True)[0]).float(),
    #                      padding=0)
    # writer.add_image('Groups - tasks connectivity (max)', x, n_iter)
    #
    # x = vutils.make_grid(torch.softmax(model['rep'].in_channel_group_connectivity, dim=1), padding=0)
    # writer.add_image('Groups - filters connectivity (softmax)', x, n_iter)
    #
    # x = vutils.make_grid((model['rep'].in_channel_group_connectivity ==
    #                       torch.max(model['rep'].in_channel_group_connectivity, dim=1, keepdim=True)[0]).float(),
    #                      padding=0)
    # writer.add_image('Groups - filters connectivity (max)', x, n_iter)

    for epoch in range(NUM_EPOCHS):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        # if (epoch + 1) % 30 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.5
        #     print('Halve the learning rate{}'.format(n_iter))
        #     # for param_group in optimizer_val.param_groups:
        #     #     param_group['lr'] *= 0.5

        for m in model:
            model[m].train()

        for batch_idx, batch in enumerate(train_loader):
            print(n_iter)
            n_iter += 1

            def get_relevant_labels_from_batch(batch):
                labels = {}
                # Read all targets of all tasks
                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels[t] = batch[i + 1]
                    labels[t] = Variable(labels[t].cuda())
                return labels

            # 2. Gradient step for normal weights using one batch from the training set:

            # First member is always images
            images = batch[0]
            images = Variable(images.cuda())
            labels = get_relevant_labels_from_batch(batch)

            loss_data = {}
            scale = {}
            for t in tasks:
                scale[t] = float(params['scales'][t])

            optimizer.zero_grad()
            loss = 0
            reps = model['rep'](images)
            for i, t in enumerate(tasks):
                rep = reps[i]
                out_t = model[t](rep)
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                loss = loss + scale[t] * loss_t
            loss.backward()
            optimizer.step()

            writer.add_scalar('training_loss', loss.item(), n_iter)
            if False and not if_train_default_resnet:
                writer.add_scalar('l1_reg_loss', loss_reg_value, n_iter)
                writer.add_scalar('training_minus_l1_reg_loss', loss.item() - loss_reg_value, n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

        for m in model:
            model[m].eval()

        tot_loss = {}
        # tot_loss['l1_reg'] = lambda_reg * torch.norm(torch.cat([model['rep'].lin_coeffs_id_zero[i].view(-1) for i in range(3)]), 1)
        tot_loss['all'] = 0.0  # tot_loss['l1_reg']#0.0
        for t in tasks:
            tot_loss[t] = 0.0

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

                val_reps = model['rep'](val_images)
                for i, t in enumerate(tasks):
                    val_rep = val_reps[i]
                    out_t_val = model[t](val_rep)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss['all'] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1

        error_sum = 0
        for t in tasks:
            writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t] / num_val_batches, n_iter)
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
                error_sum += 1 - metric_results[metric_key]
            metric[t].reset()

        error_sum /= float(len(tasks))
        writer.add_scalar('average_error', error_sum * 100, n_iter)
        print(f'average_error = {error_sum * 100}')

        writer.add_scalar('validation_loss', tot_loss['all'] / num_val_batches / len(tasks), n_iter)
        # writer.add_scalar('l1_reg_loss', tot_loss['l1_reg'] / num_val_batches, n_iter)

        # write group counts to log
        #
        # for i, block in enumerate(model['rep'].layer1):
        #     writer.add_scalar(f'group_number_1_{i}', block.conv2.num_groups, n_iter)
        #
        # for i, block in enumerate(model['rep'].layer2):
        #     writer.add_scalar(f'group_number_2_{i}', block.conv2.num_groups, n_iter)
        #
        # for i, block in enumerate(model['rep'].layer3):
        #     writer.add_scalar(f'group_number_3_{i}', block.conv2.num_groups, n_iter)
        #
        # for i, block in enumerate(model['rep'].layer4):
        #     writer.add_scalar(f'group_number_4_{i}', block.conv2.num_groups, n_iter)

        if if_show_softmax:
            x = vutils.make_grid(torch.softmax(model['rep'].out_channel_group_connectivity, dim=1), padding=0)
            writer.add_image('Groups - tasks connectivity (softmax)', x, n_iter)
        else:
            x = vutils.make_grid(model['rep'].out_channel_group_connectivity / torch.sum(model['rep'].out_channel_group_connectivity, dim=1, keepdim=True), padding=0)
            writer.add_image('Groups - tasks connectivity (normalized)', x, n_iter)

        x = vutils.make_grid((model['rep'].out_channel_group_connectivity == torch.max(model['rep'].out_channel_group_connectivity, dim=1, keepdim=True)[0]).float(), padding=0)
        writer.add_image('Groups - tasks connectivity (max)', x, n_iter)

        if if_show_softmax:
            x = vutils.make_grid(torch.softmax(model['rep'].in_channel_group_connectivity, dim=1), padding=0)
            writer.add_image('Groups - filters connectivity (softmax)', x, n_iter)
        else:
            x = vutils.make_grid(
                model['rep'].in_channel_group_connectivity / torch.sum(model['rep'].in_channel_group_connectivity,
                                                                        dim=1, keepdim=True), padding=0)
            writer.add_image('Groups - filters connectivity (normalized)', x, n_iter)

        x = vutils.make_grid((model['rep'].in_channel_group_connectivity ==
                              torch.max(model['rep'].in_channel_group_connectivity, dim=1, keepdim=True)[0]).float(),
                             padding=0)
        writer.add_image('Groups - filters connectivity (max)', x, n_iter)


        if epoch % 3 == 0 or (error_sum < error_sum_min and epoch >= 3):
            # Save after every 3 epoch
            state = {'epoch': epoch + 1,
                     'model_rep': model['rep'].state_dict(),
                     'optimizer_state': optimizer.state_dict()
                     }
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
                # to properly restore model, we need source code for it
                copy('multi_task/models/my_multi_faces_resnet.py' if not if_train_default_resnet
                     else 'multi_task/models/multi_faces_resnet.py', saved_models_prefix)

        error_sum_min = min(error_sum, error_sum_min)
        writer.flush()

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    writer.close()


if __name__ == '__main__':
    train_multi_task()
    # for lr in [0.0005, 0.00005, 0.001, 0.005]:
    #     for lambda_reg in [0.00005, 0.0001, 0.0005, 0.001, 0.01]:
    #         for weight_decay in [0.1, 0.15, 0.2, 0.25]:
    #             print(f'lr = {lr} ; lambda_reg = {lambda_reg} ; weight_decay = {weight_decay}')
    #             try:
    #                 train_multi_task('sample_all_small_reg.json', overwrite_lr=lr,
    #                              overwrite_lambda_reg=lambda_reg, overwrite_weight_decay=weight_decay)
    #             except BaseException as error:
    #                 print('An exception occurred: {}'.format(error))

    # AND then, some time later, this was run:
    # for lr in [0.0005]:
    #     for lambda_reg in [0.0001]:
    #         for weight_decay in [0.5]:
    #             print(f'lr = {lr} ; lambda_reg = {lambda_reg} ; weight_decay = {weight_decay}')
    #             try:
    #                 train_multi_task('sample_all_small_reg.json', overwrite_lr=lr,
    #                              overwrite_lambda_reg=lambda_reg, overwrite_weight_decay=weight_decay)
    #             except BaseException as error:
    #                 print('An exception occurred: {}'.format(error))
