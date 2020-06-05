from multiprocessing import set_start_method

import click
import json
import datetime
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from util.sgd_adam import SGD_Adam
import numpy as np
import torch
from load_model import load_trained_model
torch.multiprocessing.set_sharing_strategy('file_system')
from tensorboardX import SummaryWriter
import losses
import datasets
import metrics
import re
import model_selector_automl
from shutil import copy
import os
import torch
import torchvision.models

device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True
# set_start_method('spawn')

@click.command()
# @click.option('--param_file', default='old_params/sample_all.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_128_256_512_sgdadam001_pretrain.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_128_256_512_sgd001_pretrain_fc_consontop.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_4_4_8_8_16_16_32_32_sgdadam001_pretrain_condecayall2e-6.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_cifarfashionmnist.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr_fullconv_8_8_32_sgdadam001_pretrain.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_cityscapes_2tasks.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_64_64_128_128_256_256_512_512_sgdadam0004_pretrain_condecayall2e-6_bigimg.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_4_4_4_4_4_4_4_4_sgdadam0004_pretrain_fc_bigimg.json', help='JSON parameters file')
# @click.option('--param_file', default='params/binmatr2_filterwise_sgdadam001_pretrain_condecaytask1e-7.json', help='JSON parameters file')
@click.option('--param_file', default='params/binmatr2_15_8s_sgdadam001+0005_pretrain_condecayall1e-5_comeback.json', help='JSON parameters file')
@click.option('--if_debug/--not_debug', default=True, help='Whether to store results in runs_debug')
@click.option('--conn_counts_file', default='', help='Path to store number of activated connections '
                                                                  'instead of writing to tensorboard'
                                                                  'If empty, tensorboard will be used')
def train_multi_task(param_file, if_debug, conn_counts_file, overwrite_lr=None, overwrite_lambda_reg=None, overwrite_weight_decay=None):
    # print("Approx. optimal weights 0.89 0.01 0.1 (S, I, D) - from https://arxiv.org/pdf/1705.07115.pdf")
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

    def get_log_dir_name(params):
        exp_identifier = []
        for (key, val) in params.items():
            if 'tasks' in key or 'scales' in key:
                continue
            exp_identifier += ['{}={}'.format(key, val)]

        exp_identifier = '|'.join(exp_identifier)
        params['exp_id'] = exp_identifier

        run_dir_name = 'runs_debug' if if_debug else 'runsB'
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
    # summary(model['rep'], input_size=[(3, 64, 64), (1,)])
    # model = model_selector_plainnet.get_model(params)

    if_use_pretrained_resnet = params['use_pretrained_resnet']
    if if_use_pretrained_resnet:
        model_rep_dict = model['rep'].state_dict()
        def rename_key(key):
            key = re.sub('downsample', 'shortcut', key)
            if False:
                key = re.sub('(layer[34].0.conv1.)', '\g<0>ordinary_conv.', key)
            else:
                if ('layer1.0.conv1' not in key) and ('layer1.0.conv2' not in key): # a hack because it's 1 AM, this is the only layer which shouldn't be replaced in the 8-block system
                    key = re.sub('(layer[1234].[01].conv[12].)', '\g<0>ordinary_conv.', key)
                    key = re.sub('(layer[234].0.shortcut.0.)', '\g<0>ordinary_conv.', key) #"shortcut" because applied after replacing "downsample"
            return key

        if 'celeba' in params['dataset']:
            pretrained_dict = torchvision.models.resnet18(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if (rename_key(k) not in model_rep_dict) and (k != 'conv1.weight'):
                    print('ACHTUNG! Following pretrained weight was ignored: ', rename_key(k))
            pretrained_dict = {rename_key(k): v for k, v in pretrained_dict.items() if rename_key(k) in model_rep_dict and k != 'conv1.weight'}
        elif 'cityscapes' in params['dataset']:
            pretrained_dict = torchvision.models.resnet50(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if (rename_key(k) not in model_rep_dict) and (k != 'conv1.weight'):
                    print('ACHTUNG! Following pretrained weight was ignored: ', rename_key(k))
            pretrained_dict = {rename_key(k): v for k, v in pretrained_dict.items() if rename_key(k) in model_rep_dict}


        # model_rep_dict.update(pretrained_dict)
        pretrained_dict['conv1.weight'] = model_rep_dict['conv1.weight'] # this is the only difference between them, as of now. If there are any missing or extraneous keys, Pytorch throws an exception
        model['rep'].load_state_dict(pretrained_dict)


    if_continue_training = False
    if if_continue_training:
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/18_10_on_December_06/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3_4_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_04_on_December_10/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|___5_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_10_on_December_11/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3_10_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/06_58_on_February_26/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.0001|chunks=[1|_1|_16]|architecture=resnet18|width_mul=1|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximatio_4_model.pkl'
        # state = torch.load(save_model_path)
        # model['rep'].load_state_dict(state['model_rep'])
        #
        # for t in tasks:
        #     key_name = 'model_{}'.format(t)
        #     model[t].load_state_dict(state[key_name])
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_42_on_April_17/optimizer=SGD|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=True|use_pretrained_17_model.pkl'
        # param_file = 'params/binmatr2_8_8_8_sgd001_pretrain_fc_consontop.json'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_51_on_May_21/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_16_model.pkl'
        print('Continuing training from the following path:')
        print(save_model_path)
        model = load_trained_model(param_file, save_model_path, if_restore_connectivities=True)

    model_params = []
    if_freeze_normal_params_only = params['freeze_all_but_conns']

    # print('setting BatchNorm to eval')
    def set_bn_to_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    for m in model:
        if if_freeze_normal_params_only:
            model[m].apply(set_bn_to_eval)

        cur_params = model[m].parameters()
        if if_freeze_normal_params_only:
            for param in cur_params:
                param.requires_grad = False
            #need tor unfreeze connectivitities
        model_params += cur_params
        model[m].to(device)

    if if_freeze_normal_params_only:
        with torch.no_grad():
            for conn in model['rep'].connectivities:
                conn *= 0.75
                conn.requires_grad = True

    #todo: remove freezing
    # for name, param in model['rep'].named_parameters():
    #     if 'chunk_strength' in name or 'bn_bias' in name:
    #         param.requires_grad = False

    print(f'Starting training with parameters \n \t{str(params)} \n')

    lr = params['lr']
    if overwrite_lr is not None:
        lr = overwrite_lr
    weight_decay = 0.0 if 'weight_decay' not in params else params['weight_decay']
    if overwrite_weight_decay is not None:
        weight_decay = overwrite_weight_decay

    if_learn_task_specific_connections = True and 'fullconv' not in params['architecture']

    lambda_reg = params['connectivities_l1']
    if_apply_l1_to_all_conn = params['connectivities_l1_all']

    if 'SGD_Adam' in params['optimizer']:
        sgd_optimizer = torch.optim.SGD([
            {'params': model_params}],
            lr=lr, momentum=0.9)

        connectivities_lr = params['connectivities_lr']
        adam_optimizer = torch.optim.AdamW([{'params': model['rep'].connectivities, 'name':'connectivities'}],
            lr=connectivities_lr, weight_decay=weight_decay)

        optimizer = SGD_Adam(sgd_optimizer, adam_optimizer)
    elif 'Adam' in params['optimizer']:
        connectivities_lr = params['connectivities_lr']
        optimizer = torch.optim.AdamW([
            {'params': model_params, 'name':'normal_params'},
            {'params': model['rep'].connectivities, 'lr': connectivities_lr, 'name':'connectivities'}],
            lr=lr, weight_decay=weight_decay)

        #TODO: only for computational graph visulaization!
        # optimizer = torch.optim.AdamW([
        #     {'params': model_params}],
        #     lr=lr, weight_decay=weight_decay)

    elif 'SGD' in params['optimizer']:
        # optimizer = torch.optim.SGD([{'params' : model_params}, {'params':model['rep'].connectivities, 'lr' : 0.2}], lr=lr, momentum=0.9)
        optimizer = torch.optim.SGD([{'params' : model_params}, {'params':model['rep'].connectivities}], lr=lr, momentum=0.9)
    if if_continue_training:
        if 'SGD_Adam' != params['optimizer']:
            optimizer.load_state_dict(state['optimizer_state'])

    print(model['rep'])

    error_sum_min = 1.0  # highest possible error on the scale from 0 to 1 is 1

    # train2_loader_iter = iter(train2_loader)
    NUM_EPOCHS = 45
    print(f'NUM_EPOCHS={NUM_EPOCHS}')
    n_iter = 0

    scale = {}
    for t in tasks:
        scale[t] = float(params['scales'][t])

    def write_connectivities(n_iter):
        if conn_counts_file == '':
            for i, cur_con in enumerate(model['rep'].connectivities):
                for j in range(cur_con.size(0)):
                    coeffs = list(cur_con[j].cpu().detach())
                    coeffs = {str(i): coeff for i, coeff in enumerate(coeffs)}
                    writer.add_scalars(f'learning_scales_{i + 1}_{j}', coeffs, n_iter)
        else:
            with open(conn_counts_file, 'a') as f:
                f.write('\n')
                f.write(f'n_iter = {n_iter}' + '\n')
                f.write('\n')
                for i, cur_con in enumerate(model['rep'].connectivities):
                    for j in range(cur_con.size(0)):
                        coeffs = cur_con[j].cpu().detach().numpy()
                        coeffs[coeffs <= 0.5] = 0
                        coeffs[coeffs > 0.5] = 1
                        f.write(f'learning_scales_{i + 1}_{j}: {int(np.sum(coeffs))}/{coeffs.shape[0]}\n')
                    f.write('\n')
                f.write('\n')


    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(NUM_EPOCHS):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        # if (epoch + 1) % 30 == 0:
        #     lr_multiplier = 0.5
        #     for param_group in optimizer.param_groups:
        #         if param_group['name'] == 'connectivities':
        #             continue #don't wanna mess with connectivities
        #         param_group['lr'] *= lr_multiplier
        #         print(f"lr of {param_group['name']} was changed")
        #     print(f'Multiply sgd-only learning rate by {lr_multiplier} at step {n_iter}')
        #     # for param_group in optimizer_val.param_groups:
        #     #     param_group['lr'] *= 0.5

        for m in model:
            model[m].train()
            if if_freeze_normal_params_only:
                model[m].apply(set_bn_to_eval)

        for batch_idx, batch in enumerate(train_loader):
            print(n_iter)
            n_iter += 1

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
                    labels[t] = labels[t].to(device)
                return labels

            # First member is always images
            images = batch[0]
            images = images.to(device)
            labels = get_relevant_labels_from_batch(batch)

            loss_data = {}

            optimizer.zero_grad()
            if False:
                loss_reg = lambda_reg * torch.norm(torch.cat([con.view(-1) for con in model['rep'].connectivities]), 1)
            else:
                # print('Apply l1 only to task connectivities')
                # loss_reg = lambda_reg * torch.norm(model['rep'].connectivities[-1].clone().view(-1), 1)
                if if_apply_l1_to_all_conn:
                    if True:
                        loss_reg = lambda_reg * torch.norm(torch.cat([con.view(-1) for con in model['rep'].connectivities]), 1)
                    else:
                        loss_reg = 0
                        for con in model['rep'].connectivities:
                            loss_reg += torch.norm(con, 1)
                else:
                    loss_reg = lambda_reg * torch.norm(torch.cat([model['rep'].connectivities[-1].view(-1)]), 1)

                # if epoch == 0:
                #     print('ACHTUNG! NOW L1 is applied immediately! Only for connections-only learning!')
                if epoch < 5:
                    loss_reg *= 0
            loss = loss_reg
            loss_reg_value = loss_reg.item()
            reps = model['rep'](images)
            # del images
            for i, t in enumerate(tasks):
                if not if_learn_task_specific_connections:
                    rep = reps
                else:
                    rep = reps[i]
                out_t, _ = model[t](rep, None)
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                loss = loss + scale[t] * loss_t
            loss.backward()
            # plot_grad_flow(model['rep'].named_parameters())
            optimizer.step()

            writer.add_scalar('training_loss', loss.item(), n_iter)
            writer.add_scalar('l1_reg_loss', loss_reg_value, n_iter)
            writer.add_scalar('training_minus_l1_reg_loss', loss.item() - loss_reg_value, n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

            if n_iter == 1:
                #need to do it after the first forward pass because that normalizes them to the [0, 1] range
                write_connectivities(1)
                # for visualizing computation graph:
                # model['rep'].eval()
                # writer.add_graph(model['rep'], images[0][None, :, :, :])
                # model['rep'].train()

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss['l1_reg'] = lambda_reg * torch.norm(torch.cat([con.view(-1) for con in model['rep'].connectivities]), 1)
        tot_loss['all'] = tot_loss['l1_reg']#0.0
        for t in tasks:
            tot_loss[t] = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = batch_val[0].to(device)
                labels_val = get_relevant_labels_from_batch(batch_val)
                # labels_val = {}
                #
                # for i, t in enumerate(all_tasks):
                #     if t not in tasks:
                #         continue
                #     labels_val[t] = batch_val[i + 1]
                #     labels_val[t] = labels_val[t].to(device)

                val_reps = model['rep'](val_images)
                for i, t in enumerate(tasks):
                    if not if_learn_task_specific_connections:
                        val_rep = val_reps
                    else:
                        val_rep = val_reps[i]
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    # tot_loss['all'] += loss_t.item()
                    #todo: I think old way of calculating validation loss was wrong, because we also divided l1 loss by the number of tasks
                    tot_loss['all'] += scale[t] * loss_t.item()
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

        # writer.add_scalar('validation_loss', tot_loss['all'] / num_val_batches / len(tasks), n_iter)
        # todo: I think old way of calculating validation loss was wrong, because we also divided l1 loss by the number of tasks
        writer.add_scalar('validation_loss', tot_loss['all'] / num_val_batches, n_iter)
        writer.add_scalar('validation_loss_minus_l1_reg_loss', (tot_loss['all'] - tot_loss['l1_reg']) / num_val_batches , n_iter)
        # writer.add_scalar('l1_reg_loss', tot_loss['l1_reg'] / num_val_batches, n_iter)

        # write scales to log
        if not if_train_default_resnet:
            write_connectivities(n_iter)

        if epoch % 3 == 0 or (error_sum < error_sum_min and epoch >= 3) or (epoch == NUM_EPOCHS - 1):
            # Save after every 3 epoch
            state = {'epoch': epoch + 1,
                     'model_rep': model['rep'].state_dict(),
                     'connectivities': model['rep'].connectivities,
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
                # Note: for quite some time I've been saving ordinary binmatr instead of binmatr2. Yikes!
                copy('multi_task/models/binmatr2_multi_faces_resnet.py', saved_models_prefix)
                copy('multi_task/models/pspnet.py', saved_models_prefix)
                copy('multi_task/train_multi_task_binmatr.py', saved_models_prefix)

        error_sum_min = min(error_sum, error_sum_min)
        writer.flush()

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    writer.close()


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

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

