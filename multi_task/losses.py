import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def nll(pred, gt, weight=None):
    l = F.nll_loss(pred, gt, weight=weight, reduction='mean' if weight is None else 'none')
    if weight is not None:
        l = l.mean()
    return l

def focal(pred, gt, val=False, weight=None, gamma=0.5):
    log_prob = pred
    prob = torch.exp(log_prob)
    return F.nll_loss(torch.pow(1 - prob, gamma) * log_prob, gt, size_average=not val, weight=weight)

def rmse(pred, gt, val=False):
    pass

def cross_entropy2d(input, target, weight=None, val=False):
    if val:
        size_average = False
    else:
        size_average = True 
    
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def l1_loss_depth(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target != 0# target > 0
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 


def l1_loss_instance(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target!=250
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 

def get_loss(params):
    if 'mnist' == params['dataset']:
        loss_fn = {}
        for t in params['tasks']:
            loss_fn[t] = nll 
        return loss_fn

    if 'cityscapes' == params['dataset']:
        loss_fn = {}
        if 'D' in params['tasks']:
            loss_fn['D'] = rmse
        if 'S' in params['tasks']:
            loss_fn['S'] = cross_entropy2d
        if 'I' in params['tasks']:
            loss_fn['I'] = l1_loss_instance
        if 'D' in params['tasks']:
            loss_fn['D'] = l1_loss_depth
        return loss_fn

    if 'celeba' == params['dataset']:
        loss_fn = {}
        if_weighted_ce = 'weighted_ce' in params #for backward compatibility: unweighted by default
        if if_weighted_ce:
            weighted_ce_type = params['weighted_ce']
            if weighted_ce_type == 'unweighted':
                if_weighted_ce = False
            elif weighted_ce_type == 'switch_freqs':
                freqs = pd.read_csv('class_freqs.csv', header=None) #freq of true
                if False:
                    weights = torch.exp(- torch.tensor(np.vstack((1 - np.array(freqs[1]), np.array(freqs[1])))).float().cuda())
                    # weights = torch.tensor(np.vstack((np.array(freqs[1]), 1 - np.array(freqs[1])))).float().cuda()
                else:
                    freq_yes = np.array(freqs[1])
                    freq_no = 1 - freq_yes
                    weight_no = 0.5 / freq_no
                    weight_yes = (0.5 / freq_no) * (freq_no / freq_yes)
                    weights = torch.tensor(np.vstack((weight_no, weight_yes))).float().cuda()
                print(weights)
                print(weights[:, 4])
        if_ce = 'loss' in params #for backward compatibility: CrossEntropy by default
        if if_ce:
            loss_name = params['loss']
            if loss_name == 'cross-entropy':
                loss_callable = nll
            elif loss_name == 'focal':
                loss_callable = focal
        else:
            loss_callable = nll

        for t in params['tasks']:
            loss_fn[t] = lambda pred, gt: loss_callable(pred, gt, None if not if_weighted_ce else weights[:, int(t)])
        return loss_fn

    if params['dataset'] in ['cifar10', 'cifar10_singletask', 'cifarfashionmnist',
                             'imagenette_singletask']:
        loss_fn = {}
        for t in params['tasks']:
            loss_fn[t] = nll
        return loss_fn