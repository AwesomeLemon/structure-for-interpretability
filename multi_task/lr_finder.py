import json
import torch

from torch.optim import Adam, SGD
from torch_lr_finder import LRFinder
import losses as losses
import model_selector_automl as model_selector_automl
import datasets as datasets

param_file = 'params/binmatr2_imagenette_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
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

loss_fn = losses.get_loss(params)['all']
tasks = params['tasks']
model = model_selector_automl.get_model(params)
m1 = model['rep']
m2 = model['all']
model = torch.nn.Sequential(m1, m2)
optimizer = SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=1e-4)
# optimizer = Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)
lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda")
train_loader, val_loader, train2_loader = datasets.get_dataset(params, configs)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state