import torch

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
#                   r'|weight_de_58_model_additives.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/02_16_on_May_23/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
                  r'|weight_de_49_model.pkl'

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_10/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_33_on_June_10/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_03_on_June_11/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_26_on_June_12/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_31_on_June_14/optimizer=SGD_Adam|batch_size=96|lr=0.0|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_deca_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_29_on_June_15/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_49_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_51_on_June_16/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_37_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_55_on_June_16/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_50_on_June_17/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_30_on_June_18/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'


state = torch.load(save_model_path)
n_conns = len(state['connectivities'])
totals = [0] * n_conns
actives = [0] * n_conns
for i, conn in enumerate(state['connectivities']):
    totals[i] = conn.shape[0] * conn.shape[1]

    idx = conn > 0.5
    actives[i] = idx.sum().item()

print('Model path ', save_model_path[37:53], '...', save_model_path[-12:])
print('Total connections  = ', sum(totals))
print('Active connections = ', sum(actives))
print('Active % per layer = ', str([f'{(actives[i] / float(totals[i])) * 100:.0f}' for i in range(n_conns)]).replace("'", ''))
print(f'Active % =  {(sum(actives) / float(sum(totals))) * 100:.2f}')
print('Active # per layer = ', actives)