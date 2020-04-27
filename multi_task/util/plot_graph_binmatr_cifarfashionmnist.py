from graphviz import Digraph
import graphviz
import torch
import pandas as pd
import numpy as np

save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_44_on_April_25/optimizer=SGD_Adam|batch_size=256|lr=0.005|connectivities_lr=0.001|chunks=[8|_8|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_29_model.pkl'

state = torch.load(save_model_path)

connectivities = state['connectivities']
learning_scales0 = connectivities[0].detach().cpu().numpy()
learning_scales1 = connectivities[1].detach().cpu().numpy()
learning_scales2 = connectivities[2].detach().cpu().numpy().T

num_blocks = [8, 8, 4]
print(learning_scales2.shape)
print(learning_scales2)

learning_scales0_nonzero = np.abs(learning_scales0) > 0.5  # learning_scales0.mean(axis=0)
learning_scales1_nonzero = np.abs(learning_scales1) > 0.5  # learning_scales1.mean(axis=0)
learning_scales2_nonzero = np.abs(learning_scales2) > 0.5  # learning_scales2.mean(axis=0)

attr_num = 20
attr_names_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
    10: 'T-shirt/top',
    11: 'Trouser',
    12: 'Pullover',
    13: 'Dress',
    14: 'Coat',
    15: 'Sandal',
    16: 'Shirt',
    17: 'Sneaker',
    18: 'Bag',
    19: 'Ankle boot'
}

g = Digraph('G', filename='cluster.gv')
# g.graph_attr['rankdir'] = 'TB'

with g.subgraph(name='cluster_0') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    for i in range(num_blocks[0]):
        c.node(f'0_{i}')

with g.subgraph(name='cluster_1') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    for i in range(num_blocks[1]):
        c.node(f'1_{i}')

with g.subgraph(name='cluster_2') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    for i in range(num_blocks[2]):
        c.node(f'2_{i}')

assigned_to_cluster_indices = []
for j in range(num_blocks[-1]):
    with g.subgraph(name=f'cluster_{j + 3}') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        cur_cluster_attr_indices = []
        for i in range(attr_num):
            if learning_scales2_nonzero[j, i] > 0.5 and i not in assigned_to_cluster_indices:
                c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}')
                assigned_to_cluster_indices.append(i)
                cur_cluster_attr_indices.append(i)

        num_chains = 1
        l = len(cur_cluster_attr_indices)
        for cur_chain in range(num_chains):
            for x, y in zip(
                    cur_cluster_attr_indices[cur_chain * (l // num_chains) + 1:(cur_chain + 1) * (l // num_chains):2],
                    cur_cluster_attr_indices[
                    cur_chain * (l // num_chains) + 2:(cur_chain + 1) * (l // num_chains) + 1:2]):
                g.edge(f'fc_{x}', f'fc_{y}', style='invis')

if False:
    with g.subgraph(name='cluster_3') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for i in range(attr_num):
            c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}_{i}')

for i in range(num_blocks[1]):
    cur_scales = learning_scales0_nonzero[:, i]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'0_{j}', f'1_{i}')

for i in range(num_blocks[2]):
    cur_scales = learning_scales1_nonzero[:, i]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'1_{j}', f'2_{i}')

for i in range(attr_num):
    cur_scales = learning_scales2_nonzero[:, i]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'2_{j}', f'fc_{i}')

# g.edge('fc_17', 'fc_11', style='invis')
# g.edge('fc_11', 'fc_9')
# g.edge('fc_9', 'fc_8')

g.save('graph_cluster_binmatr_cifarfashionmnist.dot')
# graphviz.render('dot', 'png', 'graph_cluster.dot')

# g.view()
