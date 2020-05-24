from graphviz import Digraph
import graphviz
import torch
import pandas as pd
import numpy as np

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_24_on_April_19/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_15_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if_' \
#                   r'_16_model.pkl'
save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_44_on_April_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[4|_4|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_13_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_50_on_April_28/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0002|connectivities_l1_all=False|_24_model.pkl'

state = torch.load(save_model_path)

connectivities = state['connectivities']
learning_scales0 = connectivities[0].detach().cpu().numpy()
learning_scales1 = connectivities[1].detach().cpu().numpy()
learning_scales2 = connectivities[2].detach().cpu().numpy()

num_blocks = [4, 4, 4]
# so learning scales have shapes 4x4, 4x4, 4x40
print(learning_scales2.shape)
print(learning_scales2)

learning_scales0_nonzero = np.abs(learning_scales0) > 0.5#learning_scales0.mean(axis=0)
learning_scales1_nonzero = np.abs(learning_scales1) > 0.5#learning_scales1.mean(axis=0)
learning_scales2_nonzero = np.abs(learning_scales2) > 0.5#learning_scales2.mean(axis=0)

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
attr_num = 40
attr_names_dict = dict(zip(range(attr_num), df.columns.values))

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
            if learning_scales2_nonzero[i, j] > 0.5 and i not in assigned_to_cluster_indices:
                c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}')
                assigned_to_cluster_indices.append(i)
                cur_cluster_attr_indices.append(i)

        num_chains = 1
        l = len(cur_cluster_attr_indices)
        for cur_chain in range(num_chains):
            for x, y in zip(
                    cur_cluster_attr_indices[cur_chain * (l // num_chains) + 1:(cur_chain + 1) * (l // num_chains):2],
                    cur_cluster_attr_indices[cur_chain * (l // num_chains) + 2:(cur_chain + 1) * (l // num_chains) + 1:2]):
                c.edge(f'fc_{x}', f'fc_{y}', style='invis')

if False:
    with g.subgraph(name='cluster_3') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for i in range(attr_num):
            c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}_{i}')

for i in range(num_blocks[1]):
    cur_scales = learning_scales0_nonzero[i, :]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'0_{j}', f'1_{i}')

for i in range(num_blocks[2]):
    cur_scales = learning_scales1_nonzero[i, :]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'1_{j}', f'2_{i}')

for i in range(attr_num):
    cur_scales = learning_scales2_nonzero[i, :]
    for j, scale_bool in enumerate(cur_scales):
        if scale_bool:
            g.edge(f'2_{j}', f'fc_{i}')

# g.edge('fc_17', 'fc_11', style='invis')
# g.edge('fc_11', 'fc_9')
# g.edge('fc_9', 'fc_8')

g.save('graph_cluster_binmatr.dot')
# graphviz.render('dot', 'png', 'graph_cluster.dot')

# g.view()