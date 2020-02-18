from graphviz import Digraph
import graphviz
import torch
import pandas as pd
import numpy as np

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/13_55_on_February_11/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
state = torch.load(save_model_path)

learning_scales0 = state['learning_scales0'].cpu().numpy()
learning_scales1 = state['learning_scales1'].cpu().numpy()
learning_scales2 = state['learning_scales2'].cpu().numpy()

num_blocks = [4, 4, 4]
# so learning scales have shapes 4x4, 4x4, 4x40
print(learning_scales2.shape)

learning_scales0_nonzero = np.abs(learning_scales0) > 1e-3#learning_scales0.mean(axis=0)
learning_scales1_nonzero = np.abs(learning_scales1) > 1e-3#learning_scales1.mean(axis=0)
learning_scales2_nonzero = np.abs(learning_scales2) > 1e-3#learning_scales2.mean(axis=0)

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
attr_num = 40
attr_names_dict = dict(zip(range(attr_num), df.columns.values))

g = Digraph('G', filename='cluster.gv')

with g.subgraph(name='cluster_0') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    # c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
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

g.save('graph_cluster.dot')
# graphviz.render('dot', 'png', 'graph_cluster')

# g.view()