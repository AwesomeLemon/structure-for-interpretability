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

df = pd.read_csv('../../list_attr_celeba.txt', sep='\s+', skiprows=1)
attr_num = 40
attr_names_dict = dict(zip(range(attr_num), df.columns.values))

g = Digraph('G', filename='cluster.gv')
# g.graph_attr['rankdir'] = 'TB'

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

task_clusters_indices = [[8, 9, 11, 17], [16, 15, 3, 22, 38, 0, 12, 35, 24, 30, 7], [39, 28, 26, 23, 14, 13, 5, 4]]
assigned_to_cluster_indices = []

for j in range(len(task_clusters_indices)):
    with g.subgraph(name=f'cluster_{j + 3}') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        cur_cluster_attr_indices = []
        for i in range(attr_num):
            if i in task_clusters_indices[j]:
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

        # mid = len(cur_cluster_attr_indices) // 2
        # for x, y in zip(cur_cluster_attr_indices[:mid], cur_cluster_attr_indices[1:mid+1]):
        #     g.edge(f'fc_{x}', f'fc_{y}', style='invis')
        #
        # for x, y in zip(cur_cluster_attr_indices[mid+1:-1], cur_cluster_attr_indices[mid+2:]):
        #     g.edge(f'fc_{x}', f'fc_{y}', style='invis')

with g.subgraph(name='cluster_whatever') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    cur_cluster_attr_indices = []
    for i in range(attr_num):
        if i not in assigned_to_cluster_indices:
            c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}')
            cur_cluster_attr_indices.append(i)

    num_chains = 2
    l = len(cur_cluster_attr_indices)
    for cur_chain in range(num_chains):
        for x, y in zip(cur_cluster_attr_indices[cur_chain * (l // num_chains) + 1:(cur_chain + 1) * (l // num_chains)],
                        cur_cluster_attr_indices[cur_chain * (l // num_chains) + 2:(cur_chain + 1) * (l // num_chains) + 1]):
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

g.save('graph_cluster.dot')
# graphviz.render('dot', 'png', 'graph_cluster.dot')

# g.view()