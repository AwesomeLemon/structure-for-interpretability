from graphviz import Digraph
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_24_on_April_19/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_15_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if_' \
#                   r'_16_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_44_on_April_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[4|_4|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_13_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_50_on_April_28/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0002|connectivities_l1_all=False|_24_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_37_on_May_15/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_128|_128|_256|_256|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|co_100_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_49_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_128|_128|_256|_256|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|co_28_model.pkl'
# THE ONE BELOW!
save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_35_on_May_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_18_on_May_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_49_on_May_25/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'


state = torch.load(save_model_path)

connectivities = state['connectivities']
learning_scales = list(map(lambda conn: conn.detach().cpu().numpy(), connectivities))
for ls in learning_scales:
    print(ls.shape)
num_blocks = list(map(lambda ls: ls.shape[1], learning_scales))
# so learning scales have shapes 4x4, 4x4, 40x4
print(learning_scales[2].shape)
print(learning_scales[2])

learning_scales_binary = list(map(lambda x: np.abs(x) > 0.5, learning_scales))

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
attr_num = 40
attr_names_dict = dict(zip(range(attr_num), df.columns.values))

g = Digraph('G', filename='cluster.gv', node_attr={'shape': 'rect', #'fontsize': '20',
                                                   'width':'.5', 'height':'.2',
                                                   # 'fixedsize':'true'
                                                   },
            edge_attr={'arrowhead' : 'vee', 'penwidth' : '.5'})
# g.graph_attr['size'] = '5.75,5.25'
g.attr(None, {'nodesep' : '0.05', 'ranksep' : '1.0'})
print(g.source)
# g.graph_attr['rankdir'] = 'TB'

def delete_unused():
    potentially_good_nodes = set()
    for j in range(len(learning_scales_binary)):
        # with g.subgraph(name=f'cluster_{j}') as c:
        #     c.attr(style='filled', color='lightgrey')
        #     c.node_attr.update(style='filled', color='white')
        for i in range(num_blocks[j]):
            at_least_one_incoming = False
            at_least_one_incoming2 = False
            if j > 0:
                cur_scales_in = learning_scales_binary[j-1][i, :]
                at_least_one_incoming = np.any(cur_scales_in)
            cur_scales_out = learning_scales_binary[j][:, i]
            at_least_one_outgoing = np.any(cur_scales_out)

            if (j >= 2) and ((j - 2) % 4 == 0):
                cur_scales_in2 = learning_scales_binary[j - 2][i, :] # shortcut
                at_least_one_incoming2 = np.any(cur_scales_in2)
            if ((j > 0 and j <= len(learning_scales_binary) - 1) and ((at_least_one_incoming and at_least_one_outgoing)
                                      or (((j >= 2) and ((j - 2) % 4 == 0)) and at_least_one_incoming2 and at_least_one_outgoing))) \
                    or ((j == 0) and (at_least_one_outgoing)):
                # c.node(f'{j}_{i}')
                potentially_good_nodes.add(f'{j}_{i}')
                # print(j, i)
            else:
                learning_scales_binary[j][:, i] = False
                if j > 0:
                    learning_scales_binary[j - 1][i, :] = False
                if (j >= 2) and ((j - 2) % 4 == 0):
                    learning_scales_binary[j - 2][i, :] = False
    print(len(potentially_good_nodes))
    return potentially_good_nodes

for i in range(10):
    potentially_good_nodes = delete_unused()

print(1111)

# Create FC nodes:
if False:
    assigned_to_cluster_indices = []
    for j in range(num_blocks[-1]):
        with g.subgraph(name=f'cluster_{j + len(num_blocks)}') as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.update(style='filled', color='white')
            cur_cluster_attr_indices = []
            for i in range(attr_num):
                if learning_scales_binary[-1][i, j] > 0.5 and i not in assigned_to_cluster_indices:
                    c.node(f'fc_{i}', label=f'{attr_names_dict[i].replace("_", "")}')
                    assigned_to_cluster_indices.append(i)
                    cur_cluster_attr_indices.append(i)

            # num_chains = 1
            # l = len(cur_cluster_attr_indices)
            # for cur_chain in range(num_chains):
            #     for x, y in zip(
            #             cur_cluster_attr_indices[cur_chain * (l // num_chains) + 1:(cur_chain + 1) * (l // num_chains):2],
            #             cur_cluster_attr_indices[cur_chain * (l // num_chains) + 2:(cur_chain + 1) * (l // num_chains) + 1:2]):
            #         c.edge(f'fc_{x}', f'fc_{y}', style='invis')
else:
    with g.subgraph(name=f'cluster_{len(num_blocks) + 1}') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for i in range(attr_num):
            c.node(f'fc_{i}', label=str(attr_names_dict[i].replace("_", r"\n")))

        # num_chains = 1
        # l = 40
        # cur_cluster_attr_indices = range(attr_num)
        # for cur_chain in range(num_chains):
        #     for x, y in zip(
        #             cur_cluster_attr_indices[cur_chain * (l // num_chains) + 1:(cur_chain + 1) * (l // num_chains):2],
        #             cur_cluster_attr_indices[cur_chain * (l // num_chains) + 2:(cur_chain + 1) * (l // num_chains) + 1:2]):
        #         c.edge(f'fc_{x}', f'fc_{y}', style='invis')

# Find & save relevant edges:
actually_good_nodes = defaultdict(set)
edges_to_add = set()
for j in range(1, len(learning_scales_binary)):
    for i in range(num_blocks[j]):
        cur_scales_in = learning_scales_binary[j-1][i, :]
        # at_least_one_incoming = np.any(cur_scales_in)
        # cur_scales_out = learning_scales_binary[j][:, i]
        # at_least_one_outgoing = np.any(cur_scales_out)
        # if ((j > 0 and j < len(learning_scales_binary) - 1) and (at_least_one_incoming and at_least_one_outgoing)) \
        #         or ((j == len(learning_scales_binary) - 1) and (at_least_one_incoming and at_least_one_outgoing)):
        for k, scale_bool in enumerate(cur_scales_in):
            if scale_bool:
                if (f'{j-1}_{k}' in potentially_good_nodes) and (f'{j}_{i}' in potentially_good_nodes):
                    actually_good_nodes[j-1].add(f'{j-1}_{k}')
                    actually_good_nodes[j].add(f'{j}_{i}')
                    # g.edge(f'{j-1}_{k}', f'{j}_{i}')
                    edges_to_add.add((f'{j-1}_{k}', f'{j}_{i}'))

        if (j >= 2) and ((j - 2) % 4 == 0):
            cur_scales_in2 = learning_scales_binary[j - 2][i, :]
            for k, scale_bool in enumerate(cur_scales_in2):
                if scale_bool:
                    if (f'{j - 2}_{k}' in potentially_good_nodes) and (f'{j}_{i}' in potentially_good_nodes):
                        actually_good_nodes[j - 2].add(f'{j - 2}_{k}')
                        actually_good_nodes[j].add(f'{j}_{i}')
                        # g.edge(f'{j - 2}_{k}', f'{j}_{i}')
                        edges_to_add.add((f'{j - 2}_{k}', f'{j}_{i}'))

last_layer_num = len(num_blocks) - 1
for i in range(attr_num):
    cur_scales_in = learning_scales_binary[-1][i, :]
    for j, scale_bool in enumerate(cur_scales_in):
        if scale_bool:
            if f'{last_layer_num}_{j}' in potentially_good_nodes:
                actually_good_nodes[last_layer_num].add(f'{last_layer_num}_{j}')
                # g.edge(f'{last_layer_num}_{j}', f'fc_{i}')
                edges_to_add.add((f'{last_layer_num}_{j}', f'fc_{i}'))

for j in range(len(learning_scales_binary)):
    with g.subgraph(name=f'cluster_{j}') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in actually_good_nodes[j]:
            c.node(node)

for (src, dest) in edges_to_add:
    g.edge(src, dest)


g.save('graph_cluster_binmatr.dot')