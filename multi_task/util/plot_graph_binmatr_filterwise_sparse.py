from graphviz import Digraph
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from util import celeba_dict
from multi_task.util.util import layers


# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_24_on_April_19/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_15_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if_' \
#                   r'_16_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_44_on_April_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[4|_4|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0003|connectivities_l1_all=False|if_13_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_50_on_April_28/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0002|connectivities_l1_all=False|_24_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_37_on_May_15/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_128|_128|_256|_256|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|co_100_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_49_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_128|_128|_256|_256|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=2e-06|co_28_model.pkl'
# THE ONE BELOW!
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
#                   r'|weight_de_58_model_additives.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_35_on_May_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_18_on_May_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/10_49_on_May_25/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
#                   r'|weight_de_49_model_additives.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/18_44_on_May_27/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
#                   r'|weight_de_60_model.pkl'

# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_29_on_June_15/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_49_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_50_on_June_17/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_30_on_June_18/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_57_on_June_19/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_76_model_additives.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_41_on_June_21/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_26_on_June_23/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_120_model.pkl'
# FIRST MODEL FOR WHICH I GOT GRAPH WITH PICS IS THE ONE BELOW
save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_4_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/04_25_on_June_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_31_model.pkl'
# save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_34_on_June_29/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_22_model.pkl'

model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
im_folder_path = f'generated_imshow_{model_name_short}'

if_load_unprocessed_conns = True
if if_load_unprocessed_conns:
    state = torch.load(save_model_path)
    connectivities = state['connectivities']
    learning_scales = list(map(lambda conn: conn.detach().cpu().numpy(), connectivities))
    for ls in learning_scales:
        print(ls.shape)
    num_blocks = list(map(lambda ls: ls.shape[1], learning_scales))
    # so learning scales have shapes 4x4, 4x4, 40x4
    print(learning_scales[2].shape)
    print(learning_scales[2])

    output = {'connectivities': learning_scales}
    torch.save(output, 'visualized_connectivities_id.pkl')
else:
    state = torch.load('visualized_connectivities.pkl')
    learning_scales = state['connectivities']
    for ls in learning_scales:
        print(ls.shape)
    num_blocks = list(map(lambda ls: ls.shape[1], learning_scales))

learning_scales_binary = list(map(lambda x: np.abs(x) > 0.5, learning_scales))

if_visualize_identity_shortcuts = True

if if_visualize_identity_shortcuts:
    if if_load_unprocessed_conns:
        id_shortcut_aux_conns = []
        for i in [0, 4, 8, 12]:
            id_shortcut_aux_conns.append(
                torch.ones((learning_scales_binary[i].shape[0])) == 1)  # don't need 2 dimensions
            id_shortcut_aux_conns += [None] * 3
    else:
        print(state.keys())
        id_shortcut_aux_conns = state['auxillary_connectivities_for_id_shortcut']

proj_shortcut_aux_conns = [None] * 2
for i in [2, 6, 10]:
    proj_shortcut_aux_conns.append(np.copy(learning_scales_binary[i]))
    proj_shortcut_aux_conns += [None] * 3
proj_shortcut_aux_conns += [None] * 2

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
attr_num = 40
attr_names_dict = dict(zip(range(attr_num), df.columns.values))

g = Digraph('G', filename='cluster.gv', node_attr={'shape': 'square',   'fontsize': '15', 'fontcolor' : 'white',
                                                   'width': '.8', 'height': '.8',#'height': '.2',
                                                   'imagescale' : 'false',
                                                   'fixedsize':'true'
                                                   },
            edge_attr={'arrowhead': 'vee', 'penwidth': '.5'})
# g.graph_attr['size'] = '5.75,5.25'
g.attr(None, {'nodesep': '0.05', 'ranksep': '1.0'})
print(g.source)


# g.graph_attr['rankdir'] = 'TB'

def delete_unused():
    potentially_good_nodes = set()
    for j in range(len(learning_scales_binary)):
        for i in range(num_blocks[j]):
            at_least_one_incoming = False
            at_least_one_incoming2 = False
            at_least_one_incoming3 = False
            if j > 0:
                cur_scales_in = learning_scales_binary[j - 1][i, :]
                at_least_one_incoming = np.any(cur_scales_in)
            cur_scales_out = learning_scales_binary[j][:, i]
            at_least_one_outgoing = np.any(cur_scales_out)

            if if_visualize_identity_shortcuts:
                if (j in [0, 4, 8, 12]):  # start of the identity shortcut
                    at_least_one_outgoing = at_least_one_outgoing or id_shortcut_aux_conns[j][i]

            if (j >= 2) and ((j - 2) % 4 == 0) and (j in [2, 6, 10]): #last condition is the true one, but not generalizable; first two allow '14'
                at_least_one_outgoing = at_least_one_outgoing or np.any(proj_shortcut_aux_conns[j][:, i])

            if (j >= 4) and (j % 4 == 0):  # projection shortcut
                cur_scales_in2 = proj_shortcut_aux_conns[j - 2][i, :]
                at_least_one_incoming2 = np.any(cur_scales_in2)
            elif if_visualize_identity_shortcuts and (j >= 2) and ((j - 2) % 4 == 0):  # end of the identity shortcut
                at_least_one_incoming3 = id_shortcut_aux_conns[j - 2][i]

            if False:
                if ((j > 0 and j <= len(learning_scales_binary) - 1) and
                            ((at_least_one_incoming and at_least_one_outgoing) or (((j >= 4) and (j % 4 == 0)) and at_least_one_incoming2 and at_least_one_outgoing))) \
                        or ((j == 0) and (at_least_one_outgoing)) \
                        or (if_visualize_identity_shortcuts and (j >= 2) and (
                        (j - 2) % 4 == 0) and at_least_one_outgoing and at_least_one_incoming3):
                    potentially_good_nodes.add(f'{j}_{i}')
                    if (if_visualize_identity_shortcuts and (j >= 2) and ((j - 2) % 4 == 0) and at_least_one_outgoing and (
                            j == 2)):  # if identity shortcut, that previous node also needs to be added
                        potentially_good_nodes.add(f'{j - 2}_{i}')
                else:
                    learning_scales_binary[j][:, i] = False  # cur_scales_out
                    if j > 0:
                        learning_scales_binary[j - 1][i, :] = False  # cur_scales_in
                    if if_visualize_identity_shortcuts:
                        if (j in [0, 4, 8, 12]):  # start of the identity shortcut
                            id_shortcut_aux_conns[j][i] = False
                        if (j >= 2) and ((j - 2) % 4 == 0):
                            id_shortcut_aux_conns[j - 2][i] = False
            else:
                if_good = True
                if not at_least_one_outgoing: #everyone should have something outgoing
                    if_good = False
                if not (j == 0): # if 0-th layer has something outgoing, it's all right. Otherwise:
                    if not (at_least_one_incoming or at_least_one_incoming2 or at_least_one_incoming3):
                        if_good = False
                        # if j < 14:
                        #     if_good = False
                        # else:
                        #     if if_good:
                        #         print(j, i)
                if if_good:
                    potentially_good_nodes.add(f'{j}_{i}')
                else:
                    learning_scales_binary[j][:, i] = False  # cur_scales_out
                    if j > 0:
                        learning_scales_binary[j - 1][i, :] = False  # cur_scales_in
                    if if_visualize_identity_shortcuts:
                        if (j in [0, 4, 8, 12]):  # start of the identity shortcut
                            id_shortcut_aux_conns[j][i] = False
                        if (j >= 2) and ((j - 2) % 4 == 0):
                            id_shortcut_aux_conns[j - 2][i] = False
                    if (j >= 4) and (j % 4 == 0):  # projection shortcut
                        proj_shortcut_aux_conns[j - 2][i, :] = False
                    pass

    print(len(potentially_good_nodes))
    return potentially_good_nodes


if True:
    for i in range(10):
        potentially_good_nodes = delete_unused()
else:
    potentially_good_nodes = set()
    for i in range(100):
        for j in range(600):
            potentially_good_nodes.add(f'{i}_{j}')

print(potentially_good_nodes)

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
            c.node(f'fc_{i}', label=str(attr_names_dict[i].replace("_", r"\n")), image=f'{im_folder_path}/label/{i}.jpg', scale='False', fontsize='10')

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
        cur_scales_in = learning_scales_binary[j - 1][i, :]
        # at_least_one_incoming = np.any(cur_scales_in)
        # cur_scales_out = learning_scales_binary[j][:, i]
        # at_least_one_outgoing = np.any(cur_scales_out)
        # if ((j > 0 and j < len(learning_scales_binary) - 1) and (at_least_one_incoming and at_least_one_outgoing)) \
        #         or ((j == len(learning_scales_binary) - 1) and (at_least_one_incoming and at_least_one_outgoing)):
        for k, scale_bool in enumerate(cur_scales_in):
            if scale_bool:
                if (f'{j - 1}_{k}' in potentially_good_nodes) and (f'{j}_{i}' in potentially_good_nodes):
                    actually_good_nodes[j - 1].add(f'{j - 1}_{k}')
                    actually_good_nodes[j].add(f'{j}_{i}')
                    # g.edge(f'{j-1}_{k}', f'{j}_{i}')
                    edges_to_add.add((f'{j - 1}_{k}', f'{j}_{i}'))

        if (j >= 4) and (j % 4 == 0):  # shorctut
            cur_scales_in2 = proj_shortcut_aux_conns[j - 2][i, :]
            for k, scale_bool in enumerate(cur_scales_in2):
                if scale_bool:
                    if (f'{j - 2}_{k}' in potentially_good_nodes) and (f'{j}_{i}' in potentially_good_nodes):
                        actually_good_nodes[j - 2].add(f'{j - 2}_{k}')
                        actually_good_nodes[j].add(f'{j}_{i}')
                        # g.edge(f'{j - 2}_{k}', f'{j}_{i}')
                        edges_to_add.add((f'{j - 2}_{k}', f'{j}_{i}'))

        if if_visualize_identity_shortcuts and (j >= 2) and ((j - 2) % 4 == 0):  # identity shortcut
            if (id_shortcut_aux_conns[j - 2][i] and
                f'{j - 2}_{i}' in potentially_good_nodes) and (f'{j}_{i}' in potentially_good_nodes):
                actually_good_nodes[j - 2].add(f'{j - 2}_{i}')
                actually_good_nodes[j].add(f'{j}_{i}')
                # g.edge(f'{j - 2}_{k}', f'{j}_{i}')
                edges_to_add.add((f'{j - 2}_{i}', f'{j}_{i}'))

last_layer_num = len(num_blocks) - 1
for i in range(attr_num):
    cur_scales_in = learning_scales_binary[-1][i, :]
    for j, scale_bool in enumerate(cur_scales_in):
        if scale_bool:
            if f'{last_layer_num}_{j}' in potentially_good_nodes:
                actually_good_nodes[last_layer_num].add(f'{last_layer_num}_{j}')
                # g.edge(f'{last_layer_num}_{j}', f'fc_{i}')
                edges_to_add.add((f'{last_layer_num}_{j}', f'fc_{i}'))

print(f'Actually used nodes num: {sum(map(lambda x: len(x), actually_good_nodes.values()))}')
# list(map(lambda x: print(len(x), sorted(x)), actually_good_nodes.values()))

for j in range(len(learning_scales_binary)):
    with g.subgraph(name=f'cluster_{j}') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in actually_good_nodes[j]:
            layer_idx = int(node[:node.find('_')])
            neuron_idx = node[node.find('_') + 1:]
            c.node(node, image=f'{im_folder_path}/{layers[layer_idx]}/{neuron_idx}.jpg')

for (src, dst) in edges_to_add:
    g.edge(src, dst)

g.save('graph_cluster_binmatr.dot')

for node_name in sorted(actually_good_nodes[last_layer_num]):
    node_ind = int(node_name[node_name.find('_') + 1:])
    cur_scales_out = learning_scales_binary[-1][:, node_ind]
    print(node_name, node_ind, list(map(lambda x: celeba_dict[x], np.where(cur_scales_out)[0])))

learning_scales_binary_out = [np.zeros_like(cur) for cur in learning_scales_binary]
auxillary_connectivities_for_id_shortcut_out = [np.zeros_like(cur) if cur is not None else None
                                                for cur in id_shortcut_aux_conns]

for (src, dst) in edges_to_add:
    src_layer = int(src[:src.find('_')])
    src_neuron = int(src[src.find('_') + 1:])

    dst_neuron = int(dst[dst.find('_') + 1:])
    if 'fc' not in dst:
        dst_layer = int(dst[:dst.find('_')])
    else:
        dst_layer = len(learning_scales_binary)


    if (dst_layer - src_layer) == 1:
        learning_scales_binary_out[src_layer][dst_neuron, src_neuron] = 1.0
        continue
    else:
        assert (dst_layer - src_layer) == 2
        # need to differentiate between projection shortcut & identity shortcut
        # identity:
        if if_visualize_identity_shortcuts:
            if (src_layer in [0, 4, 8, 12]):
                assert src_neuron == dst_neuron
                auxillary_connectivities_for_id_shortcut_out[src_layer][src_neuron] = 1.0
                continue
        # projection:
        if (dst_layer >= 4) and (dst_layer % 4 == 0):
            learning_scales_binary_out[src_layer][dst_neuron, src_neuron] = 1.0
            continue

    raise ValueError(f"Edge {(src, dst)} wasn't assigned anywhere")

output = {'connectivities': learning_scales_binary_out,
          'auxillary_connectivities_for_id_shortcut': auxillary_connectivities_for_id_shortcut_out}

torch.save(output, 'visualized_connectivities.pkl')
np.save('actually_good_nodes.npy', actually_good_nodes)

print('Potential danger: for projection shortcut connectivity is shared with conv1. Thus I enable it when either of those 2 connections exist')
print("Or not. Suppose we have 2_16 -> 4_107. In the real network there'll also be 2_16 -> 3_107. If it had another incoming connections, "
      "it wouldn't be deleted unless it also didn't have outgoing connections. But if it had outgoing connections, it wouldn't be deleted"
      "because it'd have the incoming connection from 2_16")
'''
code for replacing a constant conv with a spatial bias

c = torch.nn.Conv2d(5, 3, 4, 4)
c.bias.data *= 0
x = torch.rand((1, 5, 8, 8))
c2 = copy.deepcopy(c)
disabled = 2
c3 = torch.nn.Conv2d(1,3,4,4)
c3.bias.data *= 0
c3.weight.data = c.weight.data[:, disabled:disabled+1]
c2.bias.data = c3(x[:, disabled:disabled+1]).squeeze()
c3(x[:, disabled:disabled+1]).squeeze()
c2.weight.data[:, disabled:disabled+1, :, :] *= 0
c2.bias.data = copy.deepcopy(c.bias.data)
c2(x)
c2(x) + c3(x[:, disabled:disabled+1])
c(x)

'''