import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util import celeba_dict

if_true_cond_prob = True
if_reverse_prob = False# if False: P(column | row), if True: P(column | not row)
if if_true_cond_prob:
    df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
else:
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_120_model.pkl'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_34_on_June_29/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_22_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_23_on_July_01/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_10_model.pkl'
    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    df = pd.read_csv(f'predicted_labels_celeba_{model_name_short}.csv', sep='\s+') #14_53
# print(dict(zip(range(40), df.columns.values)))
df_binary = df == 1
true_freq = df_binary.mean(axis=0)
true_freq.to_csv('class_freqs.csv')
#conditional probability P(attr2|attr1)
cond_probs = np.ones((40, 40))
for i1 in range(len(df.columns.values)):
    for i2 in range(len(df.columns.values)):
        attr1 = df.columns.values[i1]
        attr2 = df.columns.values[i2]

        df_attr1_holds = df[df[attr1] == (1 if not if_reverse_prob else -1)]
        df_attr1_and_attr2_hold = df_attr1_holds[df_attr1_holds[attr2] == 1]
        if len(df_attr1_holds) == 0:
            cond_probs[i1, i2] = 1.0
        else:
            cond_probs[i1, i2] = len(df_attr1_and_attr2_hold) / float(len(df_attr1_holds))
    print(i1, ':', len(df_attr1_holds))

f = plt.figure(figsize=(10.80 * 1.5, 10.80 * 1.5))
plt.tight_layout()
plt.pcolormesh(cond_probs, figure=f, cmap='coolwarm')#cmap='coolwarm'
if if_true_cond_prob:
    title = 'Conditional probabilities P(column|row)'
else:
    title = 'Model-based Conditional probabilities P(column|row)'
if if_reverse_prob:
    title = title.replace('|row', '|not row')
plt.title(title)
ax = plt.gca()
ax.set_yticks(np.arange(.5, 40, 1))
ax.set_yticklabels(['not ' + attr for attr in df.columns.values])
ax.set_xticks(np.arange(.5, 40, 1))
ax.set_xticklabels(df.columns.values, rotation=90)
cb = plt.colorbar(fraction=0.03, pad=0.01)
cb.ax.tick_params(labelsize=6)
if if_true_cond_prob:
    path = f'cond_prob/celeba_cond_prob.svg'
else:
    path = f'cond_prob/celeba_cond_prob_{model_name_short}.svg'
if if_reverse_prob:
    path = path.replace('.svg', '_reversed.svg')
plt.savefig(path, format='svg', bbox_inches='tight', pad_inches=0, dpi=200)

plt.show()


def plot_corr_matrix(df):
    corr_matrix = df.corr()

    mask = np.tri(corr_matrix.shape[0], k=-1)
    mask[(corr_matrix < 0.2) & (corr_matrix > -0.2)] = 1
    corr_matrix = np.ma.array(corr_matrix, mask=mask)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number, vmin=-1, vmax=1, cmap='rainbow')
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig('corrs.jpg')
    # plt.show()

# print(df[df == 1].sum() / df[df == -1].abs().sum())
# print(df.Smiling.corr(df.Wearing_Lipstick))
# print(df.corr().Smiling)

# new_df = df[["Smiling", "Wearing_Lipstick"]]
# new_df.to_csv('/mnt/raid/data/chebykin/pycharm_project_AA/smiling+wearing_lipstick.txt', sep=' ')

'''
{0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
5_o_Clock_Shadow       0.125031
Arched_Eyebrows        0.364220
Attractive             1.051303
Bags_Under_Eyes        0.257184
Bald                   0.022959
Bangs                  0.178655
Big_Lips               0.317169
Big_Nose               0.306391
Black_Hair             0.314494
Blond_Hair             0.173698
Blurry                 0.053628
Brown_Hair             0.258168
Bushy_Eyebrows         0.165729
Chubby                 0.061083
Double_Chin            0.048975
Eyeglasses             0.069655
Goatee                 0.066968
Gray_Hair              0.043787
Heavy_Makeup           0.631114
High_Cheekbones        0.834970
Male                   0.714543
Mouth_Slightly_Open    0.935838
Mustache               0.043346
Narrow_Eyes            0.130133
No_Beard               5.058401
Oval_Face              0.396926
Pale_Skin              0.044874
Pointy_Nose            0.383977
Receding_Hairline      0.086695
Rosy_Cheeks            0.070344
Sideburns              0.059895
Smiling                0.930801
Straight_Hair          0.263267
Wavy_Hair              0.469653
Wearing_Earrings       0.232931
Wearing_Hat            0.050928
Wearing_Lipstick       0.895504
Wearing_Necklace       0.140208
Wearing_Necktie        0.078417
Young                  3.417290
'''

'''
for activation-maximization experiments:

lol, High_Cheekbones correlates highly with Smiling, and the network visualizes smiles for it!
which means that we didn't really learn it! so I'll choose something else

                        (frequency ratio)       (error rate in 23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl')
19:High_Cheekbones        0.834970              12.5
31:Smiling                0.930801              7.2


'''
'''
neuron_idx = 34
wasser_dists = np.load('wasser_dist_attr_hist_lipstickonly.npy', allow_pickle=True).item()
x = np.array(list(wasser_dists[neuron_idx].values()))
x[[0, 3, 4, 7, 8, 10, 12, 13, 14, 15, 16, 17, 20, 22, 23, 28, 30, 35, 38]] *= -1
plt.scatter(x, cond_probs[:, 36]) ; plt.show()
row_corrs = []
column_corrs = []
for i in range(40):
    cur_row_corr = np.corrcoef(x, cond_probs[i, :])[0, 1]
    cur_column_corr = np.corrcoef(x, cond_probs[:, i])[0, 1]
    print(celeba_dict[i], f'{cur_row_corr:.2f}, {cur_column_corr:.2f}')
    row_corrs.append(cur_row_corr)
    column_corrs.append(cur_column_corr)
plt.plot(row_corrs, 'o')
plt.plot(column_corrs, 'o')
ax = plt.gca()
all = range(40)
ax.set_xticks(np.arange(0, len(all), 1))
ax.set_xticklabels([celeba_dict[i] for i in all], rotation=90)
plt.show()
'''