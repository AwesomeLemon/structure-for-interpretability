import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
print(dict(zip(range(40), df.columns.values)))
# corr_matrix = df.corr()
#
# mask = np.tri(corr_matrix.shape[0], k=-1)
# mask[(corr_matrix < 0.2) & (corr_matrix > -0.2)] = 1
# corr_matrix = np.ma.array(corr_matrix, mask=mask)
# f = plt.figure(figsize=(19, 15))
# plt.matshow(corr_matrix, fignum=f.number, vmin=-1, vmax=1, cmap='rainbow')
# plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
# plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

print(df[df == 1].sum() / df[df == -1].abs().sum())
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