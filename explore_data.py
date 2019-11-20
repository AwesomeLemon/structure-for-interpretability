import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1)
print(list(zip(range(40), df.columns.values)))
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
plt.show()

print(df.sum())
# print(df.Smiling.corr(df.Wearing_Lipstick))
# print(df.corr().Smiling)

# new_df = df[["Smiling", "Wearing_Lipstick"]]
# new_df.to_csv('/mnt/raid/data/chebykin/pycharm_project_AA/smiling+wearing_lipstick.txt', sep=' ')