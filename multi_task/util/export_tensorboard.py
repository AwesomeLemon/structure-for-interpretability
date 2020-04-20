from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import pyplot as plt
import pandas

size_guidance={ # see below regarding this argument
    event_accumulator.COMPRESSED_HISTOGRAMS: 0,
    event_accumulator.IMAGES: 0,
    event_accumulator.AUDIO: 0,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 0,
}

def export_avg_error():
    # chronological order:
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/11_16AM_on_November_05_2019/events.out.tfevents.1572948998.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/11_49PM_on_November_05_2019/events.out.tfevents.1572994163.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/04_44PM_on_November_06_2019/events.out.tfevents.1573055088.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/09_14AM_on_November_08_2019/events.out.tfevents.1573200854.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/02_44AM_on_November_09_2019/events.out.tfevents.1573263841.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/06_15AM_on_November_10_2019/events.out.tfevents.1573362953.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator('/mnt/antares_raid/home/awesomelemon/runs6/09_46AM_on_November_11_2019/events.out.tfevents.1573462004.eltanin', size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator(
        '/mnt/antares_raid/home/awesomelemon/runs6/12_28PM_on_November_12_2019/events.out.tfevents.1573558116.eltanin',
        size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator(
        '/mnt/antares_raid/home/awesomelemon/runs6/03_48PM_on_November_12_2019/events.out.tfevents.1573570125.eltanin',
        size_guidance=size_guidance)
    ea = event_accumulator.EventAccumulator(
        '/mnt/antares_raid/home/awesomelemon/runs6/03_50PM_on_November_12_2019/events.out.tfevents.1573570201.eltanin',
        size_guidance=size_guidance)

    ea.Reload()

    ea.Tags()

    def calc_sums():
        sums = []

        for j in range(len(ea.Scalars(f'metric_acc_0'))):
            cur_sum = 0

            for i in range(40):
                cur_sum += (1 - ea.Scalars(f'metric_acc_{i}')[j].value)

            cur_sum /= 40.
            sums.append(cur_sum)
        return sums

    ea.Scalars(f'validation_loss')

    print('smth')

    '''
    nohup sh -c "sleep 37h ; cd pycharm_project_AA_more_blocks ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_init_ones ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_different_lr_schedule ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_smaller_lr ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_batch128 ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_slower_sigmoid ; python multi_task/train_multi_task_automl.py ; cd ../pycharm_project_AA_95_5_train ; python multi_task/train_multi_task_automl.py" &
    
    '''

# Extracting connection strengths

root_path = '/mnt/antares_raid/home/awesomelemon/runs6/04_44PM_on_November_06_2019/'
# root_path = '/mnt/antares_raid/home/awesomelemon/runs6/01_54PM_on_November_19_2019/'
connection_strengths = np.ones((40, 8, 100)) * -17#tasks * connections * time points
for i in range(40):
    for j in range(8):
        ea = event_accumulator.EventAccumulator(
            root_path + f'learning_scales_3_{i}_sigmoid/{j}',
            size_guidance=size_guidance)
        ea.Reload()
        connection_strengths[i, j, :] = list(map(lambda x: x.value, ea.Scalars(f'learning_scales_3_{i}_sigmoid')))

print(connection_strengths)

# for j in range(8):
#     plt.plot(connection_strengths[0, j, :])
# plt.show()

for epoch in range(0, 21, 5):
    # connection_strengths_best = np.expand_dims(connection_strengths[:, 0, epoch], axis=1)
    connection_strengths_best = connection_strengths[:, :, epoch]
    res = np.ones((40, 40)) * -17
    for i in range(40):
        for j in range(i + 1, 40):
            res[i, j] = np.square(np.subtract(connection_strengths_best[i], connection_strengths_best[j])).mean()
    corr_matrix = res
    # df = pandas.DataFrame(connection_strengths_best.T)
    # corr_matrix = df.corr('pearson')
    mask = np.tri(corr_matrix.shape[0], k=-1)
    # mask[(corr_matrix < 0.8) & (corr_matrix > -0.8)] = 1
    corr_matrix = np.ma.array(corr_matrix, mask=mask)
    f = plt.figure(figsize=(19, 15))
    # plt.matshow(corr_matrix, fignum=f.number, vmin=-1, vmax=1, cmap='viridis')
    plt.matshow(corr_matrix, fignum=f.number, vmin=-1, cmap='viridis')
    df2 = pandas.read_csv('../../list_attr_celeba.txt', sep='\s+', skiprows=1)
    plt.xticks(range(40), df2.columns, fontsize=14, rotation=90)
    plt.yticks(range(40), df2.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(f'{epoch}')
    plt.show()

