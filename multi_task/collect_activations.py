import json
import PIL
import numpy as np
from functools import partial
from torchvision import models
from torch.nn.functional import softmax
import torch
import skimage

from multi_task import datasets
from multi_task.load_model import load_trained_model
from PIL import Image
from matplotlib import pyplot as plt
import sklearn.cluster
import sklearn.preprocessing
import sklearn.manifold
import sklearn.mixture
import sklearn_extra.cluster
import pickle


from util.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivityCollector():
    def __init__(self, model, im_size):
        self.model = model
        self.use_my_model = True
        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        for m in self.model:
            model[m].eval()

        self.feature_extractor = self.model['rep']
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)


    def img_from_np_to_torch(self, img):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.use_my_model:
            img -= np.array([0.38302392, 0.42581415, 0.50640459]) * 255#np.array([73.15835921, 82.90891754, 72.39239876])
            std = [0.2903, 0.2909, 0.3114]
        else:
            img -= np.array([0.485, 0.456, 0.406]) * 255
            std = [0.229, 0.224, 0.225]

        img = skimage.transform.resize(img, (self.size_0, self.size_1), order=3)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # HWC -> CWH
        img = img.transpose(2, 0, 1)

        for channel, _ in enumerate(img):
            img[channel] /= std[channel]

        img = torch.from_numpy(img).float()
        img.unsqueeze_(0)
        return img.to(device).requires_grad_(True)

    def img_from_torch_to_int_np(self, im_as_var):
        recreated_im = np.copy(im_as_var.cpu().data.numpy()[0])

        if not self.use_my_model or True:
            std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
            for channel, _ in enumerate(recreated_im):
                recreated_im[channel] *= std[channel]

        recreated_im = recreated_im.transpose((1, 2, 0))

        recreated_im *= 255.0
        if self.use_my_model:
            recreated_im += np.array([73.15835921, 82.90891754, 72.39239876])
        else:
            recreated_im += np.array([0.485, 0.456, 0.406]) * 255

        recreated_im[recreated_im > 255.0] = 255.0
        recreated_im[recreated_im < 0.0] = 0.0

        recreated_im = np.round(recreated_im)
        recreated_im = np.uint8(recreated_im)

        return recreated_im

    def get_activations_single_image(self, input_img_path):
        img = np.asarray(PIL.Image.open(input_img_path))
        img = self.img_from_np_to_torch(img)

        # img = self.img_from_torch_to_int_np(img)
        # im_path = 'test.jpg'
        # save_image(img, im_path)

        def save_activation(activations, name, mod, inp, out):
            if type(out) == list:
                assert len(out) == 40
                for i, out_cur in enumerate(out):
                    activations[f'features_task_{i}'] = out_cur[0].cpu()
            else:
                # why "[0]": this is batch size, which is always == 1 here.
                activations[name] = out[0].cpu()

        activations = {}

        hooks = []

        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        if False:
            self.feature_extractor(img, None)[0]
        else:
            self.feature_extractor(img)

        # a problem that I ignore for now is that for each layer there exist not only 'layer1.1' as final output, but also simply 'layer1'

        for hook in hooks:
            hook.remove()

        return activations

    def store_layer_activations_many(self, configs, target_layer_indices, if_average_spatial=True):
        target_layer_names = [layers[target_layer_idx].replace('_', '.') for target_layer_idx in target_layer_indices]
        activations = {}
        def save_activation(activations, name, mod, inp, out):
            if name in target_layer_names:
                if if_average_spatial:
                    cur_activations = out.mean(dim=(-1, -2)).detach().cpu().numpy()
                else:
                    cur_activations = out.detach().cpu().numpy()
                if name in activations:
                    cur_activations = np.append(activations[name], cur_activations, axis=0)
                activations[name] = cur_activations

        hooks = []
        for name, m in self.feature_extractor.named_modules():
            hooks.append(m.register_forward_hook(partial(save_activation, activations, name)))

        _, loader, _ = datasets.get_dataset(params, configs)
        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                if i % 10 == 0:
                    print(i)

                val_images = batch_val[0].cuda()
                self.feature_extractor(val_images)

        for hook in hooks:
            hook.remove()

        for i, idx in enumerate(target_layer_indices):
            if if_average_spatial:
                filename = f'activations_on_validation_{idx}.npy'
            else:
                filename = f'activations_on_validation_preserved_spatial_{idx}.npy'
            with open(filename, 'wb') as f:
                pickle.dump(activations[target_layer_names[i]], f, protocol=4)
        # np.save(filename, activations)

        return activations

    def cluster_stored_layer_activations(self, target_layer_indices, if_average_spatial=True):
        used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()
        representatives = {}
        for target_layer_idx in target_layer_indices:
            print(target_layer_idx)
            target_layer_name = layers[target_layer_idx].replace('_', '.')
            if if_average_spatial:
                filename = f'activations_on_validation_{target_layer_idx}.npy'
            else:
                filename = f'activations_on_validation_preserved_spatial_{target_layer_idx}.npy'
            # activations = np.load(filename, allow_pickle=True).item()
            with open(filename, 'rb') as f:
                activations = pickle.load(f)
            used_neurons_cur = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[target_layer_idx]])
            print(target_layer_name)
            if type(activations) == dict:
                x = activations[target_layer_name]
            else:
                x = activations
            x = x[:, used_neurons_cur]
            x_shape = x.shape
            x = x.reshape((x.shape[0], -1))
            # scaler = sklearn.preprocessing.StandardScaler()
            # x = scaler.fit_transform(x)
            # clustering = skcl.OPTICS(n_jobs=8, max_eps=5.0).fit(x)
            clustering = sklearn_extra.cluster.KMedoids(n_clusters=300, metric='cosine', init='k-medoids++', max_iter=2000).fit(x)
            # clustering = sklearn.cluster.DBSCAN(n_jobs=8, eps=0.15, metric='cosine', min_samples=10).fit(x)
            # labels, counts = np.unique(clustering.labels_, return_counts=True)
            # gm = sklearn.mixture.GaussianMixture(n_components=10, verbose=1).fit(x)
            # x_mdsed = sklearn.manifold.MDS(n_components=2, n_jobs=8).fit_transform(x)
            # plt.scatter(x_mdsed[:, 0], x_mdsed[:, 1])
            # plt.show()
            # print(clustering.labels_)
            representatives_cur = clustering.cluster_centers_
            representatives[target_layer_name] = representatives_cur.reshape((representatives_cur.shape[0], x_shape[1], x_shape[2], x_shape[3]))
        np.save('representatives.npy', representatives)

    def get_images_open_mouth_nonblond(self, configs):
        _, loader, _ = datasets.get_dataset(params, configs)
        res_images = []
        with torch.no_grad():
            for i, batch_val in enumerate(loader):
                if i % 10 == 0:
                    print(i)
                val_images = batch_val[0]
                label_9 = batch_val[9 + 1].detach().cpu().numpy()
                label_21 = batch_val[21 + 1].detach().cpu().numpy()

                label_9_false = label_9 == 0
                label_21_true = label_21 == 1
                both = np.logical_and(label_9_false, label_21_true)
                if len(np.where(both)[0]) > 0:
                    cur_images = val_images[both].clone().detach()
                    res_images.append(cur_images)

                break

        return res_images


    def compare_AM_images(self, class1: int, class2: int):
        activations1 = self.get_activations_single_image(f'generated_best/{class1}.jpg')
        activations2 = self.get_activations_single_image(f'generated_best/{class2}.jpg')

        # for now compare only layer4
        # shape is CHW torch.Size([512, 24, 21])
        activations1 = activations1['layer4']
        activations2 = activations2['layer4']

        strengths1 = []
        strengths2 = []

        assert activations1.size(0) == activations2.size(0)
        for i in range(activations1.size(0)):
            strengths1.append(torch.norm(activations1[i], 1).item())
            strengths2.append(torch.norm(activations2[i], 1).item())

        strengths1 = np.array(strengths1)
        strengths2 = np.array(strengths2)

        plt.hist(strengths1 - strengths2)
        plt.show()

        plt.plot(strengths1, 'o')
        plt.plot(strengths2, 'o')
        plt.show()

        print(strengths1 - strengths2)
        print(len(np.arange(512)[np.abs(strengths1 - strengths2) > 400.0]))

    def get_output_probs_single_image(self, input_img_path):
        img = np.asarray(PIL.Image.open(input_img_path))
        img = self.img_from_np_to_torch(img)

        out = self.feature_extractor(img)

        all_tasks = [str(task) for task in range(40)]
        probs = np.ones((40)) * -17
        for i, t in enumerate(all_tasks):
            out_t_val, _ = self.model[t](out[i], None)
            probs[i] = torch.nn.functional.softmax(out_t_val, dim=1)[0][1].item()

        return probs

    def get_feature_distribution_per_class(self, target: int, n_images: int, folder_suffix):
        strengths_sum = None
        for i in range(n_images):
            activations = self.get_activations_single_image(f'generated_10_{folder_suffix}/{target}_{i}.jpg')

            # for now compare only layer4
            # shape is CHW torch.Size([512, 24, 21])
            activations_last = activations['layer4']

            strengths = []
            for i in range(activations_last.size(0)):
                strengths.append(torch.norm(activations_last[i], 1).item())

            strengths = np.array(strengths)
            if strengths_sum is None:
                strengths_sum = strengths
            else:
                strengths_sum += strengths

        return strengths_sum / float(n_images)

    def get_target_probs_per_class(self, target: int, n_images: int, folder_suffix):
        probs_sum = None
        for i in range(n_images):
            probs = self.get_output_probs_single_image(f'generated_separate_{folder_suffix}/{target}_{i}.jpg')
            probs = np.array(probs)
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs

        return probs_sum / float(n_images)

    def get_feature_distribution_many(self, targets, n_images, folder_suffix):
        n_features = 512
        res = np.ones((len(targets), n_features))
        targets = [int(target) for target in targets]
        for target in targets:
            print(target)
            res[target] = self.get_feature_distribution_per_class(target, n_images, folder_suffix)
        return res

    def get_target_probs_many(self, targets, n_images, folder_suffix):
        n_tasks = 40
        res = np.ones((len(targets), n_tasks))
        targets = [int(target) for target in targets]
        for i, target in enumerate(targets):
            print(target)
            res[i] = self.get_target_probs_per_class(target, n_images, folder_suffix)
        return res

    def visualize_feature_distribution(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        feature_distr = self.get_feature_distribution_many(targets, n_images, folder_suffix)
        for i in range(512):
            # feature_distr[:, i][feature_distr[:, i] < np.percentile(feature_distr[:, i], 75)] = 0
            feature_distr[:, i] -= feature_distr[:, i].mean()
            feature_distr[:, i] /= feature_distr[:, i].std()
        f = plt.figure(figsize=(19.20 * 1.55, 10.80 * 1.5))
        plt.tight_layout()
        # plt.matshow(feature_distr, fignum=f.number, vmin=-1, vmax=1, cmap='rainbow')
        # plt.imshow(feature_distr, aspect='auto')
        plt.pcolormesh(feature_distr,figure=f)#,cmap='rainbow')#,vmax=1200)#, edgecolors='k', linewidth=1)
        plt.xticks(range(512), [])
        # plt.yticks(range(len(targets)), [celeba_dict[target] for target in targets], fontsize=14)
        ax = plt.gca()
        ax.set_yticks(np.arange(.5, len(targets), 1))
        ax.set_yticklabels([celeba_dict[target] for target in targets])
        # ax.set_yticks(np.arange(-.5, len(targets), 1), minor=True)
        ax.set_aspect('auto')
        ax.get_xaxis().set_visible(False)
        cb = plt.colorbar(fraction=0.03, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        plt.savefig(f'features_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=1200)
        plt.show()

    def visualize_feature_histograms_per_task(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        feature_distr = self.get_feature_distribution_many(targets, n_images, folder_suffix)
        for i in range(512):
            # feature_distr[:, i][feature_distr[:, i] < np.percentile(feature_distr[:, i], 75)] = 0
            feature_distr[:, i] -= feature_distr[:, i].mean()
            feature_distr[:, i] /= feature_distr[:, i].std()

        f = plt.figure(figsize=(19.20 * 1.55, 10.80 * 1.5))
        ax = f.subplots(nrows=5, ncols=8)

        for task in range(40):
            row = task // 8
            col = task - row * 8
            ax[row, col].hist(feature_distr[task, :], range=(-3, 3), bins=15)
            ax[row, col].set_ylim((0, 175))
            ax[row, col].set_title(celeba_dict[task])

        f.subplots_adjust(hspace=0.4)

        plt.savefig(f'feature_hists_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=1200)


    def visualize_probs_distribution(self, targets, n_images, folder_suffix):
        celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}
        probs = self.get_target_probs_many(targets, n_images, folder_suffix)
        f = plt.figure(figsize=(10.80 * 1.5, 10.80 * 1.5))
        plt.tight_layout()
        plt.pcolormesh(probs,figure=f,vmin=0, vmax=1)#, edgecolors='k', linewidth=1)
        ax = plt.gca()
        ax.set_yticks(np.arange(.5, len(targets), 1))
        ax.set_yticklabels([celeba_dict[target] for target in targets])
        all = range(40)
        ax.set_xticks(np.arange(.5, len(all), 1))
        ax.set_xticklabels([celeba_dict[i] for i in all], rotation=90)
        cb = plt.colorbar(fraction=0.03, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        plt.savefig(f'probs_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.show()

if __name__ == '__main__':
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
    # param_file = 'old_params/sample_all.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
    # param_file = 'params/bigger_reg_4_4_4.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_42_on_April_17/optimizer=SGD|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|if_fully_connected=True|use_pretrained_17_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgd001_pretrain_fc.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
    # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
    # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0_20_model.pkl'
    param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_nocondecay_comeback.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall2e-6_comeback_rescaled.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_6_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
    # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'

    trained_model = load_trained_model(param_file, save_model_path)

    with open(param_file) as json_params:
        params = json.load(json_params)
    # params['input_size'] = 'default'
    if params['input_size'] == 'default':
        im_size = (64, 64)
        config_path = 'configs.json'
    elif params['input_size'] == 'bigimg':
        im_size = (128, 128)
        config_path = 'configs_big_img.json'
    elif params['input_size'] == 'biggerimg':
        im_size = (192, 168)
        config_path = 'configs_bigger_img.json'

    with open(config_path) as config_params:
        configs = json.load(config_params)

    model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    ac = ActivityCollector(trained_model, im_size)
    # ac.compare_AM_images(31, 12)
    # ac.visualize_feature_distribution(range(40), 10, 'binmatr')
    # ac.visualize_feature_histograms_per_task(range(40), 10, 'binmatr')
    # ac.visualize_probs_distribution(range(40), 10, model_name_short)
    # for i in range(4):
        # print(ac.get_output_probs(f'big_diverse_generated_separate_12_18_on_June_24...46/label/33_{i}.jpg')[23])
        # print(celeba_dict[i], ac.get_output_probs_single_image(f'big_diverse_generated_separate_12_18_on_June_24...46/label/{i}_0.jpg')[9])
        # print(ac.get_output_probs_single_image(f'big_diverse_generated_separate_12_18_on_June_24...46/label/21_{i}.jpg')[9])
        # print(celeba_dict[i], ac.get_output_probs_single_image(f'generated_archive/generated_separate_12_18_on_June_24...46/label/{i}_0.jpg')[11])
        # print(ac.get_output_probs(f'generated_separate/23_{i}.jpg')[23])

    # ac.store_layer_activations_many(configs, range(1, 5), False)
    # ac.store_layer_activations_many(configs, range(5, 14), False)
    # ac.store_layer_activations_many(configs, [14], False)
    # ac.cluster_stored_layer_activations([14], False)
    ac.cluster_stored_layer_activations(list(range(15)), False)
    # ims = ac.get_images_open_mouth_nonblond(configs)
    x = 1
    '''
    out = self.feature_extractor(val_images[both].cuda())
    pred21 = self.model['21'].linear(out[21])
    pred9 = self.model['9'].linear(out[9])
    
    potential_idx = torch.nn.functional.softmax(pred9, dim=1)[:, 1] > 0.6
    potential_cur_images = cur_images[potential_idx]
    x = potential_cur_images.numpy()[:, ::-1, :, :]
    x = np.transpose(x, [0, 2, 3, 1])
    for i in range(x.shape[0]):
        plt.imshow(x[i])
        plt.show()
        
    victim = x[3]
    v = np.array(victim.copy())
    v[43:50, 26:40] = 0 
    v[43:50, 22:40] = 0 
    plt.imshow(v) ; plt.show()
    
    vt = np.array(v.copy())[:, :, ::-1].transpose([2, 0, 1])[None, ...]
    vt = torch.tensor(np.array(vt))
    out = self.feature_extractor(vt.cuda())
    pred21_2 = self.model['21'].linear(out[21])
    pred9_2 = self.model['9'].linear(out[9])
    torch.nn.functional.softmax(pred9_2, dim=1)
    torch.nn.functional.softmax(pred21_2, dim=1)
    
    v[45, 24:40] = v[44, 29]
    v[46, 26:38] = v[44, 29]
    v[47, 27:36] = v[44, 29]
    v[48, 27:36] = v[44, 29]
    '''