"""
Based on:
(1) https://github.com/utkuozbulak/pytorch-cnn-visualizations/
(2) https://github.com/tensorflow/lucid
(3) https://github.com/greentfrapp/lucent
"""
import time
from collections import OrderedDict, defaultdict

from pathlib import Path

from torch import nn
from torch.nn.functional import softmax
from torch.optim import SGD, Adam
from torchvision import models

# from multi_task.load_model import load_trained_model
try:
    from load_model import load_trained_model
    from loaders.celeba_loader import CELEBA
except:
    from multi_task.load_model import load_trained_model
    from multi_task.loaders.celeba_loader import CELEBA

# from multi_task.util.util import *
import torch
import numpy as np
from util.util import *
import skimage
import imageio
from siren import SirenNet
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size_multiplier = 4#6#4#1

def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


class ClassSpecificImageGenerator():
    def __init__(self, model, target, im_size, batch_size=1, use_my_model=True, hook_dict=None,
                 diversity_layer_name=None, force_rep_indices=[], if_cppn=False, if_cifar=False, if_imagenette=False,
                 tensor_loss_mul=1):
        self.target_layer_name, self.neuron_idx = target

        self.model = model
        self.use_my_model = use_my_model
        self.if_cifar = if_cifar
        self.if_imagenette = if_imagenette

        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        self.batch_size = batch_size
        self.execute_trunk = lambda img: self.feature_extractor(img)
        self.tensor_loss_mul = tensor_loss_mul

        if self.use_my_model:
            self.means = torch.tensor([0.38302392, 0.42581415, 0.50640459]).to(device)
            self.std = torch.tensor([0.2903, 0.2909, 0.3114]).to(device)
            if if_cifar:
                self.means = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
                # self.std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
                self.std = torch.tensor([0.24716828900163684, 0.2437065869717477, 0.26169213143713593]).to(device)
            if if_imagenette:
                self.means = torch.tensor([109.5388 / 255., 118.6897 / 255., 124.6901 / 255.]).to(device)
                self.std = torch.tensor([0.224, 0.224, 0.224]).to(device)
        else:
            self.means = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])

        rep_dict_loaded = np.load('representatives_kmeans14_50_alllayers.npy', allow_pickle=True).item()
        # rep_dict_loaded = np.load('representatives_kmeans14_50.npy', allow_pickle=True).item()
        used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()
        rep_dict = {}
        for rep_layer_idx in force_rep_indices:
            rep_layer_name = layers[rep_layer_idx].replace('_', '.')
            rep_used_neurons = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[rep_layer_idx]])
            # rep = rep_dict_loaded[rep_layer_name].reshape((rep_dict_loaded[rep_layer_name].shape[0], -1))
            rep = rep_dict_loaded[rep_layer_name][:, rep_used_neurons, :, :].mean(axis=(-1, -2))
            rep = torch.tensor(rep).to(device)
            rep_dict[rep_layer_name] = rep
            self.n_reps = rep_dict_loaded[rep_layer_name].shape[0]  # is the same everywhere

        self.hook_dict = hook_dict
        if self.use_my_model:
            self.feature_extractor = self.model['rep']
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)

            self.get_probs = None
            if self.target_layer_name == 'label':
                target_class = self.neuron_idx
                if_single_head = False
                if str(target_class) not in self.model:
                    print('Assume that this is a CIFAR10 single-head network')
                    self.classifier = self.model['all']
                    if_single_head = True
                else:
                    self.classifier = self.model[str(target_class)]
                self.classifier.eval()
                for param in self.classifier.parameters():
                    param.requires_grad_(False)
                if False:
                    self.irrelevant_idx = list(range(target_class)) + list(range(target_class + 1, 40))
                    self.irrelevant_idx = list(filter(lambda x: x != 10, self.irrelevant_idx))
                else:
                    self.irrelevant_idx = []
                print(self.irrelevant_idx)
                self.irrelevant_classifiers = {idx: self.model[str(idx)] for idx in self.irrelevant_idx}
                for c in self.irrelevant_classifiers.values():
                    c.eval()

                if False:
                    multipliers = torch.zeros((512)).to(device)
                    multipliers[[235, 316, 490, 172, 164, 342, 28]] = 0.5 * torch.tensor(
                        [2.5833, 7.0048, 3.4474, 3.2662, 5.7719, 2.6060, 1.0000]).to(device)
                    self.get_target_tensor = lambda img, trunk_features: self.classifier.linear(
                        multipliers * torch.tanh(trunk_features[target_class] / 10))[:, 1].mean()
                else:
                    if False:
                        self.get_target_tensor = lambda img, trunk_features: self.classifier.linear(
                            100 * torch.tanh(trunk_features[target_class] / 10))[:, 1].mean()
                    else:
                        if not if_single_head:
                            self.get_target_tensor = lambda img, trunk_features: self.classifier.linear(
                                trunk_features[target_class])[:, 1].mean()
                        else:
                            self.get_target_tensor = lambda img, trunk_features: self.classifier.linear(
                                trunk_features[0])[:, target_class].mean()
                    if not if_single_head:
                        self.get_probs = lambda img, trunk_features: torch.nn.functional.softmax(
                            self.classifier.linear(trunk_features[target_class]), dim=1)[:, 1].mean().item()
                    else:
                        self.get_probs = lambda img, trunk_features: torch.nn.functional.softmax(
                            self.classifier.linear(trunk_features[0]), dim=1)[:, target_class].mean().item()
            else:
                if True:
                    self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:,
                                                        self.neuron_idx].mean()
                else:
                    #optimize only for center neuron instead of the whole feature map
                    self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:,
                                                            self.neuron_idx][
                        :, self.hook_dict[self.target_layer_name].features.shape[-2] // 2,
                            self.hook_dict[self.target_layer_name].features.shape[-1] // 2]
                self.get_probs = lambda x, y: None
        else:
            model.eval()
            self.feature_extractor = self.model
            # self.get_target_tensor = lambda _, trunk_features: trunk_features[0, self.neuron_idx]
            if True:
                self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:,
                                                        self.neuron_idx].mean()
            else:
                print('optimize only for center neuron instead of the whole feature map')
                self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:,self.neuron_idx][:,
                                                        self.hook_dict[self.target_layer_name].features.shape[-2] // 2,
                                                    self.hook_dict[self.target_layer_name].features.shape[-1] // 2].mean() # mean is for batch > 1
                # self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:,
                #                                         self.neuron_idx][:,3, 3]
            self.get_probs = lambda x, y: None

        def inner_calculate_loss_f(self, image_transformed, trunk_features, n_step):
            output = self.get_target_tensor(image_transformed, trunk_features)
            target_weight = 2
            tensor_loss = -output * target_weight * self.tensor_loss_mul
            if not self.use_my_model:
                imagenet_reg = 1e4 * 1e-10 * (10e5 * 60e-9 * 30000) ** 1 * 0.0005 ** 3 * torch.norm(self.image_transformed, 1) \
                               + 1e-4 * 1e3 ** 0 * 4 ** 0 * (10e7 * 3e-8 * 45000 *1.5) ** 0 * 0.0001 * 0.001 ** 1.5 * total_variation_loss(self.image_transformed, 2)
                reg = imagenet_reg
            elif self.if_cifar:
                cifar_reg = 1e-3 * torch.norm(self.image_transformed, 1) \
                               + 1e-3 * total_variation_loss(self.image_transformed, 2)
                reg = 0#cifar_reg
            elif self.if_imagenette:
                reg = 0
            else: # celeba
                celeba_reg = 0.0005 ** 3 * torch.norm(self.image_transformed, 1) \
                               + 0.0001 * 0.001 ** 1.5 * total_variation_loss(self.image_transformed, 2)
                reg = celeba_reg
            class_loss = tensor_loss + reg
            if n_step % 60 == 0:
                probs = self.get_probs(image_transformed, trunk_features)
                if probs is not None:
                    print('Prob = ', probs)
                print('Loss without regularization = ', tensor_loss.item())

            if diversity_layer_name is not None:
                tensor = self.hook_dict[diversity_layer_name].features
                batch, channels, _, _ = tensor.shape
                flattened = tensor.view(batch, channels, -1)
                grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
                grams = torch.nn.functional.normalize(grams, p=2, dim=(1, 2))
                if True:
                    diversity_loss = sum([sum([(grams[i] * grams[j]).sum()
                                               for j in range(batch) if j != i])
                                          for i in range(batch)]) / batch
                else:
                    diversity_loss = 0.
                    for i in range(batch):
                        for j in range(batch):
                            if j != i:
                                cur = (grams[i] * grams[j]).sum()
                                if cur > 1.0:
                                    cur = cur ** 6
                                diversity_loss += cur

                    class_loss *= 0
                diversity_mul = 1.
                if True:
                    tensor_loss_power = np.log10(tensor_loss.item())
                    diversity_loss_power = np.log10(diversity_loss.item())
                    if diversity_loss_power > tensor_loss_power:
                        diversity_mul *= 10 ** (tensor_loss_power - diversity_loss_power)
                    # print(diversity_mul)
                class_loss += 0.1 ** 0 * diversity_mul * diversity_loss # * 1e-6**0 # * 10

            n_step_start_force_rep = 180
            if (len(force_rep_indices) > 0) and (n_step >= n_step_start_force_rep):
                if n_step == n_step_start_force_rep:  # this 'if' is run exactly once
                    self.target_rep_indices = np.ones((self.batch_size, 15), dtype=int) * -17
                for layer_rep_idx in force_rep_indices:
                    if layer_rep_idx == 2:
                        # TODO: for some reason, in 2nd layer the spatial dims are 64x64 in current activations, and 32x32 in rep
                        # in all the other layers spatial dims are the same between current activations and rep
                        continue
                    rep_used_neurons = np.array([int(x[x.find('_') + 1:]) for x in used_neurons[layer_rep_idx]])
                    rep_layer_name = layers[layer_rep_idx]
                    tensor = self.hook_dict[rep_layer_name].features[:, rep_used_neurons]
                    # flattened = tensor.view(self.batch_size, -1)
                    flattened = tensor.mean(axis=(-1, -2))
                    total = 0.
                    if n_step == n_step_start_force_rep:  # this 'if' is run exactly once
                        with torch.no_grad():
                            for b in range(self.batch_size):
                                if True:
                                    max_val = -1.0
                                    for r in range(self.n_reps):
                                        cur_rep = rep_dict[rep_layer_name.replace('_', '.')][r]
                                        cur_flattened = flattened[b]
                                        cosine_sim = cur_flattened.dot(cur_rep) / (
                                                torch.norm(cur_flattened, p=2) * torch.norm(cur_rep, p=2))
                                        if cosine_sim > max_val:
                                            self.target_rep_indices[b, layer_rep_idx] = r
                                else:
                                    # for getting activations of lower layers, given a k-means centroid for the last layer
                                    self.target_rep_indices[b, layer_rep_idx] = b
                        #             print(cosine_sim.item())
                        # print()

                    for i in range(self.batch_size):
                        target_rep_idx = self.target_rep_indices[i, layer_rep_idx]
                        rep = rep_dict[rep_layer_name.replace('_', '.')][target_rep_idx]
                        # dot = flattened[i].dot(rep)
                        cosine_sim = flattened[i].dot(rep) / (torch.norm(flattened[i], p=2) * torch.norm(rep, p=2))
                        total += - (((cosine_sim - 2) ** 2) ** (
                                10 / 14. + layer_rep_idx * (4 / 196.)))  # max(0.01, cosine_sim) ** 8
                        if n_step % 60 == 0:
                            print(cosine_sim.item())
                    class_loss -= 0.1 * total  # .mean()
            return class_loss

        self.calculate_loss_f = lambda img, trunk_features, n_step: inner_calculate_loss_f(self, img, trunk_features,
                                                                                           n_step)

        if self.if_imagenette or not self.use_my_model:
            color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                                     [0.27, 0.00, -0.05],
                                                     [0.27, -0.09, 0.03]]).astype("float32")
        else:
            color_correlation_svd_sqrt = np.asarray([[0.22064119, 0.15331138, 0.10992278],
                                                     [0.15331138, 0.19750742, 0.1486874],
                                                     [0.10992278, 0.1486874, 0.25062584]]).astype("float32")
            # color_correlation_svd_sqrt = np.asarray([[ 9.86012461, -7.94772474,  0.39051937],
            #                                        [-7.94772474, 15.5556668 , -5.74280639],
            #                                        [ 0.39051937, -5.74280639,  7.22573533]]).astype('float32')
            # color_correlation_svd_sqrt = np.asarray([[0.25062584, 0.1486874, 0.10992278],
            #                                          [0.1486874, 0.19750742, 0.15331138],
            #                                          [0.10992278, 0.15331138, 0.22064119]]).astype("float32")
            if self.if_cifar:
                color_correlation_svd_sqrt = np.asarray([[0.19822608, 0.11846086, 0.08812269],
                                                         [0.11846084, 0.17447087, 0.12214683],
                                                         [0.08812269, 0.12214684, 0.21400307]]).astype("float32")
        max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt  # multiply by more => brighter image

        # WARNING: the names of 2 funs below are legacy from confusing lucid stuff; they do opposite things, but not sure which is which
        def _linear_decorrelate_color(tensor):
            t_permute = tensor.permute(0, 2, 3, 1)
            t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
            tensor = t_permute.permute(0, 3, 1, 2)
            return tensor

        def _linear_correlate_color(tensor):
            t_permute = tensor.permute(0, 2, 3, 1)
            t_permute = torch.matmul(t_permute, torch.tensor(np.linalg.inv(color_correlation_normalized.T)).to(device))
            tensor = t_permute.permute(0, 3, 1, 2)
            return tensor

        if not if_cppn:
            self.if_fourier = True
            if self.if_fourier:
                h, w = self.size_0 * size_multiplier, self.size_1 * size_multiplier
                freqs = rfft2d_freqs(h, w)
                channels = 3
                init_val_size = (batch_size, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
                sd = 0.01  # 0.1
                decay_power = 1.0  # 0.6

                if_start_from_existing = False
                if if_start_from_existing:
                    if_start_from_dataset = True
                    if if_start_from_dataset:
                        with open('configs.json') as config_params:
                            configs = json.load(config_params)
                        test_dst = CELEBA(root=configs['celeba']['path'], is_transform=False, split='val',
                                          img_size=(h, w), augmentations=None)
                        # img = test_dst.__getitem__(0)[0]
                        # img = test_dst.get_item_by_path('/mnt/raid/data/chebykin/celeba/Img/img_align_celeba_png/171070.png') #broad-faced guy
                        # img = test_dst.get_item_by_path('/mnt/raid/data/chebykin/celeba/Img/img_align_celeba_png/180648.png')  # blond smiling gal
                        img = test_dst.get_item_by_path('/mnt/raid/data/chebykin/celeba/Img/img_align_celeba_png/170028.png') #black-haired dude & pink-shirt woman
                        # img = test_dst.get_item_by_path('/mnt/raid/data/chebykin/celeba/Img/img_align_celeba_png/174565.png') #brown-haired woman on orange background
                        # img = img[-175:, 15:125, :]
                    else:
                        # img = imageio.imread('12_479.jpg')
                        # img = imageio.imread('7_57.jpg')
                        img = imageio.imread('blond_hair_cppn.jpg')

                    img = img[:, :, ::-1]
                    img = img.astype(np.float64)
                    img = skimage.transform.resize(img, (h, w), order=3)
                    img = img.astype(float) / 255.0
                    # img = blur(img)
                    # plt.imshow(img) ; plt.show()
                    # HWC -> CWH
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).float().to(device)

                    img = (img - self.means[None, ..., None, None]) / self.std[None, ..., None, None]

                    img = _linear_correlate_color(img)

                    img *= 8

                    img_fourier = torch.rfft(img, normalized=True, signal_ndim=2)
                    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
                    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)
                    img_fourier = img_fourier / scale
                    self.image = img_fourier
                    self.image.requires_grad_(True)
                else:
                    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
                    self.image = spectrum_real_imag_t
                print(self.image.shape)

                def inner(spectrum_real_imag_t):
                    nonlocal decay_power
                    # 2 lines below used to be outside
                    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
                    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)
                    # decay_power *= 0.9998

                    scaled_spectrum_t = scale * spectrum_real_imag_t
                    image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
                    image = image[:batch_size, :channels, :h, :w]
                    # image += 1.0 # arbitrary additive from me for making image brighter. Works on init, but colors quickly decay from human to pale
                    magic = 8.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
                    image = image / magic
                    return _linear_decorrelate_color(image)
                    # return image

                self.image_to_bgr = inner
            else:
                intensity_offset = 100
                self.image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset,
                                               (batch_size, self.size_0 * 4, self.size_1 * 4, 3))
                # self.image = np.random.normal(0, 0.1, (batch_size, self.size_0 * 4, self.size_1 * 4, 3)) * 256 + 128
                for i in range(3):
                    self.image[:, :, :, i] -= self.means[i]
                    self.image[:, :, :, i] /= self.std[i]
                self.image /= 255
                self.image = torch.from_numpy(self.image).float()
                self.image = self.image.permute(0, 3, 1, 2)
                self.image = self.image.to(device).requires_grad_(True)
                self.image_to_bgr = lambda x: _linear_decorrelate_color(x)
                # self.image_to_bgr = lambda x: x
        else:
            r = 81 ** 0.5  # increasing this leads to smaller patterns & better loss (more negative)

            coord_range = torch.linspace(-r, r, self.size_0 * 4)
            x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
            y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)
            input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).to(device)

            if True:
                class CompositeActivation(torch.nn.Module):
                    def forward(self, x):
                        # x = torch.atan(x)
                        # return torch.cat([x / 0.67, (x * x) / 0.6], 1)
                        # x = torch.relu(x)
                        # return (x - 0.40) / 0.58
                        x = torch.sin(x)
                        return x / 0.657

                class DivideByConst(torch.nn.Module):
                    def forward(self, x):
                        return x / 2.

                class ResBlock(torch.nn.Module):
                    def __init__(self, in_planes, planes, act_fun):
                        super(ResBlock, self).__init__()
                        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0,
                                               bias=True)
                        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
                        self.shortcut = nn.Sequential()
                        self.act_fun = act_fun
                        if in_planes != planes:
                            conv_to_use = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1,
                                                    bias=True)
                            self.shortcut = nn.Sequential(conv_to_use)

                    def forward(self, x):
                        out = self.conv1(x)
                        out = self.act_fun(out)
                        out = self.conv2(out)
                        out += self.shortcut(x)
                        out = self.act_fun(out)
                        return out

                num_output_channels = 3
                num_hidden_channels = 24
                num_layers = 10
                activation_fn = CompositeActivation
                normalize = False

                layers = []
                kernel_size = 1
                for i in range(num_layers):
                    out_c = num_hidden_channels
                    in_c = out_c  # * 2  # * 2 for composite activation
                    if i == 0:
                        in_c = 2
                    if i == num_layers - 1:
                        out_c = num_output_channels
                    if False:
                        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
                        if normalize:
                            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
                        if i < num_layers - 1:
                            layers.append(('actv{}'.format(i), activation_fn()))
                        else:
                            # layers.append(('div', DivideByConst()))
                            # layers.append(('output', torch.nn.Sigmoid()))
                            layers.append(('output', torch.nn.Tanh()))
                    else:
                        if i < num_layers - 1:
                            layers.append((f'resblock{i}', ResBlock(in_c, out_c, activation_fn())))
                        else:
                            layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
                            layers.append(('output', torch.nn.Tanh()))

                # Initialize model
                net = torch.nn.Sequential(OrderedDict(layers)).to(device)
                print(net)
                # Initialize weights
                # def weights_init(module):
                #     if isinstance(module, torch.nn.Conv2d):
                #         if True:
                #             torch.nn.init.normal_(module.weight, 0, np.sqrt(1 / module.in_channels))
                #             if module.bias is not None:
                #                 torch.nn.init.zeros_(module.bias)
                #         else:
                #             w_std = (1 / module.in_channels) #if self.is_first else (math.sqrt(c / module.in_channels))
                #             torch.nn.init.uniform_(module.weight, -w_std, w_std)
                #             if module.bias is not None:
                #                 torch.nn.init.zeros_(module.bias)
                # net.apply(weights_init)
                # # Set last conv2d layer's weights to 0
                # torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
                self.image = list(net.parameters())
                self.image_to_bgr = lambda _: _linear_decorrelate_color(net(input_tensor))
            else:
                net = SirenNet(
                    dim_in=2,  # input dimension, ex. 2d coor
                    dim_hidden=256,  # hidden dimension
                    dim_out=3,  # output dimension, ex. rgb value
                    num_layers=5,  # number of layers
                    final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                    w0_initial=30.
                    # different signals may require different omega_0 in the first layer - this is a hyperparameter
                ).to(device)
                self.image = list(net.parameters())
                # print(input_tensor.shape)
                input_tensor = input_tensor.reshape(1, 2, -1)
                # print(input_tensor.shape)
                input_tensor = input_tensor.permute(0, 2, 1)
                # print(input_tensor.shape)
                # print(net(input_tensor).shape)
                # print(net(input_tensor).reshape(1, 256, 256, 3).permute(0, 3, 1, 2).shape)

                self.image_to_bgr = lambda _: _linear_decorrelate_color(
                    net(input_tensor).reshape(1, 256, 256, 3).permute(0, 3, 1, 2))

        if os.path.exists('generated'):
            shutil.rmtree('generated', ignore_errors=True)
        os.makedirs('generated')

    def normalize(self, im_as_var):
        return (im_as_var - self.means[None, ..., None, None]) / self.std[None, ..., None, None]

    def recreate_image2_batch(self, img):
        img = self.image_to_bgr(img)
        img = img.cpu().data.numpy()
        recreated_ims = []
        for i in range(img.shape[0]):
            recreated_im = np.copy(img[i])

            # CHW -> HWC
            recreated_im = recreated_im.transpose((1, 2, 0))

            for channel in range(3):
                recreated_im[:, :, channel] *= self.std[channel].item()

            recreated_im *= 255.0
            for i in range(3):
                recreated_im[:, :, i] += self.means[i].item() * 255

            recreated_im[recreated_im > 255.0] = 255.0
            recreated_im[recreated_im < 0.0] = 0.0

            recreated_im = np.round(recreated_im)
            recreated_im = np.uint8(recreated_im)

            # BGR to RGB:
            if not self.if_cifar and not self.if_imagenette and self.use_my_model: #basically, only celeba
                recreated_im = recreated_im[:, :, [2, 1, 0]]

            recreated_ims.append(recreated_im)
        return recreated_ims

    def recreate_and_save(self, img, path):
        # img = torch.sigmoid(img)
        # img = self.normalize(img)
        recreated_img = self.recreate_image2_batch(img)
        if self.batch_size == 1:
            save_image(recreated_img[0], path)
        else:
            save_image_batch(recreated_img, path)

    class ModuleHook:
        def __init__(self, module, name):
            self.hook = module.register_forward_hook(self.hook_fn)
            self.module = None
            self.features = None
            self.name = name

        def hook_fn(self, module, input, output):
            # TODO: DANGER! because output is not copied, if it's modified further in the forward pass, the values here will also be modified
            # (this happens with "+= shortcut")
            self.module = module
            if 'relu2' not in self.name and '_project_conv' not in self.name and 'depthwise_conv' not in self.name:
                self.features = output
            else:
                self.features = input[0]
            # return output.clamp(-10, 10)

        def close(self):
            self.hook.remove()

    @staticmethod
    def hook_model(model_rep):
        features = OrderedDict()

        # recursive hooking function
        def hook_layers(net, prefix):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue
                    print("_".join(prefix + [name]))
                    features["_".join(prefix + [name])] = ClassSpecificImageGenerator.ModuleHook(layer, name)
                    hook_layers(layer, prefix=prefix + [name])

        hook_layers(model_rep, [])

        # def hook(layer):
        #     return features[layer].features

        return features

    def generate(self, n_steps, if_save_intermediate=True):
        coeff_due_to_small_input = 0.35 * 2
        # coeff_due_to_small_input /= coeff_due_to_small_input #comment this line when input is small
        # 0.03  # 0.05 is good for imagenet, but not for celeba
        celeba_lr = 2 * 2 * 0.5 * 0.01 * coeff_due_to_small_input
        initial_learning_rate = celeba_lr
        if self.if_cifar:
            cifar_lr = 2 * 2 * 2 * 0.4 * 2 * 0.5 * 0.01 * coeff_due_to_small_input
            initial_learning_rate = cifar_lr
        if not self.use_my_model:
            imagenet_lr = 2 * (5) ** 1 * 2 * 0.4 * 2 * 0.5 * 0.01 * coeff_due_to_small_input
            initial_learning_rate = imagenet_lr

        # too high lr => checkerboard patterns & acid colors, too low lr => uniform grayness
        saved_iteration_probs = {}
        self.recreate_and_save(self.image, 'generated/init.jpg')
        if True:
            self.image.register_hook(normalize_grad)
        optimizer = Adam([self.image] if not type(self.image) == list else self.image, lr=initial_learning_rate,
                         eps=1e-5)
        # torch.autograd.set_detect_anomaly(True)
        for i in range(n_steps + 1):
            # if (i + 1) % 420 ==
            # 0:
            #     lr_multiplier = 0.8
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= lr_multiplier
            # if target_weight > 1.1:
            #     target_weight *= 0.95
            if self.use_my_model:
                self.feature_extractor.zero_grad()
                if self.target_layer_name == 'label':
                    self.classifier.zero_grad()
                    for icl in self.irrelevant_classifiers.values():
                        icl.zero_grad()
            else:
                self.model.zero_grad()
            optimizer.zero_grad()

            # temp = self.processed_image.clone().cpu().detach().numpy()[0, :, :, :]
            # temp = temp.permute((1, 2, 0))

            pixels_to_jitter1 = 8 * 4
            pixels_to_jitter2 = 4 * 4
            pixels_to_pad = pixels_to_jitter1 + pixels_to_jitter2
            img_width = self.size_0
            img_height = self.size_1
            ROTATE = 40 #15 is good for celeba #25 is good for cifar # 40 is good for imagenet
            scale_defining_const = 7 #7

            # pad is 'reflect' because that's what's in colab
            self.image_transformed = apply_transformations(self.image, [
                self.image_to_bgr,
                # lambda img: torch.sigmoid(img),
                # self.normalize,
                lambda img: nn.functional.pad(img, [pixels_to_pad] * 4, 'constant', 0.5),
                # lambda img: random_crop(img, img_width, img_height),
                lambda img: jitter_lucid(img, pixels_to_jitter1),
                # random_scale([1] * 10 + [1.005, 0.995]),
                random_scale([1 + i / 25. for i in range(-scale_defining_const, scale_defining_const+1)] + 5 * [1]),
                # [SCALE ** (n/10.) for n in range(-10, 11)]),
                # random_rotate([-45]),
                random_rotate(list(range(-ROTATE, ROTATE + 1)) + 5 * [0]),  # range(-ROTATE, ROTATE+1)),#,
                # blur if i % 4 == 0 else identity,
                lambda img: jitter_lucid(img, pixels_to_jitter2),
                lambda img: torch.nn.functional.interpolate(img, (self.size_0, self.size_1), mode='bicubic',
                                                            align_corners=False),
                # lambda img: center_crop(img, img_width, img_height),
                # lambda img: pad_to_correct_shape(img, img_width, img_height)
            ])
            # self.processed_image_transformed.data.clamp_(-1, 1)
            # self.image_transformed = torch.tanh(self.image_transformed)
            # if True:
            #     self.image_transformed.register_hook(normalize_grad)

            features = self.execute_trunk(self.image_transformed)
            # for f in features:
            #     f.clamp_(-3, 3)
            class_loss = self.calculate_loss_f(self.image_transformed, features, i)

            # to_print = f"Iteration {i}:\t Loss {class_loss.item():.4f} \t" \
            #            f"{softmax_prob if self.use_my_model else '':.4f} {bad_softmaxes_sum if self.use_my_model else '':.2f}".expandtabs(20)

            class_loss.backward()
            optimizer.step()

            if i % 60 == 0:  # or i < 5:
                to_print = f"Iteration {i}:\t Loss {class_loss.item():.4f} \t".expandtabs(20)
                print(to_print)

                if if_save_intermediate or i == n_steps:
                    path = 'generated/c_specific_iteration_' + str(i) + '.jpg'
                    self.recreate_and_save(self.image, path)
                # if i % 20 == 0:
                #     saved_iteration_probs[i] = softmax_prob if self.use_my_model else ''

        print(saved_iteration_probs)
        return self.image


if __name__ == '__main__':
    model_to_use = 'resnet18'
    # for epoch in [7, 22, 52, 82, 112]:
    if model_to_use == 'my':
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
        # param_file = 'old_params/sample_all.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
        # param_file = 'params/bigger_reg_4_4_4.json'

        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
        # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0_45_model.pkl'
        # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_nocondecay_comeback.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_49_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004_pretrain_condecayall2e-6_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/07_52_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=1_40_model.pkl'
        # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_condecayall1e-5_comeback.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
        #                   r'|weight_de_58_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_condecayall2e-6.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/02_16_on_May_23/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc_consontop.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_46_on_June_08/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay_60_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_50_on_June_17/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg_consontop_condecayall1e-5.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_30_on_June_18/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg_consontop_condecayall3e-5.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_57_on_June_19/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_41_on_June_21/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall2e-6_comeback_rescaled.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_50_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_' + str(epoch) + '_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_' + str(46) + '_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/04_25_on_June_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_31_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_rescaled3_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/19_15_on_June_28/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall1e-6_nocomeback_rescaled2.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/04_00_on_August_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0003|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_16_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0003_pretrain_bias_condecayall2e-6_comeback_preciserescaled_rescaled2.json'
        # single-head cifar:
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_33_on_September_16/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
        param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_19_on_October_13/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.000_240_model.pkl'
        # param_file = 'params/binmatr2_imagenette_sgd1bias_fc_batch128_weightdecay3e-4_singletask.json'
        # cifar eights-width layer4narrow
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/15_15_on_November_09/optimizer=SGD|batch_size=128|lr=0.1|connectivities_lr=0.0|chunks=[8|_8|_8|_16|_16|_16|_16|_32|_32|_32|_32|_16|_16|_16|_16]|architecture=binmatr2_resnet18|width_mul=0.125|weight_decay=0.0003|connectiv_240_model.pkl'
        # param_file = 'params/binmatr2_cifar_sgd1bias_fc_batch128_weightdecay3e-4_singletask_eightswidth_layer4narrow.json'

        if False:
            save_model_path = save_model_path[:save_model_path.find('.pkl')] + '_avgadditives' + '.pkl'
            removed_conns = defaultdict(list)
            removed_conns['shortcut'] = defaultdict(list)
            if False:
                print('Some connections were disabled')
                # removed_conns[5] = [(5, 23), (5, 25), (5, 58)]
                # removed_conns[7] = [(241, 23)]
                # removed_conns[8] = [(137, 143)]
                # removed_conns[9] = [
                #                     (142, 188),
                #                     (216, 188),
                #                     (187, 86),
                #                     (224, 86)
                #                    ]
                removed_conns[10] = [(188, 104),
                                     #(86, 104)
                                    ]
                removed_conns['shortcut'][8] = [(86, 86)]

            trained_model = load_trained_model(param_file, save_model_path, if_additives_user=True,
                                               if_store_avg_activations_for_disabling=True,
                                               conns_to_remove_dict=removed_conns)
        else:
            try:
                trained_model = load_trained_model(param_file, save_model_path)
            except:
                print('assume problem where cifar networks ignored the enable_bias parameter')
                trained_model = load_trained_model(param_file, save_model_path, if_actively_disable_bias=True)

        with open(param_file) as json_params:
            params = json.load(json_params)
        if 'input_size' not in params:
            params['input_size'] = 'default'
        if 'input_size' in params:
            if params['input_size'] == 'default':
                im_size = (64, 64)
                if 'cifar' in param_file:
                    im_size = (32, 32)
                if 'imagenette' in param_file:
                    im_size = (224, 224)
            elif params['input_size'] == 'bigimg':
                im_size = (128, 128)
            elif params['input_size'] == 'biggerimg':
                im_size = (192, 168)
        else:
            if 'bigimg' in param_file:  # I remember naming my parameter files consistently
                im_size = (128, 128)
            elif 'biggerimg' in param_file:
                im_size = (192, 168)
            else:
                im_size = (64, 64)

        hook_dict = ClassSpecificImageGenerator.hook_model(trained_model['rep'])
        model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
    else:
        # target_class = 130  # Flamingo
        # pretrained_model = models.alexnet(pretrained=True).to(device)
        if model_to_use == 'resnet18':
            trained_model = models.__dict__['resnet18'](pretrained=True).to(device)
        # trained_model = models.__dict__['vgg19_bn'](pretrained=True).to(device)
        if model_to_use == 'mobilenet':
            trained_model = models.__dict__['mobilenet_v2'](pretrained=True).to(device)
        if model_to_use == 'efficientnet':
            trained_model = EfficientNet.from_pretrained('efficientnet-b3').cuda()
        # trained_model = models.__dict__['vgg11_bn'](pretrained=True).to(device)
        im_size = (224, 224)
        hook_dict = ClassSpecificImageGenerator.hook_model(trained_model)

    if False:
        used_neurons = np.load(f'actually_good_nodes_{model_name_short}.npy', allow_pickle=True).item()

    batch_size = 2
    n_steps = 600
    tensor_loss_mul = -1 #multiplied only by the part of the loss dependent on the features, not the whole loss
    l_name = layers_bn_afterrelu[10]#'_blocks_25__project_conv'#vgg_layers_few_AM[0]#'features_6'
    # l_names = ['_blocks_15__project_conv', '_blocks_1', '_blocks_15', '_blocks_25', '_blocks_18', '_blocks_20']
    for j, layer_name in [(None, l_name)]:#list(enumerate(layers)) + [(None, 'label')]:#
    # for j, layer_name in enumerate(l_names):
        if layer_name != 'label':
            if False:
                cur_neurons = used_neurons[j]
                cur_neurons = [int(x[x.find('_') + 1:]) for x in cur_neurons]
            else:
                cur_neurons = [12]#[117]#range(128)#
            diversity_layer_name = layer_name

        else:
            cur_neurons = [8]  # range(40)  # [task_ind_from_task_name('blackhair')] #
            diversity_layer_name = layers[-1]
        for i in cur_neurons:
            print('Current: ', layer_name, i)
            # layer_name = 'layer3_1_conv1'#'layer1_1'#'layer1_1_conv1'#'label'#'layer1_0' #
            csig = ClassSpecificImageGenerator(trained_model, (layer_name, i), im_size, batch_size, model_to_use == 'my',
                   hook_dict, layer_name if (batch_size > 1) else None, [], False,
                   if_cifar=False, if_imagenette=False, tensor_loss_mul=tensor_loss_mul)  # list(range(14)))
            # ,diversity_layer_name)
            generated = csig.generate(n_steps, False)

            if True:
                imshow_path = 'feature_viz'  # f"big_ordinary_generated_imshow_{model_name_short}"
                Path(imshow_path).mkdir(parents=True, exist_ok=True)
                path_prefix = f'{model_to_use}_'
                path_postfix = f'_{"max" if tensor_loss_mul == 1 else "min"}_noreg'
                imshow_path += f"/{path_prefix}{layer_name}{path_postfix}"
                Path(imshow_path).mkdir(parents=True, exist_ok=True)

                # separate_path = f"big_ordinary_generated_separate_{model_name_short}"
                # Path(separate_path).mkdir(parents=True, exist_ok=True)
                # separate_path += f"/{layer_name}"
                # Path(separate_path).mkdir(parents=True, exist_ok=True)
                # if False:
                #     separate_path = 'generated_separate'

                copyfile(f'generated/c_specific_iteration_{n_steps}.jpg', imshow_path + f'/div_{i}.jpg')

                # recreated = csig.recreate_image2_batch(generated)
                # for j in range(batch_size):
                #     save_image(recreated[j], separate_path + f'/{i}_{j}.jpg')