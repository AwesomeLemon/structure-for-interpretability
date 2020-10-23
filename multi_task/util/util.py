import math
import os
import random
import kornia
import PIL
import numpy as np
from functools import partial
from shutil import copyfile
import json

from torch import nn
from torch.nn.functional import softmax
import torch

from PIL import Image
from skimage.filters import gaussian
from scipy.ndimage.interpolation import shift
import shutil
from matplotlib import pyplot as plt
from fuzzywuzzy import process
import pandas as pd

celeba_dict = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald',
               5: 'Bangs', 6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry',
               11: 'Brown_Hair', 12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses',
               16: 'Goatee', 17: 'Gray_Hair', 18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male',
               21: 'Mouth_Slightly_Open', 22: 'Mustache', 23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face',
               26: 'Pale_Skin', 27: 'Pointy_Nose', 28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns',
               31: 'Smiling', 32: 'Straight_Hair', 33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat',
               36: 'Wearing_Lipstick', 37: 'Wearing_Necklace', 38: 'Wearing_Necktie', 39: 'Young'}


def task_ind_from_task_name(task_name):
    task_name = process.extractOne(task_name, celeba_dict.values())[0]
    print(task_name)
    for idx, task in celeba_dict.items():
        if task == task_name:
            return idx


layers = ['layer1_0',
          'layer1_1_conv1', 'layer1_1',
          'layer2_0_conv1', 'layer2_0',
          'layer2_1_conv1', 'layer2_1',
          'layer3_0_conv1', 'layer3_0',
          'layer3_1_conv1', 'layer3_1',
          'layer4_0_conv1', 'layer4_0',
          'layer4_1_conv1', 'layer4_1',
          ]

layers_bn = ['layer1_0',
             'layer1_1_bn1', 'layer1_1',
             'layer2_0_bn1', 'layer2_0',
             'layer2_1_bn1', 'layer2_1',
             'layer3_0_bn1', 'layer3_0',
             'layer3_1_bn1', 'layer3_1',
             'layer4_0_bn1', 'layer4_0',
             'layer4_1_bn1', 'layer4_1',
             ]

layers_bn_prerelu = ['layer1_0_relu2',
             'layer1_1_bn1', 'layer1_1_relu2',
             'layer2_0_bn1', 'layer2_0_relu2',
             'layer2_1_bn1', 'layer2_1_relu2',
             'layer3_0_bn1', 'layer3_0_relu2',
             'layer3_1_bn1', 'layer3_1_relu2',
             'layer4_0_bn1', 'layer4_0_relu2',
             'layer4_1_bn1', 'layer4_1_relu2',
             ]

layers_bn_afterrelu = ['layer1_0',
             'layer1_1_relu1', 'layer1_1',
             'layer2_0_relu1', 'layer2_0',
             'layer2_1_relu1', 'layer2_1',
             'layer3_0_relu1', 'layer3_0',
             'layer3_1_relu1', 'layer3_1',
             'layer4_0_relu1', 'layer4_0',
             'layer4_1_relu1', 'layer4_1',
             ]

cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
fashionmnist_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
imagenettte_names = ['tench', 'English springer', 'cassette player',
                     'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
cifarfashion_dict = dict(zip(range(20), cifar10_names + fashionmnist_names))
cifar10_dict = dict(zip(range(10), cifar10_names))
imagenette_dict = dict(zip(range(10), imagenettte_names))

def pad_to_correct_shape(img, width, height):
    # return skimage.transform.resize(img, (width, width), order=3)
    input_width = img.shape[-2]
    input_height = img.shape[-1]

    if input_width == width:
        pixels_to_pad1 = 0
        pixels_to_pad2 = 0
    else:
        pixels_to_pad1 = (width - input_width) // 2
        pixels_to_pad2 = (width - input_width) - pixels_to_pad1

    if input_height == height:
        pixels_to_pad3 = 0
        pixels_to_pad4 = 0
    else:
        pixels_to_pad3 = (height - input_height) // 2
        pixels_to_pad4 = (height - input_height) - pixels_to_pad3
    res = nn.functional.pad(img, (pixels_to_pad3, pixels_to_pad4, pixels_to_pad1, pixels_to_pad2), 'constant', 0)
    return res


def print_shape(img):
    print(img.shape)
    return img


def center_crop(img, width, height):
    input_width = img.shape[-2]
    input_height = img.shape[-1]
    if input_width <= width and input_height <= height:
        return img
    if input_width > width and input_height > height:
        return img[:, :, input_width // 2 - width // 2: input_width // 2 + width // 2,
               input_height // 2 - height // 2: input_height // 2 + height // 2]
    if input_width > width and input_height <= height:
        return img[:, :, input_width // 2 - width // 2: input_width // 2 + width // 2, :]
    if input_width <= width and input_height > height:
        return img[:, :, :,
               input_height // 2 - height // 2: input_height // 2 + height // 2]


def random_scale(scales):
    def inner(img):
        scale = random.choice(scales)
        old_shape = np.array(img.shape[-2:], dtype='float32')
        scaled_shape = (scale * old_shape).astype(int)
        res = nn.functional.interpolate(img, size=tuple(scaled_shape)
                                        # tuple(list(img.shape[:-2]) + list(scaled_shape))
                                        , mode='bilinear', align_corners=True)
        return res

    return inner


def random_rotate(angles):
    def inner(img):
        batch_size = img.size(0)
        angle = torch.tensor(np.random.choice(angles, batch_size))
        # angle = torch.ones(1) * angle
        center = torch.ones(batch_size, 2)
        center[..., 0] = img.shape[3] / 2  # x
        center[..., 1] = img.shape[2] / 2  # y
        scale = torch.ones(batch_size)
        # compute the transformation matrix
        M = kornia.get_rotation_matrix2d(center, angle, scale)

        # apply the transformation to original image
        _, _, h, w = img.shape
        img_warped = kornia.warp_affine(img, M, dsize=(h, w))
        return img_warped

    return inner


def jitter(img):
    # img is 3 x 64 x 64
    jitter = 1  # (in pixels)
    jitter1 = np.random.randint(-jitter, jitter + 1) * np.random.randint(0, 1 + 1) * np.random.randint(0,
                                                                                                       1 + 1)  # second term is for making 0 likelier
    jitter2 = np.random.randint(-jitter, jitter + 1) * np.random.randint(0, 1 + 1) * np.random.randint(0, 1 + 1)
    return shift(img, [0, jitter1, jitter2], mode='wrap', order=0)


def jitter_lucid(img, pixels_to_jitter):
    # assert img.shape[0] == img.shape[1]
    width = img.shape[-2]
    height = img.shape[-1]
    return random_crop(img, width - pixels_to_jitter, height - pixels_to_jitter)


def random_crop(img, width, height=None):
    if height is None:
        height = width
    # assert img.shape[1] >= width
    x = random.randint(0, img.shape[-2] - width)
    y = random.randint(0, img.shape[-1] - height)
    img = img[:, :, x:x + width, y:y + height]
    return img


def identity(img):
    return img


def blur(img):
    img = gaussian(img, sigma=2.0  # 0.4
                   , multichannel=True,
                   mode='reflect', preserve_range=True)
    return img


def apply_transformations(img, transforms):
    for t in transforms:
        img = t(img)

    return img


def total_variation_loss(img, beta):
    if False:
        # this version is based on that visualization paper
        res = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)) + (
            (img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2).permute(0, 1, 3, 2))
        assert res.min() >= 0.0
        res = res.pow(beta / 2).sum()  # / (64.0 * 64.0)
    else:
        # this version is based on the tensorflow implementation that lucid uses: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/image_ops_impl.py#L2347-L2415
        res = ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs().sum(dim=-1).sum(dim=-1).sum(dim=-1)) \
              + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().sum(dim=-1).sum(dim=-1).sum(dim=-1)
        res = res.sum()
    return res


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image_batch(im_batch, path):
    fig = plt.figure(figsize=(10, 4.5))  # (20, 5)
    im_total = len(im_batch)
    rows_num = 2
    cols_num = int(math.ceil(im_total / rows_num))
    ax = fig.subplots(nrows=rows_num, ncols=cols_num, squeeze=False)
    plt.tight_layout()

    for i in range(len(im_batch)):
        row = i // cols_num
        column = i - row * cols_num
        ax[row, column].imshow(im_batch[i])
        ax[row, column].get_xaxis().set_visible(False)
        ax[row, column].get_yaxis().set_visible(False)

    fig.subplots_adjust(hspace=0.05)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def normalize_grad(grad):
    n = torch.norm(grad, 2, dim=(-1, -2, -3), keepdim=True)
    eps = 1e-8
    res = grad / (n + eps)
    # std = [0.229, 0.224, 0.225] * 4
    # for channel in range(3):
    #     res[:, channel, :, :] *= std[channel]
    # print(torch.norm(res, 2, dim=(-1, -2, -3)))
    return res


def punish_outside_center(tensor):
    # N, C, H, W
    a = b = tensor.size(2) // 2
    n = tensor.size(2)
    r = 15

    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= r * r
    mask = torch.tensor(np.repeat(mask[None, :, :], tensor.size(1), axis=0)[None, ...])
    tensor2 = tensor.clone()
    tensor2[mask] = 0
    return -tensor2.mean()


def images_list_to_grid_image(ims):
    n_ims = len(ims)
    width, height = ims[0].size
    rows_num = math.floor(math.sqrt(n_ims))
    cols_num = int(math.ceil(n_ims / rows_num))
    new_im = Image.new('RGB', (cols_num * width, rows_num * height))
    for j in range(n_ims):
        row = j // cols_num
        column = j - row * cols_num
        new_im.paste(ims[j], (column * width, row * height))
    return new_im


def proper_hist(data, title='', ax=None, xlim_left=None, xlim_right=None, bin_size=0.1, alpha=1, density=None):
    bins = math.ceil((np.max(data) - np.min(data)) / bin_size)
    if np.max(data) - np.min(data) == 0:
        bins = None
    if ax is None:
        plt.hist(data, bins, alpha=alpha, density=density)
        plt.xlim(left=xlim_left, right=xlim_right)
        plt.title(title)
    else:
        ax.hist(data, bins, alpha=alpha, density=density)
        ax.set_xlim(left=xlim_left, right=xlim_right)
        ax.set_title(title)


def recreate_image_nonceleba_batch(img, dataset_type='cifar'):
    if dataset_type == 'cifar':
        means = np.array([0.4914, 0.4822, 0.4465]) * 255
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_type == 'imagenette':
        means = np.array([109.5388, 118.6897, 124.6901])
        std = [0.224, 0.224, 0.224]
    elif dataset_type == 'imagenet_val':
        means = np.array([0.485, 0.456, 0.406]) * 255
        std = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError()
    img = img.cpu().data.numpy()
    recreated_ims = []
    for i in range(img.shape[0]):
        recreated_im = np.copy(img[i])

        # CHW -> HWC
        recreated_im = recreated_im.transpose((1, 2, 0))

        for channel in range(3):
            recreated_im[:, :, channel] *= std[channel]

        recreated_im *= 255.0
        for i in range(3):
            recreated_im[:, :, i] += means[i]

        recreated_im[recreated_im > 255.0] = 255.0
        recreated_im[recreated_im < 0.0] = 0.0

        recreated_im = np.round(recreated_im)
        recreated_im = np.uint8(recreated_im)

        if dataset_type != 'imagenette':
            # BGR to RGB:
            recreated_im = recreated_im[:, :, [2, 1, 0]]

        recreated_ims.append(recreated_im)
    return recreated_ims


def get_relevant_labels_from_batch(batch, all_tasks, tasks, params, device):
    labels = {}
    # Read all targets of all tasks
    for i, t in enumerate(all_tasks):
        if t not in tasks:
            continue
        if params['dataset'] == 'cifar10':
            labels[t] = (batch[1] == int(t)).type(torch.LongTensor)
        elif params['dataset'] == 'cifarfashionmnist':
            labels[t] = (batch[1] == int(t)).type(torch.LongTensor)
        elif params['dataset'] == 'cifar10_singletask':
            labels[t] = batch[1]
        else:
            labels[t] = batch[i + 1]
        labels[t] = labels[t].to(device)
    return labels


def store_path_to_label_dict(loader, store_path):
    '''
    Note that despite general name, this'll work only for my celeba loader
    '''
    path_to_label_dict = {}
    for batch in loader:
        paths = batch[-1]
        labels = torch.stack(batch[1:-1]).cpu().detach().numpy()
        for i, path in enumerate(paths):
            path_to_label_dict[path] = labels[:, i]

    np.save(store_path, path_to_label_dict, allow_pickle=True)


def store_path_and_label_df_broden(loader, store_path):
    max_class = 1197
    total_samples = 26707#11246
    lbls = np.zeros((max_class + 1, total_samples))
    all_paths_list = []
    all_labels_list = []
    for i, batch in enumerate(loader):
        # if i == 1:
        #     raise NotImplementedError('Assume single batch for simplicity')

        cur_paths = np.array(batch[-1])
        cur_labels = batch[1].cpu().detach().numpy().reshape((batch[1].shape[0], -1))
        cur_labels = np.unique(cur_labels, axis=1)
        print(cur_labels.shape)
        cur_labels = np.pad(cur_labels, ((0, 0), (0, 30000 - cur_labels.shape[1])), mode='constant', constant_values=0)
        # cur_labels = cur_labels[None, :]

        all_paths_list.append(cur_paths)
        all_labels_list.append(cur_labels)
            # all_paths = np.append(all_paths, cur_paths)
            # all_labels = np.append(all_labels, cur_labels, axis=0)
        print(cur_paths.shape)
        print(cur_labels.shape)

    all_paths = np.concatenate(all_paths_list)
    all_labels = np.concatenate(all_labels_list, axis=0)
    print(all_paths.shape)
    print(all_labels.shape)

    for i, path in enumerate(all_paths):
        cur_segm = all_labels[i]
        cur_labels = np.unique(cur_segm)
        cur_labels = cur_labels[cur_labels != 0]
        lbls[cur_labels, i] = 1

    sort_idx = np.argsort(all_paths)
    paths_sorted = np.array(all_paths)[sort_idx]
    lbls_sorted = lbls[:, sort_idx]
    data = []
    for i in range(max_class):
        data.append(['label', i] + list(lbls_sorted[i]))

    df = pd.DataFrame(data, columns=['layer_name', 'neuron_idx'] + list(paths_sorted))
    df.to_pickle(store_path)