"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
https://github.com/utkuozbulak/pytorch-cnn-visualizations/
"""
import os
import random
import kornia
import PIL
import numpy as np
from functools import partial
from shutil import copyfile

from torch import nn
from torch.optim import Adam, SGD
from torchvision import models
from torch.nn.functional import softmax
import torch
import skimage

from multi_task.load_model import load_trained_model
from PIL import Image
from skimage.filters import gaussian
from scipy.ndimage.interpolation import shift
import shutil
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size_0 = 192
size_1 = 168

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
        return img[:, :, input_width // 2 - width // 2: input_width // 2 + width // 2,:]
    if input_width <= width and input_height > height:
        return img[:, :, :,
               input_height // 2 - height // 2: input_height // 2 + height // 2]


def random_scale(scales):
    def inner(img):
        scale = random.choice(scales)
        old_shape = np.array(img.shape[-2:], dtype='float32')
        scaled_shape = (scale * old_shape).astype(int)
        res = nn.functional.interpolate(img, size=tuple(scaled_shape)#tuple(list(img.shape[:-2]) + list(scaled_shape))
                         , mode='bilinear', align_corners=True)
        return res

    return inner


def random_rotate(angles):
    def inner(img):
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
    return randomCrop(img, width - pixels_to_jitter, height - pixels_to_jitter)


def randomCrop(img, width, height=None):
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
    img = gaussian(img, sigma=0.4, multichannel=True,
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
        res = ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs().sum()) + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().sum()
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
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(nrows=2, ncols=5)
    plt.tight_layout()

    for i in range(len(im_batch)):
        row = 0 if i < 5 else 1
        column = i - row * 5
        ax[row, column].imshow(im_batch[i])
        ax[row, column].get_xaxis().set_visible(False)
        ax[row, column].get_yaxis().set_visible(False)

    plt.savefig(path,bbox_inches='tight',pad_inches = 0)
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

class ClassSpecificImageGenerator():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """

    def __init__(self, model, target_class, use_my_model=True):
        self.target_class = target_class

        self.model = model
        self.use_my_model = use_my_model

        if self.use_my_model:
            self.feature_extractor = self.model['rep']
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
            self.classifier = self.model[str(self.target_class)]
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad_(False)
            self.irrelevant_idx = list(range(self.target_class)) + list(range(self.target_class + 1, 40))
            self.irrelevant_idx = list(filter(lambda x: x!= 10, self.irrelevant_idx))
            print(self.irrelevant_idx)
            self.irrelevant_classifiers = {idx: self.model[str(idx)] for idx in self.irrelevant_idx}
            for c in self.irrelevant_classifiers.values():
                c.eval()
        else:
            model.eval()

        # with open('configs.json') as config_params:
        #     configs = json.load(config_params)
        # test_dst = CELEBA(root=configs['celeba']['path'], is_transform=False, split='val',
        #                   img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
        #                   augmentations=None)
        # self.created_image = test_dst.__getitem__(0)[0]

        # self.created_image = np.ones((64, 64, 3)) * 50
        # self.created_image[5:55, 5:55, :] = np.random.uniform(0 + 70, 255 - 70, (50, 50, 3))

        intensity_offset = 100
        self.created_image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset, (batch_size, size_0, size_1, 3))

        if os.path.exists('generated'):
            shutil.rmtree('generated', ignore_errors=True)
        os.makedirs('generated')

    def transform_img_batch(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.use_my_model:
            means = np.array([73.15835921, 82.90891754, 72.39239876])
        else:
            means = np.array([0.485, 0.456, 0.406]) * 255
        for i in range(3):
            img[:, :, :, i] -= means[i]

        img = torch.from_numpy(img).float()
        with torch.no_grad():
            # HWC -> CWH
            img = img.permute(0, 3, 1, 2)
            img = nn.functional.interpolate(img, size=(size_0, size_1), mode='bilinear', align_corners=True)

            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img /= 255.0

            if not self.use_my_model or True:
                std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
                for channel in range(3):
                    img[:, channel] /= std[channel]

        return img.to(device).requires_grad_(True)

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.use_my_model:
            img -= np.array([73.15835921, 82.90891754, 72.39239876])
        else:
            img -= np.array([0.485, 0.456, 0.406]) * 255

        img = skimage.transform.resize(img, (size_0, size_1), order=3)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # HWC -> CWH
        img = img.transpose(2, 0, 1)

        if not self.use_my_model or True:
            std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
            for channel, _ in enumerate(img):
                img[channel] /= std[channel]

        img = torch.from_numpy(img).float()
        img.unsqueeze_(0)
        return img.to(device).requires_grad_(True)

    def recreate_image2(self, im_as_var):
        recreated_im = np.copy(im_as_var.cpu().data.numpy()
                                   # [:, ::-1, :, :]
                               [0])  # .astype(np.float64)

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

    def recreate_image2_batch(self, im_as_var):
        im_as_var = im_as_var.cpu().data.numpy()
        recreated_ims = []
        for i in range(im_as_var.shape[0]):
            recreated_im = np.copy(im_as_var[i])

            if not self.use_my_model or True:
                std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
                for channel in range(3):
                    recreated_im[channel] *= std[channel]

            recreated_im = recreated_im.transpose((1, 2, 0))

            recreated_im *= 255.0
            if self.use_my_model:
                means = np.array([73.15835921, 82.90891754, 72.39239876])
            else:
                means = np.array([0.485, 0.456, 0.406]) * 255
            for i in range(3):
                recreated_im[:, :, i] += means[i]

            recreated_im[recreated_im > 255.0] = 255.0
            recreated_im[recreated_im < 0.0] = 0.0

            recreated_im = np.round(recreated_im)
            recreated_im = np.uint8(recreated_im)

            recreated_ims.append(recreated_im)
        return recreated_ims

    def generate(self):
        initial_learning_rate = 0.03  # 0.05 is good for imagenet, but not for celeba
        target_weight = 1.0
        # too high lr => checkerboard patterns & acid colors, too low lr => uniform grayness
        self.processed_image = self.transform_img_batch(self.created_image)
        saved_iteration_probs = {}

        self.created_image = self.recreate_image2_batch(self.processed_image)
        im_path = 'generated/init.jpg'
        save_image_batch(self.created_image, im_path)
        self.processed_image_transformed = self.processed_image
        optimizer = Adam([self.processed_image_transformed], lr=initial_learning_rate)
        for i in range(600 + 1):
            # if (i + 1) % 30 == 0:
            #     lr_multiplier = 0.9
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= lr_multiplier
            #     if target_weight > 1.1:
            #         target_weight *= 0.95
            if self.use_my_model:
                self.feature_extractor.zero_grad()
                self.classifier.zero_grad()
                for icl in self.irrelevant_classifiers.values():
                    icl.zero_grad()
            else:
                self.model.zero_grad()
            optimizer.zero_grad()

            #temp = self.processed_image.clone().cpu().detach().numpy()[0, :, :, :]
            # temp = temp.permute((1, 2, 0))

            pixels_to_pad = 12
            pixels_to_jitter1 = 8
            pixels_to_jitter2 = 4
            img_width = self.processed_image.shape[-2]
            img_height = self.processed_image.shape[-1]
            ROTATE = 10
            SCALE = 1.1

            # pad is 'reflect' because that's what's in colab
            self.processed_image_transformed = apply_transformations(self.processed_image, [
                # lambda img: np.pad(img, [(pixels_to_pad, pixels_to_pad)] * 2 + [(0, 0)], 'reflect'),
                lambda img: nn.functional.pad(img, (pixels_to_pad, pixels_to_pad, pixels_to_pad, pixels_to_pad), 'constant', 0.5),
                # constant_values=np.mean(img)),
                # print_shape,
                lambda img: jitter_lucid(img, pixels_to_jitter1),
                # random_scale([1] * 10 + [1.005, 0.995]),
                random_scale([1 + i / 25. for i in range(-5, 6)]),#[SCALE ** (n/10.) for n in range(-10, 11)]),
                # random_rotate([-45]),
                random_rotate(list(range(-ROTATE, ROTATE+1)) + 5 * [0]),#range(-ROTATE, ROTATE+1)),#,
                # blur if i % 4 == 0 else identity,
                # lambda img: jitter_lucid(img, pixels_to_jitter2),
                lambda img: center_crop(img, img_width, img_height),
                lambda img: pad_to_correct_shape(img, img_width, img_height)
            ])
            self.processed_image_transformed.data.clamp_(-1, 1)

            if self.use_my_model:
                if False:
                    features = self.feature_extractor(self.processed_image_transformed, None)[0]
                    output = features[self.target_class]
                else:
                    features = self.feature_extractor(self.processed_image_transformed)
                    output = features[self.target_class]
                output = self.classifier(output, None)[0]
                softmax_prob = torch.nn.functional.softmax(output, dim=1)[:, 1].mean().item()
                output = output[:, 1].mean()
                bad_softmaxes_sum = 0.
                bad_output = 0.
                for idx, icl in self.irrelevant_classifiers.items():
                    bad_output_cur = self.classifier(features[idx], None)[0]
                    bad_softmax_prob_cur = torch.nn.functional.softmax(bad_output_cur, dim=1)[:, 1].mean().item()
                    bad_softmaxes_sum += bad_softmax_prob_cur
                    bad_output += bad_output_cur[:, 1].mean()
                bad_softmaxes_sum /= 38.

            else:
                output = self.model(self.processed_image_transformed)[0, self.target_class]

            # 0.00005 was a good value for total variation
            class_loss = -output * target_weight \
                         + 0.001 * total_variation_loss(self.processed_image_transformed, 2) \
                         + 0.7 * bad_output / 38.

                # + 0.000001 * torch.norm(self.processed_image + 1, 1) #+1 only because I clamp_
            # + 0.1 * torch.norm(self.processed_image + (torch.tensor([73.15835921, 82.90891754, 72.39239876]) / 255.0)[None, :, None, None].cuda(), 1)
            # + 0.0001 * torch.norm(self.processed_image, 1) \
            # + 0.00001 * torch.norm(self.processed_image, 2) \
            # + 0.000001 * torch.norm(self.processed_image, 6) \
            print('Iteration:', str(i), 'Loss', "{0:.4f}".format(class_loss.item()),
                  softmax_prob if self.use_my_model else '', bad_softmaxes_sum if self.use_my_model else '')

            class_loss.backward()
            optimizer.step()

            if i % 60 == 0:# or i < 5:
                self.created_image = self.recreate_image2_batch(self.processed_image)
                im_path = 'generated/c_specific_iteration_' + str(i) + '.jpg'
                save_image_batch(self.created_image, im_path)

                if i % 20 == 0:
                    saved_iteration_probs[i] = softmax_prob if self.use_my_model else ''

        print(saved_iteration_probs)
        return self.processed_image


class ActivityCollector():
    def __init__(self, model):
        self.model = model
        self.use_my_model = True

        for m in self.model:
            model[m].eval()

        self.feature_extractor = self.model['rep']
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)

    def img_from_np_to_torch(self, img):
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.use_my_model:
            img -= np.array([73.15835921, 82.90891754, 72.39239876])
        else:
            img -= np.array([0.485, 0.456, 0.406]) * 255

        img = skimage.transform.resize(img, (size_0, size_1), order=3)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # HWC -> CWH
        img = img.transpose(2, 0, 1)

        if not self.use_my_model or True:
            std = [0.229, 0.224, 0.225]
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

    def get_activations(self, input_img_path):
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

    def compare_AM_images(self, class1: int, class2: int):
        activations1 = self.get_activations(f'generated_best/{class1}.jpg')
        activations2 = self.get_activations(f'generated_best/{class2}.jpg')

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

    def get_output_probs(self, input_img_path):
        img = np.asarray(PIL.Image.open(input_img_path))
        img = self.img_from_np_to_torch(img)

        out = self.feature_extractor(img)

        all_tasks = [str(task) for task in range(40)]
        probs = np.ones((40)) * -17
        for i, t in enumerate(all_tasks):
            out_t_val, _ = self.model[t](out[i], None)
            probs[i] = torch.nn.functional.softmax(out_t_val)[0][1].item()

        return probs

    def get_feature_distribution_per_class(self, target: int, n_images: int, folder_suffix):
        strengths_sum = None
        for i in range(n_images):
            activations = self.get_activations(f'generated_10_{folder_suffix}/{target}_{i}.jpg')

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
            probs = self.get_output_probs(f'generated_10_{folder_suffix}/{target}_{i}.jpg')
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
        res = np.ones((n_tasks, n_tasks))
        targets = [int(target) for target in targets]
        for target in targets:
            print(target)
            res[target] = self.get_target_probs_per_class(target, n_images, folder_suffix)
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
        ax.set_xticks(np.arange(.5, len(targets), 1))
        ax.set_xticklabels([celeba_dict[target] for target in targets], rotation=90)
        cb = plt.colorbar(fraction=0.03, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        plt.savefig(f'probs_{folder_suffix}.svg', format='svg', bbox_inches='tight', pad_inches=0, dpi=200)
        # plt.show()

if __name__ == '__main__':
    use_my_model = True

    if use_my_model:
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
        # param_file = 'old_params/sample_all.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
        # param_file = 'params/bigger_reg_4_4_4.json'

        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
        # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
        param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'

        trained_model = load_trained_model(param_file, save_model_path)
    else:
        target_class = 130  # Flamingo
        pretrained_model = models.alexnet(pretrained=True).to(device)

    if False:
        if use_my_model:
            batch_size = 1
            for i in [15]:
                csig = ClassSpecificImageGenerator(trained_model, i)
                generated = csig.generate()
                copyfile('generated/c_specific_iteration_600.jpg', f'generated_imshow_binmatr2/{i}.jpg')
                for j in range(batch_size):
                    recreated = csig.recreate_image2(generated[j].unsqueeze(0))
                    save_image(recreated, f'generated_10_binmatr2/{i}_{j}.jpg')

            # {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs',
            #  6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair',
            #  12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair',
            #  18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache',
            #  23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose',
            #  28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair',
            #  33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace',
            #  38: 'Wearing_Necktie', 39: 'Young'}
        else:
            csig = ClassSpecificImageGenerator(pretrained_model, target_class, False)
            csig.generate()
    else:
        ac = ActivityCollector(trained_model)
        # ac.compare_AM_images(31, 12)
        # ac.visualize_feature_distribution(range(40), 10, 'binmatr')
        ac.visualize_feature_histograms_per_task(range(40), 10, 'binmatr')
        # ac.visualize_probs_distribution(range(40), 10, 'binmatr')
