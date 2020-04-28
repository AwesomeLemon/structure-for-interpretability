"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
https://github.com/utkuozbulak/pytorch-cnn-visualizations/
"""
import os
import random

import numpy as np

from torch.optim import Adam
from torchvision import models
from torch.nn.functional import softmax
import torch
import skimage

from multi_task.load_model import load_trained_model
from PIL import Image
from skimage.filters import gaussian
from scipy.ndimage.interpolation import shift
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size_0 = 192
size_1 = 168

def pad_to_correct_shape(img, width, height):
    # return skimage.transform.resize(img, (width, width), order=3)
    input_width = img.shape[0]
    input_height = img.shape[1]

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
    return np.pad(img, [(pixels_to_pad1, pixels_to_pad2)] + [(pixels_to_pad3, pixels_to_pad4)] + [(0, 0)], 'constant',
                  constant_values=0)  # np.mean(img)+0.5)


def print_shape(img):
    print(img.shape)
    return img


def center_crop(img, width, height):
    input_width = img.shape[0]
    input_height = img.shape[1]
    if input_width <= width and input_height <= height:
        return img
    if input_width > width and input_height > height:
        return img[input_width // 2 - width // 2: input_width // 2 + width // 2,
               input_height // 2 - height // 2: input_height // 2 + height // 2,
               :]
    if input_width > width and input_height <= height:
        return img[input_width // 2 - width // 2: input_width // 2 + width // 2,
               :,
               :]
    if input_width <= width and input_height > height:
        return img[:,
               input_height // 2 - height // 2: input_height // 2 + height // 2,
               :]


def random_scale(scales):
    def inner(img):
        scale = random.choice(scales)
        old_shape = np.array(img.shape, dtype='float32')
        scaled_shape = (scale * old_shape[:-1]).astype(int)
        return skimage.transform.resize(img, scaled_shape, order=1)

    return inner


def random_rotate(angles):
    def inner(img):
        angle = random.choice(angles)
        return skimage.transform.rotate(img, angle, order=0, resize=False, mode='constant', cval=0)  # np.mean(img)+0.5)

    return inner


def jitter_lucid(img, pixels_to_jitter):
    # assert img.shape[0] == img.shape[1]
    width = img.shape[0]
    height = img.shape[1]
    return randomCrop(img, width - pixels_to_jitter, height - pixels_to_jitter)


def randomCrop(img, width, height=None):
    if height is None:
        height = width
    # assert img.shape[1] >= width
    x = random.randint(0, img.shape[0] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[x:x + width, y:y + height]
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
            self.irrelevant_idx = [8, 11]  # list(range(self.target_class)) + list(range(self.target_class + 1, 40))
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

        intensity_offset = 0
        self.created_image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset, (size_0, size_1, 3))

        if os.path.exists('generated'):
            shutil.rmtree('generated', ignore_errors=True)
        os.makedirs('generated')

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

        if self.use_my_model:
            img = skimage.transform.resize(img, (size_0, size_1), order=3)
        else:
            img = skimage.transform.resize(img, (224, 224), order=3)

        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        if not self.use_my_model or True:
            std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
            for channel, _ in enumerate(img):
                img[channel] /= std[channel]

        img = torch.from_numpy(img).float()
        img.unsqueeze_(0)
        return img.to(device).requires_grad_(True)

    def jitter(self, img):
        # img is 3 x 64 x 64
        jitter = 1  # (in pixels)
        jitter1 = np.random.randint(-jitter, jitter + 1) * np.random.randint(0, 1 + 1) * np.random.randint(0,
                                                                                                           1 + 1)  # second term is for making 0 likelier
        jitter2 = np.random.randint(-jitter, jitter + 1) * np.random.randint(0, 1 + 1) * np.random.randint(0, 1 + 1)
        return shift(img, [0, jitter1, jitter2], mode='wrap', order=0)

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

    # def normalize_img(self, img_np):

    def generate(self):
        initial_learning_rate = 0.2  # 0.05 is good for imagenet, but not for celeba
        target_weight = 2.0
        # too high lr => checkerboard patterns & acid colors, too low lr => uniform grayness
        self.processed_image = self.transform_img(self.created_image)

        saved_iteration_probs = {}

        self.created_image = self.recreate_image2(self.processed_image)
        im_path = 'generated/init.jpg'
        save_image(self.created_image, im_path)

        optimizer = Adam([self.processed_image], lr=initial_learning_rate)
        for i in range(600 + 1):
            if (i + 1) % 30 == 0:
                lr_multiplier = 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_multiplier
                if target_weight > 1.1:
                    target_weight *= 0.95
            if self.use_my_model:
                self.feature_extractor.zero_grad()
                self.classifier.zero_grad()
            else:
                self.model.zero_grad()
            optimizer.zero_grad()

            if self.use_my_model:
                if False:
                    features = self.feature_extractor(self.processed_image, None)[0]
                    output = features[self.target_class]
                else:
                    features = self.feature_extractor(self.processed_image)
                    output = features[self.target_class]
                output = self.classifier(output, None)[0][0]
                softmax_prob = torch.nn.functional.softmax(output, dim=0)[1].item()
                output = output[1]
            else:
                output = self.model(self.processed_image)[0, self.target_class]

            class_loss = -output * target_weight \
                         + 0.0001 * total_variation_loss(self.processed_image, 2) \
                # + 0.000001 * torch.norm(self.processed_image + 1, 1) #+1 only because I clamp_
            # + 0.1 * torch.norm(self.processed_image + (torch.tensor([73.15835921, 82.90891754, 72.39239876]) / 255.0)[None, :, None, None].cuda(), 1)
            # + 0.0001 * torch.norm(self.processed_image, 1) \
            # + 0.00001 * torch.norm(self.processed_image, 2) \
            # + 0.000001 * torch.norm(self.processed_image, 6) \
            print('Iteration:', str(i), 'Loss', "{0:.4f}".format(class_loss.item()),
                  softmax_prob if self.use_my_model else '')

            class_loss.backward()
            optimizer.step()

            if i % 20 == 0 or i < 10:
                self.created_image = self.recreate_image2(self.processed_image)
                im_path = 'generated/c_specific_iteration_' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

                if i % 20 == 0:
                    saved_iteration_probs[i] = softmax_prob if self.use_my_model else ''

            temp = self.processed_image.clone().cpu().detach().numpy()[0, :, :, :]
            temp = temp.transpose((1, 2, 0))

            pixels_to_pad = 2
            pixels_to_jitter1 = 2
            pixels_to_jitter2 = 1
            img_width = temp.shape[0]
            img_height = temp.shape[1]
            ROTATE = 5
            SCALE = 1.1

            # pad is 'reflect' because that's what's in colab
            temp = apply_transformations(temp, [
                # lambda img: np.pad(img, [(pixels_to_pad, pixels_to_pad)] * 2 + [(0, 0)], 'reflect'),
                # constant_values=np.mean(img)),
                # print_shape,
                # lambda img: jitter_lucid(img, pixels_to_jitter1),
                random_scale([1] * 10 + [1.005, 0.995]),
                # random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),#[1 + (i - 5) / 50. for i in range(11)]),
                random_rotate(list(range(-1, 2)) + 10 * [0]),
                # random_rotate(range(-ROTATE, ROTATE+1)),#list(range(-10, 11)) + 5 * [0]),
                # blur if i % 4 == 0 else identity,
                # lambda img: jitter_lucid(img, pixels_to_jitter2),
                lambda img: center_crop(img, img_width, img_height),
                lambda img: pad_to_correct_shape(img, img_width, img_height)
            ])
            # if i < 280 else [])

            temp = temp.transpose((2, 0, 1))
            temp = torch.from_numpy(temp).float()[None, :, :, :].to(device)
            self.processed_image.data = temp.data
            self.processed_image.data.clamp_(-1, 1)

        print(saved_iteration_probs)
        return self.processed_image


if __name__ == '__main__':
    use_my_model = True

    if not use_my_model:
        target_class = 130  # Flamingo
        pretrained_model = models.alexnet(pretrained=True).to(device)

    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'

    # todo: this was the last one I used:
    if use_my_model:
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
        # param_file = 'params/bigger_reg_4_4_4.json'

        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
        # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'

        save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
        param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'

    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
    # param_file = 'old_params/sample_all.json'

    if use_my_model:
        trained_model = load_trained_model(param_file, save_model_path)
        csig = ClassSpecificImageGenerator(trained_model,
                                           31)
        # {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', 2: 'Attractive', 3: 'Bags_Under_Eyes', 4: 'Bald', 5: 'Bangs',
        #  6: 'Big_Lips', 7: 'Big_Nose', 8: 'Black_Hair', 9: 'Blond_Hair', 10: 'Blurry', 11: 'Brown_Hair',
        #  12: 'Bushy_Eyebrows', 13: 'Chubby', 14: 'Double_Chin', 15: 'Eyeglasses', 16: 'Goatee', 17: 'Gray_Hair',
        #  18: 'Heavy_Makeup', 19: 'High_Cheekbones', 20: 'Male', 21: 'Mouth_Slightly_Open', 22: 'Mustache',
        #  23: 'Narrow_Eyes', 24: 'No_Beard', 25: 'Oval_Face', 26: 'Pale_Skin', 27: 'Pointy_Nose',
        #  28: 'Receding_Hairline', 29: 'Rosy_Cheeks', 30: 'Sideburns', 31: 'Smiling', 32: 'Straight_Hair',
        #  33: 'Wavy_Hair', 34: 'Wearing_Earrings', 35: 'Wearing_Hat', 36: 'Wearing_Lipstick', 37: 'Wearing_Necklace',
        #  38: 'Wearing_Necktie', 39: 'Young'}
    else:
        csig = ClassSpecificImageGenerator(pretrained_model, target_class,
                                           False)  # oval face == 25, mustache=22, goatee=16, heavy makeup=18,6: 'Big_Lips',9: 'Blond_Hair', 31: 'Smiling', 32: 'Straight_Hair'
    csig.generate()
