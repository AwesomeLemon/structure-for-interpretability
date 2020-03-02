"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
https://github.com/utkuozbulak/pytorch-cnn-visualizations/
"""
import json
import os
import numpy as np

from torch.optim import SGD
from torchvision import models
from torch.nn.functional import softmax
import torch
import skimage

from multi_task import datasets
from multi_task.load_model import load_trained_model
import matplotlib.pyplot as plt
from PIL import Image
from multi_task.loaders.celeba_loader import CELEBA
from PIL import ImageFilter
from skimage.filters import gaussian
from scipy.ndimage.interpolation import shift


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.target_class = target_class

        self.model = model
        self.feature_extractor = self.model['rep']
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
        self.classifier = self.model[str(self.target_class)]
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad_(False)
        self.irrelevant_idx = [8, 11]#list(range(self.target_class)) + list(range(self.target_class + 1, 40))
        self.irrelevant_classifiers = {idx : self.model[str(idx)] for idx in self.irrelevant_idx}
        for c in self.irrelevant_classifiers.values():
            c.eval()
        # Generate a random image

        with open('configs.json') as config_params:
            configs = json.load(config_params)

        # test_dst = CELEBA(root=configs['celeba']['path'], is_transform=False, split='val',
        #                   img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
        #                   augmentations=None)
        # self.created_image = test_dst.__getitem__(0)[0]

        # self.created_image = np.ones((64, 64, 3)) * 50
        # self.created_image[5:55, 5:55, :] = np.random.uniform(0 + 70, 255 - 70, (50, 50, 3))

        intensity_offset = 0
        self.created_image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset, (64, 64, 3))

        # self.created_image = np.ones((64, 64, 3)) * 127
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= np.array([73.15835921, 82.90891754, 72.39239876])
        img = skimage.transform.resize(img, (64, 64), order=3)
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        # img = self.jitter(img)
        img = torch.from_numpy(img).float()
        img.unsqueeze_(0)
        return img.cuda().requires_grad_(True)

    def jitter(self, img):
        #img is 3 x 64 x 64
        jitter = 2 # (in pixels)
        jitter1 = np.random.randint(-jitter, jitter + 1) #* np.random.randint(0, 1 + 1) * np.random.randint(0, 1 + 1)#second term is for making 0 likelier
        jitter2 = np.random.randint(-jitter, jitter + 1) #* np.random.randint(0, 1 + 1) * np.random.randint(0, 1 + 1)
        return shift(img, [0, jitter1, jitter2], mode='wrap', order=0)

    def recreate_image2(self, im_as_var, if_blur):
        recreated_im = np.copy(im_as_var.cpu().data.numpy()
                               # [:, ::-1, :, :]
                               [0]).transpose((1, 2, 0))#.astype(np.float64)
        # recreated_im = im_as_var.astype(np.float64)
        recreated_im *= 255.0
        recreated_im = recreated_im + np.array([73.15835921, 82.90891754, 72.39239876])
        # recreated_im = recreated_im.astype(float)

        recreated_im[recreated_im > 255.0] = 255.0
        recreated_im[recreated_im < 0.0] = 0.0

        if if_blur:
            print('blur!')
            recreated_im = gaussian(recreated_im, sigma=0.4, multichannel=True,
                                    mode='reflect', preserve_range=True)

        recreated_im = np.round(recreated_im)
        recreated_im = np.uint8(recreated_im)
        # recreated_im[recreated_im > 1] = 1
        # recreated_im[recreated_im < 0] = 0

        return recreated_im#/255.0

    def generate(self):
        initial_learning_rate = 6#0.01
        for i in range(1001):
            # Process image and return variable
            self.processed_image = self.transform_img(self.created_image)
            if i == 0:
                self.created_image = self.recreate_image2(self.processed_image, if_blur=False)
                im_path = '../generated/init.jpg'
                save_image(self.created_image, im_path)
            # if i == 500:
            #     initial_learning_rate /= 10.0
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)

            self.feature_extractor.zero_grad()
            self.classifier.zero_grad()
            # Forward
            features = self.feature_extractor(self.processed_image, None)[0]
            output = features[self.target_class]
            output = self.classifier(output, None)[0][0]

            # irrelevant_probs_sum = 0.0
            # for idx in self.irrelevant_idx:
            #     cur_features = features[idx]
            #     cur_classifier = self.irrelevant_classifiers[idx]
            #     cur = cur_classifier(cur_features, None)[0][0]
            #     cur_prob = torch.nn.functional.softmax(cur, dim=0)[1]
            #     irrelevant_probs_sum += cur_prob
            # Target specific class
            class_loss = -output[1] * 1 \
                         + 0.001 * total_variation_loss(self.processed_image, 2) \
                         # + 0.01 * torch.norm(self.processed_image + (torch.tensor([73.15835921, 82.90891754, 72.39239876]) / 255.0)[None, :, None, None].cuda(), 6)
            #                 # + 0.01 * torch.norm(self.processed_image, 1)
            # class_loss = irrelevant_probs_sum * 30 -output[1] + 0.01 * torch.norm(self.processed_image + (torch.tensor([73.15835921, 82.90891754, 72.39239876]) / 255.0)[None, :, None, None].cuda(), 1)
            # + 0.01 * torch.norm(self.processed_image, 1)
            print('Iteration:', str(i), 'Loss', "{0:.4f}".format(class_loss.item()), torch.nn.functional.softmax(output, dim=0)[1].item())
            # Zero grads

            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = self.recreate_image2(self.processed_image, if_blur=(i % 4) == -1)
            if i % 10 == 0 or i < 10:
                # Save image
                im_path = '../generated/c_specific_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)

        # imgs = self.processed_image.cpu().detach().numpy()[:, ::-1, :, :]
        # imgs = np.transpose(imgs, [0,2,3,1])
        #
        # plt.imshow(imgs[0])
        # plt.show()
        # im_path = '../generated/final.jpg'
        # self.created_image = self.recreate_image2(self.processed_image)
        # plt.imshow(self.created_image)
        # plt.show()
        # save_image(self.created_image, im_path)
        return self.processed_image

def total_variation_loss(img, beta):
    res = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)) + ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2).permute(0, 1, 3, 2))
    assert res.min() >= 0.0
    res = res.pow(beta / 2).sum() #/ (64.0 * 64.0)
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
        np_arr = (np_arr*255).astype(np.uint8)
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


if __name__ == '__main__':
    # target_class = 130  # Flamingo
    # pretrained_model = models.alexnet(pretrained=True)

    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
    param_file = 'params/bigger_reg_4_4_4.json'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
    # param_file = 'old_params/sample_all.json'

    trained_model = load_trained_model(param_file, save_model_path)
    csig = ClassSpecificImageGeneration(trained_model, 25) #oval face == 25, mustache=22, goatee=16, heavy makeup=18,6: 'Big_Lips',9: 'Blond_Hair', 31: 'Smiling', 32: 'Straight_Hair'
    csig.generate()