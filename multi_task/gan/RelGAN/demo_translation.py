# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

'''RelGAN: https://github.com/willylulu/RelGAN'''

import os
import sys
import random
# from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from keras import backend as K
from skimage import io, transform
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from torch import Tensor

try:
    from multi_task.gan.RelGAN.module import *
except:
    from gan.RelGAN.module import *

import argparse

# new_attrs = ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male',
#                  'Mustache', 'Pale_Skin', 'Smiling', 'Young', 'Mouth_Slightly_Open']
# new_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Mouth_Slightly_Open']
# new_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
#                             'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
#                             'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
#                             'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
#                             'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
#                             'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
#                             'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee',
             'Gray_Hair', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair',
             'Wavy_Hair', 'Wearing_Hat', 'Young']
# new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee',
#              'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair',
#              'Wavy_Hair', 'Wearing_Hat', 'Young']
n_attrs = len(new_attrs)


def tileAttr(x):
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=2)
    return tf.tile(x, [1, 256, 256, 1])


def tileAttr2(x):
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=2)
    return tf.tile(x, [1, 4, 4, 1])


# train_path = '/share/diskB/willy/GanExample/FaceAttributeChange_StarGAN/model/generator499.h5'

def generate_with_changed_attributes(relGan, img, attrs_to_change_dict):
    # img = io.imread(img)
    # img = transform.resize(img, [256, 256])
    # img = img[:, :, :3]
    # img = img * 2 - 1

    def get_attr_with_specific_nonzero(kwargs):
        temp = np.zeros([n_attrs])
        for attr_name, value in kwargs.items():
            temp[new_attrs.index(attr_name)] = value
        return temp

    attr = get_attr_with_specific_nonzero(attrs_to_change_dict)
    attr = np.concatenate([np.expand_dims(attr, axis=0)], axis=0)

    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    else:
        img = np.expand_dims(img, axis=0)
    img = np.tile(img, [1, 1, 1, 1])
    #
    # print(img.shape)
    # print(len(attr))
    # print(attr[0].shape)
    img_generated, _ = relGan.predict([img, attr])
    img_generated = (img_generated / 2 + 0.5) * 255
    img_generated = img_generated.astype(np.uint8)
    return img_generated


def orthogonal(w):
    w_kw = K.int_shape(w)[0]
    w_kh = K.int_shape(w)[1]
    w_w = K.int_shape(w)[2]
    w_h = K.int_shape(w)[3]

    temp = 0
    for i in range(w_kw):
        for j in range(w_kh):
            wwt = tf.matmul(tf.transpose(w[i, j]), w[i, j])
            mi = K.ones_like(wwt) - K.identity(wwt)
            a = wwt * mi
            a = tf.matmul(tf.transpose(a), a)
            a = a * K.identity(a)
            temp += K.sum(a)
    return 1e-4 * temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--device", type=str, default='1')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    img_shape = (256, 256, 3)
    vec_shape = (n_attrs,)
    imgA_input = Input(shape=img_shape)
    vec_input_pos = Input(shape=vec_shape)

    for idx in range(500, 1501, 50):
        train_path = f'/mnt/raid/data/chebykin/pycharm_project_AA/gans/model/generator{idx}.h5'

        g_out = generator(imgA_input, vec_input_pos, 256)
        relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)
        relGan.load_weights(train_path)

        img = io.imread('test_img/j.png')
        img = transform.resize(img, [256, 256])
        img = img[:, :, :3]
        img = img * 2 - 1
        generate_with_changed_attributes(relGan, img, {'Mouth_Slightly_Open': 1})

        # relGan.summary()
        # lengh = 10
        # temp = [None]*lengh
        # temp[0] = testPic('test_img/j.png',0)
        # temp[1] = testPic('test_img/c.2.jpg',0)
        # temp[2] = testPic('test_img/es.png',1)
        # temp[3] = testPic('test_img/e.2.png',1)
        # temp[4] = testPic('test_img/g.2.png',1)
        # temp[5] = testPic('test_img/y3.png',1)
        # temp[6] = testPic('test_img/f1.png',1,glasses=-1)
        # temp[7] = testPic('test_img/j1.png',0,glasses=-1)
        # temp[8] = testPic('test_img/c3.png',0)
        # temp[9] = testPic('test_img/g3.png',1,glasses=-1)
        #
        # new_im = Image.new('RGB', (256*10, 256*lengh))
        # for jj in range(lengh):
        #     index = jj
        #     image = temp[index]
        #     new_im.paste(Image.fromarray(image,"RGB"), (0,256*jj))
        # new_im.save(f'/mnt/raid/data/chebykin/pycharm_project_AA/relgan_test/test_v_{idx}.jpg')
