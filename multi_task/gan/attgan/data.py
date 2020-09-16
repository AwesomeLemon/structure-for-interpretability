# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""
AttGAN: https://github.com/elvisyjlin/AttGAN-PyTorch
Custom datasets for CelebA and CelebA-HQ.
"""

import torch.utils.data as data
from PIL import Image


class CustomDataset(data.Dataset):
    def __init__(self, img_paths, labels, transform_img, transform_label, image_opener=Image.open):
        self.image_opener = image_opener
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.img_paths = img_paths
        self.labels = labels

    def __getitem__(self, index):
        img = self.transform_img(self.image_opener(self.img_paths[index]))
        label = self.transform_label(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.img_paths)


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None

    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1 - att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
                    _set(att, 1 - att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
    return att_batch


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    attrs_default = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()

    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y

    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )
