import argparse
import json
import os

import numpy as np
import torch
import torchvision.utils
from PIL import Image
from keras import Input, Model
from skimage import io
from skimage import transform
from torch.utils import data
from torchvision import transforms

try:
    from multi_task.gan.RelGAN.demo_translation import generate_with_changed_attributes, n_attrs
    from multi_task.gan.RelGAN.module import generator
    from multi_task.gan.attgan.attgan import AttGAN
    from multi_task.gan.attgan.data import CustomDataset
    from multi_task.loaders.celeba_loader import CELEBA
    from multi_task.util.util import task_ind_from_task_name
except:
    from gan.RelGAN.demo_translation import generate_with_changed_attributes, n_attrs
    from gan.RelGAN.module import generator
    from gan.attgan.attgan import AttGAN
    from gan.attgan.data import CustomDataset
    from loaders.celeba_loader import CELEBA
    from util.util import task_ind_from_task_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttributeChanger:
    def __init__(self, gan_name, gan_path, gan_settings_path=None, additional_label_transform_name=None, out_path='gan_generated'):
        self.gan_name = gan_name
        if gan_name == 'AttGAN':
            with open(gan_settings_path, 'r') as f:
                gan_settings = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
            attgan = AttGAN(gan_settings)
            attgan.load(gan_path)
            attgan.eval()
            self.gan = attgan

            self.transform_img = transforms.Compose([
                transforms.Resize((gan_settings.img_size, gan_settings.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            def select_13_labels(label):
                names_13 = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses",
                            "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Young"]
                indices_13 = [task_ind_from_task_name(name) for name in names_13]
                return list(np.array(label)[np.array(indices_13)])

            def normalize_labels(label):  # {0, 1} -> {-0.5, 0.5}
                for i in range(len(label)):
                    label[i] -= 0.5
                return label

            def make_blondness_reasonable(label):
                if label[3] == .5:  # if hair are blond
                    label[3] = -0.8
                return label

            def close_mouth(label):
                label[-5] = -0.5
                return label

            def open_mouth(label):
                label[-5] = 1.0
                return label

            if additional_label_transform_name is None:
                additional_transform = lambda label: label
            elif additional_label_transform_name == 'close_mouth':
                additional_transform = close_mouth
            elif additional_label_transform_name == 'open_mouth':
                additional_transform = open_mouth

            self.transform_label = lambda lbl: torch.tensor(
                additional_transform(make_blondness_reasonable(normalize_labels(select_13_labels(lbl)))))
            self.generate_by_gan = lambda img, lbl, kwargs: self.gan.G(img, lbl)
            self.image_opener = Image.open
        elif gan_name == 'RelGAN':
            img_shape = (256, 256, 3)
            vec_shape = (n_attrs,)
            imgA_input = Input(shape=img_shape)
            vec_input_pos = Input(shape=vec_shape)
            g_out = generator(imgA_input, vec_input_pos, 256)
            relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)
            relGan.load_weights(gan_path)
            self.gan = relGan
            # self.generate_by_gan = lambda img, lbl, kwargs: torch.tensor(generate_with_changed_attributes(relGan, img, kwargs))
            self.generate_by_gan = lambda img, lbl, kwargs: generate_with_changed_attributes(relGan, img, kwargs)

            self.transform_img = lambda img: transform.resize(img, [256, 256])[:, :, :3] * 2 - 1
            self.transform_label = lambda x: torch.tensor(x)
            self.image_opener = io.imread
        else:
            raise ValueError(f'Unknown GAN name: {gan_name}')
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)

    def generate(self, im_paths, labels_true, kwargs=None):
        dataset = CustomDataset(im_paths, labels_true, self.transform_img, self.transform_label, self.image_opener)
        dataloader = data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

        for idx, (img, lbl) in enumerate(dataloader):
            img = img.to(device)
            lbl = lbl.float().to(device)
            # print(lbl)
            res = self.generate_by_gan(img, lbl, kwargs)
            print(res.shape)
            if self.gan_name == 'AttGAN':
                torchvision.utils.save_image(res, os.path.join(self.out_path, f'{idx}.jpg'), normalize=True, range=(-1., 1.))
            elif self.gan_name == 'RelGAN':
                Image.fromarray(res[0], "RGB").save(os.path.join(self.out_path, f'{idx}.jpg'))


if __name__ == '__main__':
    save_file_path = '1000opened_mouths.npy'
    if not os.path.exists(save_file_path):
        local_path = "/mnt/raid/data/chebykin/celeba"
        celeba_dataset = CELEBA(local_path, is_transform=False, augmentations=None)
        n_images = len(celeba_dataset.files[celeba_dataset.split])

        img_paths = []
        labels = []
        for i in range(n_images):
            img_path = celeba_dataset.files[celeba_dataset.split][i].rstrip()
            label = celeba_dataset.labels[celeba_dataset.split][i]

            if label[task_ind_from_task_name('Mouth_Slightly_Open')] == 1:
                img_paths.append(img_path)
                labels.append(label)

            if len(labels) == 1000:
                break

        np.save(save_file_path, (img_paths, labels))
    else:
        img_paths, labels = np.load(save_file_path, allow_pickle=True)
        ac = AttributeChanger('AttGAN', 'gans/384_shortcut1_inject0_none_hq/checkpoint/weights.149.pth',
                              'gans/384_shortcut1_inject0_none_hq/setting.txt')
        # ac = AttributeChanger('AttGAN', 'gans/128_shortcut1_inject0_none/checkpoint/weights.49.pth', 'gans/128_shortcut1_inject0_none/setting.txt')
        ac.generate(img_paths, labels)
        # ac.generate(img_paths, restore_open_mouth(closed_mouth_labels))
