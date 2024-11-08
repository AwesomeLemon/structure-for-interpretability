import glob
import re

import imageio
import numpy as np
# import scipy.misc as m
import skimage
import torch
from torch.utils import data
from PIL import Image


class CELEBA(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(32, 32), augmentations=None,
                 subtract_mean=True, custom_img_paths=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.subtract_mean = subtract_mean
        self.n_classes = 40
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])  # TODO(compute this mean)
        self.files = {}
        self.labels = {}
        self.custom_img_paths = custom_img_paths

        self.label_file = self.root + "/Anno/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'png', label_line.split(' ')[0])
            label_txt = list(map(lambda x: int(x), re.sub('-1', '0', label_line).split()[1:]))
            label_map[f_name] = label_txt

        self.all_files = glob.glob(self.root + '/Img/img_align_celeba_png/*.png')
        if self.split != 'custom':
            with open(root + '//Eval/list_eval_partition.txt', 'r') as f:
                fl = f.read().split('\n')
                fl.pop()
                if 'train' == self.split:
                    selected_files = list(filter(lambda x: x.split(' ')[1] == '0', fl))  # [:2000]
                    # selected_files = selected_files[:int(len(selected_files) * (0.9))]
                if 'train2' == self.split:
                    raise NotImplementedError('train2 is currently unavailable')
                    # selected_files = list(filter(lambda x: x.split(' ')[1] == '0', fl))
                    # selected_files = selected_files[int(len(selected_files) * (0.9)):]
                elif 'val' == self.split:
                    selected_files = list(filter(lambda x: x.split(' ')[1] == '1', fl))  # [:78]
                elif 'val_random' == self.split:
                    all_files = np.array(list(filter(lambda x: x.split(' ')[1] == '1', fl)))
                    n_pics_total = len(all_files)
                    n_pics_to_use = 1700
                    idx = np.random.choice(n_pics_total, size=n_pics_to_use, replace=False)
                    selected_files = all_files[idx]
                elif 'test' == self.split:
                    selected_files = list(filter(lambda x: x.split(' ')[1] == '2', fl))
                # elif 'custom' in self.split:
                #     selected_files = list(map(lambda x:x.split('/')[-1], self.custom_img_paths))

                selected_file_names = list(map(lambda x: re.sub('jpg', 'png', x.split(' ')[0]), selected_files))

            base_path = '/'.join(self.all_files[0].split('/')[:-1])
            self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]),
                                              set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(
                                                  set(selected_file_names))))  # [:10000]
            self.labels[self.split] = list(map(lambda x: label_map[x],
                                               set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(
                                                   set(selected_file_names))))  # [:10000]
        else:
            self.files[self.split] = self.custom_img_paths
            self.labels[self.split] = len(self.custom_img_paths) * [[-17]]  # don't have labels for arbitrary images
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                            'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                            'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                            'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        # print(img_path)
        label = self.labels[self.split][index]

        if self.augmentations is None:
            img = imageio.imread(img_path)
            if self.is_transform:
                img = self.transform_img(img)
        else:
            # ATTENTION: I wrote this more efficient pipeline when I introduced augmentations
            # img = self.augmentations(np.array(img, dtype=np.uint8))
            img = Image.open(img_path)
            # to bgr:
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r)) #no "BGR" mode, but it doesn't matter
            img = self.augmentations(img)

        return [img] + label + [img_path]

    def get_item_by_path(self, img_path):
        img = imageio.imread(img_path)
        if self.is_transform:
            img = self.transform_img(img)
        return img

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        img = img[:, :, ::-1]  # apparently BGR is used for some reason. Fine, whatever. But this screws up pretrained weights.
        img = img.astype(np.float64)
        if self.subtract_mean:
            img -= self.mean
        img = skimage.transform.resize(img, (self.img_size[0], self.img_size[1]), order=1)
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    local_path = "/mnt/raid/data/chebykin/celeba"
    dst = CELEBA(local_path, is_transform=True, augmentations=None, img_size=(128, 128))
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    for i, data in enumerate(trainloader):
        labels = data[1:]
        if labels[23].item() != 1:
            continue
        imgs = data[0].numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        plt.imshow(imgs[0])
        plt.title(labels[23].numpy())
        # f, axarr = plt.subplots(bs,4, squeeze=False)
        #
        # for j in range(bs):
        #     axarr[j][0].imshow(imgs[j])
        # axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        # axarr[j][2].imshow(imgs[j,0,:,:])
        # axarr[j][3].imshow(imgs[j,1,:,:])
        plt.show()
        # a = raw_input()
        # if a == 'ex':
        #     break
        # else:
        #     plt.close()
