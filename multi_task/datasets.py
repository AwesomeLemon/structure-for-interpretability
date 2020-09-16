import torch
from torchvision import transforms
from loaders.cifar10_loader import CIFAR10
from loaders.fashionmnist_loader import FashionMNIST
from loaders.segmentation_augmentations import *
from loaders.celeba_loader import CELEBA
from loaders.cityscapes_loader import CITYSCAPES
# from multi_task.loaders.celeba_loader import CELEBA
from torch.utils.data import ConcatDataset
# Setup Augmentations
# cityscapes_augmentations= Compose([RandomRotate(10),
#                                    RandomHorizontallyFlip()])

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' == params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'cityscapes' == params['dataset']:
        cityscapes_augmentations = Compose([RandomRotate(10),
                                            RandomHorizontallyFlip()])
        train_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['train'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']), augmentations=cityscapes_augmentations)
        val_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['val'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size']* 4, shuffle=False, num_workers=8)
        return train_loader, val_loader, None

    if 'celeba' == params['dataset']:
        train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        # train2_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train2',img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val1_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
        # train2_loader = torch.utils.data.DataLoader(train2_dst, batch_size=params['batch_size'], shuffle=True,num_workers=4)
        val_loader = torch.utils.data.DataLoader(val1_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8, pin_memory=False)
        # val2_loader = torch.utils.data.DataLoader(val2_dst, batch_size=params['batch_size'], num_workers=4,
        #                                           shuffle=True)
        return train_loader, val_loader, None#train2_loader

    if ('cifar10' == params['dataset']) or ('cifar10_singletask' == params['dataset']):
        train_dst = CIFAR10(root=configs[params['dataset']]['path'], split='train')
        val_dst = CIFAR10(root=configs[params['dataset']]['path'], split='val')

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)
        return train_loader, val_loader, None

    if 'cifarfashionmnist' == params['dataset']:
        train_dst_cifar = CIFAR10(root=configs['cifarfashionmnist']['path_cifar'], split='train')
        val_dst_cifar = CIFAR10(root=configs['cifarfashionmnist']['path_cifar'], split='val')

        train_dst_fashionmnist = FashionMNIST(root=configs['cifarfashionmnist']['path_fashionmnist'], split='train')
        val_dst_fashionmnist = FashionMNIST(root=configs['cifarfashionmnist']['path_fashionmnist'], split='val')

        train_dst = ConcatDataset([train_dst_cifar, train_dst_fashionmnist])
        val_dst = ConcatDataset([val_dst_cifar, val_dst_fashionmnist])

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)
        return train_loader, val_loader, None

#DANGER! TEST SET SHOULD NOT BE USED UNLESS IN EMERGENCY
def get_test_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'celeba' == params['dataset']:
        test_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val',
                           img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                           augmentations=None)

        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'] * 4,#78
                                                  num_workers=8,
                                                 shuffle=False)

        return test_loader

    if 'cifar10' == params['dataset']:
        test_dst = CIFAR10(root=configs['cifar10']['path'], split='test')

        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)
        return test_loader


def get_random_val_subset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'celeba' == params['dataset']:
        dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val_random',
                           img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']))

        loader = torch.utils.data.DataLoader(dst, batch_size=params['batch_size'] * 4,
                                                  num_workers=1, shuffle=False)

        return loader