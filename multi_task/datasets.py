import torch
from torchvision import transforms
import torchvision
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
from loaders.broden_loader import BrodenDataset, ScaleSegmentation


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

    if 'imagenette_singletask' == params['dataset']:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalization matches code from Network Dissection
            transforms.Normalize(mean=[109.5388 / 255., 118.6897 / 255., 124.6901 / 255.],
                                 std=[0.224, 0.224, 0.224])
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalization matches code from Network Dissection
            transforms.Normalize(mean=[109.5388 / 255., 118.6897 / 255., 124.6901 / 255.],
                                 std=[0.224, 0.224, 0.224])
        ])
        train_dst = torchvision.datasets.ImageFolder(f"{configs[params['dataset']]['path']}/train", transform=transform_train)
        val_dst = torchvision.datasets.ImageFolder(f"{configs[params['dataset']]['path']}/val", transform=transform_val)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)

        return train_loader, val_loader, None

    if 'imagenet_val' == params['dataset']:
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalization matches code from Network Dissection
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        torchvision.datasets.imagenet.ARCHIVE_DICT['devkit']['url'] = \
            "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"
        val_dst = torchvision.datasets.ImageNet('/mnt/raid/data/ni/dnn/imagenet2012', split='val', transform=transform_val,
                                                download=False)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)

        return None, val_loader, None

    if 'broden_val' == params['dataset']:
        bds = BrodenDataset('/mnt/raid/data/chebykin/NetDissect-Lite/dataset', resolution=224,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor()]),
                            transform_segment=transforms.Compose([
                        ScaleSegmentation(224, 224)
                        ]),
                            include_bincount=False, split='train', categories=["object", "part", "scene", "texture"],
                            if_include_path=True)
        val_loader = torch.utils.data.DataLoader(bds, batch_size=params['batch_size'], num_workers=8, shuffle=False)
        return None, val_loader, None


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

    if ('cifar10' == params['dataset']) or ('cifar10_singletask' == params['dataset']):
        test_dst = CIFAR10(root=configs['cifar10']['path'], split='test')

        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8)
        return test_loader

    if 'imagenet_val' == params['dataset']:
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalization matches code from Network Dissection
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        torchvision.datasets.imagenet.ARCHIVE_DICT['devkit']['url'] = \
            "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"
        val_dst = torchvision.datasets.ImageNet('/mnt/raid/data/ni/dnn/imagenet2012', split='val', transform=transform_val,
                                                download=False)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=False, num_workers=8,
                                                 pin_memory=True)

        return val_loader


def get_random_val_subset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'celeba' == params['dataset']:
        dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val_random',
                           img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']))

        loader = torch.utils.data.DataLoader(dst, batch_size=params['batch_size'] * 4,
                                                  num_workers=1, shuffle=False)

        return loader

    if ('cifar10' == params['dataset']) or ('cifar10_singletask' == params['dataset']):
        # selecting random subset seems complicated => return validation loader, but shuffled
        val_dst = CIFAR10(root=configs[params['dataset']]['path'], split='val')

        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'] * 4, shuffle=True, num_workers=8)
        return val_loader