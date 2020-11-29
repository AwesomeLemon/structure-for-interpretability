from torch.utils import data
import torch
import torchvision
import torchvision.transforms as transforms

# class CIFAR10(data.Dataset):
#     def __init__(self, root, split="train", is_transform=False, augmentations=None):
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#         self.dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, split="train", if_no_transform=False, if_6vsAll=False):

        if_test_split = False

        if split == 'train':
            torchvision.datasets.CIFAR10.train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ]
            if not if_no_transform:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                    transforms.RandomAffine(
                        degrees=(0, 5),
                        translate=(0.1, 0.1),
                        scale=(1.0, 1.2)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = transforms.ToTensor()
        elif split == 'val':
            torchvision.datasets.CIFAR10.train_list = [['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']]
            if not if_no_transform:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = transforms.ToTensor()
        elif split == 'test':
            if_test_split = True
            if not if_no_transform:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = transforms.ToTensor()
        else:
            raise NotImplementedError('Other splits not yet supported')

        target_transform = None
        if if_6vsAll:
            target_transform = lambda lbl: 1 if lbl == 6 else 0
        super().__init__(root, train=not if_test_split, download=True,
                         transform=transform, target_transform=target_transform)