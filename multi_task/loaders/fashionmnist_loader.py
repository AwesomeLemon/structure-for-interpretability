from torch.utils import data
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, split="train"):
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #I left cifar values because network takes both datasets as input, so they should be similar
        ])
        super().__init__(root, train=True, transform=transform, download=True)
        if split == 'train':
            self.data, self.targets = self.data[:-10000], self.targets[:-10000]
            self.targets += 10
        elif split == 'val':
            self.data, self.targets = self.data[-10000:], self.targets[-10000:]
            self.targets += 10
        else:
            raise NotImplementedError('Other splits not yet supported')