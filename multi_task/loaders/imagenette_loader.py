from torch.utils import data
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[109.5388/255.,118.6897/255.,124.6901/255.],
                                     std=[0.224, 0.224, 0.224])
                ])
dataset = torchvision.datasets.ImageFolder('/mnt/raid/data/chebykin/imagenette2-320/train',
                                           transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
for batch in loader:
    x = 1
