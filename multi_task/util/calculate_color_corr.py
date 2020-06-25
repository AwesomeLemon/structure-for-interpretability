import torch
import torchvision
import numpy as np
import warnings

from multi_task.loaders.celeba_loader import CELEBA

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

if_imagenet = False


if if_imagenet:
    torchvision.datasets.imagenet.ARCHIVE_DICT['devkit']['url'] = \
        "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"

    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
    ])
    data = torchvision.datasets.ImageNet('/mnt/raid/data/ni/dnn/imagenet', download=False,
                                         split='train', transform=transforms)
else:
    data = CELEBA(root="/mnt/raid/data/chebykin/celeba", is_transform=True, split='train',
                  img_size=(192, 168), augmentations=None, subtract_mean=False)
batch_size = 128
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)
if if_imagenet:
    # amount of images in train is slightly different from what's reported at https://www.tensorflow.org/datasets/catalog/imagenet2012
    total_images = 1290129 - 1 #batch_size * 10
    total_pixels = total_images * 224 * 224
else:
    total_images = 162770
    total_pixels = total_images * 192 * 168
if True:
    means = np.array([0., 0., 0.], dtype='float64')
    torch.manual_seed("0")
    for i, batch in enumerate(data_loader):
        if if_imagenet:
            images, _ = batch
        else:
            images = batch[0]
        cur = images.detach().cpu().numpy().astype('float64')
        cur = cur.transpose(1, 0, 2, 3).reshape((3, -1))
        if True:
            tmp = np.sum(cur, axis=1) / (total_pixels)
        else:
            tmp = np.zeros_like(means, dtype='float64')
            for j in range(cur.shape[1]):
                tmp += cur[:, j] / total_images / (224 * 224)

        means += tmp
        if i % 200 == 0:
            print(i, means)

        # print(cur.dtype)
        # print(tmp.dtype)
        # print(means.dtype)
        # print()
        # if i == 9:
        #     break

    print('Total images = ', i * batch_size + images.size(0)) #last batch may be incomplete
    print('Final means = ', means)
    '''
    IMAGENET:
    
    Final means =  [0.49328178 0.46366214 0.41414812]
    Final means =  [0.49332189 0.46368789 0.41414434]
    
    CELEBA:
    
    Final means =  [0.38302392 0.42581415 0.50640459]
    '''
else:
    means = np.array([0.485, 0.456, 0.406], dtype='float64')

covs = np.zeros((3, 3), dtype='float64')

torch.manual_seed("0")
for i, batch in enumerate(data_loader):
    if if_imagenet:
        images, _ = batch
    else:
        images = batch[0]
    cur = images.detach().cpu().numpy().astype('float64')
    cur = cur.transpose(1, 0, 2, 3).reshape((3, -1))
    cur -= means[:, None]
    covs[0, 1] += ((cur[0] * cur[1]).sum()) / (total_pixels)
    covs[0, 2] += ((cur[0] * cur[2]).sum()) / (total_pixels)
    covs[1, 2] += ((cur[1] * cur[2]).sum()) / (total_pixels)

    covs[0, 0] += ((cur[0] * cur[0]).sum()) / (total_pixels)
    covs[1, 1] += ((cur[1] * cur[1]).sum()) / (total_pixels)
    covs[2, 2] += ((cur[2] * cur[2]).sum()) / (total_pixels)

    if i % 200 == 0:
        print(i, covs)

    # print(cur.dtype)
    # print(means.dtype)
    # print()
    # if i == 9:
    #     break

covs[1, 0] = covs[0, 1]
covs[2, 0] = covs[0, 2]
covs[2, 1] = covs[1, 2]
print('Final covs = ', covs)
'''
IMAGENET:
Final covs =  
[[0.07815665 0.067469   0.06028684]
 [0.067469   0.07397008 0.06028684]
 [0.06028684 0.06028684 0.08139698]]
 
Final covs =  
[[0.07818374 0.06747943 0.06029495]
 [0.06747943 0.07397141 0.06029495]
 [0.06029495 0.06029495 0.0813817 ]]
damn, i had a bug where cov(1, 2) == cov(0, 2)

Final covs =  [[0.0806275  0.07111512 0.06518765]
 [0.07111512 0.07734456 0.07420234]
 [0.06518765 0.07420234 0.08621662]]
 
CELEBA:

Final covs =  [[0.08426993 0.08045107 0.07459845]
 [0.08045107 0.0846215  0.08348418]
 [0.07459845 0.08348418 0.09700427]]
 
array([[0.08426993, 0.08045107, 0.07459845],
       [0.08045107, 0.0846215 , 0.08348418],
       [0.07459845, 0.08348418, 0.09700427]])
'''

# x = cur.transpose(1, 0, 2, 3).reshape((3, -1))
# sigma = np.cov(x, rowvar=True)
# U,S,V = np.linalg.svd(sigma)
# epsilon = 1e-10
# ZCAMatrix = U @ np.diag(np.sqrt(S + epsilon)) @ U.T
