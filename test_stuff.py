import random

import numpy as np
import torch

filter_num = 6
selected_filter_idx = random.sample(range(filter_num), filter_num // 2)
print(selected_filter_idx)

filter_w = filter_h = 8
batch_size = 17
my_mask = torch.zeros((batch_size, filter_num, filter_w, filter_h))
my_mask[:, selected_filter_idx, :, :] = 1
print(my_mask[:, :, :, 0])