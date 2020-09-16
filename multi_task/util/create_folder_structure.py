from PIL import Image
import os
import glob
import math
from pathlib import Path

layers = ['layer1_0',
          'layer1_1_conv1', 'layer1_1',
          'layer2_0_conv1', 'layer2_0',
          'layer2_1_conv1', 'layer2_1',
          'layer3_0_conv1', 'layer3_0',
          'layer3_1_conv1', 'layer3_1',
          'layer4_0_conv1', 'layer4_0',
          'layer4_1_conv1', 'layer4_1',
          ]

folder_in = '/home/alex/attr_hist_for_neuron_all'
folder_out = '/home/alex/structured_attr_hist_for_neuron_all/'
Path(folder_out).mkdir(exist_ok=True)
contents = os.scandir(folder_in)
for dir_entry in contents:
    cur_name = dir_entry.name
    fst_underscore = cur_name.find('_')
    snd_underscore = cur_name.rfind('_')
    layer_idx = int(cur_name[fst_underscore + 1:snd_underscore])
    neuron_idx = cur_name[snd_underscore + 1:cur_name.find('.')]
    folder_out_cur = folder_out + layers[layer_idx] + '/'
    Path(folder_out_cur).mkdir(exist_ok=True)
    Image.open(dir_entry.path).convert('RGB').save(folder_out_cur + neuron_idx + '.jpg')