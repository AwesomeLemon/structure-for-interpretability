# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

'''RelGAN: https://github.com/willylulu/RelGAN'''

import sys
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=18, help="attribute number")
parser.add_argument("-o", "--output", type=str, default='anno_dic.npy', help="output file")
args = parser.parse_args()

annos = open('list_attr_celeba.txt').readlines()

attrs = str.split(annos[1])
print(attrs)

if args.number == 40:
    new_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
elif args.number == 18:
    new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
elif args.number == 17:
    new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
elif  args.number == 10:
    new_attrs = ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Young', 'Mouth_Slightly_Open']
elif  args.number == 9:
    new_attrs = ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Young']
elif  args.number == 5:
    new_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
elif  args.number == 4:
    new_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Mouth_Slightly_Open']
else:
    print('You can only choose 40, 17, 9, 10, 5, 4  combination')
    exit()

new_attrs_index = []
for x in new_attrs:
    new_attrs_index.append(attrs.index(x))
print(new_attrs_index)

annosAry = {}
for i in range(2,len(annos)):
    anno = str.split(annos[i])
    temp = [(int(i)+1)/2 for i in anno[1:]]
    temp2 = []
    for ii in new_attrs_index:
        temp2.append(temp[ii])
    annosAry[anno[0]] = temp2
    
print(annosAry["000001.jpg"])
print(len(annosAry["000001.jpg"]))

np.save(args.output, annosAry)

img_list = open('image_list.txt').readlines()
imgIndex = [None]*len(img_list)

for i in range(1,len(img_list)):
    temp = str.split(img_list[i])
    imgIndex[int(temp[0])] = temp[2]
    
print(imgIndex[29999])

np.save("imgIndex.npy", imgIndex)