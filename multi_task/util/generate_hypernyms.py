from collections import defaultdict

import os
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from glob import glob
from multi_task.util.dicts import imagenet_dict
import numpy as np

def synset_dict_to_names(sdict):
    new_dict = {}#defaultdict(list, sdict)
    def make_readable(s_name):
        return s_name[:s_name.find('.')].replace('_', ' ')

    for k, v in sdict.items():
        readable_key = make_readable(k.name())
        if readable_key == 'crane' and k.offset() == 2012849:
            readable_key = 'crane bird'
        elif readable_key == 'cardigan' and k.offset() == 2113186:
            readable_key = 'cardigan dog'
        elif readable_key == 'maillot' and k.offset() == 3710721:
            readable_key = 'maillot bathing'
        if readable_key in new_dict:
            print(f'Key {readable_key} is duplicated!')
        if type(v) == Synset:
            new_dict[readable_key] = make_readable(v.name())
        else:
            new_dict[readable_key] = [make_readable(x.name()) for x in v]
    return new_dict

def generate_hypernyms(original_synsets, n_stop):
    work_set_old = list(original_synsets)
    synset_sets = [set([s]) for s in original_synsets]
    hnym_dict = dict(zip(original_synsets, synset_sets))
    hnym_dict = defaultdict(set, hnym_dict)

    while len(work_set_old) > 0:
        def find_hypernym(s):
            #problem: a particular meaning of the word may have no hypernym
            #solution: find synsets correpsonding to other meanings and take their hypernym
            hypernyms = s.hypernyms()
            if len(hypernyms) > 0:
                return hypernyms[0]
            all_possible_synsets = wordnet.synsets(s.name()[:s.name().find('.')])
            for cur_alternative_synset in all_possible_synsets:
                # print(s.name()[:s.name().find('.')])
                # cur_alternative_synset = all_possible_synsets[i]
                if len(cur_alternative_synset.hypernyms()) > 0:
                    return cur_alternative_synset.hypernyms()[0]
                else:
                    print(s)
            return None
        work_set_new = [find_hypernym(s) for s in work_set_old]
        work_set_new_copy = set([x for x in work_set_new if x is not None])
        for s_child, s_parent in zip(work_set_old, work_set_new):
            if s_parent is None:
                print(s_child, s_parent)
                continue
            hnym_dict[s_parent] = hnym_dict[s_parent].union(hnym_dict[s_child])
            if len(hnym_dict[s_parent]) >= n_stop:
                if s_parent in work_set_new_copy: # can't remove twice
                    work_set_new_copy.remove(s_parent)
        work_set_old = work_set_new_copy#[x for x in work_set_new_copy if x is not None]
        print(len(work_set_old))

    # print(synset_dict_to_names(hnym_dict))
    # create reverse dict
    hnyms_sorted = list(sorted(hnym_dict.items(), key=lambda item: len(item[1]), reverse=True))
    res = {}
    for s in original_synsets:
        if_found_hnym = False
        for hnym_and_children in hnyms_sorted:
            hnym, children = hnym_and_children
            if s in children:
                if len(children) == 1:
                    print(s)
                res[s] = hnym
                if_found_hnym = True
                break
        if not if_found_hnym:
            print(s)

    return res

# imagenet_synset_names = [name.replace(' ', '_') for name in imagenet_dict.values()]
# imagenet_synsets = [wordnet.synsets(sn)[0] for sn in imagenet_synset_names]
# print(glob("/mnt/raid/data/ni/dnn/imagenet2012/val/*/"))
subfolders = list(sorted([ f.name for f in os.scandir("/mnt/raid/data/ni/dnn/imagenet2012/val") if f.is_dir() ]))
imagenet_synsets = np.array([wordnet.synset_from_pos_and_offset('n', int(folder_wnid[1:]))
                             for folder_wnid in subfolders])#[:10]#[[0, 1, 7, 84]]
res = generate_hypernyms(imagenet_synsets, n_stop=2)

res = synset_dict_to_names(res)
print(res)
x = 1
#[(parent.name(), len(children)) for parent, children in hnym_dict.items() if wordnet.synsets('groom')[0] in children]