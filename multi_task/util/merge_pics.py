from PIL import Image
import os
import glob
import math
from pathlib import Path

from multi_task.util.util import images_list_to_grid_image


def merge_pics_in_folder_structure(folders, folder_out):
    Path(folder_out).mkdir(exist_ok=True)

    contents = list(zip(*[list(os.scandir(folder)) for folder in folders]))

    for i in range(len(contents)):#assume all folders have the same number of everything
        curs = contents[i]
        if curs[0].is_dir():
            merge_pics_in_folder_structure([dir.path for dir in curs], folder_out + '/' + curs[0].name)
        else:
            ims = []
            for j in range(len(curs)):
                ims.append(Image.open(curs[j].path))

            new_im = images_list_to_grid_image(ims)
            new_im.save(folder_out + '/' + curs[0].name)


if __name__ == '__main__':
    merge_pics_in_folder_structure(['big_ordinary_generated_imshow_00_50_on_June_24...7',
                                    'big_ordinary_generated_imshow_00_50_on_June_24...22',
                                    'big_ordinary_generated_imshow_00_50_on_June_24...52',
                                    'big_ordinary_generated_imshow_00_50_on_June_24...82',
                                    'big_ordinary_generated_imshow_00_50_on_June_24...112',
                                    'big_ordinary_generated_imshow_12_18_on_June_24...7',
                                    'big_ordinary_generated_imshow_12_18_on_June_24...22',
                                    'big_ordinary_generated_imshow_12_18_on_June_24...46',
                                    ],
                                   'generated_00_50_on_June_24_merged_7_22_52_82_112_AND_12_18_on_June_24_7_22_46')
