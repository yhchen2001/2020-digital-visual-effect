import cv2
import os
import os.path as osp
import numpy as np

dir = '../data/BOT'
tar_dir = '../data/BOT_cut'

for file in np.sort(os.listdir(dir)):
    if osp.splitext(file)[1] in ['.png', '.jpg']:
        im = cv2.imread(osp.join(dir, file))
        h, w = im.shape[0:2]
        im = im[:, w//4:w*3//4]
        cv2.imwrite(osp.join(tar_dir, file), im)
