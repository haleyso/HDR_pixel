#### after process_raw_data, run this to make a train test split ###

import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

data_dir = "/media/data4b/haleyso/HDRPlusData/train"

names = os.listdir(data_dir)

min_h = 100000
min_w = 100000

min_val = 100000
max_val = 0

for name in names:

    folder = os.path.join(data_dir, name)

    raw_image = np.load(os.path.join(folder,'raw_demosaiced.npy'))
    jpg_image = plt.imread(os.path.join(folder, 'final.jpg'))

    # print(raw_image.shape, jpg_image.shape)

    if (raw_image.shape != jpg_image.shape):
        # print(name, raw_image.shape, jpg_image.shape)
        print(name)

    h,w,c = raw_image.shape
    if h < min_h:
        min_h = h
    if w < min_w:
        min_w = w

    if raw_image.max() > max_val:
        max_val = raw_image.max()
    if raw_image.min() < min_val:
        min_val = raw_image.min()

print(min_h, min_w, min_val, max_val)