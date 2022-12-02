#### after process_raw_data, run this to make a train test split ###

import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

data_dir = "/media/data4b/haleyso/HDRPlusData/train"

names = os.listdir(data_dir)

for name in names:

    folder = os.path.join(data_dir, name)

    raw_image = np.load(os.path.join(folder,'raw_demosaiced.npy'))
    jpg_image = plt.imread(os.path.join(folder, 'final.jpg'))

    # print(raw_image.shape, jpg_image.shape)

    if (raw_image.shape != jpg_image.shape):
        # print(name, raw_image.shape, jpg_image.shape)
        print(name)