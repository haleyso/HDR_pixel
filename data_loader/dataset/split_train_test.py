#### after process_raw_data, run this to make a train test split ###

import os
import numpy as np
from tqdm import tqdm
import random


data_dir = "/media/data4b/haleyso/HDRPlusData/results_20161014"
names = os.listdir(data_dir)

random.shuffle(names)

# take 10% for test so that's 15 images

test_names = names[:15]
print(test_names)


for name in names:
    origin = os.path.join(data_dir,name)
        
    if name in test_names:
        destination = os.path.join('/media/data4b/haleyso/HDRPlusData/test', name )
    else:
        destination = os.path.join('/media/data4b/haleyso/HDRPlusData/train', name )

    cmd = 'mv '+ origin + ' ' +  destination
    # print(cmd)
    os.system(cmd)
