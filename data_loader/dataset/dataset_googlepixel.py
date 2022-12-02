import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from torch.utils.data import Dataset


def get_patch(raw, target, patch_size):

    c,h,w = raw.size()
    h_start = np.random.randint(0, h-patch_size-1)
    w_start = np.random.randint(0, w-patch_size-1)

    raw_crop = raw[:,h_start:h_start+patch_size, w_start:w_start+patch_size]
    target_crop = target[:,h_start:h_start+patch_size, w_start:w_start+patch_size]

    return raw_crop, target_crop



class TrainDataset(Dataset):
    """

        raw_image: [0,65535] uint16 
                - raw_demosaiced.npy  raw 3 channel image  -- see script in dataset folder (merged.dng is the real raw image)
        jpg_image: [0,255] uint8

    """

    def __init__(self, data_dir, transform=None, patch_size=50):
        self.data_dir = data_dir
        self.names = os.listdir(data_dir)
        self.transform = transform
        self.patch_size = patch_size 

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        folder = os.path.join(self.data_dir, self.names[index])
        raw_path = os.path.join(folder, 'raw_demosaiced.npy') # merged.dng is the raw raw image. see
        jpg_path = os.path.join(folder, 'final.jpg')

        raw_image = np.load(raw_path)       # dtype('uint16')
        jgp_image = plt.imread(jpg_path)    # dtype('uint8')

        name = self.names[index]
        if raw_image.shape != jgp_image.shape:
            print('image shapes do not match', name,raw_image.shape, jgp_image.shape )
            sys.exit()
        


        # (C, H, W)
        raw_image = torch.tensor(np.transpose(raw_image / np.max(raw_image), (2, 0, 1)), dtype=torch.float32)
        jgp_image = torch.tensor(np.transpose(jgp_image/np.max(jgp_image), (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            raw_image = self.transform(raw_image)
            jgp_image = self.transform(jgp_image)
        
        # get little patches
        raw_patch, jpg_patch = get_patch(raw_image, jgp_image, self.patch_size)

        # return {'raw_image': raw_image, 'raw_patch': raw_patch, 'jgp_image': jgp_image, 'jpg_patch': jpg_patch, 'name': name}
        return {'raw_patch': raw_patch, 'jpg_patch': jpg_patch, 'name': name}


class InferDataset(Dataset):
    """
        raw_image
        jpg_image (target image) 

    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        folder = os.path.join(self.data_dir, self.names[index])
        raw_path = os.path.join(folder, 'raw_demosaiced.npy') # merged.dng is the raw raw image. see
        jpg_path = os.path.join(folder, 'final.jpg')

        raw_image = np.load(raw_path)       # dtype('uint16')
        jgp_image = plt.imread(jpg_path)    # dtype('uint8')

        if raw_image.shape != jgp_image.shape:
            print('image shapes do not match')
            sys.exit()
        
        name = self.names[index]

        # (C, H, W)
        raw_image = torch.tensor(np.transpose(raw_image / np.max(raw_image), (2, 0, 1)), dtype=torch.float32)
        jgp_image = torch.tensor(np.transpose(jgp_image/np.max(jgp_image), (2, 0, 1)), dtype=torch.float32)
        
        # get little patches
        # raw_patch, jpg_patch = get_patch(raw_image, jgp_image, self.patch_size)

        # return {'raw_image': raw_image, 'raw_patch': raw_patch, 'jgp_image': jgp_image, 'jpg_patch': jpg_patch, 'name': name}
        return {'raw_image': raw_image, 'jgp_image': jgp_image, 'name': name}


