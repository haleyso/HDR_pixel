import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
import kornia

def get_patch(raw, target, patch_size):

    c,h,w = raw.size()
    # print(h,w,patch_size)
    h_start = np.random.randint(0, h-patch_size)
    w_start = np.random.randint(0, w-patch_size)

    raw_crop = raw[:,h_start:h_start+patch_size, w_start:w_start+patch_size]
    target_crop = target[:,h_start:h_start+patch_size, w_start:w_start+patch_size]

    return raw_crop, target_crop



class TrainDataset(Dataset):
    """

        raw_image: [0,65535] uint16 
                - raw_demosaiced.npy  raw 3 channel image  -- see script in dataset folder (merged.dng is the real raw image)
        jpg_image: [0,255] uint8

    """

    def __init__(self, data_dir, transform=None, patch_size=50, data_type='rgb'):
        self.data_dir = data_dir
        self.names = os.listdir(data_dir)
        self.transform = transform
        self.patch_size = patch_size 
        self.data_type = data_type

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        folder = os.path.join(self.data_dir, self.names[index])
        raw_path = os.path.join(folder, 'raw_demosaiced.npy') # merged.dng is the raw raw image. see
        jpg_path = os.path.join(folder, 'final.jpg')

        raw_image = np.load(raw_path)       # dtype('uint16')
        jpg_image = plt.imread(jpg_path)    # dtype('uint8')

        name = self.names[index]
        if raw_image.shape != jpg_image.shape:
            print('image shapes do not match', name,raw_image.shape, jpg_image.shape )
            sys.exit()
        

        # (C, H, W)
        raw_image = torch.tensor(np.transpose(raw_image / np.max(raw_image) , (2, 0, 1)), dtype=torch.float32)
        jpg_image = torch.tensor(np.transpose(jpg_image/np.max(jpg_image)  , (2, 0, 1)), dtype=torch.float32) 

        if self.transform:
            raw_image = self.transform(raw_image)
            jpg_image = self.transform(jpg_image)
        
        # get little patches
        raw_patch, jpg_patch = get_patch(raw_image, jpg_image, self.patch_size)
        raw_image, jpg_image = get_patch(raw_image, jpg_image, 700)
        # raw_image = raw_image[:, 0:789,0:789]
        # jpg_image = jpg_image[:, 0:789,0:789]

        return {'raw_image': raw_image, 'raw_patch': raw_patch, 'jpg_image': jpg_image, 'jpg_patch': jpg_patch, 'name': name}
        # return {'raw_patch': raw_patch, 'jpg_patch': jpg_patch, 'name': name}


class LabTrainDataset(Dataset):
    """
        TO lab space -- luminance 
    """

    def __init__(self, data_dir, transform=None, patch_size=50, data_type='rgb'):
        self.data_dir = data_dir
        self.names = os.listdir(data_dir)
        self.transform = transform
        self.patch_size = patch_size 
        self.data_type = data_type

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        folder = os.path.join(self.data_dir, self.names[index])
        raw_path = os.path.join(folder, 'raw_demosaiced.npy') # merged.dng is the raw raw image. see
        jpg_path = os.path.join(folder, 'final.jpg')

        raw_image = np.load(raw_path)       # dtype('uint16')
        jpg_image = plt.imread(jpg_path)    # dtype('uint8')

        name = self.names[index]
        if raw_image.shape != jpg_image.shape:
            print('image shapes do not match', name,raw_image.shape, jpg_image.shape )
            sys.exit()
        

        # (C, H, W)
        raw_image = torch.tensor(np.transpose(raw_image / np.max(raw_image) , (2, 0, 1)), dtype=torch.float32)
        jpg_image = torch.tensor(np.transpose(jpg_image/np.max(jpg_image)  , (2, 0, 1)), dtype=torch.float32) 

        # to Lab Space
        raw_image = kornia.color.rgb_to_lab(raw_image)
        jpg_image = kornia.color.rgb_to_lab(jpg_image)

        if self.transform:
            raw_image = self.transform(raw_image)
            jpg_image = self.transform(jpg_image)
        
        # get little patches
        raw_patch, jpg_patch = get_patch(raw_image, jpg_image, self.patch_size)
        raw_image, jpg_image = get_patch(raw_image, jpg_image, 700)
        # raw_image = raw_image[:, 0:789,0:789]
        # jpg_image = jpg_image[:, 0:789,0:789]

        return {'raw_image': raw_image, 'raw_patch': raw_patch, 'jpg_image': jpg_image, 'jpg_patch': jpg_patch, 'name': name}
        # return {'raw_patch': raw_patch, 'jpg_patch': jpg_patch, 'name': name}


class InferDataset(Dataset):
    """
        raw_image
        jpg_image (target image) 

    """

    def __init__(self, data_dir, transform=None, data_type='rgb'):
        self.data_dir = data_dir
        self.names = os.listdir(data_dir)
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        folder = os.path.join(self.data_dir, self.names[index])
        raw_path = os.path.join(folder, 'raw_demosaiced.npy') # merged.dng is the raw raw image. see
        jpg_path = os.path.join(folder, 'final.jpg')

        raw_image = np.load(raw_path)       # dtype('uint16')
        jpg_image = plt.imread(jpg_path)    # dtype('uint8')

        if raw_image.shape != jpg_image.shape:
            print('image shapes do not match')
            sys.exit()
        
        h,w,c = raw_image.shape
        name = self.names[index]

        # (C, H, W)
        raw_image = torch.tensor(np.transpose(raw_image / np.max(raw_image) , (2, 0, 1)), dtype=torch.float32)
        jpg_image = torch.tensor(np.transpose(jpg_image/np.max(jpg_image)  , (2, 0, 1)), dtype=torch.float32) 


        # print(int(min(h,w)))
        # sys.exit()
        raw_image, jpg_image = get_patch(raw_image, jpg_image, int(min(h,w)-1))
        # get little patches
        # raw_patch, jpg_patch = get_patch(raw_image, jgp_image, self.patch_size)

        # return {'raw_image': raw_image, 'raw_patch': raw_patch, 'jgp_image': jgp_image, 'jpg_patch': jpg_patch, 'name': name}
        return {'raw_image': raw_image, 'jpg_image': jpg_image, 'name': name}


