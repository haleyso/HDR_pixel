from torchvision import datasets, transforms
from .dataset import dataset_googlepixel

from base import BaseDataLoader
from torch.utils.data import DataLoader
import sys


# TRAIN #
class GooglePixelTrainDataLoader(BaseDataLoader):
    """
    Google Pixel data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, patch_size, shuffle=True, validation_split=0.0, num_workers=1, data_type='rgb', training=True):
        transform = None

        self.data_dir = data_dir
        self.dataset = dataset_googlepixel.TrainDataset(data_dir, transform=transform, patch_size=patch_size, data_type=data_type)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, data_type)

# TEST #
class GooglePixelTestDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size=1, shuffle=False,validation_split=0.0, num_workers=1, data_type = 'rgb', training=False):
        transform = None
        self.dataset = dataset_googlepixel.InferDataset(data_dir, transform=transform, data_type=data_type)

        super(GooglePixelTestDataLoader, self).__init__(self.dataset)


# single
class GPixSingleTrainDataLoader(BaseDataLoader):
    """
    Google Pixel data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, patch_size, shuffle=True, validation_split=0.0, num_workers=1, data_type='rgb', training=True):
        transform = None

        self.data_dir = data_dir
        self.dataset = dataset_googlepixel.LabTrainDataset(data_dir, transform=transform, patch_size=patch_size, data_type=data_type)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, data_type)

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)