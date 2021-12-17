#based on: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# def show_hp_temp_values(power_usage, tempC_average):
#     """Show image with landmarks"""
#     print('---')
#     print(power_usage, tempC_average)
#     print('---')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        result = {}
        for feature in sample.keys():
            if feature == "timestamp":
                # skip the timestamp, because it is now present as buckets
                continue
            result[feature] = torch.from_numpy(sample[feature])
        return result

class LargeOPDSDataset(Dataset):
    """Dataset containing the Heat Pump power consumption values and the temperature at that time."""

    def __init__(self, csv_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with n columns,
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        tsv_file = pd.read_csv(csv_file, sep=',')
        self.features = tsv_file.columns
        if train:
            self.values = tsv_file[:int(len(tsv_file)*0.8)].reset_index()
        else:
            self.values = tsv_file[int(len(tsv_file)*0.8):].reset_index()

        self.transform = transform

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(len(self.hptempvalues['power_usage']))
        sample = {}
        for feature in self.features:
            sample[feature] = np.array([self.values[feature][idx]])
            # sample[feature] = np.array([float(self.values[feature][idx])])

        if self.transform:
            sample = self.transform(sample)

        return sample

###TEST FUNCTIONS
# dataset = LargeOPDSDataset()

# for i in range(len(dataset)):
#     sample = dataset[i]
#     print(sample)
#     break

# #     print(i, sample['power_usage'].shape, sample['tempC_average'].shape)
# #     show_hp_temp_values(**sample)

# #     if i > 3:
# #         break

# transformed_dataset = LargeOPDSDataset(train=True,
#                                            transform=ToTensor())


# # for i in range(len(transformed_dataset)):
# #     sample = transformed_dataset[i]
# #     print(i, sample['timestamp'].size(), sample['https://interconnectproject.eu/example/DEKNres2_GI'].size())

# #     if i == 3:
# #         break

# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)

# for sample in dataloader:
#     X = sample['timestamp']
#     y = sample['https://interconnectproject.eu/example/DEKNres2_GI']
#     print(y)
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break

# # exit()