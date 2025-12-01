import os
import re
import string
import torch

from torch.utils.data.dataset import Dataset

import pandas as pd


class MNISTDataset(Dataset):
    def __init__(self, pd_data, sample_factor = 1.0, transform=None, target_transform=None):
        self.__pd_data = pd_data
        self.__sample_factor = sample_factor
        self.__transform = transform
        self.__target_transform = target_transform
        self.__features = self.__pd_data.iloc[:, :-1]
        self.__labels = self.__pd_data.iloc[:, -1]

    def __str__(self):
        object_str = "Dataset(" + str(self.__len__()) + ")"
        return object_str

    def __repr__(self):
        object_repr = "Dataset: \n"
        object_repr += "\t Events: \t" + str(self.__len__()) + "\n"
        return object_repr

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, index: int):
        input_features = self.__features.iloc[index].values.astype(float)
        target_value = self.__labels.iloc[index]

        input_features_tensor = torch.tensor(input_features.reshape(28, 28), dtype=torch.float32)
        
        # If your convolutional layers expect a 1-channel image tensor, uncomment the following line:
        input_features_tensor = input_features_tensor.unsqueeze(0)  # Adds a channel dimension, making it 1x28x28
        
        target_value_tensor = torch.tensor(target_value, dtype=torch.long)

        if self.__transform:
            input_features_tensor = self.__transform(input_features_tensor)
        if self.__target_transform:
            target_value_tensor = self.__target_transform(target_value_tensor)

        return input_features_tensor, target_value_tensor

    def get_n_features(self):
        return self.__features.shape[1]


