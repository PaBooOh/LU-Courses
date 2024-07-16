
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import torchvision
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import datasets.mnist.loader as mnist


# extract dataset
train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
train_label = pd.read_csv('./datasets/train_out.csv',header=None).values
test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
test_label = pd.read_csv('./datasets/test_out.csv',header=None).values

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, set_X, set_Y):
        self.data_X = set_X
        self.label_Y = set_Y
    def __getitem__(self, index):
        data = self.data_X[index]
        label = self.label_Y[index]
        return data, label
    def __len__(self):
        return len(self.data_X)

torch_data = MnistDataset(train_set, train_label)
train_loader = Data.DataLoader(torch_data, batch_size=50, shuffle=True)

