import torch
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")




X_train_tensor = torch.from_numpy(X_train).float()
X_train_tensor.to(device)
del X_train
X_test_tensor = torch.from_numpy(X_test).float()
X_test_tensor.to(device)
del X_test
y_train_tensor = torch.from_numpy(y_train).int()
y_train_tensor.to(device)
del y_train
y_test_tensor = torch.from_numpy(y_test).int()
y_test_tensor.to(device)
del y_test


