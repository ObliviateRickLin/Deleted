import imp
import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import exp
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import *

class DDSafetyDataset(Dataset):
  def __init__(self, path):
    self.path = path #find the corresponding csv file
    self.df = pd.read_excel(path)
    return 

  def __getitem__(self, index):
    df = self.df
    return np.array(df.iloc[index,1:], dtype = np.float32), 0 #zero will not be used because the algorithm is unsupervised. 

  def __len__(self):
    return self.df.shape[0]

class Net(nn.Module):
    def __init__(self, dim, class_num):
        super(Net, self).__init__()
        self.class_num = class_num
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(dim, 50, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(50, 20, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(20, 4, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, dim, bias=True),
            nn.Sigmoid(),
        )
        self.cluster_layer = nn.Linear(4, class_num, bias=False)
        self.cluster_center = torch.rand([class_num, 4], requires_grad=False).cpu()

    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def cluster(self, z):
        return self.cluster_layer(z)
    
    def init_cluster_layer(self, alpha, cluster_center):
        self.cluster_layer.weight.data = 2 * alpha * cluster_center

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (
            F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha
        )

    def predict(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = torch.argmin(distance, dim=1)
        return prediction
    
    def predict_(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = F.softmin(distance, dim=1)
        return prediction

    def set_cluster_centroid(self, mu, cluster_id, alpha):
        self.cluster_layer.weight.data[cluster_id] = 2 * alpha * mu


def inference(data_loader_test, net): 
    #net.eval()
    feature_vector = []
    labels_vector = []
    hard_vector = []
    soft_vector = []
    with torch.no_grad(): 
        for step, (x, y) in enumerate(data_loader_test):
            x = x.cpu()
            with torch.no_grad():
                z = net.encode(x)
                hard = net.predict(z)
                soft = net.predict_(z)
            feature_vector.extend(z.detach().cpu().numpy())
            labels_vector.extend(y.numpy())
            hard_vector.extend(hard.detach().cpu().numpy())
            soft_vector.extend(soft.detach().cpu().numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    hard_vector = np.array(hard_vector)
    soft_vector = np.array(soft_vector)
    return feature_vector, labels_vector, hard_vector, soft_vector

def logistic_transformer(x):
    try:
        y = 1/ (1 + exp(-x))
    except:
        y = 0
    return y

