'''
created on 2.14 2019
author: chenweiwei@ict.ac.cn
'''

import time
import numpy as np 
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from custom_dataset import CustomDatasetFromCSV


class MLPNet(nn.Module):
    def __init__(self,INPUT_LENS,HIDDEN_NODE_FC1,HIDDEN_NODE_FC2,HIDDEN_NODE_FC3,N_CLASSES,keep_rate=0):
        super(MLPNet,self).__init__()
        if not keep_rate:
            keep_rate=0.5
        self.keep_rate = keep_rate

        self.fc1 = nn.Linear(INPUT_LENS,HIDDEN_NODE_FC1)
        self.relu1 = nn.ReLU()
        #self.fc1_drop = nn.Dropout(1 - keep_rate)
        self.fc2 = nn.Linear(HIDDEN_NODE_FC1,HIDDEN_NODE_FC2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN_NODE_FC2,HIDDEN_NODE_FC3)
        self.relu3 = nn.ReLU()
        #self.fc2_drop = nn.Dropout(1 - keep_rate)
        self.out = nn.Linear(HIDDEN_NODE_FC3,N_CLASSES)

    def forward(self,x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.out(out)

        return out

