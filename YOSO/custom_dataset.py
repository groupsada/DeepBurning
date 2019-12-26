import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np 
import utils


def format_data(data):
    try:
        return data.reshape((data.shape[0], 1)) if len(data.shape) == 1 else data
    except AttributeError as e:
        print('ERROR! data is not a numpy object, format_data failed!')
        exit(0)


class CustomDatasetFromCSV(Dataset):
    def __init__(self,csv_path,transform=None):
        """
        args:
            csv_path(string): path of the data csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:,0])
        print(self.labels.shape)
        self.transform = transform
        self.data_len = len(self.data.index)
    
    def __getitem__(self,index):
        single_label = self.labels[index]
        single_data = np.asarray(self.data.iloc[index][1:]).astype(np.float32)
        if self.transform is not None:
            single_data = self.transform(single_data)
        data_as_tensor = torch.from_numpy(single_data)
        return (data_as_tensor,single_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromTxt(Dataset):
    def __init__(self,app_name,train=True,transform=None):
        """
        args:
            app_name(string): such as 'blackscholes','fft','inversek2j',
                        'jmeint','jpeg','kmeans','sobel',
            transform: pytorch transforms for transforms and tensor conversion
        """
        if train == True:
            self.x = np.loadtxt('./data/' + app_name + '/train.x',dtype=np.float32)
            self.y = np.loadtxt('./data/' + app_name + '/train.y',dtype=np.float32)
        else:
            self.x  = np.loadtxt('./data/' + app_name + '/test.x',dtype=np.float32)
            self.y  = np.loadtxt('./data/' + app_name + '/test.y',dtype=np.float32)
        self.transform = transform
        self.x = format_data(self.x)
        self.y = format_data(self.y)
        self.input_size = self.x.shape[1]
        self.out_size = self.y.shape[1]
        self.data_len = self.x.shape[0]

    def __getitem__(self,index):
        x_index = self.x[index]
        y_index = self.y[index]
        if self.transform is not None:
            x_index = self.transform(x_index)
        x_index_tensor = torch.from_numpy(x_index)
        y_index_tensor = torch.from_numpy(y_index)
        return (x_index_tensor,y_index_tensor)

    def __len__(self):
        return self.data_len

    def input_out_size(self):
        return [self.input_size,self.out_size]


