import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from .transform import transform_dict

class OMNIGLOT():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index], -1
    
    def __len__(self):
        return len(self.data)

def prepare_OMNIGLOT(root, train_batch=256, test_batch=256, num_workers=1, **kwargs):
    # train_data, test_data = OMNIGLOT(root, 'train'), OMNIGLOT(root, 'test')
    # data_wb = torchvision.datasets.Omniglot('../_datasets/', background=True, download=True, transform=transform_dict['OMNIGLOT'])
    # data_nb = torchvision.datasets.Omniglot('../_datasets/', background=False, download=True, transform=transform_dict['OMNIGLOT'])
    # cds = torch.utils.data.ConcatDataset([data_wb, data_nb])
    # train_data, test_data = torch.utils.data.random_split(cds, [19280+13180-8070, 8070])
    # 
    path_train = os.path.join(root, 'omg_static', 'omg_train')
    path_test = os.path.join(root, 'omg_static', 'omg_test')
    train_data = torch.load(path_train)
    test_data = torch.load(path_test)
    train_data, test_data = OMNIGLOT(train_data), OMNIGLOT(test_data)

    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader