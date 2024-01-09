import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from .transform import transform_dict

class MNIST():
    def __init__(self, root, split='train'):
        self.data = torchvision.datasets.MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['MNIST'])

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return len(self.data)

class MNIST_binary():
    def __init__(self, root, split='train', num_train_samples=50000):
        # self.data = torchvision.datasets.MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['MNIST'])
        with open(os.path.join(root, 'binary_MNIST/', f'binarized_mnist_{split}.amat')) as f:
            lines = f.readlines()
        if split == 'train':
            self.data = self._lines_to_np_array(lines).astype('float32').reshape(-1, 1, 28, 28)[:num_train_samples]
        else:
            self.data = self._lines_to_np_array(lines).astype('float32').reshape(-1, 1, 28, 28)
        
    
    def __getitem__(self, index):
        return self.data[index], -1
    
    def __len__(self):
        return len(self.data)

    def _lines_to_np_array(self, lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

class MNIST_small_train():
    def __init__(self, root, split='train'):
        self.data = torchvision.datasets.MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['MNIST'])
        # self.data = self.data[:10000]

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return 10000

class MNIST_small_train_another():
    def __init__(self, root, split='train'):
        self.data = torchvision.datasets.MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['MNIST'])
        # self.data = self.data[:10000]

    def __getitem__(self, index):
        return self.data[index+10000][0], self.data[index+10000][1]
    
    def __len__(self):
        return 10000

def prepare_Binary_MNIST(root, train_batch=256, test_batch=256, num_workers=1, num_samples=50000, **kwargs):
    train_data, test_data = MNIST_binary(root, 'train', num_samples), MNIST_binary(root, 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader

def prepare_MNIST(root, train_batch=256, test_batch=256, num_workers=1, **kwargs):
    train_data, test_data = MNIST(root, 'train'), MNIST(root, 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader

def prepare_another_small_MNIST(root, train_batch):
    train_data = MNIST_small_train_another(root, 'train')
    train_loader = DataLoader(train_data, train_batch, True)

    return train_loader

def prepare_small_MNIST(root, train_batch=256, test_batch=256, num_workers=1, **kwargs):
    train_data, test_data = MNIST_small_train(root, 'train'), MNIST(root, 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader
