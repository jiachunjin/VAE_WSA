import torchvision
from torch.utils.data import DataLoader

from .transform import transform_dict

class CIFAR10():
    def __init__(self, root, split='train'):
        if split == 'train':
            self.data = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_dict['MNIST'])
        elif split == 'test':
            self.data = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_dict['MNIST'])
    
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


def prepare_CIFAR10(root, train_batch=256, test_batch=256, num_workers=1, **kwargs):
    train_data, test_data = CIFAR10(root, 'train'), CIFAR10(root, 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader