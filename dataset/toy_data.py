import os
import torch
import numpy as np
import torch.distributions as dist

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GaussianMixture(Dataset):
    def __init__(self, N, d, ratio):
        m1 = torch.ones(d, )
        v1 = 0.1 * torch.ones(d, )
        m2 = -torch.ones(d, )
        v2 = 0.1 * torch.ones(d, )
        a = dist.Normal(m1, v1)
        adata = a.sample((int(N * ratio),))
        alabel = 1*torch.ones((int(N * ratio),))
        b = dist.Normal(m2, v2)
        bdata = b.sample((int(N * (1-ratio)),))
        blabel = 0*torch.ones((int(N * (1-ratio)),))
        self.data = torch.concat((adata, bdata))
        self.y = torch.concat((alabel, blabel))
        self.s = self.y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        return datum, (self.y[index], self.s[index])


class GaussianMixture2(Dataset):
    def __init__(self, N, C, d):
        super().__init__()
        mix = dist.Categorical(torch.ones(C,))
        modes = dist.Normal(5*torch.randn(C, d), torch.ones((C, d)))
        comp = dist.Independent(modes, 1)
        gmm = dist.mixture_same_family.MixtureSameFamily(mix, comp)
        samples = gmm.sample((N, ))
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        return datum, (-1, -1)


class Bernoulli():
    def __init__(self, N):
        x = np.array([np.sin(2 * np.pi / 16 * i)  for i in range(256)])
        p = (x + 1) / 2
        d = dist.Bernoulli(torch.tensor(p))
        data = d.sample((N,)).float()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        return datum, -1

class Toy():
    def __init__(self, root='../_datasets/' , split='train'):
        data = torch.load(os.path.join(root, 'toy', 'toy_data'))
        if split == 'train':
            self.data = data['train']
        else:
            self.data = data['test']
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        return datum, -1


class Pinwheel(Dataset):
    def __init__(self, num_classes, num_per_class):
        super().__init__()
        data, label = Pinwheel.make_pinwheel_data(0.3, 0.05, num_classes, num_per_class, 0.25)
        self.data = data.astype(np.float32)
        self.labels = label.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        label = self.labels[index]
        return datum, (label, -1)

    @staticmethod
    def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
        # code from Johnson et. al. (2016)
        rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)
        np.random.seed(1)
        features = np.random.randn(num_classes*num_per_class, 2) * np.array([radial_std, tangential_std])
        features[:,0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)
        angles = rads[labels] + rate * np.exp(features[:,0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        feats = 10 * np.einsum('ti,tij->tj', features, rotations)
        data = np.random.permutation(np.hstack([feats, labels[:, None]]))
        return data[:, 0:2], data[:, 2].astype(np.int)


def prepare_Pinwheel(train_batch=256, test_batch=256, num_workers=1, **kwargs):
    train_data, test_data = Pinwheel(5, 200), Pinwheel(5, 200)
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader

def prepare_toy(train_batch=256, test_batch=256, num_workers=1, **kwargs):
    
    train_data, test_data = Toy('../_datasets', 'train'), Toy('../_datasets', 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader