import os
import scipy.io as scipyio
from torch.utils.data import DataLoader


class Silhouettes():
    def __init__(self, root, split='train'):
        data_path = os.path.join(root, 'caltech101_silhouettes', 'caltech101_silhouettes_28_split1.mat')
        data = scipyio.loadmat(data_path)
        if split == 'train':
            self.data = data['train_data'].astype('float32').reshape(-1, 1, 28, 28)
        elif split == 'test':
            self.data = data['test_data'].astype('float32').reshape(-1, 1, 28, 28)

    def __getitem__(self, index):
         return self.data[index], -1

    def __len__(self):
        return len(self.data)


def prepare_Silhouettes(root, train_batch=256, test_batch=256, num_workers=1, **kwargs):
    train_data, test_data = Silhouettes(root, 'train'), Silhouettes(root, 'test')
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader