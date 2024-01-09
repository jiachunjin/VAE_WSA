import copy
import torch

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from .transform import transform_dict

color_dict = {
    '0': torch.tensor([0.9, 0.5, 0.1])[None, :, None, None],
    '1': torch.tensor([0.9, 0.3, 0.5])[None, :, None, None],
    '2': torch.tensor([0.1, 0.9, 0.5])[None, :, None, None],
    '3': torch.tensor([0.7, 0.1, 0.9])[None, :, None, None],
    '4': torch.tensor([0.1, 0.7, 0.8])[None, :, None, None],
    '5': torch.tensor([0.9, 0.1, 0.1])[None, :, None, None],
    '6': torch.tensor([0.5, 0.9, 0.2])[None, :, None, None],
    '7': torch.tensor([0.8, 0.8, 0.8])[None, :, None, None],
    '8': torch.tensor([0.1, 0.1, 0.8])[None, :, None, None],
    '9': torch.tensor([0.9, 0.9, 0.3])[None, :, None, None],
}

def colorize(x, c):
    """
    colorize an input tensor

    Args:
        x: input tensor with shape (B, C, W, H)
        c: index of color

    Returns:
        colored tensor
    """
    return color_dict[str(c)] * x

def change_color(x, c):
    result = torch.zeros_like(x)
    for i in range(len(x)):
        x_i = (1 / color_dict[str(c[i])]) * (x[i])
        new_c = torch.randint(0, 10, (1,)).item()
        result[i] = (color_dict[str(new_c)] * x_i)
    return result

class coloredMNIST_S10():
    def __init__(self, root, split='train', ratio=0.9):
        data = MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['coloredMNIST_10'])
        if split == 'train':
            y_list = []
            s_list = []
            digit_list = []
            for i in range(10):
                digit =  data.data[data.targets==i].unsqueeze(dim=1) / 255.
                len_short = int(ratio * len(digit))
                len_noshort = len(digit) - len_short
                digit_list.append(colorize(digit[:len_short], i))
                s_list.append(torch.ones(len_short))
                y_list.append(i*torch.ones(len(digit)))
                for j in range(len_noshort):
                    c = torch.randint(0, 10, (1,)).item()
                    while c == i:
                        c = torch.randint(0, 10, (1,)).item()
                    digit_list.append(colorize(digit[len_short+j], c))
                s_list.append(torch.zeros(len_noshort))
            self.y = torch.cat(y_list, dim=0)
            self.X = torch.cat(digit_list, dim=0)
            self.s = torch.cat(s_list, dim=0)
        else:
            self.y = data.targets
            self.X = torch.zeros((len(self.y), 3, 28, 28))
            self.s = torch.zeros_like(self.y)
            for j in range(len(self.y)):
                c = torch.randint(0, 10, (1,)).item()
                self.X[j] = colorize(data[j][0], c)

    def __getitem__(self, index):
        return self.X[index], (self.y[index], self.s[index])

    def __len__(self):
        return len(self.y)


class coloredMNIST_S2():
    """
    Binary dataset with shortcuts
    """
    def __init__(self, root, split='train', ratio=0.9):
        data = MNIST(root, train=(split == 'train'), download=True, transform=transform_dict['coloredMNIST_2'])
        pos_label = 6
        neg_label = 9

        pos = copy.deepcopy(data)
        neg = copy.deepcopy(data)
        pos.data = pos.data[pos.targets==pos_label]
        neg.data = neg.data[neg.targets==neg_label]
        len_pos, len_neg = len(pos), len(neg)
        self.X = torch.zeros((len_neg+len_pos, 3, 28, 28))
        self.y = torch.cat((torch.zeros(len_neg, dtype=torch.long), torch.ones(len_pos, dtype=torch.long)))
        self.s = torch.ones_like(self.y)
        index = 0

        if split == 'train':
            for i in range(len_neg):
                self.X[index][0] = neg[i][0]
                if i > int(ratio * len_neg):
                    self.s[index] = 0
                    self.X[index][1] = neg[i][0]
                    self.X[index][2] = neg[i][0]
                index += 1
            for i in range(len_pos):
                self.X[index][2] = pos[i][0]
                if i > int(ratio * len_pos):
                    self.s[index] = 0
                    self.X[index][0] = pos[i][0]
                    self.X[index][1] = pos[i][0]
                index += 1
        else:
            self.s = torch.zeros_like(self.y)
            for i in range(len_neg):
                self.X[index][0] = neg[i][0]
                self.X[index][1] = neg[i][0]
                self.X[index][2] = neg[i][0]
                index += 1
            for i in range(len_pos):
                self.X[index][0] = pos[i][0]
                self.X[index][1] = pos[i][0]
                self.X[index][2] = pos[i][0]
                index += 1

    def __getitem__(self, index):
        return self.X[index], (self.y[index], self.s[index])

    def __len__(self):
        return len(self.y)

def prepare_coloredMNIST(root, train_batch=256, test_batch=256, num_workers=1, shortcut_ratio=0.9, **kwargs):
    if kwargs['num_class'] == 2:
        train_data, test_data = coloredMNIST_S2(root, 'train', shortcut_ratio), coloredMNIST_S2(root, 'test', shortcut_ratio)
    elif kwargs['num_class'] == 10:
        train_data, test_data = coloredMNIST_S10(root, 'train', shortcut_ratio), coloredMNIST_S10(root, 'test', shortcut_ratio)
    train_loader = DataLoader(train_data, train_batch, True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, True, num_workers=num_workers)

    return train_loader, test_loader
