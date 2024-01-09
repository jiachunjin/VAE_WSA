import torch
from torchvision import transforms

transform_dict = {
    'MNIST': transforms.Compose([
        transforms.ToTensor()
    ]),
    'OMNIGLOT': transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
    ]),
    'CIFAR10': transforms.Compose([
        transforms.ToTensor()
    ]),
    'coloredMNIST_2': transforms.Compose([
        transforms.ToTensor()
    ]),
    'coloredMNIST_10': transforms.Compose([
        transforms.ToTensor()
    ]),
}