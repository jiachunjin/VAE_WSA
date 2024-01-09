import torch
import torch.nn as nn
from utils import cnn_output_shape

def GCE_loss(pred, y, q):
    pred = nn.Softmax(dim=1)(pred)
    py = torch.gather(pred, 1, y.unsqueeze(1).long())
    return ((1 - py ** q) / q).mean()


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(self.size)


# class Classifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         hidden_dim = config['hidden_dim']
#         modules = []
#         dim_input = config['d']
#         for i in range(len(hidden_dim)):
#             modules.extend([
#                 nn.Linear(dim_input, hidden_dim[i]),
#                 nn.ReLU(),
#             ])
#             dim_input = hidden_dim[i]
#         modules.append(nn.Linear(dim_input, config['num_class']))
#         self.net = nn.Sequential(*modules)
    
#     def forward(self, x):
#         return self.net(x)

class Encoder_VEC_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        hidden_dim = config['hidden_dim']
        dropout_p = config['dropout_p']
        modules = []
        dim_input = 256
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
            dim_input = i
        modules.append(nn.Linear(dim_input, 2 * self.d))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        embedding = self.net(x)
        return embedding[..., :self.d], embedding[..., self.d:]

class Encoder_IMG_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        hidden_dim = config['hidden_dim']
        dropout_p = config['dropout_p']
        modules = [nn.Flatten()]
        dim_input = config['img_size'] * config['img_size'] * config['input_channel']
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
            dim_input = i
        modules.append(nn.Linear(dim_input, 2 * self.d))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        embedding = self.net(x)
        return embedding[..., :self.d], embedding[..., self.d:]


class Encoder_IMG_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        channels, kernels, strides, paddings = config['channels'], config['kernels'], config['strides'], config['paddings']
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        dropout_p = config['dropout_p']
        modules = []
        channel_in = config['input_channel']
        w = config['img_size']
        for i in range(len(channels)):
            modules.extend([
                nn.Conv2d(channel_in, channels[i], kernels[i], strides[i], paddings[i]),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
            channel_in = channels[i]
            w = cnn_output_shape(w, kernels[i], paddings[i], strides[i])
        modules.extend([
            nn.Flatten(),
            nn.Linear(channel_in * w * w, 2*self.d),
        ])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        embedding = self.net(x)
        return embedding[..., :self.d], embedding[..., self.d:]


class Decoder_IMG_CNN_32x3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        dropout_p = config['dropout_p']
        modules = [nn.Linear(self.d, 64*8*8),
                   View((-1, 64, 8, 8)),
                    nn.ConvTranspose2d(64,
                                       32,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.ConvTranspose2d(32,
                                       16,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Conv2d(16,
                              out_channels=3,
                              kernel_size=3,
                              padding=1),
                    nn.Sigmoid()
                   ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class Decoder_IMG_CNN_28_C1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        dropout_p = config['dropout_p']
        modules = [nn.Linear(self.d, 64*7*7),
                   View((-1, 64, 7, 7)),
                    nn.ConvTranspose2d(64,
                                       32,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.ConvTranspose2d(32,
                                       16,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Conv2d(16,
                              out_channels=1,
                              kernel_size=3,
                              padding=1),
                    nn.Sigmoid()
                   ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class Decoder_IMG_CNN_105_C1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        dropout_p = config['dropout_p']
        modules = [nn.Linear(self.d, 8*26*26),
                    View((-1, 8, 26, 26)),
                    nn.ConvTranspose2d(8,
                                        4,
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.ConvTranspose2d(4,
                                        2,
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Conv2d(2,
                                out_channels=1,
                                kernel_size=2,
                                padding=1),
                    nn.Sigmoid()
                    ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class Decoder_IMG_MLP_C1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        dropout_p = config['dropout_p']
        self.img_size = config['img_size']

        hidden_dim = config['hidden_dim']
        modules = []
        dim_input = self.d
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
            dim_input = i
        modules.extend([
            nn.Linear(dim_input, config['img_size'] * config['img_size'] * config['input_channel']),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).reshape(-1, 1, self.img_size, self.img_size)


class Decoder_VEC_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config['d']
        dropout_p = config['dropout_p']

        hidden_dim = config['hidden_dim']
        modules = []
        dim_input = self.d
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
            dim_input = i
        modules.extend([
            nn.Linear(dim_input, 256),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class Energy_IMG_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        modules = [nn.Flatten()]
        dim_input = config['img_size'] * config['img_size'] * config['input_channel']
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
            ])
            dim_input = i
        modules.append(nn.Linear(dim_input, 1))
        self.energy = nn.Sequential(*modules)

    def forward(self, x):
        return self.energy(x)

    def score(self, x):
        x = x.requires_grad_()
        logp = -self.energy(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]


class Binary_Classifier_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        modules = [nn.Flatten()]
        dim_input = 256
        for i in hidden_dim:
            modules.extend([
                nn.Linear(dim_input, i),
                nn.ReLU(),
            ])
            dim_input = i
        modules.append(nn.Linear(dim_input, 1))
        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.net(x)


class Binary_Classifier_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels, kernels, strides, paddings = config['channels'], config['kernels'], config['strides'], config['paddings']
        assert len(channels) == len(kernels) == len(strides) == len(paddings)

        modules = []
        channel_in = config['input_channel']
        w = config['img_size']
        for i in range(len(channels)):
            modules.extend([
                nn.Conv2d(channel_in, channels[i], kernels[i], strides[i], paddings[i]),
                nn.ReLU(),
            ])
            channel_in = channels[i]
            w = cnn_output_shape(w, kernels[i], paddings[i], strides[i])
        modules.extend([
            nn.Flatten(),
            nn.Linear(channel_in * w * w, 1),
        ])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)