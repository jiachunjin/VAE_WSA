import os
import time
import random
import yaml
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as T
from matplotlib.patches import Ellipse

color_dict = {
    'blue': '#0007ff',
    'bluegreen': '#00ffd3',
    'fenred': '#ff45d8',
    'fenblue': '#8cadfb',
    'fengreen': '#96fbaf'
}

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def dataloader_one_batch(dataset):
    x_list = []
    y_list = []
    for b, (x, y) in enumerate(dataset):
        x_list.append(x)
        y_list.append(y)
    return torch.cat(x_list), torch.cat(y_list)

def load_config(path):
    with open(path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def cnn_output_shape(width, kernel, padding, stride):
    return int(np.floor((width - kernel + 2 * padding) / stride) + 1)

def create_exp_name(comments=''):
    x = datetime.datetime.now()
    name = x.strftime("%y%m%d") + '-' + x.strftime("%X")
    if len(comments) > 0:
        name = comments + '_' + name
    return name

def setup_experiment(args):
    try:
        name = create_exp_name(args.exp_name)
        exp_dir = os.path.join('experiments', name)
        log_dir = os.path.join(exp_dir, 'logs')
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        sample_dir = os.path.join(exp_dir, 'samples')
        os.makedirs(exp_dir)
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(sample_dir)
        print(f'\nlog_dir: {log_dir}\n')
    except FileExistsError:
        t = random.randrange(1,30)
        time.sleep(t)
        name = create_exp_name(args.exp_name)
        exp_dir = os.path.join('experiments', name)
        log_dir = os.path.join(exp_dir, 'logs')
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        sample_dir = os.path.join(exp_dir, 'samples')
        os.makedirs(exp_dir)
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(sample_dir)
        print(f'\nlog_dir: {log_dir}\n')
    return exp_dir, log_dir, ckpt_dir, sample_dir

def plot_tensors(X, num_row, num_col, mode):
    """
    Use matplotlib to plot a batch of images stored in a tensor

    Args:
        X: tensor with shape (B, C, H, W)
        num_row: number of rows in the grid
        num_col: number of columns in the grid
        mode: string, 'grey' or 'rgb'        
    """
    assert mode in ['grey', 'rgb']
    N = X.shape[0]
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 2, num_row * 2))
    for ax in axes.ravel():
        ax.set_axis_off()
    if mode == 'rgb':
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(T.ToPILImage()(X[i]))
            else:
                axes[i//num_col][i%num_col].imshow(T.ToPILImage()(X[i]))
    elif mode == 'grey':
        X.squeeze_()
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(X[i, ...], cmap='Greys')
            else:
                axes[i//num_col][i%num_col].imshow(X[i, ...], cmap='Greys')
    plt.show()
    plt.close(fig)

def kl_divergence_gaussians(mu_1, mu_2, log_var_1, log_var_2):
    return torch.mean(-0.5 * torch.sum(1 + log_var_1 - log_var_2 - (mu_2 - mu_1) ** 2 / log_var_2.exp() - log_var_1.exp() / log_var_2.exp(), dim = 1), dim = 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_img_tensors(X, num_row, num_col, mode, save_path):
    assert mode in ['grey', 'rgb']
    N = X.shape[0]
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 2, num_row * 2))
    for ax in axes.ravel():
        ax.set_axis_off()
    if mode == 'rgb':
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(T.ToPILImage()(X[i]))
            else:
                axes[i//num_col][i%num_col].imshow(T.ToPILImage()(X[i]))
    elif mode == 'grey':
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(X[i, 0, ::], cmap='Greys')
            else:
                axes[i//num_col][i%num_col].imshow(X[i, 0, ::], cmap='Greys')
    plt.savefig(save_path)
    plt.close(fig)


# plot posterior

seethrough = plt.cm.colors.ListedColormap([(0,0,0,0), (0,0,0,1)])

def add_im(axs, im, xy, h=0.3, **kwargs):
    x, y = xy
    axs.imshow(im, extent = (x - h/2, x+h/2, y-h/2, y+h/2),
             cmap=seethrough, zorder=2, **kwargs)

def ellipse_coords(cov):
    """Given a Covariance matrix, return the Ellipse coordinates."""
    u, v = np.linalg.eigh(cov)
    width = np.sqrt(5.991*u[1])
    height = np.sqrt(5.991*u[0])
    angle = np.arctan2(v[1][1], v[1][0])
    return width, height, angle

def add_ellipse(axs, mean, log_var, color, alpha=0.1):
    cov = torch.exp(log_var) * np.eye(2)
    width, height, angle = ellipse_coords(cov)
    ep = Ellipse(xy=(mean[0], mean[1]),
               width=width, height=height,
               angle=np.degrees(angle),
               color=color, alpha=alpha,
               fill=False, linewidth=2)
    axs.add_artist(ep)

def show_posterior(mu, log_var, y, num_class):
    fig, axs = plt.subplots(figsize=(5,5))
    colors = plt.cm.tab20b(np.linspace(0,1,num_class))
    for i in range(len(mu)):
        add_ellipse(axs, mu[i], log_var[i], colors[y[i]])

    # for i in range(1, xx.shape[0]):
    #     add_im(axs, xx[i], mu[i], h=0.3, alpha=0.5)

    edge = np.abs(mu).max() + 2*np.abs(0.5*torch.exp(log_var)).max()
    axs.set_xlim((-edge,edge))
    axs.set_ylim((-edge,edge));