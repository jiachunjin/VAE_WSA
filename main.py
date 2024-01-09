import os
import torch
import shutil
import argparse
import pprint

from utils import load_config, setup_experiment
from trainer import All_Trainer

def train_All(args, mode, exp_dir, log_dir, ckpt_dir, sample_dir):
    if args.data == 'MNIST' or args.data == 'smallMNIST':
        config_path = os.path.join('config', 'continuous_28.yaml')
    elif args.data == 'toy':
        config_path = os.path.join('config', 'toy.yaml')
    elif args.data == 'CIFAR10':
        config_path = os.path.join('config', 'continuous_32.yaml')
    else:
        config_path = os.path.join('config', 'binary.yaml')
    config = load_config(config_path)
    shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))
    config['dataset_config']['data'] = args.data
    config['load_path'] = args.load_path
    # config['dataset_config']['num_samples'] = args.num_samples

    trainer = All_Trainer(mode, config, log_dir, ckpt_dir, sample_dir)
    trainer.train()


def run(args):
    torch.manual_seed(args.seed)
    exp_dir, log_dir, ckpt_dir, sample_dir = setup_experiment(args)
    mode = {'name': args.mode}
    mode['tensorboard'] = eval(args.tensorboard)
    if args.mode in ['wsa']:
        mode['use_weight'] = eval(args.use_weight)
    elif args.mode in ['augmentation', 'two_augmentation']:
        mode['aug_type'] = args.aug_type
    elif args.mode == 'air':
        mode['air_sigma'] = args.air_sigma
    elif args.mode == 'dropout':
        mode['dropout_p'] = args.dropout_p
    # mode = {
    #     'name': args.mode,
    #     'aug_type': args.aug_type,
    #     'air_sigma': args.air_sigma,
    #     'dropout_p': args.dropout_p,
    #     'use_weight': eval(args.use_weight), # why passing bool does not work?
    # }
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(mode)
    train_All(args, mode, exp_dir, log_dir, ckpt_dir, sample_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--data', type=str, choices=['Binary_MNIST', 'OMNIGLOT', 'Silhouettes', 'MNIST', 'smallMNIST', 'CIFAR10', 'toy'],required=True)
    parser.add_argument('--mode', type=str, choices=['vanilla', 'wsa', 'augmentation', 'two_augmentation', 'rws', 'air', 'dropout'],required=True)
    parser.add_argument('--aug_type', type=str, default='RandomRotation', choices=['RandomRotation', 'RandomHorizontalFlip'])
    parser.add_argument('--air_sigma', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--use_weight', type=str, choices=['True', 'False'], default='True')
    parser.add_argument('--comments', type=str, default='No comments')
    parser.add_argument('--tensorboard', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--load_path', type=str, default='None')
    args = parser.parse_args()
    run(args)


# def train_Tem(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'is.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = Tem_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_IS(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'is.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = IS_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_Post(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'post.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = Post_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_Threeinone(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'is.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = Threeinone_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_VAEBM(exp_dir, log_dir, ckpt_dir, sample_dir):
#     vae_config_path = os.path.join('config', 'vae.yaml')
#     vae_config = load_config(vae_config_path)
#     ebm_config_path = os.path.join('config', 'vaebm.yaml')
#     ebm_config = load_config(ebm_config_path)

#     shutil.copyfile(ebm_config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = VAEBM_Trainer(ebm_config, vae_config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_VAE(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'is.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     trainer = VAE_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

# def train_RWS(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'rws.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))
#     for i in range(200, 7000, 200):
#         print(f'============ {i} ============')
#         trainer = RWS_Trainer(config, log_dir, ckpt_dir, sample_dir)
#         path = 'experiments/231013-15:50:33_vae_small_MLP/checkpoints/ckpt_' + str(i) + '.pt'
#         trainer.train(path)

# def train_RWS_sample_prior(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'rws.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))
#     for i in range(3400, 7000, 200):
#         print(f'============ {i} ============')
#         trainer = RWS_Trainer(config, log_dir, ckpt_dir, sample_dir)
#         path = 'experiments/231013-15:50:33_vae_small_MLP/checkpoints/ckpt_' + str(i) + '.pt'
#         trainer.train_sample_prior(path)

# def optimal_inference(exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'rws.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))
#     for i in range(5400, 7000, 200):
#         print(f'============ {i} ============')
#         trainer = RWS_Trainer(config, log_dir, ckpt_dir, sample_dir)
#         path = 'experiments/231013-15:50:33_vae_small_MLP/checkpoints/ckpt_' + str(i) + '.pt'
#         trainer.optimal_inference(path)

# def train_Comparison(args, exp_dir, log_dir, ckpt_dir, sample_dir):
#     config_path = os.path.join('config', 'is.yaml')
#     config = load_config(config_path)
#     shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

#     config['dataset_config']['data'] = args.data

#     trainer = Comparison_Trainer(config, log_dir, ckpt_dir, sample_dir)
#     trainer.train()

