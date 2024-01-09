import os
import torch
from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import VAE
from utils import dataloader_one_batch, requires_grad

class RWS_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)

    def _build_model(self):
        self.vae = VAE(self.config)

    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer = Adam(
            [
                {'params': self.vae.encoder.parameters()},
                # {'params': self.vae.decoder.parameters()}, # In RWS, train encoder only
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
    
    def optimal_inference(self, pretrained_VAE_path):
        self.vae.load(pretrained_VAE_path)
        requires_grad(self.vae.decoder.parameters(), False)

        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.optimal_inference_train_epoch(epoch)
            self.num_epoch += 1
        self.evaluate()

    def optimal_inference_train_epoch(self, epoch):
        pbar = tqdm(self.test_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            losses = self.vae.loss(x)
            losses['total'].backward()
            self.optimizer.step()
            self.num_grads += 1
    
    def train(self, pretrained_VAE_path):
        self.vae.load(pretrained_VAE_path)
        requires_grad(self.vae.decoder.parameters(), False)

        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch(epoch)
            self.num_epoch += 1
        self.evaluate()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            mu, log_var = self.vae.encoder(x)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu
            recons = self.vae.decoder(z)
            x = torch.cat((x, recons), dim=0)
            self.optimizer.zero_grad()
            losses = self.vae.loss(x)
            losses['total'].backward()
            self.optimizer.step()
            self.num_grads += 1

    def train_sample_prior(self, pretrained_VAE_path):
        # so we do not use the reconstructed samples, but directly sample from model
        self.vae.load(pretrained_VAE_path)
        requires_grad(self.vae.decoder.parameters(), False)

        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch_sample_prior(epoch)
            self.num_epoch += 1
        self.evaluate()

    def train_epoch_sample_prior(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            samples = self.vae.decoder(z)
            x = torch.cat((x, samples), dim=0)
            # x = samples
            self.optimizer.zero_grad()
            losses = self.vae.loss(x)
            losses['total'].backward()
            self.optimizer.step()
            self.num_grads += 1
    
    def evaluate(self):
        with torch.no_grad():
            x_train, _ = dataloader_one_batch(self.train_loader)
            x_train = x_train.to(self.device)

            x_test, _ = dataloader_one_batch(self.test_loader)
            x_test = x_test.to(self.device)
            train_loss = round(self.vae.loss(x_train)['total'].item(), 3)
            test_loss = round(self.vae.loss(x_test)['total'].item(), 3)
            print(f'train: {train_loss}, test: {test_loss}')

    def save(self):
        pass