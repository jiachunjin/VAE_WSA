import os
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import VAE
from utils import dataloader_one_batch

class VAE_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
    
    def _build_model(self):
        self.vae = VAE(self.config)

    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer = Adam(
            [
                {'params': self.vae.encoder.parameters()},
                {'params': self.vae.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )

    def train(self):
        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch(epoch)
            self.num_epoch += 1
            if epoch == 1 or epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()
            if epoch % self.config['trainer_config']['save_every'] == 0:
                pass
                # self.save()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.vae.loss_binary_data(x)
            loss.backward()
            self.optimizer.step()
            self.num_grads += 1

    def evaluate(self):
        z = torch.randn((64, self.config['vae_config']['d']), device=self.device)
        p = self.vae.decoder(z)
        
        # samples = self.vae.decoder(z).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        samples = (torch.rand_like(p) < p).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)

        self.writer.add_images('generation', samples, self.num_epoch)

        # N = 16
        # x, _ = next(iter(self.train_loader))
        # x = x[:N].to(self.device)

        # mu, log_var = self.vae.encoder(x)
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # z = eps * std + mu
        # recons = self.vae.decoder(z).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        # origin = x.mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        # self.writer.add_images('rec', torch.cat((origin, recons), dim=0), self.num_epoch)

        with torch.no_grad():
            x_train, _ = dataloader_one_batch(self.train_loader)
            x_train = x_train.to(self.device)

            x_test, _ = dataloader_one_batch(self.test_loader)
            x_test = x_test.to(self.device)
            self.writer.add_scalars('BPD', {
                'train': self.vae.loss_binary_data(x_train).item(),
                'test': self.vae.loss_binary_data(x_test).item(),
            }, self.num_epoch)

        #     # mu_test, log_var_test = self.vae.encoder(x_test)
        #     # plt.scatter(mu_test[:, 0].detach().cpu(), mu_test[:, 1].detach().cpu(), s=2, c=y_test.cpu(), cmap='tab20')
        #     # self.writer.add_figure('posterior_test', plt.gcf(), self.num_epoch)
            self.writer.close()

    def save(self):
        save_path = os.path.join(self.ckpt_dir, f'ckpt_{self.num_epoch}.pt')
        self.vae.save(save_path)
