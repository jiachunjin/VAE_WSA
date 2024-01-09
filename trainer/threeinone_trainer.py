import os
import torch

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import Threeinone
from utils import dataloader_one_batch, save_img_tensors


class Threeinone_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
        self.num_vae_grads = 0
        self.num_ebm_grads = 0
        self.num_vae_epoch = 0
        self.num_ebm_epoch = 0

    def _build_model(self):
        self.model = Threeinone(self.config)

    def _build_optimizer(self):
        vae_optimizer_config = self.config['optimizer_config']
        ebm_optimizer_config = self.config['ebm_optimizer_config']
        self.vae_optimizer = Adam(
            [
                {'params': self.model.vae.encoder.parameters()},
                {'params': self.model.vae.decoder.parameters()},
            ],
            lr = float(vae_optimizer_config['lr']),
        )
        self.ebm_optimizer = Adam(
            [
                {'params': self.model.energy.parameters()},
            ],
            lr = float(ebm_optimizer_config['lr']),
        )

    def train(self):
        num_epoch = int(self.config['trainer_config']['num_epoch'])
        num_vae_epoch = 10
        # num_ebm_epoch = int(self.config['trainer_config']['num_ebm_epoch'])
        for epoch in trange(1, num_epoch+1):
            if epoch <= num_vae_epoch:
                self.train_vae_epoch(epoch)
                self.num_vae_epoch += 1
            else:
                self.train_vae_epoch(epoch)
                self.num_vae_epoch += 1
                self.train_ebm_epoch(epoch)
                self.num_ebm_epoch += 1
            # else:
            #     print('Start to train with simulated data.')
            #     self.train_vae_with_ebm_epoch(epoch)
            #     self.num_vae_epoch += 1
            #     self.train_ebm_epoch(epoch)
            #     self.num_ebm_epoch += 1
            self.num_epoch += 1
            if epoch == 1 or epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()
            if epoch % self.config['trainer_config']['save_every'] == 0:
                self.save()

    def train_vae_with_ebm_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.vae_optimizer.zero_grad()
            loss = self.model.loss_vae_with_ebm(x)
            loss.backward()
            self.vae_optimizer.step()
            self.num_vae_grads += 1

    def train_vae_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.vae_optimizer.zero_grad()
            loss = self.model.loss_vae(x)
            loss.backward()
            self.vae_optimizer.step()
            self.num_vae_grads += 1

    def train_both_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.vae_optimizer.zero_grad()
            self.ebm_optimizer.zero_grad()
            loss_vae = self.model.loss_vae(x)
            loss_vae.backward()
            self.vae_optimizer.step()
            self.num_vae_grads += 1

            loss_ebm = self.model.loss_ebm(x)
            loss_ebm.backward()
            self.ebm_optimizer.step()
            self.num_ebm_grads += 1

    def train_ebm_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.ebm_optimizer.zero_grad()
            loss_ebm = self.model.loss_ebm(x)
            loss_ebm.backward()
            self.ebm_optimizer.step()
            self.num_ebm_grads += 1


    def evaluate(self):
        # sample from VAE
        z = torch.randn((64, self.config['vae_config']['d']), device=self.device)
        p = self.model.vae.decoder(z)
        samples = (torch.rand_like(p) < p).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        self.writer.add_images('vae samples', samples, self.num_vae_epoch)
        # save_path_vae = os.path.join(self.sample_dir, f'vae_{self.num_vae_epoch}.png')
        # save_img_tensors(samples_vae, 8, 8, 'grey', save_path_vae)

        # sample from EBM
        num_steps = self.config['sampler_config']['num_steps_test']
        p_ebm = self.model.conditional_sample(num_steps, z)
        samples_ebm = (torch.rand_like(p_ebm) < p_ebm).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        self.writer.add_images('ebm samples', samples_ebm, self.num_ebm_epoch)
        # save_path_ebm = os.path.join(self.sample_dir, f'ebm_{self.num_ebm_epoch}.png')
        # save_img_tensors(samples_ebm, 8, 8, 'grey', save_path_ebm)

        # draw test BPD
        with torch.no_grad():
            # x_train, _ = dataloader_one_batch(self.train_loader)
            # x_train = x_train.to(self.device)

            # x_test, _ = dataloader_one_batch(self.test_loader)
            # x_test = x_test.to(self.device)
            # self.writer.add_scalars('BPD', {
            #     'train': self.model.vae.loss(x_train)['total'].item(),
            #     'test': self.model.vae.loss(x_test)['total'].item(),
            # }, self.num_vae_epoch)
            self.writer.close()

    def save(self):
        save_path = os.path.join(self.ckpt_dir, f'ckpt_{self.num_epoch}.pt')
        self.model.save(save_path)