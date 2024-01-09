import os
import torch
from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import Threeinone
from utils import dataloader_one_batch, requires_grad
from dataset import prepare_another_small_MNIST

class Post_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)

    def _build_model(self):
        self.model_1 = Threeinone(self.config)
        self.model_2 = Threeinone(self.config)
        self.model_3 = Threeinone(self.config)

    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer_1 = Adam(
            [
                {'params': self.model_1.vae.encoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
        self.optimizer_2 = Adam(
            [
                {'params': self.model_2.vae.encoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
        self.optimizer_3 = Adam(
            [
                {'params': self.model_3.vae.encoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )

    def train(self):
        for i in range(400, 10000, 100):
            pretrained_model_path = 'ckpts/3in1/ckpt_' + str(i) + '.pt'
            # pretrained_model_path = 'ckpts/3in1/ckpt_8300.pt'
            self.model_1.load(pretrained_model_path)
            self.model_2.load(pretrained_model_path)
            self.model_3.load(pretrained_model_path)
            requires_grad(self.model_1.vae.decoder.parameters(), False)
            requires_grad(self.model_1.energy.parameters(), False)
            requires_grad(self.model_2.vae.decoder.parameters(), False)
            requires_grad(self.model_2.energy.parameters(), False)
            requires_grad(self.model_3.vae.decoder.parameters(), False)
            requires_grad(self.model_3.energy.parameters(), False)

            num_epoch = int(self.config['trainer_config']['num_epoch'])
            self.another_data_loader = prepare_another_small_MNIST(self.config['dataset_config']['root'], self.config['dataset_config']['train_batch'])
            for epoch in trange(1, num_epoch+1):
                self.rws_train_epoch(epoch)
                self.post_train_epoch(epoch)
                self.optimal_inference_train_epoch(epoch)
                self.num_epoch += 1
            # for epoch in trange(1, 200+1):
            #     self.optimal_inference_train_epoch(epoch)
            self.evaluate(i)

    def evaluate(self, i):
        with torch.no_grad():
            x_test, _ = dataloader_one_batch(self.test_loader)
            x_test = x_test.to(self.device)

            self.writer.add_scalars('BPD', {
                'rws': self.model_1.loss_vae(x_test).item(),
                'ebm': self.model_2.loss_vae(x_test).item(),
                'opt': self.model_3.loss_vae(x_test).item(),
            }, i)
            self.writer.close()

    def rws_train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            samples = self.model_1.vae.decoder(z)
            x = torch.cat((x, samples), dim=0)
            self.optimizer_1.zero_grad()
            loss = self.model_1.loss_vae(x)
            loss.backward()
            self.optimizer_1.step()

    def post_train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        num_steps = self.config['sampler_config']['num_steps_test']
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            samples_ebm = self.model_2.sample_from_model(len(x), num_steps)
            x = torch.cat((x, samples_ebm), dim=0)
            self.optimizer_2.zero_grad()
            loss = self.model_2.loss_vae(x)
            loss.backward()
            self.optimizer_2.step()

    def optimal_inference_train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        another_data = iter(self.another_data_loader)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.optimizer_3.zero_grad()
            # x_ = x + 0.03*torch.randn_like(x, device=self.device)
            x_, _ = next(another_data)
            x_ = x_.to(self.device)
            x = torch.cat((x, x_), dim=0)
            loss = self.model_3.loss_vae(x)
            loss.backward()
            self.optimizer_3.step()
            self.num_grads += 1
        # pbar = tqdm(self.test_loader, leave=False)
        # for _, (x, _) in enumerate(pbar):
        #     x = x.to(self.device)
        #     self.optimizer_3.zero_grad()
        #     loss = self.model_3.loss_vae(x)
        #     loss.backward()
        #     self.optimizer_3.step()
        #     self.num_grads += 1