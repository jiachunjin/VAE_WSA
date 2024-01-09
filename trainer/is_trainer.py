import os
import torch

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import VAE, Binary_Classifier_MLP
from utils import dataloader_one_batch, save_img_tensors, requires_grad


class IS_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
    
    def _build_model(self):
        self.vae = VAE(self.config)
        self.classifier = Binary_Classifier_MLP(self.config['classifier_config']).to(self.device)
    
    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer1 = Adam(
            [
                {'params': self.classifier.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )

        self.optimizer2 = Adam(
            [
                {'params': self.vae.encoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )

    def _load_VAE(self, load_path):
        self.vae.load(load_path)
        requires_grad(self.vae.encoder.parameters(), False)
        requires_grad(self.vae.decoder.parameters(), False)

    def train(self):
        load_path = 'experiments/231013-15:50:33_vae_small_MLP/checkpoints/ckpt_' + str(1000) + '.pt'
        self._load_VAE(load_path)
        # train DRE
        for i in range(200):
            self.train_DRE_epoch(i)

        # train encoder
        requires_grad(self.vae.encoder.parameters(), True)
        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch(epoch)
            self.num_epoch += 1
            if epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            samples = self.vae.decoder(z)
            score = torch.nn.Sigmoid()(self.classifier(samples))
            weight = torch.logit(score, eps=1e-3).exp().detach()
            self.optimizer2.zero_grad()
            loss1 = self.vae.loss(x)
            loss2 = self.vae.loss(samples, weight)
            loss = loss1 + loss2
            self.writer.add_scalar('loss', loss.item(), self.num_grads)
            loss.backward()
            self.optimizer2.step()
            self.num_grads += 1


    def train_DRE_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            x_ = self.vae.decoder(z)
            labels = torch.cat((torch.ones(len(x), device=self.device), torch.zeros(len(x), device=self.device)), dim=0)
            labels = labels.float()
            x = torch.cat((x, x_), dim=0)
            self.optimizer1.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(self.classifier(x).squeeze(), labels)
            loss.backward()
            self.optimizer1.step()
            

    def evaluate(self):
        with torch.no_grad():
            x_train, _ = dataloader_one_batch(self.train_loader)
            x_train = x_train.to(self.device)

            x_test, _ = dataloader_one_batch(self.test_loader)
            x_test = x_test.to(self.device)
            self.writer.add_scalars('BPD', {
                'train': self.vae.loss(x_train).item(),
                'test': self.vae.loss(x_test).item(),
            }, self.num_epoch)
            self.writer.close()