import os
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange
from torch.nn import functional as F

from .base_trainer import Base_Trainer
from model import VAE, Binary_Classifier_MLP, Decoder_IMG_MLP_C1, Encoder_IMG_MLP, Decoder_IMG_CNN_28_C1
from utils import dataloader_one_batch


class Tem_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
        x_train, _ = dataloader_one_batch(self.train_loader)
        self.x_train = x_train.to(self.device)
        x_test, _ = dataloader_one_batch(self.test_loader)
        self.x_test = x_test.to(self.device)
    
    def _build_model(self):
        # self.vae = VAE(self.config)
        # load_path = 'ckpts/vae_small_mnist/ckpt_1000.pt'
        # self.vae.load(load_path)
        # self.encoder = Encoder_IMG_MLP(self.config['encoder_config']).to(self.device)
        self.vae1 = VAE(self.config)
        self.vae2 = VAE(self.config)
        self.G = Decoder_IMG_CNN_28_C1(self.config['decoder_config']).to(self.device)
        self.D = Binary_Classifier_MLP(self.config['classifier_config']).to(self.device)

    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer_G = Adam(
            [
                {'params': self.G.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
        self.optimizer_D = Adam(
            [
                {'params': self.D.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
        self.optimizer_VAE1 = Adam(
            [
                {'params': self.vae1.encoder.parameters()},
                {'params': self.vae1.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
            betas = eval(optimizer_config['adam_betas']),
            eps = float(optimizer_config['adam_eps']),
            weight_decay= float(optimizer_config['weight_decay']),
        )
        self.optimizer_VAE2 = Adam(
            [
                {'params': self.vae2.encoder.parameters()},
                {'params': self.vae2.decoder.parameters()},
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
            if epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()
            # if epoch % self.config['trainer_config']['save_every'] == 0:
            #     self.save()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        # for _, (x, _) in enumerate(pbar):
        #     x = x.to(self.device)
            # self.train_GAN_step(x)
        if epoch < 100:
            for _, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                self.train_VAE1_step(x)
                self.train_VAE2_step(x)
                self.train_D_step(x)
        else:
            if epoch == 100:
                self.G.load_state_dict(self.vae2.decoder.state_dict())
            for _, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                self.train_VAE1_step(x)
                if epoch < 500:
                    self.train_GAN_step(x)
                if epoch < 200:
                    self.train_VAE2_step(x)
                else:
                    self.train_VAE2_aug_step(x)
                    

    def train_VAE2_aug_step(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        z = torch.randn((num_batch_aug_samples * len(x), self.config['vae_config']['d']), device=self.device)
        samples = self.G(z)
        score = torch.nn.Sigmoid()(self.D(samples))
        weight = torch.logit(score, eps=1e-3).exp().detach()
        loss1 = self.vae2.loss(x)
        loss2 = self.vae2.loss(samples, weight)
        loss = loss1 + loss2
        self.optimizer_VAE2.zero_grad()
        loss.backward()
        self.optimizer_VAE2.step()


    def train_VAE1_step(self, x):
        self.optimizer_VAE1.zero_grad()
        loss_VAE = self.vae1.loss(x)
        loss_VAE.backward()
        self.optimizer_VAE1.step()

    def train_VAE2_step(self, x):
        self.optimizer_VAE2.zero_grad()
        loss_VAE = self.vae2.loss(x)
        loss_VAE.backward()
        self.optimizer_VAE2.step()

    def train_D_step(self, x):
        labels_real = torch.ones(len(x)).to(self.device)
        labels_fake = torch.zeros(len(x)).to(self.device)
        d_loss_real = torch.nn.BCEWithLogitsLoss()(self.D(x).squeeze(), labels_real)
        z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
        d_loss_fake = torch.nn.BCEWithLogitsLoss()(self.D(self.vae2.decoder(z)).squeeze(), labels_fake)
        d_loss = d_loss_real + d_loss_fake
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
    
    def train_GAN_step(self, x):
        # Train Discriminator
        labels_real = torch.ones(len(x)).to(self.device)
        labels_fake = torch.zeros(len(x)).to(self.device)
        d_loss_real = torch.nn.BCEWithLogitsLoss()(self.D(x).squeeze(), labels_real)
        z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
        d_loss_fake = torch.nn.BCEWithLogitsLoss()(self.D(self.G(z)).squeeze(), labels_fake)
        d_loss = d_loss_real + d_loss_fake
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        # Train Generator
        z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
        x_fake = self.G(z)
        g_loss = torch.nn.BCEWithLogitsLoss()(self.D(x_fake).squeeze(), labels_real)
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

    def train_epoch_GAN(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)

            # Train Discriminator
            labels_real = torch.ones(len(x)).to(self.device)
            labels_fake = torch.zeros(len(x)).to(self.device)
            d_loss_real = torch.nn.BCEWithLogitsLoss()(self.D(x).squeeze(), labels_real)
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            d_loss_fake = torch.nn.BCEWithLogitsLoss()(self.D(self.G(z)).squeeze(), labels_fake)
            d_loss = d_loss_real + d_loss_fake
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            # Train Generator
            z = torch.randn((len(x), self.config['vae_config']['d']), device=self.device)
            x_fake = self.G(z)
            g_loss = torch.nn.BCEWithLogitsLoss()(self.D(x_fake).squeeze(), labels_real)
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # Write tensorboard statistics
            # self.writer.add_scalars('training loss', {
            #     'd_loss': d_loss.item(),
            #     'g_loss': g_loss.item(),
            #     'total_loss': d_loss.item() + g_loss.item(),
            # }, self.num_grads)
            # self.num_grads += 1
    
    def evaluate(self):
        z = torch.randn((64, self.config['vae_config']['d']), device=self.device)
        x_fake = self.G(z)

        score = torch.nn.Sigmoid()(self.D(x_fake))
        weight = torch.logit(score, eps=1e-3).exp().detach()
        with torch.no_grad():
            self.writer.add_scalars('BPD', {
                'train1': self.vae1.loss(self.x_train).item(),
                'train2': self.vae2.loss(self.x_train).item(),
                'test1': self.vae1.loss(self.x_test).item(),
                'test2': self.vae2.loss(self.x_test).item(),
            }, self.num_epoch)

        self.writer.add_scalar('weight_mean', weight.mean(), self.num_epoch)
        
        score_test = torch.nn.Sigmoid()(self.D(self.x_test))
        acc_test = len((score_test > 0.5).nonzero()) / len(self.x_test)
        self.writer.add_scalar('acc_test_samples', acc_test, self.num_epoch)

        self.writer.add_images('generation', x_fake.mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8), self.num_epoch)

        self.writer.close()



    def save(self):
        save_path = os.path.join(self.ckpt_dir, f'GAN_{self.num_epoch}.pt')
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        }
        torch.save(state, save_path)
        print(f'Model saved at {save_path}')

    def load(self, path):
        state = torch.load(path, map_location='cpu')
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])
        print(f'Loaded a trained model from {path}.')