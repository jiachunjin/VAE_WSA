import os
import torch

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from model import VAEBM


class VAEBM_Trainer(Base_Trainer):
    def __init__(self, config, vae_config, log_dir, ckpt_dir, sample_dir):
        self.vae_config = vae_config
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
    
    def _build_model(self):
        self.vaebm = VAEBM(self.config, self.vae_config)
    
    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer = Adam(
            [
                {'params': self.vaebm.energy.parameters()},
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
                self.save()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        for _, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.vaebm.loss_EBM(x)
            loss.backward()
            self.writer.add_scalars('training loss', {
                'loss': loss.item(),
            }, self.num_grads)

            self.optimizer.step()
            self.num_grads += 1

    def evaluate(self):
        num_steps = self.config['sampler_config']['num_steps_test']
        samples = self.vaebm.sample_from_model(64, num_steps).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
        self.writer.add_images('samples', samples, self.num_epoch)

    def save(self):
        save_path = os.path.join(self.ckpt_dir, f'ckpt_energy_{self.num_epoch}.pt')
        self.vaebm.save(save_path)
