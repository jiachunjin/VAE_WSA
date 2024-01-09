import torch
import numpy as np

from torch.distributions.normal import Normal

from model import VAE
from utils import requires_grad
from .components import Energy_IMG_MLP


class VAEBM():
    def __init__(self, config, vae_config):
        self.config = config
        self.vae_config = vae_config
        self.device = config['device']
        self._build_model()
        self._load_VAE(config['load_path'])

    def _build_model(self):
        config = self.vae_config
        self.vae = VAE(config)
        self.energy = Energy_IMG_MLP(self.config['energy_config']).to(self.device)

    def _load_VAE(self, load_path):
        self.vae.load(load_path)
        requires_grad(self.vae.encoder.parameters(), False)
        requires_grad(self.vae.decoder.parameters(), False)

    def loss_EBM(self, x):
        num_steps = self.config['sampler_config']['num_steps_train']
        E_data = self.energy(x)
        simulated_x = self.sample_from_model(len(x), num_steps)
        E_model = self.energy(simulated_x)
        regularize_coeff = self.config['energy_config']['regularize_coeff']
        loss = E_data.mean() - E_model.mean() + regularize_coeff * ((E_data**2).mean() + (E_model**2).mean())
        return loss

    def sample_from_model(self, num_samples, num_steps):
        epsilon_x = torch.randn((num_samples, 1, 28, 28), device=self.device)
        alpha = self.config['sampler_config']['alpha']
        noise_std = self.config['sampler_config']['noise_std']
        for _ in range(num_steps):
            epsilon_x.requires_grad = True
            grads = torch.autograd.grad(outputs=self.energy(epsilon_x).sum(), inputs=epsilon_x)[0]
            epsilon_x = (epsilon_x - (alpha / 2) * grads + noise_std * torch.randn_like(grads)).detach()
        assert epsilon_x.requires_grad == False
        return epsilon_x
        # d = self.vae_config['vae_config']['d']
        # # First we sample epsilon_x and epsilon_z
        # epsilon_x, epsilon_z = torch.randn((num_samples, self.config['dataset_config']['input_channel'], self.config['dataset_config']['img_size'], self.config['dataset_config']['img_size']), device=self.device), torch.randn((num_samples, d), device=self.device)
        
        # alpha = self.config['sampler_config']['alpha']
        # noise_std = self.config['sampler_config']['noise_std']
        
        # p_0 = Normal(torch.zeros(d, device=self.device), torch.ones(d, device=self.device))
        # e = lambda z: self.energy(self.vae.decoder(z)) - p_0.log_prob(z)

        # for i in range(num_steps):
        #     epsilon_z.requires_grad = True
        #     grads = torch.autograd.grad(outputs=e(epsilon_z).sum(), inputs=epsilon_z)[0]
        #     epsilon_z = (epsilon_z - (alpha / 2) * grads + noise_std * torch.randn_like(grads)).detach()
        # assert epsilon_z.requires_grad == False
        # return self.vae.decoder(epsilon_z)

    def save(self, save_path):
        state = {
            'energy': self.energy.state_dict(),
        }
        torch.save(state, save_path)
        print(f'Model saved at {save_path}')

    def load(self, load_path):
        state = torch.load(load_path, map_location='cpu')
        self.energy.load_state_dict(state['energy'])
        print(f'Loaded a trained model from {load_path}.')