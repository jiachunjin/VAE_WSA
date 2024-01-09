import torch

from torch.nn import functional as F
from torch.distributions.normal import Normal

from model import VAE
from utils import requires_grad
from .components import Energy_IMG_MLP

class Threeinone():
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.d = config['vae_config']['d']
        self._build_model()

    def _build_model(self):
        self.vae = VAE(self.config)
        self.energy = Energy_IMG_MLP(self.config['energy_config']).to(self.device)

    def loss_vae(self, x):
        requires_grad(self.vae.encoder.parameters(), True)
        requires_grad(self.vae.decoder.parameters(), True)

        return self.vae.loss(x)

    def loss_ebm(self, x):
        requires_grad(self.vae.encoder.parameters(), False)
        requires_grad(self.vae.decoder.parameters(), False)

        num_steps = self.config['sampler_config']['num_steps_train']
        E_data = self.energy(x)
        simulated_x = self.sample_from_model(len(x), num_steps)
        E_model = self.energy(simulated_x)
        regularize_coeff = self.config['energy_config']['regularize_coeff']
        loss = E_data.mean() - E_model.mean() + regularize_coeff * ((E_data**2).mean() + (E_model**2).mean())
        return loss

    def sample_from_model(self, num_samples, num_steps):
        d = self.config['vae_config']['d']
        # First we sample epsilon_x and epsilon_z
        epsilon_z = torch.randn((num_samples, d), device=self.device)
        
        alpha = self.config['sampler_config']['alpha']
        noise_std = self.config['sampler_config']['noise_std']
        
        p_0 = Normal(torch.zeros(d, device=self.device), torch.ones(d, device=self.device))
        e = lambda z: self.energy(self.vae.decoder(z)) - p_0.log_prob(z)

        for _ in range(num_steps):
            epsilon_z.requires_grad = True
            grads = torch.autograd.grad(outputs=e(epsilon_z).sum(), inputs=epsilon_z)[0]
            epsilon_z = (epsilon_z - (alpha / 2) * grads + noise_std * torch.randn_like(grads)).detach()
        assert epsilon_z.requires_grad == False
        return self.vae.decoder(epsilon_z)

    def conditional_sample(self, num_steps, epsilon_z):
        d = self.config['vae_config']['d']
        alpha = self.config['sampler_config']['alpha']
        noise_std = self.config['sampler_config']['noise_std']
        
        p_0 = Normal(torch.zeros(d, device=self.device), torch.ones(d, device=self.device))
        e = lambda z: self.energy(self.vae.decoder(z)) - p_0.log_prob(z)

        for _ in range(num_steps):
            epsilon_z.requires_grad = True
            grads = torch.autograd.grad(outputs=e(epsilon_z).sum(), inputs=epsilon_z)[0]
            epsilon_z = (epsilon_z - (alpha / 2) * grads + noise_std * torch.randn_like(grads)).detach()
        assert epsilon_z.requires_grad == False
        return self.vae.decoder(epsilon_z)

    def save(self, save_path):
        state = {
            'encoder': self.vae.encoder.state_dict(),
            'decoder': self.vae.decoder.state_dict(),
            'energy': self.energy.state_dict(),
        }
        torch.save(state, save_path)
        print(f'Model saved at {save_path}')

    def load(self, load_path):
        state = torch.load(load_path, map_location='cpu')
        self.vae.encoder.load_state_dict(state['encoder'])
        self.vae.decoder.load_state_dict(state['decoder'])
        self.energy.load_state_dict(state['energy'])
        print(f'Loaded a trained model from {load_path}.')



    # def loss_vae_with_ebm(self, x):
    #     if self.config['dataset_config']['data'] == 'GaussianMixture':
    #         recons_calibration = x.shape[1]
    #     else:
    #         recons_calibration = x.shape[2] * x.shape[3]
    #     num_steps = self.config['sampler_config']['num_steps_test']
    #     simulated_x = self.sample_from_model(len(x), num_steps)

    #     # loss theta
    #     requires_grad(self.vae.encoder.parameters(), False)
    #     requires_grad(self.vae.decoder.parameters(), True)
    #     mu, log_var = self.vae.encoder(x)
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std, device=self.device)        
    #     z = eps * std + mu
    #     recons = self.vae.decoder(z)
    #     recons_loss = F.mse_loss(recons, x) * recons_calibration
    #     kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    #     loss_theta = recons_loss + self.beta * kl_loss

    #     # loss phi
    #     requires_grad(self.vae.encoder.parameters(), True)
    #     requires_grad(self.vae.decoder.parameters(), False)
    #     x = torch.cat((x, simulated_x), dim=0)
    #     mu, log_var = self.vae.encoder(x)
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std, device=self.device)        
    #     z = eps * std + mu
    #     recons = self.vae.decoder(z)
    #     recons_loss = F.mse_loss(recons, x) * recons_calibration
    #     kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    #     loss_phi = recons_loss + self.beta * kl_loss

    #     return loss_theta + loss_phi