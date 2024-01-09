import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
# import torch.distributions as dist

from .components import Encoder_IMG_MLP, Encoder_IMG_CNN, Encoder_VEC_MLP, Decoder_IMG_CNN_32x3, Decoder_VEC_MLP, Decoder_IMG_CNN_28_C1, Decoder_IMG_CNN_105_C1, Decoder_IMG_MLP_C1
# from utils import kl_divergence_gaussians



class VAE():
    def __init__(self, config):
        self.config = config
        self.device = config['vae_config']['device']
        self.d = config['vae_config']['d']
        self.beta = config['vae_config']['beta']
        self._build_autoencoder()
        assert config['vae_config']['dtype'] in ['binary', 'continuous']
        self.dtype = config['vae_config']['dtype'] = config['vae_config']['dtype']
        # if config['vae_config']['prior'] == 'vamp':
        #     self._build_vamp_prior()
        # elif config['vae_config']['prior'] == 'normal':
        #     pass

    def _build_autoencoder(self):
        config = self.config
        # build encoder
        if config['encoder_config']['type'] == 'IMG_MLP':
            self.encoder = Encoder_IMG_MLP(config['encoder_config']).to(self.device)
        elif config['encoder_config']['type'] == 'IMG_CNN':
            self.encoder = Encoder_IMG_CNN(config['encoder_config']).to(self.device)
        elif config['encoder_config']['type'] == 'VEC_MLP':
            self.encoder = Encoder_VEC_MLP(config['encoder_config']).to(self.device)
        # build decoder
        if config['decoder_config']['type'] == 'IMG_CNN_32':
            self.decoder = Decoder_IMG_CNN_32x3(config['decoder_config']).to(self.device)
        elif config['decoder_config']['type'] == 'VEC_MLP':
            self.decoder = Decoder_VEC_MLP(config['decoder_config']).to(self.device)
        elif config['decoder_config']['type'] == 'IMG_CNN_C1':
            if config['decoder_config']['img_size'] == 28:
                self.decoder = Decoder_IMG_CNN_28_C1(config['decoder_config']).to(self.device)
            elif config['decoder_config']['img_size'] == 105:
                self.decoder = Decoder_IMG_CNN_105_C1(config['decoder_config']).to(self.device)
        elif config['decoder_config']['type'] == 'IMG_MLP_C1':
            self.decoder = Decoder_IMG_MLP_C1(config['decoder_config']).to(self.device)

    def loss_AIR(self, x, air_sigma):
        x_ = x + air_sigma * torch.randn_like(x)
        mu, log_var = self.encoder(x_)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)
        z = eps * std + mu
        if self.dtype == 'binary':
            bernoullis = self.decoder(z)
            recons_loss = (torch.nn.BCELoss(reduction='none')(bernoullis, x).sum(dim=[i for i in range(1, x.dim())])).mean()
        elif self.dtype == 'continuous':
            recons = self.decoder(z)
            recons_loss = (F.mse_loss(recons, x, reduction='none').sum(dim=[i for i in range(1, x.dim())])).mean()
        kl_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)).mean()
        return recons_loss + kl_loss

    def loss(self, x, weight=None):
        if self.dtype == 'binary':
            return self.loss_binary_data(x, weight)
        elif self.dtype == 'continuous':
            return self.loss_continuous_data(x, weight)
            
    def loss_binary_data(self, x, weight=None):
        if weight == None:
            weight = torch.ones((len(x), ), device=self.device)
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)        
        z = eps * std + mu
        bernoullis = self.decoder(z)
        recons_loss = (weight * torch.nn.BCELoss(reduction='none')(bernoullis, x).sum(dim=[i for i in range(1, x.dim())])).mean()
        kl_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) * weight).mean()

        return recons_loss + kl_loss

    def loss_continuous_data(self, x, weight=None):
        '''
        https://arxiv.org/pdf/2006.10273.pdf
        '''
        if weight == None:
            weight = torch.ones((len(x), ), device=self.device)
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)        
        z = eps * std + mu
        recons = self.decoder(z)

        # recons_loss = (weight * F.mse_loss(recons, x, reduction='none').sum(dim=[i for i in range(1, x.dim())])).mean()

        l = nn.MSELoss(reduction='none')
        mseloss = l(x, recons).sum(dim=[i for i in range(1, x.dim())])
        # if self.config['dataset_config']['data'] == 'toy':
        #     D = 256
        # else:
        D = x.shape[-1] * x.shape[-2] * x.shape[-3]
        log_prob = -D / 2 * np.log(2 * np.pi) - 0.5 * mseloss
        recons_loss = -(weight * log_prob).mean()
        
        # std = torch.exp(0.5 * log_var)
        # normal = dist.Normal(recons, torch.ones_like(recons, device=self.device))
        # log_prob = normal.log_prob(x)
        # log_prob = torch.sum(log_prob, dim=[i for i in range(1, log_prob.dim())])
        # recons_loss = (weight * log_prob).mean()


        kl_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) * weight).mean()

        return recons_loss + self.beta * kl_loss
    
    def distortion(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)        
        z = eps * std + mu
        if self.dtype == 'binary':
            bernoullis = self.decoder(z)
            recons_loss = (torch.nn.BCELoss(reduction='none')(bernoullis, x).sum(dim=[i for i in range(1, x.dim())])).mean()
        elif self.dtype == 'continuous':
            recons = self.decoder(z)
            # recons_loss = (F.mse_loss(recons, x, reduction='none').sum(dim=[i for i in range(1, x.dim())])).mean()
            l = nn.MSELoss(reduction='none')
            mseloss = l(x, recons).sum(dim=[i for i in range(1, x.dim())])
            D = x.shape[-1] * x.shape[-2] * x.shape[-3]
            log_prob = -D / 2 * np.log(2 * np.pi) - 0.5 * mseloss
            recons_loss = -(log_prob).mean()
            # l = nn.MSELoss(reduction='none')
            # mseloss = l(x, recons).sum(dim=[i for i in range(1, x.dim())])
            # D = x.shape[-1] * x.shape[-2] * x.shape[-3]
            # log_prob = -D / 2 * np.log(2 * np.pi) - 0.5 * mseloss
            # recons_loss = -log_prob.mean()

        return recons_loss

    def rate(self, x):
        mu, log_var = self.encoder(x)
        kl_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)).mean()
        return kl_loss

    def save(self, save_path):
        state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }
        torch.save(state, save_path)
        print(f'Model saved at {save_path}')

    def load(self, load_path):
        state = torch.load(load_path, map_location='cpu')
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])

        print(f'Loaded a trained model from {load_path}.')

    # def _build_vamp_prior(self):
    #     config = self.config
    #     self.num_pseudodata = config['vae_config']['num_pseudodata']
    #     self.pseudo_input = torch.eye(self.num_pseudodata, requires_grad= False).to(self.device)
    #     self.embed_pseudo = nn.Sequential(
    #         nn.Linear(self.num_pseudodata, config['dataset_config']['input_channel']*config['dataset_config']['img_size']*config['dataset_config']['img_size']),
    #         nn.Hardtanh(0.0, 1.0),
    #         ).to(self.device)

    # def loss_info(self, x):
    #     if self.config['dataset_config']['data'] == 'GaussianMixture':
    #         recons_calibration = x.shape[1]
    #     else:
    #         recons_calibration = x.shape[2] * x.shape[3]
    #     mu, log_var = self.encoder(x)
    #     # mu, log_var = mu.detach(), log_var.detach()

    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std, device=self.device)        

    #     z = eps * std + mu
    #     recons_up = self.decoder(z)
    #     recons_loss = F.mse_loss(recons_up, x) * recons_calibration

    #     kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     def compute_kernel(x1, x2):
    #         eps = 1e-7
    #         z_dim = x2.size(-1)
    #         C = 2 * z_dim * 2
    #         kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    #         # Exclude diagonal elements
    #         result = kernel.sum() - kernel.diag().sum()
    #         return result


    #     def compute_mmd(z):
    #         """
    #         MMM(q || p) = E_{p(z), p(z')}[k(z, z')] - 2E_{q(z), p(z')}[k(z, z')] + E_{q(z), q(z')}[k(z, z')]
    #         Args:
    #             z: shape: (B, latent_dim), from reparameterization
    #         """
    #         prior_z = torch.randn_like(z)
    #         term_1 = compute_kernel(prior_z, prior_z)
    #         term_2 = compute_kernel(prior_z, z)
    #         term_3 = compute_kernel(z, z)
    #         mmd = term_1.mean() - 2 * term_2.mean() + term_3.mean()
    #         return mmd

    #     mmd_loss = compute_mmd(z)

    #     losses = {
    #         'total': recons_loss + 0 * kl_loss + 1000 * mmd_loss,
    #         'recons_loss': recons_loss,
    #         'kl_loss': kl_loss,
    #         'mmd_loss': mmd_loss
    #     }

    #     return losses        

    # def loss_new(self, x):
    #     if self.config['dataset_config']['data'] == 'GaussianMixture':
    #         recons_calibration = x.shape[1]
    #     else:
    #         recons_calibration = x.shape[2] * x.shape[3]
        
    #     mu, log_var = self.encoder(x)
    #     # mu, log_var = mu.detach(), log_var.detach()

    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std, device=self.device)        

    #     z = eps * std + mu
    #     recons_up = self.decoder(z)
    #     recons_loss = F.mse_loss(recons_up, x) * recons_calibration

    #     kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     from torch.distributions.normal import Normal
    #     z = torch.randn((4096, 2)).to(self.device)
    #     # x = self.decoder(z).detach()
    #     x = self.decoder(z)
    #     mu, log_var = self.encoder(x)
    #     scale = torch.exp(0.5 * log_var)
    #     n = Normal(mu, scale)
    #     new_term = n.log_prob(z).sum(1).mean()

    #     losses = {
    #         'total': recons_loss + self.beta * kl_loss - new_term,
    #         'recons_loss': recons_loss,
    #         'kl_loss': kl_loss,
    #         'new_term': new_term
    #     }

    #     return losses

    # def loss_vamp(self, x):
    #     if self.config['dataset_config']['data'] == 'GaussianMixture':
    #         recons_calibration = x.shape[1]
    #     else:
    #         recons_calibration = x.shape[2] * x.shape[3]

    #     mu, log_var = self.encoder(x)

    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std, device=self.device)        

    #     z = eps * std + mu
    #     recons_up = self.decoder(z)
    #     recons_loss = F.mse_loss(recons_up, x) * recons_calibration

    #     B, C, H, W = x.shape
    #     _, d = mu.shape
    #     pseudo_x = self.embed_pseudo(self.pseudo_input)
    #     pseudo_x = pseudo_x.view(-1, C, H, W)
    #     mu_prior, log_var_prior = self.representation(pseudo_x)

    #     z = torch.randn((self.config['vae_config']['num_kl_samples'], B, d), device=self.device) * torch.exp(0.5 * log_var) + mu

    #     E_log_q_z = torch.mean(torch.sum(-0.5 * (log_var + (z - mu) ** 2 / log_var.exp()), dim = 2))

    #     z_expand = z.unsqueeze(2)
    #     mu_prior = mu_prior.unsqueeze(0)
    #     log_var_prior = log_var_prior.unsqueeze(0)
    #     log_q_z_pseudo = torch.sum(-0.5 * (log_var_prior + (z_expand - mu_prior) ** 2 / log_var_prior.exp()), dim = 3)
    #     log_q_z_pseudo = torch.logsumexp(log_q_z_pseudo, dim = 2)
    #     E_log_p_z = torch.mean(log_q_z_pseudo)

    #     kl_loss = E_log_q_z - E_log_p_z

    #     losses = {
    #         'total': recons_loss + self.beta * kl_loss,
    #         'recons_loss': recons_loss,
    #         'kl_loss': kl_loss
    #     }

    #     return losses

    # def loss_sgm_aug(self, x):
    #     # generate some data from pretrained SGM model
    #     pass

