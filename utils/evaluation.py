import torch
import numpy as np
import torch.distributions as dist

def log_density_diagonal_gaussian_batch(x, mean, log_var):
    std = torch.exp(0.5 * log_var)
    normal = dist.Normal(mean, std)
    log_prob = normal.log_prob(x)
    log_prob = torch.sum(log_prob, dim=-1)

    return log_prob

def log_density_bernoulli(x, p):
    log_prob = x * torch.log(p) + (1 - x) * torch.log(1 - p)
    log_prob = torch.sum(log_prob, dim=(1,2,3))
    return log_prob

def L_k(vae, x, k=5000):
    mu, log_var= vae.encoder(x)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn((k, 16))
    z = eps * std + mu
    p = vae.decoder(z)
    p = torch.clamp(p, 1e-3, 1-1e-3)

    logp_xz = log_density_bernoulli(x, p)
    logp_z = log_density_diagonal_gaussian_batch(z, torch.zeros(16), torch.zeros(16))
    logq_xz = log_density_diagonal_gaussian_batch(z, mu, log_var)

    return torch.logsumexp(logp_xz + logp_z - logq_xz, 0) - np.log(k)

def MI(encoder, x):
    mix = dist.Categorical(torch.ones(len(x),))
    mu_, log_var_ = encoder(x)
    std_ = torch.exp(0.5 * log_var_)
    comp = dist.Independent(dist.Normal(mu_, std_), 1)
    agg_posterior = dist.MixtureSameFamily(mix, comp)
    l = []
    for i in range(1000):
        with torch.no_grad():
            mu, log_var = encoder(x[i])
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu
            posterior = dist.Normal(mu, std)
            mi_i = posterior.log_prob(z).sum() - agg_posterior.log_prob(z).sum()
            l.append(mi_i)
    return np.array(l).mean()

def entropy(dist, x):
    return -dist.log_prob(x).sum(dim=1).mean()