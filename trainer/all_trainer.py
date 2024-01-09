import os
import copy
import torch
import numpy as np
import torch.distributions as dist

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange

from torchvision.transforms import RandomRotation, RandomHorizontalFlip

from .base_trainer import Base_Trainer
from model import VAE, Binary_Classifier_MLP, Binary_Classifier_CNN
from utils import dataloader_one_batch, requires_grad

class All_Trainer(Base_Trainer):
    def __init__(self, mode, config, log_dir, ckpt_dir, sample_dir):
        self.mode = mode
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
        x_train, _ = dataloader_one_batch(self.train_loader) # For evaluation
        x_test, _ = dataloader_one_batch(self.test_loader)
        self.x_train = x_train.to(self.device)
        self.x_test = x_test.to(self.device)
        self.results = {
            'training_ELBO': [],
            'test_ELBO': [],
            'weight': [],
            'training_distortion': [],
            'training_rate': [],
            'training_MI': [],
            'test_distortion': [],
            'test_rate': [],
            'test_MI': [],
            'distance_train': [],
        }
    
    def _build_model(self):
        if self.mode['name'] == 'dropout':
            self.config['encoder_config']['dropout_p'] = self.mode['dropout_p']
            self.config['decoder_config']['dropout_p'] = self.mode['dropout_p']
            self.vae = VAE(self.config)
        else:
            self.vae = VAE(self.config)

        if self.mode['name'] == 'wsa':
            self._build_classifier()
        
        # if self.config['load_path'] != 'None':
        #     state = torch.load(self.config['load_path'], map_location='cpu')
        #     self.vae.encoder.load_state_dict(state['encoder'])
        #     self.vae.decoder.load_state_dict(state['decoder'])
        #     self.classifier.load_state_dict(state['classifier'])
        #     self.vae.encoder.to(self.device)
        #     self.vae.decoder.to(self.device)
        #     self.classifier.to(self.device)


        if self.mode['name'] == 'augmentation':
            if self.mode['aug_type'] == 'RandomRotation':
                self.T = RandomRotation((-30, 30))
            elif self.mode['aug_type'] == 'RandomHorizontalFlip':
                self.T = RandomHorizontalFlip(1)
            else:
                raise NotImplementedError

        if self.mode['name'] == 'two_augmentation':
            self._build_classifier()
            if self.mode['aug_type'] == 'RandomRotation':
                self.T = RandomRotation((-30, 30))
            elif self.mode['aug_type'] == 'RandomHorizontalFlip':
                self.T = RandomHorizontalFlip(1)
            else:
                raise NotImplementedError

    def _build_classifier(self):
        if self.config['classifier_config']['type'] == 'IMG_MLP':
            self.classifier = Binary_Classifier_MLP(self.config['classifier_config']).to(self.device)
        elif  self.config['classifier_config']['type'] == 'IMG_CNN':
            self.classifier = Binary_Classifier_CNN(self.config['classifier_config']).to(self.device)

    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer_vae = Adam(
            [
                {'params': self.vae.encoder.parameters()},
                {'params': self.vae.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        if self.mode['name'] in ['wsa', 'two_augmentation']:
            self.optimizer_classifier = Adam(
                [
                    {'params': self.classifier.parameters()},
                ],
                lr = float(optimizer_config['lr']),
            )
        if self.mode['name'] in ['rws']:
            self.optimizer_encoder = Adam(
                [
                    {'params': self.vae.encoder.parameters()},
                ],
                lr = float(optimizer_config['lr']),
            )
            self.optimizer_decoder = Adam(
                [
                    {'params': self.vae.decoder.parameters()},
                ],
                lr = float(optimizer_config['lr']),
            )

    def train(self):
        self.vae.encoder.train()
        self.vae.decoder.train()
        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch(epoch)
            self.num_epoch += 1
            if epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()
            if epoch % self.config['trainer_config']['save_every'] == 0:
                self.save()

    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, leave=False)
        if self.mode['name'] in ['vanilla', 'dropout']:
            # always train the model with the VAE loss
            for _, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                self._train_vae_step(x)
        else:
            if self.mode['name'] == 'augmentation':
                for _, (x, _) in enumerate(pbar):
                    x = x.to(self.device)
                    self._train_augmentation_step(x)
            elif self.mode['name'] == 'air':
                for _, (x, _) in enumerate(pbar):
                    x = x.to(self.device)
                    self._train_air_step(x)
            elif self.mode['name'] == 'rws':
                for _, (x, _) in enumerate(pbar):
                    x = x.to(self.device)
                    self._train_rws_step(x)
            else:
                # there is a two stage training
                aug_epoch = int(self.config['trainer_config']['aug_epoch'])
                if epoch < aug_epoch:
                    for _, (x, _) in enumerate(pbar):
                        x = x.to(self.device)
                        self._train_vae_step(x)
                        self._train_dre_step(x)
                else:
                    epoch_weights = []
                    for _, (x, _) in enumerate(pbar):
                        x = x.to(self.device)
                        if self.mode['name'] == 'wsa':
                            weight = self._train_self_augmentation_step(x)
                            epoch_weights.append(weight)
                        elif self.mode['name'] == 'two_augmentation':
                            weight = self._train_two_augmentation_step(x)
                            epoch_weights.append(weight)
                        self._train_dre_step(x)
                    weight_mean = np.array(epoch_weights).mean()
                    self.results['weight'].append(weight_mean)
                    if self.mode['tensorboard']:
                        self.writer.add_scalar('weight', weight_mean, self.num_epoch)

    def _train_vae_step(self, x):
        self.optimizer_vae.zero_grad()
        loss_vae = self.vae.loss(x)
        loss_vae.backward()
        self.optimizer_vae.step()
        self.num_grads += 1

    def _train_self_augmentation_step(self, x):
        samples, weight = self._self_augmentation(x)
        self.optimizer_vae.zero_grad()
        loss1 = self.vae.loss(x)
        if self.mode['use_weight']:
            loss2 = self.vae.loss(samples, weight)
        else:
            # no weighting term, this will hurt the training when samples are bad
            loss2 = self.vae.loss(samples)
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        self.optimizer_vae.step()
        self.num_grads += 1
        return weight.mean().item()

    def _train_augmentation_step(self, x):
        samples = self.T(x)
        self.optimizer_vae.zero_grad()
        loss1 = self.vae.loss(x)
        loss2 = self.vae.loss(samples)
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        self.optimizer_vae.step()
        self.num_grads += 1
    
    def _train_two_augmentation_step(self, x):
        samples, weight = self._self_augmentation(x)
        samples_aug = self.T(x)
        self.optimizer_vae.zero_grad()
        loss1 = self.vae.loss(x)
        loss2 = self.vae.loss(samples, weight)
        loss3 = self.vae.loss(samples_aug)
        loss = 0.33 * loss1 + 0.33 * loss2 + 0.33 * loss3
        loss.backward()
        self.optimizer_vae.step()
        self.num_grads += 1
        return weight.mean()

    def _train_rws_step(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        samples = self._sample_vae(self.vae, num_batch_aug_samples * len(x)).to(self.device, x.dtype)

        self.optimizer_decoder.zero_grad()
        # requires_grad(self.vae.encoder.parameters(), False)
        loss1 = self.vae.loss(x)
        loss1.backward()
        self.optimizer_decoder.step()
        # requires_grad(self.vae.encoder.parameters(), True)
        
        self.optimizer_encoder.zero_grad()
        # requires_grad(self.vae.decoder.parameters(), False)
        loss2 = 0.5 * self.vae.loss(x) + 0.5 * self.vae.loss(samples)
        # loss2 = self.vae.loss(samples)
        loss2.backward()
        self.optimizer_encoder.step()
        # requires_grad(self.vae.decoder.parameters(), True)
        self.num_grads += 1

    def _train_air_step(self, x):
        air_sigma = self.mode['air_sigma']
        self.optimizer_vae.zero_grad()
        loss1 = self.vae.loss(x)
        loss2 = self.vae.loss_AIR(x, air_sigma)
        loss = 0.5 * loss1 + 0.5 * loss2
        loss.backward()
        self.optimizer_vae.step()
        self.num_grads += 1

    def _train_dre_step(self, x):
        x_ = self._sample_vae(self.vae, len(x)).to(self.device, x.dtype)
        labels = torch.cat((torch.ones(len(x), device=self.device),
                            torch.zeros(len(x), device=self.device)), dim=0).float()
        x = torch.cat((x, x_), dim=0)
        self.optimizer_classifier.zero_grad()
        loss_dre = torch.nn.BCEWithLogitsLoss()(self.classifier(x).squeeze(), labels)
        loss_dre.backward()
        self.optimizer_classifier.step()

    def _sample_vae(self, vae, num):
        z = torch.randn((num, self.config['vae_config']['d']), device=self.device)
        p = vae.decoder(z).detach()
        if vae.dtype == 'binary':
            return (torch.rand_like(p) < p)
        elif vae.dtype == 'continuous':
            return p

    def _self_augmentation(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        weight_clamp_max = self.config['trainer_config']['weight_clamp_max']
        samples = self._sample_vae(self.vae, num_batch_aug_samples * len(x)).to(self.device, x.dtype)
        score = torch.nn.Sigmoid()(self.classifier(samples))
        weight = torch.logit(score, eps=1e-3).exp()
        weight = torch.clamp(weight, max=weight_clamp_max)
        return samples, weight
    
    def evaluate(self):
        self.vae.encoder.eval()
        self.vae.decoder.eval()

        if self.config['dataset_config']['data'] != 'toy':
            if self.mode['tensorboard']:
                samples2 = self._sample_vae(self.vae, 64).to(self.device).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)
                self.writer.add_images('generation', samples2, self.num_epoch)

        with torch.no_grad():
            training_ELBO = -self.vae.loss(self.x_train).item()
            test_ELBO = -self.vae.loss(self.x_test).item()
            train_distortion = self.vae.distortion(self.x_train).item()
            train_rate = self.vae.rate(self.x_train).item()
            test_distortion = self.vae.distortion(self.x_test).item()
            test_rate = self.vae.rate(self.x_test).item()
            if self.config['dataset_config']['data'] == 'toy':
                samples = self._sample_vae(self.vae, len(self.x_train)).to(self.device)
                distance_train = self._tensor_distance_x_train(samples)
                self.results['distance_train'].append(distance_train)
                # train_MI = self._MI(self.vae.encoder, self.x_train)
                # test_MI = self._MI(self.vae.encoder, self.x_test)
                # self.results['test_MI'].append(test_MI)

        self.results['training_ELBO'].append(training_ELBO)
        self.results['test_ELBO'].append(test_ELBO)

        self.results['training_distortion'].append(train_distortion)
        self.results['training_rate'].append(train_rate)

        self.results['test_distortion'].append(test_distortion)
        self.results['test_rate'].append(test_rate)
        if self.mode['tensorboard']:
            self.writer.add_scalars('ELBO', {
                'training': training_ELBO,
                'test': test_ELBO,
            }, self.num_epoch)
        self.writer.close()

    def save(self):
        if self.mode['name'] in ['wsa', 'two_augmentation']:
            state = {
                'encoder': self.vae.encoder.state_dict(),
                'decoder': self.vae.decoder.state_dict(),
                'classifier': self.classifier.state_dict(),
            }
        else:
            state = {
                'encoder': self.vae.encoder.state_dict(),
                'decoder': self.vae.decoder.state_dict(),
            }
        name = self.mode['name']
        result_path = os.path.join(self.ckpt_dir, f'{name}_results')
        state_path = os.path.join(self.ckpt_dir, f'{name}_ckpts_{self.num_epoch}')

        torch.save(self.results, result_path)
        torch.save(state, state_path)

    def _MI(self, encoder, x):
        mix = dist.Categorical(torch.ones(len(x),))
        mu_, log_var_ = encoder(x)
        std_ = torch.exp(0.5 * log_var_)
        comp = dist.Independent(dist.Normal(mu_, std_), 1)
        agg_posterior = dist.MixtureSameFamily(mix, comp)

        l = []
        for i in range(len(x)):
            with torch.no_grad():
                mu, log_var = encoder(x[i])
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = eps * std + mu
                posterior = dist.Normal(mu, std)
                mi_i = posterior.log_prob(z).sum() - agg_posterior.log_prob(z).sum()
                l.append(mi_i)
        return np.array(l).mean()

    def _tensor_distance_x_train(self, samples):
        nn_distance = []
        for i in range(samples.size(0)):
            x = samples[i].to(torch.float32)
            d = ((x - self.x_train)**2).mean(dim=1)
            nn_distance.append(torch.min(d))
        return np.mean(nn_distance)