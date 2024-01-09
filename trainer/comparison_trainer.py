import os
import copy
import torch

from torch.optim import Adam
from tqdm.autonotebook import tqdm, trange
from torchvision.transforms import RandomRotation, RandomHorizontalFlip

from .base_trainer import Base_Trainer
from model import VAE, Binary_Classifier_MLP, Binary_Classifier_CNN
from utils import dataloader_one_batch, save_img_tensors, requires_grad


class Comparison_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
        x_train, _ = dataloader_one_batch(self.train_loader)
        x_test, _ = dataloader_one_batch(self.test_loader)
        self.x_train = x_train.to(self.device)
        self.x_test = x_test.to(self.device)
        self.results = {
            'train_1': [],
            'train_2': [],
            'train_3': [],
            'train_4': [],
            'train_5': [],
            'train_6': [],
            'train_7': [],
            'train_8': [],
            'test_1': [],
            'test_2': [],
            'test_3': [],
            'test_4': [],
            'test_5': [],
            'test_6': [],
            'test_7': [],
            'test_8': [],
        }
    
    def _build_model(self):
        dropout_config = copy.deepcopy(self.config)
        dropout_config['encoder_config']['dropout_p'] = 0.2
        dropout_config['decoder_config']['dropout_p'] = 0.2

        self.vae_1 = VAE(self.config) # vanilla
        self.vae_2 = VAE(self.config) # mine
        self.vae_3 = VAE(self.config) # without the weighting term, this can sometimes be ignored since theoretically wrong
        self.vae_4 = VAE(self.config) # augment with some inductive bias, rotation
        self.vae_5 = VAE(self.config) # mine with rotation
        self.vae_6 = VAE(self.config) # rws
        self.vae_7 = VAE(self.config) # AIR
        self.vae_8 = VAE(dropout_config) # dropout
        if self.config['classifier_config']['type'] == 'IMG_MLP':
            self.classifier = Binary_Classifier_MLP(self.config['classifier_config']).to(self.device)
        elif  self.config['classifier_config']['type'] == 'IMG_CNN':
            self.classifier = Binary_Classifier_CNN(self.config['classifier_config']).to(self.device)
    
    def _build_optimizer(self):
        optimizer_config = self.config['optimizer_config']
        self.optimizer_VAE1 = Adam(
            [
                {'params': self.vae_1.encoder.parameters()},
                {'params': self.vae_1.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE2 = Adam(
            [
                {'params': self.vae_2.encoder.parameters()},
                {'params': self.vae_2.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE3 = Adam(
            [
                {'params': self.vae_3.encoder.parameters()},
                {'params': self.vae_3.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE4 = Adam(
            [
                {'params': self.vae_4.encoder.parameters()},
                {'params': self.vae_4.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE5 = Adam(
            [
                {'params': self.vae_5.encoder.parameters()},
                {'params': self.vae_5.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE6 = Adam(
            [
                {'params': self.vae_6.encoder.parameters()},
                {'params': self.vae_6.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE7 = Adam(
            [
                {'params': self.vae_7.encoder.parameters()},
                {'params': self.vae_7.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )
        self.optimizer_VAE8 = Adam(
            [
                {'params': self.vae_8.encoder.parameters()},
                {'params': self.vae_8.decoder.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )


        self.optimizer_classifier = Adam(
            [
                {'params': self.classifier.parameters()},
            ],
            lr = float(optimizer_config['lr']),
        )

    def train(self):
        num_epoch = int(self.config['trainer_config']['num_epoch'])
        for epoch in trange(1, num_epoch+1):
            self.train_epoch(epoch)
            self.num_epoch += 1
            if epoch % self.config['trainer_config']['evaluate_every'] == 0:
                self.evaluate()
            if epoch % self.config['trainer_config']['save_every'] == 0:
                self.save()

    def train_epoch(self, epoch):
        aug_epoch = int(self.config['trainer_config']['aug_epoch'])
        pbar = tqdm(self.train_loader, leave=False)
        # if epoch == aug_epoch:
        #     self._train_dre(aug_epoch)
        if epoch < aug_epoch:
            for _, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                self._train_vae_1_step(x)
                # self._train_vae_2_step(x)
                # self._train_vae_3_step(x)
                # self._train_vae_4_aug_step(x)
                # self._train_vae_5_first_step(x)
                # self._train_vae_6_step(x)
                # self._train_vae_7_step(x)
                # self._train_vae_8_step(x)
                # self._train_dre_step(x)
        else:
            for _, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                self._train_vae_1_step(x)
                # self._train_vae_2_aug_step(x)
                # self._train_vae_3_aug_step(x)
                # self._train_vae_4_aug_step(x)
                # self._train_vae_5_aug_step(x)
                # self._train_vae_6_aug_step(x)
                # self._train_vae_7_step(x)
                # self._train_vae_8_step(x)

                # self._train_dre_step(x)

    def _train_vae_1_step(self, x):
        self.optimizer_VAE1.zero_grad()
        loss_vae_1 = self.vae_1.loss(x)
        loss_vae_1.backward()
        self.optimizer_VAE1.step()

    def _train_vae_2_step(self, x):
        self.optimizer_VAE2.zero_grad()
        loss_vae_2 = self.vae_2.loss(x)
        loss_vae_2.backward()
        self.optimizer_VAE2.step()

    def _train_vae_3_step(self, x):
        self.optimizer_VAE3.zero_grad()
        loss_vae_3 = self.vae_3.loss(x)
        loss_vae_3.backward()
        self.optimizer_VAE3.step()

    def _train_vae_4_step(self, x):
        self.optimizer_VAE4.zero_grad()
        loss_vae_4 = self.vae_4.loss(x)
        loss_vae_4.backward()
        self.optimizer_VAE4.step()

    def _train_vae_5_first_step(self, x):
        # t = RandomRotation((-30, 30))
        t = RandomHorizontalFlip(1)
        samples = t(x)
        self.optimizer_VAE5.zero_grad()
        loss1 = self.vae_5.loss(x)
        loss2 = self.vae_5.loss(samples)
        loss = loss1 + loss2
        loss.backward()
        self.optimizer_VAE5.step()

    def _train_vae_6_step(self, x):
        self.optimizer_VAE6.zero_grad()
        loss_vae_6 = self.vae_6.loss(x)
        loss_vae_6.backward()
        self.optimizer_VAE6.step()

    def _train_vae_7_step(self, x):
        '''
        Amortized Inference Regularization with DAE
        '''
        air_sigma = self.config['trainer_config']['air_sigma']
        self.optimizer_VAE7.zero_grad()
        loss_vae_7 = self.vae_7.loss_AIR(x, air_sigma)
        loss_vae_7.backward()
        self.optimizer_VAE7.step()

    def _train_vae_8_step(self, x):
        '''
        With dropout in the encoder and the decoder
        '''
        self.vae_8.encoder.train()
        self.vae_8.decoder.train()
        self.optimizer_VAE8.zero_grad()
        loss_vae_8 = self.vae_8.loss(x)
        loss_vae_8.backward()
        self.optimizer_VAE8.step()

    def _train_vae_2_aug_step(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        weight_clamp_min = self.config['trainer_config']['weight_clamp_min']
        weight_clamp_max = self.config['trainer_config']['weight_clamp_max']
        weight_thres = self.config['trainer_config']['weight_thres']
        samples = self.sample_vae(self.vae_2, num_batch_aug_samples * len(x)).to(self.device, x.dtype)
        score = torch.nn.Sigmoid()(self.classifier(samples))
        weight = torch.logit(score, eps=1e-3).exp()
        weight = torch.clamp(weight, min=weight_clamp_min, max=weight_clamp_max)
        if weight.mean() < weight_thres:
            return None

        self.optimizer_VAE2.zero_grad()
        loss1 = self.vae_2.loss(x)
        loss2 = self.vae_2.loss(samples, weight)
        loss = 0.5 * loss1 + 0.5 * loss2
        self.writer.add_scalars('vae 2', {
            'loss1': loss1.item(),
            'loss2': loss2.item(),
        }, self.num_grads)
        self.writer.add_scalar('weight', weight.mean(), self.num_grads)
        
        loss.backward()
        self.optimizer_VAE2.step()
        self.num_grads += 1

    def _train_vae_3_aug_step(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        samples = self.sample_vae(self.vae_3, num_batch_aug_samples * len(x)).to(self.device, x.dtype)

        self.optimizer_VAE3.zero_grad()
        loss1 = self.vae_3.loss(x)
        loss2 = self.vae_3.loss(samples)
        loss = 0.5 * loss1 + 0.5 * loss2

        self.writer.add_scalars('vae 3', {
            'loss1': loss1.item(),
            'loss2': loss2.item(),
        }, self.num_grads)

        loss.backward()
        self.optimizer_VAE3.step()

    def _train_vae_4_aug_step(self, x):
        # t = RandomRotation((-30, 30))
        t = RandomHorizontalFlip(1)

        samples = t(x)

        self.optimizer_VAE4.zero_grad()
        loss1 = self.vae_4.loss(x)
        loss2 = self.vae_4.loss(samples)
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        self.optimizer_VAE4.step()

    def _train_vae_5_aug_step(self, x):
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        weight_clamp_min = self.config['trainer_config']['weight_clamp_min']
        weight_clamp_max = self.config['trainer_config']['weight_clamp_max']
        weight_thres = self.config['trainer_config']['weight_thres']
        samples = self.sample_vae(self.vae_5, num_batch_aug_samples * len(x)).to(self.device, x.dtype)
        score = torch.nn.Sigmoid()(self.classifier(samples))
        weight = torch.logit(score, eps=1e-3).exp()
        weight = torch.clamp(weight, min=weight_clamp_min, max=weight_clamp_max)
        if weight.mean() < weight_thres:
            return None
        # t = RandomRotation((-30, 30))
        t = RandomHorizontalFlip(1)
        samples_aug = t(x)
        loss1 = self.vae_5.loss(x)
        loss2 = self.vae_5.loss(samples, weight)
        loss3 = self.vae_5.loss(samples_aug)
        loss = 0.33 * loss1 + 0.33 * loss2 + 0.33 * loss3

        self.writer.add_scalar('weight', weight.mean(), self.num_grads)
        self.optimizer_VAE5.zero_grad()
        loss.backward()
        self.optimizer_VAE5.step()
    
    def _train_vae_6_aug_step(self, x):
        '''
        Only use the generated data to train the encoder
        '''
        num_batch_aug_samples = self.config['trainer_config']['num_batch_aug_samples']
        samples = self.sample_vae(self.vae_6, num_batch_aug_samples * len(x)).to(self.device, x.dtype)
        self.optimizer_VAE6.zero_grad()
        requires_grad(self.vae_6.encoder.parameters(), False)
        loss1 = self.vae_6.loss(x)
        loss1.backward()
        self.optimizer_VAE6.step()
        requires_grad(self.vae_6.encoder.parameters(), True)
        
        self.optimizer_VAE6.zero_grad()
        requires_grad(self.vae_6.decoder.parameters(), False)
        loss2 = 0.5 * self.vae_6.loss(x) + 0.5 * self.vae_6.loss(samples)
        loss2.backward()
        self.optimizer_VAE6.step()
        requires_grad(self.vae_6.decoder.parameters(), True)

    def _train_dre_step(self, x):
        x_ = self.sample_vae(self.vae_2, len(x)).to(self.device, x.dtype)
        labels = torch.cat((torch.ones(len(x), device=self.device),
                            torch.zeros(len(x), device=self.device)), dim=0).float()
        x = torch.cat((x, x_), dim=0)
        self.optimizer_classifier.zero_grad()
        loss_dre = torch.nn.BCEWithLogitsLoss()(self.classifier(x).squeeze(), labels)
        loss_dre.backward()
        self.optimizer_classifier.step()

    def sample_vae(self, vae, num):
        z = torch.randn((num, self.config['vae_config']['d']), device=self.device)
        p = vae.decoder(z).detach()
        if vae.dtype == 'binary':
            return (torch.rand_like(p) < p)
        elif vae.dtype == 'continuous':
            return p

    def evaluate(self):
        self.vae_8.encoder.eval()
        self.vae_8.decoder.eval()

        samples2 = self.sample_vae(self.vae_2, 64).to(self.device).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)

        self.writer.add_images('generation', samples2, self.num_epoch)

        samples1 = self.sample_vae(self.vae_1, 64).to(self.device).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)

        self.writer.add_images('generation', samples2, self.num_epoch)

        self.writer.add_images('generation_pure_vae', samples1, self.num_epoch)

        with torch.no_grad():
            train_1_loss = self.vae_1.loss(self.x_train).item()
            train_2_loss = self.vae_2.loss(self.x_train).item()
            train_3_loss = self.vae_3.loss(self.x_train).item()
            train_4_loss = self.vae_4.loss(self.x_train).item()
            train_5_loss = self.vae_5.loss(self.x_train).item()
            train_6_loss = self.vae_6.loss(self.x_train).item()
            train_7_loss = self.vae_7.loss(self.x_train).item()
            train_8_loss = self.vae_8.loss(self.x_train).item()

            test_1_loss = self.vae_1.loss(self.x_test).item()
            test_2_loss = self.vae_2.loss(self.x_test).item()
            test_3_loss = self.vae_3.loss(self.x_test).item()
            test_4_loss = self.vae_4.loss(self.x_test).item()
            test_5_loss = self.vae_5.loss(self.x_test).item()
            test_6_loss = self.vae_6.loss(self.x_test).item()
            test_7_loss = self.vae_7.loss(self.x_test).item()
            test_8_loss = self.vae_8.loss(self.x_test).item()

            self.results['train_1'].append(train_1_loss)
            self.results['train_2'].append(train_2_loss)
            self.results['train_3'].append(train_3_loss)
            self.results['train_4'].append(train_4_loss)
            self.results['train_5'].append(train_5_loss)
            self.results['train_6'].append(train_6_loss)
            self.results['train_7'].append(train_7_loss)
            self.results['train_8'].append(train_8_loss)

            self.results['test_1'].append(test_1_loss)
            self.results['test_2'].append(test_2_loss)
            self.results['test_3'].append(test_3_loss)
            self.results['test_4'].append(test_4_loss)
            self.results['test_5'].append(test_5_loss)
            self.results['test_6'].append(test_6_loss)
            self.results['test_7'].append(test_7_loss)
            self.results['test_8'].append(test_8_loss)

            self.writer.add_scalars('BPD', {
                'train_1': train_1_loss,
                'train_2': train_2_loss,
                'train_3': train_3_loss,
                'train_4': train_4_loss,
                'train_5': train_5_loss,
                'train_6': train_6_loss,
                'train_7': train_7_loss,
                'train_8': train_8_loss,

                'test_1': test_1_loss,
                'test_2': test_2_loss,
                'test_3': test_3_loss,
                'test_4': test_4_loss,
                'test_5': test_5_loss,
                'test_6': test_6_loss,
                'test_7': test_7_loss,
                'test_8': test_8_loss,
            }, self.num_epoch)

            # self.writer.add_scalars('rate', {
            #     'train_1': self.vae_1.rate(self.x_train).item(),
            #     'train_2': self.vae_2.rate(self.x_train).item(),
            #     'train_3': self.vae_3.rate(self.x_train).item(),
            #     'train_4': self.vae_4.rate(self.x_train).item(),
            #     'test_1': self.vae_1.rate(self.x_test).item(),
            #     'test_2': self.vae_2.rate(self.x_test).item(),
            #     'test_3': self.vae_3.rate(self.x_test).item(),
            #     'test_4': self.vae_4.rate(self.x_test).item(),
            # }, self.num_epoch)

            # self.writer.add_scalars('distortion', {
            #     'train_1': self.vae_1.distortion(self.x_train).item(),
            #     'train_2': self.vae_2.distortion(self.x_train).item(),
            #     'train_3': self.vae_3.distortion(self.x_train).item(),
            #     'train_4': self.vae_4.distortion(self.x_train).item(),
            #     'test_1': self.vae_1.distortion(self.x_test).item(),
            #     'test_2': self.vae_2.distortion(self.x_test).item(),
            #     'test_3': self.vae_3.distortion(self.x_test).item(),
            #     'test_4': self.vae_4.distortion(self.x_test).item(),
            # }, self.num_epoch)
            self.writer.close()
            

    def save(self):
        result_path = os.path.join(self.ckpt_dir, f'results_{self.num_epoch}')
        torch.save(self.results, result_path)

        state = {
            'encoder': self.vae_1.encoder.state_dict(),
            'decoder': self.vae_1.decoder.state_dict(),
            # 'classifier': self.classifier.state_dict(),
        }
        state_path = os.path.join(self.ckpt_dir, f'ckpts_vae1_{self.num_epoch}')
        torch.save(state, state_path)