from torch.utils.tensorboard import SummaryWriter

from dataset import prepare_MNIST, prepare_coloredMNIST, prepare_small_MNIST, prepare_Binary_MNIST, prepare_OMNIGLOT, prepare_Silhouettes, prepare_CIFAR10, prepare_toy


class Base_Trainer():
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        self.config = config
        self.device = config['device']
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.num_epoch = 0
        self.num_grads = 0
        self._build_model()
        self._build_dataloader()
        self._build_optimizer()
        self.writer = SummaryWriter(log_dir)

    def _build_dataloader(self):
        if self.config['dataset_config']['data'] == 'MNIST':
            train_loader, test_loader = prepare_MNIST(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'smallMNIST':
            train_loader, test_loader = prepare_small_MNIST(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'Binary_MNIST':
            train_loader, test_loader = prepare_Binary_MNIST(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'OMNIGLOT':
            train_loader, test_loader = prepare_OMNIGLOT(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'Silhouettes':
            train_loader, test_loader = prepare_Silhouettes(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'CIFAR10':
            train_loader, test_loader = prepare_CIFAR10(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'coloredMNIST':
            train_loader, test_loader = prepare_coloredMNIST(**self.config['dataset_config'])
        elif self.config['dataset_config']['data'] == 'toy':
            train_loader, test_loader = prepare_toy(**self.config['dataset_config'])
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _build_model(self):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError