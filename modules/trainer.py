import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from modules.metrics import ExactAccuracy, MAE, OneOffAccuracy, EntropyRatio, Unimodality


class BasicTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.transforms = None
        self.backbone_model = None
        self.transition_layer = None
        self.weights = None
        self.output_layers = None
        self.loss_func = None
        self.data_splits = None

        for k, v in vars(config).items():
            if not k.startswith('_'): self.hparams.__setattr__(k, v)

        self.val_entropy_ratio = EntropyRatio(output_logits=config.output_logits)

        self.test_mae = MAE()
        self.test_accuracy = ExactAccuracy(config.num_classes)
        self.test_one_off_accuracy = OneOffAccuracy()
        self.test_entropy_ratio = EntropyRatio(output_logits=config.output_logits)
        self.test_unimodality = Unimodality(output_logits=config.output_logits)

        self.test_metrics = {}

    def forward(self, x):
        x = self.backbone_model(x)
        x = self.transition_layer(x)
        x = self.output_layers(x)
        return x

    def build_data_loader(self, task):
        if task == 'train':
            shuffle = True
            batch_size = self.config.train_batch_size
            workers = self.config.train_workers
        elif task == 'test':
            shuffle = False
            batch_size = self.config.test_batch_size
            workers = self.config.test_workers
        elif task == 'val':
            shuffle = False
            batch_size = self.config.val_batch_size
            workers = self.config.val_workers
        transformed_dataset = self.dataset_class(self.config, transforms.Compose(self.transforms_list[task]), task, self.data_splits)
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        if task == 'train':
            if not self.config.not_weigted:
                labels = transformed_dataset.get_labels()
                class_props = np.array([np.sum(labels == i) for i in range(self.config.num_classes)]) / len(labels)
                class_weights = 1 / class_props
                class_weights /= np.sum(class_weights)
                class_weights = torch.tensor(class_weights).float().cuda()
            else:
                class_weights = torch.ones(self.config.num_classes).float().cuda()
            return dataloader, class_weights
        return dataloader

    def train_dataloader(self):
        loader, self.weights = self.build_data_loader('train')
        return loader

    def val_dataloader(self):
        return self.build_data_loader('val')

    def test_dataloader(self):
        return self.build_data_loader('test')

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        self.val_entropy_ratio(y_hat, y)
        self.log_dict({
            'val_loss': self.loss_func(y_hat, y),
            'val_mae': MAE()(y_hat, y),
            'val_accuracy': ExactAccuracy(self.config.num_classes)(y_hat, y),
            'val_oneoff_accuracy': OneOffAccuracy()(y_hat, y),
            'val_unimodality': Unimodality(output_logits=self.config.output_logits)(y_hat),
        }, on_step=False, on_epoch=True)

    def validation_epoch_end(self, results):
        self.log_dict({
            'val_entropy_ratio': self.val_entropy_ratio.compute(),
        }, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        y_hat = self(x)
        self.test_entropy_ratio(y_hat, y)
        self.test_mae(y_hat, y)
        self.test_accuracy(y_hat, y)
        self.test_one_off_accuracy(y_hat, y)
        self.test_unimodality(y_hat)

    def test_epoch_end(self, results):
        self.test_metrics = {
            'test_mae': self.test_mae.compute(),
            'test_accuracy': self.test_accuracy.compute(),
            'test_oneoff_accuracy': self.test_one_off_accuracy.compute(),
            'test_unimodality': self.test_unimodality.compute(),
            'test_entropy_ratio': self.test_entropy_ratio.compute(),
        }
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(v) for v in self.config.lr_sched.split(',')]),
            'interval': 'epoch',
            'frequency': 1}
        return [optimizer], [scheduler]
