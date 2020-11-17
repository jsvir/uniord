import torchvision.transforms as transforms
from torchvision.models import resnet101, resnet18

from dataset.dataset import AdienceDataset, HCIDataset, DRDataset, AbaloneDataset
from modules import BasicTrainer


class DRModule(BasicTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.backbone_model = resnet18(pretrained=True)
        self.transforms_list = {
            'train': [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=359., translate=(0.05, 0.05), scale=(0.98, 1.02), fillcolor=(128, 128, 128)),
                transforms.RandomCrop(224),
                transforms.ToTensor()],
            'val': [
                transforms.RandomCrop(224),
                transforms.ToTensor()],
            'test': [
                transforms.CenterCrop(224),
                transforms.ToTensor()]
        }
        self.dataset_class = DRDataset
        self.save_hyperparameters()

        if self.config.create_h5f_dataset:
            for i in range(self.config.data_splits):
                dr_splits = DRDataset.split_dataset(self.config.preprocessed_images_dir, self.config.meta_data_pickle, seed=i)
                if self.config.use_h5f_dataset: DRDataset.create_h5f_dataset(dr_splits, self.config.h5f_dataset_path, i)
        else:
            self.data_splits = DRDataset.split_dataset(self.config.preprocessed_images_dir, self.config.meta_data_pickle, seed=self.config.split)


class HCIModule(BasicTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.backbone_model = resnet18(pretrained=True)
        self.transforms_list = {
            'train': [
                transforms.RandomAffine(degrees=359., translate=(0.2, 0.2), scale=(0.7, 1.4)),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
            'val': [
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
            'test': [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        }
        self.dataset_class = HCIDataset
        self.save_hyperparameters()
        self.data_splits = HCIDataset.split_dataset(self.config.data_images, self.config.split)


class AdienceModule(BasicTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.backbone_model = resnet101(pretrained=True)
        self.config = config
        self.transforms_list = {
            'train': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()],
            'val': [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor()],
            'test': [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
        }
        self.dataset_class = AdienceDataset
        self.save_hyperparameters()


data_modules = {
    'DR': DRModule,
    'HCI': HCIModule,
    'Adience': AdienceModule
}
