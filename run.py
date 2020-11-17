import argparse
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from logger import MetricsLogger
from modules.models_zoo import catalog
from modules import data_modules


def init_class(dataset, method):
    method_class = catalog[method]
    data_module = data_modules[dataset]
    return type(method + dataset, (method_class, data_module), {})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def run_with_config(config):
    seed_everything(0)
    model_class = init_class(config.dataset, config.method)
    metrics_logger = MetricsLogger(config.output_dir, model_class.__name__)
    for split in range(config.data_splits):
        for fold in range(config.data_folds):
            config.fold = fold
            config.split = split
            if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)
            config.logger = TensorBoardLogger(config.output_dir, name=model_class.__name__, log_graph=False)
            trainer = Trainer.from_argparse_args(config)
            lr_log_callback = LearningRateMonitor(logging_interval='epoch')
            trainer.callbacks.append(lr_log_callback)
            config.checkpoint_callback = None
            model = model_class(config)
            trainer.fit(model)
            trainer.test(model)
            metrics_logger.update(model.test_metrics)
    metrics_logger.write()


if __name__ == "__main__":
    args = parse_args()
    config = getattr(__import__(args.config.replace('/', '.').replace('.py', ''), fromlist=['Config']), 'Config')
    run_with_config(config)
