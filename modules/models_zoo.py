import torch.nn as nn
from modules import BasicTrainer, UnimodalUniformOTLoss, SORDLoss, DLDLLoss, UnimodalNormal, OTLoss, OTLossSoft, UnimodalBeta, UnimodalBinomial


class UNIORD(BasicTrainer):
    """
     Optimal Transport Loss and Unimodal OutputProbabilities Model
    """
    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalNormal(self.hparams.num_classes, ordinal_input_dim)
        self.loss_func = OTLoss(self.hparams.num_classes)


class UNIORDSoft(UNIORD):
    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config, ordinal_input_dim)
        self.loss_func = OTLossSoft(self.hparams.num_classes)


class UNIORDBetaSoft(UNIORD):
    def __init__(self, config):
        super().__init__(config)
        self.output_layers = UnimodalBeta(self.hparams.num_classes, 1000)
        self.loss_func = OTLossSoft(self.hparams.num_classes)


class SORD(BasicTrainer):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """
    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = SORDLoss(config.num_classes)


class Liu(BasicTrainer):
    """
    https://arxiv.org/pdf/1911.02475.pdf
    """
    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = UnimodalUniformOTLoss(config.num_classes)


class DLDL(BasicTrainer):
    """
    https://arxiv.org/pdf/1611.01731.pdf
    """
    def __init__(self, config):
        config.output_logits = True
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = nn.Linear(1000, config.num_classes)
        self.loss_func = DLDLLoss(config.num_classes)


class BeckhamBinomial(BasicTrainer):
    """
    http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
    According to the paper Binomial dist produces the best results
    """
    def __init__(self, config, ordinal_input_dim=1000):
        super().__init__(config)
        self.transition_layer = nn.ReLU(True)
        self.output_layers = UnimodalBinomial(self.hparams.num_classes, ordinal_input_dim)
        self.loss_func = OTLoss(self.hparams.num_classes)


implemented_list = [
    Liu, BeckhamBinomial, UNIORD, SORD, DLDL
]

catalog = {m.__name__: m for m in implemented_list}