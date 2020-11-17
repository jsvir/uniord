import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class RegressionLoss(nn.Module):
    def forward(self, output, target):
        return nn.MSELoss()(output.squeeze(), target)


class ClassificationLoss(nn.Module):
    def forward(self, output, target, weights):
        return nn.NLLLoss(weights)(output, target.long())


class UnimodalUniformOTLoss(nn.Module):
    """
    https://arxiv.org/pdf/1911.02475.pdf
    """

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes
        self.csi = 0.15
        self.e = 0.05
        self.tau = 1.

    def forward(self, output, target):
        output = torch.softmax(output, -1)
        ranks = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device).repeat(output.size(0), 1)
        target_repeated = target.unsqueeze(1).repeat(1, self.num_classes)
        p = torch.softmax(torch.exp(-torch.abs(ranks - target_repeated) / self.tau), dim=-1)
        target_onehot = torch.nn.functional.one_hot(target.unsqueeze(0).long(), self.num_classes).squeeze()
        uniform_term = 1. / self.num_classes
        soft_target = (1 - self.csi - self.e) * target_onehot + self.csi * p + self.e * uniform_term
        loss = nn.L1Loss()(torch.cumsum(output, dim=1), torch.cumsum(soft_target, dim=1))
        return loss


class DLDLLoss(nn.Module):
    """
    https://arxiv.org/pdf/1611.01731.pdf
    """

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes

    def forward(self, output, target):
        output = torch.nn.LogSoftmax(dim=-1)(output)
        normal_dist = Normal(torch.arange(0, self.num_classes).to(output.device), torch.ones(self.num_classes).to(output.device))
        soft_target = torch.softmax(normal_dist.log_prob(target.unsqueeze(1)).exp(), -1)
        return nn.KLDivLoss()(output, soft_target)


class SORDLoss(nn.Module):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes

    def forward(self, output, target):
        output = torch.nn.LogSoftmax()(output)
        ranks = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        soft_target = -nn.L1Loss(reduction='none')(target, ranks)  # should be of size N x num_classes
        soft_target = torch.softmax(soft_target, dim=-1)
        return nn.KLDivLoss(reduction='mean')(output, soft_target)


class OTLossSoft(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes

    def forward(self, output, target):
        ranks = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        soft_target = -nn.L1Loss(reduction='none')(target, ranks)  # should be of size N x num_classes
        soft_target = torch.softmax(soft_target, dim=-1)  # like in SORD
        loss = nn.L1Loss()(torch.cumsum(output, dim=1), torch.cumsum(soft_target, dim=1))  # like in Liu 2019
        return loss


class OTLoss(nn.Module):

    def __init__(self, n_classes, cost='linear'):
        super().__init__()
        self.num_classes = n_classes
        C0 = np.expand_dims(np.arange(n_classes), 0).repeat(n_classes, axis=0) / self.num_classes
        C1 = np.expand_dims(np.arange(n_classes), 1).repeat(n_classes, axis=1) / self.num_classes

        C = np.abs(C0 - C1)
        if cost == 'quadratic':
            C = C ** 2
        elif cost == 'linear':
            pass
        self.C = torch.tensor(C).float()

    def forward(self, output_probs, target_class):
        C = self.C.cuda(output_probs.device)
        costs = C[target_class.long()]
        transport_costs = torch.sum(costs * output_probs, dim=1)
        result = torch.mean(transport_costs)
        return result

