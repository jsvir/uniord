import numpy as np
import torch
import torch.nn as nn
from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta


class UnimodalNormal(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(UnimodalNormal, self).__init__()
        self.num_classes = num_classes
        self.mu_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Tanh()
        )
        self.precision_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Softplus()
        )
        self.thresholds = np.arange(0, self.num_classes + 1) / self.num_classes * 2 - 1

    def calc_normal_output_probs(self, mu, sig):
        normal_dist = Normal(mu, sig)
        probs = torch.zeros(mu.size(0), self.num_classes).float().cuda(mu.device)
        for i in range(self.num_classes):
            probs[:, i] = (normal_dist.cdf(self.thresholds[i + 1]) - normal_dist.cdf(self.thresholds[i])).squeeze()
        # normalize probs
        norm_matrix = torch.diag(1. / torch.sum(probs, axis=1))
        return torch.matmul(norm_matrix, probs)

    def calc_output_probs(self, x):
        mu = self.mu_output(x)
        sig = 1 / self.precision_output(x).clamp(min=1e-2, max=1e2)
        output_probs = self.calc_normal_output_probs(mu=mu, sig=sig)
        return output_probs

    def forward(self, x):
        return self.calc_output_probs(x)


class UnimodalBinomial(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(UnimodalBinomial, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.tau = 1.0
        self.prob_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        success_values = torch.arange(0, self.num_classes, dtype=x.dtype, device=x.device, requires_grad=False).repeat(x.size(0), 1)
        succes_prob = self.prob_output(x).repeat(1, self.num_classes)
        scores = Binomial(total_count=self.num_classes - 1, probs=succes_prob).log_prob(success_values)
        scores = torch.softmax(scores / self.tau, dim=-1)
        return scores


class UnimodalBeta(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Softplus()
        )
        self.beta_output = nn.Sequential(
            nn.Linear(input_channels, 1),
            nn.Softplus()
        )
        self.thresholds = torch.tensor(np.arange(0, self.num_classes + 1) / self.num_classes).float().cuda().requires_grad_(False)
        self.device = None
        self.epsilon = 1e-5

    def calc_unnormalized_beta_cdf(self, b, alpha, beta, npts=100):
        bt = Beta(alpha.float(), beta.float())
        x = torch.linspace(0 + self.epsilon, b - self.epsilon, int(npts * b.cpu().numpy())).float().to(self.device)
        pdf = bt.log_prob(x).exp()
        dx = torch.tensor([1. / (npts * self.num_classes)]).float().to(self.device)
        P = pdf.sum(dim=1) * dx
        return P

    def calc_beta_output_probs(self, x):
        alpha = torch.clamp(torch.tensor(1.).to(self.device) + self.alpha_output(x), min=1, max=100)
        beta = torch.clamp(torch.tensor(1.).to(self.device) + self.beta_output(x), min=1, max=100)
        probs = torch.zeros(alpha.size(0), self.num_classes).float().to(self.device)
        for i in range(0, self.num_classes):
            cdf_next = self.calc_unnormalized_beta_cdf(self.thresholds[i + 1], alpha, beta)
            cdf_current = self.calc_unnormalized_beta_cdf(self.thresholds[i], alpha, beta)
            probs[:, i] = cdf_next - cdf_current
        norm_matrix = torch.diag(1. / probs.sum(dim=1))  # normalize probs
        return torch.matmul(norm_matrix, probs)

    def forward(self, x):
        self.device = x.device
        return self.calc_beta_output_probs(x)
