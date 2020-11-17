import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from torch.distributions import Categorical


class ExactAccuracy(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step, dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, preds):
        return torch.argmax(torch.softmax(preds, dim=-1), dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.to(preds.device)
        preds = self._input_format(preds)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class OneOffAccuracy(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step, dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, preds):
        return torch.argmax(torch.softmax(preds, dim=-1), dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.to(preds.device)
        preds = self._input_format(preds)
        self.correct += torch.sum(preds == target) + torch.sum(preds == target - 1) + torch.sum(preds == target + 1)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class MAE(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step, dist_sync_on_step)
        self.add_state("loss_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, preds):
        return torch.argmax(torch.softmax(preds, dim=-1), dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.to(preds.device)
        preds = self._input_format(preds)
        self.loss_sum += torch.nn.functional.l1_loss(preds.float(), target.float(), reduction='sum')
        self.total += target.numel()

    def compute(self):
        return self.loss_sum.float() / self.total


class EntropyRatio(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False, output_logits=False):
        super().__init__(compute_on_step, dist_sync_on_step)

        self.add_state("entropy_correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("entropy_incorrect", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.output_logits = output_logits

    def _entropy(self, pred: torch.Tensor):
        if pred.nelement() == 0: return torch.tensor(0).to(pred.device)
        return Categorical(probs=pred).entropy().mean()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.to(preds.device)
        if self.output_logits: preds = torch.softmax(preds, -1)
        preds_classes = torch.argmax(preds, dim=-1)
        self.entropy_correct += self._entropy(preds[preds_classes == target])
        self.entropy_incorrect += self._entropy(preds[preds_classes != target])

    def compute(self):
        return self.entropy_incorrect / self.entropy_correct


class Unimodality(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False, output_logits=False):
        super().__init__(compute_on_step, dist_sync_on_step)

        self.add_state("unimodal", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.output_logits = output_logits

    def _is_unimodal(self, pred: torch.Tensor):
        prob = pred.cpu().numpy()
        res = True
        argmax = np.argmax(prob)
        for i in range(argmax, 0, -1):
            res = res & (prob[i] >= prob[i - 1])
        for i in range(argmax, len(prob) - 1):
            res = res & (prob[i] >= prob[i + 1])
        return res

    def update(self, preds: torch.Tensor):
        self.to(preds.device)
        if self.output_logits: preds = torch.softmax(preds, -1)
        for i in range(preds.size(0)):
            self.unimodal += torch.tensor(self._is_unimodal(preds[i])).to(preds.device)
        self.total += preds.size(0)

    def compute(self):
        return self.unimodal.float() / self.total
