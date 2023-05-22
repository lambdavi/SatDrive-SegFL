import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def weight_train_loss(losses):
    fin_losses = {}
    c = list(losses.keys())[0]
    loss_names = list(losses[c]['loss'].keys())
    for l_name in loss_names:
        tot_loss = 0
        weights = 0
        for _, d in losses.items():
            try:
                tot_loss += d['loss'][l_name][-1] * d['num_samples']
                weights += d['num_samples']
            except:
                pass
        fin_losses[l_name] = tot_loss / weights
    return fin_losses


def weight_test_loss(losses):
    tot_loss = 0
    weights = 0
    for k, v in losses.items():
        tot_loss = tot_loss + v['loss'] * v['num_samples']
        weights = weights + v['num_samples']
    return tot_loss / weights

class SelfTrainingLoss(nn.Module):
    requires_reduction = False

    def __init__(self, conf_th=0.9, fraction=0.66, ignore_index=255, lambda_selftrain=1, **kwargs):
        super().__init__()
        self.conf_th = conf_th
        self.fraction = fraction
        self.ignore_index = ignore_index
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def set_teacher(self, model):
        self.teacher = model.cuda()

    def get_image_mask(self, prob, pseudo_lab):
        max_prob = prob.detach().clone().max(0)[0]
        mask_prob = max_prob > self.conf_th if 0. < self.conf_th < 1. else torch.zeros(max_prob.size(),
                                                                                       dtype=torch.bool).to(
            max_prob.device)
        mask_topk = torch.zeros(max_prob.size(), dtype=torch.bool).to(max_prob.device)
        if 0. < self.fraction < 1.:
            for c in pseudo_lab.unique():
                mask_c = pseudo_lab == c
                max_prob_c = max_prob.clone()
                max_prob_c[~mask_c] = 0
                _, idx_c = torch.topk(max_prob_c.flatten(), k=int(mask_c.sum() * self.fraction))
                mask_topk_c = torch.zeros_like(max_prob_c.flatten(), dtype=torch.bool)
                mask_topk_c[idx_c] = 1
                mask_c &= mask_topk_c.unflatten(dim=0, sizes=max_prob_c.size())
                mask_topk |= mask_c
        return mask_prob | mask_topk

    def get_batch_mask(self, pred, pseudo_lab):
        b, _, _, _ = pred.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(F.softmax(pred, dim=1), pseudo_lab)], dim=0)
        return mask

    def get_pseudo_lab(self, pred, imgs=None, return_mask_fract=False, model=None):
        teacher = self.teacher if model is None else model
        if teacher is not None:
            with torch.no_grad():
                try:
                    pred = teacher(imgs)['out']
                except:
                    pred = teacher(imgs)
                pseudo_lab = pred.detach().max(1)[1]
        else:
            pseudo_lab = pred.detach().max(1)[1]
        mask = self.get_batch_mask(pred, pseudo_lab)
        pseudo_lab[~mask] = self.ignore_index
        if return_mask_fract:
            return pseudo_lab, F.softmax(pred, dim=1), mask.sum() / mask.numel()
        return pseudo_lab

    def forward(self, pred, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean() * self.lambda_selftrain

class SelfTrainingLossEntropy(SelfTrainingLoss):
    def __init__(self, lambda_entropy=0.005, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy

    def cross_entropy(self, pred, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean()

    @staticmethod
    def entropy_loss(pred):
        p = F.softmax(pred, dim=1)
        logp = F.log_softmax(pred, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        ent = ent / 2.9444
        ent = ent ** 2.0 + 1e-8
        ent = ent ** 2.0
        return ent.mean()

    def forward(self, pred, imgs=None):
        ce_loss = self.cross_entropy(pred, imgs)
        entropy_loss = self.entropy_loss(pred)*self.lambda_entropy
        loss = ce_loss + entropy_loss
        return loss
    
class EntropyLoss(nn.Module):

    def __init__(self, lambda_entropy=0.005, num_classes=13, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.normalization_factor = self.__get_normalization_factor(num_classes)

    def __get_normalization_factor(self, num_classes):
        a = torch.ones((1, num_classes, 1, 1))
        a = 1 / num_classes * a
        p = F.softmax(a, dim=1)
        logp = F.log_softmax(a, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        return ent.item()

    def entropy_loss(self, pred):
        p = F.softmax(pred, dim=1)
        logp = F.log_softmax(pred, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        ent = ent / self.normalization_factor
        ent = ent ** 2.0 + 1e-8
        ent = ent ** 2.0
        return ent.mean()

    def forward(self, pred):
        loss = self.entropy_loss(pred)*self.lambda_entropy
        return loss

class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, pred_labels=None, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)

        if pred_labels is not None:
            loss = loss * pred_labels.float()
        if mask is not None:
            loss = loss * mask
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs