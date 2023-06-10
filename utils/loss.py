import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def weight_train_loss(losses):
    """
    Computes weighted training losses based on the given dictionary of losses.

    Args:
        `losses` (dict): A dictionary containing the losses for different samples.

    Returns:
        `dict`: A dictionary containing the weighted average of losses.
    """
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
    """
    Computes the weighted test loss based on the given dictionary of losses.

    Args:
        `losses` (dict): A dictionary containing the losses for different samples.

    Returns:
        float: The weighted average of the losses.
    """
    tot_loss = 0
    weights = 0
    for k, v in losses.items():
        tot_loss = tot_loss + v['loss'] * v['num_samples']
        weights = weights + v['num_samples']
    return tot_loss / weights

class IW_MaxSquareloss(nn.Module):
    """
    Implements the IW_MaxSquareloss loss function for image segmentation.

    Args:
        `ignore_index` (int): The index value to be ignored during loss calculation.
        `ratio` (float): The ratio parameter for weight calculation.

    Returns:
        torch.Tensor: The calculated loss value.
    """
    requires_reduction = False

    def __init__(self, ignore_index=255, ratio=0.2, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.ratio = ratio

    def forward(self, pred, **kwargs):
        prob = F.softmax(pred, dim=1)
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=C + 1, min=-1,
                               max=C - 1).float()
            hist = hist[1:]
            weight = \
            (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(
                argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0).unsqueeze(1)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * C)
        return loss
    
class SelfTrainingLoss(nn.Module):
    """
    Implements the self-training loss for image segmentation.

    Args:
        `conf_th` (float): Confidence threshold for pseudo-labeling.
        `fraction` (float): Fraction of top-k pixels to be considered for pseudo-labeling.
        `ignore_index` (int): The index value to be ignored during loss calculation.
        `lambda_selftrain` (float): Weighting factor for the self-training loss.

    Returns:
        torch.Tensor: The calculated loss value.
    """
    requires_reduction = False

    def __init__(self, conf_th=0.9, fraction=0.66, ignore_index=255, lambda_selftrain=1, **kwargs):
        super().__init__()
        self.conf_th = conf_th
        self.fraction = fraction
        self.ignore_index = ignore_index
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def set_teacher(self, model):
        """
        Sets the teacher model for self-training.

        Args:
            model: The teacher model.

        Returns:
            None
        """
        self.teacher = model

    def get_image_mask(self, prob, pseudo_lab):
        """
        Generates the mask for pseudo-labeling based on confidence threshold and top-k fraction.

        Args:
            prob (torch.Tensor): The predicted probabilities.
            pseudo_lab (torch.Tensor): The pseudo labels.

        Returns:
            torch.Tensor: The generated mask.
        """
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
        """
        Generates the mask for pseudo-labeling for a batch of samples.

        Args:
            pred (torch.Tensor): The predicted probabilities.
            pseudo_lab (torch.Tensor): The pseudo labels.

        Returns:
            torch.Tensor: The generated batch mask.
        """
        b, _, _, _ = pred.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(F.softmax(pred, dim=1), pseudo_lab)], dim=0)
        return mask

    def get_pseudo_lab(self, pred, imgs=None, return_mask_fract=False, model=None, seg=False):
        """
        Generates pseudo-labels for the given predictions and images.

        Args:
            pred (torch.Tensor): The predicted probabilities.
            imgs (torch.Tensor): The input images.
            return_mask_fract (bool): Whether to return the mask fraction.
            model: The teacher model (optional).
            seg (bool): Whether the prediction is segmentation logits.

        Returns:
            torch.Tensor: The generated pseudo-labels.
        """
        teacher = self.teacher if model is None else model
        if teacher is not None:
            with torch.no_grad():
                try:
                    if seg:
                        logi = self.teacher(imgs)
                        logits = logi.logits
                        pred = nn.functional.interpolate(
                                logits, 
                                size=imgs.shape[-2:], 
                                mode="bilinear", 
                                align_corners=False
                        )
                    else:
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

    def forward(self, pred, imgs=None, seg=False):
        """
        Forward pass of the self-training loss.

        Args:
            pred (torch.Tensor): The predicted logits.
            imgs (torch.Tensor): The input images (optional).
            seg (bool): Whether the prediction is segmentation logits.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        pseudo_lab = self.get_pseudo_lab(pred, imgs, seg=seg)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean() * self.lambda_selftrain

class SelfTrainingLossEntropy(SelfTrainingLoss):
    """
    Self-training loss with entropy regularization.

    Args:
        lambda_entropy (float): Weighting factor for the entropy regularization.

    Inherits from:
        SelfTrainingLoss
    """
    def __init__(self, lambda_entropy=0.005, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy

    def cross_entropy(self, pred, imgs=None):
        """
        Calculates the cross-entropy loss.

        Args:
            pred (torch.Tensor): The predicted logits.
            imgs (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The calculated cross-entropy loss.
        """
        pseudo_lab = self.get_pseudo_lab(pred, imgs)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean()

    @staticmethod
    def entropy_loss(pred):
        """
        Calculates the entropy loss.

        Args:
            pred (torch.Tensor): The predicted logits.

        Returns:
            torch.Tensor: The calculated entropy loss.
        """
        p = F.softmax(pred, dim=1)
        logp = F.log_softmax(pred, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        ent = ent / 2.9444
        ent = ent ** 2.0 + 1e-8
        ent = ent ** 2.0
        return ent.mean()

    def forward(self, pred, imgs=None):
        """
        Forward pass of the self-training loss with entropy regularization.

        Args:
            pred (torch.Tensor): The predicted logits.
            imgs (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        ce_loss = self.cross_entropy(pred, imgs)
        entropy_loss = self.entropy_loss(pred)*self.lambda_entropy
        loss = ce_loss + entropy_loss
        return loss
    
class EntropyLoss(nn.Module):
    """
    Entropy loss.

    Args:
        lambda_entropy (float): Weighting factor for the entropy loss.
        num_classes (int): Number of classes.

    Inherits from:
        nn.Module
    """
    def __init__(self, lambda_entropy=0.005, num_classes=13, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.normalization_factor = self.__get_normalization_factor(num_classes)

    def __get_normalization_factor(self, num_classes):
        """
        Calculates the normalization factor for entropy loss.

        Args:
            num_classes (int): Number of classes.

        Returns:
            float: The normalization factor.
        """
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
        """
        Forward pass of the entropy loss.

        Args:
            pred (torch.Tensor): The predicted logits.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        loss = self.entropy_loss(pred)*self.lambda_entropy
        return loss
