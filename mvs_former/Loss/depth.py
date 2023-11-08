import torch
import numpy as np
import torch.nn.functional as F


def DpethGradLoss(depth_grad_logits, depth_grad_gt, depth_grad_mask):
    B, H, W = depth_grad_logits.shape
    RB = B
    loss = 0.0
    for i in range(B):
        depth_grad_logits_ = depth_grad_logits[i]
        depth_grad_gt_ = depth_grad_gt[i]
        if torch.sum(depth_grad_gt_) == 0:
            RB = RB - 1
            continue
        depth_grad_mask_ = depth_grad_mask[i]
        pos_logits = depth_grad_logits_[depth_grad_gt_ == 1]
        depth_grad_mask_ = depth_grad_mask_ - depth_grad_gt_
        N = pos_logits.shape[0]
        neg_logits = depth_grad_logits_[depth_grad_mask_ == 1]
        shuffle_idx = np.arange(neg_logits.shape[0])
        np.random.shuffle(shuffle_idx)
        neg_logits = neg_logits[shuffle_idx[:N]]
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0
        )
        bloss = F.binary_cross_entropy_with_logits(
            logits, target=labels, reduction="mean"
        )
        loss += bloss

    loss = loss / (RB + 1e-7)
    loss = loss * 5

    return loss
