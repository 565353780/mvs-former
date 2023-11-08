import torch
import torch.nn.functional as F


def focal_loss(preds, labels, gamma=2.0):  # [B,D,H,W], [B,H,W]
    labels = labels.unsqueeze(1)
    preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
    preds_softmax = torch.exp(preds_logsoft)  # softmax

    preds_softmax = preds_softmax.gather(1, labels)
    preds_logsoft = preds_logsoft.gather(1, labels)
    loss = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)

    return loss
