import torch


def sigmoid(x, base=2.71828):
    return 1 / (1 + torch.pow(base, -x))
