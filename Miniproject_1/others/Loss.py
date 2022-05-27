import torch
from torch import nn

class L0Loss(nn.Module):

    def __init__(self, gamma=2, eps=1e-8):
        super(L0Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
        return torch.mean(loss)