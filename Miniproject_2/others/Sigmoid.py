from .MyModule import *
import torch


class Sigmoid(MyModule):

    def __init__(self) -> None:
        super().__init__()
        self.out = None

    def forward(self, input):
        out = 1.0 / (1.0 + torch.exp(-input))
        self.out = out
        return out

    def backward(self, gradwrtoutput):
        dout = gradwrtoutput * (1 - self.out) * self.out
        return dout

    def param(self):
        return [(None, None)]
