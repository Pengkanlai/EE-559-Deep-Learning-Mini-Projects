from .MyModule import *
import torch
from torch import nn


class ReLU(MyModule):

    def __init__(self, inplace=True) -> None:
        super().__init__()
        self.mask = None
        self.inplace = inplace
        self.input = None

    def forward(self, input):
        self.input = input
        self.mask = (input <= 0)
        if self.inplace:
            input[self.mask] = 0
            out = input
        else:
            out = input.clone().detach()
            out[self.mask] = 0
        return out

    def backward(self, gradwrtoutput):
        gradwrtoutput[self.mask] = 0
        dx = gradwrtoutput
        return dx

    def param(self):
        return [(None, None)]


if __name__ == '__main__':
    m = nn.ReLU()
    input = torch.randn(2)
    m_out = m(input)

    print(m_out.weight.grad)

    y = ReLU(inplace=True)
    y_out = y(input)

    torch.testing.assert_allclose(m_out, y_out)
