from .MyModule import *
from torch.nn.functional import fold, unfold
import torch
import math
import random


class TransposedConv2d(MyModule):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=0, groups=1,
                 output_padding=0, use_bias=True) -> None:
        """
        Original version of Transposed Conv2d as in PyTorch
        """
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # # Initialize kernel weights
        k = groups / (in_channels * kernel_size * 2)

        # Sample from - sqrt k to sqrt k
        # In tc2d, the position of in_channels and out_channels is switched
        self._kernel = torch.rand(in_channels, out_channels, kernel_size, kernel_size) * 2 * math.sqrt(k) - math.sqrt(k)
        self.dl_dw = torch.zeros(self._kernel.size())

        if use_bias:
            if use_bias:
                self.use_bias = True
                self._bias = torch.rand(self.out_c) * 2 * math.sqrt(k) - math.sqrt(k)
                self._bias[0] = random.random() * 2 * math.sqrt(k) - math.sqrt(k)
                self.dl_db = torch.zeros(self._bias.size())
            else:
                self.use_bias = None

        self.input_size = None
        self.flat_input = None

    def forward(self, _input):

        # Flatten input into (1,-1), flatten and transpose weight into (n, c, -1)
        # Then multiply these two
        self.input_size = _input.size()
        self.flat_input = _input.reshape(_input.shape[0], _input.shape[1], -1)

        flat_weight = self._kernel.reshape(self._kernel.size(0), -1).T
        up_input = flat_weight.matmul(self.flat_input)

        # Simplified version as we assume dilation=1
        H = int(((_input.shape[-2] - 1) * self.stride + self.output_padding + self.kernel_size - 2 * self.padding))
        W = int(((_input.shape[-1] - 1) * self.stride + self.output_padding + self.kernel_size - 2 * self.padding))

        f_out = fold(up_input, output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride,
                     padding=self.padding)

        if self.use_bias:
            f_out = f_out.add(self._bias.reshape(-1, 1, 1))

        del _input
        return f_out

    def backward(self, grad):

        grad_col = unfold(input=grad, kernel_size=self.kernel_size,
                          padding=self.padding, stride=self.stride)

        dl_dx = self._kernel.reshape(self._kernel.size(0), -1).matmul(grad_col)
        dl_dx = dl_dx.reshape(*self.input_size)

        dl_dw = grad_col.matmul(self.flat_input.transpose(1, 2)).sum(dim=0).T.reshape(*self._kernel.size())

        # flat_grad size : (Batch, C_out, out_h * out_w)
        flat_grad = grad.reshape(grad.shape[0], grad.shape[1], -1)
        dl_db = flat_grad.sum(dim=(0, 2))

        # Update
        self.dl_dw += dl_dw
        if self.use_bias:
            self.dl_db += dl_db

        return dl_dx

    @property
    def weight(self):
        return self._kernel

    @weight.setter
    def weight(self, w):
        self._kernel = w

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, b):
        self._bias = b

    def param(self):
        return [(self._kernel, self.dl_dw), (self._bias, self.dl_db)]


if __name__ == '__main__':
    from torch import nn
    from MSE import MSE

    n, c, h, w = 2, 3, 9, 9
    in_c, out_c = 3, 8
    kernel_size = 3
    padding = 1
    stride = 2

    test_input = torch.arange(n * c * h * w).reshape(n, c, h, w).float()

    # Test forward
    m = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride=stride, padding=padding)
    tc2d = TransposedConv2d(in_c, out_c, kernel_size, stride, padding)
    tc2d.weight = m.weight
    tc2d.bias = m.bias
    torch.testing.assert_allclose(tc2d(test_input), m(test_input))

    # Test forward output_padding
    n, c, h, w = 2, 3, 8, 8
    in_c, out_c = 3, 8
    kernel_size = 3
    padding = 1
    stride = 2
    output_padding = 1

    m = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding)
    tc2d = TransposedConv2d(in_c, out_c, kernel_size, stride, padding, output_padding=output_padding)
    tc2d.weight = m.weight
    tc2d.bias = m.bias
    torch.testing.assert_allclose(tc2d(test_input), m(test_input))


    # Test backward
    n, c, h, w = 3, 24, 32, 32
    in_c, out_c = 24, 3
    kernel_size = 4
    padding = 0
    stride = 2
    output_padding = 0

    test_input = torch.randn(n * c * h * w).reshape(n, c, h, w).float()
    test_input.requires_grad_()

    test_h = int(((test_input.shape[-2] - 1) * stride + output_padding + kernel_size - 2 * padding))
    test_w = int(((test_input.shape[-1] - 1) * stride + output_padding + kernel_size - 2 * padding))

    test_out = torch.randn(n * out_c * test_h * test_w).reshape(n, out_c, test_h, test_w).float()

    m = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding)
    tc2d = TransposedConv2d(in_c, out_c, kernel_size, stride, padding, output_padding=output_padding)
    tc2d.weight = m.weight
    tc2d.bias = m.bias

    out = m(test_input)
    loss = nn.MSELoss()
    l = loss(out, test_out)
    l.backward()

    pred = tc2d(test_input)
    ll = MSE()
    gd = ll(pred, test_out)
    test_grad = ll.backward()

    x_grad = tc2d.backward(test_grad)

    # test dl_dx
    torch.testing.assert_allclose(test_input.grad, x_grad)

    # test dl_db
    torch.testing.assert_allclose(m.bias.grad, tc2d.param()[1][1])

    # test dl_dw
    torch.testing.assert_allclose(m.weight.grad, tc2d.param()[0][1])


