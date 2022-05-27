from .MyModule import *
import torch
from torch.nn.functional import fold, unfold
import random
import math


class Conv2d(MyModule):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 use_bias=True):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # # Initialize kernel weights
        k = groups / (in_channels * kernel_size * 2)
        self._kernel = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)

        # Sample from - sqrt k to sqrt k
        self._kernel = torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 2 * math.sqrt(k) - math.sqrt(k)
        self._kernel = self._kernel.view(out_channels, in_channels, kernel_size, kernel_size)

        self.dl_dw = torch.zeros(self._kernel.size())

        if use_bias:
            self.use_bias = True
            self._bias = torch.rand(self.out_c) * 2 * math.sqrt(k) - math.sqrt(k)
            self._bias[0] = random.random() * 2 * math.sqrt(k) - math.sqrt(k)
            self.dl_db = torch.zeros(self._bias.size())
        else:
            self.use_bias = None

    def forward(self, input_data):
        self.input = input_data
        # input data size : (B, C, H, W)
        # Batch x Channel x Height x Width
        input_shape = input_data.shape
        B, C_in, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # kernel size : (D, C, h, w)
        # Channel_out x Channel_out x height x width
        kernel_shape = self._kernel.shape
        C_out, C, h, w = kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]

        # The number of input channel equals to number of kernel's kernel
        assert C_in == C

        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold
        # unfold_input size variation: (Batch, C, H, W) -> (Batch, L, C * (h * w))
        unfold_input = unfold(input=input_data, kernel_size=self.k_size, dilation=self.dilation, padding=self.padding,
                              stride=self.stride)
        self.unfolded = unfold_input

        # unfold_input size variation by transpose: (Batch, C_in * (h * w), L) -> (B, L, C_in*h*w)
        # kernel size variation by view and transpose: (C_out, C_in, h, w) -> (C_out, C_in*h*w) -> (C_in*h*w, C_out) flatten kernel here!
        # matmul : (B, L, C_in*h*w) * (C_in*h*w, C_out) = (B, L, C_out)
        # matmul transpose : (B, C_out, L)
        unfold_output = unfold_input.transpose(1, 2).matmul(self._kernel.view(self._kernel.size(0), -1).t())

        if self.use_bias:
            unfold_output += self._bias.reshape(1, -1)
        unfold_output = unfold_output.transpose(1, 2)

        out_h = int((input_data.shape[2] + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1)
        out_w = int((input_data.shape[3] + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1)

        # out size variation by fold : (B, C_out, L) -> (B, C_out, out_h, out_w)
        # L = out_h * out_w
        out = unfold_output.reshape(unfold_output.shape[0], unfold_output.shape[1], out_h, out_w)

        return out

    def backward(self, grad):
        # grad : The gradient with respect to output
        # a = relu(x . w + b)
        # z = x . w + b
        # grad=dy_relu
        # dl_dw = dl_da * da_dz * dz_dw = grad * dz_dw (by chaine rule)
        # grad = output.size = (B, C_out, out_h, out_w)

        # flat_grad size : (Batch, C_out, out_h * out_w)
        flat_grad = grad.reshape(grad.shape[0], grad.shape[1], -1)
        # weight size : (C_out, C_in, h, w) -> (C_out, C_in*h*w)
        weight = self._kernel.view(self._kernel.size(0), -1)
        # weight_T : (C_in*h*w, C_out)
        weight_T = weight.t()

        # dl_dx : (B, C_in*h*w, out_h*out_w)
        dl_dx = weight_T.matmul(flat_grad)
        H = (grad.shape[-2] - 1) * self.stride + 1 + self.dilation * (self.k_size - 1) - 2 * self.padding
        W = (grad.shape[-1] - 1) * self.stride + 1 + self.dilation * (self.k_size - 1) - 2 * self.padding
#         dl_dx = fold(dl_dx, output_size=(H, W), kernel_size=1)
        dl_dx = fold(dl_dx, output_size=(self.input.size(2), self.input.size(3)), kernel_size=self.k_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        # (Batch, C, H, W) -> (Batch, L, C * (h * w))
        unfold_input = unfold(input=self.input, kernel_size=self.k_size, dilation=self.dilation, padding=self.padding,
                              stride=self.stride)
        unfold_input_transpose = unfold_input.transpose(1, 2)

        # unfold_input_transpose size: (Batch, out_h * out_w, C * (h * w) )
        # flat_grad size : (Batch, C_out, out_h * out_w)
        # dl_dw_unfold size : (Batch, c_out, C*(h*w))
        dl_dw_unfold = flat_grad.matmul(unfold_input_transpose)

        # dl_dw size : (C_out, C_in, h, w)
        dl_dw = dl_dw_unfold.sum(dim=0).reshape(dl_dw_unfold.shape[1], self.in_c, self.k_size, self.k_size)
        self.dl_dw.add_(dl_dw)
        self.dl_db.add_(flat_grad.sum(dim=(0, 2)))

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
