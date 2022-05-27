from .MyModule import *
from .NearestUpsamling import NearestUpsampling
from .Conv2d import Conv2d


class Upsampling(MyModule):
    '''
    We do a Nearest Neighbor Upsampling and a convolution here.
    This class is an alternate of transpose convolution.
    '''

    def __init__(self, scale_factor, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, use_bias=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.NN_ups = NearestUpsampling(scale_factor)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias)

    def forward(self, _input):
        """
        _input size : (B,C_in,H_in,W_in)
        size after NN Upsamling : (B, C_in, H_in*scale_factor, W_in*scale_factor)
        out size : (B, C_out, H_out, W_out)
        """
        NN_ups_f = self.NN_ups.forward(_input)
        res = self.conv.forward(NN_ups_f)
        return res

    def backward(self, grad):
        dl_dx = self.conv.backward(grad)
        return self.NN_ups.backward(dl_dx)

    def param(self):
        # We did not make any updates to the weights and biases
        return self.conv.param()
