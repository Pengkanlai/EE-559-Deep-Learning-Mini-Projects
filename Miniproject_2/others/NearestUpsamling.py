from .MyModule import *
import torch
from torch import nn


class NearestUpsampling(MyModule):
    """
    Only Nearest Neighbor Upsamling here, no convolution.
    This class is functionally equivalent to nn.Upsamle(), only with parameter scale_factor.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, _input):
        """
        Input image shape : Batch, Channel, Height, Width (B,C_in,H,W)
        Output image shape : B, C_in, H*factor, W*factor
        """

        tmp = torch.repeat_interleave(_input, repeats=self.scale_factor, dim=2)
        res = torch.repeat_interleave(tmp, repeats=self.scale_factor, dim=3)
        return res

    def backward(self, grad):
        # out_grad size : B, C_in, h//scale_factor, w//scale_factor
        B, C_in, h, w = grad.shape
        out_h = h // self.scale_factor
        out_w = w // self.scale_factor
        dl_dx = torch.zeros(B, C_in, out_h, out_w)

        # We traverse the 2d Tensor and add up the derivatives of the corresponding node
        for i in range(h):
            for j in range(w):
                dl_dx[:, :, i // self.scale_factor, j // self.scale_factor] += grad[:, :, i, j]
        return dl_dx

    def param(self):
        return []


if __name__ == '__main__':
    a = torch.randn(3, 3, 32, 32)
    sc = 5

    testUpsample = NearestUpsampling(scale_factor=sc)
    torch.testing.assert_allclose(nn.Upsample(scale_factor=sc)(a),
                                  testUpsample(a))
