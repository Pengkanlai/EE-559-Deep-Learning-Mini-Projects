import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=3) -> None:
        super(UNet, self).__init__()

        self.enc_block_1 = nn.Sequential(
            # h, w unchanged in each conv layer due to padding 1
            nn.Conv2d(in_channels=in_c, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # POOL1
        )

        self.enc_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # POOL2
        )

        self.enc_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # POOL3
        )

        self.enc_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # POOL4
        )

        self.enc_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # POOL5
        )

        self.dec_block_1 = nn.Sequential(
            # h, w : 2 -> 1 * 2 = 2
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')  # UPSAMPLE5
        )

        self.dec_block_2 = nn.Sequential(
            # Concatenate output of POOL4
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')  # UPSAMPLE4
        )

        self.dec_block_3 = nn.Sequential(
            # Concatenate output of POOL3
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')  # UPSAMPLE3
        )

        self.dec_block_4 = nn.Sequential(
            # Concatenate output of POOL2
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')  # UPSAMPLE2
        )

        self.dec_block_5 = nn.Sequential(
            # Concatenate output of POOL1
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')  # UPSAMPLE1
        )

        self.dec_block_6 = nn.Sequential(
            # Concatenate input
            nn.Conv2d(in_channels=96 + in_c, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.apply(self._kaiming_weight_init)

    def forward(self, x):
        pool1 = self.enc_block_1(x)
        pool2 = self.enc_block_2(pool1)
        pool3 = self.enc_block_3(pool2)
        pool4 = self.enc_block_4(pool3)
        pool5 = self.enc_block_5(pool4)

        upsample5 = self.dec_block_1(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)

        upsample4 = self.dec_block_2(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)

        upsample3 = self.dec_block_3(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)

        upsample2 = self.dec_block_4(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)

        upsample1 = self.dec_block_5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        out = self.dec_block_6(concat1)
        return out

    def _kaiming_weight_init(self, module):
        """Kaiming Weight Initialisation"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight.data, nonlinearity='relu')