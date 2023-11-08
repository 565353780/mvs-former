import torch.nn as nn

from mvs_former.Model.conv.conv_3d import Conv3d
from mvs_former.Model.conv.deconv_3d import Deconv3d


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer

        self.conv1 = Conv3d(in_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(
            base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1
        )
        self.conv9 = Deconv3d(
            base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1
        )
        self.conv11 = Deconv3d(
            base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1
        )

        if in_channels != base_channels:
            self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
        else:
            self.inner = nn.Identity()

        if self.last_layer:
            self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


class CostRegNet2D(nn.Module):
    def __init__(self, in_channels, base_channel=8):
        super(CostRegNet2D, self).__init__()
        self.conv1 = Conv3d(
            in_channels,
            base_channel * 2,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(
            base_channel * 2,
            base_channel * 4,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(
            base_channel * 4,
            base_channel * 8,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 8,
                base_channel * 4,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 4,
                base_channel * 2,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 2,
                base_channel,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class CostRegNet3D(nn.Module):
    def __init__(self, in_channels, base_channel=8):
        super(CostRegNet3D, self).__init__()
        self.conv1 = Conv3d(
            in_channels, base_channel * 2, kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(
            base_channel * 2,
            base_channel * 4,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(
            base_channel * 4,
            base_channel * 8,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 8,
                base_channel * 4,
                kernel_size=3,
                padding=1,
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 4,
                base_channel * 2,
                kernel_size=3,
                padding=1,
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 2,
                base_channel,
                kernel_size=3,
                padding=1,
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True),
        )

        if in_channels != base_channel:
            self.inner = nn.Conv3d(in_channels, base_channel, 1, 1)
        else:
            self.inner = nn.Identity()

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        x = self.prob(x)

        return x
