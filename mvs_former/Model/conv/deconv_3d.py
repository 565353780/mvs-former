import torch.nn as nn
import torch.nn.functional as F


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        init_method="xavier",
        **kwargs,
    ):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs,
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
