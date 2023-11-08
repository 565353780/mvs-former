import torch.nn as nn
import torch.nn.functional as F
from mvs_former.Model.conv.common import init_bn, init_uniform


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
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
        kernel_size,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        norm_type="IN",
        **kwargs,
    ):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == "IN":
            self.bn = (
                nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
            )
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
