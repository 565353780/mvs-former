import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)
