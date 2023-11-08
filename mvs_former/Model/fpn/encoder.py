import torch.nn as nn

from mvs_former.Model.conv.conv_2d import Conv2d


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type="BN"):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(
            feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type
        )

        self.downsample1 = Conv2d(
            feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type
        )
        self.conv10 = Conv2d(
            feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type
        )
        self.conv11 = Conv2d(
            feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type
        )

        self.downsample2 = Conv2d(
            feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type
        )
        self.conv20 = Conv2d(
            feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type
        )
        self.conv21 = Conv2d(
            feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type
        )

        self.downsample3 = Conv2d(
            feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type
        )
        self.conv30 = Conv2d(
            feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type
        )
        self.conv31 = Conv2d(
            feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type
        )

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]
