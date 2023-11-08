import torch
import torch.nn as nn
import torch.nn.functional as F

from mvs_former.Model.layers import Swish


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[3], kernel_size=1),
            nn.BatchNorm2d(feat_chs[3]),
            Swish(),
        )

        self.inner1 = nn.Conv2d(feat_chs[2], final_ch, 1)
        self.out1 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[2]),
            Swish(),
        )

        self.inner2 = nn.Conv2d(feat_chs[1], final_ch, 1)
        self.out2 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[1]),
            Swish(),
        )

        self.inner3 = nn.Conv2d(feat_chs[0], final_ch, 1)
        self.out3 = nn.Sequential(
            nn.Conv2d(final_ch, feat_chs[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[0]),
            Swish(),
        )

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner1(conv21)
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner2(conv11)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner3(conv01)
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


class FPNDecoderV2(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoderV2, self).__init__()
        self.out1 = nn.Sequential(
            nn.Conv2d(feat_chs[3] * 2, feat_chs[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[3]),
            Swish(),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(
                feat_chs[3], feat_chs[2], kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(feat_chs[2]),
            nn.ReLU(True),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(feat_chs[2] * 2, feat_chs[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[2]),
            Swish(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(
                feat_chs[2], feat_chs[1], kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(feat_chs[1]),
            nn.ReLU(True),
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(feat_chs[1] * 2, feat_chs[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[1]),
            Swish(),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(
                feat_chs[1], feat_chs[0], kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(feat_chs[0]),
            nn.ReLU(True),
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(feat_chs[0], feat_chs[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_chs[0]),
            Swish(),
        )

    def forward(self, conv01, conv11, conv21, conv31, vit1, vit2, vit3):
        out1 = self.out1(torch.cat([conv31, vit1], dim=1))  # [B,64,H/8,W/8]

        out2 = self.upsample1(out1) + conv21
        out2 = self.out2(torch.cat([out2, vit2], dim=1))  # [B,32,H/4,W/4]

        out3 = self.upsample2(out2) + conv11
        out3 = self.out3(torch.cat([out3, vit3], dim=1))  # [B,16,H/2,W/2]

        out4 = self.upsample3(out3) + conv01
        out4 = self.out4(out4)  # [B,8,H,W]

        return [out1, out2, out3, out4]
