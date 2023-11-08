import torch.nn as nn
import torch.nn.functional as F

from mvs_former.Model.layers import Swish


class TwinDecoderStage4(nn.Module):
    def __init__(self, args):
        super(TwinDecoderStage4, self).__init__()
        ch, vit_chs = args["out_ch"], args["vit_ch"]
        ch = ch * 4  # 256
        self.upsampler0 = nn.Sequential(
            nn.ConvTranspose2d(vit_chs[-1], ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )  # 256
        self.inner1 = nn.Conv2d(vit_chs[-2], ch, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(True),
        )  # 256->128

        self.inner2 = nn.Conv2d(
            vit_chs[-3], ch // 2, kernel_size=1, stride=1, padding=0
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(ch // 2, ch // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 4),
            nn.ReLU(True),
        )  # 128->64

        self.inner3 = nn.Conv2d(
            vit_chs[-4], ch // 4, kernel_size=1, stride=1, padding=0
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(ch // 4, ch // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 4),
            Swish(),
        )  # 64->64

    def forward(self, x1, x2, x3, x4):  # in:[1/8 ~ 1/64] out:[1/2,1/4,1/8]
        x = self.smooth1(self.upsampler0(x4) + self.inner1(x3))  # 1/64->1/32
        x = self.smooth2(
            F.upsample(x, scale_factor=2, mode="bilinear", align_corners=False)
            + self.inner2(x2)
        )  # 1/32->1/16
        x = self.smooth3(
            F.upsample(x, scale_factor=2, mode="bilinear", align_corners=False)
            + self.inner3(x1)
        )  # 1/16->1/8

        return x


class TwinDecoderStage4V2(nn.Module):
    def __init__(self, args):
        super(TwinDecoderStage4V2, self).__init__()
        ch, vit_chs = args["out_ch"], args["vit_ch"]
        ch = ch * 4  # 256
        self.upsampler0 = nn.Sequential(
            nn.ConvTranspose2d(vit_chs[-1], ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )  # 256
        self.inner1 = nn.Conv2d(vit_chs[-2], ch, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.GELU(),
        )  # 256->128

        self.inner2 = nn.Conv2d(
            vit_chs[-3], ch // 2, kernel_size=1, stride=1, padding=0
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(ch // 2, ch // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 4),
            nn.GELU(),
        )  # 128->64

        self.inner3 = nn.Conv2d(
            vit_chs[-4], ch // 4, kernel_size=1, stride=1, padding=0
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(ch // 4, ch // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch // 4),
            nn.GELU(),
        )  # 64->64

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(ch // 4, ch // 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch // 8),
            nn.GELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ch // 8, ch // 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch // 16),
            nn.GELU(),
        )

    def forward(self, x1, x2, x3, x4):  # in:[1/8 ~ 1/64] out:[1/2,1/4,1/8]
        x = self.smooth1(self.upsampler0(x4) + self.inner1(x3))  # 1/64->1/32
        x = self.smooth2(
            F.upsample(x, scale_factor=2, mode="bilinear", align_corners=False)
            + self.inner2(x2)
        )  # 1/32->1/16
        out1 = self.smooth3(
            F.upsample(x, scale_factor=2, mode="bilinear", align_corners=False)
            + self.inner3(x1)
        )  # 1/16->1/8
        out2 = self.decoder1(out1)
        out3 = self.decoder2(out2)

        return out1, out2, out3
