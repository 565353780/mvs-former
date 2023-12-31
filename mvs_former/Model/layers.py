import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class AttentionFusionSimple(nn.Module):
    def __init__(self, vit_ch, out_ch, nhead):
        super(AttentionFusionSimple, self).__init__()
        self.conv_l = nn.Sequential(
            nn.Conv2d(vit_ch + nhead, vit_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(vit_ch),
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(vit_ch, vit_ch, kernel_size=3, padding=1), nn.BatchNorm2d(vit_ch)
        )
        self.act = Swish()
        self.proj = nn.Conv2d(vit_ch, out_ch, kernel_size=1)

    def forward(self, x, att):
        # x:[B,C,H,W]; att:[B,nh,H,W]
        x1 = self.act(self.conv_l(torch.cat([x, att], dim=1)))
        att = torch.mean(att, dim=1, keepdim=True)
        x2 = self.act(self.conv_r(x * att))
        x = self.proj(x1 * x2)
        return x
