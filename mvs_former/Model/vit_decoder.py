import torch.nn as nn

from mvs_former.Model.layers import AttentionFusionSimple


class VITDecoderStage4(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4, self).__init__()
        ch, vit_ch = args["out_ch"], args["vit_ch"]
        self.multi_scale_decoder = args.get("multi_scale_decoder", False)
        assert args["att_fusion"] is True
        self.attn = AttentionFusionSimple(vit_ch, ch * 4, args["nhead"])
        if self.multi_scale_decoder:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            )

            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
            )

            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch // 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch // 2, ch // 4, 4, stride=2, padding=1),
            )
        else:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch * 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            )
            self.decoder2 = nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
            )
            self.decoder3 = nn.Sequential(
                nn.BatchNorm2d(ch // 2),
                nn.GELU(),
                nn.ConvTranspose2d(ch // 2, ch // 4, 4, stride=2, padding=1),
            )

    def forward(self, x, att):
        x = self.attn(x, att)
        if self.multi_scale_decoder:
            out1 = self.decoder1(x)
            out2 = self.decoder2(x)
            out3 = self.decoder3(x)
        else:
            out1 = self.decoder1(x)
            out2 = self.decoder2(out1)
            out3 = self.decoder3(out2)

        return out1, out2, out3


class VITDecoderStage4Single(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4Single, self).__init__()
        ch, vit_ch = args["out_ch"], args["vit_ch"]
        assert args["att_fusion"] is True
        self.attn = AttentionFusionSimple(vit_ch, ch * 4, args["nhead"])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )

    def forward(self, x, att):
        x = self.attn(x, att)
        x = self.decoder(x)

        return x


class VITDecoderStage4NoAtt(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4NoAtt, self).__init__()
        ch, vit_ch = args["out_ch"], args["vit_ch"]
        self.down_sample = nn.Sequential(
            nn.Conv2d(vit_ch, ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )

    def forward(self, x, att=None):
        x = self.down_sample(x)
        x = self.decoder(x)

        return x
