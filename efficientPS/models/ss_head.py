import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import DensePredictionCell, DepthSeparableConv2d, conv_1x1_bn


class SemanticSegmentationHead(nn.Module):
    def __init__(self, num_classes, activation=nn.GELU):
        super(SemanticSegmentationHead, self).__init__()

        self.p32_dpc = DensePredictionCell()
        self.p16_dpc = DensePredictionCell()

        self.p8_lsfe = self.make_large_scale_feature_extractor()
        self.p4_lsfe = self.make_large_scale_feature_extractor()

        self.p16_mc = self.make_mismatch_correction_module()
        self.p8_mc = self.make_mismatch_correction_module()

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.up16 = nn.Upsample(scale_factor=16, mode="bilinear")

        self.conv_final = conv_1x1_bn(128, num_classes)

    def make_large_scale_feature_extractor(self):
        convolutions = [
            DepthSeparableConv2d(256, 128),
            DepthSeparableConv2d(128, 128),
        ]

        return nn.Sequential(*convolutions)

    def make_mismatch_correction_module(self):
        convolutions = [
            DepthSeparableConv2d(128, 128),
            DepthSeparableConv2d(128, 128),
        ]

        convolutions += [nn.Upsample(scale_factor=2, mode="bilinear")]

        return nn.Sequential(*convolutions)

    def forward_(self, p32, p16, p8, p4):
        p32 = self.p32_dpc(p32)
        p16 = self.p16_dpc(p16)
        p8 = self.p8_lsfe(p8)
        p4 = self.p4_lsfe(p4)

        p32_p16 = p16 + self.up2(p32)

        p8 = p8 + self.p16_mc(p32_p16)
        p4 = p4 + self.p8_mc(p8)

        u_p32 = self.up16(p32)
        u_p16 = self.up8(p16)
        u_p8 = self.up4(p8)
        u_p4 = self.up2(p4)

        block = torch.cat([u_p32, u_p16, u_p8, u_p4], dim=1)
        return self.conv_final(self.up2(block))


if __name__ == "__main__":
    ssh = SemanticSegmentationHead(10)
    ssh.cuda()
    out = ssh(
        torch.rand(1, 256, 32, 64).cuda(),
        torch.rand(1, 256, 64, 128).cuda(),
        torch.rand(1, 256, 128, 256).cuda(),
        torch.rand(1, 256, 256, 512).cuda(),
    )
    print("out", out.shape)
