"""
https://github.com/huggingface/pytorch-image-models/tree/95ba90157fbbee293e8d10ac108ced2d9b990cbc#models
"""
import timm
import torch
import torch.nn as nn


class TinyNetA(nn.Module):
    def __init__(self, pretrain=True):
        super(TinyNetA, self).__init__()

        model = timm.create_model('tinynet_a', pretrained=pretrain)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # c16 H/2
        self.block1 = model.blocks[0]
        # c24 H/4
        self.block2 = model.blocks[1]
        # c40 H/8
        self.block3 = model.blocks[2]
        # c80 H/16
        self.block4 = model.blocks[3]
        # c112 H/16
        self.block5 = model.blocks[4]
        # c192 H/32
        self.block6 = model.blocks[5]
        # c320 H/32
        self.block7 = model.blocks[6]

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        out0 = self.block1(x)
        out1 = self.block2(out0)
        out2 = self.block3(out1)
        out3 = self.block5(self.block4(out2))
        out4 = self.block7(self.block6(out3))

        return out0, out1, out2, out3, out4

    @staticmethod
    def get_stage_channels():
        return [16, 24, 40, 112, 320]