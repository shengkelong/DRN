import math
import numpy as np
import torch
import torch.nn as nn

        
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats
        
        if in_channels is None:
            in_channels = opt.n_colors
        
        if out_channels is None:
            out_channels = opt.n_colors

        
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class MSRB(nn.Module):
        def __init__(self,  n_feats=64):
            super(MSRB, self).__init__()

            kernel_size_1 = 3
            kernel_size_2 = 5
            self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1, groups=n_feats)
            self.pointwise_3_1 = nn.Conv2d(n_feats, n_feats, 1)
            self.conv_3_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_1, padding=1, groups=n_feats * 2)
            self.pointwise_3_2 = nn.Conv2d(n_feats * 2, n_feats * 2, 1)
            self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size_2, padding=2, groups=n_feats)
            self.pointwise_5_1 = nn.Conv2d(n_feats, n_feats, 1)
            self.conv_5_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_2, padding=2, groups=n_feats * 2)
            self.pointwise_5_2 = nn.Conv2d(n_feats * 2, n_feats * 2, 1)
            self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            input_1 = x
            output_3_1 = self.conv_3_1(input_1)
            output_3_1 = self.relu(self.pointwise_3_1(output_3_1))
            output_5_1 = self.conv_5_1(input_1)
            output_5_1 = self.relu(self.pointwise_5_1(output_5_1))
            input_2 = torch.cat([output_3_1, output_5_1], 1)
            output_3_2 = self.conv_3_2(input_2)
            output_3_2 = self.relu(self.pointwise_3_2(output_3_2))
            output_5_2 = self.conv_5_2(input_2)
            output_5_2 = self.relu(self.pointwise_5_2(output_5_2))
            input_3 = torch.cat([output_3_2, output_5_2], 1)
            output = self.confusion(input_3)
            output += x
            return output
class MSRN(nn.Module):
    def __init__(self):
        super(MSRN, self).__init__()

        n_feats = 64
        n_blocks = 8
        kernel_size = 3
        act = nn.ReLU(True)
        self.n_blocks = n_blocks
        # RGB mean for DIV2K
        # define head module
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB(n_feats=n_feats))

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding= kernel_size//2),
            ]
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = x
        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)
        x = self.tail(res)
        return x
