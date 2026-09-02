import torch.nn as nn
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
# from mamba_p import Spa_SSM1
from mamba_channel import Spa_SSM1


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# ----------------------------------------
#       Spe_SSM Block
# ----------------------------------------
class ESSM(nn.Module):  # [bs, 28, 256, 256]
    def __init__(
            self,
            dim,
    ):
        super().__init__()
        self.Spa_SSM = Spa_SSM1(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim,  # Model dimension d_model  dimself.token_size * self.token_size * zipdim dim
            d_state=16,  # SSM state expansion factor # 64
            expand=2,  # Block expansion factor
            use_fast_path=False,
        )

        # 0522
        self.v_in_head = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, 1, 1, bias=False, groups=1),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),  # 0531
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, bias=False, groups=1),  # 0531
            Rearrange('b c h w -> b h w c'),
        )
        self.v_out_head = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),  # 0531
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            Rearrange('b c h w -> b h w c'),
        )

    def forward(self, x):
        """
        x_in:
        return out:
        """
        b, h, w, c = x.shape
        # 0522
        x0 = self.v_in_head(x)  # +x
        x = self.Spa_SSM(x0) + x0
        x = self.v_out_head(x) + x

        return x  # x


# -------------------------------------------------------
#           FeedForward
# ------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.FF = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            # GELU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            # GELU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.FF(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# -------------------------------------------------------
#           Cascade  Transformer
# ------------------------------------------------------
class CambaBlock(nn.Module):
    def __init__(
            self, dim, num_blocks=1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                ESSM(dim=dim),  # spectral
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        # mask = mask.permute(0, 2, 3, 1)
        for (attn1, ff) in self.blocks:
            x = attn1(x) + x  # Xspe
            x = ff(x) + x  # X_all
        out = x.permute(0, 3, 1, 2)
        return out


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class up_or(nn.Module):
    def __init__(
            self, dim_stage
    ):
        super().__init__()
        self.a1 = nn.Parameter(torch.ones(1))
        self.a2 = nn.Parameter(torch.ones(1))
        self.a3 = nn.Parameter(torch.ones(1))
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_stage, 4 * dim_stage, 3, 1, 1),
            nn.PixelShuffle(2))

        self.up2 = nn.Sequential(
            nn.Conv2d(dim_stage, 4 * dim_stage, 1, 1, 0),
            nn.PixelShuffle(2))
        self.up3 = nn.Sequential(
            nn.Conv2d(dim_stage, 4 * dim_stage, 5, 1, 2),
            nn.PixelShuffle(2))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        fea1 = self.up1(x)  # 2DCNN,stride=2,become downsample
        fea2 = self.up2(x)   # 2DCNN,stride=2,become downsample
        fea3 = self.up3(x)   # 2DCNN,stride=2,become downsample
        # fea = a1(fea1) * fea1 + a2(fea2) * fea2 + a3(fea3) * fea3+ self.a3*fea3
        fea = (self.a1 * fea1 + self.a2 * fea2 + self.a3 * fea3) / 3.0
        # fea = fea1 + fea2 + fea3
        # fea = self.up1(x)
        return fea

class SDL(nn.Module):
    def __init__(self, dim=64, band=128, scale=4, stage=2, num_blocks=None):  #
        super(SDL, self).__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.dim = dim
        self.stage = 2
        self.scale = scale
        self.upstage = int(math.log2(scale))
        self.Upsample = Upsample(scale=scale, num_feat=self.dim)
        # Input projection
        self.embedding = nn.Conv2d(band, self.dim, 3, 1, 1, bias=False)

        self.upcoder_layers = nn.ModuleList([])
        dim_stage = dim

        for i in range(int(math.log2(scale))):
            self.upcoder_layers.append(
                up_or(dim_stage))

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.encoder_layers.append(nn.ModuleList([
                CambaBlock(dim=dim_stage),  # dim_stage // dim
                # nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False),
                Downsample(scale=2, num_feat=self.dim),
                nn.Conv2d(dim_stage * 2, dim_stage, 1, 1, bias=False),
            ]))
        # dim_stage

        # Bottleneck
        self.bottleneck = CambaBlock(dim=dim_stage)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(nn.ModuleList([
                Upsample(scale=2, num_feat=dim_stage),
                # nn.ConvTranspose2d(dim_stage, dim_stage, stride=2, kernel_size=2, padding=0,
                #                    output_padding=0),
                nn.Conv2d(dim_stage * 2, dim_stage, 1, 1, bias=False),
                CambaBlock(dim=dim_stage)]))

        # Output projection
        self.mapping = nn.Conv2d(self.dim, band, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # LeakyReLU

    def forward(self, x):
        """
        x: [b,c,h,w] #have been masked from a pan
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # fea = self.Upsample(fea)
        # Upcoder
        Ufea_encoder = []
        for i, FeaUPSample in enumerate(self.upcoder_layers):
            # fea = CambaBlock(fea)  # casformer don't change maskCambaBlock,
            Ufea_encoder.append(fea)
            fea = FeaUPSample(fea)
            # fea = self.lrelu(fea)
        # fea = nn.functional.interpolate(input=fea, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x1 = fea
        # Encoder
        fea_encoder = []
        for i, (CambaBlock, FeaDownSample, Fution) in enumerate(self.encoder_layers):
            fea = CambaBlock(fea)  # casformer don't change mask
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)  # 2DCNN,stride=2,become downsample
            # if self.stage - 1 - 1 - i > -1:
            #     fea = Fution(torch.cat([fea, Ufea_encoder[self.stage - 1 - 1 - i]], dim=1))

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder  RGB_list[1-i]
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea + x1)  #

        return out
