"""Models that can be used as building blocks in larger models. Model weights cannot be loaded directly."""

# standard libraries
from typing import Literal

# third party libraries
import torch.nn as nn
from torch import Tensor
from torchvision import models  # model zoo

# local packages
from base import BaseModel
import model.blocks.layers as blocks


__all__ = [
    "UNetVGG11",
    "UNetVGG11Pretrained",
    "UNetVGG16",
    "UNetVGG16Pretrained",
]


class UNetVGG11(BaseModel):
    def __init__(
        self,
        f_act: nn.Module,
        f_act_output: nn.Module,
        norm_layer: Literal["batch", "instance"] | None,
        input_channels: int,
        kernel_size: int,
        bn_output_layer: bool = False,
        kernel_size_output: int | None = None,
    ):
        super().__init__()
        shared_params = dict(
            kernel_size=kernel_size, f_act=f_act, norm_layer=norm_layer
        )

        self.down1 = blocks.UNetDown(
            input_channels, 64, **shared_params, num_convs=1, padding_mode="reflect"
        )
        self.down2 = blocks.UNetDown(64, 64, **shared_params, num_convs=2)
        self.down3 = blocks.UNetDown(64, 128, **shared_params, num_convs=2)
        self.down4 = blocks.UNetDown(128, 256, **shared_params, num_convs=2)
        self.down5 = blocks.UNetDown(256, 512, **shared_params, num_convs=2)

        self.latent = blocks.ConvBlock(
            512, 1024, **shared_params, num_convs=2, downsample=False
        )

        self.up1 = blocks.UNetUp(1024, 512, **shared_params, num_convs=2)
        self.up2 = blocks.UNetUp(512, 256, **shared_params, num_convs=2)
        self.up3 = blocks.UNetUp(256, 128, **shared_params, num_convs=2)
        self.up4 = blocks.UNetUp(128, 64, **shared_params, num_convs=2)
        self.up5 = blocks.UNetUp(64, 64, **shared_params, num_convs=2)
        norm_layer_output = norm_layer if bn_output_layer else None
        self.output = blocks.Conv(
            64,
            1,
            kernel_size=kernel_size_output or kernel_size,
            f_act=f_act_output,
            padding_mode="reflect",
            norm_layer=norm_layer_output,
        )

    def forward(self, x: Tensor) -> Tensor:
        down1_out, down1_skip = self.down1(x)
        down2_out, down2_skip = self.down2(down1_out)
        down3_out, down3_skip = self.down3(down2_out)
        down4_out, down4_skip = self.down4(down3_out)
        down5_out, down5_skip = self.down5(down4_out)

        latent = self.latent(down5_out)

        up1_out = self.up1(latent, down5_skip)
        up2_out = self.up2(up1_out, down4_skip)
        up3_out = self.up3(up2_out, down3_skip)
        up4_out = self.up4(up3_out, down2_skip)
        up5_out = self.up5(up4_out, down1_skip)
        out = self.output(up5_out)
        return out


class UNetVGG16(BaseModel):
    def __init__(
        self,
        f_act: nn.Module,
        f_act_output: nn.Module,
        norm_layer: Literal["batch", "instance"] | None,
        input_channels: int,
        kernel_size: int,
        bn_output_layer: bool = False,
    ):
        super().__init__()
        shared_params = dict(
            kernel_size=kernel_size, f_act=f_act, norm_layer=norm_layer
        )

        self.down1 = blocks.UNetDown(
            input_channels, 64, **shared_params, num_convs=2, padding_mode="reflect"
        )
        self.down2 = blocks.UNetDown(64, 128, **shared_params, num_convs=2)
        self.down3 = blocks.UNetDown(128, 256, **shared_params, num_convs=3)
        self.down4 = blocks.UNetDown(256, 512, **shared_params, num_convs=3)
        self.down5 = blocks.UNetDown(512, 512, **shared_params, num_convs=3)

        self.latent = blocks.ConvBlock(
            512, 1024, **shared_params, num_convs=3, downsample=False
        )

        self.up1 = blocks.UNetUp(1024, 512, **shared_params, num_convs=3)
        self.up2 = blocks.UNetUp(512, 512, **shared_params, num_convs=3)
        self.up3 = blocks.UNetUp(512, 256, **shared_params, num_convs=3)
        self.up4 = blocks.UNetUp(256, 128, **shared_params, num_convs=2)
        self.up5 = blocks.UNetUp(128, 64, **shared_params, num_convs=2)
        norm_layer_output = norm_layer if bn_output_layer else None
        self.output = blocks.Conv(
            64,
            1,
            kernel_size=kernel_size,
            f_act=f_act_output,
            padding_mode="reflect",
            norm_layer=norm_layer_output,
        )

    def forward(self, x: Tensor) -> Tensor:
        down1_out, down1_skip = self.down1(x)
        down2_out, down2_skip = self.down2(down1_out)
        down3_out, down3_skip = self.down3(down2_out)
        down4_out, down4_skip = self.down4(down3_out)
        down5_out, down5_skip = self.down5(down4_out)

        latent = self.latent(down5_out)

        up1_out = self.up1(latent, down5_skip)
        up2_out = self.up2(up1_out, down4_skip)
        up3_out = self.up3(up2_out, down3_skip)
        up4_out = self.up4(up3_out, down2_skip)
        up5_out = self.up5(up4_out, down1_skip)
        out = self.output(up5_out)
        return out


class UNetVGG11Pretrained(BaseModel):
    """UNet with pretraied weights from VGG11 (see ThesarusNet)."""

    def __init__(
        self,
        f_act: nn.Module,
        f_act_output: nn.Module,
        bn_output_layer: bool = False,
        norm_layer: nn.Module | None = None,  # either InstanceNorm2D or BatchNorm2D
    ):
        super().__init__()

        vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        shared_params = dict(kernel_size=3, f_act=f_act, norm_layer=norm_layer)

        self.down1 = blocks.UNetDown(
            3, 64, **shared_params, num_convs=1, pretrained=[vgg[0]]
        )
        self.down2 = blocks.UNetDown(
            64, 128, **shared_params, num_convs=1, pretrained=[vgg[3]]
        )
        self.down3 = blocks.UNetDown(
            128, 256, **shared_params, num_convs=2, pretrained=[vgg[6], vgg[8]]
        )
        self.down4 = blocks.UNetDown(
            256, 512, **shared_params, num_convs=2, pretrained=[vgg[11], vgg[13]]
        )
        self.down5 = blocks.UNetDown(
            512, 512, **shared_params, num_convs=2, pretrained=[vgg[16], vgg[18]]
        )

        self.latent = blocks.ConvBlock(
            512, 1024, **shared_params, num_convs=2, downsample=False
        )

        self.up1 = blocks.UNetUp(1024, 512, **shared_params, num_convs=2)
        self.up2 = blocks.UNetUp(512, 512, **shared_params, num_convs=2)
        self.up3 = blocks.UNetUp(512, 256, **shared_params, num_convs=2)
        self.up4 = blocks.UNetUp(256, 128, **shared_params, num_convs=2)
        self.up5 = blocks.UNetUp(128, 64, **shared_params, num_convs=2)
        norm_layer_output = norm_layer if bn_output_layer else None
        self.output = blocks.Conv(
            64,
            1,
            kernel_size=3,
            f_act=f_act_output,
            padding_mode="reflect",
            norm_layer=norm_layer_output,
        )

        self.max_pool = blocks.AvgPool()

    def forward(self, x: Tensor) -> Tensor:
        # slightly different forward pass compared to Model_0
        down1_out, down1_skip = self.down1(x)
        down2_out, down2_skip = self.down2(down1_out)
        down3_out, down3_skip = self.down3(down2_out)
        down4_out, down4_skip = self.down4(down3_out)
        down5_out, down5_skip = self.down5(down4_out)

        latent = self.latent(down5_out)

        up1_out = self.up1(latent, down5_skip)
        up2_out = self.up2(up1_out, down4_skip)
        up3_out = self.up3(up2_out, down3_skip)
        up4_out = self.up4(up3_out, down2_skip)
        up5_out = self.up5(up4_out, down1_skip)
        out = self.output(up5_out)
        return out


class UNetVGG16Pretrained(BaseModel):
    """UNet with pretraied weights from VGG16 (see ThesarusNet)."""

    def __init__(
        self,
        f_act: nn.Module,
        f_act_output: nn.Module,
        bn_output_layer: bool = False,
        norm_layer: nn.Module | None = None,  # either InstanceNorm2D or BatchNorm2D
    ):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        shared_params = dict(kernel_size=3, f_act=f_act)

        self.down1 = blocks.UNetDown(
            3, 64, **shared_params, num_convs=2, pretrained=[vgg[0], vgg[2]]
        )
        self.down2 = blocks.UNetDown(
            64, 128, **shared_params, num_convs=2, pretrained=[vgg[5], vgg[7]]
        )
        self.down3 = blocks.UNetDown(
            128, 256, **shared_params, num_convs=3, pretrained=[*vgg[10:15:2]]
        )
        self.down4 = blocks.UNetDown(
            256, 512, **shared_params, num_convs=3, pretrained=[*vgg[17:22:2]]
        )
        self.down5 = blocks.UNetDown(
            512, 512, **shared_params, num_convs=3, pretrained=[*vgg[24:29:2]]
        )

        self.latent = blocks.ConvBlock(
            512, 1024, **shared_params, num_convs=3, downsample=False
        )

        self.up1 = blocks.UNetUp(1024, 512, **shared_params, num_convs=3)
        self.up2 = blocks.UNetUp(512, 512, **shared_params, num_convs=3)
        self.up3 = blocks.UNetUp(512, 256, **shared_params, num_convs=3)
        self.up4 = blocks.UNetUp(256, 128, **shared_params, num_convs=2)
        self.up5 = blocks.UNetUp(128, 64, **shared_params, num_convs=2)
        norm_layer_output = norm_layer if bn_output_layer else None
        self.output = blocks.Conv(
            64,
            1,
            kernel_size=3,
            f_act=f_act_output,
            padding_mode="reflect",
            norm_layer=norm_layer_output,
        )

    def forward(self, x: Tensor) -> Tensor:
        # slightly different forward pass compared to Model_0
        down1_out, down1_skip = self.down1(x)
        down2_out, down2_skip = self.down2(down1_out)
        down3_out, down3_skip = self.down3(down2_out)
        down4_out, down4_skip = self.down4(down3_out)
        down5_out, down5_skip = self.down5(down4_out)

        latent = self.latent(down5_out)

        up1_out = self.up1(latent, down5_skip)
        up2_out = self.up2(up1_out, down4_skip)
        up3_out = self.up3(up2_out, down3_skip)
        up4_out = self.up4(up3_out, down2_skip)
        up5_out = self.up5(up4_out, down1_skip)
        out = self.output(up5_out)
        return out
