# standard libraries
from typing import Tuple, List, Literal

# third party libraries
import torch.nn as nn
from torch import Tensor
import torch

# local packages
from .others import *


__all__ = [
    "Conv",
    "ConvTranspose",
    "ConvBlock",
    "ConvTransposeBlock",
    "UNetUp",
    "UNetDown",
    "get_norm_layer",
]


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_act: nn.Module,
        norm_layer: Literal["batch", "instance"] | None,
        padding: str = "same",
        kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_layer=norm_layer)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=(norm_layer is None),
            **kwargs,
        )
        self.norm = (
            norm_layer(out_channels, affine=True) if norm_layer is not None else None
        )
        self.f_act = f_act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.f_act(x)
        return x


class ConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        f_act: nn.Module,
        norm_layer: Literal["batch", "instance"] | None,
        **kwargs,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_layer=norm_layer)
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
            bias=(norm_layer is None),
            # padding=scale_factor // 2,
            # output_padding=scale_factor % 2,
            **kwargs,
        )

        self.norm = (
            norm_layer(out_channels, affine=True) if norm_layer is not None else None
        )
        self.f_act = f_act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.f_act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_act: nn.Module,
        num_convs: int = 3,
        downsample: bool = True,
        pooling: nn.Module = AvgPool(),
        **kwargs,
    ):
        assert num_convs >= 1
        super().__init__()
        kwargs["padding"] = kwargs.get("padding", "same")
        layers = [Conv(in_channels, out_channels, f_act=f_act, **kwargs)]
        layers += [
            Conv(out_channels, out_channels, f_act=f_act, **kwargs)
            for _ in range(num_convs - 1)
        ]
        if downsample:
            layers.append(pooling)
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_act: nn.Module,
        scale_factor: int = 2,
        num_convs: int = 3,
        **kwargs,
    ):
        super().__init__()

        layers = [
            ConvTranspose(
                in_channels,
                out_channels,
                f_act=f_act,
                scale_factor=scale_factor,
                norm_layer=kwargs["norm_layer"],
            )
        ]
        if num_convs - 1 > 0:
            layers.append(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    f_act=f_act,
                    downsample=False,
                    num_convs=num_convs - 1,
                    **kwargs,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_act: nn.Module,
        num_convs: int = 1,
        **kwargs,
    ):
        """UNet Upsample Block.

        Args:
            in_channels (int): Number of channels of concatenated tensor. Usually `in_channels = 2 * out_channels`.
            out_channels (int): Number of output channels. Usually, `in_channels // 2`.
            f_act (nn.Module): Activation function.
            num_convs (int, optional): Number of `Conv`-Layers after upsampling. Defaults to 1.
            kwargs: Keyword arguments for `Conv`.
        """
        super().__init__()
        if out_channels == in_channels:
            # assuming that the skip connection input has same dimensionality as the output of the layer
            channels_of_skip_input = out_channels + in_channels
        else:
            channels_of_skip_input = in_channels

        self.up = ConvTranspose(
            in_channels,
            out_channels,
            scale_factor=2,
            f_act=f_act,
            norm_layer=kwargs["norm_layer"],
        )  # , padding=1, output_padding=1  # TODO: Paddings
        self.conv_block = ConvBlock(
            channels_of_skip_input,
            out_channels,
            f_act,
            num_convs=num_convs,
            downsample=False,
            **kwargs,
        )

    def forward(self, x: Tensor, skip_x: Tensor):
        x = self.up(x)
        # # Padding in case the in- and output size does not match (for odd input dimensions)
        # diffY = skip_x.size()[2] - x.size()[2]
        # diffX = skip_x.size()[3] - x.size()[3]
        # x = F.pad(x, (diffX // 2, diffX - diffX // 2,
        #               diffY // 2, diffY - diffY // 2))
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv_block(x)
        return x


class UNetDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_act: nn.Module,
        num_convs: int = 1,
        pooling: nn.Module = AvgPool(),
        pretrained: List[nn.Module] = None,
        **kwargs,
    ):
        """UNet Downsample Block.

        Args:
            in_channels (int): Number of incoming channels.
            out_channels (int): Number of output channels.
            f_act (nn.Module): Activation function.
            num_convs (int, optional): Number of channels in the convolutional block. Defaults to 1.
            batch_norm (bool, optional): Flag to determine, wheter the convolutions should have batch-norm layers.
            pooling (nn.Module, optional): Pooling layer. Basically, only MaxPool or AvgPool. Defaults to AvgPool.
            pretrained (List[nn.Module], optional): List of weights for the convolutational layer in this block, e.g.
                        from VGG11. Defaults to None.
            kwargs: Keyword arguments for `Conv`.
        """
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            f_act=f_act,
            num_convs=num_convs,
            downsample=False,
            **kwargs,
        )
        if pretrained is not None:
            assert len(pretrained) == num_convs
            for pretrained, conv_block in zip(pretrained, self.conv.model):
                conv_block.conv.weight.data.copy_(pretrained.weight.data)
                if pretrained.bias is not None:
                    conv_block.conv.bias.data.copy_(pretrained.bias.data)
                else:
                    conv_block.conv.bias.data.fill_(0)
        self.pooling = pooling

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        skip_x = self.conv(x)
        x = self.pooling(skip_x)
        return x, skip_x


def get_norm_layer(norm_layer: Literal["batch", "instance"] | None) -> nn.Module:
    match norm_layer:
        case "batch":
            norm_layer = nn.BatchNorm2d
        case "instance":
            norm_layer = nn.InstanceNorm2d
        case None:
            pass
        case _:
            raise ValueError(f"Unknown norm layer: {norm_layer}")
    return norm_layer
