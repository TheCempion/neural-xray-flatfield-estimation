# standard libraries
from typing import Literal, Dict, Any

# third party libraries
from torch import Tensor

# local packages
from base import BaseModel
import model.blocks as blocks
import model.activations as activations
from utils.datatypes import Pathlike


class UNet(BaseModel):
    def __init__(
        self,
        encoder_type: Literal[11, 16],
        pretrained: bool,
        model_weights: Pathlike | None,
        input_channels: int,
        kernel_size: int,
        f_act_output: str,
        norm_layer: Literal["batch", "instance"] | None,
        f_act: str = "LeakyReLU",
        kwargs_f_act_output: Dict[str, Any] = {},
        kwargs_f_act: Dict[str, Any] = {},
        bn_output_layer: bool = False,
    ):
        """Generic UNet class.

        Args:
            encoder_type (Literal[11, 16]): Which VGG-Network should be used: 11 -> VGG11, 16 -> VGG16.
            pretrained (bool): If True, use pretrained model parameters for the encoder, otherwise train from scratch.
                        Note, that if `model_weights` is given, those weights will be used.
            model_weights (Pathlike): Model weights for this architecture.
            input_channels (int): Number of input channels in the first layer.
            kernel_size (int): Kernel size used in every layer.
            f_act_output (str): Select the activation function in the output layer.
            f_act (str, optional): Select the activation function in the hidden layers. Defaults to "LeakyReLU".
            kwargs_f_act_output (Dict[str, Any], optional): Keyword arguments for the output activation. Defaults to {}.
            kwargs_f_act (Dict[str, Any], optional): Keyword arguments for the activation function. Defaults to {}.
            bn_output_layer (bool, optional): If True, add a BatchNorm layer to the output layer. Defaults to False.

        Raises:
            ValueError: Invalid encoder type. Must be 11 or 16.

        """
        super().__init__()

        f_act = getattr(activations, f_act)(**kwargs_f_act)
        f_act_output = getattr(activations, f_act_output)(**kwargs_f_act_output)

        # possibly add an input-layer; only if a pretrained encoder will be used.
        if input_channels != 3 and pretrained:
            self.input = self.input = blocks.Conv(
                input_channels,
                3,
                f_act,
                kernel_size=kernel_size,
                padding_mode="reflect",
                norm_layer=norm_layer,
            )
        else:
            self.input = None

        # select UNet Type
        if encoder_type == 11:
            if pretrained:
                self.unet = blocks.UNetVGG11Pretrained(
                    f_act=f_act,
                    f_act_output=f_act_output,
                    bn_output_layer=bn_output_layer,
                )
            else:
                self.unet = blocks.UNetVGG11(
                    f_act=f_act,
                    f_act_output=f_act_output,
                    input_channels=input_channels,
                    kernel_size=kernel_size,
                    bn_output_layer=bn_output_layer,
                    norm_layer=norm_layer,
                )
        elif encoder_type == 16:
            if pretrained:
                self.unet = blocks.UNetVGG16Pretrained(
                    f_act=f_act,
                    f_act_output=f_act_output,
                    bn_output_layer=bn_output_layer,
                )
            else:
                self.unet = blocks.UNetVGG16(
                    f_act=f_act,
                    f_act_output=f_act_output,
                    input_channels=input_channels,
                    kernel_size=kernel_size,
                    bn_output_layer=bn_output_layer,
                    norm_layer=norm_layer,
                )
        else:
            raise ValueError(
                f"`encoder_type` must be 11 for VGG11 or 16 for VGG16, but got: {encoder_type}"
            )

        if model_weights is not None:
            self.load_weights(model_weights)
        elif not pretrained:
            self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        if self.input is not None:
            x = self.input(x)
        out = self.unet(x)
        return out


# class MultiscaleUNet(BaseModel):
#     """UNet with pretraied weights from VGG11 (see ThesarusNet). Only makes sense for input_channels < 3."""

#     def __init__(self, f_act: nn.Module = activations.LeakyReLU(), input_channels: int = 1):
#         super().__init__()
#         self.input = blocks.Conv(input_channels, 3, f_act=f_act, kernel_size=3, padding_mode="reflect")
#         self.unet = blocks.UNetVGG11(f_act=f_act)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.input(x)  # 1 -> 3 channels
#         out = self.unet(x)
#         return out
