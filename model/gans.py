# standard libraries
from typing import Literal, Dict, Any

# third party libraries

# local packages
from base import BaseGAN
import model.models as models
from model.discriminators import PatchGANDiscriminator
from utils.datatypes import Pathlike


class UNetPatchGAN(BaseGAN):
    def __init__(
        self,
        encoder_type: Literal[11, 16],
        pretrained: bool,
        norm_layer: Literal["batch", "instance"],
        model_weights: Pathlike | None = None,
        model_weights_generator: Pathlike | None = None,
        model_weights_discriminator: Pathlike | None = None,
        input_channels: int = 1,
        kernel_size: int = 3,
        f_act_output: str = "ReLU",
        kwargs_patch_gan: Dict[str, Any] = {},
        **kwargs,
    ):
        """GAN architecture with UNet-shaped encoder and PatchGAN discriminator.

        Args:
            encoder_type (Literal[11, 16]): Which VGG-Network should be used: 11 -> VGG11, 16 -> VGG16.
            pretrained (bool): If True, use pretrained model parameters for the encoder, otherwise train from scratch.
                        Note, that if `model_weights` is given, those weights will be used.
            model_weights (Pathlike | None, optional): Model weights for this architecture. Defaults to None.
            model_weights_generator (Pathlike | None, optional): Model weights for the generator. Defaults to None.
            model_weights_discriminator (Pathlike | None, optional): Model weights for the discriminator.
                        Defaults to None.
            input_channels (int, optional): Number of input channels in the first layer. If `input_channels > 1`, the
                        discriminator also accepts as many channels (conditioned). Defaults to 1.
            kernel_size (int, optional): Kernel size used in every layer. Defaults to 3.
            f_act_output (str, optional): Select the activation function in the output layer. Defaults to "ReLU".
        """
        netG = models.UNet(
            encoder_type=encoder_type,
            pretrained=pretrained,
            model_weights=model_weights_generator,
            input_channels=input_channels,
            kernel_size=kernel_size,
            f_act_output=f_act_output,
            norm_layer=norm_layer,
            **kwargs,
        )
        netD = PatchGANDiscriminator(
            input_channel=input_channels,
            model_weights=model_weights_discriminator,
            norm_layer=norm_layer,
            **kwargs_patch_gan,
        )
        super().__init__(netG=netG, netD=netD)
        if model_weights is not None:
            self.load_weights(model_weights=model_weights)
