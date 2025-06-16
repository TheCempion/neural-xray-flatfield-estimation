# standard libraries
from typing import Literal

# third party libraries
import torch
from torch import nn

# local packages
from base import BaseModel
import model.blocks as blocks
from utils.datatypes import Pathlike


# Implementation from PhaseGAN (https://github.com/pvilla/PhaseGAN), slightly adjusted to better fit my code
class PatchGANDiscriminator(BaseModel):
    """PatchGAN discriminator as in CycleGAN (Also PhaseGAN) and pix2pix(?)"""

    def __init__(
        self,
        input_channel: int,
        norm_layer: Literal["batch", "instance"] | None,
        ndf=64,
        n_layers=3,
        model_weights: Pathlike | None = None,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_channel (int): The number of channels in input images.
            ndf (int): The number of filters in the last conv layer.
            n_layers (int): The number of conv layers in the discriminator.
            model_weights (Pathlike, optional): Model weights. Defaults to None.
        """
        super().__init__()
        kernel_size = 4
        padding = 1
        layers = [
            blocks.Conv(
                input_channel,
                ndf,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                norm_layer=norm_layer,
                f_act=nn.LeakyReLU(0.2, True),
            )
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                blocks.Conv(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    norm_layer=norm_layer,
                    f_act=nn.LeakyReLU(0.2, True),
                )
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            blocks.Conv(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                norm_layer=norm_layer,
                f_act=nn.LeakyReLU(0.2, True),
            )
        ]
        # output 1 channel prediction map
        layers += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding
            )
        ]
        self.model = nn.Sequential(*layers)
        if model_weights is not None:
            self.load_weights(model_weights)
        else:
            self.init_weights()

    def forward(self, input):
        return torch.sigmoid(self.model(input))
