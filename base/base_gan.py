# standard libraries
from typing import Any

# third party libraries
import torch
import torch.nn as nn

# local packages
from base import BaseModel


__all__ = [
    "BaseGAN",
]


class BaseGAN(BaseModel):
    def __init__(self, netG: nn.Module, netD: nn.Module):
        super().__init__()
        self._netG = netG
        self._netD = netD

    def train(self) -> None:
        self._netG.train()
        self._netD.train()
        self.training = True

    def eval(self) -> None:
        self._netG.eval()
        self._netD.eval()
        self.training = False

    def forward(self, *args, **kwargs) -> Any:
        if self.training:
            raise NotImplementedError(
                "During training, call the generator `netG` and discriminator `netD` separately."
            )
        else:
            return self.netG(*args, **kwargs)

    def generator(self, *args, **kwargs) -> torch.Tensor:
        """Wrapper to forward trough the generator."""
        return self.netG(*args, **kwargs)

    def discriminator(self, *args, **kwargs) -> torch.Tensor:
        """Wrapper to forward trough the discriminator."""
        return self.netD(*args, **kwargs)

    @property
    def netG(self) -> nn.Module:
        return self.module._netG if isinstance(self, nn.DataParallel) else self._netG

    @property
    def netD(self) -> nn.Module:
        return self.module._netD if isinstance(self, nn.DataParallel) else self._netD
