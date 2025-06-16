# standard librariesr
from abc import abstractmethod

# third party libraries
import torch
import torch.nn as nn
import numpy as np

# local packages
from utils.datatypes import Pathlike
from utils.weights_init import weights_init
import utils.constants as const


__all__ = [
    "BaseModel",
]


class BaseModel(nn.Module):
    """Base class for all models"""

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def load_weights(self, model_weights: Pathlike | None) -> None:
        """Load only the weights of the saved model.

        Args:
            model_weights (Pathlike | None): Full path to the model weights.
        """
        if model_weights is not None:
            state_dict = torch.load(model_weights)[const.PREF_MODEL_STATE_DICT]

            # Check if 'module.' prefix is present and adjust accordingly
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("module.", ""): value
                    for key, value in state_dict.items()
                }

            # GAN Specific: need to check for legacy code, before introducing the nets as properties
            if any(
                key.startswith("netG.") or key.startswith("netD.")
                for key in state_dict.keys()
            ):
                state_dict = {f"_{key}": value for key, value in state_dict.items()}

            self.load_state_dict(state_dict=state_dict)

    def init_weights(self) -> None:
        self.apply(weights_init)
