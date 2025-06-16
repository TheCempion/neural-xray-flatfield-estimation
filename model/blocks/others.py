# standard libraries

# third party libraries
from torch import Tensor
import torch.nn as nn

# local packages


__all__ = [
    "MaxPool",
    "AvgPool",
    "Reshape",
]


class MaxPool(nn.MaxPool2d):
    def __init__(self, **kwargs) -> None:
        super().__init__(2, 2, **kwargs)


class AvgPool(nn.AvgPool2d):
    def __init__(self, **kwargs) -> None:
        super().__init__(2, 2, **kwargs)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), *self.shape)
