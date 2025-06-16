# standard libraries
from typing import List, Tuple, Literal
from abc import ABC, abstractmethod

# third party libraries
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

# local packages
from utils.datatypes import Tensorlike


__all__ = [
    "FlexTransform",
    "FlexCompose",
    "NumpyTo3DTensor",
    "Resize",
    "RandomCrop",
    "VerticalFlip",
    "HorizontalFlip",
    "RandomFixedRotation",  # not Used
    "get_transform",
    "get_transforms_train",
    "get_transforms_test",
]


class FlexTransform(ABC):
    @abstractmethod
    def __call__(
        self, x: Tensorlike, y: Tensorlike | None
    ) -> Tensorlike | Tuple[Tensorlike, Tensorlike]:
        raise NotImplementedError("Must be implemented in child class.")


class FlexCompose(T.Compose):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if y is not None:
            for t in self.transforms:
                x, y = t(x, y)
            return x, y
        else:
            for t in self.transforms:
                x = t(x)
            return x

    def __getitem__(self, idx: int) -> FlexTransform:
        return self.transforms[idx]

    def __add__(self, other):
        if not isinstance(other, FlexCompose):
            raise ValueError(f"type {type(other)} is not allowed for {type(self)}")
        return FlexCompose(self.transforms + other.transforms)


class NumpyTo3DTensor(FlexTransform):
    def __call__(
        self, x: np.ndarray, y: np.ndarray | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        assert x.ndim <= 3
        while x.ndim < 3:
            x = np.expand_dims(x, axis=0)
        if y is not None:
            assert y.ndim <= 3
            while y.ndim < 3:
                y = np.expand_dims(y, axis=0)
        return (torch.tensor(x), torch.tensor(y)) if y is not None else torch.tensor(x)


class Resize(FlexTransform):
    def __init__(self, size: int, *args, **kwargs) -> None:
        self.t = T.Resize(size=(size, size), *args, **kwargs)

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return (self.t(x), self.t(y)) if y is not None else self.t(x)


class RandomCrop(FlexTransform):
    def __init__(self, *, size: int):
        self.size = (size, size)

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        params = T.RandomCrop.get_params(x, output_size=self.size)
        return (
            (F.crop(x, *params), F.crop(y, *params))
            if y is not None
            else F.crop(x, *params)
        )


class VerticalFlip(FlexTransform):
    """Randomly flip the image vertically with a probability."""

    def __init__(self, *, prob: float = 0.5):
        """Initialize with the probability of applying the flip.

        Args:
            training (bool): If True, also expecting target to be transformed.
            prob (float, optional): The probability of flipping the image. Default is 0.5.
        """
        self.prob = prob

    def __call__(self, x: Tensor, y: Tensor | None = None) -> Tuple[Tensor, ...]:
        """Randomly flip the image vertically with the given probability.

        Args:
            x (Tensor): Input tensor (image).
            y (Tensor): Target tensor (label).

        Returns:
            Tensor: Flipped (or unchanged) tensor.
        """
        if np.random.rand() < self.prob:
            return (x, y) if y is not None else x
        return (F.vflip(x), F.vflip(y)) if y is not None else F.vflip(x)


class HorizontalFlip(FlexTransform):
    """Randomly flip the image horizontally with a probability."""

    def __init__(self, *, prob: float = 0.5):
        """Initialize with the probability of applying the flip.

        Args:
            training (bool): If True, also expecting target to be transformed.
            prob (float, optional): The probability of flipping the image. Default is 0.5.
        """
        self.prob = prob

    def __call__(self, x: Tensor, y: Tensor | None = None) -> Tuple[Tensor, ...]:
        """Randomly flip the image horizontally with the given probability.

        Args:
            x (Tensor): Input tensor (image).
            y (Tensor): Target tensor (label).

        Returns:
            Tensor: Flipped (or unchanged) tensor.
        """
        if np.random.rand() < self.prob:
            return (x, y) if y is not None else x
        return (F.hflip(x), F.hflip(y)) if y is not None else F.hflip(x)


class RandomFixedRotation(FlexTransform):
    """Rotate by one of the given angles."""

    def __init__(self, *, angles: List[int] = [90, 180, 270]):
        self.angles = angles

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        angle = np.random.choice(self.angles)
        return (
            (F.rotate(x, angle), F.rotate(y, angle))
            if y is not None
            else F.rotate(x, angle)
        )


def get_transform(
    training: bool,
    *,
    input_size: int | None = None,
    resize_method: str | None = None,
) -> FlexCompose:
    if training:
        return get_transforms_train(input_size=input_size, resize_method=resize_method)
    else:
        return get_transforms_test()


def get_transforms_train(
    input_size: int | None = None,
    resize_method: Literal["crop", "resize"] | None = "crop",
) -> FlexCompose:
    if input_size is None or resize_method is None:
        return FlexCompose([NumpyTo3DTensor()])

    match resize_method.lower():
        case "crop":
            resize_transform = RandomCrop
        case "resize":
            resize_transform = Resize
        case _:
            raise ValueError(f"Unknown resizing method: {resize_method}.")

    return FlexCompose([NumpyTo3DTensor(), resize_transform(size=input_size)])


def get_transforms_test() -> FlexCompose:
    return FlexCompose([NumpyTo3DTensor()])


# %%################################################# Not Used ATM #####################################################


class TensorToNumpy(FlexTransform):

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        return (x.numpy(), y.numpy()) if y is not None else x.numpy()
