# standard libraries
from abc import ABC
from typing import Tuple, Literal, Callable

# third party libraries
import torch
from torch import Tensor

# local packages
import utils.constants as const


__all__ = [
    "DataNormalizer",
    "DummyNormalizer",
    "NormalizeByMax",
    "NormalizeMinMax",
    "NormalizeMinMaxConditioned",
    "NormalizeByMaxConditioned",
    "NormalizeByMaxBatch",
    "NormalizeMinMaxBatch",
    "NormalizeLogarithmic",
    "NormalizeSqrt",
]


class DataNormalizer(ABC):
    def __init__(self):
        pass

    def __call__(self, x: Tensor, y: Tensor | None) -> Tensor:
        pass

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def _condition_scaling(
        self,
        normalizer: Callable[
            [Tensor, Tensor], Tuple[Tensor, Tensor]
        ],  # the __call__ function
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if len(x[1]) <= 1 or y is None:
            return normalizer(x, y)

        conditions = x[
            :, 1:, :, :
        ].clone()  # BxCxHxW -> Bx(C-1)xHxW; should also be y[:, 1:, :, :] if GAN Training
        target_cloned = y[:, 0, :, :].clone()  # BxCxHxW -> BxHxW
        for i in range(1, len(x[1])):
            # fill with dummy values to not include condition in normalization
            x[:, i, :, :] = target_cloned
            if len(y[1]) > 1:  # Multichannel GAN Training
                y[:, i, :, :] = target_cloned

        normalized_x, normalized_y = normalizer(
            x, y
        )  # solely normalize H and FF, without considering C

        # Compute ymin and ymax for each sample in the batch along H and W
        ymin = normalized_y[:, :1, :, :].amin(
            dim=(2, 3), keepdim=True
        )  # Shape: (B, 1, 1, 1)
        ymax = normalized_y[:, :1, :, :].amax(
            dim=(2, 3), keepdim=True
        )  # Shape: (B, 1, 1, 1)
        cmin = conditions.amin(dim=(2, 3), keepdim=True)  # Shape: (B, (C-1), 1, 1)
        cmax = conditions.amax(dim=(2, 3), keepdim=True)  # Shape: (B, (C-1), 1, 1)

        # Generate new lower and upper bounds for each sample in the batch
        new_lower = torch.clamp(
            torch.normal(mean=ymin, std=const.FF_STD_MIN), min=0.0
        )  # Shape: (B, 1, 1)„
        new_upper = torch.clamp(
            torch.normal(mean=ymax, std=const.FF_STD_MAX), max=1.0
        )  # Shape: (B, 1, 1)
        new_lower = torch.where(
            (ymax > ymin) & (new_upper > new_lower), new_lower, ymin
        )
        new_upper = torch.where(
            (ymax > ymin) & (new_upper > new_lower), new_upper, ymax
        )

        new_condition = new_lower + (new_upper - new_lower) / (cmax - cmin) * (
            conditions - cmin
        )
        normalized_x[:, 1:, :, :] = new_condition
        if len(y[1]) > 1:  # multichannel GAN training
            normalized_y[:, 1:, :, :] = new_condition
        return normalized_x, normalized_y


class DummyNormalizer(DataNormalizer):
    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return x, y

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space.
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        return x


class NormalizeByMax(DataNormalizer):
    """Normalize all values to lie in [x.min() / x.max(), 1]."""

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        # for stability reasons, also consider y.max(), even though in general x >= y
        if y is not None:
            max_vals = torch.max(
                x.amax(dim=(1, 2, 3), keepdim=True), y.amax(dim=(1, 2, 3), keepdim=True)
            )
            return x / max_vals, y / max_vals
        else:
            max_vals = x.amax(dim=(1, 2, 3), keepdim=True)
            return x / max_vals

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space.
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        return x * y.amax(dim=(1, 2, 3), keepdim=True)


class NormalizeMinMax(DataNormalizer):
    """Normalize all values to lie in [0, 1]."""

    def __init__(self):
        self.normalize = lambda x, low, high: torch.clip(
            (x - low) / (high - low + const.EPS), min=0, max=1
        )

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        # for stability reasons, also consider y.max() and x.min(), even though in general x >= y
        # retransform is not affected
        if y is not None:
            min_vals = torch.min(
                x.amin(dim=(1, 2, 3), keepdim=True), y.amin(dim=(1, 2, 3), keepdim=True)
            )
            max_vals = torch.max(
                x.amax(dim=(1, 2, 3), keepdim=True), y.amax(dim=(1, 2, 3), keepdim=True)
            )
            normalized_x = self.normalize(x, min_vals, max_vals)
            normalized_y = self.normalize(y, min_vals, max_vals)
            return normalized_x, normalized_y
        else:
            min_vals = x.amin(dim=(1, 2, 3), keepdim=True)
            max_vals = x.amax(dim=(1, 2, 3), keepdim=True)
            return self.normalize(x, min_vals, max_vals)

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space.
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        min_vals = y.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = y.amax(dim=(1, 2, 3), keepdim=True)
        return x * (max_vals - min_vals) + min_vals


class NormalizeMinMaxConditioned(NormalizeMinMax):
    """Normalize all values to lie in [0, 1]. Condition will be scaled to be close to the target."""

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return self._condition_scaling(super().__call__, x, y)


class NormalizeByMaxConditioned(NormalizeByMax):
    """Normalize all values to lie in [0, 1]. Condition will be scaled to be close to the target."""

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return self._condition_scaling(super().__call__, x, y)


################################################## LEGACY CODE BELOW ###################################################


class NormalizeByMaxBatch(DataNormalizer):
    """Normalize all values to lie in [x.min() / x.max(), 1]."""

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        # for stability reasons, also consider y.max(), even though in general x >= y
        max_val = max(x.max(), y.max()) if y is not None else x.max()
        return (x / max_val, y / max_val) if y is not None else x / max_val

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space.
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        return x * y.max()


class NormalizeMinMaxBatch(DataNormalizer):
    """Normalize all values to lie in [0, 1]."""

    def __init__(self):
        self.normalize = lambda x, low, high: torch.clip(
            (x - low) / (high - low), min=0, max=1
        )

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        # for stability reasons, also consider y.max() and x.min(), even though in general x >= y
        # retransform is not affected
        max_val = max(x.max(), y.max()) if y is not None else x.max()
        min_val = min(x.min(), y.min()) if y is not None else x.min()
        return (
            (self.normalize(x, min_val, max_val), self.normalize(y, min_val, max_val))
            if y is not None
            else self.normalize(x, min_val, max_val)
        )

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space.
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        min_val, max_val = y.min(), y.max()
        return x * (max_val - min_val) + min_val


class NormalizeLogarithmic(DataNormalizer):
    """Normalize all logarithmically scaled values to lie in [0, 1]."""

    def __init__(self, norm: Literal["minmax", "bymax"] = "minmax"):
        match norm:
            case "minmax":
                self.norm = NormalizeMinMax()
            case "bymax":
                self.norm = NormalizeByMax()
            case _:
                raise ValueError(f"Invalid normalization to [0,1]: {norm}")
        self.log = lambda x: torch.log(x + 1)

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return (
            self.norm(self.log(x), self.log(y))
            if y is not None
            else self.norm(self.log(x))
        )

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space: min_max(ln(x + 1)).
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        log_scale_y = self.log(y)
        log_scale_x = self.norm.invert(x, log_scale_y)
        return torch.exp(log_scale_x) - 1


class NormalizeSqrt(DataNormalizer):
    """Normalize all logarithmically scaled values to lie in [0, 1]."""

    def __init__(self, norm: Literal["minmax", "bymax"] = "minmax"):
        match norm:
            case "minmax":
                self.norm = NormalizeMinMax()
            case "bymax":
                self.norm = NormalizeByMax()
            case _:
                raise ValueError(f"Invalid normalization to [0,1]: {norm}")
        self.sqrt = lambda x: torch.sqrt(x)

    def __call__(
        self, x: Tensor, y: Tensor | None = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return (
            self.norm(self.sqrt(x), self.sqrt(y))
            if y is not None
            else self.norm(self.sqrt(x))
        )

    def invert(self, x: Tensor, y: Tensor) -> Tensor:
        """Invert the normalization process by scaling `x` back to its original range given by `y`.

        Args:
            x (Tensor): Tensor in 'normalized' space: by_max(sqrt(x)).
            y (Tensor): Tensor in 'original' space.

        Returns:
            Tensor: Retransformed Tensor.
        """
        sqrt_scale_y = self.sqrt(y)
        sqrt_scale_x = self.norm.invert(x, sqrt_scale_y)
        return sqrt_scale_x**2
