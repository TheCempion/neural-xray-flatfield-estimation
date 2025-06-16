# standard libraries

# third party libraries
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

# local packages


__all__ = [
    "ReLU",
    "LeakyReLU",
    "Abs",
    "Linear",
    "SiLU",
    "SoftAbs",
    "SoftPlus",
    "SoftPlusFixedGradient",
    "SoftPlusDynamicFixedGradient",
]


class ReLU(nn.ReLU):
    pass


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.2, inplace: bool = False):
        super().__init__(negative_slope=negative_slope, inplace=inplace)


class Abs(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


class Linear(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x


class SiLU(nn.SiLU):
    pass


class SoftAbs(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return (self.alpha * x.abs() ** 3) / (1 + self.alpha * x**2)


class SoftPlus(nn.Softplus):
    pass


class SoftPlusWithFixedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, fixed_grad, beta: float = 1):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.fixed_grad = fixed_grad
        ctx.beta = beta
        return torch.nn.functional.softplus(x, beta=beta)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        threshold = ctx.threshold
        fixed_grad = ctx.fixed_grad
        beta = ctx.beta

        # Compute the gradient
        grad_softplus = torch.sigmoid(beta * x)
        grad_modified = torch.where(
            x < threshold, torch.full_like(grad_softplus, fixed_grad), grad_softplus
        )
        return grad_output * grad_modified, None, None, None


class SoftPlusFixedGradient(nn.Module):
    def __init__(self, threshold: float = -2.19722, fixed_grad: float = 0.1):
        """Softplus activation that has minimum fixed gradient to mitigate vanishing gradients.

        Note:
            d/dx SoftPlus(x) = 0.1 when x ≈ -2.19722.

        Args:
            threshold (float, optional): Set the fixed gradients for all x < threshold. Defaults to -2.19722.
            fixed_grad (float, optional): The minimum gradient. Defaults to 0.1.
        """
        super().__init__()
        self.threshold = threshold
        self.fixed_grad = fixed_grad
        self.beta = 1

    def forward(self, x):
        return SoftPlusWithFixedGradient.apply(
            x, self.threshold, self.fixed_grad, self.beta
        )


class SoftPlusDynamicFixedGradient(nn.Module):
    def __init__(self, fixed_grad: float = 0.1, beta: float = 1.0):
        """Softplus activation that has minimum fixed gradient to mitigate vanishing gradients.

        Args:
            fixed_grad (float, optional): Minimum gradient for all x: d/dx SoftPlus(x) < fixed_grad. Defaults to 0.1.
            beta (float, optional): Beta parameter for SoftPlus activation function. Defaults to 1.
        """
        super().__init__()
        assert 0 < fixed_grad < 1
        decimals = int(-np.floor(np.log10(fixed_grad)))
        self.beta = beta
        self.threshold = (
            round(np.log(fixed_grad / (1 - fixed_grad)) / beta, decimals) - fixed_grad
        )  # for stability
        self.fixed_grad = fixed_grad

    def forward(self, x):
        return SoftPlusWithFixedGradient.apply(
            x, self.threshold, self.fixed_grad, self.beta
        )
