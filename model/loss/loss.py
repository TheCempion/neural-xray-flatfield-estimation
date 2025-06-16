# standard libraries
from typing import Dict, Literal, List
from abc import abstractmethod

# third part libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import piqa

# local packages
from utils.torch_settings import get_torch_device
from utils.datatypes import Loss_t
import utils.constants as const
import utils.flatfield_correction as FFC


__all__ = [
    "LossModule",
    "L1_Loss",
    "L2_Loss",
    "SSIM_Loss",
    "DSSIM_Loss",
    "MS_SSIM_Loss",
    "MS_DSSIM_Loss",
    "FRC_Loss",
    "GAN_Loss",
    "Variance_Loss",
    "Multigrid_Variance_Loss",
    "MS_Variance_Loss",
    "Range_Loss",
    "ssim_loss_fn",
    "mse_loss_fn",
    "l1_loss_fn",
]


class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = get_torch_device()
        self.num_training_steps = 0
        self.milestones = []

    @abstractmethod
    def last_loss_terms(self) -> Dict[str, float]:
        raise NotImplementedError("Method needs to be defined in child class.")

    def keys(self) -> List[str]:
        return list(self.last_loss_terms().keys())

    def __len__(self) -> int:
        return len(self.last_loss_terms())

    def increment(self) -> bool:
        """Increment the counter of how many training steps were performed.

        Returns:
            bool: True, if a change of the loss weighting and the early stopping must be reset. False otherwise.
        """
        self.num_training_steps += 1
        return self.num_training_steps - 1 in self.milestones

    def update_iter_count(self, new_count: int) -> None:
        self.num_training_steps = new_count


class L1_Loss(LossModule):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_1"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.loss = self.lam * F.l1_loss(output, target)
        return self.loss


class L2_Loss(LossModule):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_2"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.loss = self.lam * F.mse_loss(output, target)
        return self.loss


class SSIM_Loss(LossModule):
    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__()
        self.lam = lam
        kwargs["n_channels"] = kwargs.get("n_channels", 1)
        self.ms_ssim = piqa.SSIM(**kwargs)
        self.to(get_torch_device())

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{SSIM}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.ssim_value = self.lam * self.ms_ssim(
            torch.clip(output, min=const.EPS, max=1 - const.EPS),
            torch.clip(target, min=const.EPS, max=1 - const.EPS),
        )
        return self.ssim_value


class MS_SSIM_Loss(LossModule):
    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__()
        self.lam = lam
        kwargs["n_channels"] = kwargs.get("n_channels", 1)
        self.ms_ssim = piqa.MS_SSIM(**kwargs)
        self.to(get_torch_device())

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{MS\_SSIM}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.loss = self.lam * self.ms_ssim(
            torch.clip(output, min=const.EPS, max=1 - const.EPS),
            torch.clip(target, min=const.EPS, max=1 - const.EPS),
        )
        return self.loss


class DSSIM_Loss(LossModule):
    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__()
        self.lam = lam
        self.ssim = SSIM_Loss(**kwargs)
        self.to(get_torch_device())

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{DSSIM}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_value = self.ssim(
            torch.clip(output, min=const.EPS, max=1 - const.EPS),
            torch.clip(target, min=const.EPS, max=1 - const.EPS),
        )
        self.loss = self.lam * (1 - ssim_value)
        return self.loss


class MS_DSSIM_Loss(LossModule):
    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__()
        self.lam = lam
        self.ssim = MS_SSIM_Loss(**kwargs)
        self.to(get_torch_device())

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{MS\_DSSIM}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_value = self.ssim(
            torch.clip(output, min=const.EPS, max=1 - const.EPS),
            torch.clip(target, min=const.EPS, max=1 - const.EPS),
        )
        self.loss = self.lam * (1 - ssim_value)
        return self.loss


class GAN_Loss(LossModule):
    def __init__(self, lam: float = 1.0):
        assert lam > 0
        super().__init__()
        self.bce = nn.BCELoss(reduction=const.BCE_LOSS_REDUCTION_METHOD)
        self.lam = lam

    def last_loss_terms(self, ext: str = "") -> Dict[str, float]:
        key = rf"\mathcal{{L}}_{{GAN}}_{{{ext}}}" if ext != "" else r"\mathcal{L}_{GAN}"
        try:
            return {
                key: self.loss.item()
            }  # NOTE: Probably only works when reduction="mean"; .view(-1).mean()?
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self._get_target_tensor(output, target_is_real)
        self.loss = self.lam * self.bce(output, target_tensor)
        return self.loss

    def _get_target_tensor(
        self, output: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Create label tensors with the same size as the input.

        Args:
            output (tensor): Typically the prediction from a discriminator
            target_is_real (bool): If the ground truth label is for real images or fake images

        Returns:
            (Tensor) A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = torch.tensor(const.GAN_REAL_LABEL)
        else:
            target_tensor = torch.tensor(const.GAN_FAKE_LABEL)
        return target_tensor.expand_as(output).to(self.device)


class Variance_Loss(LossModule):
    def __init__(
        self,
        lam: float = 1.0,
        *,
        weights: float | List[float] | None = 1.0,
        mode: Literal["sub", "div", "ffc"] = "sub",
    ):
        """Initialize the `Multigrid_Variance_Loss` function.

        Args:
            lam (float): Weighting of the loss.
            weights (float | List[float], optional): The weighting for the loss of different patch-sizes. The smallest
                        patch size will be `size // 2**(len(weights) - 1)`, where size is the spatial dimension of the
                        image. if `weights` is a scalar or None, a single weight with value 1.0 will be used instead,
                        i.e. original scale of the image. Defaults to 1.0.
        """
        assert lam > 0
        if weights is None or isinstance(weights, float):
            weights = [1.0]
        assert all([w >= 0 for w in weights])
        assert any([w > 0 for w in weights])
        assert mode in ["sub", "div", "ffc"]

        super().__init__()
        self.lam = lam
        self.mode = mode

        if not (W := sum(weights)) == 1:
            weights = [w / W for w in weights]
        self.weights = torch.tensor(weights, device=self.device)

    def last_loss_terms(self) -> Dict[str, float]:
        key_var = r"\mathcal{L}_{\sigma^2}"
        try:
            return {key_var: self.loss.item()}
        except:
            return {key_var: None}

    def _do_ffc(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "div":
            ffc = (target / (output + const.EPS)).clip(max=10)
        elif self.mode == "sub":
            ffc = target - output
        elif self.mode == "ffc":
            ffc = FFC.correct_flatfield(target, output)
        return ffc

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Basic variance calculation."""
        assert len(self.weights) == 1  # -> otherwise use weight == 1.0

        ffc = self._do_ffc(output, target)
        patch_vars = ffc.var(dim=(2, 3), unbiased=False)  # Variance across spatial dims
        self.loss = self.lam * patch_vars.mean()
        return self.loss


class Multigrid_Variance_Loss(Variance_Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(self.weights) == 1:
            return super().forward(output, target)

        ffc = self._do_ffc(output, target)

        B, C, H, W = ffc.shape
        assert (
            H // (2 ** (len(self.weights) - 1)) >= 1
        ), f"Minimum patch size must be at least 1: {target.shape=}, {self.weights=}"

        total_loss = torch.zeros(1, device=self.device)
        for i, weight in enumerate(self.weights):
            if weight == 0.0:
                continue
            elif (
                patch_size := H // 2**i
            ) == 1:  # variance 0 for all pixel if patch_size == 1
                continue

            unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
            patches = unfold(ffc)
            patches = patches.view(
                B, C, patch_size**2, -1
            )  # Shape (B, C, P**2, N_patches)

            # Compute mean and variance for each patch
            patch_vars = patches.var(
                dim=2, unbiased=False
            )  # Variance along patch pixels: (B, C, num_patches)
            total_loss += weight * patch_vars.mean()

        # Combine losses
        self.loss = self.lam * total_loss
        return self.loss


class MS_Variance_Loss(Variance_Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(self.weights) == 1:
            return super().forward(output, target)

        ffc = self._do_ffc(output, target)

        B, C, H, W = ffc.shape
        assert (
            H / 2 ** (len(self.weights) - 1) >= 1
        ), f"Minimum scaled down image must be at least 1: {target.shape=}, {self.weights=}"

        total_loss = torch.zeros(1, device=self.device)

        # Iterate over scales
        for i, weight in enumerate(self.weights):
            if weight == 0.0:
                continue

            # Rescale tensor
            if i == 0:  # no rescaling needed
                scaled_ffc = ffc
            else:
                scaled_ffc = F.interpolate(
                    ffc, scale_factor=1 / (2**i), mode="bilinear", align_corners=False
                )

            # Compute variance for each scaled version
            patch_vars = scaled_ffc.var(
                dim=(2, 3), unbiased=False
            )  # Variance across spatial dims
            total_loss += weight * patch_vars.mean()

        # Combine losses
        self.loss = self.lam * total_loss
        return self.loss


class Range_Loss(LossModule):
    def __init__(
        self,
        lam: float = 1.0,
        metric: Literal["L1", "L2"] = "L2",
        reduction: Literal["mean", "poi"] = "mean",
    ):
        super().__init__()
        self.lam = lam
        assert reduction in ["mean", "poi"]  # poi = pixel of interest
        match metric:
            case "L1":
                self.metric = L1_Loss(lam=1)
            case "L2":
                self.metric = L2_Loss(lam=1)
            case _:
                raise ValueError(f"Unknown distance metric: {metric}")
        self.reduction = reduction

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{Range}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # upper_loss = nn.ReLU(output - target.max())
        upper = target.amax(dim=(1, 2, 3), keepdim=True)
        lower = target.amin(dim=(1, 2, 3), keepdim=True)

        upper_loss = self.metric(nn.ReLU()(output - upper), torch.zeros_like(target))
        lower_loss = self.metric(output * (output < lower), lower.expand_as(target))
        if self.reduction == "mean":
            self.loss = self.lam * (upper_loss + lower_loss).mean()
        elif self.reduction == "poi":  # pixel of interest
            loss = upper_loss + lower_loss
            if (divisor := (loss > 0).sum()) > 0:
                self.loss = self.lam * loss / divisor
            else:
                self.loss = torch.zeros(1, device=self.device)
        return self.loss


# https://github.com/pvilla/PhaseGAN/blob/master/models/trainer.py
# line 181ff
class FRC_Loss(LossModule):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam
        self.to(get_torch_device())

    def last_loss_terms(self) -> Dict[str, float]:
        key = r"\mathcal{L}_{FRC}"
        try:
            return {key: self.loss.item()}
        except:
            return {key: None}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier Ring Correlation Loss for 3D images.
        Args:
            output (Tensor): torch.Tensor of shape (batch_size, 1, H, W).
            target (Tensor): torch.Tensor of shape (batch_size, 1, H, W).

        Returns:
            Scalar loss value.
        """
        # assert output.shape == target.shape, "Input and target must have the same shape"
        output = output.squeeze(1)  # Resulting shape: (B, H, W)
        target = target.squeeze(1)  # Resulting shape: (B, H, W)

        # Image dimensions
        B, H, W = output.shape
        rnyquist = H // 2  # Assuming H == W

        # Create 2D frequency grid
        x = torch.fft.fftfreq(H, 1 / H).to(self.device)  # Frequency range for x
        y = torch.fft.fftfreq(W, 1 / W).to(self.device)  # Frequency range for y
        X, Y = torch.meshgrid(x, y, indexing="ij")
        radial_map = torch.sqrt(X**2 + Y**2)
        radial_indices = torch.round(radial_map).long()

        # Compute 2D FFTs
        F1 = torch.fft.fft2(output, norm="forward")  # Shape: (B, H, W)
        F2 = torch.fft.fft2(target, norm="forward")  # Shape: (B, H, W)

        # Initialize accumulators for each radial frequency
        C_r = torch.zeros((rnyquist + 1, B), device=self.device)
        C_i = torch.zeros((rnyquist + 1, B), device=self.device)
        C1 = torch.zeros((rnyquist + 1, B), device=self.device)
        C2 = torch.zeros((rnyquist + 1, B), device=self.device)

        # Loop through radial bins
        for r in range(rnyquist + 1):
            mask = radial_indices == r  # Shape: (H, W)
            mask = mask.unsqueeze(0)  # Broadcast to batch, shape: (1, H, W)

            # Masked FFT components
            F1_r = F1 * mask  # Shape: (B, H, W)
            F2_r = F2 * mask  # Shape: (B, H, W)

            # Compute correlations and energy for each radial frequency
            C_r[r] = torch.sum(
                F1_r.real * F2_r.real + F1_r.imag * F2_r.imag, dim=(-2, -1)
            )
            C_i[r] = torch.sum(
                F1_r.imag * F2_r.real - F1_r.real * F2_r.imag, dim=(-2, -1)
            )
            C1[r] = torch.sum(F1_r.abs() ** 2, dim=(-2, -1))
            C2[r] = torch.sum(F2_r.abs() ** 2, dim=(-2, -1))

        # Compute FRC for each radial bin
        frc = torch.sqrt(C_r**2 + C_i**2) / (torch.sqrt(C1 * C2) + const.EPS)

        # Compute loss (penalizing deviation from ideal FRC=1)
        self.loss = self.lam * torch.mean((1 - frc) ** 2)
        return self.loss


def ssim_loss_fn(**kwargs) -> piqa.SSIM:
    return piqa.SSIM(**kwargs)


def mse_loss_fn() -> Loss_t:
    return F.mse_loss


def l1_loss_fn() -> Loss_t:
    return F.l1_loss
