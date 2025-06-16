# standard libraries
from typing import Dict, Any, List, Literal
from dataclasses import dataclass
import sys

# third part libraries
from torch import Tensor
import torch

# local packages
import model.loss as module_loss


__all__ = [
    "ComposedLoss",
    "EpochBasedComposedLoss",
    "GANRegLoss",
    "DiracLoss",
]


class ComposedLoss(module_loss.LossModule):
    def __init__(self, losses: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.loss_functions: List[module_loss.LossModule] = []
        for loss_name, kwargs_loss_fn in losses.items():
            try:
                loss_fn = getattr(module_loss, loss_name)
            except:
                loss_fn = getattr(sys.modules[__name__], loss_name)
            finally:
                self.loss_functions.append(loss_fn(**kwargs_loss_fn))
        assert len(self.loss_functions) > 0

    def __len__(self) -> int:
        return len(self.loss_functions)

    def last_loss_terms(self) -> Dict[str, float]:
        last_loss_terms = {}
        for loss_fn in self.loss_functions:
            last_loss_terms |= loss_fn.last_loss_terms()
        return last_loss_terms

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return sum(list([loss_fn(output, target) for loss_fn in self.loss_functions]))


class EpochBasedComposedLoss(module_loss.LossModule):
    def __init__(
        self, losses: Dict[str, Dict[Any, Any]]
    ):  # can also contain start_iter: int and/or ent_iter: int
        super().__init__()

        @dataclass
        class IntervalLoss:
            loss_fn: module_loss.LossModule
            start_iter: int | None
            end_iter: int | None

        self.loss_functions: List[IntervalLoss] = (
            []
        )  # actually need to store start, end, loss
        for loss_name, kwargs_loss_fn in losses.items():
            try:
                loss_fn = getattr(module_loss, loss_name)
            except:
                loss_fn = getattr(sys.modules[__name__], loss_name)
            finally:
                start_iter = kwargs_loss_fn.pop("start_iter", None)
                end_iter = kwargs_loss_fn.pop("end_iter", None)
                if start_iter is not None and end_iter is not None:
                    assert (
                        start_iter < end_iter
                    ), f"{loss_fn}\t{start_iter=}\t{end_iter=}"
                if start_iter is not None:
                    self.milestones.append(start_iter)
                if end_iter is not None:
                    self.milestones.append(end_iter)

                self.loss_functions.append(
                    IntervalLoss(
                        loss_fn=loss_fn(**kwargs_loss_fn),
                        start_iter=start_iter,
                        end_iter=end_iter,
                    )
                )

        assert len(self.loss_functions) > 0
        assert any([loss.start_iter in [None, 0] for loss in self.loss_functions])
        assert any([loss.end_iter is None for loss in self.loss_functions])

    def __len__(self) -> int:
        return len(self.loss_functions)

    def last_loss_terms(self) -> Dict[str, float]:
        last_loss_terms = {}
        for loss in self.loss_functions:
            last_loss_terms |= loss.loss_fn.last_loss_terms()
        return last_loss_terms

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        total_losses = []
        for loss in self.loss_functions:
            if (
                loss.start_iter is None or loss.start_iter <= self.num_training_steps
            ) and (loss.end_iter is None or loss.end_iter > self.num_training_steps):
                total_losses.append(loss.loss_fn(output, target))
        return sum(total_losses)


class GANRegLoss(module_loss.LossModule):
    def __init__(self, lambda_gan: float, **kwargs):
        super().__init__()
        self.gan_loss = module_loss.GAN_Loss(lam=lambda_gan)
        self.criterion = EpochBasedComposedLoss(**kwargs)

    def __len__(self) -> int:
        raise NotImplementedError(
            "Need to call `__len__()` for `gan_loss` and `criterion` loss, separately."
        )

    def last_loss_terms(self) -> Dict[str, float]:
        raise NotImplementedError(
            "Need to call `last_loss_terms()` for `gan_loss` and `criterion` loss, separately."
        )

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError(
            "Need to call `forward` for `gan_loss` and `criterion` loss, separately."
        )


class DiracLoss(module_loss.LossModule):
    def __init__(
        self,
        lambda_l2: float,
        lambda_var: float,
        *,
        patch_mode: Literal["multigrid", "mg", "ms", "multiscale"] | None,
        use_dirac_label: bool = False,
        **kwargs_var_loss,
    ):
        super().__init__()
        match patch_mode:
            case pm if pm in ["mg", "multigrid"]:
                self.variance_loss = module_loss.Multigrid_Variance_Loss(
                    lambda_var, **kwargs_var_loss
                )
            case pm if pm in ["ms", "multiscale"]:
                self.variance_loss = module_loss.MS_Variance_Loss(
                    lambda_var, **kwargs_var_loss
                )
            case None:
                self.variance_loss = module_loss.Variance_Loss(
                    lambda_var, **kwargs_var_loss
                )
            case _:
                raise ValueError(
                    f"patchmode must be either `patch` or `ms`, not {patch_mode=}"
                )
        self.use_dirac_label = use_dirac_label
        self.l2_loss = module_loss.L2_Loss(lambda_l2)

    def last_loss_terms(self):
        if self.use_dirac_label:
            key = r"\mathcal{L}_{\delta}"
            try:
                return {key: self.loss.item()}
            except:
                return {key: None}
        else:
            return self.l2_loss.last_loss_terms() | self.variance_loss.last_loss_terms()

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        self.loss = self.l2_loss(output, target) + self.variance_loss(output, target)
        return self.loss
