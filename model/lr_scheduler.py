# standard libraries
from typing import List

# third party libraries
import torch.optim.lr_scheduler as LR
from torch.optim.optimizer import Optimizer

# local libraries


__all__ = [
    "WarmupCosineAnnealingLR",
    "WarmupStepLR",
    "DynamicWarmupCosineAnnealingLR",
]


class WarmupCosineAnnealingLR(LR.SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        eta_min: float,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.eta_min = eta_min
        warmup = LR.LinearLR(
            optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps
        )
        cosine_annealing = LR.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=eta_min, last_epoch=last_epoch
        )
        super().__init__(
            optimizer, schedulers=[warmup, cosine_annealing], milestones=[warmup_steps]
        )


class WarmupStepLR(LR.SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        warmup = LR.LinearLR(
            optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps
        )
        step_scheduler = LR.StepLR(optimizer, step_size=step_size, gamma=gamma)
        super().__init__(
            optimizer,
            schedulers=[warmup, step_scheduler],
            milestones=[warmup_steps],
            last_epoch=last_epoch,
        )


class DynamicWarmupCosineAnnealingLR(WarmupCosineAnnealingLR):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int] = None,
        step_size: int = None,
        gamma: float = None,
        **kwargs,
    ):
        """Also c

        Args:
            optimizer (Optimizer): Optimizer wrapper.
            milestones (List[int], optional): List of epoch indices. Must be increasing. If given, step_size must be
                        None. Defaults to None.
            step_size (int, optional): Period of learning rate decay. If given, milestones must be None.
                        Defaults to None.
            gamma (float): Multiplicative factor of learning rate decay. Must be given if either milestones is not None
                        or step_size is not None. Defaults to None.
        """
        super().__init__(optimizer=optimizer, **kwargs)

        assert not (
            milestones is None and step_size is None
        ), f"Only one must be not-None, but got: {milestones=} and {step_size=}"

        if step_size is not None:
            self.stepLR = LR.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif milestones is not None:
            self.stepLR = LR.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            self.stepLR = None

        if self.stepLR is not None:
            assert 0 < gamma < 1, f"{gamma=}"

    def step(self):
        """Override step method to adjust the maximum learning rate dynamically."""
        # if self.last_epoch in self.milestones:
        #     for param_group in self.optimizer.param_groups:
        #         param_group["lr"] *= self.gamma
        if self.stepLR is not None:
            self.stepLR.step()
        super().step()
