# standard libraries
import logging
from typing import Dict
import logging

# third party libraries
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np

# local packages
from utils.datatypes import Pathlike
from parse_config import ConfigParser
import utils.constants as const


__all__ = [
    "EarlyStopping",
    "DummyEarlyStopping",
]


class DummyEarlyStopping:
    """Dummy Early Stopping."""

    def __init__(self, *args, **kwargs):
        self.counter = -1

    def reset(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> None:
        return False

    def save_state(self, *args, **kwargs) -> None:
        pass


class EarlyStopping(DummyEarlyStopping):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int,
        optimizer_dict: Dict[str, Optimizer],
        config: ConfigParser,
        save_path: Pathlike,
        logger: logging.Logger,
        lr_scheduler_dict: Dict[str, LRScheduler] | None = None,
        delta: float = 0,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            optimizer (Optimizer): Optimizer wrapper used for the training.
            config (ConfigParser): Config used for training.
            save_path (Pathlike, optional): Path for the checkpoint to be saved to. Defaults to 'model_best.pth'.
            logger: (logging.Logger): For logging.
            lr_scheduler (LRScheduler, optional): Learning rate scheduler wrapper used for the training.
                        Defaults to None.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement.
                        Defaults to 0.
        """
        self.patience = patience
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler_dict = lr_scheduler_dict
        self.config = config
        self.delta = delta
        self.save_path = save_path
        self.logger = logger
        self.counter = 0
        self.val_loss_min = np.inf
        self.stop_early = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        *,
        epoch: int,
        logger: logging.Logger | None = None,
    ) -> bool:
        if self.stop_early:
            return True

        if val_loss < self.val_loss_min + self.delta:
            self.val_loss_min = val_loss
            self.counter = 0
            self.save_state(model, epoch)
        else:
            self.counter += 1
        if stop_early := self.counter >= self.patience:
            if logger:
                logger.info(
                    f"Validation performance didn't improve for {self.patience} steps. Training stops."
                )
        self.stop_early = stop_early
        return stop_early

    def save_state(
        self, model: nn.Module, epoch: int, *, path: Pathlike | None = None
    ) -> None:
        if path is None:
            path = self.save_path

        state = {
            "arch": type(model).__name__,
            const.PREF_MODEL_STATE_DICT: model.state_dict(),
            **{
                const.PREF_OPTIM_STATE_DICT + key: optim.state_dict()
                for key, optim in self.optimizer_dict.items()
            },
            **{
                const.PREF_LR_SCHEDULER_STATE_DICT + key: scheduler.state_dict()
                for key, scheduler in self.lr_scheduler_dict.items()
            },
            "config": self.config,
            "epoch": epoch,
        }

        torch.save(state, self.save_path)
        self.logger.info("Saved new best model.")

    def reset(self) -> None:
        self.logger.debug("Restting Early Stopper...")
        self.val_loss_min = np.inf
        self.counter = 0
