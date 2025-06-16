# standard libraries
from typing import Dict, Any, Tuple

# third party libraries
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# local packages
from base import BaseTrainer
from parse_config import ConfigParser
from utils.data_normalization import DataNormalizer
import utils.plotting as plotting
from model.loss import LossModule, Variance_Loss
from utils.torch_settings import get_torch_device


__all__ = [
    "Trainer",
]


class Trainer(BaseTrainer):
    """Trainer class."""

    def __init__(
        self,
        model: nn.Module,
        criterion: LossModule,
        optimizer: Optimizer,
        config: ConfigParser,
        device: torch.device,
        data_loader: DataLoader,
        *,
        data_normalizer: DataNormalizer,
        lr_scheduler: LRScheduler | None = None,
        valid_data_loader: DataLoader | None = None,  # if validation_split > 0
        valid_data_loader_2: (
            DataLoader | None
        ) = None,  # if another validation dataloader was given
    ):
        optimizer_dict = {"": optimizer}
        lr_scheduler_dict = {"": lrs for lrs in [lr_scheduler] if lrs is not None}
        super().__init__(
            model,
            criterion,
            optimizer_dict=optimizer_dict,
            config=config,
            device=device,
            data_loader=data_loader,
            data_normalizer=data_normalizer,
            lr_scheduler_dict=lr_scheduler_dict,
            valid_data_loader=valid_data_loader,
            valid_data_loader_2=valid_data_loader_2,
        )

    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Train and validate an epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            Dict[str, Any]: Result in format {"training_loss": avg_train_loss, "validation_loss": avg_valid_loss}".
                    Note: The validation is `None` if `self.do_validation` is False.
        """
        if epoch == 1:
            # store model output before training
            self._store_samples()

        self.model.train()
        self.data_loader.train()

        total_loss = 0
        total_accum_loss = 0
        has_zero_grad = True
        train_step_in_epoch = 0
        accum_batch_divisor = min(
            self.len_epoch, self.accum_batches
        )  # accumulate that number of batches for next step
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = Trainer.prepare_batches(
                data, target, data_normalizer=self.data_normalizer
            )
            output = self.model(data)
            loss = self.criterion(output, target) / accum_batch_divisor
            loss.backward()
            has_zero_grad = False
            total_loss += (
                loss.item() * accum_batch_divisor
            )  # Need to consider actual loss, since it will be averaged
            total_accum_loss += loss.item()

            if (batch_idx + 1) % self.accum_batches == 0:  # accumulating gradient
                self._step()
                accum_batch_divisor = min(
                    self.len_epoch - (batch_idx + 1), self.accum_batches
                )  # next num of accum. batches. if accum_batch_divisor == 0 => end of epoch => no div by 0
                has_zero_grad = True

                sample_idx = (batch_idx + 1) * self.data_loader.batch_size
                self._log_stats(
                    epoch=epoch,
                    step=self.training_iter_step,
                    idx=sample_idx,
                    loss=total_accum_loss,
                )
                self.loss_tracker.add_train_loss(
                    total_accum_loss, train_step=self.training_iter_step
                )
                total_accum_loss = 0

                if self.do_validation and (train_step_in_epoch in self.log_step):
                    val_loss, term_losses = self._validate(
                        self.training_iter_step, back_to_train_mode=True
                    )
                    self.loss_tracker.add_val_loss(
                        val_loss, train_step=self.training_iter_step
                    )
                    self.loss_tracker.add_other_losses(
                        term_losses, train_step=self.training_iter_step
                    )
                    self._log_summary({"validation_loss": val_loss, **term_losses})
                    if self.early_stop(
                        val_loss=val_loss,
                        model=self.model,
                        epoch=epoch,
                        logger=self.logger,
                    ):
                        # Validation was done here, so it does not need to be done again at the end of method
                        self.do_validation = False
                        break
                    if self.early_stop.counter == 0:  # new best model
                        self._store_samples()

                if self.writer is not None:
                    self.writer.set_step(self.training_iter_step - 1)
                    self.writer.add_scalar("loss", loss.item())

                self.training_iter_step += (
                    1  # increase afterwards, because loss is from before the step
                )
                train_step_in_epoch += 1
                self._increment_loss_iteratation()  # need to be called AFTER validation for same loss function

            if batch_idx == self.len_epoch:
                break

        # might need to still make a step, due to accumulated batches
        if not has_zero_grad:
            self._step()

            sample_idx = (batch_idx + 1) * self.data_loader.batch_size
            self.loss_tracker.add_train_loss(
                total_accum_loss, train_step=self.training_iter_step
            )
            self._log_stats(
                epoch=epoch,
                step=self.training_iter_step,
                idx=sample_idx,
                loss=total_accum_loss,
            )
            self.training_iter_step += 1

        epoch_summary = {
            "training_loss": total_loss / (batch_idx + 1),
            "validation_loss": None,
        }
        if self.do_validation:
            val_loss, term_losses = self._validate(
                self.training_iter_step - 1, back_to_train_mode=False
            )
            # NOTE: Need to subtract a possible extra step, for the loss tracker, otherwise problems with indices
            self.loss_tracker.add_val_loss(
                val_loss, train_step=self.training_iter_step - 1
            )
            self.loss_tracker.add_other_losses(
                term_losses, train_step=self.training_iter_step - 1
            )
            epoch_summary |= {
                "validation_loss": val_loss,
                **term_losses,
            }
        self._increment_loss_iteratation()  # need to be called AFTER validation for same loss function

        if epoch == self.epochs or self.early_stop(
            val_loss=val_loss, model=self.model, epoch=epoch, logger=self.logger
        ):
            self._store_samples()
        return epoch_summary

    def _validate(
        self, iteration: int, *, back_to_train_mode: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Validate after training some training iteration steps.

        Args:
            iteration (int): Current training iteration.
            back_to_train_mode (bool, optional): If True, set model to train mode after validation. Defaults to True.

        Returns:
            Tuple[float, Dict[str, float]]: Averaged validation loss and dictionary containing the averaged loss terms.
        """
        self.logger.debug("Validating...")
        self.model.eval()
        total_loss = 0
        total_batches = 0
        total_loss_terms = {
            loss_term: 0 for loss_term in self.criterion.last_loss_terms().keys()
        }
        total_variance_loss = 0
        variance_loss = Variance_Loss(lam=1)
        with torch.no_grad():
            for valid_data_loader in [self.valid_data_loader, self.valid_data_loader_2]:
                if valid_data_loader is None or len(valid_data_loader) == 0:
                    continue
                log_images = (
                    True  # once per valid_data_loader log images, if writer exists
                )
                valid_data_loader.eval()
                for batch_idx, (data, target) in enumerate(valid_data_loader):
                    data, target = Trainer.prepare_batches(
                        data, target, data_normalizer=self.data_normalizer
                    )
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    total_variance_loss += variance_loss(target, output).item()

                    total_loss += loss.item()
                    total_batches += 1
                    for (
                        loss_term,
                        loss_value,
                    ) in self.criterion.last_loss_terms().items():
                        if loss_value is not None:
                            total_loss_terms[loss_term] += loss_value

                    if self.writer is not None:
                        self.writer.set_step(
                            (iteration - 1) * len(valid_data_loader) + batch_idx,
                            "valid",
                        )
                        self.writer.add_scalar("val_loss", loss.item())
                        if log_images:
                            log_images = False
                            self.writer.add_image(
                                "input", make_grid(data.cpu(), nrow=8, normalize=True)
                            )
                            self.writer.add_image(
                                "target",
                                make_grid(target.cpu(), nrow=8, normalize=True),
                            )
                            self.writer.add_image(
                                "output",
                                make_grid(output.cpu(), nrow=8, normalize=True),
                            )

        if back_to_train_mode:
            self.model.train()

        assert (
            total_batches > 0
        ), f"{total_batches=}; no batch in the validation data loader and that's really stupid."
        self.loss_tracker.add_variance_loss(
            total_variance_loss / total_batches, train_step=iteration
        )
        total_loss_terms = {
            loss_term: loss / total_batches
            for loss_term, loss in total_loss_terms.items()
        }
        return total_loss / total_batches, total_loss_terms

    @plotting.change_mpl_settings("font", size=20)
    def _store_samples(self, back_to_train_mode: bool = True) -> None:
        super().store_examples(
            self.model,
            data_loader=self.valid_data_loader,
            iter_or_epoch=self.training_iter_step,
            prepare_batch_fn=Trainer.prepare_batches,
            num_samples=16,
            grid_size=(4, 4),
            back_to_train_mode=back_to_train_mode,
        )

    @staticmethod
    def prepare_batches(
        data: Tensor, target: Tensor, *, data_normalizer: DataNormalizer
    ) -> Tuple[Tensor, Tensor]:
        device = get_torch_device()
        data, target = data_normalizer(data, target)
        data, target = data.to(device), target.to(device)
        return data, target
