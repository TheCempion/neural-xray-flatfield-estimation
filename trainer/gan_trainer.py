# standard libraries
from typing import Dict, Any, Tuple

# third party libraries
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# local packages
from parse_config import ConfigParser
from base import BaseGAN, BaseTrainer
import utils.plotting as plotting
import utils.data_normalization as norm
from model.loss import GANRegLoss, Variance_Loss
from utils.torch_settings import get_torch_device


__all__ = [
    "GANTrainer",
]


class GANTrainer(BaseTrainer):
    """Trainer class for GANs."""

    def __init__(
        self,
        gan: BaseGAN,
        loss: GANRegLoss,
        optimizerG: Optimizer,
        optimizerD: Optimizer,
        config: ConfigParser,
        device: torch.device,
        data_loader: DataLoader,
        *,
        data_normalizer: norm.DataNormalizer,
        lr_schedulerG: LRScheduler | None = None,
        lr_schedulerD: LRScheduler | None = None,
        valid_data_loader: DataLoader | None = None,
        valid_data_loader_2: DataLoader | None = None,
    ):
        assert isinstance(loss, GANRegLoss)

        optimizer_dict = {"G": optimizerG, "D": optimizerD}
        lr_scheduler_dict = {
            suffix: scheduler
            for suffix, scheduler in zip(["D", "G"], [lr_schedulerD, lr_schedulerG])
            if scheduler is not None
        } or None

        super().__init__(
            model=None,  # will be set below
            criterion=loss.criterion,
            optimizer_dict=optimizer_dict,
            config=config,
            device=device,
            data_loader=data_loader,
            data_normalizer=data_normalizer,
            lr_scheduler_dict=lr_scheduler_dict,
            valid_data_loader=valid_data_loader,
            valid_data_loader_2=valid_data_loader_2,
        )

        # (mostly) GAN specific Settings
        self.model = gan
        self.netG = self.model.netG
        self.netD = self.model.netD
        self.netG_params_list = list(self.netG.parameters())
        self.gan_loss = loss.gan_loss

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
        self.data_loader.train()  # NOTE: Should already be True by default

        self.total_accum_loss_D = 0  # calculate the loss per training step
        self.total_accum_loss_G = 0  # calculate the loss per training step
        self.total_accum_loss = 0  # calculate loss per train_step (GAN Loss + Criterion) -> loss_tracker.add_train_loss

        total_loss = (
            0  # training loss: GAN Loss + criterion Loss ->  averaged for epoch summary
        )
        has_zero_grad = True
        train_step_in_epoch = 0
        accum_batch_divisor = min(
            self.len_epoch, self.accum_batches
        )  # accumulate that number of batches for next step
        for batch_idx, (data, targetG, condition) in enumerate(self.data_loader):
            do_step = (batch_idx + 1) % self.accum_batches == 0
            data, targetG, real_batch = GANTrainer.prepare_batches(
                data, targetG, condition, data_normalizer=self.data_normalizer
            )

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Forward pass real batch through D
            output = self.netD(real_batch)
            loss_D_real = (
                self.gan_loss(output, target_is_real=True) / accum_batch_divisor
            )
            loss_D_real.backward()
            has_zero_grad = False

            # create fake batch with G and forward through D
            generator_output = self.netG(data)
            fake_batch_discr = torch.cat(
                [generator_output, real_batch[:, 1:, ...].clone()], dim=1
            )  # concat with conditions
            output = self.netD(fake_batch_discr.detach())
            loss_D_fake = (
                self.gan_loss(output, target_is_real=False) / accum_batch_divisor
            )
            loss_D_fake.backward()

            loss_D = (loss_D_real.item() + loss_D_fake.item()) * 0.5
            self.total_accum_loss_D += loss_D

            if do_step:  # accumulating gradient
                self._step("D")  # G still has non-zero grad -> has_zero_grad = False

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            predD = self.netD(fake_batch_discr)
            loss_G_GAN = self.gan_loss(
                predD, target_is_real=True
            )  # target is real, because G wants to fool D

            loss_G_criterion = self.criterion(
                generator_output, targetG
            )  # Calculate other losses, e.g. SSIM

            loss_G = (loss_G_GAN + loss_G_criterion) / accum_batch_divisor
            loss_G.backward(
                inputs=self.netG_params_list
            )  # ONLY Accumulate gradients for generator

            # accumulate losses in this epoch
            self.total_accum_loss += loss_G.item()
            self.total_accum_loss_G += loss_G_GAN.item() / accum_batch_divisor
            total_loss += loss_G.item() * accum_batch_divisor

            if do_step:  # accumulating gradient
                self._step("G")
                has_zero_grad = True  # Now did both steps for G and D
                accum_batch_divisor = min(
                    self.len_epoch - (batch_idx + 1), self.accum_batches
                )  # next num of accum. batches. if accum_batch_divisor == 0 => end of epoch => no div by 0

                sample_idx = (
                    batch_idx + 1
                ) * self.data_loader.batch_size  # needed for logging
                self._log_stats(
                    epoch=epoch,
                    step=self.training_iter_step,
                    idx=sample_idx,
                    loss=self.total_accum_loss,
                )
                self._track_train_losses()

                if self.do_validation and (train_step_in_epoch in self.log_step):
                    val_loss, val_summary = self._run_validate(self.training_iter_step)
                    self._log_summary(val_summary)
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

                self.training_iter_step += (
                    1  # increase afterwards, because loss is from before the step
                )
                train_step_in_epoch += 1
                self._increment_loss_iteratation()  # need to be called AFTER validation for same loss function

            if batch_idx == self.len_epoch:
                break

        # might need to still make a step, due to accumulated batches
        if not has_zero_grad:
            self._step("D")
            self._step("G")

            sample_idx = (
                batch_idx + 1
            ) * self.data_loader.batch_size  # needed for logging
            self._track_train_losses()
            self._log_stats(
                epoch=epoch,
                step=self.training_iter_step,
                idx=sample_idx,
                loss=self.total_accum_loss,
            )
            self.training_iter_step += 1

        epoch_summary = {
            "training_loss": total_loss / (batch_idx + 1),
            "validation_loss": None,
        }

        if self.do_validation:
            val_loss, val_summary = self._run_validate(
                step=self.training_iter_step - 1, back_to_train_mode=True
            )
            epoch_summary |= val_summary
        self._increment_loss_iteratation()  # need to be called AFTER (possible) validation for same loss function

        if epoch == self.epochs or self.early_stop(
            val_loss=val_loss, model=self.model, epoch=epoch, logger=self.logger
        ):
            self._store_samples()
        return epoch_summary

    def _track_train_losses(self) -> None:
        self.loss_tracker.add_train_loss(
            loss=self.total_accum_loss, train_step=self.training_iter_step
        )
        self.loss_tracker.add_generator_gan_loss(
            loss=self.total_accum_loss_G, train_step=self.training_iter_step
        )
        self.loss_tracker.add_discriminator_loss(
            loss=self.total_accum_loss_D, train_step=self.training_iter_step
        )
        self.total_accum_loss = 0
        self.total_accum_loss_G = 0
        self.total_accum_loss_D = 0

    def _run_validate(
        self, step: int, *, back_to_train_mode: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        val_loss, term_losses, gan_loss_terms = self._validate(
            iteration=step, back_to_train_mode=back_to_train_mode
        )
        self.loss_tracker.add_val_loss(val_loss, train_step=step)
        self.loss_tracker.add_other_losses(term_losses, train_step=step)
        summary = {
            "validation_loss": val_loss,
            **term_losses,
        }
        # track validation lyosses for generator losses seperately
        self.loss_tracker.add_discriminator_loss(
            gan_loss_terms["D"], train_step=step, val=True
        )
        self.loss_tracker.add_generator_gan_loss(
            gan_loss_terms["G"], train_step=step, val=True
        )
        return val_loss, summary

    def _validate(
        self, iteration: int, *, back_to_train_mode: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Validate after training some training iteration steps.

        Note:
            Ignoring tensorboard writer.

        Args:
            back_to_train_mode (bool, optional): If True, set model to train mode after validation. Defaults to True.

        Returns:
            Tuple[float, Dict[str, float], ict[str, float]]: Averaged validation loss, dictionary containing the
                        averaged loss terms, and gan lossses for both discriminator and generator.
        """
        self.logger.debug("Validating...")
        self.model.eval()
        total_loss = 0
        total_loss_D = 0
        total_loss_G = 0
        total_batches = 0
        total_variance_loss = 0
        variance_loss = Variance_Loss(lam=1)
        total_loss_terms = {
            loss_term: 0 for loss_term in self.criterion.last_loss_terms().keys()
        } | {loss_term: 0 for loss_term in self.gan_loss.last_loss_terms().keys()}
        with torch.no_grad():
            for valid_data_loader in [self.valid_data_loader, self.valid_data_loader_2]:
                if valid_data_loader is None or len(valid_data_loader) == 0:
                    continue
                valid_data_loader.eval()
                for batch_idx, (data, targetG, condition) in enumerate(
                    valid_data_loader
                ):
                    data, targetG, real_batch = GANTrainer.prepare_batches(
                        data, targetG, condition, data_normalizer=self.data_normalizer
                    )

                    # discriminator (not really important; just tracking its progress)
                    output = self.netD(real_batch)
                    loss_D_real = self.gan_loss(output, target_is_real=True)

                    generator_output = self.netG(data)
                    fake_batch_discr = torch.cat(
                        [generator_output, real_batch[:, 1:, ...].clone()], dim=1
                    )  # concat with conditions
                    pred_D_fake_batch = self.netD(fake_batch_discr)
                    loss_D_fake = self.gan_loss(pred_D_fake_batch, target_is_real=False)
                    total_loss_D += (loss_D_real + loss_D_fake).item() * 0.5

                    # generator validation loss
                    loss_G_GAN = self.gan_loss(
                        pred_D_fake_batch, target_is_real=True
                    ).item()  # new last loss term
                    loss_G_criterion = self.criterion(
                        generator_output, targetG
                    ).item()  # Calculate other losses
                    total_loss_G += loss_G_GAN
                    total_loss += loss_G_criterion + loss_G_GAN
                    total_variance_loss += variance_loss(
                        generator_output, targetG
                    ).item()

                    total_batches += 1
                    for loss_fn in [self.criterion, self.gan_loss]:
                        for loss_term, loss_value in loss_fn.last_loss_terms().items():
                            if loss_value is not None:
                                total_loss_terms[loss_term] += loss_value

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
        gan_loss_terms = {
            "G": total_loss_G / total_batches,
            "D": total_loss_D / total_batches,
        }
        return total_loss / total_batches, total_loss_terms, gan_loss_terms

    @plotting.change_mpl_settings("font", size=20)
    def _store_samples(self, back_to_train_mode: bool = True) -> None:
        super().store_examples(
            self.netG,
            data_loader=self.valid_data_loader,
            iter_or_epoch=self.training_iter_step,
            prepare_batch_fn=GANTrainer.prepare_batches,
            num_samples=16,
            grid_size=(4, 4),
            back_to_train_mode=back_to_train_mode,
        )

    @staticmethod
    def prepare_batches(
        data: Tensor,
        target: Tensor,
        condition: Tensor | None,
        *,
        data_normalizer: norm.DataNormalizer,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if type(data_normalizer) == norm.NormalizeMinMaxConditioned:
            if condition is not None:
                x = torch.cat([data.clone(), condition.clone()], axis=1)
                y = torch.cat([target.clone(), condition.clone()], axis=1)
            else:
                x, y = data.clone(), target.clone()
            data, real_batch = data_normalizer(x, y)
            target = real_batch[:, :1, :, :].clone()
        else:
            if condition is not None:
                # assuming that both target and condition are already 4D Tensors (NxCxHxW)
                real_batch = torch.cat([target.clone(), condition.clone()], axis=1)
            else:
                real_batch = target.clone()

            concat_for_normalization = torch.cat(
                [data.clone(), real_batch.clone(), target.clone()], axis=1
            )

            _, real_batch = data_normalizer(
                concat_for_normalization, real_batch
            )  # normalize based on model input
            _, target = data_normalizer(concat_for_normalization, target)

            # need to do this after normalization, otherwise normalization might consider `condition`
            _, data = data_normalizer(concat_for_normalization, data)
            if condition is not None:
                _, condition = data_normalizer(concat_for_normalization, condition)
                data = torch.cat([data, condition.clone()], axis=1)
        device = get_torch_device()
        data, target = data.to(device), target.to(device)
        real_batch = real_batch.to(device)
        return data, target, real_batch
