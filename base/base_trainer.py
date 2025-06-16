# standard libraries
from typing import Dict, Tuple, Callable, List
from abc import abstractmethod
from datetime import datetime, timedelta
import math
from pathlib import Path

# third party libraries
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# local packages
from logger import TensorboardWriter
from parse_config import ConfigParser
from utils.loss_tracker import LossTracker
from utils.early_stopping import EarlyStopping, DummyEarlyStopping
from utils import ensure_dir
import utils.constants as const
import utils.plotting as plotting
import utils.plotting.mpl_constants as mpl_constants
from utils.datatypes import Tensorlike, Pathlike
import utils.data_normalization as norm

from model.loss import LossModule
from base import BaseModel
import utils.flatfield_correction as ffc


__all__ = [
    "BaseTrainer",
]


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(
        self,
        model: BaseModel,
        criterion: LossModule,
        optimizer_dict: Dict[str, Optimizer],
        config: ConfigParser,
        device: torch.device,
        data_loader: DataLoader,
        data_normalizer: norm.DataNormalizer,
        lr_scheduler_dict: Dict[str, LRScheduler],
        valid_data_loader: DataLoader | None,
        valid_data_loader_2: DataLoader | None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler_dict = lr_scheduler_dict
        self.device = device
        self.data_normalizer = data_normalizer
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.valid_data_loader_2 = valid_data_loader_2
        self.do_validation = bool(self.valid_data_loader) or bool(
            self.valid_data_loader_2
        )

        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])
        self.output_dir = config.save_dir
        self.output_dir_plots = ensure_dir(self.output_dir / "plots")

        self.cfg_trainer = config["trainer"]
        self.epochs = self.cfg_trainer["epochs"]
        self.save_period = self.cfg_trainer.get("save_period", 0)
        self.what_to_plot = self.cfg_trainer.get("what_to_plot", "all")

        # initialize early stopping for storing best model
        self.early_stop = (
            EarlyStopping(
                patience=self.cfg_trainer["early_stop"],
                optimizer_dict=self.optimizer_dict,
                config=self.config,
                save_path=self.output_dir / "model_best.pth",
                logger=self.logger,
                lr_scheduler_dict=self.lr_scheduler_dict,
            )
            if ((p := self.cfg_trainer.get("early_stop", 0)) == "inf" or p > 0)
            and self.do_validation
            else DummyEarlyStopping()
        )

        # setup visualization writer instance
        if self.cfg_trainer["tensorboard"]:
            self.writer = TensorboardWriter(
                config.save_dir, self.logger, self.cfg_trainer["tensorboard"]
            )
        else:
            self.writer = None

        # track training progress
        self.loss_tracker = LossTracker(**self.cfg_trainer["loss_tracker_args"])

        self.accum_batches = self.cfg_trainer.get("accum_batches", 1)
        assert self.accum_batches > 0
        self.len_epoch = len(self.data_loader)
        assert (
            self.accum_batches <= self.len_epoch
        ), "Need to do at least one 'proper' training step."

        self.num_train_samples = self.len_epoch * self.data_loader.batch_size
        self.num_steps_per_epoch = math.ceil(self.len_epoch / self.accum_batches)

        self.start_epoch = (self.config.resume_epoch or 0) + 1
        self.training_iter_step = (self.start_epoch - 1) * self.num_steps_per_epoch
        self.plain_train_time = timedelta(0)

        # set logging step points during epoch
        # also include validation step after epoch (hence, +1), even though it will not be checked
        num_valid_steps = (
            self.num_steps_per_epoch // const.MAX_NUM_STEPS_BETWEEN_VALID
        ) + 1
        if num_valid_steps == 1:
            self.log_step = [self.num_steps_per_epoch // 2 - 1]
        else:
            self.log_step = [
                ((i + 1) * self.num_steps_per_epoch) // (num_valid_steps) - 1
                for i in range(num_valid_steps)
            ]

        self.logger.debug(f"{self.num_steps_per_epoch=}")
        self.logger.debug(f"{num_valid_steps=}")
        self.logger.debug(f"{self.log_step=}")

    @abstractmethod
    def _train_epoch(self, epoch: int):
        """Training logic for an epoch

        Args:
            epoch (int): Current epoch number.
        """
        raise NotImplementedError

    def train(self) -> None:
        """Full training logic."""
        for optim in self.optimizer_dict.values():
            optim.zero_grad()
        if self.config.resume:
            self._resume()

        start_time = datetime.now()
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d: %H:%M:%S')}")
        self._train_time_start()
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                epoch_result = self._train_epoch(epoch)
                self._log_summary(epoch_result)

                if self.early_stop.stop_early:  # will be evaluated in child class
                    break

                if self.save_period > 0 and epoch % self.save_period == 0:
                    self._save_checkpoint(epoch_or_iter=epoch)

                self.loss_tracker.save(self.output_dir)
        except Exception as e:
            import traceback

            self.logger.error(
                f"Training failed in epoch {epoch} at {datetime.now().strftime('%Y-%m-%d: %H:%M:%S')}"
            )
            traceback.print_exc()
            self.logger.error(e)
            self._save_checkpoint(model_name=f"model_interrupted{epoch}")
        else:
            self._save_checkpoint(model_name="model_last")
        finally:
            self._train_time_pause()
            end_time = datetime.now()
            self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d: %H:%M:%S')}")

            # calculate duration:
            def format_datetime(duration: datetime) -> str:
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            self.logger.info(
                f"Plain training time: {format_datetime(self.plain_train_time)}"
            )
            self.logger.info(f"Duration: {format_datetime(end_time - start_time)}")
        self.loss_tracker.save(self.output_dir)

    def _increment_loss_iteratation(self) -> None:
        if self.criterion.increment():
            self.early_stop.reset()

    def _log_summary(self, summary: Dict[str, float]) -> None:
        for key, value in summary.items():  # print results
            if value is None:
                continue
            self.logger.info(f"\t{str(key):15s}: {value}")

    def _save_checkpoint(
        self, *, epoch_or_iter: int | None = None, model_name: str | None = None
    ) -> None:
        """Saving model weights at checkpoint.

        Args:
            epoch_or_iter (int, optional): current epoch or training iteration. Defaults to None.
            model_name (str, optional): Name of the file the model should be stored. Defaults to None.

        Note:
            Exactly one of both arguments must be not None.
        """
        assert (epoch_or_iter is None) != (model_name is None)
        if epoch_or_iter is not None:
            ouput_dir = self.output_dir / "checkpoints"
            ouput_dir.mkdir(exist_ok=True)
        else:
            ouput_dir = self.output_dir

        model_name = (
            f"checkpoint-{epoch_or_iter}.pth"
            if epoch_or_iter is not None
            else f"{model_name}.pth"
        )
        filename = ouput_dir / model_name
        save_dict = {
            const.PREF_MODEL_STATE_DICT: self.model.state_dict(),
            **{
                const.PREF_OPTIM_STATE_DICT + key: optim.state_dict()
                for (key, optim) in self.optimizer_dict.items()
            },
            **{
                const.PREF_LR_SCHEDULER_STATE_DICT + key: scheduler.state_dict()
                for (key, scheduler) in self.lr_scheduler_dict.items()
            },
        }
        torch.save(save_dict, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")

    def _resume(self) -> None:
        self.logger.info(f"Resuming training. Loading weights: {self.config.resume}")
        self.model.load_weights(self.config.resume)
        state_dict = torch.load(self.config.resume)
        for suffix, optim in self.optimizer_dict.items():
            # NOTE: might crash for GANs, but on the other hand, model_best will not be stored for GANs
            optim.load_state_dict(state_dict[const.PREF_OPTIM_STATE_DICT + suffix])

        for suffix, scheduler in self.lr_scheduler_dict.items():
            # NOTE: might crash for GANs, but on the other hand, model_best will not be stored for GANs
            scheduler.load_state_dict(
                state_dict[const.PREF_LR_SCHEDULER_STATE_DICT + suffix]
            )

        if (
            train_hist_filename := (self.output_dir / "training_history.csv")
        ).is_file():
            self.loss_tracker.load_csv_file(train_hist_filename)
        self.criterion.update_iter_count(self.loss_tracker.get_iter_count())

    def _log_stats(self, epoch: int, step: int, idx: int, loss: float) -> None:
        self.logger.info(
            f"Epoch: {epoch} | Step: {step:5} {self._progress(idx)} Loss: {loss:.6f}"
        )

    def _progress(self, sample_idx: int) -> str:
        total = self.num_train_samples
        emtpy_spaces = math.ceil(math.log10(total))
        return f"[{sample_idx:{emtpy_spaces}}/{total} ({int(100.0 * sample_idx / total):3}%)]"

    def _step(self, key: str = "") -> None:
        """Perform optimizer and LR-scheduler steps.

        Args:
            key (str, optional): Key to optimizer_dict and lr_scheduler_dict. Defaults to "".
        """
        self.optimizer_dict[key].step()
        self.optimizer_dict[key].zero_grad()
        if self.lr_scheduler_dict.get(key, None) is not None:
            self.lr_scheduler_dict[key].step()

    def store_examples(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        iter_or_epoch: int,
        prepare_batch_fn: Callable[[Tensor, *Tuple[Tensor, ...]], Tuple[Tensor, ...]],
        num_samples: int = 16,
        grid_size: Tuple[int, int] = (4, 4),
        back_to_train_mode: bool = True,
    ) -> None:
        """Store examples of inference.

        Args:
            model (nn.Module): Underlying model.
            data_loader (DataLoader): Data loader to load the data for inference.
            iter_or_epoch (int): Extension for images.
            prepare_batch_fn (Callable[[Tensor, *Tuple[Tensor, ...]], Tensor | Tuple[Tensor, ...]]): Function that
                        prepares the batch for inference. Most important, this should include normalization. It should
                        either return a Tensor or a tuple of Tensors. Nonetheless, the first element of the tuple needs
                        to be the model input, since this is the only thing that is needed in the forward pass.
            num_samples (int, optional): Number of samples that should be plotted. Defaults to 16.
            grid_size (Tuple[int, int], optional): Grid of how the images should be layed (HxW). Defaults to (4, 4).
            back_to_train_mode (bool, optional): Set the model back to train mode afterwards. Defaults to True.
        """
        if not self.what_to_plot:
            return

        assert num_samples == grid_size[0] * grid_size[1]
        if data_loader is None:
            return
        self._train_time_pause()  # train time without storing examples

        data_loader.eval()
        model.eval()
        inputs = []
        outputs = []
        targets = []
        ffc_input_output = []
        ffc_target_output = []

        conditions_1 = (
            []
        )  # only look at at a single condition channel (if there are more, ignore them)
        store_samples = True
        with torch.no_grad():
            for (
                batch
            ) in (
                data_loader
            ):  # batch should look have form (data, target, <optional_stuff>)
                if not store_samples:
                    break

                prepared_batch = prepare_batch_fn(
                    *batch, data_normalizer=self.data_normalizer
                )
                data, target = (
                    (prepared_batch[0], prepared_batch[1])
                    if type(prepared_batch) == tuple
                    else (prepared_batch, None)
                )
                output = model(data)
                inputs += [
                    img[0].cpu() for img in data
                ]  # should become HxW from NxCxHxW, with N=batch_size
                outputs += [
                    img[0].cpu() for img in output
                ]  # should become HxW from NxCxHxW, with N=batch_size
                ffc_input_output += [
                    ffc.correct_flatfield(inp[0].cpu(), out[0].cpu())
                    for inp, out in zip(data, output)
                ]
                if data.shape[1] > 1:  # input has conditions
                    conditions_1 += [img[1].cpu() for img in data]

                if target is not None:
                    targets += [
                        img[0].cpu() for img in target
                    ]  # should become HxW from NxCxHxW, with N=batch_size
                    ffc_target_output += [
                        ffc.correct_flatfield(targ[0].cpu(), out[0].cpu())
                        for targ, out in zip(target, output)
                    ]
                store_samples = len(outputs) < num_samples

        # plot inputs and outputs as grid
        filename_outputs = (
            self.output_dir_plots / "D_outputs" / f"{iter_or_epoch:3}.{const.FILE_EXT}"
        )
        self._plot_grid_and_single(
            outputs, grid_size, filename_outputs, Path(f"D_outputs/{iter_or_epoch}")
        )

        # flat-field corrected input
        ffc_holo_file = (
            self.output_dir_plots / "E_ffc_holo" / f"{iter_or_epoch:3}.{const.FILE_EXT}"
        )
        self._plot_grid_and_single(  # using different threshold
            ffc_input_output,
            grid_size,
            ffc_holo_file,
            Path(f"E_ffc_holo/{iter_or_epoch}"),
            threshold=5.0,
        )

        # flat-field corrected target
        if targets != []:
            # NOTE: F-prefix for a folder already exists, however, but too much effor to change
            ffc_holo_file = self.output_dir_plots / f"ffc_targets.{const.FILE_EXT}"
            self._plot_grid_and_single(
                ffc_target_output,
                grid_size,
                ffc_holo_file,
                Path(f"F_ffc_target/{iter_or_epoch}"),
            )

        # store once in the beginnning, since these will not change
        inputs_file = self.output_dir_plots / f"inputs.{const.FILE_EXT}"
        store_inputs_and_targets = (
            not inputs_file.is_file()
        )  # do not store inputs and targets multiple times
        if store_inputs_and_targets:
            self._plot_grid_and_single(inputs, grid_size, inputs_file, "A_inputs")
            if targets != []:
                targets_file = inputs_file = (
                    self.output_dir_plots / f"targets.{const.FILE_EXT}"
                )
                self._plot_grid_and_single(
                    targets, grid_size, targets_file, "B_targets"
                )
            if conditions_1 != []:
                conditions_file = self.output_dir_plots / f"conditions.{const.FILE_EXT}"
                self._plot_grid_and_single(
                    conditions_1, grid_size, conditions_file, "C_conditions"
                )

        if targets == []:
            targets = [None] * len(
                outputs
            )  # fill dummy values to be able to iterate over them

        for i, (input, output, target) in enumerate(zip(inputs, outputs, targets)):
            filename = f"{i:02}.{const.FILE_EXT}"

            if "L" in self.what_to_plot or self.what_to_plot == "all":
                self._plot_L(
                    filename, input=input, output=output, step=iter_or_epoch, line=line
                )

            line = output.shape[0] // 2  # to plot pixel values along this line
            if "K" in self.what_to_plot or self.what_to_plot == "all":
                # had to change the naming from C to K, because now I also store conditions
                self._plot_K(filename, output=output, step=iter_or_epoch, line=line)

            if target is not None:
                if "E" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_E(
                        filename,
                        target=target,
                        output=output,
                        step=iter_or_epoch,
                        line=line,
                    )

                if "F" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_F(
                        filename,
                        input=input,
                        target=target,
                        output=output,
                        step=iter_or_epoch,
                        line=line,
                    )

                if "G" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_G(
                        filename,
                        input=input,
                        target=target,
                        output=output,
                        step=iter_or_epoch,
                        line=line,
                    )

                if "H" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_H(
                        filename,
                        input=input,
                        target=target,
                        output=output,
                        step=iter_or_epoch,
                    )

                if "I" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_I(
                        filename, input=input, output=output, step=iter_or_epoch
                    )

                if "J" in self.what_to_plot or self.what_to_plot == "all":
                    self._plot_J(
                        filename, target=target, output=output, step=iter_or_epoch
                    )

                if store_inputs_and_targets and (
                    "Z" in self.what_to_plot or self.what_to_plot == "all"
                ):
                    self._plot_Z(filename, target=target, step=iter_or_epoch)

        if back_to_train_mode:
            model.train()
            data_loader.train()
        self._train_time_start()  # train time without storing examples

    def _train_time_start(self) -> None:
        self.temp_time_delta = datetime.now()

    def _train_time_pause(self) -> None:
        if self.temp_time_delta is not None:
            self.plain_train_time += datetime.now() - self.temp_time_delta
            self.temp_time_delta = None  # reset to 0 (None to check in if condition)

    ################################################## PLOTTING BELOW ##################################################

    def _plot_grid_and_single(
        self,
        imgs: List[Tensorlike],
        grid_size: Tuple[int, int],
        output_file_grid: Pathlike,
        output_subfolder: Pathlike,
        threshold: float | None = None,
    ) -> None:
        output_folder_single = ensure_dir(self.output_dir_plots / output_subfolder)
        output_folder_single_plain = ensure_dir(output_folder_single / "plain")
        plotting.plot_grid(imgs, grid_size=grid_size, filename=output_file_grid)
        # store single conditions
        for i, img in enumerate(imgs):
            filename = f"{i:02}.{const.FILE_EXT}"
            fig, ax = plotting.subplots(1, 1)
            plotting.imshow_colorbar(ax, img, vmax=threshold)
            fig.savefig(output_folder_single / filename)
            plt.close(fig)
            plotting.imsave_min_max(
                output_folder_single_plain / filename, img, threshold=threshold
            )

    # NOTE: `plotting.subplot_and_store` not integrated
    def _plot_K(
        self, filename: str, output: Tensor, step: int, sub_path: str = ""
    ) -> None:
        # store single output
        fig, ax = plotting.subplots(1, 1)
        plotting.imshow_colorbar(ax, fig, output)
        fig.savefig(
            ensure_dir(self.output_dir_plots / sub_path / "K_output_single" / str(step))
            / filename
        )
        plt.close(fig)

    # NOTE: `plotting.subplot_and_store` not integrated
    def _plot_L(
        self,
        filename: str,
        input: Tensor,
        output: Tensor,
        step: int,
        line: int,
        sub_path: str = "",
    ) -> None:
        # store: input next to output with lines
        fig, axs = plotting.subplots(1, 3)
        plotting.imshow_colorbar(axs[0], fig, input)
        plotting.imshow_colorbar(axs[1], fig, output)
        axs[0].axhline(line)
        axs[1].plot(output[line, :])
        axs[0].axhline(line, color=mpl_constants.MPL_COLORS["blue"])
        axs[1].axhline(line, color=mpl_constants.MPL_COLORS["orange"])
        axs[2].plot(input[line, :], label="Input")
        axs[2].plot(output[line, :], label="Output", alpha=0.8)
        axs[2].legend(loc="upper right")
        axs[2].axhline(0, linestyle="--", color="black", alpha=0.5)
        axs[2].axhline(1, linestyle="--", color="black", alpha=0.5)
        axs[0].set_title("Input")
        axs[1].set_title("Output")
        axs[2].set_title("Pixel Values")
        axs[2].set_xlabel("Pixel")
        axs[2].set_ylabel("Value / A.U.")
        fig.savefig(
            ensure_dir(
                self.output_dir_plots
                / sub_path
                / "L_input_vs_output_with_line"
                / str(step)
            )
            / filename
        )
        plt.close(fig)

    # NOTE: `plotting.subplot_and_store` not integrated
    def _plot_E(
        self,
        filename: str,
        target: Tensor,
        output: Tensor,
        step: int,
        line: int,
        sub_path: str = "",
    ) -> None:
        # store: target next to output and with lines
        fig, axs = plotting.subplots(1, 3)
        plotting.imshow_colorbar(axs[0], fig, target).set_title("Target")
        plotting.imshow_colorbar(axs[1], fig, output).set_title("Output")
        axs[0].axhline(line, color=mpl_constants.MPL_COLORS["blue"])
        axs[1].axhline(line, color=mpl_constants.MPL_COLORS["orange"])

        axs[2].set_title("Values")
        axs[2].plot(target[line, :], label="Target")
        axs[2].plot(output[line, :], label="Output", alpha=0.8)
        axs[2].legend(loc="upper right")
        axs[2].axhline(0, linestyle="--", color="black", alpha=0.5)
        axs[2].axhline(1, linestyle="--", color="black", alpha=0.5)

        fig.savefig(
            ensure_dir(
                self.output_dir_plots / sub_path / "E_target_vs_output" / str(step)
            )
            / filename
        )
        plt.close(fig)

    def _plot_F(
        self,
        filename: str,
        input: Tensor,
        target: Tensor,
        output: Tensor,
        step: int,
        line: int,
        sub_path: str = "",
    ) -> None:
        # store: input, output, target with lines
        output_dir = ensure_dir(
            self.output_dir_plots / sub_path / "F_in_vs_target_vs_out" / str(step)
        )

        fig, axs = plotting.subplots(2, 2)
        imgs = [input, target, output]
        titles = [const.ANNOT_MODEL_IN, const.ANNOT_MODEL_TARGET, const.ANNOT_MODEL_OUT]
        colors = ["blue", "orange", "green"]

        for i in range(len(imgs)):
            plotting.subplot_and_store(
                axs.flat[i],
                plot_func=plotting.imshow_colorbar,
                title=titles[i],
                output_dir=output_dir,
                filename=filename,
                axhline_wrapper=plotting.axhline_wrapper(
                    line, color=mpl_constants.MPL_COLORS[colors[i]]
                ),
                data=imgs[i],
            )

        plotting.subplot_and_store(
            axs[1, 1],
            plotting.plot_line_values,
            title="Comparison",
            output_dir=output_dir,
            filename=filename,
            data=imgs,
            line=line,
            labels=titles,
            do_upper_lower=True,
        )

        fig.savefig(output_dir / filename)
        plt.close(fig)

        # also store line values in csv file for later things
        data = {label: img[line, :] for img, label in zip(imgs, titles)}
        df = pd.DataFrame(data)
        output_file_lines = (
            ensure_dir(output_dir / "line_data") / filename
        ).with_suffix(".csv")
        df.to_csv(output_file_lines, index=False)

    # NOTE: `plotting.subplot_and_store` not integrated
    def _plot_G(
        self,
        filename: str,
        input: Tensor,
        target: Tensor,
        output: Tensor,
        step: int,
        line: int,
        sub_path: str = "",
    ) -> None:
        # store: input, output, target with lines
        fig, axs = plotting.subplots(2, 2)
        plotting.imshow_colorbar(axs[0, 0], fig, input).set_title("Input")
        plotting.imshow_colorbar(axs[0, 1], fig, target).set_title("Target")
        plotting.imshow_colorbar(axs[1, 0], fig, output).set_title("Output")
        axs[0, 0].axhline(line, color=mpl_constants.MPL_COLORS["blue"])
        axs[0, 1].axhline(line, color=mpl_constants.MPL_COLORS["orange"])
        axs[1, 0].axhline(line, color=mpl_constants.MPL_COLORS["green"])

        axs[1, 1].plot(
            input[line, :],
            color=mpl_constants.MPL_COLORS["blue"],
            label="Input",
            alpha=0.5,
        )
        axs[1, 1].plot(
            target[line, :], color=mpl_constants.MPL_COLORS["orange"], label="Target"
        )
        axs[1, 1].plot(
            output[line, :],
            color=mpl_constants.MPL_COLORS["green"],
            label="Output",
            alpha=0.8,
        )
        axs[1, 1].legend(loc="upper right")

        axs[1, 1].axhline(0, linestyle="--", color="black", alpha=0.5)
        axs[1, 1].axhline(1, linestyle="--", color="black", alpha=0.5)
        # add lines for min/max values of target and outout
        axs[1, 1].axhline(
            target.min(),
            linestyle="--",
            color=mpl_constants.MPL_COLORS["orange"],
            alpha=0.5,
            label="Min/max Target",
        )
        axs[1, 1].axhline(
            target.max(),
            linestyle="--",
            color=mpl_constants.MPL_COLORS["orange"],
            alpha=0.5,
        )
        axs[1, 1].axhline(
            output.min(),
            linestyle=":",
            color=mpl_constants.MPL_COLORS["green"],
            alpha=0.5,
            label="Min/max Output",
        )
        axs[1, 1].axhline(
            output.max(),
            linestyle=":",
            color=mpl_constants.MPL_COLORS["green"],
            alpha=0.5,
        )
        axs[1, 1].legend(loc="upper right")

        # save second version WITH min/max of output/target
        fig.savefig(
            ensure_dir(
                self.output_dir_plots
                / sub_path
                / "G_in_vs_target_vs_out_minmax"
                / str(step)
            )
            / filename
        )
        plt.close(fig)

    def _plot_H(
        self, filename: str, input: Tensor, target: Tensor, output: Tensor, step: int
    ) -> None:
        # store: input, output, target with lines, but ZOOMED in
        size = input.shape[0] // 4
        rect = (size, size, size, size)  # should be the same for all
        line = size // 2
        input_crop = F.crop(input, *rect)
        target_crop = F.crop(target, *rect)
        output_crop = F.crop(output, *rect)

        sub_path = "H_zoomed_in"
        self._plot_F(
            filename,
            input=input_crop,
            target=target_crop,
            output=output_crop,
            line=line,
            step=step,
            sub_path=sub_path,
        )
        self._plot_I(
            filename, input=input_crop, output=output_crop, step=step, sub_path=sub_path
        )
        self._plot_J(
            filename,
            target=target_crop,
            output=output_crop,
            step=step,
            sub_path=sub_path,
        )

    def _plot_I(
        self,
        filename: str,
        input: Tensor,
        output: Tensor,
        step: int,
        sub_path: str = "",
    ) -> None:
        # do flat-field correction on validation data
        output_dir = ensure_dir(
            self.output_dir_plots / sub_path / "I_FFC_holo" / str(step)
        )
        plotting.plot_ffc_process(
            input,
            output,
            output_dir / filename,
            data_title=const.ANNOT_MODEL_IN,
            show_mean=True,
        )

    def _plot_J(
        self,
        filename: str,
        target: Tensor,
        output: Tensor,
        step: int,
        sub_path: str = "",
    ) -> None:
        # do flat-field correction on validation data
        output_dir = ensure_dir(
            self.output_dir_plots / sub_path / "J_FFC_target" / str(step)
        )
        plotting.plot_ffc_process(
            target,
            output,
            output_dir / filename,
            data_title=const.ANNOT_MODEL_TARGET,
            show_stats=True,
        )

    # NOTE: `plotting.subplot_and_store` not integrated
    def _plot_Z(
        self, filename: str, target: Tensor, step: int, sub_path: str = ""
    ) -> None:
        fig, axs = plotting.subplots(1, 2)
        plotting.imshow_colorbar(axs[0], fig, target)
        line = target.shape[0] // 2
        axs[0].axhline(line)
        axs[1].plot(target[line, :])
        axs[1].axhline(0, linestyle="--", color="black", alpha=0.5)
        axs[1].axhline(1, linestyle="--", color="black", alpha=0.5)
        axs[1].set_xlabel("Pixel Values")
        axs[1].set_ylabel("Value / A.U.")
        axs[0].set_title("Target")
        axs[1].set_title("Pixel")

        output_dir = ensure_dir(
            self.output_dir_plots / sub_path / "Z_target_with_line" / str(step)
        )
        fig.savefig(output_dir / filename)
        plt.close(fig)
