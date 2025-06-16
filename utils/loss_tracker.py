# standard libraries
from typing import Dict, Optional, List, Any
from pathlib import Path

# third party libraries
import matplotlib.pyplot as plt
import pandas as pd

# local packages
from utils.datatypes import Pathlike
import utils.plotting as plotting
import utils.plotting.mpl_constants as mpl_constants
import utils.constants as const

__all__ = [
    "LossTracker",
]


class LossTracker:
    def __init__(
        self,
        ymax: Optional[float] = None,
        plot_other_losses: bool = True,
        smoothing: float = 0.1,
        show_title: bool = False,
    ):
        self.ymax = ymax
        self.plot_other_losses = plot_other_losses
        assert 0 <= smoothing <= 1
        self.smoothing = smoothing
        self.show_title = show_title
        self.train_loss: Dict[int, float] = {}
        self.val_loss: Dict[int, float] = {}
        self.other_losses: Dict[str, dict[int, float]] = {}
        self.train_loss_D: Dict[int, float] = (
            {}
        )  # only for GAN Training; only BCE for discriminator (train loss)
        self.train_loss_G: Dict[int, float] = (
            {}
        )  # only for GAN Training; only BCE for generator (train loss)
        self.val_loss_D: Dict[int, float] = {}  # only for GAN Training (val loss)
        self.val_loss_G: Dict[int, float] = {}  # only for GAN Training (val loss)
        self.variance_losses: Dict[int, float] = (
            {}
        )  # always store the variance loss when validating the model

    def add_train_loss(self, loss: float, train_step: int) -> None:
        self.train_loss[train_step] = loss

    def add_val_loss(self, loss: float, train_step: int) -> None:
        self.val_loss[train_step] = loss

    def add_discriminator_loss(
        self, loss: float, train_step: int, val: bool = False
    ) -> None:
        if val:
            self.val_loss_D[train_step] = loss
        else:
            self.train_loss_D[train_step] = loss

    def add_generator_gan_loss(
        self, loss: float, train_step: int, val: bool = False
    ) -> None:
        if val:
            self.val_loss_G[train_step] = loss
        else:
            self.train_loss_G[train_step] = loss

    def add_other_losses(self, losses: Dict[str, float], train_step: int) -> None:
        for loss, value in losses.items():
            if self.other_losses.get(loss, None) is None:
                self.other_losses[loss] = {}
            self.other_losses[loss][train_step] = value

    def add_variance_loss(self, loss: float, train_step: float) -> None:
        self.variance_losses[train_step] = loss

    def save(self, output_dir: Path) -> None:
        if list(self.train_loss.values()) == []:
            return
        self.save_training_plot(
            output_dir / f"new_training_history.{const.FILE_EXT}", y_logscale=False
        )
        self.save_training_plot(
            output_dir / f"new_training_history_log.{const.FILE_EXT}", y_logscale=True
        )
        self.save_gan_loss_plot(
            output_dir / f"new_training_history_gan_train.{const.FILE_EXT}",
            plot_val_losses=False,
        )
        self.save_gan_loss_plot(
            output_dir / f"new_training_history_gan_val.{const.FILE_EXT}",
            plot_val_losses=True,
        )
        self.save_training_raw(output_dir / f"new_training_history.csv")
        self.save_training_raw_gan_losses(output_dir / f"new_training_history_gan.csv")

        self.save_variance_loss_raw(output_dir / f"new_variance_loss.csv")
        self.save_variance_plot(
            output_dir / f"new_variance_loss.{const.FILE_EXT}", y_logscale=True
        )

    def load_csv_file(self, filename: Pathlike) -> None:
        df = pd.read_csv(filename)

        # Reconstruct train_loss and val_loss dictionaries
        self.train_loss = self._safe_dict(df, "train_loss")
        self.val_loss = self._safe_dict(df, "val_loss")

        # Extract other_losses (handling missing values per column dynamically)
        self.other_losses = {
            col: self._safe_dict(df, col)
            for col in df.columns
            if col not in ["iteration", "train_loss", "val_loss"]
            and self._safe_dict(df, col) != {}
        }
        if (gan_file := Path(filename).parent / "training_history_gan.csv").is_file():
            self.load_csv_file_gan_losses(gan_file)

        if (variances_file := Path(filename).parent / "variance_loss.csv").is_file():
            self.load_csv_file_variance_losses(variances_file)

    def load_csv_file_gan_losses(self, filename: Pathlike) -> None:
        df = pd.read_csv(filename)
        self.train_loss_G = self._safe_dict(df, "train_loss_G")
        self.train_loss_D = self._safe_dict(df, "train_loss_D")
        self.val_loss_G = self._safe_dict(df, "val_loss_G")
        self.val_loss_D = self._safe_dict(df, "val_loss_D")

    def load_csv_file_variance_losses(self, filename: Pathlike) -> None:
        self.variance_losses: Dict[int, float] = self._safe_dict(
            pd.read_csv(filename), "variance_loss"
        )

    def _safe_dict(self, df: pd.DataFrame, column_name: str) -> Dict[int, float]:
        """Create a dictionary while excluding NaN values for each iteration."""
        iterations = df["iteration"].tolist()
        if column_name in df:
            return {
                iteration: value
                for iteration, value in zip(iterations, df[column_name])
                if pd.notna(value)
            }
        return {}

    def save_training_raw(self, filename: Pathlike) -> None:
        df = pd.DataFrame(
            {
                "iteration": self.train_loss.keys(),
                "train_loss": pd.Series(
                    self.train_loss.values(), index=self.train_loss.keys()
                ),
                "val_loss": pd.Series(
                    self.val_loss.values(), index=self.val_loss.keys()
                ),
                **{
                    name: pd.Series(loss.values(), index=loss.keys())
                    for (name, loss) in self.other_losses.items()
                },
            }
        )
        df.to_csv(filename, index=False)

    def save_variance_loss_raw(self, filename: Pathlike) -> None:
        if self.variance_losses == {}:
            return

        df = pd.DataFrame(
            {
                "iteration": self.variance_losses.keys(),
                "variance_loss": pd.Series(
                    self.variance_losses.values(), index=self.variance_losses.keys()
                ),
            }
        )
        df.to_csv(filename, index=False)

    def save_training_raw_gan_losses(self, filename: Pathlike) -> None:
        if self.train_loss_G == {}:
            return

        df = pd.DataFrame(
            {
                "iteration": self.train_loss_G.keys(),
                "train_loss_G": pd.Series(
                    self.train_loss_G.values(), index=self.train_loss_G.keys()
                ),
                "val_loss_G": pd.Series(
                    self.val_loss_G.values(), index=self.val_loss_G.keys()
                ),
                "train_loss_D": pd.Series(
                    self.train_loss_D.values(), index=self.train_loss_D.keys()
                ),
                "val_loss_D": pd.Series(
                    self.val_loss_D.values(), index=self.val_loss_D.keys()
                ),
            }
        )
        df.to_csv(filename, index=False)

    def _smooth(self, loss: List[float]) -> List[float]:
        if 0 <= self.smoothing < 1:  # Apply exponential smoothing
            smoothed_loss = [loss[0]]
            for i in range(1, len(loss)):
                smoothed_value = (
                    self.smoothing * loss[i] + (1 - self.smoothing) * smoothed_loss[-1]
                )
                smoothed_loss.append(smoothed_value)
        else:
            smoothed_loss = loss
        return smoothed_loss

    def _smooth_and_plot(
        self,
        loss_dict: Dict[int, float],
        *,
        color: str,
        label: str,
        reference: List[Any] | None = None,
        alpha: float = 0.3,
    ) -> None:
        x_vals = list(loss_dict.keys())
        loss = list(loss_dict.values())
        x_vals, loss = [], []
        for (
            x_val,
            l,
        ) in (
            loss_dict.items()
        ):  # do not plot any 0-values or Nones, so there will be no jumps in the plot
            if l is not None and l > 0:
                x_vals.append(x_val)
                loss.append(l)

        color = mpl_constants.MPL_COLORS[color]
        if reference is None or len(reference) == len(loss):
            smoothed_loss = self._smooth(loss)
            plt.plot(x_vals, smoothed_loss, color=color, label=label)
            plt.plot(x_vals, loss, color=color, alpha=alpha)
        else:
            plt.plot(x_vals, loss, color=color, label=label)

    def save_training_plot(self, filename: Pathlike, y_logscale: bool) -> None:
        filename = Path(filename)
        plt.figure(figsize=plotting.get_figsize())

        train_loss = list(self.train_loss.values())
        mpl_colors = list(mpl_constants.MPL_COLORS.keys())
        mpl_colors_idx = 0

        self._smooth_and_plot(
            self.train_loss,
            color=mpl_colors[mpl_colors_idx],
            label="Training Loss",
            reference=None,
        )
        mpl_colors_idx += 1

        if (
            val_losses := [loss is not None for loss in self.val_loss.values()]
        ) != [] and all(val_losses):
            color = mpl_colors[mpl_colors_idx]
            mpl_colors_idx += 1
            self._smooth_and_plot(
                self.val_loss,
                color=color,
                label="Validation Loss",
                reference=train_loss,
            )

        plt.xlabel("Training Iteration")
        plt.legend(loc="upper right")
        if y_logscale:
            plt.yscale("log")
            plt.ylabel(r"$\log(Loss)$ / A.U.")
            if self.show_title:
                plt.title("Training History (Log Scale)")
        else:
            plt.ylabel(r"$Loss$ / A.U.")
            if self.show_title:
                plt.title("Training History")

        if self.ymax is not None and not y_logscale:
            upper = min(self.ymax, max(train_loss))
            plt.ylim((0, upper))
        plt.savefig(filename)  # store train and validation loss only

        if len(self.other_losses) > 0:
            for loss_term, loss_term_history in self.other_losses.items():
                color = mpl_colors[mpl_colors_idx]
                mpl_colors_idx += 1
                self._smooth_and_plot(
                    loss_term_history,
                    label=rf"$\mathsf{{{loss_term}}}$",
                    color=color,
                    alpha=0.2,
                    reference=train_loss,
                )
            plt.legend(loc="upper right")
            other_losses_filename = filename.parent / (
                filename.stem + "_multi-loss" + filename.suffix
            )
            plt.savefig(other_losses_filename)  # also store other losses
        plt.close()

    def save_variance_plot(self, filename, y_logscale: bool = True) -> None:
        if self.variance_losses == {}:
            return

        filename = Path(filename)
        plt.figure(figsize=plotting.get_figsize())
        self._smooth_and_plot(
            self.variance_losses,
            color="blue",
            label=None,
            reference=list(self.train_loss.values()),
        )

        plt.xlabel("Training Iteration")
        if y_logscale:
            plt.yscale("log")
            plt.ylabel(r"$\log(Loss)$ / A.U.")
            if self.show_title:
                plt.title("Variance Loss (Log Scale)")
        else:
            plt.ylabel(r"$Loss$ / A.U.")
            if self.show_title:
                plt.title("Variance History")

        if self.ymax is not None and not y_logscale:
            upper = min(self.ymax, max())
            plt.ylim((0, upper))
        plt.savefig(filename)
        plt.close()

    def save_gan_loss_plot(
        self, filename: Pathlike, plot_val_losses: bool = False
    ) -> None:
        if self.train_loss_G == {}:
            return
        elif plot_val_losses and self.val_loss_G == {}:
            return

        filename = Path(filename)
        plt.figure(figsize=plotting.get_figsize())

        mpl_colors = list(mpl_constants.MPL_COLORS.keys())

        if not plot_val_losses:
            loss_G = self.train_loss_G
            loss_D = self.train_loss_D
            if self.show_title:
                plt.title("Training History Generator and Discriminator")
        else:
            loss_G = self.val_loss_G
            loss_D = self.val_loss_D
            if self.show_title:
                plt.title("Validation History Generator and Discriminator")

        self._smooth_and_plot(
            loss_G, color=mpl_colors[0], label="Generator Loss", reference=None
        )
        self._smooth_and_plot(
            loss_D, color=mpl_colors[1], label="Discriminator Loss", reference=None
        )

        plt.legend(loc="upper right")
        plt.xlabel("Training Iteration")
        plt.ylabel(r"$Loss$ / A.U.")
        plt.savefig(filename)  # store train and validation loss only
        plt.close()

    def get_iter_count(self) -> int:
        return len(self.train_loss)
