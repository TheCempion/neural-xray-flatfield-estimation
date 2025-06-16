# standard libraries
from pathlib import Path
from typing import Dict, List
from logging import Logger

# third party libraries
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import torch

# local packages
from utils import ensure_dir
from utils.torch_settings import get_torch_device
import utils.plotting as plotting
import utils.constants as const
import model.loss.loss as loss
from utils.data_normalization import NormalizeByMax


__all__ = [
    "Statistician",
]


class Statistician:
    def __init__(self, output_path: Path, eval_on_gt: bool, override: bool):
        self.output_path = ensure_dir(output_path / "stats")
        self.hdf5_filename = output_path / "results.hdf5"
        self.eval_on_gt = eval_on_gt
        self.override = override
        self.crop_types = [
            "center_small",
            "center_large",
            "left",
            "right",
            "top",
            "bottom",
        ]
        self.csv_dir_profiles = None  # will be set later in store_profiles

        with h5py.File(self.hdf5_filename, "r") as hdf5:
            # Load full-scale data
            self.full_scale: Dict[str, np.ndarray] = {
                name: ds[:] for name, ds in hdf5["full_scale"].items()
            }

            # Load cropped data
            self.cropped: Dict[str, Dict[str, np.ndarray]] = {
                area_name: {name: ds[:] for name, ds in area.items()}
                for area_name, area in hdf5["cropped"].items()
            }

    def run_all(self, logger: Logger) -> None:
        logger.info("Running all statistical methods.")
        logger.info("Calculating base stats...")
        self.base_stats_ffc_on_target()
        logger.info("Done.")
        logger.info("Plotting stats...")
        self.plot_statistics()
        logger.info("Done.")
        logger.info("Scatter stats...")
        self.scatter_statistics()
        logger.info("Done.")
        logger.info("Storing profiles...")
        self.store_profiles()
        logger.info("Done.")
        logger.info("Calculating losses...")
        self.calculate_losses()
        logger.info("Done.")
        logger.info("Creating histograms of FFCs...")
        self.hist_statistics()
        logger.info("Done.")
        logger.info("Done with all statistical methods.")

    def base_stats_ffc_on_target(self) -> Dict[str, Dict[str, np.ndarray]] | None:
        # return stuff if I want to get stats for different models in a single script
        if not self.eval_on_gt:
            return

        if (
            not self.override
            and [f for f in self.output_path.iterdir() if "csv" in f.suffix] == []
        ):
            return

        norm = NormalizeByMax()

        self.stats = {"means": {}, "stds": {}, "vars": {}}
        data_in_scope = ["ffc_dl_target", "ffc_pca_target"]

        for dataset_name in data_in_scope:
            imgs = torch.tensor(
                self.full_scale[dataset_name], device=get_torch_device()
            ).unsqueeze(1)
            normed_imgs = norm(imgs).squeeze(1).cpu().numpy()
            self.stats["means"][f"full_scale_{dataset_name}"] = normed_imgs.mean(
                axis=(-2, -1)
            )
            self.stats["stds"][f"full_scale_{dataset_name}"] = normed_imgs.std(
                axis=(-2, -1)
            )
            self.stats["vars"][f"full_scale_{dataset_name}"] = normed_imgs.var(
                axis=(-2, -1)
            )

            for crop_type in self.crop_types:
                if not crop_type in self.cropped:
                    continue
                key = f"crop_{dataset_name}_{crop_type}"
                imgs = torch.tensor(
                    self.cropped[crop_type][dataset_name], device=get_torch_device()
                ).unsqueeze(1)
                normed_imgs = norm(imgs).squeeze(1).cpu().numpy()
                self.stats["means"][key] = normed_imgs.mean(axis=(-2, -1))
                self.stats["stds"][key] = normed_imgs.std(axis=(-2, -1))
                self.stats["vars"][key] = normed_imgs.var(axis=(-2, -1))

        dfs = {}
        for stat_type, data in self.stats.items():
            dfs[stat_type] = (df := pd.DataFrame(data))
            df.to_csv(self.output_path / f"ffc_on_target_{stat_type}.csv", index=False)

        return self.stats

    def plot_statistics(self) -> None:
        """Plot statistics for means, stds, and vars."""
        if not self.eval_on_gt:
            return

        output_dir = ensure_dir(self.output_path / "plots_basestats")
        if not self.override and any(output_dir.iterdir()):
            return

        if not hasattr(self, "stats"):
            self.base_stats_ffc_on_target()

        data_in_scope = ["ffc_dl_target", "ffc_pca_target"]
        clip_at = 10
        for stat_type, stat_values in self.stats.items():
            # Plot statistics for full-scale (means, stds, vars)
            if all(
                [
                    f"full_scale_{dataset_name}" in stat_values.keys()
                    for dataset_name in data_in_scope
                ]
            ):
                fig, ax = plotting.subplots()
                need_to_clip = False
                for dataset_name in data_in_scope:
                    key = f"full_scale_{dataset_name}"
                    label = "PCA" if "pca" in dataset_name else "DL"
                    ax.plot(stat_values[key], ".", label=label)
                    need_to_clip = need_to_clip or any(
                        [val > clip_at for val in stat_values[key]]
                    )
                ax.legend()
                ax.set_xlabel("Sample Index")
                ax.set_ylabel(stat_type[:-1].capitalize())  # remove the s
                fig.savefig(output_dir / f"full_scale_{stat_type}.{const.FILE_EXT}")
                if need_to_clip:
                    ax.set_ylim(0, clip_at)
                    fig.savefig(
                        output_dir / f"full_scale_{stat_type}_clipped.{const.FILE_EXT}"
                    )
                plt.close(fig)

            # Plot cropped statistics for each crop type
            for crop_type in self.crop_types:
                if all(
                    [
                        f"crop_{dataset_name}_{crop_type}" in stat_values.keys()
                        for dataset_name in data_in_scope
                    ]
                ):
                    fig, ax = plotting.subplots()
                    need_to_clip = False
                    for dataset_name in data_in_scope:
                        key = f"crop_{dataset_name}_{crop_type}"
                        if key in stat_values:
                            label = "PCA" if "pca" in dataset_name else "DL"
                            ax.plot(stat_values[key], ".", label=label)
                            need_to_clip = need_to_clip or any(
                                [val > clip_at for val in stat_values[key]]
                            )
                    ax.legend()
                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel(stat_type.capitalize())
                    fig.savefig(
                        output_dir / f"crop_{crop_type}_{stat_type}.{const.FILE_EXT}"
                    )
                    if need_to_clip:
                        ax.set_ylim(0, clip_at)
                        fig.savefig(
                            output_dir
                            / f"crop_{crop_type}_{stat_type}_clipped.{const.FILE_EXT}"
                        )
                    plt.close(fig)

    # TODO: Create scatter plot for base stats
    def scatter_statistics(self) -> None:
        """Scatter statistics for means, stds, and vars."""
        if not self.eval_on_gt:
            return

        output_dir = ensure_dir(self.output_path / "plots_basestats")
        if not self.override and any(output_dir.iterdir()):
            return

        if not hasattr(self, "stats"):
            self.base_stats_ffc_on_target()

        data_in_scope = ["ffc_dl_target", "ffc_pca_target"]
        clip_at = 10
        for stat_type, stat_values in self.stats.items():
            # Plot statistics for full-scale (means, stds, vars)
            if all(
                [
                    f"full_scale_{dataset_name}" in stat_values.keys()
                    for dataset_name in data_in_scope
                ]
            ):
                for dataset_name in data_in_scope:
                    key = f"full_scale_{dataset_name}"
                    label = "PCA" if "pca" in dataset_name else "DL"

                    # Create a separate scatter plot for each dataset
                    fig, ax = plotting.subplots()
                    ax.scatter(
                        np.arange(len(stat_values[key])),
                        stat_values[key],
                        label=label,
                        marker="o",
                    )

                    # Set labels and legend
                    ax.legend()
                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel(stat_type[:-1].capitalize())  # remove trailing "s"

                    # Save the plot
                    filename = f"full_scale_{stat_type}_{label}.{const.FILE_EXT}"
                    fig.savefig(output_dir / filename)

                    # Handle clipping if needed
                    if stat_values[key].max() > clip_at:
                        ax.set_ylim(0, clip_at)
                        fig.savefig(
                            output_dir
                            / f"full_scale_{stat_type}_{label}_clipped.{const.FILE_EXT}"
                        )
                    plt.close(fig)

            # Cropped statistics plot
            # for crop_type in self.crop_types:
            #     if all([f"crop_{dataset_name}_{crop_type}" in stat_values for dataset_name in data_in_scope]):
            #         fig, ax = plotting.subplots()
            #         need_to_clip = False
            #         for dataset_name in data_in_scope:
            #             key = f"crop_{dataset_name}_{crop_type}"
            #             label = "PCA" if "pca" in dataset_name else "DL"
            #             if key in stat_values:
            #                 ax.scatter(np.arange(len(stat_values[key])), stat_values[key], label=label)
            #                 if stat_values[key].max() > clip_at:
            #                     need_to_clip = True
            #         ax.legend()
            #         ax.set_xlabel("Sample Index")
            #         ax.set_ylabel(stat_type.capitalize())
            #         fig.savefig(output_dir / f"crop_{crop_type}_{stat_type}.{const.FILE_EXT}")

            #         if need_to_clip:
            #             ax.set_ylim(0, clip_at)
            #             fig.savefig(output_dir / f"crop_{crop_type}_{stat_type}_clipped.{const.FILE_EXT}")
            #         plt.close(fig)

    def hist_statistics(self) -> None:
        """Plot statistics for means, stds, and vars."""
        if not self.eval_on_gt:
            return

        output_dir = ensure_dir(self.output_path / "histograms_ffc_on_target")
        if not self.override and any([d.is_dir() for d in output_dir.iterdir()]):
            return

        data_in_scope = ["ffc_dl_target", "ffc_pca_target"]

        def _plot_hist_wrapper(data: np.ndarray, filename: Path) -> None:
            fig, ax = plotting.subplots()
            plotting.histogram(ax, data=data, show_mean=True, show_var=True)
            fig.savefig(filename)
            plt.close(fig)

        # Plot statistics for full-scale (means, stds, vars)
        for dataset_name in data_in_scope:
            sub_dir_path = ensure_dir(output_dir / f"full_scale_{dataset_name}")
            for i, data in enumerate(self.full_scale[dataset_name]):
                _plot_hist_wrapper(data, sub_dir_path / f"{i:04}.{const.FILE_EXT}")
                if (data > 3).any():
                    _plot_hist_wrapper(
                        data[data <= 3],
                        sub_dir_path / f"{i:04}_clipped.{const.FILE_EXT}",
                    )

            # Plot cropped statistics for each crop type
            for crop_type in [
                "center_small",
                "center_large",
                "left",
                "right",
                "top",
                "bottom",
            ]:
                if not crop_type in self.cropped.keys():
                    continue
                sub_dir_path = ensure_dir(
                    output_dir / f"crop_{dataset_name}_{crop_type}"
                )
                for i, data in enumerate(self.cropped[crop_type][dataset_name]):
                    _plot_hist_wrapper(data, sub_dir_path / f"{i:04}.{const.FILE_EXT}")
                    if (data > 3).any():
                        _plot_hist_wrapper(
                            data[data <= 3],
                            sub_dir_path / f"{i:04}_clipped.{const.FILE_EXT}",
                        )

    def store_profiles(self) -> Path:
        self.csv_dir_profiles = ensure_dir(self.output_path / "profiles")

        if not self.override and any(self.csv_dir_profiles.iterdir()):
            return self.csv_dir_profiles

        def _store_profiles_wrapper(
            data: Dict[str, np.ndarray], output_path: Path
        ) -> None:
            num_samples = len(list(data.values())[0])
            raw = {
                i: {} for i in range(num_samples)
            }  # later: store each i in a separate file
            for dataset_name, imgs in data.items():
                for i, img in enumerate(imgs):
                    if img.ndim == 3:
                        img = img[0]
                    line = img.shape[0] // 2
                    profile = img[line, :]
                    raw[i][dataset_name] = profile

            for i, profiles in raw.items():
                csv_filename = ensure_dir(output_path) / f"{i:04}.csv"
                df = pd.DataFrame(profiles)
                df.to_csv(csv_filename, index=False)

        _store_profiles_wrapper(self.full_scale, self.csv_dir_profiles / "full_scale")
        for crop_type, cropped_data in self.cropped.items():
            _store_profiles_wrapper(
                cropped_data, self.csv_dir_profiles / f"crop_{crop_type}"
            )
        return self.csv_dir_profiles

    def calculate_losses(self) -> Path:
        if not self.eval_on_gt:
            self.basepath_losses = None
            return

        self.basepath_losses = ensure_dir(self.output_path / "losses")
        loss_names = [
            "L1_Loss",
            "L2_Loss",
            "SSIM_Loss",
            "DSSIM_Loss",
            "MS_SSIM_Loss",
            "MS_DSSIM_Loss",
            "FRC_Loss",
            "Variance_Loss",
        ]
        losses = {
            loss_name: getattr(loss, loss_name)(lam=1.0) for loss_name in loss_names
        }
        device = get_torch_device()

        def _wrapper(data: np.ndarray, sub_dir: Path) -> None:
            output_dir = ensure_dir(self.basepath_losses / sub_dir)
            for method in ["dl", "pca"]:
                loss_dict = {loss_name: [] for loss_name in losses.keys()}
                for normalized_sff, noralized_gt_ff in zip(
                    data[f"sff_{method}_normalized"], data["normed_target"]
                ):
                    for loss_name, loss_fn in losses.items():
                        sff_tensor = (
                            torch.tensor(normalized_sff, device=device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        gt_ff_tensor = (
                            torch.tensor(noralized_gt_ff, device=device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        l = loss_fn(sff_tensor, gt_ff_tensor)
                        loss_dict[loss_name].append(l.cpu().item())
                df = pd.DataFrame(loss_dict)
                df.to_csv(output_dir / f"losses_{method}.csv", index=False)
                # calculate stats for the losses
                means = {
                    loss_name: [np.array(loss_vals).mean()]
                    for loss_name, loss_vals in loss_dict.items()
                }
                stds = {
                    loss_name: [np.array(loss_vals).std()]
                    for loss_name, loss_vals in loss_dict.items()
                }
                vars = {
                    loss_name: [np.array(loss_vals).var()]
                    for loss_name, loss_vals in loss_dict.items()
                }
                pd.DataFrame(means).to_csv(
                    output_dir / f"losses_{method}_means.csv", index=False
                )
                pd.DataFrame(stds).to_csv(
                    output_dir / f"losses_{method}_stds.csv", index=False
                )
                pd.DataFrame(vars).to_csv(
                    output_dir / f"losses_{method}_vars.csv", index=False
                )

        _wrapper(self.full_scale, "full_scale")
        for crop_type, cropped_data in self.cropped.items():
            _wrapper(cropped_data, f"crop_{crop_type}")

        return self.basepath_losses
