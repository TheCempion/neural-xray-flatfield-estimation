# standard libraries
from pathlib import Path
from typing import Dict, List
from logging import Logger

# third party libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# local packages
from utils import ensure_dir
import utils.plotting as plotting
import utils.constants as const
from utils.datatypes import Pathlike
import model.loss.loss as loss
import utils.fileIO as fileIO


__all__ = [
    "Visualizer",
]


class Visualizer:
    def __init__(
        self,
        hdf5_filename: Path,
        eval_on_gt: bool,
        override: bool,
        csv_dir_profiles: Path,
        basepath_losses: Path,
    ):
        self.hdf5_filename = hdf5_filename
        self.output_path = ensure_dir(hdf5_filename.parent / "plots")
        self.eval_on_gt = eval_on_gt
        self.override = override
        self.csv_dir_profiles = csv_dir_profiles
        self.basepath_losses = basepath_losses
        self.crop_types = [
            "center_small",
            "center_large",
            "left",
            "right",
            "top",
            "bottom",
        ]

        with h5py.File(hdf5_filename, "r") as hdf5:
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
        logger.info("Running all visualizer methods.")
        logger.info("Do something with the losses...")
        # self.do_domething_with_the_losses()       # TODO:
        logger.info("Done.")
        logger.info("Plotting profile comparisons...")
        # self.profile_comparison() # TODO:
        logger.info("Done.")
        logger.info("Store EVERY image...")
        self.store_all()
        logger.info("Done.")
        logger.info("Done with all visualizer methods.")

    def store_all(self) -> None:
        def _store_all_wrapper(data: np.ndarray, name: str) -> None:
            output_dir_cbar = ensure_dir(self.output_path / "cbar" / name)
            basepath_plain = ensure_dir(self.output_path / "plain")
            output_dir_plain_pdf = ensure_dir(basepath_plain / "pdf" / name)
            if "full_scale" in name:
                output_dir_plain_tiff = ensure_dir(basepath_plain / "tiff" / name)
            output_dir_with_line = ensure_dir(self.output_path / "with_lines" / name)
            for i, img in enumerate(data):
                filename = f"{i:04}.{const.FILE_EXT}"
                if img.ndim == 3:
                    img = img[
                        0
                    ]  # condition -> just take first channel, because I don't have more
                if self.override or not any(output_dir_cbar.iterdir()):
                    fig, ax = plotting.subplots()
                    match dataset_name:
                        case ds_name if any(
                            [s in ds_name for s in ["inputs_model", "gt_hologram"]]
                        ):
                            vmin = None
                            vmax = 5.0  # holo
                        case ds_name if any(
                            [
                                s in ds_name
                                for s in ["inputs_raw", "sff_dl", "sff_pca", "gt_ff"]
                            ]
                        ):
                            vmin = img.min()
                            vmax = img.max()
                        case _:
                            vmin = None
                            vmax = None
                    plotting.imshow_colorbar(ax, img, vmin=vmin, vmax=vmax)
                    fig.savefig(output_dir_cbar / filename)
                    ax.axhline(img.shape[0] // 2, color="red")
                    fig.savefig(output_dir_with_line / filename)
                    plt.close(fig)
                if self.override or not any(output_dir_plain_pdf.iterdir()):
                    plt.imsave(output_dir_plain_pdf / filename, img)

                    if "full_scale" in name:
                        # store only full-scales as tiff
                        fileIO.save_img(output_dir_plain_tiff / f"{i:04}.tiff", img)
                break

        for dataset_name, data in self.full_scale.items():
            _store_all_wrapper(data, f"full_scale_{dataset_name}")

        for crop_type, cropped_data in self.cropped.items():
            for dataset_name, data in cropped_data.items():
                _store_all_wrapper(data, f"crop_{dataset_name}_{crop_type}")

    def profile_comparison(self) -> None:
        output_dir = ensure_dir(self.output_path / "compare_profiles")

        def _wrapper(
            csv_file_dir: Path,
            sub_dir: Pathlike,
            cols: List[str],
            labels: List[str],
            alphas: List[float] | None = None,
        ) -> None:
            # csv_file_dir contains all enumerated csv files
            if alphas is None:
                alphas = [1.0] * len(cols)

            assert csv_file_dir.is_dir(), f"{csv_file_dir}="
            assert (
                len(cols) == len(labels) == len(alphas)
            ), f"{len(cols)=}, {len(labels)=}, {len(alphas)=}"

            profiles_output_dir = ensure_dir(output_dir / csv_file_dir.name / sub_dir)
            for i, csv_file in enumerate(csv_file_dir.iterdir()):
                if not csv_file.is_file():
                    continue  # safety measure

                output_filename = profiles_output_dir / f"{i:04}.{const.FILE_EXT}"
                if not self.override and output_filename.is_file():
                    continue  # skip file instead of returning from function

                df = pd.read_csv(csv_file)
                fig, ax = plotting.subplots()
                for c, l, a in zip(cols, labels, alphas):
                    ax.plot(df[c], label=l, alpha=a)
                ax.set_xlabel("Pixel")
                ax.set_ylabel("Value / A.U.")
                ax.legend(loc="upper right")
                fig.savefig(output_filename)
                plt.close(fig)

        for csv_dir in self.csv_dir_profiles.iterdir():
            ## ALWAYS (A - H):
            # H_raw vs SFF_DL -> "inputs_raw" vs "sff_dl"
            _wrapper(
                csv_dir,
                "A1_H_raw_vs_SFF_DL",
                ["inputs_raw", "sff_dl"],
                [const.ANNOT_H_RAW, const.ANNOT_SFF],
            )
            _wrapper(
                csv_dir,
                "A2_H_raw_vs_SFF_DL",
                ["inputs_raw", "sff_dl"],
                [const.ANNOT_H_RAW, const.ANNOT_SFF_DL],
            )

            # H_raw vs SFF_PCA -> "inputs_raw", "sff_pca"
            _wrapper(
                csv_dir,
                "B1_H_raw_vs_SFF_PCA",
                ["inputs_raw", "sff_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_SFF],
            )
            _wrapper(
                csv_dir,
                "B2_H_raw_vs_SFF_PCA",
                ["inputs_raw", "sff_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_SFF_PCA],
            )

            # H_raw vs SFF_DL vs SFF_PCA -> "inputs_raw", "sff_dl" vs "sff_pca"
            _wrapper(
                csv_dir,
                "C1_H_raw_vs_SFF_DL_vs_PCA",
                ["inputs_raw", "sff_dl", "sff_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_SFF_DL, const.ANNOT_SFF_PCA],
                alphas=[0.5, 1.0, 0.8],
            )

            # SFF_DL vs SFF_PCA -> "sff_dl", "sff_pca"
            _wrapper(
                csv_dir,
                "D1_SFF_DL_vs_PCA",
                ["sff_dl", "sff_pca"],
                [const.ANNOT_SFF_DL, const.ANNOT_SFF_PCA],
            )

            # H_raw vs FFC_DL vs FFC_PCA -> "inputs_raw", "ffc_dl" vs "ffc_pca"
            _wrapper(
                csv_dir,
                "E1_H_raw_vs_FFC_DL_vs_PCA",
                ["inputs_raw", "ffc_dl", "ffc_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_H_CORR_DL, const.ANNOT_H_CORR_PCA],
                alphas=[0.5, 1.0, 0.8],
            )

            # H_raw vs FFC_DL -> "inputs_raw", "ffc_dl"
            _wrapper(
                csv_dir,
                "F1_H_raw_vs_FFC_DL",
                ["inputs_raw", "ffc_dl"],
                [const.ANNOT_H_RAW, const.ANNOT_H_CORR],
            )
            _wrapper(
                csv_dir,
                "F2_H_raw_vs_FFC_DL",
                ["inputs_raw", "ffc_dl"],
                [const.ANNOT_H_RAW, const.ANNOT_H_CORR_DL],
            )

            # H_raw vs FFC_PCA -> "inputs_raw", "ffc_pca"
            _wrapper(
                csv_dir,
                "G1_H_raw_vs_FFC_PCA",
                ["inputs_raw", "ffc_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_H_CORR],
            )
            _wrapper(
                csv_dir,
                "G2_H_raw_vs_FFC_PCA",
                ["inputs_raw", "ffc_pca"],
                [const.ANNOT_H_RAW, const.ANNOT_H_CORR_PCA],
            )

            # FFC_DL vs FFC_PCA -> "ffc_dl", "ffc_pca"
            _wrapper(
                csv_dir,
                "H1_FFC_DL_vs_PCA",
                ["ffc_dl", "ffc_pca"],
                [const.ANNOT_H_CORR_DL, const.ANNOT_H_CORR_PCA],
            )

            ## IF SELF.EVAL_ON_GT (K - V):
            if self.eval_on_gt:
                # H_raw vs SFF_DL vs target -> "inputs_raw", "sff_dl" vs gt_ff
                _wrapper(
                    csv_dir,
                    "K1_H_raw_vs_Target_vs_SFF_DL",
                    ["inputs_raw", "gt_ff", "sff_dl"],
                    [const.ANNOT_H_RAW, const.ANNOT_FF_REAL, const.ANNOT_SFF],
                    alphas=[0.5, 1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "K2_H_raw_vs_Target_vs_SFF_DL",
                    ["inputs_raw", "gt_ff", "sff_dl"],
                    [const.ANNOT_H_RAW, const.ANNOT_FF_REAL, const.ANNOT_SFF_DL],
                    alphas=[0.5, 1.0, 0.8],
                )

                # H_raw vs SFF_PCA vs target -> "inputs_raw", "sff_pca" vs gt_ff
                _wrapper(
                    csv_dir,
                    "L1_H_raw_vs_Target_vs_SFF_PCA",
                    ["inputs_raw", "gt_ff", "sff_pca"],
                    [const.ANNOT_H_RAW, const.ANNOT_FF_REAL, const.ANNOT_SFF],
                    alphas=[0.5, 1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "L2_H_raw_vs_Target_vs_SFF_PCA",
                    ["inputs_raw", "gt_ff", "sff_pca"],
                    [const.ANNOT_H_RAW, const.ANNOT_FF_REAL, const.ANNOT_SFF_PCA],
                    alphas=[0.5, 1.0, 0.8],
                )

                # H_raw vs SFF_DL vs SFF_PCA vs target  -> "inputs_raw", "sff_dl" vs sff_pca vs gt_ff
                _wrapper(
                    csv_dir,
                    "M1_H_raw_vs_Target_vs_SFF_DL_vs_FF_PCA",
                    ["inputs_raw", "gt_ff", "sff_dl", "sff_pca"],
                    [
                        const.ANNOT_H_RAW,
                        const.ANNOT_FF_REAL,
                        const.ANNOT_SFF_DL,
                        const.ANNOT_SFF_PCA,
                    ],
                    alphas=[0.5, 0.8, 1.0, 0.8],
                )

                # SFF_DL vs SFF_PCA vs target -> "sff_dl", "sff_pca" vs gt_ff
                _wrapper(
                    csv_dir,
                    "N1_Target_vs_SFF_DL_vs_FF_PCA",
                    ["gt_ff", "sff_dl", "sff_pca"],
                    [const.ANNOT_FF_REAL, const.ANNOT_SFF_DL, const.ANNOT_SFF_PCA],
                    alphas=[0.5, 1.0, 0.8],
                )

                # SFF_DL vs target -> "sff_dl", "gt_ff"
                _wrapper(
                    csv_dir,
                    "O1_Target_vs_SFF_DL",
                    ["gt_ff", "sff_dl"],
                    [const.ANNOT_FF_REAL, const.ANNOT_SFF],
                    alphas=[1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "O2_Target_vs_SFF_DL",
                    ["gt_ff", "sff_dl"],
                    [const.ANNOT_FF_REAL, const.ANNOT_SFF_DL],
                    alphas=[1.0, 0.8],
                )

                # SFF_PCA vs target -> "sff_pca", "gt_ff"
                _wrapper(
                    csv_dir,
                    "P1_Target_vs_SFF_PCA",
                    ["gt_ff", "sff_pca"],
                    [const.ANNOT_FF_REAL, const.ANNOT_SFF],
                    alphas=[1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "P2_Target_vs_SFF_PCA",
                    ["gt_ff", "sff_pca"],
                    [const.ANNOT_FF_REAL, const.ANNOT_SFF_PCA],
                    alphas=[1.0, 0.8],
                )

                # FFC_DL_on_target vs FFC_PCA_on_target -> "ffc_dl_target", "ffc_pca_target"
                _wrapper(
                    csv_dir,
                    "Q1_FFC_on_target_DL_vs_PCA",
                    ["ffc_dl_target", "ffc_pca_target"],
                    [const.ANNOT_F_CORR_DL, const.ANNOT_F_CORR_PCA],
                    alphas=[1.0, 0.8],
                )

                # H_raw vs H_gt vs FFC_DL -> "inputs_raw", "gt_hologram" vs ffc_dl
                _wrapper(
                    csv_dir,
                    "R1_H_raw_vs_H_gt_vs_FFC_DL",
                    ["inputs_raw", "gt_hologram", "ffc_dl"],
                    [const.ANNOT_H_RAW, const.ANNOT_H_REAL, const.ANNOT_H_CORR],
                    alphas=[0.5, 1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "R2_H_raw_vs_H_gt_vs_FFC_DL",
                    ["inputs_raw", "gt_hologram", "ffc_dl"],
                    [const.ANNOT_H_RAW, const.ANNOT_H_REAL, const.ANNOT_H_CORR_DL],
                    alphas=[0.5, 1.0, 0.8],
                )

                # H_raw vs H_gt vs FFC_PCA -> "inputs_raw", "gt_hologram" vs ffc_pca
                _wrapper(
                    csv_dir,
                    "S1_H_raw_vs_H_gt_vs_FFC_PCA",
                    ["inputs_raw", "gt_hologram", "ffc_pca"],
                    [const.ANNOT_H_RAW, const.ANNOT_H_REAL, const.ANNOT_H_CORR],
                    alphas=[0.5, 1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "S2_H_raw_vs_H_gt_vs_FFC_PCA",
                    ["inputs_raw", "gt_hologram", "ffc_pca"],
                    [const.ANNOT_H_RAW, const.ANNOT_H_REAL, const.ANNOT_H_CORR_PCA],
                    alphas=[0.5, 1.0, 0.8],
                )

                # H_gt vs FFC_DL vs FFC_PCA -> "gt_hologram", "ffc_dl" vs "ffc_pca"
                _wrapper(
                    csv_dir,
                    "T1_H_gt_vs_FFC_DL_vs_FFC_PCA",
                    ["gt_hologram", "ffc_dl", "ffc_pca"],
                    [const.ANNOT_H_REAL, const.ANNOT_H_CORR_DL, const.ANNOT_H_CORR_PCA],
                    alphas=[0.5, 1.0, 0.8],
                )

                # H_gt vs FFC_DL -> "gt_hologram", "ffc_dl"
                _wrapper(
                    csv_dir,
                    "U1_H_gt_vs_FFC_DL",
                    ["gt_hologram", "ffc_dl"],
                    [const.ANNOT_H_REAL, const.ANNOT_H_CORR],
                    alphas=[1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "U2_H_gt_vs_FFC_DL",
                    ["gt_hologram", "ffc_dl"],
                    [const.ANNOT_H_REAL, const.ANNOT_H_CORR_DL],
                    alphas=[1.0, 0.8],
                )

                # H_gt vs FFC_PCA -> "gt_hologram", "ffc_pca"
                _wrapper(
                    csv_dir,
                    "V1_H_gt_vs_FFC_PCA",
                    ["gt_hologram", "ffc_pca"],
                    [const.ANNOT_H_REAL, const.ANNOT_H_CORR],
                    alphas=[1.0, 0.8],
                )
                _wrapper(
                    csv_dir,
                    "V2_H_gt_vs_FFC_PCA",
                    ["gt_hologram", "ffc_pca"],
                    [const.ANNOT_H_REAL, const.ANNOT_H_CORR_PCA],
                    alphas=[1.0, 0.8],
                )

            ## IF CONDITIONS:   # TODO: Skip for now, dont want to lol
            # cond_raw vs SFF_DL -> conditions_raw vs sff_dl
            # cond_raw vs SFF_DL vs. target -> conditions_raw vs sff_dl vs gt_ff

    def do_domething_with_the_losses(self) -> None:
        if not self.eval_on_gt:
            return

        basepath = ensure_dir(self.output_path / "losses")

        def _wrapper(path: Path) -> None:
            output_path = ensure_dir(basepath / path.name)
            losses_dl = pd.read_csv(path / "losses_dl.csv").to_dict()
            losses_pca = pd.read_csv(path / "losses_pca.csv").to_dict()
            for loss_name in losses_dl.keys():
                loss_fn = getattr(loss, loss_name)()
                key = loss_fn.keys()[0]
                loss_values_dl = losses_dl[loss_name].values()
                loss_values_pca = losses_pca[loss_name].values()

                # normal plot
                fig, ax = plotting.subplots()
                ax.plot(loss_values_dl, label="DL")
                ax.plot(loss_values_pca, label="PCA")
                ax.set_xlabel("Sample")
                ax.set_ylabel(rf"${key}$ / A.U.")
                ax.legend(loc="upper right")
                fig.savefig(output_path / f"plot_{loss_name}.{const.FILE_EXT}")
                plt.close(fig)

                # scatter plot
                fig, ax = plotting.subplots()
                ax.scatter(loss_values_dl, loss_values_pca)
                # ax.set_ylabel(rf"${key}$ / A.U.")
                ax.set_xlabel("DL")
                ax.set_ylabel("PCA")
                # ax.legend(loc="upper right")
                fig.savefig(output_path / f"scatter_{loss_name}.{const.FILE_EXT}")
                plt.close(fig)

        for crop_stype_path in self.basepath_losses.iterdir():
            _wrapper(crop_stype_path)
