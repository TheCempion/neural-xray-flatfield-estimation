# standard libraries
from pathlib import Path
from typing import List
from logging import Logger

# third party libraries
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import FastICA
from tqdm import tqdm
import pandas as pd
import h5py

# local packages
import utils.constants as const
from utils.datatypes import Pathlike, callback_t, stats_callback_t
from utils.flatfield_correction import correct_flatfield
import utils.fileIO as fileIO
import utils.pca as pca
import utils.plotting as plot
from utils import ensure_dir
from utils.data_normalization import DataNormalizer
import model.loss as loss_fn

from utils.holowizard_livereco_server.api.parameters import (
    FlatfieldCorrectionParams as FFCParams,
)


__all__ = [
    "Callbacks",
]


class Callbacks:
    def __init__(
        self,
        logger: Logger,
        basepath: Pathlike,
        model_name: str,
        data_normalizer: DataNormalizer,  # TODO: Docstring
        *,
        callbacks: List[str],
        statistical_callbacks: List[str],
        pca_file: Pathlike,
        eval_on_ground_truth: bool = False,
        basepath_suffix: str = "",
        override_hdf5: bool = False,
        override_stats: bool = False,
        override_plots: bool = False,
    ):
        """Analize model performance by evaluating the output based on different metrics.

        Args:
            logger (Logger): Logger.
            basepath (Pathlike): Output directory, where the evaluation results are written to.
            model_name (str): Model that is evaluated. Used as subfolder in `basepath`.
            callbacks (List[str]): List of class method names to apply for each forward pass individually.
            statistical_callbacks (List[str]):  List of class method names to apply on all results. Start with "stats".
            pca_file (Pathlike): File, that contains the pca-model. Used for benchmarking.
            eval_on_ground_truth (bool, optional): If True, then the target is known and can be used to directly compare
                        the SFF with the ground truth label. Defaults to `False`.
            basepath_suffix (str, optional): Add an optional suffix to the `model_name` to distinguish between
                        different evaluations of the same model, e.g. on different data. Defaults to "".
            override_hdf5 (bool, optional): If True, override existing hdf5-file with all processed data.
                        Defaults to False.
            override_stats (bool, optional): If False, skip callbacks that were already run. Defaults to False.
            override_plots (bool, optional): If False, skip callbacks that were already run. Defaults to False.
        """
        self.logger = logger
        self.basepath = Path(basepath) / f"{model_name}{basepath_suffix}"
        self.data_normalizer = data_normalizer

        self.callbacks = self._load_callbacks(callbacks)
        self.statistical_callbacks = self._load_statistical_callbacks(
            statistical_callbacks
        )
        self.pca_model = self._load_pca_model(pca_file=pca_file)
        self.eval_on_ground_truth = eval_on_ground_truth
        self.override_hdf5 = override_hdf5
        self.override_plots = override_plots
        self.override_stats = override_stats

        self.hdf5_filename = self.basepath / "results.hdf5"

        self.basepath_all_single_images = ensure_dir(self.basepath / "A_single_images")

        self.basepath_multiplots = ensure_dir(self.basepath / "I_Multiplots")
        self.basepath_stats = ensure_dir(self.basepath / "J_Stats")

        # initialize all data to be stored
        self.inputs_raw: List[np.ndarray] = (
            []
        )  # unnormalized model input (only preprocessed)
        self.inputs_model: List[np.ndarray] = []  # Model input
        self.outputs_model: List[np.ndarray] = []  # model output
        self.sff_dl: List[np.ndarray] = []  # Rescaled model output
        self.ffc_dl: List[np.ndarray] = []  # FFC with SSF of DL-model
        self.sff_pca: List[np.ndarray] = []  # SSF retrieved with PCA Model
        self.ffc_pca: List[np.ndarray] = []  # FFC with SSF of PCA

        if self.eval_on_ground_truth:
            self.gt_ff: List[np.ndarray] = (
                []
            )  # unnormalized label / flat-field (only preprocessed)
            self.gt_hologram: List[np.ndarray] = (
                []
            )  # Ground truth FFC holograms, i.e. without multiplying by FF
            self.ffc_dl_target: List[np.ndarray] = (
                []
            )  # flatfield correction on known target (= gt_ff)
            self.ffc_pca_target: List[np.ndarray] = (
                []
            )  # flatfield correction on known target (= gt_ff)

        self.conditions: List[np.ndarray] = []  # possibly add conditions
        self.conditions_raw: List[np.ndarray] = []  # possibly add conditions

    def __call__(
        self,
        input: Tensor,
        output: Tensor,
        input_raw: Tensor,
        gt_ff: Tensor | None = None,
        gt_hologram: Tensor | None = None,
    ) -> None:
        """Add idata for callbacks.

        Args:
            input (Tensor): Model Input, i.e. raw hologram.
            output (Tensor): Model Output, i.e. synthetic flat-field.
            input_raw (Tensor): Model Input before normalization to [0, 1].
        """
        if self.hdf5_filename.is_file():
            raise FileExistsError(
                f"{self.hdf5_filename=} already exists. Need some override flag."
            )

        model_input = input.cpu().numpy()
        model_output = output.cpu().numpy()
        input_raw = input_raw.cpu().numpy()

        if self.eval_on_ground_truth:
            assert gt_ff is not None
            assert gt_hologram is not None
            gt_ff = gt_ff.cpu().numpy()
            gt_hologram = gt_hologram.cpu().numpy()
        else:
            gt_ff = [None] * len(input_raw)
            gt_hologram = [None] * len(input_raw)

        remove_dim = lambda a: (
            a[0] if a.ndim == 3 else a
        )  # matplotlib expects 2D arrays
        for input, output, raw_in, raw_out, gt_holo in zip(
            model_input, model_output, input_raw, gt_ff, gt_hologram
        ):
            if input.shape[0] > 1:  # has conditions:
                self.conditions.append(input[1:, :, :])
                self.conditions_raw.append(input_raw[1:, :, :])

            input, output, raw_in = map(
                remove_dim, (input, output, raw_in)
            )  # reduce dimension if needed

            sff_dl = remove_dim(
                self.data_normalizer.invert(torch.tensor(output), torch.tensor(raw_in))
            )
            ffc_dl = correct_flatfield(image=raw_in, synthetic_flat_field=sff_dl)
            sff_pca = pca.calc_synth_flatfield_pca(
                image=raw_in, pca_model=self.pca_model
            )  # TODO: stop time for this operation
            ffc_pca = correct_flatfield(image=raw_in, synthetic_flat_field=sff_pca)

            # Append processed results to respective lists
            self.inputs_raw.append(np.array(raw_in))
            self.inputs_model.append(np.array(input))
            self.outputs_model.append(np.array(output))
            self.sff_dl.append(np.array(sff_dl))
            self.ffc_dl.append(np.array(ffc_dl))
            self.sff_pca.append(np.array(sff_pca))
            self.ffc_pca.append(np.array(ffc_pca))

            if self.eval_on_ground_truth:
                raw_out = remove_dim(raw_out)
                self.gt_ff.append(np.array(raw_out))
                self.gt_hologram.append(np.array(remove_dim(gt_holo)))
                self.ffc_dl_target.append(correct_flatfield(raw_out, sff_dl))
                self.ffc_pca_target.append(correct_flatfield(raw_out, sff_pca))

    def __len__(self):
        return len(self.inputs_model)

    def _get_class_methods(self) -> List[str]:
        return [
            name
            for name, obj in Callbacks.__dict__.items()
            if callable(obj) and not name.startswith("_") and name != "evaluate"
        ]

    def _load_callbacks(self, callbacks: List[str] | str) -> List[callback_t]:
        cbs = []
        if not callbacks:
            return cbs
        methods = [name for name in self._get_class_methods() if name.startswith("cb")]
        if callbacks == "all" or "all" in callbacks:
            callbacks = methods
        for callback_name in callbacks:
            cbs.append(getattr(self, callback_name))
        return cbs

    def _load_statistical_callbacks(
        self, callbacks: List[str] | str
    ) -> List[stats_callback_t]:
        cbs = []
        if not callbacks:
            return cbs
        methods = [
            name for name in self._get_class_methods() if name.startswith("stats")
        ]
        if callbacks == "all" or "all" in callbacks:
            callbacks = methods
        for callback_name in callbacks:
            cbs.append(getattr(self, callback_name))
        return cbs

    # TODO: Rethink usefulness and/or what other stats might be of interest
    def _prepare_stats(self) -> None:
        ffc_dl = np.array(self.ffc_dl)
        ffc_pca = np.array(self.ffc_pca)

        self.means_ffc_model = ffc_dl.mean(axis=0)
        self.stds_ffc_model = ffc_dl.std(axis=0)
        self.means_ffc_pca = ffc_pca.mean(axis=0)
        self.stds_ffc_pca = ffc_pca.std(axis=0)

    def _load_pca_model(self, pca_file: Pathlike) -> FastICA:
        ffc_params = FFCParams(image=None, pca_path=pca_file)
        pca_model = pca.load_pca_model(ffc_params)
        return pca_model

    def _dont_override(self, path: Path) -> bool:
        """Check if output can be overriden.

        Args:
            path (Path): Directory to check if that exists.

        Returns:
            bool: True, if the path exists and does not need to be overriden.
        """
        return not self.override and not path.exists() and not any(path.iterdir())

    def cb_store_every_single_onle(self, idx: int) -> None:
        raise NotImplementedError("Need to implement before running tests")
        # TODO: store every image (from every list) with cbar and then plain with csv-file
        # TODO: BUT actually: do NOT make this as callbacks, but as a flag-that can be passed, like: store_all_as_single
        #       but set it to True, to have flexibility
        #       Additionally, if real world data, store SFF_DL each as TIFF
        #       Also stor ethe min/max values in csv-files. One file for each dataset (compare with `stats_to_csv`)
        # TODO: Need to include possible conditions

    def cb_multiplot_forward_pass_dl(self, idx: int) -> None:
        output_dir = self.basepath_multiplots / "A_Forward_Pass_DL"
        if self._dont_override(output_dir):
            return

        input_raw, sff_dl, ffc_dl = (
            self.inputs_raw[idx],
            self.sff_dl[idx],
            self.ffc_dl[idx],
        )
        fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
        fig.suptitle("Flat-field Correction (DL)")
        plot.imshow_colorbar(ax1, fig, data=input_raw, title="Raw Hologram")
        plot.imshow_colorbar(ax2, fig, data=sff_dl, title="Synthetic Flat-field")
        plot.imshow_colorbar(ax3, fig, data=ffc_dl, title="Flat-field Corrected")
        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_model_in_out(self, idx: int) -> None:
        input, output = self.inputs_model[idx], self.outputs_model[idx]
        fig, (ax1, ax2) = plot.subplots(1, 2)
        fig.suptitle("DL Model Input and Output")

        plot.imshow_colorbar(ax1, fig, data=input, title="Normalized Input")
        plot.imshow_colorbar(ax2, fig, data=output, title="Output")

        output_dir = self.basepath_multiplots / "B_Input_Output_Model"
        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_forward_pass_pca(self, idx: int) -> None:
        output_dir = self.basepath_multiplots / "C_Forward_Pass_PCA"
        if self._dont_override(output_dir):
            return
        input_raw, sff_pca, ffc_pca = (
            self.inputs_raw[idx],
            self.sff_pca[idx],
            self.ffc_pca[idx],
        )
        fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
        fig.suptitle("Flat-field Correction (PCA)")
        plot.imshow_colorbar(ax1, fig, data=input_raw, title="Raw Hologram")
        plot.imshow_colorbar(ax2, fig, data=sff_pca, title="Synthetic Flat-field")
        plot.imshow_colorbar(ax3, fig, data=ffc_pca, title="Flat-field Corrected")
        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_ffc_comparison(self, idx: int) -> None:
        output_dir = self.basepath_multiplots / "D_FFC_DL_vs_PCA"
        if self._dont_override(output_dir):
            return
        input, ffc_dl, ffc_pca = (
            self.inputs_raw[idx],
            self.ffc_dl[idx],
            self.ffc_pca[idx],
        )
        fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
        fig.suptitle("Flat-field Correction DL vs. PCA")
        plot.imshow_colorbar(ax1, fig, data=input, title="Raw Hologram")
        plot.imshow_colorbar(
            ax2, fig, data=ffc_dl, title="Flat-field Corrected (Model)"
        )
        plot.imshow_colorbar(ax3, fig, data=ffc_pca, title="Flat-field Corrected (PCA)")
        fileIO.savefig(output_dir=output_dir, idx=idx)

    @plot.change_mpl_settings("font", size=14)
    def cb_multiplot_raw_in_out_sff_ffc(self, idx: int) -> None:
        """Plot the Raw image, model input, model output and the SFF in the first row and the histograms in the second.

        Args:
            idx (int): Index of sample.
        """
        output_dir = self.basepath_multiplots / "E_Raw_In_Out_SFF_FFC"
        if self._dont_override(output_dir):
            return

        raw = self.inputs_raw[idx]
        input = self.inputs_model[idx]
        output = self.outputs_model[idx]
        sff = self.sff_dl[idx]
        ffc = self.ffc_dl[idx]

        if self.eval_on_ground_truth:
            gt_ff = self.gt_ff[idx]

            fig, axs = plot.subplots(3, 4)

            # first row
            plot.imshow_colorbar(axs[0, 0], fig, data=raw, title="Raw Hologram")
            plot.histogram(axs[0, 1], data=raw, title="Raw Hologram")

            plot.imshow_colorbar(axs[0, 2], fig, data=gt_ff, title="GT Flat-field")
            plot.histogram(axs[0, 3], data=gt_ff, title="GT Flat-field")

            # second row
            plot.imshow_colorbar(axs[1, 0], fig, data=input, title="Model Input")
            plot.histogram(axs[1, 1], data=input, title="Model Input")

            plot.imshow_colorbar(axs[1, 2], fig, data=sff, title="SFF")
            plot.histogram(axs[1, 3], data=sff, title="SFF")

            # third row
            plot.imshow_colorbar(axs[2, 0], fig, data=output, title="Model Output")
            plot.histogram(axs[2, 1], data=output, title="Model Output")

            plot.imshow_colorbar(axs[2, 2], fig, data=ffc, title="FFC")
            plot.histogram(axs[2, 3], data=ffc, title="FFC")

        else:

            fig, axs = plot.subplots(2, 5)
            plot.imshow_colorbar(axs[0, 0], fig, data=raw, title="Raw Hologram")
            plot.imshow_colorbar(axs[0, 1], fig, data=input, title="Model Input")
            plot.imshow_colorbar(axs[0, 2], fig, data=output, title="Model Output")
            plot.imshow_colorbar(axs[0, 3], fig, data=sff, title="SFF")
            plot.imshow_colorbar(axs[0, 4], fig, data=ffc, title="FFC")

            plot.histogram(axs[1, 0], data=raw, title="")
            plot.histogram(axs[1, 1], data=input, title="")
            plot.histogram(axs[1, 2], data=output, title="")
            plot.histogram(axs[1, 3], data=sff, title="")
            plot.histogram(axs[1, 4], data=ffc, title="")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    @plot.change_mpl_settings("font", size=14)
    def cb_multiplot_raw_in_gtff_sff_gtffc_ffc(self, idx: int) -> None:
        """Plot the Raw image, model input, grount truth flat-flatfield (target), SFF, the target FFC hologram, and the
        FCC with the model, and the corresponding histograms.

        Note:
            This plot will only be created when the ground truths are known, i.e. self.eval_on_ground_truth is known.

        Args:
            idx (int): Index of sample.
        """
        if not self.eval_on_ground_truth:
            return

        output_dir = (
            self.basepath_multiplots / "F_Raw_In_GTFF_SFF_GTFFC_FFC"
        )  # GT := Grount truth
        if self._dont_override(output_dir):
            return

        raw = self.inputs_raw[idx]
        input = self.inputs_model[idx]
        gt_ff = self.gt_ff[idx]
        sff = self.sff_dl[idx]
        ffc = self.ffc_dl[idx]
        gt_hologram = self.gt_hologram[idx]

        fig, axs = plot.subplots(3, 4)

        # first row
        plot.imshow_colorbar(axs[0, 0], fig, data=raw, title="Raw Hologram")
        plot.histogram(axs[0, 1], data=raw, title="Raw Hologram")

        plot.imshow_colorbar(axs[0, 2], fig, data=input, title="Model Input")
        plot.histogram(axs[0, 3], data=input, title="Model Input")

        # second row
        plot.imshow_colorbar(axs[1, 0], fig, data=gt_ff, title="GT Flat-field")
        plot.histogram(axs[1, 1], data=gt_ff, title="GT Flat-field")

        plot.imshow_colorbar(axs[1, 2], fig, data=sff, title="SFF")
        plot.histogram(axs[1, 3], data=sff, title="SFF")

        # third row
        plot.imshow_colorbar(axs[2, 0], fig, data=gt_hologram, title="GT FFC")
        plot.histogram(axs[2, 1], data=gt_hologram, title="GT FFC")

        plot.imshow_colorbar(axs[2, 2], fig, data=ffc, title="FFC")
        plot.histogram(axs[2, 3], data=ffc, title="FFC")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    @plot.change_mpl_settings("font", size=14)
    def cb_multiplot_raw_in_gtff_sff_gtffc_ffc_wo_hist(self, idx: int) -> None:
        """Plot the Raw image, model input, grount truth flat-flatfield (target), SFF, the target FFC hologram, and the
        FCC with the model, WITHOUT the corresponding histograms.

        Note:
            This plot will only be created when the ground truths are known, i.e. self.eval_on_ground_truth is known.

        Args:
            idx (int): Index of sample.
        """
        if not self.eval_on_ground_truth:
            return

        output_dir = (
            self.basepath_multiplots / "G_Raw_In_GTFF_SFF_GTFFC_FFC_wo_hist"
        )  # GT := Grount truth
        if self._dont_override(output_dir):
            return

        raw = self.inputs_raw[idx]
        input = self.inputs_model[idx]
        gt_ff = self.gt_ff[idx]
        sff = self.sff_dl[idx]
        ffc = self.ffc_dl[idx]
        gt_hologram = self.gt_hologram[idx]

        fig, axs = plot.subplots(2, 3)

        # first row
        plot.imshow_colorbar(axs[0, 0], fig, data=raw, title="Raw Hologram")
        plot.imshow_colorbar(axs[0, 1], fig, data=gt_ff, title="GT Flat-field")
        plot.imshow_colorbar(axs[0, 2], fig, data=gt_hologram, title="GT FFC")

        # second row
        plot.imshow_colorbar(axs[1, 0], fig, data=input, title="Model Input")
        plot.imshow_colorbar(axs[1, 1], fig, data=sff, title="SFF")
        plot.imshow_colorbar(axs[1, 2], fig, data=ffc, title="FFC")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_gtffc_vs_ffc(self, idx: int) -> None:
        """Plot the groud truth flat-field corrected hologram vs. the flat-field corrected.
        Note:
            This plot will only be created when the ground truths are known, i.e. self.eval_on_ground_truth is known.

        Args:
            idx (int): Index of sample.
        """
        if not self.eval_on_ground_truth:
            return

        output_dir = self.basepath_multiplots / "H_GTFFC_vs_FFC"  # GT := Grount truth
        if self._dont_override(output_dir):
            return

        gt_hologram = self.gt_hologram[idx]
        ffc = self.ffc_dl[idx]

        fig, axs = plot.subplots(1, 3)

        row = ffc.shape[0] // 2
        plot.imshow_colorbar(axs[0, 0], fig, data=gt_hologram, title="GT FFC")
        plot.imshow_colorbar(axs[0, 1], fig, data=ffc, title="FFC")
        axs[0, 0].axhline(row, linestyle="dashed", lw=1, color="red")
        axs[0, 1].axhline(row, linestyle="dashed", lw=1, color="blue")
        axs[0, 2].plot(gt_hologram[row], color="red", label="GT FFC")
        axs[0, 2].plot(ffc[row], color="blue", label="FFC")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_difference_map_gtff_sff(self, idx: int) -> None:
        if not self.eval_on_ground_truth:
            return

        output_dir = self.basepath_multiplots / "I_GTFF-SFF"  # GT := Grount truth
        if self._dont_override(output_dir):
            return

        gt_ff = self.gt_ff[idx]
        sff = self.sff_dl[idx]
        fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
        plot.imshow_colorbar(ax1, fig, data=gt_ff, title="GT FF")
        plot.imshow_colorbar(ax2, fig, data=sff, title="SFF")
        plot.show_difference_map(ax3, fig, gt=gt_ff, other=sff, title="GT_FF - SFF")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_multiplot_difference_map_gtffc_ffc(self, idx: int) -> None:
        if not self.eval_on_ground_truth:
            return

        output_dir = self.basepath_multiplots / "J_GTFFC-FFC"  # GT := Grount truth
        if self._dont_override(output_dir):
            return

        gt_ffc = self.gt_hologram[idx]
        ffc = self.ffc_dl[idx]
        fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
        plot.imshow_colorbar(ax1, fig, data=gt_ffc, title="GT FFC")
        plot.imshow_colorbar(ax2, fig, data=ffc, title="FFC")
        plot.imshow_colorbar(ax3, fig, gt=gt_ffc, other=ffc, title="GT_FFC - FFC")

        fileIO.savefig(output_dir=output_dir, idx=idx)

    def cb_single_plots(self, idx: int) -> None:
        """Store all relevant images in different "versions".

        Args:
            idx (int): index of the example.
        """
        data = [
            ("A_Input_Raw", self.inputs_raw, "Hologram"),  # loaded by dataloader
            ("B_Input_Model", self.inputs_model, "DL Model Input"),  # model input
            ("C_Output_Model", self.outputs_model, "DL Model Output"),  # model output
            (
                "D_SFF_DL",
                self.sff_dl,
                "Synthetic Flat-field (DL)",
            ),  # Model output retransformed to raw-scale
            ("E_FFC_DL", self.ffc_dl, "Flat-field Corrected (DL)"),  # := A / D
            ("F_SSF_PCA", self.sff_pca, "Synthetic Flat-field (PCA)"),
            ("G_FFC_PCA", self.ffc_pca, "Flat-field Corrected (PCA)"),
        ]
        for sub_path, imgs, title in data:
            img = imgs[idx]

            output_dir_tiff = ensure_dir(self.basepath / sub_path)
            if self._dont_override(output_dir_tiff):
                continue
            fileIO.save_img(output_dir_tiff / f"{idx:03}.tiff", img)

            # simply store single images as pdf
            plot.plot_single_img(img=img, title=title)
            fileIO.savefig(self.basepath / sub_path / "pdf", idx=idx)

            fig, (ax1, ax2) = plot.subplots(1, 2)
            fig.suptitle(title)

            row = (img.shape[0] - 1) // 2
            line_color = "red"
            ax1.axhline(row, linestyle="dashed", lw=1, color=line_color)
            ax1.imshow(img, cmap=const.CMAP)
            ax1.set_title("Hologram")

            ax2.plot(img[row, :], label=f"Row {row}", color=line_color)
            ax2.set_title(f"Pixel Values Along Selected Row")
            ax2.set_xlabel("Column Index")
            ax2.set_ylabel("Pixel Value")
            ax2.legend()
            fileIO.savefig(self.basepath / sub_path / "pdf_include_lines", idx=idx)

    def stats_to_csv(self) -> None:
        csv_file = self.basepath_stats / "stats_all.csv"
        if csv_file.exists():
            return
        data = {
            "Raw": self.inputs_raw,
            "SFF_DL": self.sff_dl,
            "FFC_DL": self.ffc_dl,
            "SSF_PCA": self.sff_pca,
            "FFC_DL": self.ffc_pca,
        }
        min_max_vals = {
            "idx": list(range(len(self.sff_pca))),
            **{
                f"{sub_path}_{min_max}": []
                for sub_path in data.keys()
                for min_max in ["min", "max", "mean", "std"]
            },
        }
        for col_name, imgs in data.items():
            for img in imgs:
                min_max_vals[f"{col_name}_min"].append(img.min())
                min_max_vals[f"{col_name}_max"].append(img.max())
                min_max_vals[f"{col_name}_mean"].append(img.mean())
                min_max_vals[f"{col_name}_std"].append(img.std())
        df = pd.DataFrame(min_max_vals)
        df.to_csv(csv_file, index=False)

    def stats_mean_std_ffc(self) -> None:
        plot_mean_std_ffc_path = self.basepath_stats / "plot_mean_std_ffc.pdf"
        plot_mean_std_ffc_shaded_path = (
            self.basepath_stats / "plot_mean_std_ffc_shaded.pdf"
        )
        if (
            not self.override
            and plot_mean_std_ffc_path.exists()
            and plot_mean_std_ffc_shaded_path.exists()
        ):
            return

        fig, (ax1, ax2) = plot.subplots(1, 2, sharex=True)
        fig.suptitle("Mean and Std of Flat-field Corrected Holograms")
        ax1.set_title("Mean")
        ax1.plot(self.means_ffc_model, label="DL")
        ax1.plot(self.means_ffc_pca, label="PCA")
        ax1.set_xlabel("Index of Sample")
        ax1.set_ylabel("Value")
        ax1.legend(loc="upper right")

        ax2.set_title("Std")
        ax2.plot(self.stds_ffc_model, label="DL")
        ax2.plot(self.stds_ffc_pca, label="PCA")
        ax2.set_xlabel("Index of Sample")
        ax2.set_ylabel("Value")
        ax2.legend(loc="upper right")
        plt.savefig(plot_mean_std_ffc_path)

        fig, (ax1, ax2) = plot.subplots(1, 2, sharex=True)
        fig.suptitle("Mean and Std of Flat-field Corrected Holograms")
        x = np.linspace(0, len(self.means_ffc_model) - 1, len(self.means_ffc_model))
        ax1.set_title("DL Model")
        ax1.plot(self.means_ffc_model, label="Mean")
        ax1.fill_between(
            x,
            self.means_ffc_model - self.stds_ffc_model,
            self.means_ffc_model + self.stds_ffc_model,
            color="blue",
            alpha=0.2,
            label="Mean ± Std",
        )
        ax1.set_xlabel("Index of Sample")
        ax1.set_ylabel("Value")
        ax1.legend(loc="upper right")

        ax2.set_title("PCA")
        ax2.plot(self.means_ffc_pca, label="Mean")
        ax2.fill_between(
            x,
            self.means_ffc_pca - self.stds_ffc_pca,
            self.means_ffc_pca + self.stds_ffc_pca,
            color="blue",
            alpha=0.2,
            label="Mean ± Std",
        )
        ax2.set_xlabel("Index of Sample")
        # ax2.set_ylabel("Value")
        ax2.legend(loc="upper right")
        plt.savefig(plot_mean_std_ffc_shaded_path)

    def stats_plot_mean_std_ffc_shaded_DL(self) -> None:
        plot_mean_std_ffc_shaded_path = (
            self.basepath_stats / "plot_mean_std_ffc_shaded_dl.pdf"
        )
        if not self.override and plot_mean_std_ffc_shaded_path.exists():
            return

        fig = plt.figure(figsize=plot.get_figsize())
        x = np.linspace(0, len(self.means_ffc_model) - 1, len(self.means_ffc_model))
        plt.plot(self.means_ffc_model, label="Mean")
        plt.fill_between(
            x,
            self.means_ffc_model - self.stds_ffc_model,
            self.means_ffc_model + self.stds_ffc_model,
            color="blue",
            alpha=0.2,
            label="Mean ± Std",
        )
        plt.xlabel("Index of Sample")
        plt.ylabel("Value")
        plt.legend(loc="upper right")

        plt.savefig(plot_mean_std_ffc_shaded_path)

    def stats_eval_gt_flatfields(self) -> None:
        """Calculate different error metrics to quantify the performances on flat-field generation.

        Possible metrics are:
            - MSE
            - MAE
            - SSIM
            - Total Variance (P_0005:eq.3)

        Note:
            This can only be done if the target is known. Then, the metrics can be calculated for the inputs compared
            with the ground truth.
        """
        if not self.eval_on_ground_truth:
            return

        output_file = self.basepath_stats / "ground_truth_eval_ff.txt"
        if not self.override and output_file.exists():
            return
        total_vars_dl = []
        total_vars_pca = []
        ssim_dl = []
        ssim_pca = []
        mse_dl = []
        mse_pca = []
        l1_dl = []
        l1_pca = []

        ssim_loss_fn = loss_fn.SSIM_Loss()
        mse_loss_fn = lambda x, y: loss_fn.mse_loss_fn()(
            torch.tensor(x), torch.tensor(y)
        )
        l1_loss_fn = lambda x, y: loss_fn.l1_loss_fn()(torch.tensor(x), torch.tensor(y))

        add_dims = lambda array: torch.tensor(array).unsqueeze(0).unsqueeze(0)
        div_by = lambda x, y: (
            add_dims(x / max(x.max(), y.max())),
            add_dims(y / max(x.max(), y.max())),
        )

        for target, sff_dl, sff_pca in zip(self.gt_ff, self.sff_dl, self.sff_pca):
            divisor = target.max()
            target /= divisor
            sff_dl /= divisor
            sff_pca /= divisor

            total_vars_dl.append(np.var(target - sff_dl))
            total_vars_pca.append(np.var(target - sff_pca))

            ssim_dl.append(ssim_loss_fn(*div_by(target, sff_dl)))
            ssim_pca.append(ssim_loss_fn(*div_by(target, sff_pca)))

            mse_dl.append(mse_loss_fn(target, sff_dl))
            mse_pca.append(mse_loss_fn(target, sff_pca))

            l1_dl.append(l1_loss_fn(target, sff_dl))
            l1_pca.append(l1_loss_fn(target, sff_pca))

        all_losses = [
            torch.tensor(loss)
            for loss in [
                total_vars_dl,
                total_vars_pca,
                ssim_dl,
                ssim_pca,
                mse_dl,
                mse_pca,
                l1_dl,
                l1_pca,
            ]
        ]
        stats = {
            "Loss": [
                "Total Variance DL",
                "Total Variance PCA",
                "SSIM DL",
                "SSIM PCA",
                "MSE DL",
                "MSE PCA",
                "L1 DL",
                "L1 PCA",
            ],
            "Means": [loss.mean().item() for loss in all_losses],
            "Std": [loss.std().item() for loss in all_losses],
        }
        pd.DataFrame(stats).to_csv(output_file, index=False, sep="\t")

        for counter, (loss_name, loss) in enumerate(zip(stats["Loss"], all_losses)):
            if counter % 2 == 0:
                plt.figure(figsize=plot.get_figsize())
                plt.title(" ".join(loss_name.split(" ")[:-1]))
            plt.plot(loss, label=f"{loss_name.split(' ')[-1]}")

            if counter % 2 == 1:
                plt.legend()
                # TODO: Need to remove the type, e.g. pca/model from the name
                plt.savefig(
                    self.basepath_stats
                    / f"{' '.join(loss_name.lower().split(' ')[:-1])}.pdf"
                )
                plt.close()

    def stats_eval_gt_ssim_ffc(self) -> None:
        """Calculate different error metrics to quantify the performances on flat-field generation.

        Note:
            This can only be done if the target is known. Then, the metrics can be calculated for the inputs compared
            with the ground truth.
        """
        if not self.eval_on_ground_truth:
            return

        output_file = self.basepath_stats / "ground_truth_eval_ssim_eval.pdf"
        if not self.override and output_file.exists():
            return
        ssim_dl = []
        ssim_pca = []

        ssim_loss_fn = loss_fn.SSIM_Loss()

        add_dims = lambda array: torch.tensor(array).unsqueeze(0).unsqueeze(0)
        div_by = lambda x, y: (
            add_dims(x / max(x.max(), y.max())),
            add_dims(y / max(x.max(), y.max())),
        )

        for gt_ffc, ffc_dl, ffc_pca in zip(self.gt_ff, self.ffc_dl, self.ffc_pca):
            divisor = gt_ffc.max()
            gt_ffc /= divisor
            ffc_dl /= divisor
            ffc_pca /= divisor

            ssim_dl.append(ssim_loss_fn(*div_by(gt_ffc, ffc_dl)))
            ssim_pca.append(ssim_loss_fn(*div_by(gt_ffc, ffc_pca)))

        plt.figure(figsize=plot.get_figsize())
        plt.plot(ssim_dl, label="DL")
        plt.plot(ssim_pca, label="PCA")
        plt.legend(loc="upper right")
        plt.xlabel("Sample")
        plt.ylabel("SSIM / A.U.")
        plt.title("SSIM of Flat-field Corrected Holograms")
        plt.savefig(output_file)
        plt.close()
