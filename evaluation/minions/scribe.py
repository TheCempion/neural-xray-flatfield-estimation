# standard libraries
from typing import List, Tuple
import time
from pathlib import Path

# third party libraries
import numpy as np
from torch import Tensor
import torch
from sklearn.decomposition import FastICA
import h5py

# local packages
from utils.datatypes import Pathlike
from utils.flatfield_correction import correct_flatfield
import utils.pca as pca
from utils import ensure_dir
from utils.torch_settings import get_torch_device
from utils.data_normalization import DataNormalizer

from livereco_server.api.parameters import FlatfieldCorrectionParams as FFCParams


__all__ = [
    "Scribe",
]


class Scribe:
    def __init__(
        self,
        output_path: Pathlike,
        data_normalizer: DataNormalizer,
        pca_file: Pathlike,
        eval_on_gt: bool,
        override: bool = False,
    ):
        self.output_path = ensure_dir(output_path)
        self._hdf5_filename = self.output_path / "results.hdf5"
        self.data_normalizer = data_normalizer

        self.pca_model = self._load_pca_model(pca_file=pca_file)
        self.total_inference_time_pca = 0
        self.override = override
        self.eval_on_gt = eval_on_gt

        self.device = get_torch_device()
        self.already_did_inference = self.hdf5_filename.is_file() and not self.override

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
        self.sff_pca_normalized: List[np.ndarray] = []  # normalized SFF of PCA

        if self.eval_on_gt:
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
            self.normed_target: List[np.ndarray] = (
                []
            )  # target as it would be for training

        self.conditions: List[np.ndarray] = []  # possibly add conditions
        self.conditions_raw: List[np.ndarray] = []  # possibly add conditions

    def __len__(self):
        return len(self.inputs_model)

    def _load_pca_model(self, pca_file: Pathlike) -> FastICA:
        ffc_params = FFCParams(image=None, pca_path=pca_file)
        pca_model = pca.load_pca_model(ffc_params)
        return pca_model

    def _calc_sff_pca(self, img: np.ndarray) -> np.ndarray:
        t_0 = time.time()
        sff = pca.calc_synth_flatfield_pca(image=img, pca_model=self.pca_model)
        self.total_inference_time_pca += time.time() - t_0
        return sff

    def get_inference_time_pca(self) -> Tuple[float, float]:
        return self.total_inference_time_pca, self.total_inference_time_pca / len(self)

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
        if not self.override and self.hdf5_filename.is_file():
            return

        synth_flatfield_dl = (
            self.data_normalizer.invert(output, input_raw.to(self.device)).cpu().numpy()
        )
        model_input = input.cpu().numpy()
        model_output = output.cpu().numpy()
        input_raw = input_raw.cpu().numpy()

        if self.eval_on_gt:
            assert gt_ff is not None
            assert gt_hologram is not None
            _, normed_target = self.data_normalizer(
                torch.tensor(input_raw, device=self.device), gt_ff.to(self.device)
            )
            normed_target = normed_target.cpu().numpy()
            gt_ff = gt_ff.cpu().numpy()
            gt_hologram = gt_hologram.cpu().numpy()

        remove_dim = lambda a: (
            a[0] if a.ndim == 3 else a
        )  # matplotlib expects 2D arrays

        for i in range(len(model_input)):
            input = model_input[i]
            output = model_output[i]
            raw_in = input_raw[i]
            sff_dl = synth_flatfield_dl[i]

            if input.shape[0] > 1:  # has conditions:
                self.conditions.append(input[1:, :, :])
                self.conditions_raw.append(raw_in[1:, :, :])

            # reduce dimension if needed
            input, output, raw_in, sff_dl = map(
                remove_dim, (input, output, raw_in, sff_dl)
            )

            ffc_dl = correct_flatfield(image=raw_in, synthetic_flat_field=sff_dl)
            sff_pca = self._calc_sff_pca(raw_in)
            ffc_pca = correct_flatfield(image=raw_in, synthetic_flat_field=sff_pca)
            _, sff_pca_normalized = self.data_normalizer(
                torch.tensor(raw_in, device=self.device).unsqueeze(0).unsqueeze(0),
                torch.tensor(sff_pca, device=self.device).unsqueeze(0).unsqueeze(0),
            )
            sff_pca_normalized = sff_pca_normalized[0, 0, ...].cpu()

            # Append processed results to respective lists
            self.inputs_raw.append(np.array(raw_in))
            self.inputs_model.append(np.array(input))
            self.outputs_model.append(np.array(output))
            self.sff_dl.append(np.array(sff_dl))
            self.ffc_dl.append(np.array(ffc_dl))
            self.sff_pca.append(np.array(sff_pca))
            self.ffc_pca.append(np.array(ffc_pca))
            self.sff_pca_normalized.append(np.array(sff_pca_normalized))

            if self.eval_on_gt:
                raw_out = remove_dim(gt_ff[i])
                self.gt_ff.append(raw_out)
                self.gt_hologram.append(remove_dim(gt_hologram[i]))
                self.ffc_dl_target.append(correct_flatfield(raw_out, sff_dl))
                self.ffc_pca_target.append(correct_flatfield(raw_out, sff_pca))
                self.normed_target.append(remove_dim(normed_target[i]))

    def to_hdf5_file(self) -> Path:
        if self.hdf5_filename.is_file() and not self.override:
            raise FileExistsError(
                f"File {self.hdf5_filename} already exists. Set `override_hdf5` to override."
            )

        with h5py.File(self.hdf5_filename, "w") as hdf5:
            # full-scale images
            hdf5_full_scale = hdf5.create_group("full_scale")
            hdf5_full_scale.create_dataset(
                "inputs_raw", data=np.array(self.inputs_raw), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "inputs_model", data=np.array(self.inputs_model), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "outputs_model", data=np.array(self.outputs_model), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "sff_dl_normalized", data=np.array(self.outputs_model), dtype=np.float32
            )  # same as outputs model
            hdf5_full_scale.create_dataset(
                "sff_dl", data=np.array(self.sff_dl), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "ffc_dl", data=np.array(self.ffc_dl), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "sff_pca", data=np.array(self.sff_pca), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "ffc_pca", data=np.array(self.ffc_pca), dtype=np.float32
            )
            hdf5_full_scale.create_dataset(
                "sff_pca_normalized",
                data=np.array(self.sff_pca_normalized),
                dtype=np.float32,
            )

            if self.eval_on_gt:
                hdf5_full_scale.create_dataset(
                    "gt_ff", data=np.array(self.gt_ff), dtype=np.float32
                )
                hdf5_full_scale.create_dataset(
                    "gt_hologram", data=np.array(self.gt_hologram), dtype=np.float32
                )
                hdf5_full_scale.create_dataset(
                    "ffc_dl_target", data=np.array(self.ffc_dl_target), dtype=np.float32
                )
                hdf5_full_scale.create_dataset(
                    "ffc_pca_target",
                    data=np.array(self.ffc_pca_target),
                    dtype=np.float32,
                )
                hdf5_full_scale.create_dataset(
                    "normed_target", data=np.array(self.normed_target), dtype=np.float32
                )

            if self.conditions != []:
                hdf5_full_scale.create_dataset(
                    "conditions", data=np.array(self.conditions), dtype=np.float32
                )
                hdf5_full_scale.create_dataset(
                    "conditions_raw",
                    data=np.array(self.conditions_raw),
                    dtype=np.float32,
                )

            # store cropped patches
            crop_params = {
                "center_small": (True, (768, 768, 512, 512)),
                "center_large": (False, (512, 512, 1024, 1024)),
                "left": (False, (768, 0, 512, 512)),
                "right": (False, (768, 1536, 512, 512)),
                "top": (False, (0, 768, 512, 512)),
                "bottom": (False, (1536, 768, 512, 512)),
            }
            crop = lambda a, p: np.array(a)[..., p[0] : p[0] + p[2], p[1] : p[1] + p[3]]

            hdf5_cropped = hdf5.create_group("cropped")
            for area_name, (do_crop, params) in crop_params.items():
                if (
                    self.eval_on_gt and not do_crop
                ):  # not (not self.eval_on_gt or do_crop) (basically: not (A => B))
                    continue  # only do multiple crops on experimental data
                g_cropped = hdf5_cropped.create_group(area_name)
                g_cropped.create_dataset(
                    "inputs_raw", data=crop(self.inputs_raw, params), dtype=np.float32
                )
                g_cropped.create_dataset(
                    "inputs_model",
                    data=crop(self.inputs_model, params),
                    dtype=np.float32,
                )
                g_cropped.create_dataset(
                    "outputs_model",
                    data=crop(self.outputs_model, params),
                    dtype=np.float32,
                )
                g_cropped.create_dataset(
                    "sff_dl_normalized",
                    data=crop(self.outputs_model, params),
                    dtype=np.float32,
                )  # same as outputs model
                g_cropped.create_dataset(
                    "sff_dl", data=crop(self.sff_dl, params), dtype=np.float32
                )
                g_cropped.create_dataset(
                    "ffc_dl", data=crop(self.ffc_dl, params), dtype=np.float32
                )
                g_cropped.create_dataset(
                    "sff_pca", data=crop(self.sff_pca, params), dtype=np.float32
                )
                g_cropped.create_dataset(
                    "ffc_pca", data=crop(self.ffc_pca, params), dtype=np.float32
                )
                g_cropped.create_dataset(
                    "sff_pca_normalized",
                    data=crop(self.sff_pca_normalized, params),
                    dtype=np.float32,
                )

                if self.eval_on_gt:
                    g_cropped.create_dataset(
                        "gt_ff", data=crop(self.gt_ff, params), dtype=np.float32
                    )
                    g_cropped.create_dataset(
                        "gt_hologram",
                        data=crop(self.gt_hologram, params),
                        dtype=np.float32,
                    )
                    g_cropped.create_dataset(
                        "ffc_dl_target",
                        data=crop(self.ffc_dl_target, params),
                        dtype=np.float32,
                    )
                    g_cropped.create_dataset(
                        "ffc_pca_target",
                        data=crop(self.ffc_pca_target, params),
                        dtype=np.float32,
                    )
                    g_cropped.create_dataset(
                        "normed_target",
                        data=crop(self.normed_target, params),
                        dtype=np.float32,
                    )

                if self.conditions != []:
                    g_cropped.create_dataset(
                        "conditions",
                        data=crop(self.conditions, params),
                        dtype=np.float32,
                    )
                    g_cropped.create_dataset(
                        "conditions_raw",
                        data=crop(self.conditions_raw, params),
                        dtype=np.float32,
                    )
        return self.hdf5_filename

    @property
    def hdf5_filename(self) -> Path:
        return self._hdf5_filename
