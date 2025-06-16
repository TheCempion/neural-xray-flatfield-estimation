# standard libraries
import glob
import pickle
from pathlib import Path
from typing import Optional

# third party libraries
import torch
from torch import Tensor
import numpy as np
from sklearn.decomposition import FastICA

# local libraries
from utils.torch_settings import get_torch_device
from utils.datatypes import Pathlike
import utils.constants as const
from utils.flatfield_correction import set_lower_bound

from livereco_server.api.base.calc_flatfield_pca_io import calc_flatfield_pca
from livereco_server.api.serialization.params_serializer import ParamsSerializer
from livereco_server.api.parameters import FlatfieldPcaParams
from livereco_server.api.parameters import FlatfieldCorrectionParams


__all__ = [
    "calc_pca_for_measurement",
    "load_pca_model",
    "get_pca_params",
    "get_ffc_params",
    "calc_synth_flatfield_pca",
    "correct_flatfield_pca",
]


def calc_pca_for_measurement(
    refs_root_dir: Pathlike,
    output_dir_pca: Pathlike,
    num_pca_components: int = 28,
    prefix: str = "ref_",
    pca_filename: Optional[str] = None,
) -> FlatfieldCorrectionParams:

    flatfield_pca_params = get_pca_params(
        refs_root_dir,
        output_dir_pca,
        num_pca_components,
        prefix=prefix,
        pca_filename=pca_filename,
    )
    pca_file = flatfield_pca_params.save_path

    params_dir = Path(output_dir_pca)

    # image not used/ needed, hence, None
    flatfield_correction_params = FlatfieldCorrectionParams(
        image=None, pca_path=pca_file
    )

    ParamsSerializer.serialize(
        flatfield_pca_params, params_dir / "flatfield_pca_params.pkl"
    )
    ParamsSerializer.serialize(
        flatfield_correction_params, params_dir / "flatfield_correction_params.pkl"
    )

    if not Path(pca_file).is_file():
        calc_flatfield_pca(flatfield_pca_params)
    return flatfield_correction_params


def get_pca_params(
    refs_root_dir: Pathlike,
    output_dir_pca: Pathlike,
    num_pca_components: int,
    prefix: str = "ref_",
    pca_filename: Optional[str] = None,
) -> FlatfieldPcaParams:
    refs_root_dir = str(refs_root_dir)
    output_dir_pca = str(output_dir_pca)

    ref_files = glob.glob(f"{refs_root_dir}/{prefix}*")
    ref_files.sort()

    pca_file = f"{output_dir_pca}/{const.PCA_FILE_NAME if pca_filename is None else pca_filename}"
    Path(pca_file).parents[0].mkdir(exist_ok=True)
    flatfield_pca_params = FlatfieldPcaParams(
        measurements=ref_files,
        num_components=num_pca_components,
        save_path=pca_file,
    )
    return flatfield_pca_params


def get_ffc_params(
    refs_root_dir: Pathlike, output_dir_pca: Pathlike, num_pca_components: int = 28
) -> FlatfieldCorrectionParams:
    flatfield_pca_params = get_pca_params(
        refs_root_dir, output_dir_pca, num_pca_components
    )
    pca_file = flatfield_pca_params.save_path
    flatfield_correction_params = FlatfieldCorrectionParams(
        image=None, pca_path=pca_file
    )
    return flatfield_correction_params


def load_pca_model(flatfield_correction_params: FlatfieldCorrectionParams) -> FastICA:
    with open(flatfield_correction_params.pca_path, "rb") as file:
        pca_model = pickle.load(file)
    return pca_model


def calc_synth_flatfield_pca(
    image: np.ndarray, pca_model: FastICA, log_space: bool = False
) -> np.ndarray:
    image = set_lower_bound(Tensor(image), log_space=log_space)
    if not log_space:
        image = torch.log(image)
    cur_data = image.reshape(np.prod(image.shape)).to(get_torch_device())
    pca_model.mean_ = Tensor(pca_model.mean_).to(get_torch_device())
    pca_model.components_ = Tensor(pca_model.components_).to(get_torch_device())

    synthetic_flat_field = cur_data - pca_model.mean_
    synthetic_flat_field = torch.matmul(
        pca_model.components_.float(), synthetic_flat_field.float()
    )
    synthetic_flat_field = torch.matmul(
        torch.transpose(pca_model.components_.float(), 0, 1),
        synthetic_flat_field.float(),
    )
    synthetic_flat_field += pca_model.mean_
    synthetic_flat_field = synthetic_flat_field.reshape(image.shape)
    return torch.exp(synthetic_flat_field).cpu().numpy()
