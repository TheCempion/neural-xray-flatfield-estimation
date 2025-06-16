import logging
import pickle
import torch

import livereco
from livereco.core.preprocessing import (
    correct_flatfield as correct_flatfield_internal,
)
from livereco.core.utils.fileio import load_img_data

from holowizard_livereco_server.api.parameters import FlatfieldCorrectionParams


def correct_flatfield(flatfield_correction_params: FlatfieldCorrectionParams):
    with open(flatfield_correction_params.components_path, "rb") as file:
        components = pickle.load(file)

    logging.debug("Load image from " + flatfield_correction_params.image)
    image_to_correct = torch.tensor(
        load_img_data(flatfield_correction_params.image),
        device=livereco.torch_running_device,
    )
    logging.image_info("raw", image_to_correct.cpu().numpy())

    logging.debug("Correct flatfield")

    corrected_img_data = correct_flatfield_internal(image_to_correct, components)

    logging.image_info("flatfield_corrected", corrected_img_data.cpu().numpy())

    return corrected_img_data
