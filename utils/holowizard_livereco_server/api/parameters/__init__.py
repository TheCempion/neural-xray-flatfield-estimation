import sys

member_value_adapter = None

try:
    import torch
except Exception:
    pass

if "torch" in sys.modules:
    from holowizard_livereco_server.api.parameters.type_conversion.member_value_adapter_torch import (
        MemberValueAdapterTorch,
    )

    member_value_adapter = MemberValueAdapterTorch
else:
    from holowizard_livereco_server.api.parameters.type_conversion.member_value_adapter_numpy import (
        MemberValueAdapterNumpy,
    )

    member_value_adapter = MemberValueAdapterNumpy

from holowizard_livereco_server.api.parameters.beam_setup import BeamSetup
from holowizard_livereco_server.api.parameters.data_dimensions import DataDimensions
from holowizard_livereco_server.api.parameters.measurement import Measurement
from holowizard_livereco_server.api.parameters.options import Options
from holowizard_livereco_server.api.parameters.padding import Padding
from holowizard_livereco_server.api.parameters.reco_params import RecoParams
from holowizard_livereco_server.api.parameters.regularization import Regularization
from holowizard_livereco_server.api.parameters.flatfield_components_params import (
    FlatfieldComponentsParams,
)
from holowizard_livereco_server.api.parameters.flatfield_correction_params import (
    FlatfieldCorrectionParams,
)
