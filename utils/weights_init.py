# standard libraries

# third party libraries
import torch.nn as nn

# local packages


__all__ = [
    "weights_init",
]


def weights_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):  # Initialize Conv2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):  # Initialize ConvTranspose2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(
        m, (nn.BatchNorm2d, nn.InstanceNorm2d)
    ):  # Handle normalization layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif hasattr(m, "conv") or hasattr(m, "norm_layer"):  # Handle nested submodules
        if hasattr(m, "conv"):  # Check if it contains a convolutional layer
            submodule = m.conv
            if isinstance(submodule, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(submodule.weight.data, 0.0, 0.02)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias.data, 0)
        if hasattr(m, "norm_layer") and isinstance(
            m.norm_layer, (nn.BatchNorm2d, nn.InstanceNorm2d)
        ):
            nn.init.normal_(m.norm_layer.weight.data, 1.0, 0.02)
            if m.norm_layer.bias is not None:
                nn.init.constant_(m.norm_layer.bias.data, 0)


def weights_init_old(m: nn.Module):
    if isinstance(m, nn.Conv2d):  # Initialize Conv2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):  # Initialize ConvTranspose2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):  # Initialize BatchNorm2d layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif hasattr(m, "conv") or hasattr(
        m, "bn"
    ):  # Handle nested submodules like ConvBN or ConvTransposeBN
        if hasattr(m, "conv"):  # Check if it contains a convolutional layer
            submodule = m.conv
            if isinstance(submodule, nn.Conv2d) or isinstance(
                submodule, nn.ConvTranspose2d
            ):
                nn.init.normal_(submodule.weight.data, 0.0, 0.02)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias.data, 0)
        if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):  # Check for BatchNorm
            nn.init.normal_(m.bn.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bn.bias.data, 0)
