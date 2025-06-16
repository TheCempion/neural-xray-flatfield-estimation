# Neural X-ray Flat-Field Estimation

This project implements UNet-based neural networks for flat-field estimation in X-ray imaging. It supports encoder initialization with pretrained ImageNet weights (VGG11 and VGG16), includes PatchGAN-based discriminator models, multiple loss functions, and quantitative evaluation metrics. The framework was applied to reconstruct flat-fields from holographic data and benchmarked against PCA-based flat-field estimation.

The code provides tools for training, evaluation, and comparison of flat-field estimation models and can be adapted to various X-ray imaging modalities beyond near-field holography, such as XFEL data.

## Credits

Some functions in this code were adapted from other projects:

- The modules in `utils.holowizard_livereco_server` as well as the `remove_outliers` function in `utils.remove_outliers.py` were taken from the **HoloWizard** package: [Zenodo](https://zenodo.org/records/14024980).
- The PatchGAN implementation is highly inspired by **PhaseGAN**: [GitHub](https://github.com/pvilla/PhaseGAN), [Paper](https://arxiv.org/abs/2011.08660).
