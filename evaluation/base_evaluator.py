# standard libraries
from typing import Dict, Any
from pathlib import Path

# third party libraries

# local packages
import data_loader as module_data
import data_loader.datasets as module_datasets
import model.models as module_arch
import model.gans as module_gan
import utils.data_normalization as module_norm
import utils.torch_settings as torch_settings
from utils import init_dataloader, gpu_has_80GB, ensure_dir
from parse_config import ConfigParser
from base import BaseDataLoader, BaseModel, BaseGAN
from evaluation.minions import Scribe, Statistician, Visualizer


__all__ = [
    "BaseEvaluator",
]


class BaseEvaluator:
    def __init__(self, config: ConfigParser, eval_on_gt: bool):
        self.config = config
        self.logger = config.get_logger("test")
        self.cfg_eval: Dict[str, Any] = config["evalutation"]

        self.basepath = ensure_dir(
            Path(self.cfg_eval["basepath"]) / f"cfg_{self.config.cfg_fname.stem}"
        )
        self.model = self._load_model()
        self.dataloader = self._get_dataloader()
        self.data_normalizer = self._get_data_normalizer()
        self.eval_on_gt = eval_on_gt
        self.scribe = Scribe(
            output_path=self.basepath,
            data_normalizer=self.data_normalizer,
            pca_file=self.cfg_eval["pca_file"],
            eval_on_gt=eval_on_gt,
            override=self.cfg_eval.get("override_hdf5", False),
        )

        self.statistician = None
        self.visiualizer = None

    def _load_model(self) -> BaseModel:
        # build model architecture
        try:
            model: BaseModel = self.config.init_obj("arch", module_arch)
        except AttributeError:
            model: BaseGAN = self.config.init_obj(
                "arch", module_gan
            )  # specifically: BaseGAN
        self.logger.info(model)
        device = torch_settings.get_torch_device()
        model = model.to(device)
        model.eval()
        return model

    def _get_dataloader(self) -> BaseDataLoader:
        if gpu_has_80GB():
            self.config["data_loader"]["args"]["batch_size"] = 6

        dataset = self.config.init_obj("dataset", module_datasets)
        data_loader_class = getattr(module_data, self.config["data_loader"]["type"])

        data_loader, _ = init_dataloader(
            dataset,
            data_loader_class,
            training=False,
            **self.config["data_loader"]["args"],
        )
        return data_loader

    def _get_data_normalizer(self) -> module_norm.DataNormalizer:
        # self.data_normalizer = config.init_obj("data_normalizer", module_norm)
        # if self.config.get("data_normalizer", None) is None:
        #     data_normalizer = module_norm.DummyNormalizer()
        # else:
        #     data_normalizer = module_norm.NormalizeByMax()
        #     self.logger.warning(f"Using `NormalizeByMax` by default. Ignoring normalizer from config.")
        data_normalizer = module_norm.NormalizeByMax()
        self.logger.warning(
            f"Using `NormalizeByMax` by default. Ignoring normalizer from config."
        )
        return data_normalizer

    def do_inference(self, *args, **kwargs) -> None:
        raise NotImplementedError("Call method in child classes.")

    def _finish_inference(self, total_time_dl: float) -> Path:
        avg_time = total_time_dl / len(self.dataloader.dataset)
        batch_size = self.dataloader.batch_size

        _, avg_time_pca = self.scribe.get_inference_time_pca()
        self.logger.info(
            f"Total time: {total_time_dl} s (DL, batch size = {batch_size})."
        )
        self.logger.info(
            f"Average inference time: Deep learning: {avg_time} s\tPCA: {avg_time_pca}."
        )
        hdf5_filename = self.scribe.to_hdf5_file()
        self.logger.info("Finished.")
        return hdf5_filename

    def create_stats(self) -> None:
        if self.statistician is None:
            self.statistician = Statistician(
                self.basepath,
                self.eval_on_gt,
                self.cfg_eval.get("override_stats", False),
            )
        self.statistician.run_all(self.logger)

    def create_plots(self) -> None:
        if self.visiualizer is None:
            self.visiualizer = Visualizer(
                self.scribe.hdf5_filename,
                self.eval_on_gt,
                self.cfg_eval.get("override_plots", False),
                self.statistician.csv_dir_profiles,
                basepath_losses=self.statistician.basepath_losses,
            )
        self.visiualizer.run_all(self.logger)
