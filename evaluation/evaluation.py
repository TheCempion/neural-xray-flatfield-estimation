# standard libraries
import time
from pathlib import Path

# third party libraries
from tqdm import tqdm
import torch

# local packages
import utils.torch_settings as torch_settings
from parse_config import ConfigParser
from .base_evaluator import BaseEvaluator


__all__ = [
    "SynthDataEvaluator",
    "RealDataEvaluator",
]


class SynthDataEvaluator(BaseEvaluator):
    def __init__(self, config: ConfigParser):
        super().__init__(config, eval_on_gt=True)

    def do_inference(self) -> Path:
        if self.scribe.already_did_inference:
            return self.scribe.hdf5_filename
        device = torch_settings.get_torch_device()
        total_time = 0
        self.logger.info("Inference...")
        with torch.no_grad():
            for H_raw, gt_ff, gt_hologram in tqdm(self.dataloader):
                input = self.data_normalizer(H_raw)
                input = input.to(device)
                start_time = time.time()
                output = self.model(input)
                total_time += time.time() - start_time
                self.scribe(
                    input=input,
                    output=output,
                    input_raw=H_raw,
                    gt_ff=gt_ff,
                    gt_hologram=gt_hologram,
                )
        return super()._finish_inference(total_time)


class RealDataEvaluator(BaseEvaluator):
    def __init__(self, config: ConfigParser):
        if config.get("data_normalizer") is not None:
            config["data_normalizer"]["type"] = "NormalizeByMax"

        super().__init__(config, eval_on_gt=False)

    def do_inference(self) -> Path:
        device = torch_settings.get_torch_device()
        total_time = 0
        self.logger.info("Inference...")
        with torch.no_grad():
            for H_raw in tqdm(self.dataloader):
                input = self.data_normalizer(H_raw)
                input = input.to(device)
                start_time = time.time()
                output = self.model(input)
                total_time += time.time() - start_time
                self.scribe(input=input, output=output, input_raw=H_raw)
        return super()._finish_inference(total_time)
