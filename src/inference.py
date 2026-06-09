from typing import Optional

import hydra
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.models.inference_model import load_inference_model, merge_inference_model
from src.utils import extras

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

# Disable tokenizers parallelism to avoid warnings when using multiprocessing
import os

if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # If a merged inference ckpt is provided, just load it.
    inference_ckpt_path = cfg.get("inference_ckpt_path")
    if inference_ckpt_path:
        model = load_inference_model(inference_ckpt_path)
    # Otherwise merge model from two checkpoints
    else:
        model = merge_inference_model(cfg, save_ckpt=True)

    # TODO: do what you need with the inference model

    return


if __name__ == "__main__":
    main()
