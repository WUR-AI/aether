from typing import Optional

import hydra
import rootutils
from omegaconf import DictConfig

from src.utils import extras
from src.utils.experiment_tracking import clean_local_ckpts, get_experiments_from_wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="wandb.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)

    df = get_experiments_from_wandb(cfg)
    clean_local_ckpts(cfg, df)


if __name__ == "__main__":
    main()
