import os
from typing import Optional

import hydra
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.models.inference_model import load_inference_model, merge_inference_model
from src.utils import extras

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

# Disable tokenizers parallelism to avoid warnings when using multiprocessing
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model(cfg):
    # If a merged inference ckpt is provided, just load it.
    inference_ckpt_path = cfg.get("inference_ckpt_path")
    if inference_ckpt_path:
        model = load_inference_model(inference_ckpt_path, cfg.paths.cache_dir)
    else:
        model = merge_inference_model(cfg, save_ckpt=True)

    return model


def get_geo_data(cfg, device="cpu"):
    import pandas as pd

    params = cfg["geo_data"]
    modality = list(params.keys())[0]
    params = params[modality]
    dtype = params.get("dtype", torch.float32)
    dim = params.get("dimension", 64 if modality == "aef_avr" else 128)

    path = params.get("path", KeyError(f"Please specify {modality} path to csv file"))
    assert os.path.exists(path), FileNotFoundError(f"{path} does not exist.")
    df = pd.read_csv(path)

    # Filter out locations without data for embeddings
    emb_cols = [f"emb_{i}" for i in range(dim)]

    geo_data = torch.tensor(df[emb_cols].to_numpy(), dtype=dtype, device=device)
    return {
        "eo": {modality: geo_data},
        "name_loc": df.name_loc.to_list(),
        "lat": df.lat.to_list(),
        "lon": df.lon.to_list(),
        "split": df.split.to_list(),
    }


@torch.no_grad()
@hydra.main(version_base="1.3", config_path="../configs", config_name="packaged_inference.yaml")
def main(cfg: DictConfig, save_results: bool = False) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :param save_results: Whether to save inference results.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    model = get_model(cfg)
    model.to(cfg.device)

    # TODO: do what you need with the inference model
    # Supply text in a list. Examples
    text = ["Forested area", "Area with water bodies near-by"]
    # or
    # text = [input('Enter a concept')]
    # or
    # with open('../outputs/example_concepts.txt', 'r') as f:
    #     text = f.readlines()
    #     text = [t.strip() for t in text]

    text_embeds = model.forward_text(text)

    # supply geo data as csv file
    if cfg.get("geo_data"):
        b = get_geo_data(
            cfg, device=cfg.get("device", "cpu")
        )  # batch has name_loc, lat, lon and split arguments
        geo_embeds, pred = model.forward_geo(b)

    geo_embeds = F.normalize(geo_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    similarity_matrix = geo_embeds @ text_embeds.T

    results = torch.cat([similarity_matrix, pred], dim=1)

    if cfg.get("save_output") or save_results:
        df = pd.DataFrame(
            results.cpu(),
            columns=text + [f"target_{i}" for i in range(pred.shape[-1])],
            index=b.get("name_loc"),
        )
        df.reset_index(inplace=True, names=["name_loc"])
        path = cfg.get("save_output")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"Cosine similarities are saved to {path}")
    else:
        names = b["name_loc"]
        for i in range(len(names)):
            print(f"Location {names[i]} similarity with:")
            for j, t in enumerate(text):
                print(f" - {t}: {similarity_matrix[i, j]:.2f} similarity")
            print(f"And prediction vector: {pred[i].detach().cpu().numpy()}")

    return


if __name__ == "__main__":
    main()
