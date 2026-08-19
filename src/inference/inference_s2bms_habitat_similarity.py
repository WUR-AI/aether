import json
import pickle  # nosec B403
from typing import Optional

import hydra
import numpy as np
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

import src.data_preprocessing.data_utils as du
from src.models.inference_model import load_inference_model, merge_inference_model
from src.utils import extras

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

# Disable tokenizers parallelism to avoid warnings when using multiprocessing
import os

if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(
    version_base="1.3",
    config_path="../../configs/",
    config_name="inference_s2bms_habitat_similarity.yaml",
)
def main(cfg: DictConfig, save_results=False) -> Optional[float]:
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
        model = load_inference_model(inference_ckpt_path, cfg.paths.cache_dir)
    # Otherwise merge model from two checkpoints
    else:
        model = merge_inference_model(cfg, save_ckpt=True)
    model.to("mps")

    if cfg.data:
        datamodule = hydra.utils.instantiate(cfg.get("data"))
        datamodule.setup()

        concepts = [
            c["concept_caption"] for c in datamodule.caption_builder.__dict__["concepts"]
        ]  # or other source of concepts
        text_embeds = model.forward_text(concepts).squeeze(0).detach().cpu().numpy()

        print("Concept caption embeddings shape:", text_embeds.shape)

        filepath_inference_captions = os.path.join(
            cfg.paths.data_dir, "s2bms/inference_captions/habitat_preferences.json"
        )
        with open(filepath_inference_captions) as f:
            inference_captions = json.load(f)
        assert type(inference_captions) is list, "Expected inference_captions to be a list"

        inference_captions_embedded = {}
        target_id_inference_captions = {}
        for item in inference_captions:
            key = item["common_name"]
            if key == "NAME":
                continue
            value = item["inference_captions"]
            target_id = item["target_id"]
            inference_captions_embedded[key] = (
                model.forward_text(value).squeeze(0).detach().cpu().numpy()
            )
            inference_captions_embedded[key] = inference_captions_embedded[key] / np.linalg.norm(
                inference_captions_embedded[key], axis=1, keepdims=True
            )  # Normalize caption embeddings
            target_id_inference_captions[key] = target_id
            print(f"Embedded caption for {key}: {inference_captions_embedded[key].shape}")

        # per location (batching uses location text generation)
        for i_d, d in enumerate(datamodule.data_test):
            b = {"eo": {"aef_avr": d["eo"]["aef_avr"].unsqueeze(0).to("mps")}}
            geo_embeds, pred = model.forward_geo(b)

            if i_d == 0:
                geo_embeds_all = geo_embeds.detach().cpu()
                pred_all = pred.detach().cpu()
                labels_all = (
                    d["target"].detach().cpu()[None, :]
                )  # Assuming target is a tensor and we want the first element
            else:
                geo_embeds_all = torch.cat((geo_embeds_all, geo_embeds.detach().cpu()), dim=0)
                pred_all = torch.cat((pred_all, pred.detach().cpu()), dim=0)
                labels_all = torch.cat(
                    (labels_all, d["target"].detach().cpu()[None, :]), dim=0
                )  # Concatenate labels as well

        labels_all = labels_all.numpy()
        print("Labels shape:", labels_all.shape)

        geo_embeds_all = geo_embeds_all.numpy()
        geo_embeds_all = geo_embeds_all / np.linalg.norm(
            geo_embeds_all, axis=1, keepdims=True
        )  # Normalize geo embeddings

        pred_all = pred_all.numpy()

        print("Geo embeddings shape:", geo_embeds_all.shape)
        print("Predictions shape:", pred_all.shape)

        mse_per_species = np.mean((pred_all - labels_all) ** 2, axis=0)
        print("MSE per species:", mse_per_species)
        print(
            "Best species: ", np.argsort(mse_per_species)[:10]
        )  # Print indices of top 5 best predicted species

        for key, embed in inference_captions_embedded.items():
            print(f"Similarity between geo embeds and {key} caption embedding:")
            similarity = np.dot(geo_embeds_all, embed.T)
            sim_sum = similarity.sum(axis=1)
            print(
                "Similarity vs labels: ",
                np.corrcoef(sim_sum, labels_all[:, target_id_inference_captions[key]])[0, 1],
            )  # Correlation between summed similarity and labels
            print(
                "Similarity vs predictions: ",
                np.corrcoef(sim_sum, pred_all[:, target_id_inference_captions[key]])[0, 1],
            )  # Correlation between predictions and labels

        results_store = {
            "test": {
                "geo_embeddings": geo_embeds_all,
                "inference_caption_embeddings": inference_captions_embedded,
                "target_id_inference_captions": target_id_inference_captions,
                "labels": labels_all,
                "predictions": pred_all,
            }
        }

        if save_results:
            timestamp = du.create_timestamp()
            results_folder = os.path.join(cfg.paths.data_dir, "outputs")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            results_path = os.path.join(results_folder, f"inference_results_{timestamp}.pkl")
            with open(results_path, "wb") as f:
                pickle.dump(results_store, f)
            print(f"Saved inference results to {results_path}")

    return results_store


if __name__ == "__main__":
    _ = main()
