<div align="center">

# Domain-specific language alignment of multimodal embeddings towards explainable predictions

[![arXiv](https://img.shields.io/badge/arXiv-2610.23110-b31b1b.svg)](https://arxiv.org/abs/2610.23110)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/PyTorch--Lightning-792EE5?style=flat&logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/WUR-AI/aether/pulls)
[![Issues](https://img.shields.io/github/issues/WUR-AI/aether)](https://github.com/WUR-AI/aether/issues)
![GitHub Tag](https://img.shields.io/github/v/tag/WUR-AI/aether)
[![test-main](https://github.com/WUR-AI/aether/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/WUR-AI/aether/actions/workflows/test.yml)
[![contributors](https://img.shields.io/github/contributors/WUR-AI/aether.svg)](https://github.com/WUR-AI/aether/graphs/contributors)

</div>

## Description

This repository is the code for the pre-print
[*Domain-specific language alignment of multimodal embeddings towards explainable predictions*](https://arxiv.org/abs/2610.23110).

It provides a framework to language-align any embedding space (e.g. Earth observation embeddings such as AlphaEarth) using interpretable auxiliary data. It consists of:

- **captioning functions** that turn auxiliary data (land cover, climate, ...) into text captions;
- a **soft contrastive loss** that aligns geo-embeddings with text embeddings of those captions;
- a **rank-based concept retrieval** evaluation of the aligned space;
- **prediction models** that predict target variables (e.g. species occurrence) from the same embeddings, so that predictions can be linked to aligned concepts.

### Use cases

The paper evaluates the framework on two species distribution modelling datasets:

- **S2BMS** (UK butterflies)
- **SatBird** (USA summer birds)

The repository also contains code and configs for use cases that we explored during development but that are **not part of the paper**: urban heat in Guatemala City and Kraków (`heat_*`) and crop yield in Africa (`yield_africa_*`). They are kept for reference and may be less maintained.

## Main results

All numbers are taken from the paper and reported as mean ± SEM over 3 seeds on the test split. Rows marked with * are taken from previous work and were not re-run in this repository. See the paper for the full ablation tables.

### Prediction

AlphaEarth embeddings beat all baselines, including the previous state of the art, on 6 of 7 metrics.

**S2BMS** (62 butterfly species, 1,329 locations, UK)

| Modality (best config.)                        | Top-10 \[%\] ↑ | Top-5 \[%\] ↑  | MSE \[1e−2\] ↓  |
| ---------------------------------------------- | -------------- | -------------- | --------------- |
| Mean rate\*                                    | 67.3           | 58.7           | 1.39            |
| Sentinel-2 (RGB)                               | 69.5 ± 0.2     | 62.9 ± 0.5     | 1.23 ± 0.02     |
| Sentinel-2 (RGB + IR) + contrastive \[SOTA\]\* | 70.4 ± 0.2     | 63.0 ± 0.9     | 1.20 ± 0.03     |
| Linear regression (aux. data)                  | 72.4 ± 0.1     | 64.3 ± 0.4     | 1.10 ± 0.00     |
| MLP (aux. data)                                | 72.2 ± 0.3     | 64.7 ± 0.5     | 1.09 ± 0.02     |
| GeoCLIP                                        | 72.0 ± 0.4     | 65.8 ± 0.7     | 1.12 ± 0.01     |
| SatCLIP                                        | 70.8 ± 0.6     | 63.5 ± 0.4     | 1.14 ± 0.01     |
| Tessera                                        | 72.7 ± 0.2     | 65.8 ± 0.2     | 1.01 ± 0.00     |
| **AlphaEarth**                                 | **73.0 ± 0.1** | **67.2 ± 0.4** | **0.98 ± 0.00** |

**SatBird USA-summer** (670 bird species, 122K locations, USA)

| Modality (best config.)        | Top-10 \[%\] ↑ | Top-30 \[%\] ↑ | MSE \[1e−2\] ↓  | MAE \[1e−2\] ↓  |
| ------------------------------ | -------------- | -------------- | --------------- | --------------- |
| Mean rate\*                    | 26.9           | 38.6           | 0.9             | 3.1             |
| Sentinel-2 RGB\*               | 35.5 ± 0.1     | 52.2 ± 0.2     | 0.8 ± 0.0       | 2.6 ± 0.02      |
| Sentinel-2 RGB + env.\*        | 46.3 ± 0.2     | 65.5 ± 0.2     | 0.64 ± 0.0      | 2.2 ± 0.02      |
| MiTREE \[SOTA\]\*              | 47.3           | **66.5**       | 0.62            | 2.09            |
| MIAM (best of input subsets)\* | 45.6           | 62.0           | -               | 1.94            |
| Linear regression (aux. data)  | 38.8 ± 0.0     | 52.3 ± 0.0     | 0.75 ± 0.00     | 2.42 ± 0.00     |
| MLP (aux. data)                | 43.2 ± 0.1     | 56.5 ± 0.0     | 0.67 ± 0.00     | 2.20 ± 0.00     |
| GeoCLIP                        | 40.0 ± 0.2     | 53.6 ± 0.2     | 0.73 ± 0.00     | 2.34 ± 0.01     |
| SatCLIP                        | 39.0 ± 0.1     | 52.8 ± 0.1     | 0.73 ± 0.00     | 2.37 ± 0.00     |
| **AlphaEarth**                 | **49.7 ± 0.1** | 61.5 ± 0.0     | **0.58 ± 0.00** | **1.93 ± 0.00** |

### Alignment (concept retrieval)

Scores are the chance-corrected top-k retrieval index I<sub>top-k</sub> (0 = random, 1 = perfect), averaged per auxiliary data group. *Av. (sub)* is the validation criterion used for model selection. AEF = AlphaEarth Foundations embeddings aligned with our soft contrastive loss.

**S2BMS**

| Model                  | Biocl.          | CORINE          | Human footpr.   | UKCEH LC        | PM2.5       | Av. (sub)       |
| ---------------------- | --------------- | --------------- | --------------- | --------------- | ----------- | --------------- |
| GeoCLIP                | 0.01            | 0.13            | 0.07            | 0.21            | **0.64**    | 0.06            |
| RemoteCLIP             | 0.04            | 0.31            | 0.23            | 0.29            | 0.27        | 0.21            |
| CLIP                   | 0.07            | 0.28            | 0.25            | 0.26            | 0.20        | 0.23            |
| GeoRSCLIP              | 0.03            | 0.35            | 0.33            | 0.39            | -0.02       | 0.24            |
| SkyCLIP                | 0.07            | 0.33            | 0.33            | 0.34            | 0.27        | 0.25            |
| DOFA-CLIP              | 0.04            | 0.34            | 0.27            | 0.44            | 0.13        | 0.29            |
| AEF, mixed captions    | 0.10 ± 0.01     | **0.48 ± 0.01** | **0.37 ± 0.03** | 0.44 ± 0.01     | 0.54 ± 0.02 | 0.42 ± 0.01     |
| AEF, separate captions | **0.20 ± 0.01** | **0.48 ± 0.01** | 0.11 ± 0.05     | **0.47 ± 0.01** | 0.49 ± 0.00 | **0.45 ± 0.00** |

UKCEH land cover and PM2.5 were not used for captioning; they test transfer to out-of-distribution concepts.

**SatBird USA-summer**

| Model                  | Biocl.      | Dynamic World LC | Soil        | Av. (sub)       |
| ---------------------- | ----------- | ---------------- | ----------- | --------------- |
| GeoCLIP                | 0.05        | 0.11             | 0.03        | 0.06            |
| RemoteCLIP             | 0.05        | 0.33             | 0.03        | 0.15            |
| CLIP                   | 0.12        | 0.28             | 0.05        | 0.18            |
| DOFA-CLIP              | 0.04        | 0.45             | **0.06**    | 0.19            |
| SkyCLIP                | 0.11        | 0.38             | 0.02        | 0.20            |
| GeoRSCLIP              | **0.15**    | 0.42             | 0.03        | 0.23            |
| AEF, separate captions | 0.08 ± 0.01 | **0.56 ± 0.00**  | 0.03 ± 0.02 | **0.24 ± 0.01** |

## Getting started

### Installation

Clone the repo and create a virtual environment (or use conda):

```bash
git clone https://github.com/WUR-AI/aether
cd aether

python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# install uv manager
pip install uv

# install all Python dependencies
uv sync # reads pyproject.toml + uv.lock

# install project locally (editable)
uv pip install -e .
```

Note, running `uv sync` in the venv will always update the package to the most up-to-date version (as defined by the repo's `pyproject.toml` file).

### Paths

Copy the example environment file and edit it for your system:

```bash
cp .env.example .env
```

Adjust the paths in `.env` to match your local system. **At a minimum, you must set `PROJECT_ROOT`.**

> **Important:** This repository only includes a small portion of the data. The full dataset is available on Zenodo: https://doi.org/10.5281/zenodo.23336313

After downloading the full dataset, make sure `DATA_DIR` is set up in one of two ways:

1. **Default:** `DATA_DIR` points to `aether/data/`. Place the downloaded data there.
2. **Custom location:** `DATA_DIR` points to another folder (e.g., `my/local/data/`). In this case, copy the contents of `aether/data/` into `my/local/data/`.

Any additional data is downloaded and organised automatically by [`pooch`](https://www.fatiando.org/pooch/), into `DATA_DIR` where possible. If automatic download fails, copy the files over manually.

Data folders should follow the following directory structure within `DATA_DIR`:

```
├── registry.txt                         <- Pooch config file, don't change.
├── s2bms/                               <- Dataset folder.
│   ├── model_ready_s2bms.csv            <- "name_loc" id, locations, aux data and target data.
│   ├── aux_classes.csv                  <- Explanations for aux data class names.
│   ├── caption_templates/
│   │   └── v1.json                      <- Caption templates (referencing aux column names).
│   ├── splits/                          <- Torch data splits.
│   ├── source/                          <- Optional: source data used to create the model_ready csv.
│   └── eo/                              <- EO data modalities.
│       ├── s2/                          <- e.g. Sentinel-2
│       │   ├── s2_<NAME_LOC_1>.tif      <- One file per location, indexed by unique <NAME_LOC>.
│       │   └── s2_<NAME_LOC_2>.tif
│       └── aef/                         <- e.g. AlphaEarth Foundations embeddings
├── satbird-USA-summer/
│   └── ...                              <- same structure as s2bms
└── other_dataset/
```

### Verify the installation

```bash
pytest --use-mock -m "not slow"
```

All tests should pass.

## Training

Two model types are implemented:

- a **prediction model** that predicts target variables from EO embeddings.
- an **alignment model** that aligns EO embeddings with text embeddings of captions;

Experiments (data, encoders, hyperparameters, ...) are configured with [Hydra](https://hydra.cc/) in `configs/experiment/`. For example, to train the S2BMS alignment model on AlphaEarth embeddings with the soft contrastive loss:

```bash
python src/train.py experiment=s2bms_alignment_aef/aef_soft_labelled_lr
```

and to train a SatBird prediction model on AlphaEarth embeddings:

```bash
python src/train.py experiment=satbird_sum_prediction/avr_aef_128_256_mlp
```

Please see the [Hydra](https://hydra.cc/) and [Hydra-Lightning template](https://github.com/ashleve/lightning-hydra-template) documentation for further examples of how to configure training runs.

### Reproducing the paper

Every model in the tables above has an experiment config. Most configs run as Hydra multiruns over the ablation grid from the paper's appendix. The best configuration was selected on MSE for prediction and on *Av. (sub)* retrieval for alignment.

| Paper result                 | S2BMS (`configs/experiment/...`)                                  | SatBird (`configs/experiment/...`)                            |
| ---------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| Prediction: AlphaEarth       | `s2bms_prediction/avr_aef_1_128_256_{lin,mlp,deeper_mlp}`         | `satbird_sum_prediction/avr_aef_128_256_{lin,mlp,deeper_mlp}` |
| Prediction: Tessera          | `s2bms_prediction/avr_tessera_v11_1_128_256_{lin,mlp,deeper_mlp}` | not evaluated                                                 |
| Prediction: GeoCLIP          | `s2bms_prediction/geoclip_{lin,mlp,deeper_mlp}`                   | `satbird_sum_prediction/geoclip_{lin,mlp,deeper_mlp}`         |
| Prediction: SatCLIP          | `s2bms_prediction/satclip_{lin,mlp,deeper_mlp}`                   | `satbird_sum_prediction/satclip_{lin,mlp,deeper_mlp}`         |
| Prediction: aux. data        | `s2bms_prediction/baselines/tabular_{linear,mlp}`                 | `satbird_sum_prediction/baselines/tabular_{linear,mlp}`       |
| Prediction: Sentinel-2 (CNN) | `s2bms_prediction/s2_*`                                           | taken from previous work                                      |
| Alignment: AEF (ours)        | `s2bms_alignment_aef/aef_soft_{labelled,unlabelled}_lr_v11`       | `satbird_alignment/aef_s01`                                   |
| Alignment: VLM baselines     | `s2bms_alignment_baselines/*`                                     | `satbird_alignment_baselines/*`                               |

For example:

```bash
python src/train.py experiment=s2bms_alignment_aef/aef_soft_unlabelled_lr_v11   # ours, S2BMS
python src/train.py experiment=s2bms_alignment_baselines/dofa_clip              # DOFA-CLIP baseline
```

## Baselines

### Prediction baselines

All embedding baselines share the same prediction heads (linear, 1- or 2-hidden-layer MLP, see `src/models/components/pred_heads/`). Only the geo-encoder in `src/models/components/geo_encoders/` changes:

| Baseline          | Encoder implementation                                           | Source / weights                                                                                                                              |
| ----------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| AlphaEarth        | `average_encoder.py` (spatial average pooling), `cnn_encoder.py` | Pre-computed [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) embeddings via Google Earth Engine (`scripts/download_gee_data.py`)   |
| Tessera           | `average_encoder.py`, `cnn_encoder.py`                           | Pre-computed [Tessera](https://arxiv.org/abs/2506.20380) embeddings via the `geotessera` package (`src/data_preprocessing/tessera_embeds.py`) |
| GeoCLIP           | `geoclip.py`                                                     | Location encoder from the [`geoclip`](https://github.com/VicenteVivan/geo-clip) package                                                       |
| SatCLIP           | `satclip.py`                                                     | [`microsoft/SatCLIP-ViT16-L40`](https://huggingface.co/microsoft/SatCLIP-ViT16-L40) ([repo](https://github.com/microsoft/satclip))            |
| Sentinel-2 images | `cnn_encoder.py` (ResNet-18/34/50)                               | Random, ImageNet or SSL4EO initialisation                                                                                                     |
| Auxiliary data    | `tabular_encoder.py`                                             | Linear regression / MLP on the auxiliary variables                                                                                            |

The mean-rate and previous state-of-the-art rows are taken from [PECL](https://github.com/vdplasthijs/PECL/) (van der Plas et al., 2025) for S2BMS, and from [SatBird](https://github.com/RolnickLab/SatBird) (Teng et al., 2023), MiTREE (Chen & Chiang, 2024) and MIAM (Zbinden et al., 2026) for SatBird.

### Alignment baselines

The vision-language baselines are evaluated zero-shot (frozen, `trainable_modules: []`) on Sentinel-2 RGB images, centre-cropped to 224 × 224 (DOFA-CLIP is also run on RGB + NIR), with the same concept retrieval metric as our models. Each baseline is wrapped as a `TextAlignmentModel`, so it runs through the same training and evaluation pipeline:

| Baseline   | Implementation                                                              | Backbones evaluated                | Pre-trained weights                                                                                                            |
| ---------- | --------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| CLIP       | `geo_encoders/clip_img.py`, `text_encoders/clip_text_encoder.py`            | ViT-B-32, ViT-B-16, ViT-L-14, RN50 | [OpenAI CLIP](https://github.com/openai/CLIP) (`openai/clip-vit-*` from Hugging Face, RN50 via `open_clip`)                    |
| RemoteCLIP | `src/models/remote_clip_baseline.py`                                        | ViT-B-32, ViT-L-14, RN50           | [`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP) ([repo](https://github.com/ChenDelong1999/RemoteCLIP)) |
| GeoRSCLIP  | `src/models/georsclip.py`                                                   | ViT-B-32, ViT-H-14                 | [`Zilun/GeoRSCLIP`](https://huggingface.co/Zilun/GeoRSCLIP) ([repo](https://github.com/om-ai-lab/RS5M))                        |
| SkyCLIP    | `src/models/skyclip_baseline.py`                                            | ViT-L-14 / ViT-B-32 variants       | [SkyScript](https://github.com/wangzhecheng/SkyScript) checkpoints (downloaded automatically)                                  |
| DOFA-CLIP  | `src/models/dofa_clip_baseline.py`, vendored code in `src/utils/dofa_clip/` | ViT-B-16, ViT-L-14                 | [`earthflow/GeoLB-*`](https://huggingface.co/earthflow) ([repo](https://github.com/xiong-zhitong/DOFA-CLIP))                   |
| GeoCLIP    | `geo_encoders/geoclip.py`                                                   | coordinates → CLIP space           | [`geoclip`](https://github.com/VicenteVivan/geo-clip) package                                                                  |

Each baseline config also sweeps three Sentinel-2 preprocessing options (`div_2000`, `div_10000`, `stretch_2_98`). Pre-trained weights are downloaded to `paths.huggingface_cache` on first use.

## Inference

1. Copy the provided `inference/` folder into the repository root.
2. Edit `configs/packaged_inference.yaml`:
   - Set the checkpoint, e.g. `inference_ckpt_path: ${paths.root_dir}/inference/s2bms/v1/example.ckpt`.
   - Choose the locations to evaluate:
     ```yaml
     geo_data:
       aef_avr:
         path: ${paths.root_dir}/inference/s2bms/avr_aef_256.csv # can be a subset or just one entry
     ```
   - Set `device` to `mps`, `cuda` or `cpu`.
   - To save results (cosine similarities and predictions), set `save_output: ${paths.root_dir}/inference/s2bms/v1/${now:%Y-%m-%d}_${now:%H-%M-%S}.csv`.
3. Run:
   ```bash
   python src/packaged_inference.py
   ```

## Directory structure

We follow the directory structure from the [Hydra-Lightning template](https://github.com/ashleve/lightning-hydra-template), which looks like:

```
├── configs                <- Hydra configs
│   ├── data                  <- Dataset configs
│   ├── experiment            <- Experiment configs (one folder per use case)
│   ├── model                 <- Model configs
│   ├── metrics               <- Metric configs
│   ├── ...                   <- callbacks, logger, trainer, paths, hydra, ...
│   ├── train.yaml            <- Main training config
│   ├── eval.yaml             <- Main evaluation config
│   └── packaged_inference.yaml
├── data                   <- Project data (can live elsewhere, see `DATA_DIR`)
├── docs                   <- Figures
├── inference              <- Example inference inputs and checkpoint
├── notebooks              <- Jupyter notebooks
├── scripts                <- Shell scripts
├── src
│   ├── data                  <- Datasets, datamodules and caption builders
│   ├── data_preprocessing    <- Scripts to build model-ready data
│   ├── models                <- Models, encoders, projectors and losses
│   ├── utils                 <- Utilities
│   ├── train.py              <- Training entry point
│   ├── eval.py               <- Evaluation entry point
│   └── packaged_inference.py <- Inference entry point
├── tests
├── .env.example           <- Template for local environment variables
├── pyproject.toml         <- Dependencies and tool configuration
└── uv.lock                <- Locked dependencies for uv
```

## Citation

If you use this code, please cite:

```bibtex
@article{tijunaityte2026domain,
  title   = {Domain-specific language alignment of multimodal embeddings towards explainable predictions},
  author  = {Tij{\=u}naityt{\.e}, Gabriel{\.e} and van der Plas, Thijs L. and Belaid, Mohamed-Bachir and Knapen, Rob and Soja, Maciej J. and Athanasiadis, Ioannis N.},
  journal = {arXiv preprint arXiv:2610.23110},
  year    = {2026}
}
```

## Acknowledgements

This repo is based on the [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).
Some code was adapted from [PECL](https://github.com/vdplasthijs/PECL/).
Please also cite the original S2BMS and SatBird datasets when using them.
