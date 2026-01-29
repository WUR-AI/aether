<div align="center">

# AETHER-xAI

![python](https://img.shields.io/badge/python-3.12%2B-blue)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/PyTorch--Lightning-792EE5?style=flat&logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/WUR-AI/aether/blob/main/LICENSE) <br>
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/WUR-AI/aether/pulls)
[![Issues](https://img.shields.io/github/issues/vdplasthijs/aether)](https://github.com/WUR-AI/aether/issues)
![GitHub Tag](https://img.shields.io/github/v/tag/vdplasthijs/aether)
[![test-main](https://github.com/WUR-AI/aether/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/WUR-AI/aether/actions/workflows/test.yml)
[![contributors](https://img.shields.io/github/contributors/WUR-AI/aether.svg)](https://github.com/WUR-AI/aether//graphs/contributors)

</div>

## Description

This project develops an EO embedding/language model that can be used for explainable predictions from EO data.

## Getting Started

### Virtual environment

To install the dependencies in a venv using [uv](https://docs.astral.sh/uv/getting-started/installation/), first, clone the repo:

```bash
# clone project
git clone https://github.com/WUR-AI/aether
cd aether
```

Then, create a virtual environment (or alternatively via conda):
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate
```

Then, install `uv` and use this to install all packages.
```bash
# install uv manager
pip install uv

# install all Python dependencies
uv sync # reads pyproject.toml + uv.lock

# install project locally (editable)
uv pip install -e .
```

Note, running `uv sync` in the venv will always update the package to the most up-to-date version (as defined by the repo's `pyproject.toml` file).

### Set paths

Next, create a file in your local repo parent folder `aether/` called `.env` and copy the contents of `aether/.env.example`:

```bash
cp .env.example .env
```
Adjust the paths in `.env` to your local system. **At a minimum, you should set PROJECT_ROOT!**. 

**Important**: `DATA_DIR` should either point to `aether/data/` (default setting) OR if it points to another folder (e.g., `my/local/data/`) then copy the contents of the `aether/data/` folder to `my/local/data/` to ensure the butterfly use case runs using the provided example data. Other data will automatically be downloaded and organised by `pooch` if possible into `DATA_DIR`, or should be copied manually.

Data folders should follow the following directory structure within `DATA_DIR`:

```
├── registry.txt                         <- Pooch config file, don't change.
├── s2bms/                               <- Dataset folder.
│   ├── model_ready_s2bms.csv            <- Csv file with "name_loc" id, locations, aux data and target data.
│   ├── aux_classes.csv                  <- Csv file with explanations for aux data class names.
│   ├── caption_templates.json           <- Json file with list of caption templates (referencing aux column names).
│   ├── splits/                          <- Torch data splits
│   ├── source/                          <- Optional: source data used to create model_ready csv.
│   ├── eo/                              <- EO data modalities
│       ├── s2/                          <- Modality 1: (e.g. sentinel-2)
│           ├── s2_<NAME_LOC_1>.tif      <- EO modality data for a single location (indexed by unique <NAME_LOC>)
│           ├── s2_<NAME_LOC_2>.tif
│       ├── aef/                         <- Modality 2: (e.g. AEF)
│       ├── other_modality/
├── other_dataset/
```

### Verify installation:

To verify whether the installation was successful, run the tests in `aether/` using:
```bash
pytest --use-mock -m "not slow"
```
which should pass all tests.


## Training

Currently, we have implemented 2 models: a prediction model (that predicts target variables from EO data) and an alignment model (that aligns EO embeddings with text embeddings).

Experiment configurations (such as choosing data, encoders, hyperparameters etc.) are managed through [Hydra](https://hydra.cc/) configurations. Define your experiment configurations in `configs/experiments/experiment_name.yaml`, for example to train predictive model with GeoCLIP coordinate encoder for the Butterfly data using `configs/experiments/prediction.yaml` (copied below)

```yaml
# @package _global_
# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

defaults:
  - override /model: predictive_geoclip
  - override /data: butterfly_coords


tags: ["prediction", "geoclip_coords"]

seed: 12345

trainer:
  min_epochs: 1
  max_epochs: 100

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "predictive"
  aim:
    experiment: "predictive"
```

To execute this experiment run (inside your venv):

```bash
python train.py experiment=prediction
```

Please see the [Hydra](https://hydra.cc/) and [Hydra-Lightning template](https://github.com/ashleve/lightning-hydra-template) documentation for further examples of how to configure training runs.

## Directory structure

We follow the directory structure from the [Hydra-Lightning template](https://github.com/ashleve/lightning-hydra-template), which looks like:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data (for aether, this can also be elsewhere, see environment paths).
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── data_prepocessing        <- Data preprocessing scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Environment requirements, configuration options for testing and linting,
├── setup.py                  <- File for installing project as a package
├── uv.lock                   <- A frozen snapshot of exact dependencies for the uv package manager.
└── README.md
```

## Attribution

This repo is based on the [Hydra-Lightning template](https://github.com/ashleve/lightning-hydra-template).
Some code was adapted from [github.com/vdplasthijs/PECL/](github.com/vdplasthijs/PECL/).
