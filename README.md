![Python](https://img.shields.io/badge/python-3.12+-blue)
[![License](https://img.shields.io/github/license/vdplasthijs/aether.svg)](LICENSE)
![Issues](https://img.shields.io/github/issues/vdplasthijs/aether)
![GitHub Tag](https://img.shields.io/github/v/tag/vdplasthijs/aether)

# AETHER

Some code was adapted from [github.com/vdplasthijs/PECL/](github.com/vdplasthijs/PECL/). 

## Installation
Use conda to create a virtual environment from `aether.yml` or pip to install packages from `requirements.txt`. 

To use Google Earth engine (GEE) utilities, you need [a GEE API key](https://developers.google.com/earth-engine/guides/app_key). To use that key here, create the file `content/api_keys.py` that contains one line `GEE_API = '<your_api_key>'`. 

Data paths are automatically retrieved from `content/data_paths.json`. You can add your profile (login name) here, or alternatively the 'default' profile will be used.

## Getting started
Have a look at the Jupyter notebooks in `notebooks/` to see how the functions in `src/` are used. 

## Data:
The S2BMS coordinates and species occurrence probabilities are stored in `content/`. All satellite images of the full S2-BMS data set are available on [Zenodo](https://zenodo.org/records/15198884). 
