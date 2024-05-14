# MLDE Notebooks

Evalute ML approaches to emulate a CPM and downscale rainfall.

## Setup

This guide assumes you are using conda (or mamba) to manage packages and python environments.

1. Install conda environment:
  * If you wish to re-use the exact environment: `conda env create --file environment.lock.yml` and activate it: `conda activate downscaling-notebooks`
  * OR install the needed conda and pip packages to your own environment: `conda install --file environment.txt`
2. Install this package (including a few pip dependencies which may not have been included in the previous step): `pip install -e .`
3. Create .env file: cp .env.example .env and then update to match your needs:
  * `DERIVED_DATA`: path to where derived data such datasets and model artefacts are kept

## Running

### Interactive

Use jupyter: `jupyter lab`

### Batch

This can be used with a helper script to run a notebook in batch mode along with an (optional) parameter files: `bin/run-notebook nbs/my/notebook.ipynb nbs/my/parameters/setA.yml`

## Development

There are a set of development dependencies if you do more than just tweak a notebook: `pip install -e '.[dev]'`
