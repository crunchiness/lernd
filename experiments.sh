#!/usr/bin/env bash

# Get conda.sh path
CONDABIN="condabin/conda"
CONDA_SCRIPT="etc/profile.d/conda.sh"
CONDABIN_PATH=$(command -v conda)
CONDA_SCRIPT_PATH="${CONDABIN_PATH/$CONDABIN/$CONDA_SCRIPT}"

# shellcheck source=/opt/anaconda3/etc/profile.d/conda.sh
. "$CONDA_SCRIPT_PATH"

conda activate lernd
python3 -m lernd.experiments
