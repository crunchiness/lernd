#!/usr/bin/env bash
set -euo pipefail

. /opt/anaconda3/etc/profile.d/conda.sh
conda activate py37
python3 -m lernd.main
