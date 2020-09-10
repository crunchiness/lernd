# LERND

Lernd stands for [Learning Explanatory Rules from Noisy Data](https://arxiv.org/abs/1711.04574).
It is my implementation of the algorithm in the linked paper.

## Demo Jupyter notebooks
Demo Jupyter notebooks are available online on Kaggle for a quick look into how it works:
1. https://www.kaggle.com/ingvaras/lernd-intro-predecessor
2. https://www.kaggle.com/ingvaras/lernd-even

Notebook files for local use can be found on https://github.com/crunchiness/lernd-notebooks

## Set up and run

### Step 1
Run Lernd on Python 3.7+

You may create a conda environment (here named "lernd"):
```bash
conda create -n lernd python=3.7
```

(Follow instructions at [anaconda.com](https://www.anaconda.com) to get and install the conda package manager.)


Activate the environment:
```bash
conda activate lernd
```

### Step 2
Install requirements:
```bash
pip install -r requirements.txt
```

### Step 3
Run experiments.

Some benchmark problems are defined in file `lernd/experiments.py`.

You may run lernd on them:
```bash
conda activate lernd  # activate environment if using conda
python -m lernd.experiments
```


## Tests

Unit tests are in `lernd/test.py`. 

Run them:
```bash
conda activate lernd  # activate environment if using conda
python -m lernd.test
```
