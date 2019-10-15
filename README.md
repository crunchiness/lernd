# LERND (work in progress)

My attempt to implement the algorithm from the paper
[Learning Explanatory Rules from Noisy Data](https://arxiv.org/abs/1711.04574).

## Set up

First you'll need to install conda package manager. Follow instructions at [anaconda.com](https://www.anaconda.com).
This project uses standard Python 3.7 environment provided by Anaconda.

To create the environment (named lernd):
```bash
conda create -n lernd python=3.7 scipy=1.3.1
```

Activate the environment:
```bash
conda activate lernd
```

Install requirements from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Run tests

To run tests:
```bash
./test.sh
```
or
```bash
python3 -m lernd.experiments  # with "lernd" env active
```

## Current state of affairs

Main issues at the moment are:
* inferrer is too slow,
* autograd does not work due to ints, so need to try and figure out how to circumvent that (try JAX),
* optimizer (like SGD) is not implemented


## Experiments

To run experiments (not fully implemented, but makes inferrer work):
```bash
./experiments.sh
```
or
```bash
python3 -m lernd.experiments  # with "lernd" env active
```
