# Design Baselines for Model-Based Optimization

This repository contains several design baselines for model-based optimization. Our hope is that a common evaluation protocol will encourage future research and comparability in model-based design.

## Available Baselines

We provide the following list of baseline algorithms.

* Conditioning by Adaptive Sampling: `from design_baselines.cbas import cbas`
* Model Inversion Networks: `from design_baselines.mins import mins`
* Forward Ensemble: `from design_baselines.forward_ensemble import forward_ensemble`

## Setup

You can install the algorithms by cloning this repository and using anaconda.

```bash
git clone https://github.com/brandontrabucco/design-baselines
conda env create -f design-baselines/environment.yml
```

## Usage

Every algorithm is implemented as a function that accepts a dictionary of hyper parameters called `config`. This makes interfacing with hyper parameter tuning platforms such as `ray.tune`, simple. 

```python
from design_baselines.forward_ensemble import forward_ensemble
forward_ensemble({
  "logging_dir": "forward-ensemble",
  "task": "HopperController-v0",
  "task_kwargs": {},
  "val_size": 200,
  "batch_size": 128,
  "bootstraps": 1,
  "epochs": 200,
  "hidden_size": 2048,
  "initial_max_std": 1.5,
  "initial_min_std": 0.5,
  "forward_model_lr": 0.001,
  "solver_samples": 128,
  "solver_lr": 0.0005,
  "solver_steps": 1000})
```

## Choosing Which Task

You may notice in the previous example that the `task` parameter is set to `HopperController-v0`. These baselines are tightly integrated with our [design-bench](https://github.com/brandontrabucco/design-bench). These will automatically be installed with anaconda. For more information on which tasks are currently available for use, or how to register new tasks, please check out [design-bench](https://github.com/brandontrabucco/design-bench).
