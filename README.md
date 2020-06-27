# Design Benchmarks for Model-Based Optimization

This repository contains several benchmarks of design problems for model-based optimization. Have Fun! -Brandon

In particular, we provide the following family of design problems:

* ROBEL D'Kitty Morphology Design
* MuJoCo Ant Morphology Design 
* MuJoCo Dog Morphology Design 
* Fluorescent Protein Design

## Setup

You can install our benchmarks with the following command.

```bash
pip install git+git://github.com/brandontrabucco/design-bench.git
```

## Usage

You can instantiate a design problem using the factory module.

```python
import design_bench.factory as f

# create a design problem using a curated dataset
p = f.AntMorphology()

# sample designs from the dataset
designs = p.sample(n=10)
```
