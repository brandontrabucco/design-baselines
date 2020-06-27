# Baselines for Model-Based Optimization

This repository provides a common interface for solving model-based optimization problems.

In particular, we provide the following family of design problems:

* ROBEL D'Kitty Morphology Design
* MuJoCo Ant Morphology Design 
* MuJoCo Dog Morphology Design 
* Fluorescent Protein Design

## Setup

You can install the baselines benchmarks with the following command.

```bash
git clone github.com/brandontrabucco/design-baselines.git
pip install -e ./design-baselines
pip install -e ./design-baselines/design-bench
```

## Usage

You can instantiate a design problem using the factory module.

```python
import design_bench.factory as fty

# create a design problem using a curated dataset
p = fty.AntMorphology()

# sample designs from the dataset
design = p.sample(n=1)
```

You can find samples that solve a design problem using an algorithm.

```python
import design_bench.factory as fty
import design_baselines.dev as dev

# create a design problem using a curated dataset
p = fty.AntMorphology()

# create a design problem using a curated dataset
a = dev.ForwardModel(p)

# sample designs from the dataset
design = a.solve()
```

## Contributing

    is_offline, is_batched, design_space, score

New design problems can be added to the `design-bench` repository in the factory module. These design problems can have both continuous and discrete components, and  must implement the abstract methods found in the base `DesignProblem` class. In particular, these methods are `is_offline`, `is_batched`, `design_space`, and `score.`

New model-based optimization algorithms can be added to the `design-baselines` repository in the dev module. Similar to the benchmarks code, algorithms must inherit from the base `Algorithm` class, and implement all abstract methods. These methods are `solve.`
