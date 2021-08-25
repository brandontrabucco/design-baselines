# Design-Baselines

Design-Baselines is a set of **baseline algorithms** for solving automatic design problems that involve choosing an input that maximizes a black-box function. This type of optimization is used across scientific and engineering disciplines in ways such as designing proteins and DNA sequences with particular functions, chemical formulas and molecule substructures, the morphology and controllers of robots, and many more applications. 

These applications have significant potential to accelerate research in biochemistry, chemical engineering, materials science, robotics and many other disciplines. We hope this set of baselines serves as a robust platform to drive these applications and create widespread excitement for model-based optimization.

**COMs Website**: [link](https://sites.google.com/berkeley.edu/coms/home?authuser=0) | **COMs Paper**: [arXiv](https://arxiv.org/abs/2107.06882)

If these applications interest you, consider using our benchmark: [design-bench](https://github.com/brandontrabucco/design-bench), which you may install to python and automatically download benchmark data using `pip install design-bench==2.0.12`.

## Offline Model-Based Optimization

![Offline Model-Based Optimization](https://storage.googleapis.com/design-bench/mbo.png)

The goal of model-based optimization is to find an input **x** that maximizes an unknown black-box function **f**. This function is frequently difficulty or costly to evaluate---such as requiring wet-lab experiments in the case of protein design. In these cases, **f** is described by a set of function evaluations: D = {(x_0, y_0), (x_1, y_1), ... (x_n, y_n)}, and optimization is performed without querying **f** on new data points.

## Installation

Design-Baselines can be downloaded from github and installed using anaconda.

```bash
git clone git@github.com:brandontrabucco/design-baselines.git
conda create -f design-baselines/environment.yml
```

## Performance Of Baselines

We benchmark a set of 9 methods for solving offline model-based optimization problems. Performance is reported in normalized form, where the 100th percentile score of 128 candidate designs is evaluated and normalized such that a 1.0 corresponds to performance equivalent to the best performing design in the *full unobserved* dataset assoctated with each model-based optimization task. A 0.0 corresponds to performance equivalent to the worst performing design in the *full unobserved* dataset. In circumstances where an exact oracle is not available, this *full unobserved* dataset is used for training the approximate oracle that is used for evaluation of candidate designs proposed by each method. The symbol ± indicates the empirical standard deviation of reported performance across 8 trials.

### Performance On Continuous Tasks

Method \ Task                 | Superconductor | Ant Morphology | D'Kitty Morphology | Hopper Controller 
----------------------------- | -------------- | -------------- | ------------------ | -----------------
D (best)                      |          0.399 |          0.565 |              0.884 |               1.0
Auto. CbAS                    |  0.421 ± 0.045 |  0.884 ± 0.046 |      0.906 ± 0.006 |     0.137 ± 0.005 
CbAS                          |  0.503 ± 0.069 |  0.879 ± 0.032 |      0.892 ± 0.008 |     0.141 ± 0.012 
BO-qEI                        |  0.402 ± 0.034 |  0.820 ± 0.000 |      0.896 ± 0.000 |     0.550 ± 0.118 
CMA-ES                        |  0.465 ± 0.024 |  1.219 ± 0.738 |      0.724 ± 0.001 |     0.604 ± 0.215 
Grad.                         |  0.518 ± 0.024 |  0.291 ± 0.023 |      0.874 ± 0.022 |     1.035 ± 0.482 
Grad. Min                     |  0.506 ± 0.009 |  0.478 ± 0.064 |      0.889 ± 0.011 |     1.391 ± 0.589 
Grad. Mean                    |  0.499 ± 0.017 |  0.444 ± 0.081 |      0.892 ± 0.011 |     1.586 ± 0.454 
MINs                          |  0.469 ± 0.023 |  0.916 ± 0.036 |      0.945 ± 0.012 |     0.424 ± 0.166 
REINFORCE                     |  0.481 ± 0.013 |  0.263 ± 0.032 |      0.562 ± 0.196 |    -0.020 ± 0.067 
**COMs (Ours)**               |  0.439 ± 0.033 |  0.944 ± 0.016 |      0.949 ± 0.015 |     2.056 ± 0.314

### Performance On Discrete Tasks

Method \ Task                 |            GFP |      TF Bind 8 |            UTR 
----------------------------- | -------------- | -------------- | -------------- 
D (best)                      |          0.789 |          0.439 |          0.593
Auto. CbAS                    |  0.865 ± 0.000 |  0.910 ± 0.044 |  0.691 ± 0.012 
CbAS                          |  0.865 ± 0.000 |  0.927 ± 0.051 |  0.694 ± 0.010 
BO-qEI                        |  0.254 ± 0.352 |  0.798 ± 0.083 |  0.684 ± 0.000 
CMA-ES                        |  0.054 ± 0.002 |  0.953 ± 0.022 |  0.707 ± 0.014 
Grad.                         |  0.864 ± 0.001 |  0.977 ± 0.025 |  0.695 ± 0.013 
Grad. Min                     |  0.864 ± 0.000 |  0.984 ± 0.012 |  0.696 ± 0.009 
Grad. Mean                    |  0.864 ± 0.000 |  0.986 ± 0.012 |  0.693 ± 0.010 
MINs                          |  0.865 ± 0.001 |  0.905 ± 0.052 |  0.697 ± 0.010 
REINFORCE                     |  0.865 ± 0.000 |  0.948 ± 0.028 |  0.688 ± 0.010
**COMs (Ours)**               |  0.864 ± 0.000 |  0.945 ± 0.033 |  0.699 ± 0.011

## Reproducing Baseline Performance

To reproduce the performance of baseline algorithms reported in our work, you may then run the following series of commands in a bash terminal using the command-line interface exposed in design-baselines. Also, please ensure that the conda environment `design-baselines` is activated in the bash session that you run these commands from in order to access the `design-baselines` command-line interface.

```bash
# set up machine parameters
NUM_CPUS=32
NUM_GPUS=8

for TASK_NAME in \
    superconductor \
    ant \
    dkitty \
    hopper \
    gfp \
    tf-bind-8 \
    utr; do
    
  for ALGORITHM_NAME in \
      autofocused-cbas \
      cbas \
      bo-qei \
      cma-es \
      gradient-ascent \
      gradient-ascent-min-ensemble \
      gradient-ascent-mean-ensemble \
      mins \
      reinforce; do
  
    # launch several model-based optimization algorithms using the command line interface
    # for example: 
    # (design-baselines) name@computer:~/$ cbas gfp \
    #                                        --local-dir ~/db-results/cbas-gfp \
    #                                        --cpus 32 \
    #                                        --gpus 8 \
    #                                        --num-parallel 8 \
    #                                        --num-samples 8
    $ALGORITHM_NAME $TASK_NAME \
      --local-dir ~/db-results/$ALGORITHM_NAME-$TASK_NAME \
      --cpus $NUM_CPUS \
      --gpus $NUM_GPUS \
      --num-parallel 8 \
      --num-samples 8
    
  done
  
done

# generate the main performance table of the paper
design-baselines make-table --dir ~/db-results/ --percentile 100th

# generate the performance tables in the appendix
design-baselines make-table --dir ~/db-results/ --percentile 50th
design-baselines make-table --dir ~/db-results/ --percentile 100th --no-normalize
```

These commands will run several model-based optimization algorithms (such as [CbAS](http://proceedings.mlr.press/v97/brookes19a.html)) contained in design-baselines on all tasks released with the design-bench benchmark, and will then generate three performance tables from those results, and print a latex rendition of these performance tables to stdout.

## Running COMs

You may run COMs using the `design-baselines` command line interface in a bash session where the `design-baselines` anaconda environments is activated and the `design-baselines` pip package is installed. Below is an example command that will run COMs on the task `HopperController-Exact-v0` from [design-bench](https://github.com/brandontrabucco/design-bench).

```bash
coms --logging-dir ./coms-hopper \
     --not-fast \
     --task HopperController-Exact-v0 \
     --no-task-relabel \
     --normalize-ys \
     --normalize-xs \
     --particle-lr 0.05 \
     --particle-train-gradient-steps 50 \
     --particle-evaluate-gradient-steps 50 \
     --particle-entropy-coefficient 0.0 \
     --forward-model-activations relu \
     --forward-model-activations relu \
     --forward-model-hidden-size 2048 \
     --forward-model-lr 0.0003 \
     --forward-model-alpha 0.1 \
     --forward-model-alpha-lr 0.01 \
     --forward-model-overestimation-limit 0.5 \
     --forward-model-noise-std 0.0 \
     --forward-model-batch-size 128 \
     --forward-model-val-size 500 \
     --forward-model-epochs 50 \
     --evaluation-samples 128
```
