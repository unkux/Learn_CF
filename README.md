# Learn CF

Repo. for our paper "Predicting Cascading Failures with a Hyperparametric Diffusion Model".

## Running Environments

It's currently running on `Linux` with `Python 3.10.10`, can be ported to other OS.

## Requirements

It supports running on both general CPU and Intel GPU (via OpenCL standard), the latter requires package [numba-dpex](https://github.com/IntelPython/numba-dpex).

## Manual

The main file is `run.py` or a shell script `run.sh`. Besides, `mle_run.py, features.py, samples.py, simulate_igraph.py` can run separately by needs.

```
NAME
    run.py

SYNOPSIS
    run.py INSTANCE <flags>

POSITIONAL ARGUMENTS
    INSTANCE

FLAGS
    -o, --output=OUTPUT
        Default: ''
        Output directory.
    -v, --verbose=VERBOSE
        Default: 0
    --method=METHOD
        Default: 'lbfgsb'
        Optimization algorithm.
    -g, --gpu=GPU
        Default: False
    --poly_fts=POLY_FTS
        Default: False
        Use polynomial features.
    -d, --dist=DIST
        Default: False
        Use distance features.
    -c, --corr=CORR
        Default: 0.9
        Pearson correlation coefficient.
    --scaler=SCALER
        Default: 'max_abs'
        Feature scaler.
    --block=BLOCK
        Default: 1
        Split data.
    --sel=SEL
        Default: -1
        Select feature dimension.
    --a1=A1
        Default: 0.001
        L1 regularization.
    --a2=A2
        Default: 0.01
        L2 regularization.
    -B, --B=B
        Default: 'inf'
        Box boundary of hyperparameters.
    --tol=TOL
        Default: 1e-06
        Optimization tolerance.
    --maxiter=MAXITER
        Default: 300
        Optimization iteration.
    --pfunc=PFUNC
        Default: 'logistic'
        Influence probability function.
    --theta=THETA
        Default: 0
        Specify value for hyperparameters.
    --test=TEST
        Type: Optional[]
        Default: None
        Test instances.
    --rerun=RERUN
        Default: False
    --fr=FR
        Default: 1
        Feature rank filter.
    --rank=RANK
        Type: Optional[]
        Default: None
    -w, --weight=WEIGHT
        Default: []
        Sample weights.
    -b, --build_only=BUILD_ONLY
        Default: False
        Only build features and samples.
    --fc=FC
        Default: 0.05
        Cascading failures filter.
    -n, --no_mc=NO_MC
        Default: -1
        No. of Monte Carlo simulation.
    --fp=FP
        Default: 1
        Probability filter.
    --resample=RESAMPLE
        Default: False
        Resampling strategy.
    --precision=PRECISION
        Default: 6
    -i, --init_failures=INIT_FAILURES
        Default: ''
    -k, --k=K
        Default: 1
        N-k contingencies.
    --max_workers=MAX_WORKERS
        Default: 1
        Parallelization on CPU.
```

## Datasets

DATA STRUCTURES:
```
Instances
│
├── 1.00  # power demands factor
│   │
│   ├── generations.csv  # cascading failures
│   │
│   ├── ig_data.mat  # power grid data
│   │
│   ├── res_cnt_size.mat  # benchmarks
│   │
│   └── sub
│       ├── 9 -> ..  # for managing instances, can be any value
│       └── ...
├── ...
│
└── 2.00
    └── ...
```
