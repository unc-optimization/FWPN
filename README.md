# FWPN
A Frank-Wolfe Proximal Newton Algorithm

## Introduction
This Algorithm can solve the following constrained convex optimization problem:

<img src="https://render.githubusercontent.com/render/math?math=$\min_{x \in \mathcal{X} \subseteq \mathbb{R}^p} f(x) $"> 


where <img src="https://render.githubusercontent.com/render/math?math=$f:\mathbb{R}^p\to\mathbb{R}\cup\{+\infty\}$"> is self-concordant and <img src="https://render.githubusercontent.com/render/math?math=$\mathcal{X}$"> is a compact convex set, whose linear optimization oracle is easy to find.

## Prerequisites

The code is tested under Matlab R2018b and it doesn't requires additional MATLAB toolbox. Before running the examples, please change the  your MATLAB's current work directory to

```
path/to/FWPN/
```

## Running the examples

we implemented three examples to test our algorithm.

### 1. Portfolio optimization example

This example is implemented in the `Port_Opt_example.m` script which can both run real and synthetic data.

For example, set the varibles `use_real_data = 1` and `id = 1` and run the `Port_Opt_example.m` script to test for the real stock data `473500_wk.mat`.

Set the varibles `use_real_data = 0`, `n = 1e+4`, `p = 1e+3`and run the `Port_Opt_example.m` script to test for the synthetic stock data with size (1e+4, 1e+3). 

### 2. D-optimal design example

This example is implemented in the `Dopt_example.m` script.

Set the varibles `n = 1e+2`, `p = 1e+3`and run the `Dopt_example.m` script to test for the data with size (1e+2, 1e+3). 

### 3. Logistic Regression example

This example is implemented in the `Log_Reg_example.m` script.

We support LIBSVM datasets which can be downloaded [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The downloaded file should be unzipped and put in the following folder

```
path/to/FWPN/data/log_reg/
```

Suppose the file name is `w8a.t`, set the variable `fname = 'w8a.t'` and run the `Log_Reg_example.m` script to test for `w8a` testing dataset in LIBSVM.
