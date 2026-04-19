# Gaussian Process Regression for CO2 Data

This repository contains the Python code used for the bachelor thesis "Gaussian Process Regression with Applications to Modeling and Predicting Atmospheric 
CO² Concentrations"

## Main files
- `algorithms.py`: Contains just the algorithm given in the thesis and is also given in the appendix. The python files do not access this file. Instead gp_algorithm.py is containing all the general GP algorithms needed and the hyper parameter optimization (alg9 and alg. 12) is stretched into 3 different files, cause it  is easier to code it efficient in that way. The 3 hyperparameter optimization algorithms do still work the same way. The optimization routines use logarithmic hyperparameters and bounds are imposed to improve numerical stability.
- `gp_algorithms.py`: contains the algorithm given in the thesis + more stable versions using numpy algorithms for the application of Predicting Atmospheric CO2 concentrations
- `gp_kernels.py`: kernel definitions
- `optimize_se.py`: Hyperparamter optimization given an SE kernel
- `optimize_se_fixed_noise.py`: Hyperparamter optimization given an SE kernel with fixed noise
- `optimize_periodic_only.py`: Hyperparameter optimization given an SE + periodic kernel with fixed SE parameters
- `plotting_ch3.py`: figures for Chapter 3
- `plotting_ch4.py`: figures for Chapter 4
- `plotting_ch5.py`: figures for Chapter 5
- `plotting_ch6.py`: figures for Chapter 6 and contourplot definition

## Data

The CO2 data file is stored in `data/co2_data.csv`.


