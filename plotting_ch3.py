import numpy as np
import matplotlib.pyplot as plt

from gp_algorithms import covariancematrix
from gp_kernels import (
    exponential_covariance_function,
    rational_quadratic_kernel,
    squared_exponential_covariance_function,
    periodic_kernel
)


def draw_samples(covariance_function, inputs, times=2):
    inputs = np.asarray(inputs, dtype=float)
    cov = covariancematrix(covariance_function, inputs)
    mean = np.zeros(len(inputs))
    for _ in range(times):
        sample = np.random.multivariate_normal(mean, cov)
        upcrossings = np.sum((sample[:-1] < 0) & (sample[1:] > 0))
        print("Zero upcrossings:", upcrossings)
        plt.figure(figsize=(8, 4))
        plt.plot(inputs, sample)
        plt.xlabel(r"input,$x$", fontsize=18)
        plt.ylabel(r"output,$f(x)$", fontsize=18)
        plt.tick_params(axis='both', labelsize=15)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
def upcrossings(covariance_function, inputs, times=100):
    inputs = np.asarray(inputs, dtype=float)
    cov = covariancematrix(covariance_function, inputs)
    mean = np.zeros(len(inputs))
    total_upcrossings = 0
    for _ in range(times):
        sample = np.random.multivariate_normal(mean, cov)
        total_upcrossings += np.sum((sample[:-1] < 0) & (sample[1:] > 0))
    return total_upcrossings / times


def run():
    inputs = np.linspace(0, 100, 1000)
    draw_samples(rational_quadratic_kernel(1, 1, 0.1), inputs, times=1)
    draw_samples(rational_quadratic_kernel(1, 1, 10), inputs, times=1)
    #figure 3.1
    draw_samples(squared_exponential_covariance_function(1, 1), inputs, times=1)
    draw_samples(squared_exponential_covariance_function(10,1),inputs,times=1)
    #figure 3.3
    draw_samples(exponential_covariance_function(1, 1), inputs, times=1)
    draw_samples(exponential_covariance_function(1, 1), np.linspace(0, 100, 2000), times=1)
    #figure 3.5
   
    
    #figure 3.7
    draw_samples(periodic_kernel(1, 10, 1), inputs, times=1)
    draw_samples(periodic_kernel(1, 20, 1), inputs, times=1)
    #figure 3.2
    grid = np.linspace(0, 10, 1000)
    plt.figure(figsize=(8, 5))
    for l in [0.5, 1, 2, 4]:
        plt.plot(grid, np.exp(-grid / l), label=rf"$\ell={l}$")
    plt.xlabel(r"input distance,$r$", fontsize=13.5)
    plt.ylabel(r"covariance,$\kappa(r)$", fontsize=13.5)
    plt.tick_params(axis='both', labelsize=11.25)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    #figure 3.4
    plt.figure(figsize=(8, 5))
    for alpha in [0.5, 1, 2, 4]:
        vals = [rational_quadratic_kernel(1, 1, alpha)(0.0, r) for r in grid]
        plt.plot(grid, vals, label=rf"$\alpha={alpha}$")
    plt.xlabel(r"input distance,$r$", fontsize=13.5)
    plt.ylabel(r"covariance,$\kappa(r)$", fontsize=13.5)
    plt.tick_params(axis='both', labelsize=11.25)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    #figure 3.6
    plt.figure(figsize=(8, 5))
    for p in [5, 10, 20, 50]:
        vals = [periodic_kernel(1, p, 1)(0.0, r) for r in grid]
        plt.plot(grid, vals, label=rf"$p={p}$")
    plt.xlabel(r"input distance,$r$", fontsize=13.5)
    plt.ylabel(r"covariance,$\kappa(r)$", fontsize=13.5)
    plt.tick_params(axis='both', labelsize=11.25)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    #upcrossings
    print("Upcrossings (RQ, alpha=1):", upcrossings(rational_quadratic_kernel(1, 1, 1), inputs, times=100))
    #example result: Upcrossings (RQ, alpha=1)=16.14
    print("Upcrossings (SE):", upcrossings(squared_exponential_covariance_function(1, 1), inputs, times=100))
    #example results: Upcrossings (SE): 15.69
    print("Upcrossings (Periodic):", upcrossings(periodic_kernel(1, 10, 1), inputs, times=100))
    #example results: Upcrossings (Periodic): 10.1

if __name__ == "__main__":
    run()
