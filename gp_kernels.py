import numpy as np


def squared_exponential_covariance_function(lengthscale: float = 1.0, signalvariance: float = 1.0):
    def squared(x1, x2):
        return signalvariance * np.exp(-((x1 - x2) ** 2) / (2.0 * lengthscale ** 2))
    return squared


def exponential_covariance_function(lengthscale: float = 1.0, signalvariance: float = 1.0):
    def kernel(x1, x2):
        return signalvariance * np.exp(-np.abs(x1 - x2) / lengthscale)
    return kernel


def rational_quadratic_kernel(lengthscale: float = 1.0, signalvariance: float = 1.0, alpha: float = 1.0):
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    def kernel(x1, x2):
        r = np.abs(x1 - x2)
        return signalvariance * (1.0 + (r ** 2) / (2.0 * alpha * lengthscale ** 2)) ** (-alpha)
    return kernel


def periodic_kernel(lengthscale: float = 1.0, period: float = 1.0, signalvariance: float = 1.0):
    def kernel(x1, x2):
        diff = np.pi * (x1 - x2) / period
        return signalvariance * np.exp(-2.0 * np.sin(diff) ** 2 / (lengthscale ** 2))
    return kernel


def sum_kernels(kernel1, kernel2):
    def kernel(x1, x2):
        return kernel1(x1, x2) + kernel2(x1, x2)
    return kernel
