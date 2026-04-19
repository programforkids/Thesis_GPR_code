import numpy as np
import matplotlib.pyplot as plt

from gp_algorithms import gaussian_process_regression,draw_posterior
from gp_kernels import squared_exponential_covariance_function
from optimize_se_fixed_noise import maximize_log_marginal_likelihood_se_fixed_noise, unpack_log_theta
from optimize_se import maximize_log_marginal_likelihood_se, unpack_log_theta_se
from plotting_ch5 import contourplot_se_fixed_noise

plt.rcParams.update({
    "font.size": 16,          
    "axes.titlesize": 14,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def run():
    X = np.array([0, 0.3, 1, 3.1, 4.7], dtype=float)
    Y = np.array([1, 0, 1.4, 0, -0.9], dtype=float)
    grid = np.linspace(-3, 8, 500)
    theta0_log=np.log(np.array([1.0, 1.0,0.1]))
    theta_opt_log,history,grad,steps = maximize_log_marginal_likelihood_se(X=X,Y=Y,theta0_log=theta0_log,verbose=True)
    sigma2, ell, noise2 = unpack_log_theta_se(theta_opt_log)
    print("SE optimum:", sigma2, ell, noise2, history[-1])
    #SE optimum: 0.2791599098963631 1.6243003330003665 0.4621657421601115 -6.1454749148264
    #figure 16
    fig,ax=plt.subplots()
    ax.plot(history, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log marginal likelihood")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(grad, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(steps, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    plt.show() 
    mean, cov, _ = gaussian_process_regression(X, Y, lambda x: 0, squared_exponential_covariance_function(ell, sigma2), noise2, grid)
    #figure 17
    draw_posterior(mean, cov, grid,number_samples=4, x_obs=X, y_obs=Y)
    theta0_log = np.log(np.array([1.0, 1.0]))
    theta_opt_log, history, grad_norms, step_sizes = maximize_log_marginal_likelihood_se_fixed_noise(
        X=X,
        Y=Y,
        theta0_log=theta0_log,
        noise2_fixed=0.1,
        alpha0=2.0,
        verbose=True,
    )
    sigma2_se, ell_se = unpack_log_theta(theta_opt_log)
    print("SE fixed noise optimum:", sigma2_se, ell_se, history[-1])
    #SE fixed noise optimum: 0.6539968115106783 0.05450065254517818 -6.388785388610272
    #figure 18
    fig,ax=plt.subplots()
    ax.plot(history, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log marginal likelihood")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(grad_norms, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(step_sizes, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    plt.show() 
    sigma2, ell = unpack_log_theta(theta_opt_log)
    mean, cov, _ = gaussian_process_regression(X, Y, lambda x: 0, squared_exponential_covariance_function(ell, sigma2), 0.1, grid)
    #figure 19
    draw_posterior(mean, cov, grid,number_samples=4, x_obs=X, y_obs=Y)
    #figure 20
    contourplot_se_fixed_noise(
        X, Y,
        log_noise2_fixed=theta_opt_log[1],
        log_sigma2_range=(theta_opt_log[0] - 4.0, theta_opt_log[0] + 4.0),
        log_ell_range=(theta_opt_log[1] - 4.0, theta_opt_log[1] + 4.0),
        theta_opt_log=theta_opt_log,
        subsample_step=1, levels=50
    )


if __name__ == "__main__":
    run()
