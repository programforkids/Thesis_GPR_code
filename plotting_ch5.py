from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gp_kernels import periodic_kernel, squared_exponential_covariance_function, sum_kernels
from gp_algorithms import precompute_sin2_term, precompute_sqdist, gp_predict
from optimize_periodic_only import (
    FIXED_ELL_SE,
    FIXED_PERIOD,
    FIXED_SIGMA2_SE,
    log_marginal_likelihood_periodic_from_precomputed,
    maximize_log_marginal_likelihood_periodic_only,
    unpack_log_theta_periodic_only,
)
from optimize_se import (
    log_marginal_likelihood_se_from_sqdist,
    maximize_log_marginal_likelihood_se,
    unpack_log_theta_se,
)
plt.rcParams.update({
    "font.size": 16,          
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
DATA_PATH = Path(__file__).resolve().parent / "data" / "co2_data.csv"


def load_co2_data():
    df = pd.read_csv(DATA_PATH, sep=",")
    df["Date"] = df["Date"].str.replace(".", "", regex=False).astype(float) / 10000
    df["CO2"] = df["CO2"].astype(float)
    X = df["Date"].values.astype(float)
    Y_raw = df["CO2"].values.astype(float)
    Y_centered = Y_raw - Y_raw.mean()
    return df, X, Y_centered, Y_raw.mean()


def contourplot_se_fixed_noise(X, Y, log_noise2_fixed, log_sigma2_range, log_ell_range, theta_opt_log=None, subsample_step=1, levels=20):
    X_plot = np.asarray(X, dtype=float)[::subsample_step]
    Y_plot = np.asarray(Y, dtype=float)[::subsample_step]
    sqdist = precompute_sqdist(X_plot)

    log_sigma2_vals = np.linspace(log_sigma2_range[0], log_sigma2_range[1], 30)
    log_ell_vals = np.linspace(log_ell_range[0], log_ell_range[1], 30)

    Z = np.empty((len(log_ell_vals), len(log_sigma2_vals)), dtype=float)

    for i, log_ell in enumerate(log_ell_vals):
        for j, log_sigma2 in enumerate(log_sigma2_vals):
            Z[i, j] = log_marginal_likelihood_se_from_sqdist(
                Y_plot, sqdist, log_sigma2, log_ell, log_noise2_fixed
            )
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(log_sigma2_vals, log_ell_vals, Z, levels=levels)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Log marginal likelihood")
    ax.set_xlabel(r"$\log \sigma^2_{\mathrm{SE}}$")
    ax.set_ylabel(r"$\log \ell_{\mathrm{SE}}$")
    if theta_opt_log is not None:
        ax.plot(theta_opt_log[0], theta_opt_log[1],
                marker="x", markersize=10, markeredgewidth=2,
                color="red", label="Estimated optimum")
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def contourplot_periodic_fixed_se_noise(X, Y, sigma2_se_fixed, ell_se_fixed, noise2_fixed, log_sigma2_periodic_range, log_ell_periodic_range, theta_opt_log=None, subsample_step=1, levels=20, period_fixed=1.0):
    X_plot = np.asarray(X, dtype=float)[::subsample_step]
    Y_plot = np.asarray(Y, dtype=float)[::subsample_step]
    sqdist = precompute_sqdist(X_plot)
    sin2_term = precompute_sin2_term(X_plot, period=period_fixed)
    log_sigma2_vals = np.linspace(log_sigma2_periodic_range[0], log_sigma2_periodic_range[1], 30)
    log_ell_vals = np.linspace(log_ell_periodic_range[0], log_ell_periodic_range[1], 30)
    Z = np.empty((len(log_ell_vals), len(log_sigma2_vals)), dtype=float)
    for i, log_ell in enumerate(log_ell_vals):
        for j, log_sigma2 in enumerate(log_sigma2_vals):
            Z[i, j] = log_marginal_likelihood_periodic_from_precomputed(
                Y=Y_plot,
                sqdist=sqdist,
                sin2_term=sin2_term,
                log_sigma2_periodic=log_sigma2,
                log_ell_periodic=log_ell,
                sigma2_se_fixed=sigma2_se_fixed,
                ell_se_fixed=ell_se_fixed,
                noise2_fixed=noise2_fixed,
            )
    fig,ax=plt.subplots(figsize=(8, 6))
    contour = ax.contourf(log_sigma2_vals, log_ell_vals, Z, levels=levels)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Log marginal likelihood")
    ax.set_xlabel(r"$\log \sigma^2_{\mathrm{per}}$")
    ax.set_ylabel(r"$\log \ell_{\mathrm{per}}$")
    if theta_opt_log is not None:
        ax.plot(theta_opt_log[0], theta_opt_log[1], marker="x", markersize=10, markeredgewidth=2, color="red", label="Estimated optimum")
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def draw_posterior_CO2(mean, cov, inputs):
    stds = np.sqrt(np.maximum(np.diag(cov), 0))
    plt.figure(figsize=(10, 5))
    plt.fill_between(inputs, mean - 3 * stds, mean + 3 * stds, alpha=0.2, label=r'±3 std')
    plt.fill_between(inputs, mean - 2 * stds, mean + 2 * stds, alpha=0.3, label=r'±2 std')
    plt.fill_between(inputs, mean - stds, mean + stds, alpha=0.4, label=r'±1 std')
    plt.plot(inputs, mean, linewidth=2, label=r'mean')
    plt.xlabel(r"years")
    plt.ylabel(r"CO$_2$ (ppm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run():
    df, X, Y, Y_mean = load_co2_data()
    #figure 5.1
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['CO2'],label=r"$\mathrm{CO}_2$ concentration", linewidth=2)
    plt.xlabel(r'Year')
    plt.ylabel(r'CO$_2$ (ppm)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    #figure 5.5
    df_2024 = df[(df["Date"] >= 2024.0) & (df["Date"] < 2025.0)].copy()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    plt.figure(figsize=(10,5))
    plt.plot(months, df_2024["CO2"], marker="o", linewidth=2,label=r"$\mathrm{CO}_2$ concentration in 2024")
    plt.xlabel(r"Month")
    plt.ylabel(r"CO$_2$(ppm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Optimizing SE model...")
    theta0_log = np.log(np.array([66 ** 2, 67, 2.44 ** 2], dtype=float))
    theta_se_opt, hist_se, grad_se, steps_se = maximize_log_marginal_likelihood_se(
    X=X,
    Y=Y,
    theta0_log=theta0_log,
    tol=1e-3,
    max_iter=100,
    alpha0=0.05,
    c=1e-4,
    rho=0.5,
    verbose=True
    )
    sigma2_se, ell_se, noise2_se = unpack_log_theta_se(theta_se_opt)
    print("SE optimum:", sigma2_se, ell_se, noise2_se, hist_se[-1])
    #SE optimum: 4177.913672312089 36.5715721099202 4.76076515293103 -1791.1259319227797
    #figure 5.2
    fig,ax=plt.subplots()
    ax.plot(hist_se, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log marginal likelihood")
    fig.tight_layout()
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(grad_se, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(steps_se, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    plt.show()
    #figure 5.3
    contourplot_se_fixed_noise(
        X, Y,
        log_noise2_fixed=theta_se_opt[2],
        log_sigma2_range=(theta_se_opt[0] - 4.0, theta_se_opt[0] + 4.0),
        log_ell_range=(theta_se_opt[1] - 4.0, theta_se_opt[1] + 4.0),
        theta_opt_log=theta_se_opt,
        subsample_step=1, levels=20
    )
    grid = np.linspace(X[0], X[-1] + 20, 1000)
    mean_se, cov_se, _ = gp_predict(X, Y + Y_mean, squared_exponential_covariance_function(ell_se, sigma2_se), noise2_se, grid)
    #figure 5.4
    draw_posterior_CO2(mean_se + Y_mean, cov_se, grid)
    print("Optimizing periodic-only extension...")
    theta0_per = np.log(np.array([2.44 ** 2, 1.3, 4.76076515293103], dtype=float))
    theta_per_opt, hist_per, grad_per, steps_per = maximize_log_marginal_likelihood_periodic_only(X, Y, theta0_per, max_iter=100, alpha0=0.01, verbose=True)
    sigma2_per, ell_per, noise2_per = unpack_log_theta_periodic_only(theta_per_opt)
    print("Periodic optimum:", sigma2_per, ell_per, noise2_per, hist_per[-1])
    #periodic optimum: 7.996450966501127 1.4484533046867 0.3001718169244376 -709.4581887386555
    
    #figure 5.6
    fig,ax=plt.subplots()
    ax.plot(hist_per, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log marginal likelihood")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(grad_per, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    plt.show()
    fig,ax=plt.subplots()
    ax.plot(steps_per, marker="o")
    ax.grid(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    plt.show()         
    #figure 5.7
    contourplot_periodic_fixed_se_noise(
        X, Y,
        sigma2_se_fixed=FIXED_SIGMA2_SE,
        ell_se_fixed=FIXED_ELL_SE,
        noise2_fixed=noise2_per,
        log_sigma2_periodic_range=(theta_per_opt[0] - 3.0, theta_per_opt[0] + 3.0),
        log_ell_periodic_range=(theta_per_opt[1] - 3.0, theta_per_opt[1] + 3.0),
        theta_opt_log=theta_per_opt,
        subsample_step=1,levels=20,
        period_fixed=FIXED_PERIOD,
    )
    
    kernel = sum_kernels(
        squared_exponential_covariance_function(FIXED_ELL_SE, FIXED_SIGMA2_SE),
        periodic_kernel(ell_per, FIXED_PERIOD, sigma2_per),
    )
    mean_per, cov_per, _ = gp_predict(X, Y + Y_mean, kernel, noise2_per, grid)
    #figure 5.8
    draw_posterior_CO2(mean_per + Y_mean, cov_per, grid)




if __name__ == "__main__":
    run()
