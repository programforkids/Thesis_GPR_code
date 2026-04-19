import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from gp_algorithms import gaussian_process_regression, gaussian_process_regression_finite,covariancematrix,draw_posterior, draw_posterior_without_samples
from gp_kernels import squared_exponential_covariance_function

plt.rcParams.update({
    "font.size": 16,          
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
 
# figure 1
X = [0.1, 0.2, 0.5, 1, 3, 3.5, 5]
Y = [1, 2, 1.7, 4, 6, 5.5, 2]

#a)
fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.axvline(x=2, color='red', linestyle='--')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
fig.tight_layout()
plt.show()

# linear vs GPR
testinputs = np.linspace(0, 6, 1000)

# linear regression
X_design = np.vstack([np.ones(len(X)), X]).T
theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y
b = theta[0]
w = theta[1]
y_pred = b + w * testinputs

# GPR
means, C, l = gaussian_process_regression(X, Y, lambda x:0 ,squared_exponential_covariance_function(1.5, 1),0.01, testinputs)
stds = np.sqrt(np.diag(C))
#b)
fig, ax = plt.subplots()
ax.scatter(X, Y, s=40, edgecolors="black", linewidths=0.7, label="data", zorder=3)
ax.axvline(x=2, color="red", linestyle="--", linewidth=1.2, label=r"$x=2$")
ax.plot(testinputs, y_pred, color="black", linestyle=":", linewidth=1.5, label="linear regression")
ax.plot(testinputs, means, color="black", linewidth=2.0, label="GPR mean")
ax.fill_between(testinputs, means - stds, means + stds,
                color="#9ecae1", alpha=0.35, label=r"GPR $\pm 1$ std.")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# figure 2
colors = ['red', 'blue', 'green', 'orange']  # colors for 4 different samples

# mean and covariance
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8],
                  [0.8, 2]])

# density
rv = multivariate_normal(mu, Sigma)
samples = np.random.multivariate_normal(mu, Sigma, 4)

# grid for ellipsoid visualization
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X_grid, Y_grid = np.meshgrid(x, y)
pos = np.dstack((X_grid, Y_grid))
Z = rv.pdf(pos)

#a)
fig, ax = plt.subplots()
ax.contour(X_grid, Y_grid, Z, levels=6, colors='black', linewidths=1)
ax.scatter(mu[0], mu[1], color='red')
for i in range(4):
    ax.scatter(samples[i, 0], samples[i, 1], color=colors[i])
ax.set_xlabel(r"$f(1)$")
ax.set_ylabel(r"$f(2)$")
fig.tight_layout()
plt.show()

#b)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot([1, 2], samples[i], marker='o', color=colors[i])
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xlabel(r"input index $x$")
ax.set_ylabel(r"sample value $f(x)$")
fig.tight_layout()
plt.show()

# figure 3
# 5D
mu2 = np.zeros(5)
Sigma2 = [[1 - 0.1 * abs(i - j) for j in range(5)] for i in range(5)]
samples2 = np.random.multivariate_normal(mu2, Sigma2, 4)

#a)
fig, ax = plt.subplots()
ax.plot([(j+1) for j in range(5)], mu2, label='mean')
stds = np.sqrt(np.diag(Sigma2))
ax.errorbar([(j+1) for j in range(5)], mu2, yerr=stds, fmt='none',
            ecolor='#6baed6', capsize=3,
            label='mean plus/minus 1*standard deviation')
for i in range(4):
    ax.plot([(j+1) for j in range(5)], samples2[i], marker='o', color=colors[i])
ax.set_xticks([(k+1) for k in range(5)])
ax.set_xlabel(r"input index $x$")
ax.set_ylabel(r"sample value $f(x)$")
fig.tight_layout()
plt.show()

# 20D
mu3 = np.zeros(20)
Sigma3 = [[1 - 0.05 * abs(i - j) for j in range(20)] for i in range(20)]
samples3 = np.random.multivariate_normal(mu3, Sigma3, 4)

#b)
fig, ax = plt.subplots()
ax.plot([(j+1) for j in range(20)], mu3, label='mean')
stds = np.sqrt(np.diag(Sigma3))
ax.errorbar([(j+1) for j in range(20)], mu3, yerr=stds, fmt='none',
            ecolor='#6baed6', capsize=3,
            label='mean plus/minus 1*standard deviation')
for i in range(4):
    ax.plot([(j+1) for j in range(20)], samples3[i], marker='o', color=colors[i])
ax.set_xticks([2*(k+1) for k in range(10)])
ax.set_xlabel(r"input index $x$")
ax.set_ylabel(r"sample value $f(x)$")
fig.tight_layout()
plt.show()

mu3=np.zeros(20)
Sigma3=[[1-0.05*abs(i-j) for j in range(20)]for i in range(20)]
inputs= np.array([1,8])
targets= np.array([0,2])
testinputs = np.arange(1,21)
#20 dimensional conditioning example
mu3=np.zeros(20)
Sigma3=[[1-0.05*abs(i-j) for j in range(20)]for i in range(20)]
inputs= np.array([1,8])
targets= np.array([0,2])
testinputs = np.arange(1,21)
means,C,l=gaussian_process_regression_finite(inputs,targets,Sigma3,0.001,testinputs)
#samples
samples3=np.random.multivariate_normal(means, C, 4)
colors = ['red','blue','green','orange']
#4a)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(testinputs, samples3[i], marker='o', color=colors[i])
    ax.set_xticks([2*(k+1) for k in range(10)])
ax.set_xlabel(r"input index $x$")
ax.set_ylabel(r"function value $f(x)$")
fig.tight_layout()
fig.show()
#means  with variances
stds=np.sqrt((np.diag(C)))
#4b)
fig, ax=plt.subplots()
ax.scatter(testinputs,means)
ax.errorbar(testinputs,means,yerr=stds,      fmt='none',   ecolor='#6baed6',capsize=3,label='mean plus/minus 1*standard deviation')
ax.set_xticks([(2*(k+1)) for k in range(10)])
ax.set_xlabel(r"input index $x$")
ax.set_ylabel(r"$m(x)\pm std(x)$")
fig.tight_layout()
plt.show()

grid = np.linspace(0, 10, 1000)
lengthscales = [0.5, 1, 2, 4]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
#figure 5
plt.figure(figsize=(8,5))
for l, c in zip(lengthscales, colors):
    k = squared_exponential_covariance_function(lengthscale=l,signalvariance=1)(0, grid)
    plt.plot(grid, k, color=c, linewidth=2, label=rf"$\ell={l}$")
plt.xlabel(r"distance $r = |x-x'|$", fontsize=12)
plt.ylabel(r"$\kappa(r)$", fontsize=12)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

C=covariancematrix(squared_exponential_covariance_function(1,1),grid) 
#figure 6
draw_posterior(np.zeros(len(grid)),C,grid,number_samples=20, x_obs=None, y_obs=None)

# Data
x_values = np.array([0, 0.3, 1, 3.1, 4.7])
y_values = np.array([1, 0, 1.4, 0, -0.9])
#figure 7
means, C, l = gaussian_process_regression(x_values, y_values,lambda x:0, squared_exponential_covariance_function(1,1), 0.001, grid)
#a)
draw_posterior_without_samples(means, C, grid, x_obs=x_values, y_obs=y_values)

means, C, l = gaussian_process_regression(x_values, y_values,lambda x:0, squared_exponential_covariance_function(1,1), 0.1, grid)
#b)
draw_posterior_without_samples(means, C, grid, x_obs=x_values, y_obs=y_values)

means, C, l = gaussian_process_regression(x_values, y_values,lambda x:0, squared_exponential_covariance_function(1,1), 1, grid)
#c)
draw_posterior_without_samples(means, C, grid, x_obs=x_values, y_obs=y_values)

def Updatinggp(covariancefunction, noise, inputs, outputs, grid):
    inputs = np.array(inputs, dtype=float)
    outputs = np.array(outputs, dtype=float)

    for i in range(1, len(inputs) + 1):
        current_x = inputs[:i]
        current_y = outputs[:i]

        mean, C, lml = gaussian_process_regression(current_x, current_y, lambda x: 0, covariancefunction, noise, grid)
        draw_posterior_without_samples(mean, C, grid,x_obs=current_x, y_obs=current_y)
#figure 8
Updatinggp(
    squared_exponential_covariance_function(1, 1),
    0.1,
    x_values,
    y_values,
    grid
)   
