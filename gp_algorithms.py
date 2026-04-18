import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 16,          
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def cholesky_decomposition(A):
    """Implements Algorithm 1 from thesis"""
    A = np.array(A, dtype=float, copy=True)
    n, m = A.shape
    L = np.zeros((n, n))
    for k in range(n):
        if A[k, k] <= 0:
            raise ValueError("Matrix is not positive definite")
        else:
            L[k, k] = np.sqrt(A[k, k])
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / L[k, k]
            for j in range(k + 1, i + 1):
                A[i, j] -= L[i, k] * L[j, k]
    return L

def forward_substitution(L, b):
    """Implements Algorithm 2 from thesis"""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def forward_substitution_matrix(L, B):
    """Implements Algorithm 3 from thesis"""
    n = L.shape[0]
    d = B.shape[1]
    Y = np.zeros((n, d))
    for i in range(d):
        Y[:, i] = forward_substitution(L, B[:, i])
    return Y

def backward_substitution(R, y):
    """Implements Algorithm 4 from thesis"""
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x

def backward_substitution_matrix(R, Y):
    """Implements Algorithm 5 from thesis"""
    n = R.shape[0]
    d = Y.shape[1]
    X = np.zeros((n, d))
    for i in range(d):
        X[:, i] = backward_substitution(R, Y[:, i])
    return X

def solve_cholesky(L, b):
    """Implements Algorithm 6 from thesis"""
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x

def solve_cholesky_matrix(L, B):
    """Implements Algorithm 7 from thesis"""
    Y = forward_substitution_matrix(L, B)
    X = backward_substitution_matrix(L.T, Y)
    return X

#==================helper functions for Gaussian process regression==================#
def covariancematrix(covariancefunction, inputs):
    n = len(inputs)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = covariancefunction(inputs[i], inputs[j])
    return A

def crosscovariancematrix(inputs, testinputs, covariancefunction):
    return np.array([[covariancefunction(xi, xj) for xj in testinputs] for xi in inputs])

def sum_log_diag(L):
    return np.sum(np.log(np.diag(L)))
#===============================GPR Algorithm===============================#
def gaussian_process_regression(inputs, targets, meanfunction, covariancefunction, noiselevel, testinputs):
    """Implements Algorithm 8 from thesis"""
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets, dtype=float)
    testinputs = np.array(testinputs, dtype=float)

    K = covariancematrix(covariancefunction, inputs)
    K += noiselevel * np.eye(len(inputs))

    L = cholesky_decomposition(K)
    alpha = solve_cholesky(L, targets - meanfunction(inputs))

    Kstar = crosscovariancematrix(inputs, testinputs, covariancefunction)
    predictive_mean = meanfunction(testinputs) + Kstar.T @ alpha

    v = forward_substitution_matrix(L, Kstar)
    predictive_covariance_matrix = covariancematrix(covariancefunction, testinputs) - np.transpose(v) @ v

    log_marginal_likelihood = (
        -0.5 * float(np.transpose(targets - meanfunction(inputs)) @ alpha)
        - sum_log_diag(L)
        - (len(inputs) / 2.0) * np.log(2 * np.pi)
    )

    return predictive_mean, predictive_covariance_matrix, log_marginal_likelihood
#We use this algorithm for Conditioning in finite dimension even though this is not needed since a small matrix is easily invertible. In this case it is only for prior mean=0
def gaussian_process_regression_finite(inputs, targets, covariancematrix, noiselevel, testinputs):
    inputs = np.array(inputs)
    targets = np.array(targets)
    K= np.zeros((len(inputs),len(inputs)))
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            K[i][j]=covariancematrix[inputs[i]-1][inputs[j]-1]
    K += noiselevel * np.eye(len(inputs))
    L = cholesky_decomposition(K)
    alpha = solve_cholesky(L, targets)
    Kstar= np.zeros((len(inputs),len(testinputs)))
    for i in range(len(inputs)):
        for j in range(len(testinputs)):
            Kstar[i][j]=covariancematrix[inputs[i]-1][testinputs[j]-1]
    means= np.transpose(Kstar) @ alpha
    v = forward_substitution_matrix(L,Kstar)
    Kstarstar= np.zeros((len(testinputs),len(testinputs)))
    for i in range(len(testinputs)):
        for j in range(len(testinputs)):
            Kstarstar[i][j]=covariancematrix[testinputs[i]-1][testinputs[j]-1]    
    C = Kstarstar - np.transpose(v) @ v
    log_marginal_likelihood = -0.5 * float(np.transpose(targets) @ alpha) - sum_log_diag(L) - (len(inputs)/2.0)*np.log(2*np.pi)
    return means, C, log_marginal_likelihood
#===============================Visualization for posterior ====================='
def draw_posterior(mean, cov, inputs, number_samples, x_obs=None, y_obs=None):
    stds = np.sqrt(np.maximum(np.diag(cov), 0))
    fig1, ax1 = plt.subplots()
    #uncertainty bands
    ax1.fill_between(inputs, mean - 3 * stds, mean + 3 * stds, alpha=0.2)
    ax1.fill_between(inputs, mean - 2 * stds, mean + 2 * stds, alpha=0.3)
    ax1.fill_between(inputs, mean - stds, mean + stds, alpha=0.4)
    #mean
    ax1.plot(inputs, mean, linewidth=2)
    #observations
    if x_obs is not None and y_obs is not None:
        ax1.scatter(x_obs, y_obs, marker='x', s=80)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$f(x)$")
    ax1.grid(True)
    fig1.tight_layout()
    plt.show()
    fig2, ax2 = plt.subplots()
    #samples
    samples = np.random.multivariate_normal(mean, cov, size=number_samples)
    for i in range(number_samples):
        ax2.plot(inputs, samples[i], alpha=0.8)
    if x_obs is not None and y_obs is not None:
        ax2.scatter(x_obs, y_obs, marker='x', s=80, color='red')
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$f(x)$")
    ax2.grid(True)
    fig2.tight_layout()
    plt.show()
def draw_posterior_without_samples(mean, cov, inputs, x_obs=None, y_obs=None):
    stds = np.sqrt(np.maximum(np.diag(cov), 0))
    fig, ax = plt.subplots()
    #uncertainty bands
    ax.fill_between(inputs, mean - 3 * stds, mean + 3 * stds, alpha=0.2)
    ax.fill_between(inputs, mean - 2 * stds, mean + 2 * stds, alpha=0.3)
    ax.fill_between(inputs, mean - stds, mean + stds, alpha=0.4)
    #mean
    ax.plot(inputs, mean, linewidth=2)
    ax.scatter(x_obs, y_obs, marker='x', s=80)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
#=============stable versions used for hyperparameters optimization and big data===============#
def stable_cholesky_fast(
    A: np.ndarray,
    base_jitter: float = 1e-10,
    max_tries: int = 10,
) -> tuple[np.ndarray, float]:
    """stable version of algorithm 1 from thesis, returns Cholesky factor and jitter added"""
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix contains NaN or inf.")

    A = 0.5 * (A + A.T)
    jitter = 0.0
    for k in range(max_tries):
        try:
            if k == 0:
                return np.linalg.cholesky(A), jitter
            jitter = base_jitter * (10 ** (k - 1))
            return np.linalg.cholesky(A + jitter * np.eye(n)), jitter
        except np.linalg.LinAlgError:
            continue

    raise np.linalg.LinAlgError("Cholesky failed even after adding jitter.")
def solve_via_cholesky_fast(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """stable version of algorithm 6 from thesis, that solves linear system"""
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)
def solve_via_cholesky_matrix_fast(L: np.ndarray, A: np.ndarray) -> np.ndarray:
    """stable version of algorithm 7 from thesis, that solves linear system with multiple right hand sides"""
    C = np.linalg.solve(L, A)
    return np.linalg.solve(L.T, C)
def gp_predict(
    inputs: np.ndarray,
    targets: np.ndarray,
    covariance_function,
    noise2: float,
    test_inputs: np.ndarray,
):
    """fast version of Algorithm 1"""
    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    test_inputs = np.asarray(test_inputs, dtype=float)

    K = covariancematrix(covariance_function, inputs) + noise2 * np.eye(len(inputs))
    L, _ = stable_cholesky_fast(K)
    alpha = solve_via_cholesky_fast(L, targets)

    K_star = crosscovariancematrix(inputs, test_inputs, covariance_function)
    mean = K_star.T @ alpha

    v = np.linalg.solve(L, K_star)
    K_star_star = covariancematrix(covariance_function, test_inputs)
    cov = K_star_star - v.T @ v

    log_ml = (
        -0.5 * float(targets.T @ alpha)
        - float(np.sum(np.log(np.diag(L))))
        - 0.5 * len(inputs) * np.log(2.0 * np.pi)
    )
    return mean, cov, float(log_ml)
#==================helper functions for hyperparamter optimization==================#
def precompute_sqdist(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    return (X - X.T) ** 2


def precompute_sin2_term(X: np.ndarray, period: float = 1.0) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    absdiff = np.abs(X - X.T)
    return np.sin(np.pi * absdiff / period) ** 2

#==================Algorithm 11==============================#
def backtracking_line_search_ascent(
    f,
    theta_k: np.ndarray,
    g_k: np.ndarray,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 20,
) -> float:
    alpha = float(alpha0)
    f_k = float(f(theta_k))
    grad_sq_norm = float(np.dot(g_k, g_k))
    if grad_sq_norm == 0.0:
        return 0.0

    for _ in range(max_steps):
        candidate = theta_k + alpha * g_k
        f_candidate = float(f(candidate))
        if f_candidate >= f_k + c * alpha * grad_sq_norm:
            return alpha
        alpha *= rho
    return alpha
