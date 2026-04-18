import numpy as np
#==================linear algebra building blocks==================#
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

#===================helper functions for GPR==================#
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
#====================GPR========================#
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
#====================hyperparamater optimization=======================#
def log_marginal_likelihood_and_gradient(
    inputs,
    targets,
    covariance_function_from_log_parameters,
    derivatives_wrt_log_parameters,
    log_hyperparameters,
):
    """Implements Algorithm 9 from thesis"""
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets, dtype=float)

    covariancefunction, noiselevel = covariance_function_from_log_parameters(log_hyperparameters)
    K = covariancematrix(covariancefunction, inputs)
    K += noiselevel * np.eye(len(inputs))

    L= cholesky_decomposition(K)
    alpha = solve_cholesky(L, targets)

    log_marginal_likelihood = (
        -0.5 * float(targets.T @ alpha)
        - sum_log_diag(L)
        - (len(inputs) / 2.0) * np.log(2 * np.pi)
    )

    K_inverse = solve_cholesky_matrix(L, np.eye(len(inputs)))
    B = np.outer(alpha, alpha) - K_inverse

    derivatives = derivatives_wrt_log_parameters(inputs, log_hyperparameters)
    gradient = np.zeros(len(log_hyperparameters))

    for j in range(len(log_hyperparameters)):
        gradient[j] = 0.5 * np.sum(B * derivatives[j])

    return log_marginal_likelihood, gradient

def backtracking_line_search_ascent(
    f,
    theta_k: np.ndarray,
    g_k: np.ndarray,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 20,
) -> float: 
    """Implements algorithm 11 from the thesis"""
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

def maximize_log_marginal_likelihood(
    inputs,
    targets,
    covariance_function_from_log_parameters,
    derivatives_wrt_log_parameters,
    theta0_log,
    tol=1e-5,
    max_iter=100,
    alpha0=1.0,
    c=1e-4,
    rho=0.5,
    verbose=False,

):
    """Implements Algorithm 12 from thesis"""
    inputs = np.asarray(inputs, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    theta = np.asarray(theta0_log, dtype=float).copy()

    def objective(th):
        try:
            logml, _ = log_marginal_likelihood_and_gradient(
                inputs=inputs,
                targets=targets,
                covariance_function_from_log_parameters=covariance_function_from_log_parameters,
                derivatives_wrt_log_parameters=derivatives_wrt_log_parameters,
                log_hyperparameters=th,
            )
            return logml if np.isfinite(logml) else -np.inf
        except Exception:
            return -np.inf

    history = []
    grad_norms = []
    step_sizes = []

    for k in range(max_iter):
        logml, grad = log_marginal_likelihood_and_gradient(
            inputs=inputs,
            targets=targets,
            covariance_function_from_log_parameters=covariance_function_from_log_parameters,
            derivatives_wrt_log_parameters=derivatives_wrt_log_parameters,
            log_hyperparameters=theta,
        )

        grad_norm = float(np.linalg.norm(grad))
        history.append(logml)
        grad_norms.append(grad_norm)

        if verbose:
            print(f"iter={k:3d} logML={logml: .8f} ||g||={grad_norm: .3e} theta={theta}")

        if grad_norm <= tol:
            break

        alpha = backtracking_line_search_ascent(
            objective,
            theta,
            grad,
            alpha0=alpha0,
            c=c,
            rho=rho,
        )
        step_sizes.append(alpha)
        theta = theta + alpha * grad
    return theta, history[-1], grad_norms[-1]      
