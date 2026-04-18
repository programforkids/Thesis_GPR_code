import numpy as np

from gp_algorithms import (
    backtracking_line_search_ascent,
    precompute_sqdist,
    solve_via_cholesky_fast,
    solve_via_cholesky_matrix_fast,
    stable_cholesky_fast,
)

LOG_SIGMA2_MIN = np.log(1e-6)
LOG_SIGMA2_MAX = np.log(1e6)
LOG_ELL_MIN = np.log(1e-3)
LOG_ELL_MAX = np.log(1e3)


def unpack_log_theta(theta_log: np.ndarray) -> tuple[float, float]:
    theta_log = np.asarray(theta_log, dtype=float)
    return float(np.exp(theta_log[0])), float(np.exp(theta_log[1]))


def clip_theta(theta_log: np.ndarray) -> np.ndarray:
    theta_log = np.asarray(theta_log, dtype=float).copy()
    theta_log[0] = np.clip(theta_log[0], LOG_SIGMA2_MIN, LOG_SIGMA2_MAX)
    theta_log[1] = np.clip(theta_log[1], LOG_ELL_MIN, LOG_ELL_MAX)
    return theta_log


def logml_and_grad_se_fixed_noise_precomputed(
    Y: np.ndarray,
    sqdist: np.ndarray,
    I: np.ndarray,
    theta_log: np.ndarray,
    noise2_fixed: float,
) -> tuple[float, np.ndarray]:
    theta_log = clip_theta(theta_log)
    sigma2, ell = unpack_log_theta(theta_log)
    K = sigma2 * np.exp(-sqdist / (2.0 * ell ** 2))
    K_theta = K + noise2_fixed * I

    L, _ = stable_cholesky_fast(K_theta)
    alpha = solve_via_cholesky_fast(L, Y)
    logml = float(-0.5 * (Y.T @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2.0 * np.pi))

    grad_mats = [K, K * (sqdist / (ell ** 2))]
    grad = np.zeros(2, dtype=float)
    for j, A_j in enumerate(grad_mats):
        beta_j = solve_via_cholesky_matrix_fast(L, A_j)
        grad[j] = 0.5 * float(alpha.T @ A_j @ alpha) - 0.5 * float(np.trace(beta_j))
    return logml, grad


def maximize_log_marginal_likelihood_se_fixed_noise(
    X: np.ndarray,
    Y: np.ndarray,
    theta0_log: np.ndarray,
    noise2_fixed: float,
    tol: float = 1e-5,
    max_iter: int = 100,
    alpha0: float = 0.1,
    c: float = 1e-4,
    rho: float = 0.5,
    verbose: bool = False,
):
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    sqdist = precompute_sqdist(X)
    I = np.eye(len(X))
    theta = clip_theta(theta0_log)

    def objective(th: np.ndarray) -> float:
        try:
            val, _ = logml_and_grad_se_fixed_noise_precomputed(Y, sqdist, I, th, noise2_fixed)
            return val if np.isfinite(val) else -np.inf
        except Exception:
            return -np.inf

    history = []
    grad_norms = []
    step_sizes = []
    for k in range(max_iter):
        logml, grad = logml_and_grad_se_fixed_noise_precomputed(Y, sqdist, I, theta, noise2_fixed)
        grad_norm = float(np.linalg.norm(grad))
        history.append(logml)
        grad_norms.append(grad_norm)
        if verbose:
            sigma2, ell = unpack_log_theta(theta)
            print(f"iter={k:3d} logML={logml: .8f} ||g||={grad_norm: .3e} sigma2={sigma2:.5f} ell={ell:.5f}")
        if grad_norm <= tol:
            break
        alpha = backtracking_line_search_ascent(objective, theta, grad, alpha0=alpha0, c=c, rho=rho)
        step_sizes.append(alpha)
        theta = clip_theta(theta + alpha * grad)
    return theta, history, grad_norms, step_sizes
