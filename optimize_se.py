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
LOG_NOISE2_MIN = np.log(1e-6)
LOG_NOISE2_MAX = np.log(1e2)


def unpack_log_theta_se(theta_log: np.ndarray) -> tuple[float, float, float]:
    theta_log = np.asarray(theta_log, dtype=float)
    return float(np.exp(theta_log[0])), float(np.exp(theta_log[1])), float(np.exp(theta_log[2]))


def clip_theta_se(theta_log: np.ndarray) -> np.ndarray:
    theta_log = np.asarray(theta_log, dtype=float).copy()
    theta_log[0] = np.clip(theta_log[0], LOG_SIGMA2_MIN, LOG_SIGMA2_MAX)
    theta_log[1] = np.clip(theta_log[1], LOG_ELL_MIN, LOG_ELL_MAX)
    theta_log[2] = np.clip(theta_log[2], LOG_NOISE2_MIN, LOG_NOISE2_MAX)
    return theta_log


def prepare_se_quantities(X: np.ndarray):
    X = np.asarray(X, dtype=float).reshape(-1)
    sqdist = precompute_sqdist(X)
    I = np.eye(len(X))
    return sqdist, I


def logml_and_grad_se_precomputed(
    Y: np.ndarray,
    sqdist: np.ndarray,
    I: np.ndarray,
    theta_log: np.ndarray,
) -> tuple[float, np.ndarray]:
    theta_log = clip_theta_se(theta_log)
    sigma2_se, ell_se, noise2 = unpack_log_theta_se(theta_log)

    K_se = sigma2_se * np.exp(-sqdist / (2.0 * ell_se ** 2))
    K_theta = K_se + noise2 * I

    L, _ = stable_cholesky_fast(K_theta)
    alpha = solve_via_cholesky_fast(L, Y)

    logml = (
        -0.5 * float(Y.T @ alpha)
        - float(np.sum(np.log(np.diag(L))))
        - 0.5 * len(Y) * np.log(2.0 * np.pi)
    )

    grad_mats = [K_se, K_se * (sqdist / (ell_se ** 2)), noise2 * I]
    grad = np.zeros(3, dtype=float)
    for j, A_j in enumerate(grad_mats):
        beta_j = solve_via_cholesky_matrix_fast(L, A_j)
        grad[j] = 0.5 * float(alpha.T @ A_j @ alpha) - 0.5 * float(np.trace(beta_j))

    return float(logml), grad


def maximize_log_marginal_likelihood_se(
    X: np.ndarray,
    Y: np.ndarray,
    theta0_log: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 100,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    verbose: bool = False,
):
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    sqdist, I = prepare_se_quantities(X)
    theta = clip_theta_se(theta0_log)

    def objective(th: np.ndarray) -> float:
        try:
            val, _ = logml_and_grad_se_precomputed(Y, sqdist, I, th)
            return val if np.isfinite(val) else -np.inf
        except Exception:
            return -np.inf

    history, grad_norms, step_sizes = [], [], []

    for k in range(max_iter):
        logml, grad = logml_and_grad_se_precomputed(Y, sqdist, I, theta)
        grad_norm = float(np.linalg.norm(grad))
        history.append(logml)
        grad_norms.append(grad_norm)

        if verbose:
            sigma2, ell, noise2 = unpack_log_theta_se(theta)
            print(
                f"iter={k:3d} logML={logml: .8f} ||g||={grad_norm: .3e} "
                f"sigma2={sigma2:.5f} ell={ell:.5f} noise2={noise2:.5f}"
            )

        if grad_norm <= tol:
            break

        alpha = backtracking_line_search_ascent(objective, theta, grad, alpha0=alpha0, c=c, rho=rho)
        step_sizes.append(alpha)
        theta = clip_theta_se(theta + alpha * grad)

    return theta, history, grad_norms, step_sizes


def log_marginal_likelihood_se_from_sqdist(
    Y: np.ndarray,
    sqdist: np.ndarray,
    log_sigma2_se: float,
    log_ell_se: float,
    log_noise2: float,
) -> float:
    theta = clip_theta_se(np.array([log_sigma2_se, log_ell_se, log_noise2], dtype=float))
    sigma2_se, ell_se, noise2 = unpack_log_theta_se(theta)
    K = sigma2_se * np.exp(-sqdist / (2.0 * ell_se ** 2)) + noise2 * np.eye(len(Y))
    try:
        L, _ = stable_cholesky_fast(K)
        alpha = solve_via_cholesky_fast(L, Y)
        return float(-0.5 * (Y.T @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        return -np.inf
