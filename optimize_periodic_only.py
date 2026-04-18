import numpy as np

from gp_algorithms import (
    backtracking_line_search_ascent,
    precompute_sin2_term,
    precompute_sqdist,
    solve_via_cholesky_fast,
    solve_via_cholesky_matrix_fast,
    stable_cholesky_fast,
)

FIXED_SIGMA2_SE = 4322.987322334175
FIXED_ELL_SE = 36.72171458829093
FIXED_PERIOD = 1.0

LOG_SIGMA2_PER_MIN = np.log(1e-6)
LOG_SIGMA2_PER_MAX = np.log(1e4)
LOG_ELL_PER_MIN = np.log(1e-3)
LOG_ELL_PER_MAX = np.log(10.0)
LOG_NOISE2_MIN = np.log(1e-6)
LOG_NOISE2_MAX = np.log(1e2)


def unpack_log_theta_periodic_only(theta_log: np.ndarray) -> tuple[float, float, float]:
    theta_log = np.asarray(theta_log, dtype=float)
    return (
        float(np.exp(theta_log[0])),
        float(np.exp(theta_log[1])),
        float(np.exp(theta_log[2])),
    )


def clip_theta_periodic_only(theta_log: np.ndarray) -> np.ndarray:
    theta_log = np.asarray(theta_log, dtype=float).copy()
    theta_log[0] = np.clip(theta_log[0], LOG_SIGMA2_PER_MIN, LOG_SIGMA2_PER_MAX)
    theta_log[1] = np.clip(theta_log[1], LOG_ELL_PER_MIN, LOG_ELL_PER_MAX)
    theta_log[2] = np.clip(theta_log[2], LOG_NOISE2_MIN, LOG_NOISE2_MAX)
    return theta_log


def prepare_periodic_only_quantities(X: np.ndarray):
    X = np.asarray(X, dtype=float).reshape(-1)
    sqdist = precompute_sqdist(X)
    sin2_term = precompute_sin2_term(X, period=FIXED_PERIOD)
    K_se_fixed = FIXED_SIGMA2_SE * np.exp(-sqdist / (2.0 * FIXED_ELL_SE ** 2))
    I = np.eye(len(X))
    return sqdist, sin2_term, K_se_fixed, I


def logml_and_grad_periodic_only_precomputed(
    Y: np.ndarray,
    sin2_term: np.ndarray,
    K_se_fixed: np.ndarray,
    I: np.ndarray,
    theta_log: np.ndarray,
) -> tuple[float, np.ndarray]:
    theta_log = clip_theta_periodic_only(theta_log)
    sigma2_periodic, ell_periodic, noise2 = unpack_log_theta_periodic_only(theta_log)
    K_per = sigma2_periodic * np.exp(-2.0 * sin2_term / (ell_periodic ** 2))
    K_theta = K_se_fixed + K_per + noise2 * I

    L, _ = stable_cholesky_fast(K_theta)
    alpha = solve_via_cholesky_fast(L, Y)
    logml = float(-0.5 * (Y.T @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2.0 * np.pi))

    grad_mats = [
        K_per,
        K_per * (4.0 * sin2_term / (ell_periodic ** 2)),
        noise2 * I,
    ]
    grad = np.zeros(3, dtype=float)
    for j, A_j in enumerate(grad_mats):
        beta_j = solve_via_cholesky_matrix_fast(L, A_j)
        grad[j] = 0.5 * float(alpha.T @ A_j @ alpha) - 0.5 * float(np.trace(beta_j))
    return logml, grad


def maximize_log_marginal_likelihood_periodic_only(
    X: np.ndarray,
    Y: np.ndarray,
    theta0_log: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 100,
    alpha0: float = 0.01,
    c: float = 1e-4,
    rho: float = 0.5,
    verbose: bool = False,
):
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    _, sin2_term, K_se_fixed, I = prepare_periodic_only_quantities(X)
    theta = clip_theta_periodic_only(theta0_log)

    def objective(th: np.ndarray) -> float:
        try:
            val, _ = logml_and_grad_periodic_only_precomputed(Y, sin2_term, K_se_fixed, I, th)
            return val if np.isfinite(val) else -np.inf
        except Exception:
            return -np.inf

    history, grad_norms, step_sizes = [], [], []
    for k in range(max_iter):
        logml, grad = logml_and_grad_periodic_only_precomputed(Y, sin2_term, K_se_fixed, I, theta)
        grad_norm = float(np.linalg.norm(grad))
        history.append(logml)
        grad_norms.append(grad_norm)
        if verbose:
            sigma2_per, ell_per, noise2 = unpack_log_theta_periodic_only(theta)
            print(
                f"iter={k:3d} logML={logml: .8f} ||g||={grad_norm: .3e} "
                f"sigma2_per={sigma2_per:.5f} ell_per={ell_per:.5f} noise2={noise2:.5f}"
            )
        if grad_norm <= tol:
            break
        alpha = backtracking_line_search_ascent(objective, theta, grad, alpha0=alpha0, c=c, rho=rho)
        step_sizes.append(alpha)
        theta = clip_theta_periodic_only(theta + alpha * grad)
    return theta, history, grad_norms, step_sizes


def log_marginal_likelihood_periodic_from_precomputed(
    Y: np.ndarray,
    sqdist: np.ndarray,
    sin2_term: np.ndarray,
    log_sigma2_periodic: float,
    log_ell_periodic: float,
    sigma2_se_fixed: float,
    ell_se_fixed: float,
    noise2_fixed: float,
) -> float:
    theta = clip_theta_periodic_only(np.array([log_sigma2_periodic, log_ell_periodic, np.log(noise2_fixed)], dtype=float))
    sigma2_periodic, ell_periodic, noise2 = unpack_log_theta_periodic_only(theta)
    K_se_fixed = sigma2_se_fixed * np.exp(-sqdist / (2.0 * ell_se_fixed ** 2))
    K_periodic = sigma2_periodic * np.exp(-2.0 * sin2_term / (ell_periodic ** 2))
    K_theta = K_se_fixed + K_periodic + noise2 * np.eye(len(Y))
    try:
        L, _ = stable_cholesky_fast(K_theta)
        alpha = solve_via_cholesky_fast(L, Y)
        return float(-0.5 * (Y.T @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        return -np.inf
