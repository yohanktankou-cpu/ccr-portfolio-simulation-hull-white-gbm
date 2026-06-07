from __future__ import annotations

import numpy as np
import pandas as pd


def validate_correlation_matrix(correlation_matrix: np.ndarray) -> None:
    """Validate that a correlation matrix is square, symmetric and positive semi-definite."""
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-10):
        raise ValueError("Correlation matrix must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("Correlation matrix must be positive semi-definite.")


def generate_correlated_normals(
    n_paths: int,
    n_steps: int,
    correlation_matrix: np.ndarray,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate correlated standard-normal shocks.

    Returns an array with shape `(n_paths, n_steps, n_factors)`.
    """
    validate_correlation_matrix(correlation_matrix)
    rng = np.random.default_rng(random_seed)
    n_factors = correlation_matrix.shape[0]
    independent = rng.standard_normal(size=(n_paths, n_steps, n_factors))
    cholesky = np.linalg.cholesky(correlation_matrix)
    return independent @ cholesky.T


def simulate_hull_white_1f(
    r0: float,
    a: float,
    theta: float,
    sigma: float,
    shocks: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Simulate a simplified Hull-White one-factor / OU short-rate process.

    Dynamic used in this public educational implementation:
        dr_t = a(theta - r_t)dt + sigma dW_t
    """
    n_paths, n_steps = shocks.shape
    rates = np.empty((n_paths, n_steps + 1))
    rates[:, 0] = r0

    for step in range(n_steps):
        rates[:, step + 1] = (
            rates[:, step]
            + a * (theta - rates[:, step]) * dt
            + sigma * np.sqrt(dt) * shocks[:, step]
        )
    return rates


def simulate_gbm(
    spot0: float,
    mu: float,
    sigma: float,
    shocks: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Simulate a Geometric Brownian Motion process."""
    n_paths, n_steps = shocks.shape
    spots = np.empty((n_paths, n_steps + 1))
    spots[:, 0] = spot0

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion_scale = sigma * np.sqrt(dt)
    for step in range(n_steps):
        spots[:, step + 1] = spots[:, step] * np.exp(drift + diffusion_scale * shocks[:, step])
    return spots


def scenario_frame(time_grid: np.ndarray, scenarios: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert simulated scenarios into a long pandas DataFrame."""
    frames = []
    for name, paths in scenarios.items():
        frame = pd.DataFrame(paths, columns=time_grid)
        frame["path_id"] = np.arange(paths.shape[0])
        frame = frame.melt(id_vars="path_id", var_name="time", value_name="value")
        frame["risk_factor"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
