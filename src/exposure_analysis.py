from __future__ import annotations

import numpy as np
import pandas as pd


def expected_exposure(exposure_paths: np.ndarray) -> np.ndarray:
    """Expected Exposure (EE): average positive exposure at each time."""
    return np.mean(np.maximum(exposure_paths, 0.0), axis=0)


def potential_future_exposure(exposure_paths: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """Potential Future Exposure (PFE): exposure quantile at each time."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    return np.quantile(np.maximum(exposure_paths, 0.0), quantile, axis=0)


def exposure_summary_frame(
    time_grid: np.ndarray,
    netting_set_mtm: np.ndarray,
    collateral_adjusted_exposure: np.ndarray,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Build a summary DataFrame with MtM and exposure metrics."""
    raw_positive_exposure = np.maximum(netting_set_mtm, 0.0)
    return pd.DataFrame(
        {
            "time": time_grid,
            "average_netting_set_mtm": np.mean(netting_set_mtm, axis=0),
            "ee_raw": expected_exposure(raw_positive_exposure),
            f"pfe_{int(quantile * 100)}_raw": potential_future_exposure(raw_positive_exposure, quantile),
            "ee_after_collateral": expected_exposure(collateral_adjusted_exposure),
            f"pfe_{int(quantile * 100)}_after_collateral": potential_future_exposure(
                collateral_adjusted_exposure, quantile
            ),
        }
    )
