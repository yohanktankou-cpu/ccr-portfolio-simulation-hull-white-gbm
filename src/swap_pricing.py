from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterestRateSwap:
    """Simplified vanilla interest-rate swap representation."""

    trade_id: str
    currency: str
    notional: float
    fixed_rate: float
    maturity_years: float
    pay_receive_fixed: str  # "pay" or "receive"


def annuity_from_flat_rate(rate: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    """Approximate swap annuity under a flat rate curve.

    This is intentionally simplified for a public educational GitHub version.
    """
    safe_rate = np.maximum(rate, 1e-6)
    return (1.0 - np.exp(-safe_rate * maturity)) / safe_rate


def par_swap_rate_from_flat_rate(rate: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    """Approximate par swap rate from a flat continuously compounded curve."""
    annuity = annuity_from_flat_rate(rate, maturity)
    discount_factor = np.exp(-rate * maturity)
    return (1.0 - discount_factor) / np.maximum(annuity, 1e-12)


def price_swap_paths(swap: InterestRateSwap, rates: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """Compute simplified MtM paths for an interest-rate swap.

    Positive MtM means value for the counterparty receiving the computed trade value.
    """
    remaining_maturity = np.maximum(swap.maturity_years - time_grid, 0.0)
    annuity = annuity_from_flat_rate(rates, remaining_maturity)
    par_rate = par_swap_rate_from_flat_rate(rates, remaining_maturity)

    if swap.pay_receive_fixed.lower() == "receive":
        direction = 1.0
    elif swap.pay_receive_fixed.lower() == "pay":
        direction = -1.0
    else:
        raise ValueError("pay_receive_fixed must be either 'pay' or 'receive'.")

    mtm = direction * swap.notional * (swap.fixed_rate - par_rate) * annuity
    mtm[:, remaining_maturity <= 0.0] = 0.0
    return mtm
