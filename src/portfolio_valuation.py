from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .scenario_generation import ScenarioSet
from .swap_pricing import InterestRateSwap, price_swap_paths


def build_portfolio(config: Dict) -> list[InterestRateSwap]:
    """Build the synthetic portfolio from configuration."""
    trades = []
    for trade in config["portfolio"]["trades"]:
        trades.append(
            InterestRateSwap(
                trade_id=trade["trade_id"],
                currency=trade["currency"],
                notional=float(trade["notional"]),
                fixed_rate=float(trade["fixed_rate"]),
                maturity_years=float(trade["maturity_years"]),
                pay_receive_fixed=trade["pay_receive_fixed"],
            )
        )
    return trades


def value_trades(trades: Iterable[InterestRateSwap], scenarios: ScenarioSet) -> Dict[str, np.ndarray]:
    """Value all swaps under the simulated scenarios."""
    results = {}
    for trade in trades:
        if trade.currency.upper() == "EUR":
            mtm = price_swap_paths(trade, scenarios.eur_rate, scenarios.time_grid)
        elif trade.currency.upper() == "USD":
            usd_mtm = price_swap_paths(trade, scenarios.usd_rate, scenarios.time_grid)
            mtm = usd_mtm / scenarios.eur_usd_fx  # convert USD value to EUR using EUR/USD
        else:
            raise ValueError(f"Unsupported currency: {trade.currency}")
        results[trade.trade_id] = mtm
    return results


def aggregate_netting_set(trade_mtms: Dict[str, np.ndarray]) -> np.ndarray:
    """Aggregate trade MtMs at netting-set level."""
    return np.sum(np.stack(list(trade_mtms.values()), axis=0), axis=0)


def apply_simple_collateral(
    netting_set_mtm: np.ndarray,
    threshold: float = 0.0,
    minimum_transfer_amount: float = 1.0,
) -> np.ndarray:
    """Apply a simplified collateral rule to positive MtM only.

    Collateral posted is max(MtM - threshold, 0) if it exceeds the minimum transfer amount.
    Exposure after collateral is MtM minus collateral, floored at zero.
    """
    positive_mtm = np.maximum(netting_set_mtm, 0.0)
    collateral = np.maximum(positive_mtm - threshold, 0.0)
    collateral = np.where(collateral >= minimum_transfer_amount, collateral, 0.0)
    return np.maximum(positive_mtm - collateral, 0.0)


def trade_mtms_to_frame(time_grid: np.ndarray, trade_mtms: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert trade MtM arrays into a long DataFrame."""
    frames = []
    for trade_id, mtm in trade_mtms.items():
        frame = pd.DataFrame(mtm, columns=time_grid)
        frame["path_id"] = np.arange(mtm.shape[0])
        frame = frame.melt(id_vars="path_id", var_name="time", value_name="mtm_eur")
        frame["trade_id"] = trade_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
