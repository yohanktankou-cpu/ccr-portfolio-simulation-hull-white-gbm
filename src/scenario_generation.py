from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .risk_factors import generate_correlated_normals, scenario_frame, simulate_gbm, simulate_hull_white_1f


@dataclass(frozen=True)
class ScenarioSet:
    """Container for simulated risk-factor scenarios."""

    time_grid: np.ndarray
    eur_rate: np.ndarray
    usd_rate: np.ndarray
    eur_usd_fx: np.ndarray
    correlation_matrix: np.ndarray
    labels: list[str]

    def to_frame(self) -> pd.DataFrame:
        return scenario_frame(
            self.time_grid,
            {
                "eur_rate": self.eur_rate,
                "usd_rate": self.usd_rate,
                "eur_usd_fx": self.eur_usd_fx,
            },
        )


def generate_scenarios(config: Dict) -> ScenarioSet:
    """Generate correlated EUR rate, USD rate and EUR/USD FX scenarios."""
    simulation = config["simulation"]
    rf = config["risk_factors"]
    correlation = config["correlation_matrix"]

    n_paths = int(simulation["n_paths"])
    steps_per_year = int(simulation["steps_per_year"])
    end_year = float(simulation["end_year"])
    n_steps = int(end_year * steps_per_year)
    dt = 1.0 / steps_per_year
    time_grid = np.linspace(0.0, end_year, n_steps + 1)

    labels = correlation["labels"]
    corr_matrix = np.array(correlation["values"], dtype=float)
    shocks = generate_correlated_normals(
        n_paths=n_paths,
        n_steps=n_steps,
        correlation_matrix=corr_matrix,
        random_seed=int(simulation["random_seed"]),
    )

    eur = rf["eur_rate"]
    usd = rf["usd_rate"]
    fx = rf["eur_usd_fx"]

    eur_rate = simulate_hull_white_1f(
        r0=eur["r0"], a=eur["a"], theta=eur["theta"], sigma=eur["sigma"], shocks=shocks[:, :, 0], dt=dt
    )
    usd_rate = simulate_hull_white_1f(
        r0=usd["r0"], a=usd["a"], theta=usd["theta"], sigma=usd["sigma"], shocks=shocks[:, :, 1], dt=dt
    )
    eur_usd_fx = simulate_gbm(
        spot0=fx["spot0"], mu=fx["mu"], sigma=fx["sigma"], shocks=shocks[:, :, 2], dt=dt
    )

    return ScenarioSet(
        time_grid=time_grid,
        eur_rate=eur_rate,
        usd_rate=usd_rate,
        eur_usd_fx=eur_usd_fx,
        correlation_matrix=corr_matrix,
        labels=labels,
    )
