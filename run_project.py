from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.exposure_analysis import exposure_summary_frame
from src.portfolio_valuation import (
    aggregate_netting_set,
    apply_simple_collateral,
    build_portfolio,
    trade_mtms_to_frame,
    value_trades,
)
from src.scenario_generation import generate_scenarios


def ensure_output_dir() -> Path:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_correlation_matrix(correlation_matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(correlation_matrix, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            ax.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Risk-Factor Correlation Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sample_paths(time_grid: np.ndarray, scenarios, output_path: Path, n_sample_paths: int = 50) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    scenario_arrays = [scenarios.eur_rate, scenarios.usd_rate, scenarios.eur_usd_fx]
    titles = ["EUR Rate Paths", "USD Rate Paths", "EUR/USD FX Paths"]
    ylabels = ["Rate", "Rate", "Spot"]

    for ax, values, title, ylabel in zip(axes, scenario_arrays, titles, ylabels):
        ax.plot(time_grid, values[:n_sample_paths].T, alpha=0.35, linewidth=0.8)
        ax.plot(time_grid, values.mean(axis=0), linewidth=2.0, label="Mean")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time in years")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_exposure_profile(exposure_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(exposure_frame["time"], exposure_frame["ee_raw"], label="EE raw")
    ax.plot(exposure_frame["time"], exposure_frame["pfe_95_raw"], label="PFE 95% raw")
    ax.plot(exposure_frame["time"], exposure_frame["ee_after_collateral"], label="EE after collateral")
    ax.plot(exposure_frame["time"], exposure_frame["pfe_95_after_collateral"], label="PFE 95% after collateral")
    ax.set_title("Portfolio Exposure Profile")
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Exposure in EUR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_output_dir()
    config = load_config()

    scenarios = generate_scenarios(config)
    trades = build_portfolio(config)
    trade_mtms = value_trades(trades, scenarios)
    netting_set_mtm = aggregate_netting_set(trade_mtms)

    portfolio_config = config["portfolio"]
    exposure_after_collateral = apply_simple_collateral(
        netting_set_mtm,
        threshold=float(portfolio_config["collateral_threshold"]),
        minimum_transfer_amount=float(portfolio_config["minimum_transfer_amount"]),
    )

    # Keep GitHub outputs lightweight: save only a sample of path-level data.
    # Full arrays are used for exposure metrics, but writing every simulated path to CSV
    # would create very large files.
    max_paths_to_export = 200
    sampled_scenarios = type(scenarios)(
        time_grid=scenarios.time_grid,
        eur_rate=scenarios.eur_rate[:max_paths_to_export],
        usd_rate=scenarios.usd_rate[:max_paths_to_export],
        eur_usd_fx=scenarios.eur_usd_fx[:max_paths_to_export],
        correlation_matrix=scenarios.correlation_matrix,
        labels=scenarios.labels,
    )
    sampled_trade_mtms = {key: value[:max_paths_to_export] for key, value in trade_mtms.items()}

    scenario_df = sampled_scenarios.to_frame()
    trade_mtm_df = trade_mtms_to_frame(scenarios.time_grid, sampled_trade_mtms)
    exposure_df = exposure_summary_frame(scenarios.time_grid, netting_set_mtm, exposure_after_collateral, quantile=0.95)

    scenario_df.to_csv(output_dir / "scenario_paths_sample.csv", index=False)
    trade_mtm_df.to_csv(output_dir / "trade_mtms_sample.csv", index=False)
    exposure_df.to_csv(output_dir / "portfolio_exposures.csv", index=False)

    plot_correlation_matrix(scenarios.correlation_matrix, scenarios.labels, output_dir / "correlation_matrix.png")
    plot_sample_paths(scenarios.time_grid, scenarios, output_dir / "risk_factor_paths.png")
    plot_exposure_profile(exposure_df, output_dir / "exposure_profile.png")

    print("Project run completed.")
    print(f"Generated files in: {output_dir.resolve()}")
    print(exposure_df.head())


if __name__ == "__main__":
    main()
