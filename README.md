# CCR Portfolio Simulation with Hull-White and GBM Risk Factors

This repository provides a clean, synthetic and GitHub-ready implementation of a counterparty credit risk (CCR) portfolio simulation workflow.

The project simulates correlated EUR and USD interest-rate risk factors, an EUR/USD FX risk factor, values a small interest-rate swap portfolio under Monte Carlo scenarios, and computes exposure metrics such as Expected Exposure (EE), Potential Future Exposure (PFE) and collateral-adjusted exposure.

> Note: this public version uses synthetic parameters and self-contained code. It does not include proprietary course code, private datasets, group member identifiers, or internal framework dependencies.

## Objective

The goal is to demonstrate how a quantitative risk workflow can be structured from risk-factor simulation to portfolio exposure analysis:

1. Generate coherent market scenarios.
2. Simulate EUR and USD short-rate paths using a simplified Hull-White one-factor model.
3. Simulate EUR/USD FX paths using Geometric Brownian Motion.
4. Value interest-rate swaps under simulated scenarios.
5. Aggregate portfolio MtM at netting-set level.
6. Apply a simplified collateral rule.
7. Compute EE and PFE exposure profiles.
8. Produce plots and CSV outputs for analysis.

## Risk Factors

The simulation includes:

- EUR short-rate / zero-yield proxy
- USD short-rate / zero-yield proxy
- EUR/USD FX rate

Interest-rate factors are modelled using a simplified Hull-White one-factor / Ornstein-Uhlenbeck dynamic:

```text
dr_t = a(theta - r_t) dt + sigma dW_t
```

The FX rate is modelled using Geometric Brownian Motion:

```text
dS_t / S_t = mu dt + sigma dW_t
```

The Brownian shocks are correlated through a user-defined correlation matrix.

## Main Features

- Monthly scenario generation
- 10,000 Monte Carlo paths by default
- Correlated EUR rates, USD rates and EUR/USD FX simulation
- Simplified interest-rate swap valuation under scenarios
- Portfolio aggregation with two swaps
- Netting-set exposure calculation
- Simplified collateral threshold and minimum transfer amount logic
- EE and PFE exposure profile computation
- CSV exports and plots for GitHub presentation

## Quantitative Methods

- Monte Carlo simulation
- Hull-White one-factor style interest-rate dynamics
- Geometric Brownian Motion
- Correlated risk-factor modelling
- Interest-rate swap MtM approximation
- Netting and collateral adjustment
- Expected Exposure and Potential Future Exposure
- Sensitivity-friendly modular Python code

## Repository Structure

```text
ccr-portfolio-simulation-hull-white-gbm/
│
├── README.md
├── requirements.txt
├── run_project.py
├── .gitignore
│
├── data/
│   └── synthetic_parameters.json
│
├── notebooks/
│   └── ccr_portfolio_simulation.ipynb
│
├── outputs/
│   ├── README.md
│   └── generated after running the project
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── risk_factors.py
    ├── scenario_generation.py
    ├── swap_pricing.py
    ├── portfolio_valuation.py
    └── exposure_analysis.py
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python run_project.py
```

The script will generate:

- `outputs/scenario_paths_sample.csv`
- `outputs/trade_mtms_sample.csv`
- `outputs/portfolio_exposures.csv`
- `outputs/correlation_matrix.png`
- `outputs/risk_factor_paths.png`
- `outputs/exposure_profile.png`

## Skills Demonstrated

- Python for quantitative finance
- Monte Carlo simulation
- Derivatives valuation
- Counterparty credit risk analytics
- Portfolio aggregation
- Exposure modelling
- Data handling with pandas
- Financial visualisation with matplotlib
- Clean project structuring for GitHub

## Relevance

This project is relevant for roles in:

- Quantitative Risk Modelling
- Counterparty Credit Risk
- XVA / CCR
- Model Risk
- Market Risk
- Derivatives Analytics
- Banking Risk Modelling
- Stress Testing

## CV Bullet Example

**CCR Portfolio Simulation with Hull-White and GBM Risk Factors**  
Built a Python-based Monte Carlo framework for correlated EUR/USD interest-rate and FX scenarios; valued a swap portfolio under simulated paths, aggregated exposures at netting-set level, applied simplified collateral logic and computed EE/PFE profiles for counterparty credit risk analysis.
