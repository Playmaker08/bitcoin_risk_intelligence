# ₿ Bitcoin Risk Intelligence Dashboard

A quantitative **risk-monitoring dashboard** for Bitcoin built with **Python, Pandas, Plotly, and Streamlit**.  
This project transforms historical Bitcoin market data into a structured **risk intelligence system** that tracks volatility, downside risk, tail-loss severity, and market risk regimes.

The system is designed to mimic the type of monitoring workflow used in **risk management, quantitative research, and fintech analytics**.

---

## Project Objective

Bitcoin is one of the most volatile and regime-dependent financial assets in the world.  
Traditional price charts alone do not fully capture the market’s **tail risk, downside exposure, or changing stress conditions**.

This project was built to answer the following questions:

- How volatile is Bitcoin over time?
- How severe are downside losses under normal and stressed conditions?
- How often do actual losses exceed modeled risk thresholds?
- Can Bitcoin market conditions be classified into interpretable **risk regimes**?

To answer these questions, the project builds a complete analytics pipeline covering:

1. data engineering  
2. market behavior analysis  
3. volatility modeling  
4. Value-at-Risk estimation  
5. Expected Shortfall analysis  
6. risk regime classification  
7. interactive dashboard deployment  

---

## End Product

The final product is an interactive **Bitcoin Risk Intelligence Dashboard** that provides:

- executive risk summary
- rolling volatility monitoring
- Value-at-Risk (VaR) tracking
- Expected Shortfall (ES) tail-risk analysis
- market regime classification
- recent stress-event logging

The dashboard is intended to function like a lightweight **crypto risk monitoring platform**.

---

## Methodology Overview

The project was developed in six analytical stages before deployment.

### 1. Data Engineering

Historical Bitcoin price data was cleaned, transformed, and resampled into daily observations suitable for risk modeling and dashboard deployment.

Core tasks included:

- timestamp conversion
- resampling minute-level data to daily frequency
- handling missing values
- creating daily return series
- preparing a deployment-friendly dataset

This step ensured the data pipeline was stable, lightweight, and reproducible.

---

### 2. Market Behavior Analysis

The second stage explored the statistical properties of Bitcoin returns.

Key findings:

- Bitcoin returns are **non-Gaussian**
- the return distribution exhibits **heavy tails**
- volatility clusters through time
- drawdowns are deep and persistent
- squared returns show evidence of **volatility persistence**

These patterns suggest that Bitcoin risk is **time-varying**, not constant, and motivate the need for dynamic risk metrics.

---

### 3. Volatility Modeling

Volatility dynamics were modeled using **GARCH-family models** to understand conditional risk behavior.

Models compared:

- GARCH
- GJR-GARCH
- EGARCH

Key insight:

- Bitcoin volatility is **highly persistent**
- asymmetric models added limited incremental forecasting value relative to the baseline GARCH structure
- volatility spikes tend to cluster during stress periods

This stage confirmed that Bitcoin risk should be modeled as a **conditional and regime-dependent process**.

---

### 4. Value at Risk (VaR)

The project then estimated **Value at Risk** using rolling historical methods.

Metrics implemented:

- 5% VaR
- 1% VaR

VaR interpretation:

- 5% VaR estimates a threshold such that only 5% of returns are expected to fall below it
- 1% VaR focuses on more extreme downside scenarios

Backtesting showed that:

- the 5% parametric-style risk threshold was the most practical calibration benchmark
- exceedances occur in clusters rather than as isolated events
- downside risk in Bitcoin is substantial even outside crisis periods

This stage converted the project from pure market analysis into a true **risk engine**.

---

### 5. Expected Shortfall (ES)

To go beyond threshold-based risk, the project estimated **Expected Shortfall**, also known as **Conditional VaR**.

Expected Shortfall answers a more severe question:

> If returns fall beyond VaR, how bad is the average loss?

Metrics implemented:

- 5% ES
- 1% ES

Key findings:

- Expected Shortfall is consistently more negative than VaR
- Bitcoin tail losses are materially deeper than threshold metrics alone imply
- empirical tail-risk estimates are generally more conservative
- the 1% tail reveals particularly severe downside exposure

This step strengthened the project significantly by measuring **tail-loss severity**, not just tail-loss frequency.

---

### 6. Risk Regime Classification

The final analytical layer was a rule-based **risk regime model**.

A composite risk score was built using:

- rolling realized volatility
- historical VaR
- historical Expected Shortfall

The model classifies market states into:

- Low Risk
- Moderate Risk
- High Risk
- Extreme Risk

This creates a more interpretable framework for understanding whether the market is currently calm, elevated, stressed, or extreme.

Key insight:

- Bitcoin spends most of its history in **low-to-moderate risk states**
- extreme-risk states are relatively rare
- however, extreme states are highly informative because they align with periods of concentrated market stress

This regime layer transforms the project from a collection of metrics into a true **risk intelligence system**.

---

## Core Risk Metrics

### Rolling Volatility

The dashboard monitors short- and medium-term realized volatility using:

- 30-day rolling volatility
- 60-day rolling volatility

These metrics provide a direct view of changing market instability.

---

### Value at Risk (VaR)

VaR is estimated using rolling historical simulation.

Implemented levels:

- 5% VaR
- 1% VaR

Interpretation:

- 5% VaR estimates a typical downside threshold under moderately adverse conditions
- 1% VaR estimates a more extreme loss threshold

---

### Expected Shortfall (ES)

Expected Shortfall captures the **average loss beyond VaR**.

Implemented levels:

- 5% ES
- 1% ES

Why this matters:

- VaR tells you where the tail begins
- ES tells you how deep the tail becomes once you are already inside it

---

### Risk Regime Score

A composite risk score is computed using percentile-ranked versions of:

- realized volatility
- VaR
- ES

Weighting scheme:

- 40% volatility percentile
- 30% VaR percentile
- 30% ES percentile

This score is then mapped into discrete market states.

---

## Major Insights

Across the full project pipeline, several important conclusions emerge.

### 1. Bitcoin risk is regime-based

Bitcoin does not behave like an asset with stable variance.  
Instead, it transitions between calmer periods and concentrated stress periods.

---

### 2. Bitcoin returns are heavy-tailed

The return distribution exhibits fat tails, meaning extreme gains and losses occur more frequently than under a normal distribution assumption.

---

### 3. Volatility is highly persistent

Once volatility rises, it tends to stay elevated for a period of time rather than reverting immediately.

---

### 4. Tail losses are much deeper than VaR alone suggests

Expected Shortfall shows that average realized losses in extreme scenarios are materially worse than the threshold indicated by VaR.

---

### 5. Extreme-risk states are rare but meaningful

The market spends relatively little time in extreme-risk regimes, but those periods contain the most important downside information.

---

### 6. Risk metrics are more informative than price alone

High prices do not necessarily imply high risk, and falling prices do not always imply maximum stress.  
The dashboard focuses on **risk state**, not just price level.

---

## Dashboard Features

The deployed dashboard includes:

- executive summary of current market risk
- current Bitcoin price
- 30-day volatility
- selected VaR and ES levels
- current risk regime and composite risk score
- market overview section
- tail-risk monitoring section
- regime classification section
- regime summary table
- recent stress-event log
- methodology notes

This makes the project suitable for demonstrating both **financial reasoning** and **product-oriented analytics design**.

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Plotly**
- **Streamlit**

---

## Repository Structure

```text
bitcoin_risk_intelligence/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── btc_daily.csv
│
├── notebooks/
│   ├── 01_data_engineering.ipynb
│   ├── 02_market_behavior.ipynb
│   ├── 03_volatility_modeling.ipynb
│   ├── 04_risk_modeling_var.ipynb
│   ├── 05_garch_expected_shortfall.ipynb
│   └── 06_regime_detection.ipynb
│
├── requirements.txt
└── README.md
