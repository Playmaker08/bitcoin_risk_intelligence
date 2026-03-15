# ₿ Bitcoin Risk Intelligence Dashboard

A dynamic **risk-monitoring dashboard** for Bitcoin markets built using **Python, Streamlit, and quantitative risk analytics**.  
The system analyzes historical Bitcoin price data to estimate **volatility regimes, Value-at-Risk (VaR), Expected Shortfall (ES), and composite risk regimes**, allowing investors and analysts to monitor evolving tail risk conditions.

The dashboard provides an **interactive executive-level view of market risk**, combining traditional financial risk metrics with modern visualization tools.

---

# Project Overview

Bitcoin markets experience extreme volatility and fat-tailed return distributions.  
Traditional portfolio monitoring tools often fail to capture **tail-risk dynamics and regime shifts** in crypto markets.

This project builds a **risk intelligence system** that:

- Calculates **realized volatility**
- Estimates **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**
- Detects **risk regimes** based on percentile-ranked metrics
- Tracks **tail-risk exceedances**
- Provides a **real-time interactive dashboard**

The goal is to replicate the type of **risk monitoring system used in institutional trading desks or hedge funds**.

---

# Dashboard Preview

## Executive Risk Overview
Displays the current market state including price, volatility, VaR, ES, and the active risk regime.

<img src="images/dashboard_summary.png" width="900">

---

## Market Overview
Tracks price evolution and rolling volatility.

<img src="images/market_overview.png" width="900">

---

## Tail Risk Monitor
Compares daily returns against estimated **VaR and Expected Shortfall thresholds**.

<img src="images/tail_risk_monitor.png" width="900">

---

## Risk Regime Detection
Classifies historical market periods into **Low / Moderate / High / Extreme risk regimes**.

<img src="images/risk_regime.png" width="900">

---

# Key Risk Metrics

The dashboard computes several institutional-grade risk indicators.

### Realized Volatility
Rolling volatility is calculated using:
