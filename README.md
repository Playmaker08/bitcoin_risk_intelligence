# ₿ Bitcoin Risk Intelligence Dashboard

A live-updated quantitative **Bitcoin risk intelligence platform** built with **Python, Pandas, NumPy, Plotly, Streamlit, and CoinGecko market data**.

The project transforms historical and live Bitcoin price information into a structured financial risk-monitoring system that measures **volatility, downside exposure, tail-loss severity, market risk regimes, model calibration, and stress-event clustering**.

Rather than functioning as a conventional cryptocurrency price dashboard, the system is designed to mimic a lightweight analytical workflow used in **risk management, quantitative research, fintech analytics, and portfolio monitoring**.

---

## Project Objective

Bitcoin is one of the most volatile and regime-dependent financial assets in modern markets.

Traditional price charts provide useful market context, but price alone does not reveal:

- how unstable the return process currently is
- how severe potential downside losses may be
- whether tail-risk thresholds are being exceeded
- whether losses occur independently or cluster during stress periods
- whether the market is transitioning into a different risk environment
- whether a risk model remains statistically well calibrated over time

This project was built to answer the following questions:

- How volatile is Bitcoin over time?
- How severe are downside losses under normal and stressed conditions?
- How often do actual losses exceed modeled risk thresholds?
- Are VaR exceedances independent, or do they cluster during stress periods?
- Can Bitcoin market conditions be classified into interpretable risk regimes?
- Do higher current risk regimes correspond to higher future realized risk?
- Can historical and live market data be combined into a reliable monitoring workflow?

To answer these questions, the project builds an end-to-end analytics pipeline covering:

1. data engineering
2. market behavior analysis
3. volatility modeling
4. Value-at-Risk estimation
5. Expected Shortfall analysis
6. causal risk regime classification
7. live market-data ingestion
8. automated historical backfilling
9. statistical VaR validation
10. exceedance-clustering diagnostics
11. forward regime validation
12. interactive dashboard deployment

---

## End Product

The final product is a deployed **Bitcoin Risk Intelligence Dashboard** combining long-run historical market data with live Bitcoin observations.

The platform provides:

- live Bitcoin market snapshot
- automated historical-data freshness checks
- automatic backfilling of missing daily observations
- executive risk summary
- rolling volatility monitoring
- Value-at-Risk tracking
- Expected Shortfall tail-risk analysis
- causal market regime classification
- rule-based risk intelligence signals
- VaR breach monitoring
- statistical VaR backtesting
- exceedance-clustering diagnostics
- forward-looking regime validation
- recent stress-event logging

The dashboard therefore functions less like a static visualization project and more like a lightweight **crypto market-risk monitoring platform**.

---

## System Architecture

```text
Historical Bitcoin Data
        │
        ▼
Data Cleaning / Daily Resampling
        │
        ▼
Historical Data Freshness Check
        │
        ├── stale ──► CoinGecko Historical Backfill
        │
        ▼
Updated Daily Price Series
        │
        ├───────────────┐
        │               │
        │         CoinGecko Live API
        │               │
        └───────┬───────┘
                ▼
      Provisional Live Observation
                │
                ▼
         Return Engineering
                │
                ▼
        Quantitative Risk Engine
        │       │       │
        ▼       ▼       ▼
   Volatility   VaR     Expected Shortfall
        │       │       │
        └───────┼───────┘
                ▼
       Causal Risk Score
                │
                ▼
        Market Risk Regime
                │
                ▼
       Risk Intelligence Layer
                │
                ▼
       Model Validation Layer
                │
                ▼
        Streamlit Dashboard
```

---

# Methodology Overview

The project evolved through several analytical and engineering stages before deployment.

## 1. Data Engineering

The original dataset contained high-frequency Bitcoin market observations.

The data pipeline transformed this information into a deployment-ready daily time series.

Core tasks included:

- timestamp conversion
- chronological sorting
- minute-to-daily resampling
- missing-value handling
- daily return construction
- validation of date continuity
- preparation of a lightweight deployment dataset

The production dashboard uses a compact daily dataset rather than loading the full high-frequency source file directly.

This separation between research-scale raw data and deployment-scale analytical data improves:

- application speed
- reproducibility
- maintainability
- cloud deployment reliability

---

## 2. Market Behavior Analysis

The second stage examined the statistical behavior of Bitcoin returns.

Key findings included:

- returns are strongly **non-Gaussian**
- the return distribution exhibits **heavy tails**
- large positive and negative observations occur more frequently than a normal distribution would imply
- volatility clusters through time
- large drawdowns can persist
- squared returns display evidence of volatility persistence

These findings indicate that Bitcoin risk is neither constant nor well summarized by simple price movements.

Instead, risk behaves as a **time-varying process** with periods of calm interrupted by concentrated stress episodes.

---

## 3. Volatility Modeling

GARCH-family models were explored during the research stage to understand conditional volatility behavior.

Models examined included:

- GARCH
- GJR-GARCH
- EGARCH

The modeling exercise reinforced several important conclusions:

- Bitcoin volatility is highly persistent
- volatility shocks can remain relevant for extended periods
- conditional variance changes significantly through time
- asymmetric specifications offered useful context, although the baseline persistence result remained the dominant insight

The GARCH research helped motivate the broader project architecture even though the deployed dashboard ultimately emphasizes transparent rolling realized-risk measures.

---

## 4. Value at Risk

The production risk engine estimates rolling historical **Value at Risk (VaR)** at two levels:

- 5% VaR
- 1% VaR

VaR answers the question:

> What loss threshold should only be exceeded with a specified probability under the recent empirical return distribution?

The 5% measure represents a more common adverse scenario, while 1% VaR focuses on more extreme downside events.

The rolling methodology allows the threshold to evolve as the recent return distribution changes.

---

## 5. Expected Shortfall

Value at Risk identifies where the downside tail begins, but it does not describe the severity of losses once the threshold has been crossed.

For this reason, the project also estimates **Expected Shortfall (ES)**, sometimes referred to as Conditional VaR.

Expected Shortfall answers:

> If the market moves beyond the VaR threshold, what is the average loss in those tail observations?

Implemented levels:

- 5% Expected Shortfall
- 1% Expected Shortfall

Expected Shortfall is consistently more severe than VaR and provides a more complete representation of Bitcoin's downside tail.

Conceptually:

```text
VaR → where the downside tail begins

ES  → how severe losses become once inside the tail
```

---

## 6. Causal Risk Regime Classification

A composite risk score combines:

- rolling realized volatility
- historical VaR
- historical Expected Shortfall

The weighting scheme is:

- **40% volatility percentile**
- **30% VaR percentile**
- **30% Expected Shortfall percentile**

The score maps observations into:

- Low Risk
- Moderate Risk
- High Risk
- Extreme Risk

A key methodological improvement is that percentile ranks are **causal**.

Each historical observation is ranked only relative to information available up to that point in time.

This prevents future observations from influencing historical risk classifications and eliminates a source of **look-ahead bias**.

---

## 7. Live Market Data and Automatic Backfilling

The deployed platform integrates live Bitcoin market information through the **CoinGecko API**.

The live layer includes:

- Bitcoin price
- 24-hour percentage change
- 24-hour trading volume
- market capitalization
- API update timestamp

Before incorporating the live observation into risk calculations, the application checks whether the historical dataset is current.

If missing daily observations are detected, the application automatically retrieves the missing market history before calculating the current return.

This prevents a stale multi-day price difference from being incorrectly interpreted as a one-day return.

Once the historical series is current, the live Bitcoin price is inserted as the latest provisional daily observation and the risk engine is recalculated.

The resulting architecture is therefore:

```text
Historical Dataset
        ↓
Freshness Validation
        ↓
Historical Backfill
        ↓
Live Observation
        ↓
Return Calculation
        ↓
Risk Engine
```

---

## 8. Risk Intelligence Layer

The dashboard converts quantitative metrics into interpretable monitoring signals.

The system reports:

- **Risk Trend** — compares the current composite score with its recent level
- **Volatility State** — classifies the current volatility percentile
- **VaR Status** — identifies whether the latest return breached VaR
- **Tail-Risk Trend** — evaluates whether Expected Shortfall is improving or deteriorating

These signals create a bridge between raw quantitative output and practical risk interpretation.

The purpose is not to generate trading signals, but to summarize the current **risk environment**.

---

## 9. VaR Model Validation

VaR performance is evaluated through both descriptive and formal backtesting.

The validation framework includes:

- empirical breach frequency
- Kupiec unconditional coverage test
- Christoffersen independence test
- conditional coverage testing

The empirical breach frequencies are close to their nominal probability levels, indicating strong **unconditional calibration**.

However, formal testing reveals that exceedances are not fully independent through time.

This distinction is important because a VaR model can correctly estimate the long-run number of failures while still failing to adapt quickly enough during concentrated volatility shocks.

---

## 10. Exceedance-Clustering Diagnostics

To investigate this issue further, the project measures the spacing between VaR breaches.

Diagnostics include:

- average number of days between breaches
- median breach spacing
- minimum breach spacing
- number of breaches occurring within five days of another breach
- historical exceedance timeline

The analysis shows that VaR failures are meaningfully concentrated during periods of market stress rather than being uniformly distributed through time.

This makes the temporal structure of model failures visible rather than reducing validation to a single breach-rate statistic.

---

## 11. Forward Regime Validation

The regime framework is also validated using subsequent realized market behavior.

For each current risk regime, the system evaluates future seven-day realized volatility.

The objective is to determine whether:

```text
Low Risk
    <
Moderate Risk
    <
High Risk
    <
Extreme Risk
```

in terms of subsequent realized instability.

The observed results show **strong separation** between risk categories.

This supports the interpretation that the regime framework captures meaningful information about the near-term risk environment rather than simply describing contemporaneous conditions.

---

# Core Risk Metrics

## Rolling Volatility

The dashboard monitors:

- 30-day realized volatility
- 60-day realized volatility

These measures capture changes in short- and medium-term market instability.

---

## Value at Risk

Historical VaR is calculated using rolling return windows.

Implemented thresholds:

- 5% VaR
- 1% VaR

VaR identifies the estimated downside threshold associated with different levels of tail probability.

---

## Expected Shortfall

Expected Shortfall estimates the average realized loss conditional on returns falling below VaR.

Implemented levels:

- 5% ES
- 1% ES

Conceptually:

```text
VaR → where the tail begins

ES  → how deep the tail becomes
```

---

## Composite Risk Score

The market-risk score combines causal percentile rankings of:

- volatility
- VaR
- Expected Shortfall

Weights:

```text
40% Volatility
30% VaR
30% Expected Shortfall
```

The resulting score is translated into an interpretable market regime.

---

# Major Insights

Across the full project pipeline, several important conclusions emerge.

## 1. Bitcoin risk is regime-dependent

Bitcoin does not behave like an asset with stable variance.

Instead, market behavior transitions between calm environments and concentrated periods of elevated instability.

---

## 2. Bitcoin returns are heavy-tailed

Extreme positive and negative observations occur more frequently than a Gaussian framework would suggest.

This makes tail-risk modeling essential.

---

## 3. Volatility is highly persistent

Once volatility increases, elevated risk conditions often persist rather than immediately mean-reverting.

This creates clustering in both realized volatility and tail-loss events.

---

## 4. Tail losses are significantly deeper than VaR alone suggests

Expected Shortfall demonstrates that the average loss after entering the tail can be substantially worse than the VaR threshold itself.

---

## 5. Extreme-risk states are rare but economically meaningful

Bitcoin spends most of its time in lower-risk regimes.

However, high and extreme regimes contain disproportionate levels of realized volatility and downside tail severity.

---

## 6. Risk metrics provide information that price alone cannot

A high Bitcoin price does not necessarily imply high market risk, while a declining price does not automatically imply maximum stress.

Risk conditions depend on the distribution and dynamics of returns rather than price level alone.

---

## 7. VaR can be well calibrated and still fail dynamically

One of the most important findings from the project is that a model may produce approximately the correct long-run number of VaR exceedances while those exceedances remain temporally clustered.

This illustrates the distinction between:

- **unconditional calibration**
- **dynamic adequacy**

---

## 8. Risk regimes contain meaningful forward information

Higher current risk regimes correspond to higher subsequent realized volatility.

This suggests that the composite risk-state framework captures economically meaningful differences in market conditions.

---

# Strategy, Quantitative Realizations, and Recommendations

One of the most important lessons from this project is that quantitative risk management is not primarily about finding a single sophisticated model that produces the "correct" answer. It is about building a system of measurements, diagnostics, and decision rules that collectively describe how uncertainty is evolving. At the beginning of the project, the natural temptation was to think of risk modeling as a competition between individual methods: historical simulation, GARCH, Value at Risk, Expected Shortfall, or regime classification. As the analysis developed, it became increasingly clear that these tools answer different questions. Volatility measures instability, VaR estimates a loss threshold, Expected Shortfall measures the severity of losses after the threshold is crossed, and regime classification converts several dimensions of risk into a more interpretable state. Their value therefore comes from how they complement one another rather than from selecting a single "best" statistic.

The project also reinforced the difference between **descriptive accuracy and decision usefulness**. A metric can be statistically valid without being sufficient for monitoring. Historical VaR is a good example. The observed exceedance frequency can be close to its intended probability level, which initially appears to indicate a successful model. However, when those exceedances are examined through an independence test and breach-spacing diagnostics, a more nuanced picture emerges. Failures tend to cluster during high-volatility episodes. From a risk-management perspective, this matters because losses occurring repeatedly within a short period are far more consequential than the same number of losses evenly distributed across many years. A portfolio manager, treasury team, or risk committee cares not only about whether a model is correct on average, but also about whether it remains informative precisely when markets become unstable.

That realization changed the strategic direction of the project. Rather than continuously adding more models, the project increasingly emphasized **validation, monitoring, and interpretation**. This is an important quantitative lesson: complexity is not automatically sophistication. A transparent model with well-understood weaknesses, strong diagnostics, and a clear monitoring framework may be more useful than a highly complex model whose behavior is difficult to interpret. The deployed system therefore retains relatively transparent rolling risk measures while surrounding them with model-validation tools. The objective is not to claim that historical VaR perfectly describes Bitcoin risk. Instead, the objective is to understand where it performs well, where it fails, and how those failures should affect interpretation.

Another major realization involved **look-ahead bias**. Early versions of the regime framework ranked historical observations against the entire dataset. Although acceptable for a purely descriptive historical analysis, that design implicitly allowed future market information to influence classifications in the past. Once the project evolved toward a live risk-monitoring platform, that assumption became inconsistent with the intended use case. Replacing full-sample rankings with expanding causal percentiles substantially improved the conceptual integrity of the model. Each day's regime is now determined only by information that would have been available at that date. This change may appear small from a programming perspective, but it represents an important transition from retrospective analytics toward genuine out-of-sample thinking.

The live-data architecture generated a similar insight. Connecting an API is relatively easy; creating a trustworthy live analytical pipeline is not. A live price cannot simply be appended to a stale historical dataset. Doing so could convert weeks or months of price movement into a single daily return, corrupting volatility, VaR, Expected Shortfall, and regime calculations simultaneously. The automatic freshness check and historical backfill were therefore not merely engineering conveniences. They were quantitative controls designed to protect the validity of downstream calculations. This reinforced the idea that in financial analytics, **data engineering and model risk are inseparable**. A mathematically correct model operating on incorrectly aligned data can produce misleading results just as easily as a poorly specified model.

The regime-validation results also changed how I think about classification systems in finance. A risk regime should not be considered useful simply because its labels appear intuitive on a historical chart. A meaningful regime should correspond to observable differences in financial outcomes. By evaluating subsequent seven-day realized volatility across current regime categories, the project tested whether the classification contained information about future instability. The strong separation between lower and higher risk states provides evidence that the regime framework captures meaningful differences in the underlying market environment. This does not mean the regime classifier should be treated as a price-prediction model. Rather, it suggests that it can function as a **risk-conditioning mechanism**: when the system identifies a higher-risk state, the distribution of near-term volatility is materially different.

This distinction between **risk forecasting and return forecasting** is strategically important. The project does not attempt to predict whether Bitcoin will rise or fall tomorrow. Instead, it focuses on estimating the environment in which future returns will occur. This is closer to how many institutional risk systems are designed. A portfolio manager may not know the direction of the next market move, but knowing that volatility, tail severity, and downside-risk measures are simultaneously elevated can still influence position sizing, hedging, liquidity planning, or risk limits. In that sense, the dashboard is intentionally designed as a decision-support system rather than a trading-signal generator.

A further lesson is that model failures can themselves become valuable information. The clustering of VaR exceedances is not simply a weakness that should be hidden. It reveals that the underlying return process changes more rapidly during stress than a rolling historical distribution can always accommodate. That finding provides a natural motivation for possible future research into dynamic volatility models such as GARCH-Student-t, filtered historical simulation, Extreme Value Theory, or volatility-scaled VaR. The important point is that future model development would be driven by an observed empirical limitation rather than by a desire to add complexity for its own sake.

From a practical strategy perspective, the dashboard should therefore be interpreted hierarchically. The live market snapshot answers what the market is doing now. Rolling volatility describes current instability. VaR and Expected Shortfall quantify downside exposure. The composite regime places these values within historical context. The risk-intelligence layer summarizes directional changes in risk conditions. Finally, the validation layer answers whether the models themselves deserve confidence. This hierarchy mirrors a broader principle in financial decision-making: **market information, risk measurement, interpretation, and model validation should remain separate but connected layers**.

The primary recommendation from this project is therefore not to use a single metric as a trading or investment rule. Instead, the system should be used to identify changes in the **risk environment**. A transition from Low to Moderate or High Risk, rising volatility percentiles, worsening Expected Shortfall, and increasingly clustered VaR exceedances should collectively be interpreted as evidence that market uncertainty is becoming more concentrated. In a real portfolio context, such information could justify reviewing position sizes, liquidity exposure, leverage, stop-loss assumptions, or hedging requirements. Conversely, a Low Risk regime should not automatically be interpreted as a bullish signal; it simply indicates that recent market behavior is comparatively less stressed.

For future development, the strongest extensions would be those that directly address limitations identified through validation. A GARCH-Student-t benchmark could test whether conditional volatility improves VaR independence. Filtered historical simulation could preserve the empirical return distribution while adjusting for time-varying volatility. Extreme Value Theory could provide a more explicit model of the far-left tail. Multi-asset extensions could determine whether similar regime logic generalizes to Ethereum or broader crypto portfolios. However, these extensions should be treated as research questions rather than mandatory additions. The current project already demonstrates an important quantitative principle: a useful risk system is not defined by the number of models it contains, but by whether its data pipeline is reliable, its assumptions are transparent, its outputs are interpretable, and its failures are actively measured.

Ultimately, the project changed from an exercise in calculating risk statistics into an exercise in **thinking like a risk analyst**. The central question became less "What model can I build?" and more "What would I need to know before trusting this model in a real decision-making environment?" That shift—from metric construction to model skepticism, validation, and monitoring—is arguably the most important quantitative realization produced by the project.

---

# Dashboard Features

## Live Market Snapshot

The dashboard displays live market information including:

- Bitcoin price
- 24-hour price change
- 24-hour trading volume
- market capitalization
- market-data update timestamp

---

## Executive Risk Summary

The executive panel reports:

- current BTC price
- 30-day volatility
- selected VaR level
- selected Expected Shortfall level
- current market-risk regime
- composite risk score

---

## Risk Intelligence

The interpretation layer reports:

- 7-day composite risk trend
- current volatility state
- VaR breach status
- 30-day tail-risk trend

The system also generates risk alerts when selected stress conditions are triggered.

---

## Market Overview

Interactive charts visualize:

- Bitcoin price history
- 30-day volatility
- 60-day volatility

---

## Tail-Risk Monitor

The dashboard displays:

- daily returns
- VaR thresholds
- Expected Shortfall thresholds
- VaR exceedance events
- historical evolution of 5% and 1% tail-risk metrics

---

## Model Validation

VaR performance is evaluated using:

- empirical breach rates
- Kupiec unconditional coverage test
- Christoffersen independence test
- conditional coverage analysis

The purpose is to distinguish between simple long-run calibration and true dynamic adequacy.

---

## Exceedance-Clustering Diagnostics

The dashboard measures:

- average gap between VaR failures
- median gap between failures
- minimum breach spacing
- closely spaced breach counts
- historical clustering of exceedance events

These diagnostics reveal whether tail-risk failures are distributed independently or concentrated during stress episodes.

---

## Regime Validation

The application compares current regimes with subsequent realized risk.

Validation metrics include:

- average absolute return
- average realized volatility
- forward seven-day absolute return
- forward seven-day volatility
- Expected Shortfall

This provides an empirical test of whether the regime classifier captures economically meaningful differences in future market instability.

---

## Risk Regime Classification

The dashboard visualizes:

- historical regime transitions
- regime frequencies
- risk-state distribution through time

---

## Regime Summary

Each risk category is summarized using metrics such as:

- average return
- realized return volatility
- average rolling volatility
- average VaR
- average Expected Shortfall
- average composite risk score
- number of observations

---

## Recent Stress Events

The dashboard maintains a stress-event log containing recent VaR exceedances and their associated:

- BTC closing price
- realized return
- VaR threshold
- Expected Shortfall
- risk regime

---

# Tech Stack

## Data and Analytics

- **Python**
- **Pandas**
- **NumPy**

## Visualization and Application

- **Plotly**
- **Streamlit**

## Market Data

- **CoinGecko REST API**

## Quantitative Methods

- rolling realized volatility
- historical Value at Risk
- Expected Shortfall
- causal percentile ranking
- composite risk scoring
- risk regime classification
- empirical VaR backtesting
- Kupiec unconditional coverage testing
- Christoffersen independence testing
- conditional coverage analysis
- exceedance-clustering diagnostics
- forward realized-risk validation

---

# Repository Structure

```text
bitcoin_risk_intelligence/
│
├── dashboard/
│   ├── app.py
│   └── market_data.py
│
├── data/
│   └── btc_daily.csv
│
├── notebooks/
│   ├── 01_data_engineering.ipynb
│   ├── 02_market_behavior.ipynb
│   ├── 03_volatility_modeling.ipynb
│   ├── 04_risk_modeling_var.ipynb
│   ├── 05_expected_shortfall.ipynb
│   └── 06_regime_detection.ipynb
│
├── requirements.txt
└── README.md
```

---

# Model Limitations

The project intentionally emphasizes transparent and interpretable risk models, but several limitations remain.

## Historical Dependence

Historical VaR and Expected Shortfall assume that recent historical observations provide a useful representation of future downside risk.

Rapid structural changes can weaken this assumption.

---

## Exceedance Clustering

Although unconditional VaR calibration is strong, exceedances demonstrate temporal dependence during high-volatility episodes.

This indicates that rolling historical VaR may adapt too slowly during abrupt volatility transitions.

---

## Provisional Live Observation

The live Bitcoin price is treated as the current provisional daily observation.

Risk metrics can therefore evolve intraday before the final daily close is known.

---

## Rule-Based Regime Classification

The risk regime is a rule-based analytical framework rather than a structural Markov regime-switching model.

Its purpose is interpretation and monitoring rather than probabilistic latent-state estimation.

---

## Single-Asset Scope

The current implementation focuses only on Bitcoin.

Cross-asset correlations, portfolio diversification, contagion effects, and portfolio-level risk aggregation are outside the current scope.

---

# Potential Future Extensions

Possible future research directions include:

- GARCH-Student-t VaR benchmarking
- filtered historical simulation
- Extreme Value Theory
- volatility-scaled VaR
- rolling Expected Shortfall backtesting
- probability-based regime modeling
- multi-asset crypto portfolio risk
- correlation and contagion analysis
- scenario analysis
- stress testing
- portfolio-level VaR
- automated risk notifications

These extensions are intentionally treated as future research rather than necessary components of the current system.

---

# Key Takeaway

The central result of this project is not simply that Bitcoin is volatile.

The deeper conclusion is that **Bitcoin risk is dynamic, persistent, heavy-tailed, and strongly regime-dependent**.

A useful monitoring framework therefore requires more than price visualization.

It requires a combination of:

```text
Reliable Data
      +
Risk Measurement
      +
Tail Analysis
      +
Regime Detection
      +
Live Monitoring
      +
Model Validation
      +
Interpretation
```

The project demonstrates how these components can be integrated into a single deployed quantitative risk-intelligence workflow.

More importantly, the project demonstrates that **model validation can be as informative as model construction itself**. A risk model should not only produce a number; its assumptions, failures, calibration, and behavior during periods of stress should also be understood.

---

# Disclaimer

This project is intended for **educational, analytical, and research purposes only**.

It does not constitute investment advice, a trading recommendation, or a production risk-management system.
