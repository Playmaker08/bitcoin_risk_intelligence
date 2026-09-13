import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from market_data import (
    get_live_btc_market_data,
    get_historical_btc_data,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bitcoin Risk Intelligence Dashboard",
    page_icon="₿",
    layout="wide"
)


# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.badge-low {
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background:#163824;
    color:#9be7b1;
    font-weight:600;
}
.badge-moderate {
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background:#4a3b14;
    color:#ffd76a;
    font-weight:600;
}
.badge-high {
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background:#4a2914;
    color:#ffb26a;
    font-weight:600;
}
.badge-extreme {
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background:#4a1414;
    color:#ff8e8e;
    font-weight:600;
}
.section-note {
    color: #b8b8b8;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def historical_var(x: pd.Series, alpha: float = 0.05) -> float:
    return np.quantile(x, alpha)


def historical_es(x: pd.Series, alpha: float = 0.05) -> float:
    var = np.quantile(x, alpha)
    tail = x[x <= var]
    return tail.mean() if len(tail) > 0 else np.nan


def classify_regime(score: float) -> str:
    if score < 0.50:
        return "Low Risk"
    elif score < 0.80:
        return "Moderate Risk"
    elif score < 0.95:
        return "High Risk"
    return "Extreme Risk"


def regime_badge(regime: str) -> str:
    classes = {
        "Low Risk": "badge-low",
        "Moderate Risk": "badge-moderate",
        "High Risk": "badge-high",
        "Extreme Risk": "badge-extreme",
    }
    return f"<span class='{classes.get(regime, 'badge-low')}'>{regime}</span>"


def risk_interpretation(latest_row: pd.Series) -> str:
    regime = latest_row["risk_regime"]
    vol = latest_row["vol_30d"]
    var_5 = latest_row["VaR_5"]
    es_5 = latest_row["ES_5"]
    score = latest_row["risk_score"]

    if regime == "Low Risk":
        return (
            f"Bitcoin is currently in a {regime} state. Recent 30-day realized volatility "
            f"({vol:.2f}%) is relatively calm versus historical stress periods. The 5% VaR "
            f"({var_5:.2f}%) and 5% Expected Shortfall ({es_5:.2f}%) suggest downside risk remains present "
            f"but not elevated. Composite risk score: {score:.3f}."
        )
    elif regime == "Moderate Risk":
        return (
            f"Bitcoin is currently in a {regime} state. Risk conditions are elevated relative to calmer periods, "
            f"but still below severe stress thresholds. The latest VaR ({var_5:.2f}%) and ES ({es_5:.2f}%) imply "
            f"meaningful downside risk if volatility rises further. Composite risk score: {score:.3f}."
        )
    elif regime == "High Risk":
        return (
            f"Bitcoin is currently in a {regime} state. Market conditions show significantly elevated volatility "
            f"and heavier downside risk. The latest VaR ({var_5:.2f}%) and ES ({es_5:.2f}%) indicate that tail-loss "
            f"risk is materially higher than normal. Composite risk score: {score:.3f}."
        )
    else:
        return (
            f"Bitcoin is currently in an {regime} state. Risk conditions are near the most stressed levels observed "
            f"in the sample. The latest VaR ({var_5:.2f}%) and ES ({es_5:.2f}%) point to severe downside exposure. "
            f"Composite risk score: {score:.3f}."
        )


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
        hovermode="x unified"
    )
    return fig


def filter_by_range(df: pd.DataFrame, range_option: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if range_option == "Full History":
        return df.copy()

    end_date = df.index.max()

    range_map = {
        "Last 6M": pd.DateOffset(months=6),
        "Last 1Y": pd.DateOffset(years=1),
        "Last 2Y": pd.DateOffset(years=2),
        "Last 5Y": pd.DateOffset(years=5),
    }

    offset = range_map.get(range_option)

    if offset is None:
        return df.copy()

    start_date = end_date - offset

    return df.loc[df.index >= start_date].copy()


def latest_change(series: pd.Series, periods: int = 1):
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    return clean.iloc[-1] - clean.iloc[-1 - periods]

def causal_percentile_rank(
    series: pd.Series,
    min_periods: int = 250
) -> pd.Series:
    """
    Percentile rank of each observation using only information
    available up to that date.

    No future observations are used.
    """

    return (
        series
        .expanding(min_periods=min_periods)
        .apply(
            lambda x: np.mean(x <= x[-1]),
            raw=True
        )
    )

def classify_risk_trend(
    current_score: float,
    previous_score: float,
    threshold: float = 0.05
) -> str:
    """
    Classify short-term direction of composite risk.
    """

    change = current_score - previous_score

    if change > threshold:
        return "Rising"
    elif change < -threshold:
        return "Falling"

    return "Stable"


def classify_volatility_state(vol_percentile: float) -> str:
    """
    Convert current causal volatility percentile
    into an interpretable market state.
    """

    if vol_percentile < 0.50:
        return "Calm"
    elif vol_percentile < 0.80:
        return "Normal"
    elif vol_percentile < 0.95:
        return "Elevated"

    return "Extreme"


def classify_var_status(
    current_return: float,
    var_5: float,
    var_1: float
) -> str:
    """
    Determine whether today's return breached
    the 5% or 1% VaR threshold.
    """

    if current_return < var_1:
        return "Severe Breach"

    elif current_return < var_5:
        return "Breach"

    return "Normal"


def classify_tail_risk_trend(
    current_es: float,
    previous_es: float,
    threshold: float = 0.25
) -> str:
    """
    Compare current Expected Shortfall with
    its value 30 observations earlier.

    More negative ES = worsening tail risk.
    """

    change = current_es - previous_es

    if change < -threshold:
        return "Deteriorating"

    elif change > threshold:
        return "Improving"

    return "Stable"

def calculate_var_backtest(df: pd.DataFrame) -> dict:
    """
    Calculate empirical VaR exceedance rates.
    """

    valid = df.dropna(
        subset=["return_pct", "VaR_5", "VaR_1"]
    ).copy()

    if valid.empty:
        return {
            "observations": 0,
            "breaches_5": 0,
            "breaches_1": 0,
            "rate_5": np.nan,
            "rate_1": np.nan,
        }

    n = len(valid)

    breaches_5 = int(
        (valid["return_pct"] < valid["VaR_5"]).sum()
    )

    breaches_1 = int(
        (valid["return_pct"] < valid["VaR_1"]).sum()
    )

    return {
        "observations": n,
        "breaches_5": breaches_5,
        "breaches_1": breaches_1,
        "rate_5": breaches_5 / n,
        "rate_1": breaches_1 / n,
    }

def interpret_var_calibration(
    observed_rate: float,
    expected_rate: float,
    tolerance: float = 0.25
) -> str:
    """
    Compare observed breach frequency with nominal VaR level.
    """

    if np.isnan(observed_rate):
        return "N/A"

    lower = expected_rate * (1 - tolerance)
    upper = expected_rate * (1 + tolerance)

    if observed_rate < lower:
        return "Conservative"

    elif observed_rate > upper:
        return "Underestimating Risk"

    return "Well Calibrated"

def check_regime_ordering(validation_df: pd.DataFrame) -> str:
    """
    Check whether forward volatility generally rises
    across Low -> Moderate -> High -> Extreme regimes.
    """

    values = (
        validation_df["avg_forward_7d_vol"]
        .dropna()
        .values
    )

    if len(values) < 2:
        return "Insufficient Data"

    if all(
        values[i] <= values[i + 1]
        for i in range(len(values) - 1)
    ):
        return "Strong Separation"

    return "Mixed Separation"

def log_likelihood_term(count: int, probability: float) -> float:
    """
    Compute count * log(probability) safely.
    Uses the convention 0 * log(0) = 0.
    """

    if count == 0:
        return 0.0

    if probability <= 0 or probability >= 1:
        return -np.inf

    return count * math.log(probability)

def kupiec_test(
    exceedances: pd.Series,
    alpha: float
) -> dict:
    """
    Kupiec Proportion-of-Failures test.

    H0: empirical VaR exceedance probability = alpha.
    """

    breaches = exceedances.dropna().astype(bool)

    n = len(breaches)
    x = int(breaches.sum())

    if n == 0:
        return {
            "lr_uc": np.nan,
            "p_value": np.nan,
            "result": "N/A",
        }

    p_hat = x / n

    # Null likelihood
    log_l0 = (
        log_likelihood_term(n - x, 1 - alpha)
        + log_likelihood_term(x, alpha)
    )

    # Alternative likelihood
    if p_hat == 0:
        log_l1 = 0.0
    elif p_hat == 1:
        log_l1 = 0.0
    else:
        log_l1 = (
            log_likelihood_term(n - x, 1 - p_hat)
            + log_likelihood_term(x, p_hat)
        )

    lr_uc = max(
        0.0,
        -2 * (log_l0 - log_l1)
    )

    # Chi-square(1) survival function
    p_value = math.erfc(
        math.sqrt(lr_uc / 2)
    )

    return {
        "lr_uc": lr_uc,
        "p_value": p_value,
        "result": (
            "Pass"
            if p_value >= 0.05
            else "Reject"
        ),
    }

def christoffersen_independence_test(
    exceedances: pd.Series
) -> dict:
    """
    Christoffersen independence test.

    Tests whether VaR breaches are independent over time.
    """

    breaches = (
        exceedances
        .dropna()
        .astype(int)
        .values
    )

    if len(breaches) < 2:
        return {
            "lr_ind": np.nan,
            "p_value": np.nan,
            "result": "N/A",
        }

    previous = breaches[:-1]
    current = breaches[1:]

    n00 = int(
        ((previous == 0) & (current == 0)).sum()
    )

    n01 = int(
        ((previous == 0) & (current == 1)).sum()
    )

    n10 = int(
        ((previous == 1) & (current == 0)).sum()
    )

    n11 = int(
        ((previous == 1) & (current == 1)).sum()
    )

    denom_0 = n00 + n01
    denom_1 = n10 + n11

    if denom_0 == 0 or denom_1 == 0:
        return {
            "lr_ind": np.nan,
            "p_value": np.nan,
            "result": "Insufficient Data",
        }

    p01 = n01 / denom_0
    p11 = n11 / denom_1

    total_transitions = (
        n00 + n01 + n10 + n11
    )

    p = (
        n01 + n11
    ) / total_transitions

    # Independence null
    log_l0 = (
        log_likelihood_term(
            n00 + n10,
            1 - p
        )
        + log_likelihood_term(
            n01 + n11,
            p
        )
    )

    # First-order Markov alternative
    log_l1 = (
        log_likelihood_term(n00, 1 - p01)
        + log_likelihood_term(n01, p01)
        + log_likelihood_term(n10, 1 - p11)
        + log_likelihood_term(n11, p11)
    )

    lr_ind = max(
        0.0,
        -2 * (log_l0 - log_l1)
    )

    p_value = math.erfc(
        math.sqrt(lr_ind / 2)
    )

    return {
        "lr_ind": lr_ind,
        "p_value": p_value,
        "result": (
            "Pass"
            if p_value >= 0.05
            else "Reject"
        ),
    }

def conditional_coverage_test(
    kupiec_result: dict,
    independence_result: dict
) -> dict:

    lr_uc = kupiec_result["lr_uc"]
    lr_ind = independence_result["lr_ind"]

    if (
        np.isnan(lr_uc)
        or np.isnan(lr_ind)
    ):
        return {
            "lr_cc": np.nan,
            "p_value": np.nan,
            "result": "N/A",
        }

    lr_cc = lr_uc + lr_ind

    # Chi-square with 2 degrees of freedom:
    # survival function = exp(-x / 2)
    p_value = math.exp(
        -lr_cc / 2
    )

    return {
        "lr_cc": lr_cc,
        "p_value": p_value,
        "result": (
            "Pass"
            if p_value >= 0.05
            else "Reject"
        ),
    }



# =========================================================
# DATA PIPELINE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "btc_daily.csv"


@st.cache_data(show_spinner=False)
def load_historical_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean historical daily Bitcoin data.
    """

    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        st.error("btc_daily.csv must contain a 'date' column.")
        st.stop()

    if "Close" not in df.columns:
        st.error("btc_daily.csv must contain a 'Close' column.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Always recalculate returns to guarantee consistency
    df["return_pct"] = df["Close"].pct_change() * 100

    return df


def append_live_observation(
    df: pd.DataFrame,
    live_market: dict
) -> pd.DataFrame:
    """
    Append or update today's provisional Bitcoin close
    using the latest live CoinGecko price.
    """

    combined = df.copy()

    if live_market.get("status") != "live":
        return combined

    live_price = live_market.get("price")

    if live_price is None:
        return combined

    live_timestamp = live_market.get("last_updated")

    if live_timestamp is not None:
        today = (
            pd.Timestamp(live_timestamp)
            .tz_localize(None)
            .normalize()
        )
    else:
        today = (
            pd.Timestamp.utcnow()
            .tz_localize(None)
            .normalize()
        )

    # Add/update today's provisional close
    combined.loc[today, "Close"] = float(live_price)

    combined = combined.sort_index()

    # Recalculate returns after live observation is inserted
    combined["return_pct"] = combined["Close"].pct_change() * 100

    return combined


def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling volatility, historical VaR,
    Expected Shortfall, exceedances, and risk regimes.
    """

    risk_df = df.copy()

    # -----------------------------------------------------
    # REALIZED VOLATILITY
    # -----------------------------------------------------

    risk_df["vol_30d"] = (
        risk_df["return_pct"]
        .rolling(30)
        .std()
    )

    risk_df["vol_60d"] = (
        risk_df["return_pct"]
        .rolling(60)
        .std()
    )

    # -----------------------------------------------------
    # HISTORICAL VALUE AT RISK
    # -----------------------------------------------------

    risk_df["VaR_5"] = (
        risk_df["return_pct"]
        .rolling(250)
        .quantile(0.05)
    )

    risk_df["VaR_1"] = (
        risk_df["return_pct"]
        .rolling(250)
        .quantile(0.01)
    )

    # -----------------------------------------------------
    # EXPECTED SHORTFALL
    # -----------------------------------------------------

    risk_df["ES_5"] = (
        risk_df["return_pct"]
        .rolling(250)
        .apply(
            lambda x: historical_es(pd.Series(x), 0.05),
            raw=False
        )
    )

    risk_df["ES_1"] = (
        risk_df["return_pct"]
        .rolling(250)
        .apply(
            lambda x: historical_es(pd.Series(x), 0.01),
            raw=False
        )
    )

    # -----------------------------------------------------
    # RISK SCORE INPUTS
    # -----------------------------------------------------

    risk_df["vol_score"] = risk_df["vol_30d"]

    # VaR and ES are negative.
    # More negative = greater downside risk.
    risk_df["var_score"] = -risk_df["VaR_5"]
    risk_df["es_score"] = -risk_df["ES_5"]

    risk_df = risk_df.dropna().copy()


    # -----------------------------------------------------
    # CAUSAL HISTORICAL PERCENTILE RANKS
    # -----------------------------------------------------

    risk_df["vol_pct"] = causal_percentile_rank(
        risk_df["vol_score"],
        min_periods=250
    )

    risk_df["var_pct"] = causal_percentile_rank(
        risk_df["var_score"],
        min_periods=250
    )

    risk_df["es_pct"] = causal_percentile_rank(
        risk_df["es_score"],
        min_periods=250
    )

    risk_df = risk_df.dropna(
        subset=["vol_pct", "var_pct", "es_pct"]
    ).copy()


    # -----------------------------------------------------
    # COMPOSITE RISK SCORE
    # -----------------------------------------------------

    risk_df["risk_score"] = (
        0.40 * risk_df["vol_pct"]
        + 0.30 * risk_df["var_pct"]
        + 0.30 * risk_df["es_pct"]
    )

    risk_df["risk_regime"] = (
        risk_df["risk_score"]
        .apply(classify_regime)
    )

    regime_map = {
        "Low Risk": 1,
        "Moderate Risk": 2,
        "High Risk": 3,
        "Extreme Risk": 4,
    }

    risk_df["regime_code"] = (
        risk_df["risk_regime"]
        .map(regime_map)
    )


    # -----------------------------------------------------
    # VAR EXCEEDANCES
    # -----------------------------------------------------

    risk_df["exceed_5"] = (
        risk_df["return_pct"] < risk_df["VaR_5"]
    )

    risk_df["exceed_1"] = (
        risk_df["return_pct"] < risk_df["VaR_1"]
    )

    return risk_df


# =========================================================
# HISTORICAL BACKFILL
# =========================================================

def backfill_missing_history(
    historical_df: pd.DataFrame,
    gap_days: int
) -> pd.DataFrame:
    """
    Backfill missing daily BTC prices using CoinGecko historical data.
    """

    if gap_days <= 1:
        return historical_df.copy()

    api_history = get_historical_btc_data(
        days=gap_days + 2
    )

    if api_history.empty:
        return historical_df.copy()

    api_history = api_history.set_index("date")

    combined = pd.concat([
        historical_df[["Close"]],
        api_history[["Close"]],
    ])

    combined = (
        combined[
            ~combined.index.duplicated(keep="last")
        ]
        .sort_index()
    )

    combined["return_pct"] = (
        combined["Close"]
        .pct_change()
        * 100
    )

    return combined


# =========================================================
# BUILD ANALYTICS PIPELINE
# =========================================================

historical_data = load_historical_data(DATA_PATH)

if historical_data.empty:
    st.error("No historical data available.")
    st.stop()


# ---------------------------------------------------------
# HISTORICAL DATA FRESHNESS
# ---------------------------------------------------------

historical_end = (
    historical_data.index
    .max()
    .normalize()
)

today_utc = (
    pd.Timestamp.utcnow()
    .tz_localize(None)
    .normalize()
)

gap_days = (
    today_utc - historical_end
).days


# ---------------------------------------------------------
# BACKFILL MISSING DAILY DATA
# ---------------------------------------------------------

updated_history = backfill_missing_history(
    historical_data,
    gap_days
)

updated_end = (
    updated_history.index
    .max()
    .normalize()
)

remaining_gap = (
    today_utc - updated_end
).days


# ---------------------------------------------------------
# LIVE MARKET SNAPSHOT
# ---------------------------------------------------------

live_market = get_live_btc_market_data()


# ---------------------------------------------------------
# BUILD HYBRID HISTORICAL + LIVE SERIES
# ---------------------------------------------------------

if (
    live_market.get("status") == "live"
    and remaining_gap <= 1
):
    combined_data = append_live_observation(
        updated_history,
        live_market
    )

    live_risk_enabled = True

else:
    combined_data = updated_history.copy()
    live_risk_enabled = False


# ---------------------------------------------------------
# RUN RISK ENGINE
# ---------------------------------------------------------

data = calculate_risk_metrics(
    combined_data
)

# =========================================================
# REGIME VALIDATION FEATURES
# =========================================================

data["abs_return"] = data["return_pct"].abs()

data["forward_7d_abs_return"] = (
    data["return_pct"]
    .abs()
    .rolling(7)
    .mean()
    .shift(-7)
)

data["forward_7d_vol"] = (
    data["return_pct"]
    .rolling(7)
    .std()
    .shift(-7)
)

regime_validation = (
    data
    .groupby("risk_regime")
    .agg(
        avg_abs_return=("abs_return", "mean"),
        avg_vol_30d=("vol_30d", "mean"),
        avg_forward_7d_abs_return=("forward_7d_abs_return", "mean"),
        avg_forward_7d_vol=("forward_7d_vol", "mean"),
        avg_es_5=("ES_5", "mean"),
        observations=("risk_regime", "count"),
    )
    .reindex([
        "Low Risk",
        "Moderate Risk",
        "High Risk",
        "Extreme Risk",
    ])
)
regime_validation_status = check_regime_ordering(
    regime_validation
)

var_backtest = calculate_var_backtest(data)

# =========================================================
# FORMAL VAR BACKTESTS
# =========================================================

kupiec_5 = kupiec_test(
    data["exceed_5"],
    alpha=0.05
)

independence_5 = (
    christoffersen_independence_test(
        data["exceed_5"]
    )
)

conditional_5 = conditional_coverage_test(
    kupiec_5,
    independence_5
)


kupiec_1 = kupiec_test(
    data["exceed_1"],
    alpha=0.01
)

independence_1 = (
    christoffersen_independence_test(
        data["exceed_1"]
    )
)

conditional_1 = conditional_coverage_test(
    kupiec_1,
    independence_1
)

formal_var_tests = pd.DataFrame({
    "VaR Level": [
        "5%",
        "1%",
    ],

    "Kupiec p-value": [
        kupiec_5["p_value"],
        kupiec_1["p_value"],
    ],

    "Coverage": [
        kupiec_5["result"],
        kupiec_1["result"],
    ],

    "Independence p-value": [
        independence_5["p_value"],
        independence_1["p_value"],
    ],

    "Independence": [
        independence_5["result"],
        independence_1["result"],
    ],

    "Conditional Coverage p-value": [
        conditional_5["p_value"],
        conditional_1["p_value"],
    ],

    "Overall": [
        conditional_5["result"],
        conditional_1["result"],
    ],
})

calibration_5 = interpret_var_calibration(
    var_backtest["rate_5"],
    0.05
)

calibration_1 = interpret_var_calibration(
    var_backtest["rate_1"],
    0.01
)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Dashboard Controls")
view_range = st.sidebar.selectbox(
    "Date Range",
    ["Last 6M", "Last 1Y", "Last 2Y", "Last 5Y", "Full History"],
    index=2
)

tail_level = st.sidebar.selectbox(
    "Tail-Risk Level",
    ["5%", "1%"],
    index=0
)

show_exceedances = st.sidebar.checkbox("Highlight Exceedances", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Data Status")

if live_market.get("status") == "live":
    st.sidebar.success("CoinGecko Feed: LIVE")
else:
    st.sidebar.error("CoinGecko Feed: OFFLINE")


if live_risk_enabled:
    st.sidebar.success("Risk Engine: LIVE")
else:
    st.sidebar.warning("Risk Engine: HISTORICAL")


st.sidebar.caption(
    f"Risk data through: "
    f"{updated_end.strftime('%Y-%m-%d')}"
)

display_df = filter_by_range(data, view_range)
latest = data.iloc[-1]

# =========================================================
# RISK INTELLIGENCE SIGNALS
# =========================================================

# 7-day risk-score trend
if len(data) >= 8:
    risk_score_7d_ago = data["risk_score"].iloc[-8]

    risk_trend = classify_risk_trend(
        latest["risk_score"],
        risk_score_7d_ago
    )

    risk_score_change_7d = (
        latest["risk_score"]
        - risk_score_7d_ago
    )

else:
    risk_trend = "N/A"
    risk_score_change_7d = np.nan


# Current volatility state
volatility_state = classify_volatility_state(
    latest["vol_pct"]
)


# Current VaR breach status
var_status = classify_var_status(
    latest["return_pct"],
    latest["VaR_5"],
    latest["VaR_1"]
)


# 30-day Expected Shortfall trend
if len(data) >= 31:
    es_30d_ago = data["ES_5"].iloc[-31]

    tail_risk_trend = classify_tail_risk_trend(
        latest["ES_5"],
        es_30d_ago
    )

    es_change_30d = (
        latest["ES_5"]
        - es_30d_ago
    )

else:
    tail_risk_trend = "N/A"
    es_change_30d = np.nan

selected_var = "VaR_5" if tail_level == "5%" else "VaR_1"
selected_es = "ES_5" if tail_level == "5%" else "ES_1"
selected_exceed = "exceed_5" if tail_level == "5%" else "exceed_1"


# =========================================================
# TITLE
# =========================================================
st.title("Bitcoin Risk Intelligence Dashboard")
st.markdown(
    "<div class='section-note'>"
    "A live-updated Bitcoin market risk platform combining historical "
    "price data with live market observations, volatility analysis, "
    "Value-at-Risk, Expected Shortfall, and regime classification."
    "</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# LIVE MARKET SNAPSHOT
# =========================================================
st.subheader("Live Market Snapshot")

if live_market["status"] == "live":

    live1, live2, live3, live4 = st.columns(4)

    with live1:
        st.metric(
            "Live BTC Price",
            f"${live_market['price']:,.0f}"
        )

    with live2:
        change = live_market["change_24h"]

        st.metric(
            "24H Change",
            f"{change:.2f}%",
            delta=f"{change:.2f}%"
        )

    with live3:
        st.metric(
            "24H Trading Volume",
            f"${live_market['volume_24h'] / 1e9:,.2f}B"
        )

    with live4:
        st.metric(
            "Market Cap",
            f"${live_market['market_cap'] / 1e12:,.2f}T"
        )

    if live_market["last_updated"] is not None:
        st.caption(
            "Live market feed • Last updated: "
            f"{live_market['last_updated'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

else:
    st.warning(
        "Live market feed is temporarily unavailable. "
        "Historical risk analytics remain available."
    )

st.markdown("---")

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================
st.subheader("Executive Summary")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

price_delta = latest_change(data["Close"], periods=1)
vol_delta = latest_change(data["vol_30d"], periods=1)
var_delta = latest_change(data[selected_var], periods=1)
es_delta = latest_change(data[selected_es], periods=1)


with kpi1:
    st.metric(
        "BTC Price" if live_risk_enabled else "Historical Close",
        f"${latest['Close']:,.0f}",
        None if price_delta is None else f"{price_delta:,.0f}"
    )

with kpi2:
    st.metric(
        "30D Volatility",
        f"{latest['vol_30d']:.3f}%",
        None if vol_delta is None else f"{vol_delta:.3f}"
    )

with kpi3:
    st.metric(
        f"VaR {tail_level}",
        f"{latest[selected_var]:.3f}%",
        None if var_delta is None else f"{var_delta:.3f}"
    )

with kpi4:
    st.metric(
        f"ES {tail_level}",
        f"{latest[selected_es]:.3f}%",
        None if es_delta is None else f"{es_delta:.3f}"
    )

with kpi5:
    st.markdown("**Current Risk Regime**")
    st.markdown(regime_badge(latest["risk_regime"]), unsafe_allow_html=True)
    st.caption(f"Risk Score: {latest['risk_score']:.3f}")

st.info(risk_interpretation(latest))

if live_risk_enabled:
    st.success(
        "Live risk engine active — the latest CoinGecko BTC price "
        "is incorporated as today's provisional observation."
    )

    st.caption(
        "Volatility, VaR, Expected Shortfall, and risk regime "
        "are recalculated using the latest provisional daily BTC price."
    )

else:
    if live_market.get("status") != "live":
        st.warning(
            "Live market feed is unavailable. "
            "Risk analytics are based on historical observations."
        )

    elif gap_days > 1:
        st.warning(
            f"Live risk engine disabled because the historical dataset "
            f"is {gap_days} days behind the live market date. "
            "Live price is displayed separately but is not included "
            "in return or risk calculations."
        )

st.markdown("#### Statistical VaR Backtesting")

st.dataframe(
    formal_var_tests.style.format({
        "Kupiec p-value": "{:.3f}",
        "Independence p-value": "{:.3f}",
        "Conditional Coverage p-value": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

if (
    conditional_5["result"] == "Pass"
    and conditional_1["result"] == "Pass"
):
    st.success(
        "Formal VaR backtesting passed at both confidence levels: "
        "the models show acceptable unconditional coverage and "
        "breach independence under the combined conditional coverage test."
    )

else:
    st.warning(
        "At least one VaR model fails the formal conditional coverage test. "
        "Review exceedance frequency and breach clustering before interpreting "
        "the model as fully calibrated."
    )

# =========================================================
# RISK INTELLIGENCE
# =========================================================

st.markdown("---")
st.subheader("Risk Intelligence")

intel1, intel2, intel3, intel4 = st.columns(4)


with intel1:
    st.metric(
        "Risk Trend (7D)",
        risk_trend,
        None if np.isnan(risk_score_change_7d)
        else f"{risk_score_change_7d:+.3f}"
    )


with intel2:
    st.metric(
        "Volatility State",
        volatility_state,
        f"{latest['vol_pct']:.0%} percentile"
    )


with intel3:
    st.metric(
        "VaR Status",
        var_status,
        f"Return: {latest['return_pct']:.2f}%"
    )


with intel4:
    st.metric(
        "Tail-Risk Trend (30D)",
        tail_risk_trend,
        None if np.isnan(es_change_30d)
        else f"{es_change_30d:+.3f}"
    )

risk_signal_text = (
    f"Composite risk is currently **{risk_trend.lower()}** over the past 7 days. "
    f"Realized volatility is in the **{latest['vol_pct']:.0%} historical percentile**, "
    f"corresponding to a **{volatility_state.lower()} volatility environment**. "
    f"The latest daily return shows **{var_status.lower()}** relative to current VaR thresholds. "
    f"Expected Shortfall indicates tail risk is **{tail_risk_trend.lower()}** versus 30 days ago."
)

st.info(risk_signal_text)

# =========================================================
# RISK ALERT
# =========================================================

if var_status == "Severe Breach":
    st.error(
        "Critical Risk Alert: today's return has breached the 1% VaR threshold."
    )

elif var_status == "Breach":
    st.warning(
        "Risk Alert: today's return has breached the 5% VaR threshold."
    )

elif latest["risk_regime"] == "Extreme Risk":
    st.error(
        "Critical Risk Alert: composite market risk is in the Extreme Risk regime."
    )

elif latest["risk_regime"] == "High Risk":
    st.warning(
        "Elevated Risk Alert: composite market risk is in the High Risk regime."
    )

# =========================================================
# MARKET OVERVIEW
# =========================================================
st.subheader("Market Overview")

col_left, col_right = st.columns(2)

with col_left:
    fig_price = px.line(
        display_df,
        x=display_df.index,
        y="Close",
        labels={"x": "Date", "Close": "Price (USD)"},
        title="BTC Price"
    )
    apply_theme(fig_price)
    st.plotly_chart(fig_price, use_container_width=True)

with col_right:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["vol_30d"],
        mode="lines",
        name="30D Volatility"
    ))
    fig_vol.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["vol_60d"],
        mode="lines",
        name="60D Volatility"
    ))
    fig_vol.update_layout(
        title="Rolling Volatility",
        yaxis_title="Volatility (%)",
        xaxis_title="Date"
    )
    apply_theme(fig_vol)
    st.plotly_chart(fig_vol, use_container_width=True)


# =========================================================
# TAIL RISK MONITOR
# =========================================================
st.subheader("Tail-Risk Monitor")

col_left, col_right = st.columns(2)

with col_left:
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["return_pct"],
        mode="lines",
        name="Daily Return (%)",
        opacity=0.55
    ))
    fig_var.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df[selected_var],
        mode="lines",
        name=f"VaR {tail_level}"
    ))
    fig_var.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df[selected_es],
        mode="lines",
        name=f"ES {tail_level}"
    ))

    if show_exceedances:
        exceed_mask = display_df[selected_exceed]
        fig_var.add_trace(go.Scatter(
            x=display_df.index[exceed_mask],
            y=display_df.loc[exceed_mask, "return_pct"],
            mode="markers",
            name="Exceedances",
            marker=dict(size=7)
        ))

    fig_var.update_layout(
        title=f"Returns vs VaR / ES ({tail_level})",
        yaxis_title="Return / Risk Threshold (%)",
        xaxis_title="Date"
    )
    apply_theme(fig_var)
    st.plotly_chart(fig_var, use_container_width=True)

with col_right:
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["VaR_5"],
        mode="lines",
        name="VaR 5%"
    ))
    fig_risk.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["ES_5"],
        mode="lines",
        name="ES 5%"
    ))
    fig_risk.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["VaR_1"],
        mode="lines",
        name="VaR 1%"
    ))
    fig_risk.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["ES_1"],
        mode="lines",
        name="ES 1%"
    ))
    fig_risk.update_layout(
        title="Tail-Risk Metrics Over Time",
        yaxis_title="Risk Threshold (%)",
        xaxis_title="Date"
    )
    apply_theme(fig_risk)
    st.plotly_chart(fig_risk, use_container_width=True)

# =========================================================
# MODEL VALIDATION
# =========================================================

st.subheader("Model Validation")

bt1, bt2, bt3, bt4 = st.columns(4)

with bt1:
    st.metric(
        "5% VaR Breach Rate",
        "N/A"
        if np.isnan(var_backtest["rate_5"])
        else f"{var_backtest['rate_5']:.2%}",
        "Expected: 5%"
    )

with bt2:
    st.metric(
        "1% VaR Breach Rate",
        "N/A"
        if np.isnan(var_backtest["rate_1"])
        else f"{var_backtest['rate_1']:.2%}",
        "Expected: 1%"
    )

with bt3:
    st.metric(
        "5% VaR Breaches",
        f"{var_backtest['breaches_5']:,}"
    )

with bt4:
    st.metric(
        "1% VaR Breaches",
        f"{var_backtest['breaches_1']:,}"
    )

st.info(
    f"5% VaR calibration: **{calibration_5}** "
    f"(observed breach rate "
    f"{var_backtest['rate_5']:.2%} vs nominal 5%). "
    f"1% VaR calibration: **{calibration_1}** "
    f"(observed breach rate "
    f"{var_backtest['rate_1']:.2%} vs nominal 1%)."
)

# =========================================================
# REGIME VALIDATION
# =========================================================

st.subheader("Regime Validation")

st.markdown(
    "This section evaluates whether higher risk regimes correspond "
    "to larger realized and forward-looking market risk."
)

st.dataframe(
    regime_validation.style.format({
        "avg_abs_return": "{:.3f}%",
        "avg_vol_30d": "{:.3f}%",
        "avg_forward_7d_abs_return": "{:.3f}%",
        "avg_forward_7d_vol": "{:.3f}%",
        "avg_es_5": "{:.3f}%",
        "observations": "{:,.0f}",
    }),
    use_container_width=True
)


regime_plot_df = (
    regime_validation
    .reset_index()
    .rename(columns={
        "risk_regime": "Risk Regime",
        "avg_forward_7d_vol": "Forward 7D Volatility"
    })
)

fig_forward = px.bar(
    regime_plot_df,
    x="Risk Regime",
    y="Forward 7D Volatility",
    title="Forward 7-Day Volatility by Current Risk Regime"
)

apply_theme(fig_forward)

st.plotly_chart(
    fig_forward,
    use_container_width=True
)


if regime_validation_status == "Strong Separation":
    st.success(
        "Regime validation: Strong Separation — higher current "
        "risk regimes are associated with higher forward 7-day volatility."
    )

elif regime_validation_status == "Mixed Separation":
    st.warning(
        "Regime validation: Mixed Separation — the regime system "
        "captures meaningful risk differences, but forward volatility "
        "is not strictly monotonic across all regimes."
    )

else:
    st.info(
        "Regime validation: insufficient observations for a reliable comparison."
    )

# =========================================================
# REGIME SECTION
# =========================================================
st.subheader("Risk Regime Classification")

col_left, col_right = st.columns(2)

with col_left:
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df["regime_code"],
        mode="lines",
        line_shape="hv",
        name="Risk Regime"
    ))
    fig_regime.update_layout(
        title="Risk Regime Over Time",
        yaxis=dict(
            title="Regime",
            tickmode="array",
            tickvals=[1, 2, 3, 4],
            ticktext=["Low", "Moderate", "High", "Extreme"]
        ),
        xaxis_title="Date"
    )
    apply_theme(fig_regime)
    st.plotly_chart(fig_regime, use_container_width=True)

with col_right:
    regime_counts = data["risk_regime"].value_counts().reindex(
        ["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"]
    ).fillna(0)
    regime_bar = px.bar(
        x=regime_counts.index,
        y=regime_counts.values,
        labels={"x": "Regime", "y": "Count"},
        title="Historical Regime Counts"
    )
    apply_theme(regime_bar)
    st.plotly_chart(regime_bar, use_container_width=True)


# =========================================================
# REGIME SUMMARY TABLE
# =========================================================
st.subheader("Regime Summary")

regime_summary = data.groupby("risk_regime").agg(
    avg_return=("return_pct", "mean"),
    return_vol=("return_pct", "std"),
    avg_vol_30d=("vol_30d", "mean"),
    avg_var_5=("VaR_5", "mean"),
    avg_es_5=("ES_5", "mean"),
    avg_risk_score=("risk_score", "mean"),
    observations=("risk_regime", "count")
).reindex(["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"])

st.dataframe(
    regime_summary.style.format({
        "avg_return": "{:.3f}",
        "return_vol": "{:.3f}",
        "avg_vol_30d": "{:.3f}",
        "avg_var_5": "{:.3f}",
        "avg_es_5": "{:.3f}",
        "avg_risk_score": "{:.3f}",
        "observations": "{:,.0f}",
    }),
    use_container_width=True
)


# =========================================================
# RECENT STRESS EVENTS
# =========================================================
st.subheader("Recent Stress Events")

recent_events = data.loc[
    data["exceed_5"],
    ["Close", "return_pct", "VaR_5", "ES_5", "risk_regime"]
].tail(15)

recent_events = recent_events.rename(columns={
    "Close": "BTC Close",
    "return_pct": "Actual Return (%)",
    "VaR_5": "VaR 5% (%)",
    "ES_5": "ES 5% (%)",
    "risk_regime": "Risk Regime"
})

st.dataframe(
    recent_events.style.format({
        "BTC Close": "{:,.0f}",
        "Actual Return (%)": "{:.3f}",
        "VaR 5% (%)": "{:.3f}",
        "ES 5% (%)": "{:.3f}",
    }),
    use_container_width=True
)


# =========================================================
# FOOTER / METHOD NOTE
# =========================================================
st.markdown("---")
st.markdown(
    """
**Methodology Notes**

- Historical daily BTC prices provide the long-run analytical base.
- When the historical dataset is current, the latest CoinGecko price is inserted as a provisional daily observation.
- Daily returns are recalculated after the live observation is incorporated.
- Rolling volatility is estimated over 30-day and 60-day windows.
- VaR and Expected Shortfall are estimated from 250-day rolling historical returns.
- Risk regimes are derived from a causal composite percentile score using volatility, VaR, and Expected Shortfall; each observation is ranked only against information available up to that date.
- If historical data are stale, live market data remain visible but are excluded from risk calculations to prevent invalid multi-day returns from being treated as one-day returns.
- Short-term risk trend compares the current composite risk score with its level seven observations earlier.
- Volatility state, VaR breach status, and Expected Shortfall trend provide a rule-based interpretation layer over the underlying quantitative risk metrics.
"""
)
