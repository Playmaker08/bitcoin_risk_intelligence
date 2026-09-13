import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from market_data import get_live_btc_market_data

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

    # VaR and ES are negative:
    # more negative = more downside risk
    risk_df["var_score"] = -risk_df["VaR_5"]
    risk_df["es_score"] = -risk_df["ES_5"]

    risk_df = risk_df.dropna().copy()

    # -----------------------------------------------------
    # HISTORICAL PERCENTILE RANKS
    # -----------------------------------------------------

    risk_df["vol_pct"] = (
        risk_df["vol_score"]
        .rank(pct=True)
    )

    risk_df["var_pct"] = (
        risk_df["var_score"]
        .rank(pct=True)
    )

    risk_df["es_pct"] = (
        risk_df["es_score"]
        .rank(pct=True)
    )

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
# BUILD ANALYTICS PIPELINE
# =========================================================

historical_data = load_historical_data(DATA_PATH)

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


# Backfill missing historical observations
updated_history = backfill_missing_history(
    historical_data,
    gap_days
)


# Fetch live market snapshot
live_market = get_live_btc_market_data()


# Add today's provisional observation
combined_data = append_live_observation(
    updated_history,
    live_market
)


# Run risk engine
data = calculate_risk_metrics(
    combined_data
)


live_risk_enabled = (
    live_market.get("status") == "live"
    and len(updated_history) > len(historical_data)
)

if historical_data.empty:
    st.error("No historical data available.")
    st.stop()

historical_end = historical_data.index.max().normalize()

today_utc = (
    pd.Timestamp.utcnow()
    .tz_localize(None)
    .normalize()
)

gap_days = (today_utc - historical_end).days

from market_data import (
    get_live_btc_market_data,
    get_historical_btc_data,
)

def backfill_missing_history(
    historical_df: pd.DataFrame,
    gap_days: int
) -> pd.DataFrame:

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
        combined[~combined.index.duplicated(
            keep="last"
        )]
        .sort_index()
    )

    combined["return_pct"] = (
        combined["Close"]
        .pct_change()
        * 100
    )

    return combined
# Fetch current market observation
live_market = get_live_btc_market_data()


# ---------------------------------------------------------
# DATA FRESHNESS CHECK
# ---------------------------------------------------------

historical_end = historical_data.index.max()

updated_end = (
    updated_history.index
    .max()
    .normalize()
)

remaining_gap = (
    today_utc - updated_end
).days

live_risk_enabled = (
    live_market.get("status") == "live"
    and remaining_gap <= 1
)

if live_market.get("last_updated") is not None:
    live_date = (
        pd.Timestamp(live_market["last_updated"])
        .tz_localize(None)
        .normalize()
    )
else:
    live_date = (
        pd.Timestamp.utcnow()
        .tz_localize(None)
        .normalize()
    )


gap_days = (live_date - historical_end.normalize()).days


# ---------------------------------------------------------
# HYBRID HISTORICAL + LIVE DATA
# ---------------------------------------------------------

if gap_days <= 1:
    combined_data = append_live_observation(
        historical_data,
        live_market
    )

    live_risk_enabled = (
        live_market.get("status") == "live"
    )

else:
    # Do not treat a stale historical close as yesterday's close
    combined_data = historical_data.copy()
    live_risk_enabled = False


# Run risk analytics
data = calculate_risk_metrics(combined_data)


if data.empty:
    st.error("No data available after risk calculations.")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Dashboard Controls")
CoinGecko Feed: LIVE
Historical Backfill: CURRENT
Risk Engine: LIVE
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
    f"Historical data through: "
    f"{historical_end.strftime('%Y-%m-%d')}"
)

display_df = filter_by_range(data, view_range)
latest = data.iloc[-1]

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
- Risk regimes are derived from a composite percentile score using volatility, VaR, and Expected Shortfall.
- If historical data are stale, live market data remain visible but are excluded from risk calculations to prevent invalid multi-day returns from being treated as one-day returns.
"""
)
