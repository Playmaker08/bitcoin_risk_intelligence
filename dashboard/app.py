import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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
.metric-card {
    padding: 14px 16px;
    border-radius: 14px;
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
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


@st.cache_data(show_spinner=False)
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("date").sort_index()

    btc_daily = df["Close"].resample("1D").last().dropna().to_frame()
    btc_daily["return_pct"] = btc_daily["Close"].pct_change() * 100
    btc_daily = btc_daily.dropna()

    # Rolling market metrics
    btc_daily["vol_30d"] = btc_daily["return_pct"].rolling(30).std()
    btc_daily["vol_60d"] = btc_daily["return_pct"].rolling(60).std()

    btc_daily["VaR_5"] = btc_daily["return_pct"].rolling(250).apply(
        lambda x: historical_var(x, 0.05), raw=False
    )
    btc_daily["VaR_1"] = btc_daily["return_pct"].rolling(250).apply(
        lambda x: historical_var(x, 0.01), raw=False
    )

    btc_daily["ES_5"] = btc_daily["return_pct"].rolling(250).apply(
        lambda x: historical_es(x, 0.05), raw=False
    )
    btc_daily["ES_1"] = btc_daily["return_pct"].rolling(250).apply(
        lambda x: historical_es(x, 0.01), raw=False
    )

    # Regime inputs
    btc_daily["vol_score"] = btc_daily["vol_30d"]
    btc_daily["var_score"] = -btc_daily["VaR_5"]
    btc_daily["es_score"] = -btc_daily["ES_5"]

    regime_df = btc_daily.dropna().copy()

    regime_df["vol_pct"] = regime_df["vol_score"].rank(pct=True)
    regime_df["var_pct"] = regime_df["var_score"].rank(pct=True)
    regime_df["es_pct"] = regime_df["es_score"].rank(pct=True)

    regime_df["risk_score"] = (
        0.4 * regime_df["vol_pct"] +
        0.3 * regime_df["var_pct"] +
        0.3 * regime_df["es_pct"]
    )

    regime_df["risk_regime"] = regime_df["risk_score"].apply(classify_regime)

    regime_map = {
        "Low Risk": 1,
        "Moderate Risk": 2,
        "High Risk": 3,
        "Extreme Risk": 4
    }
    regime_df["regime_code"] = regime_df["risk_regime"].map(regime_map)

    # Exceedances
    regime_df["exceed_5"] = regime_df["return_pct"] < regime_df["VaR_5"]
    regime_df["exceed_1"] = regime_df["return_pct"] < regime_df["VaR_1"]

    return regime_df


def filter_by_range(df: pd.DataFrame, range_option: str) -> pd.DataFrame:
    if range_option == "Full History":
        return df.copy()
    if range_option == "Last 5Y":
        return df.last("1825D").copy()
    if range_option == "Last 2Y":
        return df.last("730D").copy()
    if range_option == "Last 1Y":
        return df.last("365D").copy()
    if range_option == "Last 6M":
        return df.last("180D").copy()
    return df.copy()


def latest_change(series: pd.Series, periods: int = 1) -> float | None:
    if len(series.dropna()) <= periods:
        return None
    return series.iloc[-1] - series.iloc[-1 - periods]


# =========================================================
# LOAD DATA
# =========================================================
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "btc_daily.csv"

@st.cache_data(show_spinner=False)
def load_and_prepare_data(csv_path):
    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Nếu file đã có return_pct rồi thì giữ nguyên
    # nếu chưa có thì tính lại
    if "return_pct" not in df.columns:
        df["return_pct"] = df["Close"].pct_change() * 100

    # Rolling market metrics
    df["vol_30d"] = df["return_pct"].rolling(30).std()
    df["vol_60d"] = df["return_pct"].rolling(60).std()

    df["VaR_5"] = df["return_pct"].rolling(250).quantile(0.05)
    df["VaR_1"] = df["return_pct"].rolling(250).quantile(0.01)

    def historical_es(x, alpha=0.05):
        var = x.quantile(alpha)
        tail = x[x <= var]
        return tail.mean() if len(tail) > 0 else None

    df["ES_5"] = df["return_pct"].rolling(250).apply(
        lambda x: historical_es(pd.Series(x), 0.05), raw=False
    )
    df["ES_1"] = df["return_pct"].rolling(250).apply(
        lambda x: historical_es(pd.Series(x), 0.01), raw=False
    )

    df["vol_score"] = df["vol_30d"]
    df["var_score"] = -df["VaR_5"]
    df["es_score"] = -df["ES_5"]

    regime_df = df.dropna().copy()

    regime_df["vol_pct"] = regime_df["vol_score"].rank(pct=True)
    regime_df["var_pct"] = regime_df["var_score"].rank(pct=True)
    regime_df["es_pct"] = regime_df["es_score"].rank(pct=True)

    regime_df["risk_score"] = (
        0.4 * regime_df["vol_pct"] +
        0.3 * regime_df["var_pct"] +
        0.3 * regime_df["es_pct"]
    )

    def classify_regime(score):
        if score < 0.50:
            return "Low Risk"
        elif score < 0.80:
            return "Moderate Risk"
        elif score < 0.95:
            return "High Risk"
        return "Extreme Risk"

    regime_df["risk_regime"] = regime_df["risk_score"].apply(classify_regime)

    regime_map = {
        "Low Risk": 1,
        "Moderate Risk": 2,
        "High Risk": 3,
        "Extreme Risk": 4
    }
    regime_df["regime_code"] = regime_df["risk_regime"].map(regime_map)

    regime_df["exceed_5"] = regime_df["return_pct"] < regime_df["VaR_5"]
    regime_df["exceed_1"] = regime_df["return_pct"] < regime_df["VaR_1"]

    return regime_df


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

display_df = filter_by_range(DATA_PATH, view_range)
latest = data.iloc[-1]

selected_var = "VaR_5" if tail_level == "5%" else "VaR_1"
selected_es = "ES_5" if tail_level == "5%" else "ES_1"
selected_exceed = "exceed_5" if tail_level == "5%" else "exceed_1"


# =========================================================
# TITLE
# =========================================================
st.title("Bitcoin Risk Intelligence Dashboard")
st.markdown(
    "<div class='section-note'>A dynamic risk-monitoring dashboard built from historical Bitcoin price data, volatility analysis, VaR, Expected Shortfall, and regime classification.</div>",
    unsafe_allow_html=True
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
        "BTC Price",
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

recent_events = data.loc[data["exceed_5"], ["Close", "return_pct", "VaR_5", "ES_5", "risk_regime"]].tail(15)
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
- Daily BTC prices are created by resampling historical minute-level data.
- Rolling volatility is computed using 30-day and 60-day realized volatility.
- VaR and Expected Shortfall are estimated from 250-day rolling historical returns.
- Regime classification is based on a composite risk score using volatility, VaR, and ES percentiles.
"""
)
