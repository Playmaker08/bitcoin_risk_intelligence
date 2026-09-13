import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone


COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_btc_data(days: int) -> pd.DataFrame:
    """
    Fetch historical BTC/USD market data from CoinGecko.
    """

    url = (
        "https://api.coingecko.com/api/v3/coins/"
        "bitcoin/market_chart"
    )

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        payload = response.json()

        prices = payload.get("prices", [])

        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(
            prices,
            columns=["timestamp", "Close"]
        )

        df["date"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        df["date"] = (
            df["date"]
            .dt.tz_localize(None)
            .dt.normalize()
        )

        df = (
            df[["date", "Close"]]
            .drop_duplicates("date")
            .sort_values("date")
        )

        return df

    except Exception:
        return pd.DataFrame()
    
@st.cache_data(ttl=300, show_spinner=False)
def get_live_btc_market_data():
    """
    Fetch latest Bitcoin market data from CoinGecko.

    Cached for 5 minutes to reduce unnecessary API calls.
    """

    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }

    try:
        response = requests.get(
            COINGECKO_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()

        if "bitcoin" not in payload:
            raise ValueError("Bitcoin data not found in API response.")

        btc = payload["bitcoin"]

        timestamp = btc.get("last_updated_at")

        if timestamp is not None:
            updated_time = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc
            )
        else:
            updated_time = None

        return {
            "price": btc.get("usd"),
            "market_cap": btc.get("usd_market_cap"),
            "volume_24h": btc.get("usd_24h_vol"),
            "change_24h": btc.get("usd_24h_change"),
            "last_updated": updated_time,
            "status": "live",
            "error": None,
        }

    except Exception as e:
        return {
            "price": None,
            "market_cap": None,
            "volume_24h": None,
            "change_24h": None,
            "last_updated": None,
            "status": "unavailable",
            "error": str(e),
        }
