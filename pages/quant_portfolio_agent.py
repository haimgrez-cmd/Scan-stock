"""
quant_portfolio_agent.py
=========================
Pure statistical mean-reversion agent (Simons-style).
No analyst targets, no fundamentals, no human opinion - only price z-score.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio_state.json")

LOOKBACK_DAYS = 20        # window for rolling mean/std
Z_ENTRY = -2.0             # price at least 2 std below its own 20-day mean -> candidate
Z_EXIT = -0.5              # once z recovers above this, the anomaly is gone -> exit
MAX_POSITIONS = 5
UNIVERSE_SIZE = 100
BATCH_SIZE = 50


@dataclass
class Holding:
    ticker: str
    name: str
    price: float
    z_score: float
    weight: float
    entered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def status(self) -> str:
        if self.z_score > Z_EXIT:
            return "EXIT"     # reverted to normal - thesis played out
        if self.z_score > Z_ENTRY:
            return "TRIM"     # reverting, signal weakening
        return "HOLD"         # still statistically anomalous


@st.cache_data(ttl=86400)
def get_sp500_universe(n: int) -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 50:
            return tickers[:n]
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")
    # fallback - small manual list
    fallback = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "XOM",
        "MRK", "ABBV", "BAC", "AVGO", "LLY", "TMO", "CSCO", "MCD", "ACN", "ABT",
    ]
    return fallback[:n]


def calc_zscore_batch(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """Returns {ticker: (last_price, z_score)} using rolling 20-day mean/std."""
    results = {}
    try:
        raw = yf.download(
            tickers, period="90d", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        logger.error(e)
        return results

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            df = raw[t].copy() if is_multi else raw.copy()
            c = df["Close"].dropna()
            if len(c) < LOOKBACK_DAYS + 5:
                continue
            window = c.tail(LOOKBACK_DAYS)
            mean = float(window.mean())
            std = float(window.std())
            if std == 0 or np.isnan(std):
                continue
            last = float(c.iloc[-1])
            z = (last - mean) / std
            if np.isnan(z):
                continue
            results[t] = (round(last, 2), round(z, 2))
        except Exception as e:
            logger.warning(f"{t}: {e}")
    return results


def scan_universe(tickers: list[str]) -> list[Holding]:
    candidates = []
    progress = st.progress(0.0, text="Scanning universe for z-score anomalies...")
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    for i, start in enumerate(range(0, len(tickers), BATCH_SIZE)):
        batch = tickers[start:start + BATCH_SIZE]
        scores = calc_zscore_batch(batch)
        for t, (price, z) in scores.items():
            candidates.append(Holding(ticker=t, name=t, price=price, z_score=z, weight=0.0))
        progress.progress((i + 1) / total_batches, text=f"Scanned {min(start + BATCH_SIZE, len(tickers))}/{len(tickers)}")
    progress.empty()
    return candidates


def load_portfolio() -> list[Holding]:
    if not PORTFOLIO_FILE.exists():
        return []
    raw = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    return [Holding(**h) for h in raw]


def save_portfolio(holdings: list[Holding]) -> None:
    PORTFOLIO_FILE.write_text(
        json.dumps([asdict(h) for h in holdings], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_decision_engine(current: list[Holding], candidates: list[Holding]) -> dict:
    exits = [h for h in current if h.status == "EXIT"]
    survivors = [h for h in current if h.status != "EXIT"]
    open_slots = MAX_POSITIONS - len(survivors)

    current_tickers = {h.ticker for h in current}
    # only take candidates that are still deeply anomalous (HOLD, i.e. z <= Z_ENTRY)
    fresh = [c for c in candidates if c.ticker not in current_tickers and c.status == "HOLD"]
    fresh.sort(key=lambda h: h.z_score)  # most negative z first = most anomalous
    entries = fresh[:max(0, open_slots)]

    final_count = len(survivors) + len(entries)
    equal_weight = round(100 / final_count, 1) if final_count else 0.0
    for h in survivors:
        h.weight = equal_weight
    for h in entries:
        h.weight = equal_weight

    return {"exits": exits, "entries": entries, "survivors": survivors, "equal_weight": equal_weight}


def main():
    st.set_page_config(page_title="Quant Portfolio Agent", layout="wide")
    st.markdown(
        "<style>html, body, [class*='css'] { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Quant Portfolio Agent - Pure Statistical (Simons Style)")
    st.caption(
        f"Entry: price <= {Z_ENTRY} std below its own {LOOKBACK_DAYS}-day mean. "
        f"Exit: z-score reverts above {Z_EXIT}. No analyst targets, no fundamentals."
    )

    current = load_portfolio()

    universe_n = st.slider("Universe size (S&P 500 slice)", 20, 200, UNIVERSE_SIZE, step=10)
    universe = get_sp500_universe(universe_n)
    st.caption(f"Scanning {len(universe)} tickers from the S&P 500.")

    if not current:
        st.info("No existing portfolio yet. Run a scan below to find the first candidates.")

    if st.button("Run scan + update decisions", type="primary"):
        with st.spinner("Fetching price history and computing z-scores..."):
            if current:
                fresh_scores = calc_zscore_batch([h.ticker for h in current])
                for h in current:
                    if h.ticker in fresh_scores:
                        h.price, h.z_score = fresh_scores[h.ticker]
            candidates = scan_universe(universe)

        plan = run_decision_engine(current, candidates)
        st.subheader("Action plan")

        if plan["exits"]:
            st.error("Close position (z-score reverted, anomaly gone):")
            for h in plan["exits"]:
                st.write(f"EXIT {h.ticker} - price {h.price:.2f}, z={h.z_score:+.2f}")

        if plan["entries"]:
            st.success("Open new position (statistically anomalous):")
            for h in plan["entries"]:
                st.write(f"NEW {h.ticker} - price {h.price:.2f}, z={h.z_score:+.2f}")

        if not plan["exits"] and not plan["entries"]:
            st.info("No change suggested.")

        st.session_state["pending_plan"] = plan

    if "pending_plan" in st.session_state:
        if st.button("Confirm and update portfolio"):
            plan = st.session_state["pending_plan"]
            new_portfolio = plan["survivors"] + plan["entries"]
            save_portfolio(new_portfolio)
            del st.session_state["pending_plan"]
            st.success("Portfolio updated.")
            st.rerun()

    if current:
        st.divider()
        st.subheader("Current portfolio status")
        for h in current:
            col1, col2, col3 = st.columns([2, 2, 2])
            col1.write(f"[{h.status}] {h.ticker}")
            col2.write(f"Price: {h.price:.2f}")
            col3.write(f"Z-score: {h.z_score:+.2f}")


if __name__ == "__main__":
    main()
