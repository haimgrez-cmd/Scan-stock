"""
quant_portfolio_agent.py
=========================
סוכן "סימונס" — סורק מועמדים חדשים + מנהל את תיק ההחזקות הקיים לפי כללים מכניים.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_STABLE = "https://financialmodelingprep.com/stable"
PORTFOLIO_FILE = Path("portfolio_state.json")

EXIT_GAP_PCT = -2.0
TRIM_GAP_PCT = 8.0
MAX_POSITIONS = 5
MAX_WEIGHT_PER_POSITION = 0.20

DEFAULT_UNIVERSE = [
    "LAD", "SYBT", "IBCP", "KNSA", "WKC", "MU", "PLTR", "NBIX",
    "AMZN", "MSFT", "V", "AAPL",
]


@dataclass
class Holding:
    ticker: str
    name: str
    price: float
    target: float
    weight: float
    entered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def gap_pct(self) -> float:
        if self.price == 0:
            return 0.0
        return (self.target - self.price) / self.price * 100

    @property
    def status(self) -> str:
        if self.gap_pct <= EXIT_GAP_PCT:
            return "EXIT"
        if self.gap_pct < TRIM_GAP_PCT:
            return "TRIM"
        return "HOLD"


def _api_key() -> str:
    key = st.secrets.get("FMP_API_KEY", os.environ.get("FMP_API_KEY", ""))
    if not key:
        st.error("חסר FMP_API_KEY ב-secrets או במשתני הסביבה.")
        st.stop()
    return key


@st.cache_data(ttl=3600)
def fetch_quote(ticker: str):
    url = f"{FMP_BASE}/quote/{ticker}"
    try:
        r = requests.get(url, params={"apikey": _api_key()}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=3600)
def fetch_price_target(ticker: str):
    url = f"{FMP_STABLE}/price-target-consensus"
    try:
        r = requests.get(url, params={"symbol": ticker, "apikey": _api_key()}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            return data[0].get("targetConsensus") or data[0].get("targetMedian")
    except requests.RequestException:
        pass
    return None


def scan_universe(tickers):
    candidates = []
    progress = st.progress(0.0, text="סורק מועמדים...")
    for i, t in enumerate(tickers):
        quote = fetch_quote(t)
        target = fetch_price_target(t)
        if quote and target:
            candidates.append(
                Holding(ticker=t, name=quote.get("name", t), price=quote["price"], target=target, weight=0.0)
            )
        progress.progress((i + 1) / len(tickers), text=f"סורק מועמדים... {t}")
    progress.empty()
    return candidates


def load_portfolio():
    if not PORTFOLIO_FILE.exists():
        return []
    raw = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    return [Holding(**h) for h in raw]


def save_portfolio(holdings):
    PORTFOLIO_FILE.write_text(
        json.dumps([asdict(h) for h in holdings], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_decision_engine(current, candidates):
    exits = [h for h in current if h.status == "EXIT"]
    survivors = [h for h in current if h.status != "EXIT"]
    open_slots = MAX_POSITIONS - len(survivors)
    current_tickers = {h.ticker for h in current}
    fresh = [c for c in candidates if c.ticker not in current_tickers and c.status == "HOLD"]
    fresh.sort(key=lambda h: h.gap_pct, reverse=True)
    entries = fresh[:max(0, open_slots)]
    final_count = len(survivors) + len(entries)
    equal_weight = round(100 / final_count, 1) if final_count else 0.0
    for h in survivors:
        h.weight = equal_weight
    for h in entries:
        h.weight = equal_weight
    return {"exits": exits, "entries": entries, "survivors": survivors, "equal_weight": equal_weight}


def main():
    st.set_page_config(page_title="סוכן תיק — סימונס", layout="wide")
    st.markdown(
        """<style>html, body, [class*="css"] { direction: rtl; text-align: right; }</style>""",
        unsafe_allow_html=True,
    )
    st.title("🎯 סוכן תיק כמותי — סגנון סימונס")
    st.caption("סורק מועמדים חדשים ומנהל את הפוזיציות הקיימות לפי פער מחיר-יעד בלבד.")

    current = load_portfolio()
    if not current:
        st.info("אין תיק קיים. נטען תיק ברירת מחדל בלחיצת הכפתור.")
        if st.button("טען תיק התחלתי לדוגמה"):
            defaults = [
                Holding("LAD", "Lithia Motors", 372, 396, 25),
                Holding("SYBT", "Stock Yards Bancorp", 81.5, 77.25, 25),
                Holding("IBCP", "Independent Bank Corp", 38.5, 39.4, 25),
                Holding("KNSA", "Kiniksa Pharmaceuticals", 75, 88, 25),
            ]
            save_portfolio(defaults)
            st.rerun()
        return

    universe = st.multiselect("יקום סריקה", options=DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)

    if st.button("🔍 הרץ סריקה + עדכן החלטות", type="primary"):
        with st.spinner("מושך מחירים ויעדי אנליסטים מ-FMP..."):
            for h in current:
                q = fetch_quote(h.ticker)
                t = fetch_price_target(h.ticker)
                if q:
                    h.price = q["price"]
                if t:
                    h.target = t
            candidates = scan_universe(universe)

        plan = run_decision_engine(current, candidates)
        st.subheader("📋 תוכנית פעולה")

        if plan["exits"]:
            st.error("**לסגור פוזיציה:**")
            for h in plan["exits"]:
                st.write(f"🔴 {h.ticker} — מחיר {h.price:.2f}$ מול יעד {h.target:.2f}$
