"""
Net-Net Stock Scanner - Graham Style
yקום: NASDAQ API (כל המניות) + SEC EDGAR כגיבוי
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime
from typing import Optional

st.set_page_config(page_title="Net-Net Scanner", page_icon="💎", layout="wide")

st.markdown("""
<style>
.metric-box {
    background: linear-gradient(135deg, #1a1f2e, #252d3d);
    border: 1px solid #2d3748; border-radius: 8px;
    padding: 16px; text-align: center; margin-bottom: 8px;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #48bb78; }
.metric-lbl { font-size: 0.78rem; color: #a0aec0; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Net-Net Stock Scanner — Graham Style")
st.caption("Price < 2/3 × NCAV per Share  |  NCAV = Current Assets − Total Liabilities")

# ── UNIVERSE OPTIONS (ללא אמוג'י במזהה) ──────────────────────────────────────
UNIVERSE_LABELS = [
    "Micro Cap (< $300M)",
    "Small Cap ($300M - $2B)",
    "Micro + Small Cap",
    "S&P 500",
    "Custom Tickers",
]
UNIVERSE_KEYS = ["micro", "small", "micro_small", "sp500", "custom"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    uni_idx = st.selectbox(
        "Universe",
        options=range(len(UNIVERSE_LABELS)),
        format_func=lambda i: UNIVERSE_LABELS[i],
        key="netnet_universe_idx",
    )
    universe_key = UNIVERSE_KEYS[uni_idx]

    custom_input = ""
    if universe_key == "custom":
        custom_input = st.text_area(
            "Tickers (comma / space / newline)",
            placeholder="AAPL, MSFT, GOOG",
            key="netnet_custom_input",
        )

    max_scan = st.number_input(
        "Max tickers to scan",
        min_value=50, max_value=3000, value=300, step=50,
        key="netnet_max_scan",
    )

    st.markdown("---")
    st.subheader("Filters")

    price_min = st.number_input(
        "Min price ($)", min_value=0.0, value=0.5, step=0.5,
        key="netnet_price_min",
    )
    price_max = st.number_input(
        "Max price ($)", min_value=0.0, value=200.0, step=10.0,
        key="netnet_price_max",
    )
    ncav_max = st.slider(
        "Max P/NCAV (Graham: 0.67)",
        min_value=0.10, max_value=1.50, value=0.67, step=0.01,
        key="netnet_ncav_max",
    )
    show_con = st.checkbox(
        "Show Conservative NCAV", value=True,
        key="netnet_show_con",
    )
    excl_fin = st.checkbox(
        "Exclude banks / insurance", value=True,
        key="netnet_excl_fin",
    )

    st.markdown("---")
    delay_ms = st.slider(
        "Delay between tickers (ms)",
        min_value=50, max_value=400, value=120, step=25,
        key="netnet_delay_ms",
    )

# ── DATA SOURCES ──────────────────────────────────────────────────────────────
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nasdaq.com/",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_nasdaq_all() -> pd.DataFrame:
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&download=true"
    )
    r = requests.get(url, headers=NASDAQ_HEADERS, timeout=25)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    df = pd.DataFrame(rows)

    def parse_cap(v):
        if not v or str(v).strip() in ("", "N/A"):
            return None
        s = str(v).strip().upper().replace(",", "")
        try:
            if s.endswith("T"): return float(s[:-1]) * 1e12
            if s.endswith("B"): return float(s[:-1]) * 1e9
            if s.endswith("M"): return float(s[:-1]) * 1e6
            return float(s)
        except Exception:
            return None

    df["cap"] = df["marketCap"].apply(parse_cap)
    df = df.rename(columns={"symbol": "ticker"})
    df = df[df["ticker"].notna() & (df["ticker"] != "")]
    df = df[~df["ticker"].str.contains(r"[/\^~+]", na=False, regex=True)]
    df = df[df["ticker"].str.len() <= 5]
    return df[["ticker", "cap"]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_tickers() -> list:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "research@example.com"},
        timeout=15,
    )
    r.raise_for_status()
    return [v["ticker"] for v in r.json().values() if v.get("ticker")]


@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500() -> list:
    df = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def get_tickers(ukey: str, custom: str, max_n: int) -> list:
    if ukey == "sp500":
        return load_sp500()
    if ukey == "custom":
        parts = custom.replace(",", " ").split()
        return [t.strip().upper() for t in parts if t.strip()]

    # NASDAQ API
    try:
        df_all = load_nasdaq_all()
    except Exception as e:
        st.warning(f"NASDAQ API failed ({e}) — falling back to SEC EDGAR")
        import random
        t = load_sec_tickers()
        random.shuffle(t)
        return t[:max_n]

    has_cap = df_all["cap"].notna()
    if ukey == "micro":
        mask = has_cap & (df_all["cap"] < 300e6)
    elif ukey == "small":
        mask = has_cap & (df_all["cap"] >= 300e6) & (df_all["cap"] < 2e9)
    else:
        mask = has_cap & (df_all["cap"] < 2e9)

    result = df_all[mask].sort_values("cap")["ticker"].tolist()
    total = len(result)
    result = result[:max_n]
    st.info(f"Found {total:,} tickers in category — scanning first {len(result)} (smallest first)")
    return result


# ── ANALYZER ──────────────────────────────────────────────────────────────────
FIN_SECTORS = {"Financial Services", "Financial", "Banks"}
FIN_INDS = {"Bank", "Insur", "REIT", "Mortgage", "Thrift", "Brokerage"}


def safe(info: dict, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def bval(bs, col, *names) -> Optional[float]:
    for n in names:
        if n in bs.index:
            v = bs.loc[n, col]
            if pd.notna(v):
                return float(v)
    return None


def analyze(ticker: str) -> Optional[dict]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet
        if bs is None or bs.empty:
            return None

        price = safe(info, "currentPrice", "regularMarketPrice", "previousClose")
        if not price or price <= 0:
            return None
        if not (price_min <= price <= price_max):
            return None

        if excl_fin:
            sec = info.get("sector", "")
            ind = info.get("industry", "")
            if sec in FIN_SECTORS or any(k in ind for k in FIN_INDS):
                return None

        col = bs.columns[0]

        ca = bval(bs, col, "Current Assets", "Total Current Assets",
                  "CurrentAssets", "TotalCurrentAssets")
        tl = bval(bs, col,
                  "Total Liabilities Net Minority Interest",
                  "TotalLiabilitiesNetMinorityInterest",
                  "Total Liabilities", "TotalLiabilities")

        if ca is None or tl is None:
            return None

        shares = safe(info, "sharesOutstanding", "impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        ncav = ca - tl
        ncav_ps = ncav / shares
        if ncav_ps <= 0:
            return None

        p_ncav = price / ncav_ps
        if p_ncav > ncav_max:
            return None

        # Conservative NCAV
        con_ps = None
        if show_con:
            cash = bval(bs, col, "Cash And Cash Equivalents", "Cash",
                        "CashAndCashEquivalents",
                        "Cash Cash Equivalents And Short Term Investments") or 0.0
            ar   = bval(bs, col, "Net Receivables", "Receivables",
                        "Accounts Receivable", "AccountsReceivable") or 0.0
            inv  = bval(bs, col, "Inventory", "Inventories") or 0.0
            other = max(0.0, ca - cash - ar - inv)
            con = cash * 1.0 + ar * 0.75 + inv * 0.5 + other * 0.25 - tl
            con_ps = con / shares

        cl = bval(bs, col, "Current Liabilities", "Total Current Liabilities",
                  "CurrentLiabilities", "TotalCurrentLiabilities")
        cr = ca / cl if cl and cl > 0 else None
        mc = safe(info, "marketCap")

        row = {
            "Ticker":          ticker,
            "Name":            info.get("shortName", ticker),
            "Sector":          info.get("sector", ""),
            "Price":           round(price, 2),
            "NCAV/Share":      round(ncav_ps, 2),
            "P/NCAV":          round(p_ncav, 3),
            "Margin of Safety": round((1 - p_ncav) * 100, 1),
            "CA ($M)":         round(ca / 1e6, 1),
            "TL ($M)":         round(tl / 1e6, 1),
            "NCAV ($M)":       round(ncav / 1e6, 1),
            "Curr.Ratio":      round(cr, 2) if cr else None,
            "Mkt Cap ($M)":    round(mc / 1e6, 1) if mc else None,
            "P/B":             safe(info, "priceToBook"),
            "Country":         info.get("country", ""),
        }
        if show_con:
            row["Con.NCAV/Share"] = round(con_ps, 2) if con_ps is not None else None
            row["P/Con.NCAV"]     = round(price / con_ps, 3) if con_ps and con_ps > 0 else None

        return row

    except Exception:
        return None


# ── SCAN BUTTON ───────────────────────────────────────────────────────────────
st.markdown("---")
run = st.button("Run Scan", type="primary", key="netnet_run_btn")

if run:
    tickers = get_tickers(universe_key, custom_input, int(max_scan))

    if not tickers:
        st.error("No tickers found. Check input.")
        st.stop()

    st.write(f"Scanning **{len(tickers)}** tickers …")
    bar   = st.progress(0)
    info  = st.empty()
    found = []

    for i, tk in enumerate(tickers):
        info.text(f"[{i+1}/{len(tickers)}]  {tk:<8}  found so far: {len(found)}")
        res = analyze(tk)
        if res:
            found.append(res)
            if len(found) >= 100:
                st.info("Reached 100 Net-Net stocks — stopping early.")
                break
        bar.progress((i + 1) / len(tickers))
        time.sleep(delay_ms / 1000)

    bar.empty()
    info.empty()

    if not found:
        st.warning("No Net-Net stocks found. Try raising Max P/NCAV or changing universe.")
        st.stop()

    df = pd.DataFrame(found).sort_values("P/NCAV")

    # ── Summary metrics ──
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, str(len(df)),                          "Net-Net stocks found"),
        (c2, f"{df['Margin of Safety'].mean():.1f}%", "Avg margin of safety"),
        (c3, f"{df.iloc[0]['Ticker']} ({df.iloc[0]['P/NCAV']})", "Cheapest (P/NCAV)"),
        (c4, str(len(tickers)),                     "Tickers scanned"),
    ]:
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader(f"Results — {len(df)} Net-Net stocks")

    # ── Color coding ──
    def color_p(val):
        try:
            v = float(val)
            if v < 0.33: return "color: #68d391; font-weight:bold"
            if v < 0.50: return "color: #f6e05e"
            return "color: #fc8181"
        except Exception:
            return ""

    # ── Column order ──
    base_cols = ["Ticker", "Name", "Sector", "Price", "NCAV/Share",
                 "P/NCAV", "Margin of Safety"]
    if show_con:
        base_cols += ["Con.NCAV/Share", "P/Con.NCAV"]
    base_cols += ["CA ($M)", "TL ($M)", "NCAV ($M)",
                  "Curr.Ratio", "Mkt Cap ($M)", "P/B", "Country"]

    styler = df[base_cols].style.format(na_rep="—", precision=2)
    # pandas >= 2.1 uses .map(); older versions use .applymap()
    try:
        styler = styler.map(color_p, subset=["P/NCAV"])
    except AttributeError:
        styler = styler.applymap(color_p, subset=["P/NCAV"])

    st.dataframe(styler, use_container_width=True, height=480)

    # ── Export ──
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "Download CSV",
        data=buf.getvalue().encode("utf-8-sig"),
        file_name=f"netnet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="netnet_download",
    )

    with st.expander("Methodology"):
        st.markdown("""
**NCAV** = Current Assets − Total Liabilities  
**Graham Criterion**: Price < 2/3 × NCAV per Share (P/NCAV < 0.67)  
**Conservative NCAV** = Cash×100% + AR×75% + Inventory×50% + OtherCA×25% − All Liabilities  

**Universe source**: NASDAQ Screener API (all NASDAQ/NYSE/AMEX listed stocks with market cap).  
**Fallback**: SEC EDGAR `company_tickers.json` (~12,000 tickers).  
Sorted by market cap ascending — smallest first, as Net-Nets mostly appear there.
        """)

st.markdown("---")
st.caption("Not investment advice. Net-Net requires broad diversification. Verify data via SEC Edgar.")
"""
Net-Net Stock Scanner — Benjamin Graham Style
==============================================
יקום מניות: NASDAQ API + SEC EDGAR → אלפי small/micro caps
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Net-Net Scanner | Graham Style",
    page_icon="💎",
    layout="wide",
)

st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #48bb78; }
    .metric-lbl { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }
    .info-box {
        background: #1a2744;
        border-left: 4px solid #4299e1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #bee3f8;
    }
    .warn-box {
        background: #2d1b00;
        border-left: 4px solid #ed8936;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #fbd38d;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("💎 Net-Net Stock Scanner")
st.markdown(
    "<div class='info-box'>"
    "מחפש מניות לפי שיטת בנג'מין גרהאם: <b>Price &lt; 2/3 × NCAV per Share</b><br>"
    "NCAV = רכוש שוטף − כל ההתחייבויות | "
    "יקום: כל המניות ב-NASDAQ/NYSE/AMEX כולל small &amp; micro cap"
    "</div>",
    unsafe_allow_html=True,
)

# ─── UNIVERSE LOADERS ────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasdaq_universe() -> pd.DataFrame:
    """
    שולף את כל המניות מה-NASDAQ Screener API.
    מחזיר DataFrame עם: ticker, company, mktcap_usd, sector, industry, exchange.
    """
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&download=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    rows = data["data"]["rows"]
    df = pd.DataFrame(rows)

    def parse_mktcap(v):
        if not v or str(v).strip() in ("", "N/A", "None"):
            return None
        v = str(v).strip().upper().replace(",", "")
        try:
            if v.endswith("T"):
                return float(v[:-1]) * 1e12
            if v.endswith("B"):
                return float(v[:-1]) * 1e9
            if v.endswith("M"):
                return float(v[:-1]) * 1e6
            return float(v)
        except Exception:
            return None

    df["mktcap_usd"] = df["marketCap"].apply(parse_mktcap)
    df = df.rename(columns={"symbol": "ticker", "name": "company"})
    df = df[df["ticker"].notna() & (df["ticker"] != "")].copy()

    # הסר ETFs / warrants / units
    df = df[~df["ticker"].str.contains(r"[/\^~\+]", na=False, regex=True)]
    df = df[df["ticker"].str.len() <= 5]

    return df[["ticker", "company", "mktcap_usd", "sector", "industry", "exchange"]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sec_tickers() -> list:
    """גיבוי: שולף את כל הטיקרים מ-SEC EDGAR (~12k חברות)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(
        url,
        headers={"User-Agent": "research@example.com"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return [v["ticker"] for v in data.values() if v.get("ticker")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list:
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ הגדרות סריקה")

UNIVERSE_OPTIONS = {
    "🔬 Micro Cap  (< $300M)":          "micro",
    "📦 Small Cap  ($300M – $2B)":      "small",
    "🔬+📦 Micro + Small Cap":          "micro_small",
    "🏛️ S&P 500":                       "sp500",
    "✏️ טיקרים ידניים":                 "custom",
}
universe_choice = st.sidebar.selectbox(
    "🌐 יקום מניות", list(UNIVERSE_OPTIONS.keys()), key="nn_universe"
)

custom_tickers_raw = ""
if universe_choice == "✏️ טיקרים ידניים":
    custom_tickers_raw = st.sidebar.text_area(
        "הכנס טיקרים (פסיק / רווח / שורה חדשה)",
        placeholder="AAPL, MSFT, GOOG ...",
        height=120,
        key="nn_custom_tickers",
    )

max_tickers = st.sidebar.number_input(
    "מקסימום מניות לסריקה",
    min_value=50, max_value=3000, value=300, step=50,
    help="~300 מניות ≈ 10 דקות. הגדל בזהירות.",
    key="nn_max_tickers",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 פילטרים")

price_min = st.sidebar.number_input(
    "מחיר מינימום ($)", value=0.5, step=0.5, min_value=0.0, key="nn_price_min"
)
price_max = st.sidebar.number_input(
    "מחיר מקסימום ($)", value=200.0, step=10.0, key="nn_price_max"
)

ncav_ratio_max = st.sidebar.slider(
    "P/NCAV מקסימום",
    min_value=0.1, max_value=1.5, value=0.67, step=0.01,
    help="Graham: 0.67 = 2/3 NCAV",
    key="nn_ncav_ratio",
)

show_conservative = st.sidebar.checkbox(
    "📊 Conservative NCAV", value=True, key="nn_conservative"
)
exclude_financials = st.sidebar.checkbox(
    "🏦 הסר בנקים/ביטוח", value=True, key="nn_excl_fin"
)

st.sidebar.markdown("---")
delay_ms = st.sidebar.slider(
    "השהיה בין טיקרים (ms)", 50, 400, 120, 25, key="nn_delay"
)

FINANCIAL_KEYWORDS_SECTOR = {"Financial Services", "Financial", "Banks"}
FINANCIAL_KEYWORDS_INDUSTRY = {"Bank", "Insur", "REIT", "Mortgage", "Thrift", "Brokerage"}


# ─── ANALYZER ────────────────────────────────────────────────────────────────
def safe_get(info, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def row_val(bs, col, *names):
    for n in names:
        if n in bs.index:
            v = bs.loc[n, col]
            if pd.notna(v):
                return float(v)
    return None


def analyze_ticker(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet

        price = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
        if not price or price <= 0:
            return None
        if not (price_min <= price <= price_max):
            return None

        if exclude_financials:
            sector = info.get("sector", "")
            industry = info.get("industry", "")
            if (sector in FINANCIAL_KEYWORDS_SECTOR or
                    any(kw in industry for kw in FINANCIAL_KEYWORDS_INDUSTRY)):
                return None

        if bs is None or bs.empty:
            return None
        col = bs.columns[0]

        current_assets = row_val(bs, col,
            "Current Assets", "Total Current Assets",
            "CurrentAssets", "TotalCurrentAssets")
        total_liabilities = row_val(bs, col,
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities", "TotalLiabilities")

        if current_assets is None or total_liabilities is None:
            return None

        shares = safe_get(info, "sharesOutstanding", "impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares
        if ncav_per_share <= 0:
            return None

        p_ncav = price / ncav_per_share
        if p_ncav > ncav_ratio_max:
            return None

        # Conservative NCAV
        con_ncav_ps = None
        if show_conservative:
            cash = row_val(bs, col,
                "Cash And Cash Equivalents", "Cash",
                "CashAndCashEquivalents",
                "Cash Cash Equivalents And Short Term Investments") or 0.0
            ar = row_val(bs, col,
                "Net Receivables", "Receivables",
                "Accounts Receivable", "AccountsReceivable") or 0.0
            inv = row_val(bs, col, "Inventory", "Inventories") or 0.0
            other_ca = max(0.0, current_assets - cash - ar - inv)
            con_ncav = (
                cash * 1.0
                + ar * 0.75
                + inv * 0.5
                + other_ca * 0.25
                - total_liabilities
            )
            con_ncav_ps = con_ncav / shares if shares else None

        mktcap = safe_get(info, "marketCap")
        mktcap_m = mktcap / 1e6 if mktcap else None

        cl = row_val(bs, col,
            "Current Liabilities", "Total Current Liabilities",
            "CurrentLiabilities", "TotalCurrentLiabilities")
        current_ratio = current_assets / cl if cl and cl > 0 else None

        return {
            "Ticker":            ticker,
            "שם":                info.get("shortName", ticker),
            "סקטור":             info.get("sector", ""),
            "מחיר ($)":          round(price, 2),
            "NCAV/מניה ($)":     round(ncav_per_share, 2),
            "P/NCAV":            round(p_ncav, 3),
            "שולי ביטחון %":     round((1 - p_ncav) * 100, 1),
            "Con.NCAV/מניה":     round(con_ncav_ps, 2) if con_ncav_ps is not None else None,
            "P/Con.NCAV":        round(price / con_ncav_ps, 3) if (con_ncav_ps and con_ncav_ps > 0) else None,
            "CA ($M)":           round(current_assets / 1e6, 1),
            "TL ($M)":           round(total_liabilities / 1e6, 1),
            "NCAV ($M)":         round(ncav / 1e6, 1),
            "Current Ratio":     round(current_ratio, 2) if current_ratio else None,
            "שווי שוק ($M)":     round(mktcap_m, 1) if mktcap_m else None,
            "P/B":               safe_get(info, "priceToBook"),
            "Country":           info.get("country", ""),
            "בורסה":             info.get("exchange", ""),
        }
    except Exception:
        return None


# ─── BUILD TICKER LIST ───────────────────────────────────────────────────────
def build_ticker_list(choice_key: str) -> list:
    key = UNIVERSE_OPTIONS[choice_key]

    if key == "sp500":
        with st.spinner("טוען S&P 500..."):
            return get_sp500_tickers()

    if key == "custom":
        raw = custom_tickers_raw.replace(",", " ").split()
        return [t.strip().upper() for t in raw if t.strip()]

    # ── NASDAQ API ──
    placeholder = st.empty()
    placeholder.info("⏳ שולף רשימת מניות מ-NASDAQ API (~5,000–8,000 מניות)...")
    try:
        df_all = fetch_nasdaq_universe()
        placeholder.empty()
    except Exception as e:
        placeholder.warning(
            f"NASDAQ API נכשל ({e})\n"
            "עובר ל-SEC EDGAR (ללא סינון שווי שוק)..."
        )
        import random
        tickers = fetch_sec_tickers()
        random.shuffle(tickers)
        return tickers[:int(max_tickers)]

    # סנן לפי שווי שוק
    has_cap = df_all["mktcap_usd"].notna()
    if key == "micro":
        mask = has_cap & (df_all["mktcap_usd"] < 300e6)
    elif key == "small":
        mask = has_cap & (df_all["mktcap_usd"] >= 300e6) & (df_all["mktcap_usd"] < 2e9)
    else:  # micro_small
        mask = has_cap & (df_all["mktcap_usd"] < 2e9)

    df_filtered = df_all[mask].sort_values("mktcap_usd")  # הכי קטנות קודם
    tickers = df_filtered["ticker"].tolist()

    total_available = len(tickers)
    tickers = tickers[:int(max_tickers)]

    st.info(
        f"🗂️ נמצאו **{total_available:,}** מניות בקטגוריה | "
        f"סורק **{len(tickers)}** ראשונות (לפי שווי שוק עולה)"
    )
    return tickers


# ─── MAIN SCAN ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([2, 5])
run_scan = col1.button("🔍 הרץ סריקה", type="primary", use_container_width=True)
col2.markdown(
    "<div class='warn-box'>⚠️ Micro cap = זמן ארוך. "
    "מומלץ להתחיל ב-200–300 מניות. "
    "yfinance עלול להחזיר נתונים חסרים — תאמת מול SEC Edgar.</div>",
    unsafe_allow_html=True,
)

if run_scan:
    tickers = build_ticker_list(universe_choice)

    if not tickers:
        st.error("לא נמצאו טיקרים. בדוק קלט.")
        st.stop()

    progress_bar = st.progress(0)
    status_txt = st.empty()
    results = []
    skipped = 0

    for i, ticker in enumerate(tickers):
        status_txt.text(
            f"[{i+1}/{len(tickers)}]  {ticker:<8}  |  נמצאו: {len(results)}"
        )
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
            if len(results) >= 100:
                st.info("נמצאו 100 מניות Net-Net — עוצר סריקה מוקדם.")
                break
        else:
            skipped += 1

        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(delay_ms / 1000)

    progress_bar.empty()
    status_txt.empty()

    if not results:
        st.warning("לא נמצאו מניות Net-Net. נסה להרחיב P/NCAV מקסימום או לשנות יקום.")
        st.stop()

    df = pd.DataFrame(results).sort_values("P/NCAV")

    # ── מדדים עיקריים ──
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (str(len(df)), "מניות Net-Net שנמצאו"),
        (f"{df['שולי ביטחון %'].mean():.1f}%", "שולי ביטחון ממוצעים"),
        (f"{df.iloc[0]['Ticker']} ({df.iloc[0]['P/NCAV']})", "הזול ביותר (P/NCAV)"),
        (f"{len(tickers) - skipped:,}", "טיקרים נסרקו בהצלחה"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader(f"📋 תוצאות ({len(df)} מניות Net-Net)")

    def color_pncav(val):
        if isinstance(val, (int, float)):
            if val < 0.33: return "color: #68d391; font-weight: bold"
            if val < 0.5:  return "color: #f6e05e"
            return "color: #fc8181"
        return ""

    cols_show = [
        "Ticker", "שם", "סקטור", "מחיר ($)",
        "NCAV/מניה ($)", "P/NCAV", "שולי ביטחון %",
    ]
    if show_conservative:
        cols_show += ["Con.NCAV/מניה", "P/Con.NCAV"]
    cols_show += [
        "CA ($M)", "TL ($M)", "NCAV ($M)",
        "Current Ratio", "שווי שוק ($M)", "P/B", "Country",
    ]

    # pandas >= 2.1: applymap → map on Styler
    _styler = df[cols_show].style
    try:
        styled = _styler.map(color_pncav, subset=["P/NCAV"]).format(na_rep="—", precision=2)
    except AttributeError:
        styled = _styler.applymap(color_pncav, subset=["P/NCAV"]).format(na_rep="—", precision=2)
    st.dataframe(styled, use_container_width=True, height=500)

    # ייצוא CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ ייצא לCSV",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"netnet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    with st.expander("📖 מתודולוגיה"):
        st.markdown("""
**NCAV** = Current Assets − Total Liabilities  
**Graham Criterion**: Price < ⅔ × NCAV per Share  
**Conservative NCAV** = Cash×100% + AR×75% + Inventory×50% + OtherCA×25% − All Liabilities  

**מקור יקום**:  
• `NASDAQ Screener API` — כל המניות ב-NASDAQ / NYSE / AMEX עם שווי שוק  
• גיבוי: `SEC EDGAR company_tickers.json` (~12,000 חברות)  

הרשימה ממוינת לפי שווי שוק עולה — הכי קטנות נסרקות ראשונות,  
כי Net-Nets אמיתיים נמצאים בעיקר שם.
        """)

st.markdown("---")
st.caption("לא המלצת השקעה. Net-Net דורש פיזור רחב. אמת נתונים מול SEC Edgar.")
"""
Net-Net Stock Scanner — Benjamin Graham Style
==============================================
יקום מניות: NASDAQ API + SEC EDGAR → אלפי small/micro caps
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Net-Net Scanner | Graham Style",
    page_icon="💎",
    layout="wide",
)

st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #48bb78; }
    .metric-lbl { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }
    .info-box {
        background: #1a2744;
        border-left: 4px solid #4299e1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #bee3f8;
    }
    .warn-box {
        background: #2d1b00;
        border-left: 4px solid #ed8936;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #fbd38d;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("💎 Net-Net Stock Scanner")
st.markdown(
    "<div class='info-box'>"
    "מחפש מניות לפי שיטת בנג'מין גרהאם: <b>Price &lt; 2/3 × NCAV per Share</b><br>"
    "NCAV = רכוש שוטף − כל ההתחייבויות | "
    "יקום: כל המניות ב-NASDAQ/NYSE/AMEX כולל small &amp; micro cap"
    "</div>",
    unsafe_allow_html=True,
)

# ─── UNIVERSE LOADERS ────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasdaq_universe() -> pd.DataFrame:
    """
    שולף את כל המניות מה-NASDAQ Screener API.
    מחזיר DataFrame עם: ticker, company, mktcap_usd, sector, industry, exchange.
    """
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&download=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    rows = data["data"]["rows"]
    df = pd.DataFrame(rows)

    def parse_mktcap(v):
        if not v or str(v).strip() in ("", "N/A", "None"):
            return None
        v = str(v).strip().upper().replace(",", "")
        try:
            if v.endswith("T"):
                return float(v[:-1]) * 1e12
            if v.endswith("B"):
                return float(v[:-1]) * 1e9
            if v.endswith("M"):
                return float(v[:-1]) * 1e6
            return float(v)
        except Exception:
            return None

    df["mktcap_usd"] = df["marketCap"].apply(parse_mktcap)
    df = df.rename(columns={"symbol": "ticker", "name": "company"})
    df = df[df["ticker"].notna() & (df["ticker"] != "")].copy()

    # הסר ETFs / warrants / units
    df = df[~df["ticker"].str.contains(r"[/\^~\+]", na=False, regex=True)]
    df = df[df["ticker"].str.len() <= 5]

    return df[["ticker", "company", "mktcap_usd", "sector", "industry", "exchange"]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sec_tickers() -> list:
    """גיבוי: שולף את כל הטיקרים מ-SEC EDGAR (~12k חברות)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(
        url,
        headers={"User-Agent": "research@example.com"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return [v["ticker"] for v in data.values() if v.get("ticker")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list:
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ הגדרות סריקה")

UNIVERSE_OPTIONS = {
    "🔬 Micro Cap  (< $300M)":          "micro",
    "📦 Small Cap  ($300M – $2B)":      "small",
    "🔬+📦 Micro + Small Cap":          "micro_small",
    "🏛️ S&P 500":                       "sp500",
    "✏️ טיקרים ידניים":                 "custom",
}
universe_choice = st.sidebar.selectbox("🌐 יקום מניות", list(UNIVERSE_OPTIONS.keys()))

custom_tickers_raw = ""
if universe_choice == "✏️ טיקרים ידניים":
    custom_tickers_raw = st.sidebar.text_area(
        "הכנס טיקרים (פסיק / רווח / שורה חדשה)",
        placeholder="AAPL, MSFT, GOOG ...",
        height=120,
    )

max_tickers = st.sidebar.number_input(
    "מקסימום מניות לסריקה",
    min_value=50, max_value=3000, value=300, step=50,
    help="~300 מניות ≈ 10 דקות. הגדל בזהירות.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 פילטרים")

price_min = st.sidebar.number_input("מחיר מינימום ($)", value=0.5, step=0.5, min_value=0.0)
price_max = st.sidebar.number_input("מחיר מקסימום ($)", value=200.0, step=10.0)

ncav_ratio_max = st.sidebar.slider(
    "P/NCAV מקסימום",
    min_value=0.1, max_value=1.5, value=0.67, step=0.01,
    help="Graham: 0.67 = 2/3 NCAV",
)

show_conservative = st.sidebar.checkbox("📊 Conservative NCAV", value=True)
exclude_financials = st.sidebar.checkbox("🏦 הסר בנקים/ביטוח", value=True)

st.sidebar.markdown("---")
delay_ms = st.sidebar.slider("השהיה בין טיקרים (ms)", 50, 400, 120, 25)

FINANCIAL_KEYWORDS_SECTOR = {"Financial Services", "Financial", "Banks"}
FINANCIAL_KEYWORDS_INDUSTRY = {"Bank", "Insur", "REIT", "Mortgage", "Thrift", "Brokerage"}


# ─── ANALYZER ────────────────────────────────────────────────────────────────
def safe_get(info, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def row_val(bs, col, *names):
    for n in names:
        if n in bs.index:
            v = bs.loc[n, col]
            if pd.notna(v):
                return float(v)
    return None


def analyze_ticker(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet

        price = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
        if not price or price <= 0:
            return None
        if not (price_min <= price <= price_max):
            return None

        if exclude_financials:
            sector = info.get("sector", "")
            industry = info.get("industry", "")
            if (sector in FINANCIAL_KEYWORDS_SECTOR or
                    any(kw in industry for kw in FINANCIAL_KEYWORDS_INDUSTRY)):
                return None

        if bs is None or bs.empty:
            return None
        col = bs.columns[0]

        current_assets = row_val(bs, col,
            "Current Assets", "Total Current Assets",
            "CurrentAssets", "TotalCurrentAssets")
        total_liabilities = row_val(bs, col,
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities", "TotalLiabilities")

        if current_assets is None or total_liabilities is None:
            return None

        shares = safe_get(info, "sharesOutstanding", "impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares
        if ncav_per_share <= 0:
            return None

        p_ncav = price / ncav_per_share
        if p_ncav > ncav_ratio_max:
            return None

        # Conservative NCAV
        con_ncav_ps = None
        if show_conservative:
            cash = row_val(bs, col,
                "Cash And Cash Equivalents", "Cash",
                "CashAndCashEquivalents",
                "Cash Cash Equivalents And Short Term Investments") or 0.0
            ar = row_val(bs, col,
                "Net Receivables", "Receivables",
                "Accounts Receivable", "AccountsReceivable") or 0.0
            inv = row_val(bs, col, "Inventory", "Inventories") or 0.0
            other_ca = max(0.0, current_assets - cash - ar - inv)
            con_ncav = (
                cash * 1.0
                + ar * 0.75
                + inv * 0.5
                + other_ca * 0.25
                - total_liabilities
            )
            con_ncav_ps = con_ncav / shares if shares else None

        mktcap = safe_get(info, "marketCap")
        mktcap_m = mktcap / 1e6 if mktcap else None

        cl = row_val(bs, col,
            "Current Liabilities", "Total Current Liabilities",
            "CurrentLiabilities", "TotalCurrentLiabilities")
        current_ratio = current_assets / cl if cl and cl > 0 else None

        return {
            "Ticker":            ticker,
            "שם":                info.get("shortName", ticker),
            "סקטור":             info.get("sector", ""),
            "מחיר ($)":          round(price, 2),
            "NCAV/מניה ($)":     round(ncav_per_share, 2),
            "P/NCAV":            round(p_ncav, 3),
            "שולי ביטחון %":     round((1 - p_ncav) * 100, 1),
            "Con.NCAV/מניה":     round(con_ncav_ps, 2) if con_ncav_ps is not None else None,
            "P/Con.NCAV":        round(price / con_ncav_ps, 3) if (con_ncav_ps and con_ncav_ps > 0) else None,
            "CA ($M)":           round(current_assets / 1e6, 1),
            "TL ($M)":           round(total_liabilities / 1e6, 1),
            "NCAV ($M)":         round(ncav / 1e6, 1),
            "Current Ratio":     round(current_ratio, 2) if current_ratio else None,
            "שווי שוק ($M)":     round(mktcap_m, 1) if mktcap_m else None,
            "P/B":               safe_get(info, "priceToBook"),
            "Country":           info.get("country", ""),
            "בורסה":             info.get("exchange", ""),
        }
    except Exception:
        return None


# ─── BUILD TICKER LIST ───────────────────────────────────────────────────────
def build_ticker_list(choice_key: str) -> list:
    key = UNIVERSE_OPTIONS[choice_key]

    if key == "sp500":
        with st.spinner("טוען S&P 500..."):
            return get_sp500_tickers()

    if key == "custom":
        raw = custom_tickers_raw.replace(",", " ").split()
        return [t.strip().upper() for t in raw if t.strip()]

    # ── NASDAQ API ──
    placeholder = st.empty()
    placeholder.info("⏳ שולף רשימת מניות מ-NASDAQ API (~5,000–8,000 מניות)...")
    try:
        df_all = fetch_nasdaq_universe()
        placeholder.empty()
    except Exception as e:
        placeholder.warning(
            f"NASDAQ API נכשל ({e})\n"
            "עובר ל-SEC EDGAR (ללא סינון שווי שוק)..."
        )
        import random
        tickers = fetch_sec_tickers()
        random.shuffle(tickers)
        return tickers[:int(max_tickers)]

    # סנן לפי שווי שוק
    has_cap = df_all["mktcap_usd"].notna()
    if key == "micro":
        mask = has_cap & (df_all["mktcap_usd"] < 300e6)
    elif key == "small":
        mask = has_cap & (df_all["mktcap_usd"] >= 300e6) & (df_all["mktcap_usd"] < 2e9)
    else:  # micro_small
        mask = has_cap & (df_all["mktcap_usd"] < 2e9)

    df_filtered = df_all[mask].sort_values("mktcap_usd")  # הכי קטנות קודם
    tickers = df_filtered["ticker"].tolist()

    total_available = len(tickers)
    tickers = tickers[:int(max_tickers)]

    st.info(
        f"🗂️ נמצאו **{total_available:,}** מניות בקטגוריה | "
        f"סורק **{len(tickers)}** ראשונות (לפי שווי שוק עולה)"
    )
    return tickers


# ─── MAIN SCAN ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([2, 5])
run_scan = col1.button("🔍 הרץ סריקה", type="primary", use_container_width=True)
col2.markdown(
    "<div class='warn-box'>⚠️ Micro cap = זמן ארוך. "
    "מומלץ להתחיל ב-200–300 מניות. "
    "yfinance עלול להחזיר נתונים חסרים — תאמת מול SEC Edgar.</div>",
    unsafe_allow_html=True,
)

if run_scan:
    tickers = build_ticker_list(universe_choice)

    if not tickers:
        st.error("לא נמצאו טיקרים. בדוק קלט.")
        st.stop()

    progress_bar = st.progress(0)
    status_txt = st.empty()
    results = []
    skipped = 0

    for i, ticker in enumerate(tickers):
        status_txt.text(
            f"[{i+1}/{len(tickers)}]  {ticker:<8}  |  נמצאו: {len(results)}"
        )
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
            if len(results) >= 100:
                st.info("נמצאו 100 מניות Net-Net — עוצר סריקה מוקדם.")
                break
        else:
            skipped += 1

        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(delay_ms / 1000)

    progress_bar.empty()
    status_txt.empty()

    if not results:
        st.warning("לא נמצאו מניות Net-Net. נסה להרחיב P/NCAV מקסימום או לשנות יקום.")
        st.stop()

    df = pd.DataFrame(results).sort_values("P/NCAV")

    # ── מדדים עיקריים ──
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (str(len(df)), "מניות Net-Net שנמצאו"),
        (f"{df['שולי ביטחון %'].mean():.1f}%", "שולי ביטחון ממוצעים"),
        (f"{df.iloc[0]['Ticker']} ({df.iloc[0]['P/NCAV']})", "הזול ביותר (P/NCAV)"),
        (f"{len(tickers) - skipped:,}", "טיקרים נסרקו בהצלחה"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader(f"📋 תוצאות ({len(df)} מניות Net-Net)")

    def color_pncav(val):
        if isinstance(val, (int, float)):
            if val < 0.33: return "color: #68d391; font-weight: bold"
            if val < 0.5:  return "color: #f6e05e"
            return "color: #fc8181"
        return ""

    cols_show = [
        "Ticker", "שם", "סקטור", "מחיר ($)",
        "NCAV/מניה ($)", "P/NCAV", "שולי ביטחון %",
    ]
    if show_conservative:
        cols_show += ["Con.NCAV/מניה", "P/Con.NCAV"]
    cols_show += [
        "CA ($M)", "TL ($M)", "NCAV ($M)",
        "Current Ratio", "שווי שוק ($M)", "P/B", "Country",
    ]

    # pandas >= 2.1: applymap → map on Styler
    _styler = df[cols_show].style
    try:
        styled = _styler.map(color_pncav, subset=["P/NCAV"]).format(na_rep="—", precision=2)
    except AttributeError:
        styled = _styler.applymap(color_pncav, subset=["P/NCAV"]).format(na_rep="—", precision=2)
    st.dataframe(styled, use_container_width=True, height=500)

    # ייצוא CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ ייצא לCSV",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"netnet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    with st.expander("📖 מתודולוגיה"):
        st.markdown("""
**NCAV** = Current Assets − Total Liabilities  
**Graham Criterion**: Price < ⅔ × NCAV per Share  
**Conservative NCAV** = Cash×100% + AR×75% + Inventory×50% + OtherCA×25% − All Liabilities  

**מקור יקום**:  
• `NASDAQ Screener API` — כל המניות ב-NASDAQ / NYSE / AMEX עם שווי שוק  
• גיבוי: `SEC EDGAR company_tickers.json` (~12,000 חברות)  

הרשימה ממוינת לפי שווי שוק עולה — הכי קטנות נסרקות ראשונות,  
כי Net-Nets אמיתיים נמצאים בעיקר שם.
        """)

st.markdown("---")
st.caption("לא המלצת השקעה. Net-Net דורש פיזור רחב. אמת נתונים מול SEC Edgar.")
"""
Net-Net Stock Scanner — Benjamin Graham Style
==============================================
יקום מניות: NASDAQ API + SEC EDGAR → אלפי small/micro caps
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Net-Net Scanner | Graham Style",
    page_icon="💎",
    layout="wide",
)

st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #48bb78; }
    .metric-lbl { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }
    .info-box {
        background: #1a2744;
        border-left: 4px solid #4299e1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #bee3f8;
    }
    .warn-box {
        background: #2d1b00;
        border-left: 4px solid #ed8936;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #fbd38d;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("💎 Net-Net Stock Scanner")
st.markdown(
    "<div class='info-box'>"
    "מחפש מניות לפי שיטת בנג'מין גרהאם: <b>Price &lt; 2/3 × NCAV per Share</b><br>"
    "NCAV = רכוש שוטף − כל ההתחייבויות | "
    "יקום: כל המניות ב-NASDAQ/NYSE/AMEX כולל small &amp; micro cap"
    "</div>",
    unsafe_allow_html=True,
)

# ─── UNIVERSE LOADERS ────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasdaq_universe() -> pd.DataFrame:
    """
    שולף את כל המניות מה-NASDAQ Screener API.
    מחזיר DataFrame עם: ticker, company, mktcap_usd, sector, industry, exchange.
    """
    url = (
        "https://api.nasdaq.com/api/screener/stocks"
        "?tableonly=true&limit=10000&offset=0&download=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    rows = data["data"]["rows"]
    df = pd.DataFrame(rows)

    def parse_mktcap(v):
        if not v or str(v).strip() in ("", "N/A", "None"):
            return None
        v = str(v).strip().upper().replace(",", "")
        try:
            if v.endswith("T"):
                return float(v[:-1]) * 1e12
            if v.endswith("B"):
                return float(v[:-1]) * 1e9
            if v.endswith("M"):
                return float(v[:-1]) * 1e6
            return float(v)
        except Exception:
            return None

    df["mktcap_usd"] = df["marketCap"].apply(parse_mktcap)
    df = df.rename(columns={"symbol": "ticker", "name": "company"})
    df = df[df["ticker"].notna() & (df["ticker"] != "")].copy()

    # הסר ETFs / warrants / units
    df = df[~df["ticker"].str.contains(r"[/\^~\+]", na=False, regex=True)]
    df = df[df["ticker"].str.len() <= 5]

    return df[["ticker", "company", "mktcap_usd", "sector", "industry", "exchange"]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sec_tickers() -> list:
    """גיבוי: שולף את כל הטיקרים מ-SEC EDGAR (~12k חברות)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(
        url,
        headers={"User-Agent": "research@example.com"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return [v["ticker"] for v in data.values() if v.get("ticker")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list:
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ הגדרות סריקה")

UNIVERSE_OPTIONS = {
    "🔬 Micro Cap  (< $300M)":          "micro",
    "📦 Small Cap  ($300M – $2B)":      "small",
    "🔬+📦 Micro + Small Cap":          "micro_small",
    "🏛️ S&P 500":                       "sp500",
    "✏️ טיקרים ידניים":                 "custom",
}
universe_choice = st.sidebar.selectbox("🌐 יקום מניות", list(UNIVERSE_OPTIONS.keys()))

custom_tickers_raw = ""
if universe_choice == "✏️ טיקרים ידניים":
    custom_tickers_raw = st.sidebar.text_area(
        "הכנס טיקרים (פסיק / רווח / שורה חדשה)",
        placeholder="AAPL, MSFT, GOOG ...",
        height=120,
    )

max_tickers = st.sidebar.number_input(
    "מקסימום מניות לסריקה",
    min_value=50, max_value=3000, value=300, step=50,
    help="~300 מניות ≈ 10 דקות. הגדל בזהירות.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 פילטרים")

price_min = st.sidebar.number_input("מחיר מינימום ($)", value=0.5, step=0.5, min_value=0.0)
price_max = st.sidebar.number_input("מחיר מקסימום ($)", value=200.0, step=10.0)

ncav_ratio_max = st.sidebar.slider(
    "P/NCAV מקסימום",
    min_value=0.1, max_value=1.5, value=0.67, step=0.01,
    help="Graham: 0.67 = 2/3 NCAV",
)

show_conservative = st.sidebar.checkbox("📊 Conservative NCAV", value=True)
exclude_financials = st.sidebar.checkbox("🏦 הסר בנקים/ביטוח", value=True)

st.sidebar.markdown("---")
delay_ms = st.sidebar.slider("השהיה בין טיקרים (ms)", 50, 400, 120, 25)

FINANCIAL_KEYWORDS_SECTOR = {"Financial Services", "Financial", "Banks"}
FINANCIAL_KEYWORDS_INDUSTRY = {"Bank", "Insur", "REIT", "Mortgage", "Thrift", "Brokerage"}


# ─── ANALYZER ────────────────────────────────────────────────────────────────
def safe_get(info, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def row_val(bs, col, *names):
    for n in names:
        if n in bs.index:
            v = bs.loc[n, col]
            if pd.notna(v):
                return float(v)
    return None


def analyze_ticker(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet

        price = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
        if not price or price <= 0:
            return None
        if not (price_min <= price <= price_max):
            return None

        if exclude_financials:
            sector = info.get("sector", "")
            industry = info.get("industry", "")
            if (sector in FINANCIAL_KEYWORDS_SECTOR or
                    any(kw in industry for kw in FINANCIAL_KEYWORDS_INDUSTRY)):
                return None

        if bs is None or bs.empty:
            return None
        col = bs.columns[0]

        current_assets = row_val(bs, col,
            "Current Assets", "Total Current Assets",
            "CurrentAssets", "TotalCurrentAssets")
        total_liabilities = row_val(bs, col,
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities", "TotalLiabilities")

        if current_assets is None or total_liabilities is None:
            return None

        shares = safe_get(info, "sharesOutstanding", "impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares
        if ncav_per_share <= 0:
            return None

        p_ncav = price / ncav_per_share
        if p_ncav > ncav_ratio_max:
            return None

        # Conservative NCAV
        con_ncav_ps = None
        if show_conservative:
            cash = row_val(bs, col,
                "Cash And Cash Equivalents", "Cash",
                "CashAndCashEquivalents",
                "Cash Cash Equivalents And Short Term Investments") or 0.0
            ar = row_val(bs, col,
                "Net Receivables", "Receivables",
                "Accounts Receivable", "AccountsReceivable") or 0.0
            inv = row_val(bs, col, "Inventory", "Inventories") or 0.0
            other_ca = max(0.0, current_assets - cash - ar - inv)
            con_ncav = (
                cash * 1.0
                + ar * 0.75
                + inv * 0.5
                + other_ca * 0.25
                - total_liabilities
            )
            con_ncav_ps = con_ncav / shares if shares else None

        mktcap = safe_get(info, "marketCap")
        mktcap_m = mktcap / 1e6 if mktcap else None

        cl = row_val(bs, col,
            "Current Liabilities", "Total Current Liabilities",
            "CurrentLiabilities", "TotalCurrentLiabilities")
        current_ratio = current_assets / cl if cl and cl > 0 else None

        return {
            "Ticker":            ticker,
            "שם":                info.get("shortName", ticker),
            "סקטור":             info.get("sector", ""),
            "מחיר ($)":          round(price, 2),
            "NCAV/מניה ($)":     round(ncav_per_share, 2),
            "P/NCAV":            round(p_ncav, 3),
            "שולי ביטחון %":     round((1 - p_ncav) * 100, 1),
            "Con.NCAV/מניה":     round(con_ncav_ps, 2) if con_ncav_ps is not None else None,
            "P/Con.NCAV":        round(price / con_ncav_ps, 3) if (con_ncav_ps and con_ncav_ps > 0) else None,
            "CA ($M)":           round(current_assets / 1e6, 1),
            "TL ($M)":           round(total_liabilities / 1e6, 1),
            "NCAV ($M)":         round(ncav / 1e6, 1),
            "Current Ratio":     round(current_ratio, 2) if current_ratio else None,
            "שווי שוק ($M)":     round(mktcap_m, 1) if mktcap_m else None,
            "P/B":               safe_get(info, "priceToBook"),
            "Country":           info.get("country", ""),
            "בורסה":             info.get("exchange", ""),
        }
    except Exception:
        return None


# ─── BUILD TICKER LIST ───────────────────────────────────────────────────────
def build_ticker_list(choice_key: str) -> list:
    key = UNIVERSE_OPTIONS[choice_key]

    if key == "sp500":
        with st.spinner("טוען S&P 500..."):
            return get_sp500_tickers()

    if key == "custom":
        raw = custom_tickers_raw.replace(",", " ").split()
        return [t.strip().upper() for t in raw if t.strip()]

    # ── NASDAQ API ──
    placeholder = st.empty()
    placeholder.info("⏳ שולף רשימת מניות מ-NASDAQ API (~5,000–8,000 מניות)...")
    try:
        df_all = fetch_nasdaq_universe()
        placeholder.empty()
    except Exception as e:
        placeholder.warning(
            f"NASDAQ API נכשל ({e})\n"
            "עובר ל-SEC EDGAR (ללא סינון שווי שוק)..."
        )
        import random
        tickers = fetch_sec_tickers()
        random.shuffle(tickers)
        return tickers[:int(max_tickers)]

    # סנן לפי שווי שוק
    has_cap = df_all["mktcap_usd"].notna()
    if key == "micro":
        mask = has_cap & (df_all["mktcap_usd"] < 300e6)
    elif key == "small":
        mask = has_cap & (df_all["mktcap_usd"] >= 300e6) & (df_all["mktcap_usd"] < 2e9)
    else:  # micro_small
        mask = has_cap & (df_all["mktcap_usd"] < 2e9)

    df_filtered = df_all[mask].sort_values("mktcap_usd")  # הכי קטנות קודם
    tickers = df_filtered["ticker"].tolist()

    total_available = len(tickers)
    tickers = tickers[:int(max_tickers)]

    st.info(
        f"🗂️ נמצאו **{total_available:,}** מניות בקטגוריה | "
        f"סורק **{len(tickers)}** ראשונות (לפי שווי שוק עולה)"
    )
    return tickers


# ─── MAIN SCAN ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([2, 5])
run_scan = col1.button("🔍 הרץ סריקה", type="primary", use_container_width=True)
col2.markdown(
    "<div class='warn-box'>⚠️ Micro cap = זמן ארוך. "
    "מומלץ להתחיל ב-200–300 מניות. "
    "yfinance עלול להחזיר נתונים חסרים — תאמת מול SEC Edgar.</div>",
    unsafe_allow_html=True,
)

if run_scan:
    tickers = build_ticker_list(universe_choice)

    if not tickers:
        st.error("לא נמצאו טיקרים. בדוק קלט.")
        st.stop()

    progress_bar = st.progress(0)
    status_txt = st.empty()
    results = []
    skipped = 0

    for i, ticker in enumerate(tickers):
        status_txt.text(
            f"[{i+1}/{len(tickers)}]  {ticker:<8}  |  נמצאו: {len(results)}"
        )
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
            if len(results) >= 100:
                st.info("נמצאו 100 מניות Net-Net — עוצר סריקה מוקדם.")
                break
        else:
            skipped += 1

        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(delay_ms / 1000)

    progress_bar.empty()
    status_txt.empty()

    if not results:
        st.warning("לא נמצאו מניות Net-Net. נסה להרחיב P/NCAV מקסימום או לשנות יקום.")
        st.stop()

    df = pd.DataFrame(results).sort_values("P/NCAV")

    # ── מדדים עיקריים ──
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (str(len(df)), "מניות Net-Net שנמצאו"),
        (f"{df['שולי ביטחון %'].mean():.1f}%", "שולי ביטחון ממוצעים"),
        (f"{df.iloc[0]['Ticker']} ({df.iloc[0]['P/NCAV']})", "הזול ביותר (P/NCAV)"),
        (f"{len(tickers) - skipped:,}", "טיקרים נסרקו בהצלחה"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader(f"📋 תוצאות ({len(df)} מניות Net-Net)")

    def color_pncav(val):
        if isinstance(val, (int, float)):
            if val < 0.33: return "color: #68d391; font-weight: bold"
            if val < 0.5:  return "color: #f6e05e"
            return "color: #fc8181"
        return ""

    cols_show = [
        "Ticker", "שם", "סקטור", "מחיר ($)",
        "NCAV/מניה ($)", "P/NCAV", "שולי ביטחון %",
    ]
    if show_conservative:
        cols_show += ["Con.NCAV/מניה", "P/Con.NCAV"]
    cols_show += [
        "CA ($M)", "TL ($M)", "NCAV ($M)",
        "Current Ratio", "שווי שוק ($M)", "P/B", "Country",
    ]

    styled = (
        df[cols_show]
        .style
        .applymap(color_pncav, subset=["P/NCAV"])
        .format(na_rep="—", precision=2)
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # ייצוא CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ ייצא לCSV",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"netnet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    with st.expander("📖 מתודולוגיה"):
        st.markdown("""
**NCAV** = Current Assets − Total Liabilities  
**Graham Criterion**: Price < ⅔ × NCAV per Share  
**Conservative NCAV** = Cash×100% + AR×75% + Inventory×50% + OtherCA×25% − All Liabilities  

**מקור יקום**:  
• `NASDAQ Screener API` — כל המניות ב-NASDAQ / NYSE / AMEX עם שווי שוק  
• גיבוי: `SEC EDGAR company_tickers.json` (~12,000 חברות)  

הרשימה ממוינת לפי שווי שוק עולה — הכי קטנות נסרקות ראשונות,  
כי Net-Nets אמיתיים נמצאים בעיקר שם.
        """)

st.markdown("---")
st.caption("לא המלצת השקעה. Net-Net דורש פיזור רחב. אמת נתונים מול SEC Edgar.")
"""
Net-Net Stock Scanner — Benjamin Graham Style
==============================================
מחפש מניות שנסחרות מתחת ל-2/3 מה-NCAV שלהן (Net Current Asset Value)
NCAV = Current Assets - Total Liabilities
Graham Criterion: Price < (2/3) * NCAV per Share
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import io
from datetime import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Net-Net Scanner | Graham Style",
    page_icon="💎",
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-box {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #48bb78; }
    .metric-lbl { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    h1 { color: #e2e8f0 !important; }
    .info-box {
        background: #1a2744;
        border-left: 4px solid #4299e1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #bee3f8;
    }
    .warn-box {
        background: #2d1b00;
        border-left: 4px solid #ed8936;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #fbd38d;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("💎 Net-Net Stock Scanner")
st.markdown(
    "<div class='info-box'>מחפש מניות לפי שיטת בנג'מין גרהאם: מניות הנסחרות <b>מתחת ל-2/3 מה-NCAV שלהן</b>.<br>"
    "NCAV = רכוש שוטף - <i>כל</i> ההתחייבויות (שוטפות + ארוכות טווח)<br>"
    "ניסחור מתחת ל-NCAV מציע שולי ביטחון גבוהים מאוד — המניה נסחרת מתחת לערך הפירוק שלה.</div>",
    unsafe_allow_html=True,
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ הגדרות סריקה")

UNIVERSE_OPTIONS = {
    "S&P 500": "sp500",
    "Nasdaq 100": "nasdaq100",
    "Russell 2000 (דוגמה — 500 מניות)": "russell_sample",
    "הכנס טיקרים ידנית": "custom",
}
universe_choice = st.sidebar.selectbox("🌐 יקום מניות", list(UNIVERSE_OPTIONS.keys()))

custom_tickers_raw = ""
if universe_choice == "הכנס טיקרים ידנית":
    custom_tickers_raw = st.sidebar.text_area(
        "הכנס טיקרים (מופרדים בפסיק או רווח)",
        placeholder="AAPL, MSFT, GOOG, ...",
        height=120,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 פילטרים")

price_min = st.sidebar.number_input("מחיר מינימום ($)", value=0.5, step=0.5, min_value=0.0)
price_max = st.sidebar.number_input("מחיר מקסימום ($)", value=500.0, step=10.0)

mktcap_min_m = st.sidebar.number_input(
    "שווי שוק מינימום (מיליון $)", value=10.0, step=5.0, min_value=0.0
)
mktcap_max_m = st.sidebar.number_input(
    "שווי שוק מקסימום (מיליון $)", value=5000.0, step=100.0
)

ncav_ratio_max = st.sidebar.slider(
    "P/NCAV מקסימום (Graham: 0.67)",
    min_value=0.1,
    max_value=1.5,
    value=0.67,
    step=0.01,
    help="Graham ממליץ לקנות במחיר מתחת ל-2/3 מה-NCAV, כלומר P/NCAV < 0.67",
)

show_conservative = st.sidebar.checkbox(
    "📊 הצג Conservative NCAV (נוסחת הנחות פירוק)",
    value=True,
    help="Cash×100% + AR×75% + Inventory×50% + OtherCA×25% - All Liabilities",
)

exclude_financials = st.sidebar.checkbox("🏦 הסר בנקים וביטוח", value=True)

st.sidebar.markdown("---")
delay_ms = st.sidebar.slider("השהיה בין טיקרים (ms)", 50, 500, 150, 50)

# ─── UNIVERSE LOADERS ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_sp500_tickers() -> list[str]:
    """שולף רשימת S&P 500 מ-Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        st.warning(f"לא ניתן לטעון S&P 500 מ-Wikipedia: {e}\nמשתמש ברשימת ברירת מחדל.")
        return DEFAULT_SP500


@st.cache_data(ttl=3600)
def get_nasdaq100_tickers() -> list[str]:
    """שולף רשימת Nasdaq 100 מ-Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns:
                return t["Ticker"].tolist()
            if "Symbol" in t.columns:
                return t["Symbol"].tolist()
        raise ValueError("לא נמצאה עמודת טיקר בטבלה")
    except Exception as e:
        st.warning(f"לא ניתן לטעון Nasdaq 100: {e}")
        return []


# דוגמה — Russell 2000 מלא לא זמין ב-Wikipedia בקלות,
# אז נשתמש בדוגמה של מניות small-cap ידועות
RUSSELL_SAMPLE = [
    "ACLS","ACM","ACCO","ACLX","ADMA","ADUS","AEIS","AGIO","AGYS","AHCO",
    "AHH","AHPI","AIN","AIRC","AIRT","AIV","ALEX","ALGT","ALHC","ALKS",
    "ALRM","AMBC","AMCX","AMD","AMEH","AMKR","AMMO","AMNB","AMPH","AMRK",
    "AMSC","AMT","AMWD","ANIK","ANIP","APAM","APLE","APOG","APPF","APPN",
    "APRE","ARCB","ARCH","ARCO","ARCT","ARDX","ARIS","ARKO","ARQT","ARTW",
    "ASIX","ASLE","ASND","ASRT","ASTE","ATEX","ATNI","ATNX","ATOS","ATRC",
    "ATRI","ATSG","ATXI","AUDC","AVDL","AVNS","AVRO","AWIN","AXGN","AXSM",
    "AZEK","AZPN","AZTA","BAND","BANF","BANR","BATRA","BCAL","BCBP","BCEI",
    "BCO","BCRX","BCSA","BCTX","BDGE","BFAM","BFST","BGFV","BHLB","BHVN",
    "BJ","BKD","BKNG","BLD","BLFS","BLI","BLKB","BLMN","BLOW","BLUE",
    "BLZE","BMI","BMRC","BMTC","BNFT","BOC","BODY","BOOT","BOWX","BPMC",
    "BRBR","BRBS","BRC","BRKL","BRP","BRSP","BSIG","BSVN","BTAI","BUSE",
    "BW","BWA","BWXT","BYND","CAC","CADE","CAKE","CALX","CAMP","CARA",
    "CASH","CASS","CBFV","CBNK","CBRL","CBU","CCBG","CCOI","CCRN","CCS",
    "CDLX","CDMO","CDNA","CDRE","CDXS","CECO","CENT","CENTA","CENX","CEVA",
    "CFFN","CFFI","CFR","CFRX","CGBD","CGEN","CGNT","CHCO","CHEF","CHX",
    "CIFR","CIZN","CK","CKHUY","CLBK","CLFD","CLMT","CLPS","CLPT","CLVS",
    "CLXT","CMCO","CMGE","CMLF","CMPR","CMRX","CNCE","CNDT","CNMD","CNOB",
    "CNSL","COHU","COLB","COMM","CONN","COOP","CORE","CORR","CORT","COST",
    "CPIX","CPLG","CPRX","CPSS","CPTN","CRBP","CRCT","CREE","CRGY","CRIS",
    "CRNX","CROX","CRSP","CRTX","CRWD","CSGP","CSII","CSTL","CTBI","CTLP",
]

DEFAULT_SP500 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","TSLA","JPM","JNJ",
    "V","PG","MA","UNH","HD","CVX","MRK","PEP","ABBV","LLY","AVGO","COST",
    "TMO","WMT","BAC","CSCO","ACN","MCD","ORCL","DHR","LIN","ABT","NEE",
    "TXN","CRM","AMGN","QCOM","PFE","HON","MS","BMY","UPS","RTX","SPGI",
    "SBUX","GS","INTC","BLK","CAT","GE","ELV","ISRG","ADP","MDLZ","GILD",
    "DE","SO","REGN","ZTS","AON","AXP","TGT","CI","MMC","BDX","VRTX","SYK",
    "CB","DUK","HUM","PLD","LOW","ITW","ADI","MO","KLAC","ETN","BKNG",
]

FINANCIAL_SECTORS = {"Financial Services", "Banks", "Insurance", "Diversified Financials"}

# ─── DATA FETCHER ─────────────────────────────────────────────────────────────
def safe_get(info: dict, *keys, default=None):
    """שולף ערך ראשון שנמצא מרשימת מפתחות אפשריים."""
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def analyze_ticker(ticker: str) -> dict | None:
    """
    מחשב את מדדי Net-Net עבור טיקר נתון.
    מחזיר None אם הנתונים חסרים / הטיקר לא עומד בפילטרים.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet  # עמודות = תאריכים, שורות = items

        # ── מחיר נוכחי ──
        price = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
        if not price or price <= 0:
            return None

        # ── פילטר מחיר ──
        if not (price_min <= price <= price_max):
            return None

        # ── שווי שוק ──
        mktcap = safe_get(info, "marketCap")
        if mktcap:
            mktcap_m = mktcap / 1e6
            if not (mktcap_min_m <= mktcap_m <= mktcap_max_m):
                return None
        else:
            mktcap_m = None

        # ── פילטר סקטור פיננסי ──
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        if exclude_financials and (sector in FINANCIAL_SECTORS or "Bank" in industry or "Insur" in industry):
            return None

        # ── מאזן ──
        if bs is None or bs.empty:
            return None

        col = bs.columns[0]  # הדוח האחרון

        def row(name_candidates):
            for n in name_candidates:
                if n in bs.index:
                    v = bs.loc[n, col]
                    if pd.notna(v):
                        return float(v)
            return None

        current_assets = row([
            "Current Assets", "Total Current Assets",
            "CurrentAssets", "TotalCurrentAssets",
        ])
        total_liabilities = row([
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities",
            "TotalLiabilities",
        ])
        if current_assets is None or total_liabilities is None:
            return None

        # ── NCAV ──
        shares = safe_get(info, "sharesOutstanding", "impliedSharesOutstanding")
        if not shares or shares <= 0:
            return None

        ncav = current_assets - total_liabilities
        ncav_per_share = ncav / shares

        if ncav_per_share <= 0:
            return None  # חברה עם NCAV שלילי — לא Net-Net קלאסי

        p_ncav = price / ncav_per_share

        if p_ncav > ncav_ratio_max:
            return None  # לא עומד בקריטריון Graham

        # ── Conservative NCAV (Graham's liquidation value) ──
        con_ncav_ps = None
        if show_conservative:
            cash = row([
                "Cash And Cash Equivalents", "Cash",
                "CashAndCashEquivalents",
                "Cash Cash Equivalents And Short Term Investments",
            ]) or 0.0
            receivables = row([
                "Net Receivables", "Receivables",
                "Accounts Receivable", "AccountsReceivable",
                "Net Receivables",
            ]) or 0.0
            inventory = row([
                "Inventory", "Inventories",
            ]) or 0.0
            other_ca = max(0.0, current_assets - cash - receivables - inventory)

            con_ncav = (
                cash * 1.0
                + receivables * 0.75
                + inventory * 0.5
                + other_ca * 0.25
                - total_liabilities
            )
            con_ncav_ps = con_ncav / shares if shares else None

        # ── נתונים נוספים ──
        current_liabilities = row([
            "Current Liabilities", "Total Current Liabilities",
            "CurrentLiabilities", "TotalCurrentLiabilities",
        ])
        current_ratio = (
            current_assets / current_liabilities
            if current_liabilities and current_liabilities > 0
            else None
        )

        debt = safe_get(info, "totalDebt") or 0
        pe = safe_get(info, "trailingPE", "forwardPE")
        pb = safe_get(info, "priceToBook")
        roe = safe_get(info, "returnOnEquity")
        revenue = safe_get(info, "totalRevenue")
        revenue_growth = safe_get(info, "revenueGrowth")
        name = info.get("shortName", ticker)
        exchange = info.get("exchange", "")

        return {
            "Ticker": ticker,
            "שם": name,
            "סקטור": sector,
            "תעשייה": industry[:30] if industry else "",
            "מחיר ($)": round(price, 2),
            "NCAV/מניה ($)": round(ncav_per_share, 2),
            "P/NCAV": round(p_ncav, 3),
            "שולי ביטחון %": round((1 - p_ncav) * 100, 1),
            "Con.NCAV/מניה ($)": round(con_ncav_ps, 2) if con_ncav_ps is not None else None,
            "P/Con.NCAV": round(price / con_ncav_ps, 3) if (con_ncav_ps and con_ncav_ps > 0) else None,
            "רכוש שוטף ($M)": round(current_assets / 1e6, 1),
            "התחייבויות ($M)": round(total_liabilities / 1e6, 1),
            "NCAV ($M)": round(ncav / 1e6, 1),
            "Current Ratio": round(current_ratio, 2) if current_ratio else None,
            "שווי שוק ($M)": round(mktcap_m, 0) if mktcap_m else None,
            "P/E": round(pe, 1) if pe and abs(pe) < 999 else None,
            "P/B": round(pb, 2) if pb else None,
            "ROE%": round(roe * 100, 1) if roe else None,
            "Revenue ($M)": round(revenue / 1e6, 0) if revenue else None,
            "Rev.Growth%": round(revenue_growth * 100, 1) if revenue_growth else None,
            "בורסה": exchange,
        }

    except Exception:
        return None


# ─── MAIN SCAN BUTTON ────────────────────────────────────────────────────────
st.markdown("---")
col_btn1, col_btn2 = st.columns([2, 5])
run_scan = col_btn1.button("🔍 הרץ סריקה", type="primary", use_container_width=True)
col_btn2.markdown(
    "<div class='warn-box'>⚠️ yfinance עשוי להחזיר נתונים לא מדויקים לעיתים."
    " תמיד אמת נתונים מול מקור ראשוני (SEC Edgar, Macrotrends).</div>",
    unsafe_allow_html=True,
)

if run_scan:
    # ── בנה רשימת טיקרים ──
    with st.spinner("טוען רשימת טיקרים..."):
        key = UNIVERSE_OPTIONS[universe_choice]
        if key == "sp500":
            tickers = get_sp500_tickers()
        elif key == "nasdaq100":
            tickers = get_nasdaq100_tickers()
        elif key == "russell_sample":
            tickers = RUSSELL_SAMPLE
        else:  # custom
            raw = custom_tickers_raw.replace(",", " ").split()
            tickers = [t.strip().upper() for t in raw if t.strip()]

    if not tickers:
        st.error("לא נמצאו טיקרים. בדוק את הקלט.")
        st.stop()

    st.info(f"סורק **{len(tickers)}** מניות...")

    # ── Progress ──
    progress_bar = st.progress(0)
    status_txt = st.empty()
    results = []
    errors = 0

    for i, ticker in enumerate(tickers):
        status_txt.text(f"בודק: {ticker}  ({i+1}/{len(tickers)})")
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
        else:
            errors += 1

        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(delay_ms / 1000)

    progress_bar.empty()
    status_txt.empty()

    # ── תוצאות ──
    if not results:
        st.warning("לא נמצאו מניות Net-Net העומדות בקריטריונים. נסה להרחיב את הפילטרים.")
    else:
        df = pd.DataFrame(results).sort_values("P/NCAV")

        # ── מדדים עיקריים ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val'>{len(df)}</div>"
                f"<div class='metric-lbl'>מניות Net-Net שנמצאו</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            avg_margin = df["שולי ביטחון %"].mean()
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val'>{avg_margin:.1f}%</div>"
                f"<div class='metric-lbl'>שולי ביטחון ממוצעים</div></div>",
                unsafe_allow_html=True,
            )
        with m3:
            best = df.iloc[0]["Ticker"]
            best_ratio = df.iloc[0]["P/NCAV"]
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val'>{best}</div>"
                f"<div class='metric-lbl'>הזול ביותר (P/NCAV={best_ratio})</div></div>",
                unsafe_allow_html=True,
            )
        with m4:
            scanned_ok = len(tickers) - errors
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val'>{scanned_ok}</div>"
                f"<div class='metric-lbl'>מניות נסרקו בהצלחה</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader(f"📋 תוצאות הסריקה ({len(df)} מניות)")

        # ── עיצוב טבלה ──
        def color_pncav(val):
            if isinstance(val, float):
                if val < 0.33:
                    return "color: #68d391; font-weight: bold"
                elif val < 0.5:
                    return "color: #f6e05e"
                else:
                    return "color: #fc8181"
            return ""

        def color_margin(val):
            if isinstance(val, float):
                if val > 65:
                    return "background-color: #1a3a2a; color: #68d391"
                elif val > 40:
                    return "background-color: #2d2a00; color: #f6e05e"
            return ""

        # עמודות להצגה
        display_cols = [
            "Ticker", "שם", "סקטור", "מחיר ($)", "NCAV/מניה ($)", "P/NCAV",
            "שולי ביטחון %",
        ]
        if show_conservative:
            display_cols += ["Con.NCAV/מניה ($)", "P/Con.NCAV"]
        display_cols += [
            "רכוש שוטף ($M)", "התחייבויות ($M)", "NCAV ($M)",
            "Current Ratio", "שווי שוק ($M)", "P/E", "P/B",
        ]

        styled = (
            df[display_cols]
            .style
            .applymap(color_pncav, subset=["P/NCAV"])
            .applymap(color_margin, subset=["שולי ביטחון %"])
            .format(na_rep="—", precision=2)
        )

        st.dataframe(styled, use_container_width=True, height=500)

        # ── פירוט מורחב ──
        with st.expander("📊 נתונים מורחבים (Revenue, ROE, Growth)"):
            extra_cols = ["Ticker", "שם", "P/E", "P/B", "ROE%", "Revenue ($M)", "Rev.Growth%", "בורסה"]
            st.dataframe(df[extra_cols].set_index("Ticker"), use_container_width=True)

        # ── ייצוא CSV ──
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ ייצא לCSV",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name=f"netnet_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # ── הסבר מתודולוגיה ──
        with st.expander("📖 מתודולוגיה — כיצד מחושב NCAV"):
            st.markdown("""
### NCAV קלאסי (גרהאם)
```
NCAV = רכוש שוטף (Current Assets) - כל ההתחייבויות (Total Liabilities)
NCAV per Share = NCAV / מניות במחזור
קריטריון Graham: קנה כאשר Price < 2/3 × NCAV per Share
```

### Conservative NCAV (ערך פירוק שמרני)
```
= מזומן × 100%
+ חייבים (AR) × 75%
+ מלאי × 50%
+ נכסים שוטפים אחרים × 25%
- כל ההתחייבויות × 100%
```

### למה Net-Net עובד?
- הרעיון: גם אם החברה תפסיק לפעול מחר, בעלי המניות יקבלו יותר ממה ששילמו
- גרהאם בנה פורטפוליו מגוון של עשרות מניות כאלה
- **אזהרה**: Net-Nets נוטות להיות חברות בקשיים — צריך פיזור רחב!
- חברות פיננסיות (בנקים, ביטוח) מוצאות מהסריקה כי המאזן שלהן עובד אחרת
            """)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ לא המלצת השקעה. נתוני yfinance עשויים להיות שגויים — תמיד אמת מול SEC Edgar. "
    "Net-Net requires portfolio diversification. Past performance ≠ future results."
)
