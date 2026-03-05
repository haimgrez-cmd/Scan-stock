import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum Backtest 20Y", layout="wide")
st.title("📊 Backtest מומנטום — 10 שנים | רבעוני | Top 3-5 מניות")
st.caption("יקום: S&P 500 | איזון: סוף כל רבעון | בנצ'מרק: SPY")

# ─── קבועים ────────────────────────────────────────────────────────────────
DATA_START      = "2013-01-01"   # 2 שנות warmup לפני תחילת הבאקטסט
BACKTEST_START  = "2015-01-01"   # 10 שנה אחורה
DATA_END        = datetime.now().strftime("%Y-%m-%d")


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        t  = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        t   = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
            "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC","AVGO"]


# ─── הורדת נתונים ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="מוריד 20 שנות נתונים — עשוי לקחת כמה דקות...")
def download_data(tickers: list[str]):
    all_t = ["SPY"] + tickers
    raw   = yf.download(
        all_t, start=DATA_START, end=DATA_END,
        auto_adjust=True, progress=False, threads=True, group_by="ticker"
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw.xs("Close",  axis=1, level=1).ffill()
        volume = raw.xs("Volume", axis=1, level=1).ffill()
    else:
        close  = raw[["Close"]].ffill()
        volume = raw[["Volume"]].ffill()

    # מסנן עמודות ריקות
    close  = close.dropna(how="all",  axis=1)
    volume = volume.dropna(how="all", axis=1)
    return close, volume


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return float(100.0) if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── ציון מניה ביום נתון ──────────────────────────────────────────────────
def score_on_date(c: pd.Series, v: pd.Series, date) -> float:
    """מחשב ציון ROC משוקלל ביום נתון — רק על נתונים עד אותו יום."""
    c = c.loc[:date].dropna()
    v = v.loc[:date].dropna()

    if len(c) < 252:
        return 0.0

    last = c.iloc[-1]
    sma50  = c.iloc[-50:].mean()  if len(c) >= 50  else np.nan
    sma100 = c.iloc[-100:].mean() if len(c) >= 100 else np.nan
    sma200 = c.iloc[-200:].mean() if len(c) >= 200 else np.nan

    if any(np.isnan(x) for x in [sma50, sma100, sma200]):
        return 0.0
    if not (sma50 > sma100 > sma200):   # חייב מבנה מדורג
        return 0.0
    if last < sma200:                    # חייב מעל ממוצע 200
        return 0.0

    roc3  = (c.iloc[-1] / c.iloc[-63]  - 1) * 100 if len(c) >= 63  else np.nan
    roc6  = (c.iloc[-1] / c.iloc[-126] - 1) * 100 if len(c) >= 126 else np.nan
    roc12 = (c.iloc[-1] / c.iloc[-252] - 1) * 100 if len(c) >= 252 else np.nan

    if any(np.isnan(x) for x in [roc3, roc6, roc12]):
        return 0.0

    rsi = calc_rsi(c)
    if rsi > 75:   # קנייה יתר קיצונית — לא נכנסים
        return 0.0

    vol20 = v.iloc[-20:].mean() if len(v) >= 20 else 0
    vol50 = v.iloc[-50:].mean() if len(v) >= 50 else 0
    vol_ok = 1.0 if (vol50 > 0 and vol20 > vol50 * 1.2) else 0.0

    # ציון משוקלל: ROC 6M חשוב ביותר, ואחריו ROC 12M ו-3M
    score = (roc6 * 0.5) + (roc12 * 0.3) + (roc3 * 0.2) + (vol_ok * 5)
    return float(score) if score > 0 else 0.0


# ─── Walk-Forward Backtest ──────────────────────────────────────────────────
def run_backtest(close, volume, top_n: int) -> tuple[pd.Series, pd.DataFrame]:
    tickers     = [c for c in close.columns if c != "SPY"]
    trading_days = close.index
    # ימי המסחר האחרונים של כל רבעון
    quarter_ends = pd.date_range(BACKTEST_START, DATA_END, freq="QE")

    port_values = [1.0]
    dates_out   = [trading_days[trading_days.searchsorted(quarter_ends[0])]]
    log_rows    = []

    for i in range(len(quarter_ends) - 1):
        e_date = trading_days[trading_days.searchsorted(quarter_ends[i],     side="left")]
        x_date = trading_days[trading_days.searchsorted(quarter_ends[i + 1], side="left")]

        if e_date >= trading_days[-1] or x_date >= trading_days[-1]:
            break

        # חישוב ציון לכל מניה ביום הכניסה
        scores = {}
        for t in tickers:
            if t not in close.columns or t not in volume.columns:
                continue
            s = score_on_date(close[t], volume[t], e_date)
            if s > 0:
                scores[t] = s

        if not scores:
            port_values.append(port_values[-1])
            dates_out.append(x_date)
            log_rows.append({
                "רבעון":          e_date.strftime("%Y-Q%q" if hasattr(e_date, 'quarter') else "%Y-%m"),
                "מניות":          "— מזומן",
                "תשואה %":        0.0,
                "ציון ממוצע":     0.0,
            })
            continue

        # בחירת Top N
        top = sorted(scores, key=scores.get, reverse=True)[:top_n]

        ep = close.loc[e_date, top].dropna()
        xp = close.loc[x_date, ep.index].dropna()
        common = ep.index.intersection(xp.index)

        if common.empty:
            port_values.append(port_values[-1])
            dates_out.append(x_date)
            continue

        # ─── פילטר נתונים שגויים ──────────────────────────────────────
        # פוסל מניה שעלתה/ירדה מעל 60% ברבעון — כנראה נתון זבל
        raw_rets = (xp[common] / ep[common]) - 1
        valid    = raw_rets[raw_rets.abs() < 0.60].index
        if valid.empty:
            port_values.append(port_values[-1])
            dates_out.append(x_date)
            continue

        # ─── הגבלת הפסד לפוזיציה — Stop Loss 20% ─────────────────────
        # אם מניה ירדה יותר מ-20% — מגבילים את ההפסד ל-20%
        capped = raw_rets[valid].clip(lower=-0.20)
        ret    = capped.mean()

        port_values.append(port_values[-1] * (1 + ret))
        dates_out.append(x_date)

        log_rows.append({
            "רבעון":      f"{e_date.year}-Q{(e_date.month-1)//3+1}",
            "מניות":      "  |  ".join(common.tolist()),
            "תשואה %":    round(ret * 100, 2),
            "ציון ממוצע": round(np.mean([scores[t] for t in common]), 1),
        })

    return pd.Series(port_values, index=dates_out), pd.DataFrame(log_rows)


# ─── ממשק ──────────────────────────────────────────────────────────────────
top_n_ui = st.slider("כמה מניות לבחור כל רבעון", 3, 5, 3)
st.divider()

if st.button("▶️ הרץ Backtest 20 שנה", type="primary"):

    tickers      = get_tickers()
    close, volume = download_data(tickers)

    st.success(f"נטענו {close.shape[1]-1} מניות | {close.shape[0]:,} ימי מסחר")

    with st.spinner("מריץ Walk-Forward על 80 רבעונים..."):
        portfolio, log = run_backtest(close, volume, top_n_ui)

    # ─── SPY נורמלי ────────────────────────────────────────────────────────
    spy      = close["SPY"].dropna()
    spy_bt   = spy[spy.index >= portfolio.index[0]]
    spy_norm = spy_bt / spy_bt.iloc[0]

    # ─── מדדי ביצוע ────────────────────────────────────────────────────────
    port_ret  = portfolio.iloc[-1] - 1
    spy_ret   = spy_norm.iloc[-1]  - 1
    port_q    = portfolio.pct_change().dropna()
    sharpe    = (port_q.mean() / port_q.std() * np.sqrt(4)) if port_q.std() > 0 else 0
    drawdown  = (portfolio / portfolio.cummax()) - 1
    max_dd    = drawdown.min()
    pos_q     = (port_q > 0).sum()

    # ─── גרף ───────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio.values,
        name=f"מומנטום Top {top_n_ui}",
        line=dict(color="#00C851", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name="S&P 500 (SPY)",
        line=dict(color="#2196F3", width=2, dash="dot")
    ))
    fig.update_layout(
        title=f"תשואה מצטברת — 10 שנים | Top {top_n_ui} מניות רבעוני",
        yaxis_title="ערך (1.0 = נקודת פתיחה)",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── מדדים ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("תשואה כוללת",       f"{port_ret*100:.0f}%",
              delta=f"{(port_ret-spy_ret)*100:.0f}% vs SPY")
    m2.metric("SPY תשואה",         f"{spy_ret*100:.0f}%")
    m3.metric("Sharpe (רבעוני×√4)", f"{sharpe:.2f}")
    m4.metric("Max Drawdown",       f"{max_dd*100:.1f}%")
    m5.metric("רבעונים חיוביים",   f"{pos_q}/{len(port_q)}")

    # ─── לוג רבעוני ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 לוג רבעוני מלא — אילו מניות נבחרו בכל רבעון"):
        st.dataframe(log, use_container_width=True)

    csv = log.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא לוג", csv,
                       f"backtest_20y_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
