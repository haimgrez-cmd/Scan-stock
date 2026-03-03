import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum Backtest", layout="wide")
st.title("📊 Backtest מומנטום - Walk-Forward 10 שנים vs S&P 500")
st.caption("יקום: S&P 500 | איזון מחדש: חודשי | בנצ'מרק: SPY")

# ─── קבועים ────────────────────────────────────────────────────────────────
LOOKBACK_YEARS = 10
# מורידים 12 שנה כדי שיהיה warmup של שנתיים לחישוב SMA200 ו-ROC 12M
DATA_START = (datetime.now() - timedelta(days=365 * 12)).strftime("%Y-%m-%d")
DATA_END   = datetime.now().strftime("%Y-%m-%d")
BACKTEST_START = (datetime.now() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")


# ─── טעינת נתונים ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        # fallback מינימלי אם ויקיפדיה לא זמינה
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","UNH",
            "V","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC",
            "AVGO","LLY","COST","TMO","CSCO","MCD","ACN","ABT","WMT","DHR"
        ]

@st.cache_data(ttl=86400, show_spinner="מוריד נתונים היסטוריים (עשוי לקחת כדקה)...")
def download_all_data(tickers: list[str], start: str, end: str):
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True, group_by="ticker"
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw.xs("Close",  axis=1, level=1).ffill().dropna(how="all", axis=1)
        volume = raw.xs("Volume", axis=1, level=1).ffill().dropna(how="all", axis=1)
    else:
        # טיקר בודד
        close  = raw[["Close"]].ffill()
        volume = raw[["Volume"]].ffill()
    return close, volume


# ─── חישוב אינדיקטורים ─────────────────────────────────────────────────────
def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=length - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=length - 1, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

@st.cache_data(ttl=86400, show_spinner="מחשב אינדיקטורים לכל המניות...")
def compute_scores(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    מחשב ציון מומנטום (0-8) לכל מניה בכל יום.
    ללא ADX (קשה לוקטוריזציה יעילה) — שמרנו 8 קריטריונים.
    """
    scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for ticker in close.columns:
        c = close[ticker].dropna()
        v = volume[ticker].reindex(c.index).fillna(0)

        if len(c) < 260:   # צריך לפחות שנה+ של נתונים
            continue

        sma50  = c.rolling(50).mean()
        sma100 = c.rolling(100).mean()
        sma200 = c.rolling(200).mean()
        roc3   = c.pct_change(63)  * 100   # ~3 חודשים
        roc6   = c.pct_change(126) * 100   # ~6 חודשים
        roc12  = c.pct_change(252) * 100   # ~12 חודשים
        vol20  = v.rolling(20).mean()
        vol50  = v.rolling(50).mean()
        rsi    = rsi_series(c, 14)

        s = (
            (c > sma50).astype(float)                              +  # מעל SMA50
            (c > sma100).astype(float)                             +  # מעל SMA100
            (c > sma200).astype(float)                             +  # מעל SMA200
            ((sma50 > sma100) & (sma100 > sma200)).astype(float)  +  # ממוצעים מדורגים
            (roc3  > 5).astype(float)                              +  # ROC 3M > 5%
            (roc6  > 10).astype(float)                             +  # ROC 6M > 10%
            (roc12 > 15).astype(float)                             +  # ROC 12M > 15%
            (vol20 > vol50 * 1.1).astype(float)                      # ווליום תומך
        )

        s[rsi > 80] = 0   # פסילה: קנייה יתר קיצונית

        scores[ticker] = s.reindex(close.index)

    return scores


# ─── Walk-Forward Backtest ──────────────────────────────────────────────────
def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    spy: pd.Series,
    start_date: str,
    min_score: int,
    top_n: int,
) -> tuple[pd.Series, pd.DataFrame]:

    trading_days = close.index
    month_ends   = pd.date_range(start_date, close.index[-1], freq="QE")

    port_values  = [1.0]
    dates_out    = [trading_days[trading_days.searchsorted(month_ends[0])]]
    log_rows     = []

    for i in range(len(month_ends) - 1):
        e_idx = trading_days.searchsorted(month_ends[i],     side="left")
        x_idx = trading_days.searchsorted(month_ends[i + 1], side="left")
        if e_idx >= len(trading_days) or x_idx >= len(trading_days):
            continue

        entry_date = trading_days[e_idx]
        exit_date  = trading_days[x_idx]

        # ─── בדיקת ציונים ───────────────────────────────────────────────
        day_scores = scores.loc[entry_date].dropna()
        qualified  = day_scores[day_scores >= min_score]

        if qualified.empty:
            port_values.append(port_values[-1])
            dates_out.append(exit_date)
            log_rows.append({
                "חודש": entry_date.strftime("%Y-%m"),
                "מניות שנבחרו": "— (כסף מזומן)",
                "מספר": 0,
                "תשואה חודשית %": 0.0,
            })
            continue

        top = qualified.nlargest(top_n).index
        ep  = close.loc[entry_date, top].dropna()
        xp  = close.loc[exit_date,  ep.index].dropna()
        common = ep.index.intersection(xp.index)

        if common.empty:
            port_values.append(port_values[-1])
            dates_out.append(exit_date)
            continue

        monthly_ret = ((xp[common] / ep[common]) - 1).mean()
        port_values.append(port_values[-1] * (1 + monthly_ret))
        dates_out.append(exit_date)

        log_rows.append({
            "חודש": entry_date.strftime("%Y-%m"),
            "מניות שנבחרו": "  |  ".join(common[:8].tolist()) + ("  ..." if len(common) > 8 else ""),
            "מספר": len(common),
            "תשואה חודשית %": round(monthly_ret * 100, 2),
        })

    return pd.Series(port_values, index=dates_out), pd.DataFrame(log_rows)


# ─── ממשק משתמש ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ איך ה-Backtest עובד?"):
    st.markdown("""
    **Walk-Forward Methodology:**
    בכל סוף חודש לאורך 10 השנים האחרונות:
    1. מחשבים ציון מומנטום לכל מניה ב-S&P 500 **רק על נתונים שהיו זמינים באותו יום** (ללא look-ahead bias)
    2. בוחרים את N המניות עם הציון הגבוה ביותר
    3. "מחזיקים" אותן חודש בחלוקה שווה
    4. משווים לתשואת SPY באותה תקופה
    
    **הערה:** לא כולל עמלות מסחר, מיסים, או השפעת מחיר (slippage)
    """)

c1, c2 = st.columns(2)
top_n_in     = c1.slider("מספר מניות להחזיק בכל חודש", 5, 50, 20)
min_score_in = c2.slider("ציון מינימלי (מתוך 8)", 4, 8, 6)

if st.button("▶️ הרץ Backtest", type="primary"):

    # 1. טעינה
    with st.spinner("טוען רשימת S&P 500..."):
        tickers = get_sp500_tickers()

    close, volume = download_all_data(["SPY"] + tickers, DATA_START, DATA_END)
    st.success(f"נטענו {close.shape[1]-1} מניות + SPY | {close.shape[0]:,} ימי מסחר")

    # 2. ציונים
    score_tickers = [t for t in close.columns if t != "SPY"]
    scores = compute_scores(close[score_tickers], volume[score_tickers])

    # 3. Backtest
    with st.spinner("מריץ Walk-Forward..."):
        portfolio, holdings_log = run_backtest(
            close[score_tickers], scores, close["SPY"], BACKTEST_START, min_score_in, top_n_in
        )

    # ─── בנצ'מרק SPY ──────────────────────────────────────────────────────
    spy       = close["SPY"].dropna()
    spy_bt    = spy[spy.index >= BACKTEST_START]
    spy_norm  = spy_bt / spy_bt.iloc[0]

    # ─── מדדי ביצוע ───────────────────────────────────────────────────────
    port_ret   = portfolio.iloc[-1] - 1
    spy_ret    = spy_norm.iloc[-1] - 1
    port_mon   = portfolio.pct_change().dropna()
    spy_mon    = spy_norm.pct_change().dropna()

    # Maximum Drawdown
    roll_max   = portfolio.cummax()
    drawdown   = (portfolio / roll_max) - 1
    max_dd     = drawdown.min()

    # Sharpe (שנתי, risk-free ≈ 0 לפשטות)
    sharpe = (port_mon.mean() / port_mon.std()) * np.sqrt(12) if port_mon.std() > 0 else 0

    # ─── גרף ──────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio.values,
        name="אסטרטגיית מומנטום",
        line=dict(color="#00C851", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name="S&P 500 (SPY)",
        line=dict(color="#2196F3", width=2, dash="dot")
    ))
    fig.add_hrect(y0=0, y1=1, fillcolor="red", opacity=0.03, line_width=0)
    fig.update_layout(
        title=f"תשואה מצטברת ({LOOKBACK_YEARS} שנים)",
        xaxis_title="תאריך",
        yaxis_title="ערך (1.0 = נקודת פתיחה)",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── כרטיסי מדדים ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("תשואה כוללת - מומנטום",  f"{port_ret*100:.1f}%",
              delta=f"{(port_ret - spy_ret)*100:.1f}% vs SPY")
    m2.metric("תשואה כוללת - SPY",       f"{spy_ret*100:.1f}%")
    m3.metric("תשואה חודשית ממוצעת",    f"{port_mon.mean()*100:.2f}%")
    m4.metric("Sharpe Ratio (שנתי)",     f"{sharpe:.2f}")
    m5.metric("Maximum Drawdown",        f"{max_dd*100:.1f}%")

    # ─── לוג חודשי ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 לוג חודשי מלא"):
        log_df = pd.DataFrame(holdings_log)
        # צביעת שורות לפי תשואה
        pos_months = (log_df["תשואה חודשית %"] > 0).sum()
        st.caption(f"חודשים חיוביים: {pos_months}/{len(log_df)}  |  "
                   f"תשואה חודשית ממוצעת: {log_df['תשואה חודשית %'].mean():.2f}%")
        st.dataframe(log_df, use_container_width=True)

    csv = holdings_log.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא לוג ל-CSV", csv, "momentum_backtest.csv", "text/csv")
