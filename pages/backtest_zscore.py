"""
backtest_zscore.py
Backtest of the z-score mean-reversion rule used by quant_portfolio_agent.py.
Signal at close of day t, execution at close of day t+1 (no look-ahead).
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def run_backtest(close, lookback=20, z_entry=-2.0, z_exit=-0.5,
                 max_pos=5, cost=0.001, start_cash=100000.0):
    close = close.ffill()
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    z = (close - mean) / std.replace(0, np.nan)

    dates = close.index
    px = close.values
    zv = z.values
    cols = list(close.columns)

    cash = start_cash
    shares = {}          # col index -> shares
    entry = {}           # col index -> (entry_day, entry_price)
    trades = []          # (ticker, days_held, return)
    equity = pd.Series(np.nan, index=dates)
    equity.iloc[lookback] = start_cash

    for i in range(lookback, len(dates) - 1):
        j = i + 1
        # exits decided on day i signal, executed at close j
        for k in list(shares.keys()):
            zi = zv[i, k]
            if np.isfinite(zi) and zi > z_exit and np.isfinite(px[j, k]):
                value = shares[k] * px[j, k]
                cash += value * (1 - cost)
                d0, p0 = entry[k]
                trades.append((cols[k], j - d0, px[j, k] / p0 - 1 - 2 * cost))
                del shares[k]
                del entry[k]

        slots = max_pos - len(shares)
        if slots > 0:
            cand = []
            for k in range(len(cols)):
                if k in shares:
                    continue
                zi = zv[i, k]
                if np.isfinite(zi) and zi <= z_entry and np.isfinite(px[j, k]):
                    cand.append((zi, k))
            cand.sort()
            equity_now = cash + sum(shares[k] * px[j, k] for k in shares if np.isfinite(px[j, k]))
            for zi, k in cand[:slots]:
                target = equity_now / max_pos
                spend = min(target, cash)
                if spend <= 0:
                    break
                shares[k] = spend * (1 - cost) / px[j, k]
                entry[k] = (j, px[j, k])
                cash -= spend

        equity.iloc[j] = cash + sum(shares[k] * px[j, k] for k in shares if np.isfinite(px[j, k]))

    equity = equity.dropna()
    return equity, trades


def stats(equity, years_label=""):
    rets = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    return {"total": total, "cagr": cagr, "maxdd": dd, "sharpe": sharpe}


@st.cache_data(ttl=86400)
def load_prices(tickers, years):
    raw = yf.download(list(tickers), period=f"{years + 1}y", interval="1d",
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return close.dropna(axis=1, how="all")


def fetch_sp500_symbols():
    """Returns (symbols, source). Tries Wikipedia with a browser User-Agent, then a GitHub CSV."""
    import io
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_html(io.StringIO(r.text))[0]
        syms = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(syms) > 400:
            return syms, "wikipedia"
    except Exception:
        pass
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        syms = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(syms) > 400:
            return syms, "github"
    except Exception:
        pass
    return [], "failed"

@st.cache_data(ttl=86400)
def sp500_list(n):
    syms, source = fetch_sp500_symbols()
    if not syms:
        raise RuntimeError("Could not download the S&P 500 list from Wikipedia or GitHub.")
    return tuple(syms[:n])


def main():
    st.set_page_config(page_title="Backtest", layout="wide")
    st.title("Backtest - Z-score mean reversion")
    st.caption("Signal at close of day t, execution at close of day t+1. Cost applied on each side.")

    c1, c2, c3, c4 = st.columns(4)
    years = c1.slider("Years", 3, 10, 10)
    n = c2.slider("Universe size", 20, 200, 100, step=10)
    z_entry = c3.number_input("Entry z", value=-2.0, step=0.25)
    z_exit = c4.number_input("Exit z", value=-0.5, step=0.25)
    cost = st.number_input("Cost per side (%)", value=0.1, step=0.05) / 100

    if st.button("Run backtest", type="primary"):
        with st.spinner("Downloading prices and simulating..."):
            tickers = sp500_list(n)
            close = load_prices(tickers, years)
            spy = load_prices(("SPY",), years)["SPY"]
            equity, trades = run_backtest(close, z_entry=z_entry, z_exit=z_exit, cost=cost)

        spy = spy.reindex(equity.index).ffill()
        bench = spy / spy.iloc[0] * equity.iloc[0]

        s, b = stats(equity), stats(bench)
        wins = [t for t in trades if t[2] > 0]
        a, bcol = st.columns(2)
        a.subheader("Strategy")
        a.write(f"Total return: {s['total']:.1%}")
        a.write(f"CAGR: {s['cagr']:.1%}")
        a.write(f"Max drawdown: {s['maxdd']:.1%}")
        a.write(f"Sharpe: {s['sharpe']:.2f}")
        a.write(f"Trades: {len(trades)}")
        if trades:
            a.write(f"Win rate: {len(wins) / len(trades):.1%}")
            a.write(f"Avg holding days: {np.mean([t[1] for t in trades]):.1f}")
            a.write(f"Avg trade return: {np.mean([t[2] for t in trades]):.2%}")
        bcol.subheader("SPY buy and hold")
        bcol.write(f"Total return: {b['total']:.1%}")
        bcol.write(f"CAGR: {b['cagr']:.1%}")
        bcol.write(f"Max drawdown: {b['maxdd']:.1%}")
        bcol.write(f"Sharpe: {b['sharpe']:.2f}")

        st.line_chart(pd.DataFrame({"Strategy": equity, "SPY": bench}))
        st.warning("Survivorship bias: the universe is today's S&P 500 members, "
                   "so companies that dropped out or went bankrupt are missing. "
                   "Real results would likely be worse.")


if __name__ == "__main__":
    main()
