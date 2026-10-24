"""
score_model_backtest.py
=======================
סורק + בדיקה אחורה למודל דירוג 0-100 (מבוסס מחירים בלבד).

עקרונות (לפי המדריך):
- היקום נבנה מחדש בכל תאריך איזון מרשימת חברות S&P 500 היסטורית (Point-in-Time)
- דירוג בסגירה של יום t, ביצוע במחיר הפתיחה של יום המסחר הבא
- איזון רבעוני, Top 30 במשקל שווה, מכירה רק כשמניה יוצאת מה-Top 50
- עלויות מסחר על כל קנייה ומכירה
- בדיקת רגישות, פיצול תקופות, והשוואה ל-SPY ולכלל היקום במשקל שווה

מגבלה שחשוב להכיר: Yahoo Finance שומר היסטוריה רק למניות שעדיין נסחרות,
לכן חברות שנמחקו מהמדד ופשטו רגל לרוב חסרות. המסך מציג כמה מהיקום
ההיסטורי באמת נבדק, והתוצאה צריכה להיחשב כגבול עליון.
"""

import bisect
import warnings
from datetime import datetime, timedelta
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)

FACTORS = ["mom12_1", "mom6_1", "trend", "lowvol", "high52"]
FACTOR_LABELS = {
    "mom12_1": "מומנטום 12 חודשים (בלי החודש האחרון)",
    "mom6_1": "מומנטום 6 חודשים (בלי החודש האחרון)",
    "trend": "מרחק מממוצע 200 יום",
    "lowvol": "תנודתיות נמוכה (126 יום)",
    "high52": "קרבה לשיא 52 שבועות",
}
BASE_WEIGHTS = {"mom12_1": 35, "mom6_1": 25, "trend": 15, "lowvol": 15, "high52": 10}


# ---------------------------------------------------------------------------
# יקום היסטורי (Point-in-Time)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def load_membership():
    df = pd.read_csv(MEMBERSHIP_URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    sets = [
        frozenset(t.strip().replace(".", "-") for t in str(s).split(",") if t.strip())
        for s in df["tickers"]
    ]
    return list(df["date"]), sets


def members_on(membership, date):
    dates, sets = membership
    i = bisect.bisect_right(dates, pd.Timestamp(date)) - 1
    return sets[i] if i >= 0 else frozenset()


def tickers_ever(membership, since):
    dates, sets = membership
    out = set()
    start_i = max(bisect.bisect_right(dates, pd.Timestamp(since)) - 1, 0)
    for s in sets[start_i:]:
        out |= s
    return sorted(out)


# ---------------------------------------------------------------------------
# הורדת מחירים
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _download_batch(tickers, start):
    try:
        raw = yf.download(
            list(tickers), start=start, interval="1d", auto_adjust=True,
            group_by="ticker", progress=False, threads=True,
        )
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    frames = {"Open": {}, "Close": {}, "Volume": {}}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if multi else raw
            if "Close" not in sub.columns:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            for k in frames:
                frames[k][t] = sub[k]
        except Exception:
            continue
    if not frames["Close"]:
        return None
    return {k: pd.DataFrame(v) for k, v in frames.items()}


def load_prices(tickers, start, batch=100, label="מוריד מחירים"):
    tickers = sorted(set(tickers))
    parts = {"Open": [], "Close": [], "Volume": []}
    nb = max(1, ceil(len(tickers) / batch))
    bar = st.progress(0.0, text=label)
    for i in range(nb):
        b = tuple(tickers[i * batch:(i + 1) * batch])
        if not b:
            continue
        res = _download_batch(b, start)
        if res:
            for k in parts:
                parts[k].append(res[k])
        done = min((i + 1) * batch, len(tickers))
        bar.progress((i + 1) / nb, text=f"{label}... {done}/{len(tickers)}")
    bar.empty()
    if not parts["Close"]:
        return None
    out = {}
    for k, v in parts.items():
        df = pd.concat(v, axis=1).sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = df.loc[:, ~df.columns.duplicated()]
        out[k] = df
    return out


# ---------------------------------------------------------------------------
# גורמים וציון 0-100
# ---------------------------------------------------------------------------

def compute_factors(C, V):
    ret = C.pct_change(fill_method=None)
    f = {}
    f["mom12_1"] = C.shift(21) / C.shift(252) - 1
    f["mom6_1"] = C.shift(21) / C.shift(126) - 1
    f["trend"] = C / C.rolling(200, min_periods=200).mean() - 1
    f["lowvol"] = ret.rolling(126, min_periods=126).std()
    f["high52"] = C / C.rolling(252, min_periods=252).max()
    f["dollarvol"] = (C * V).rolling(60, min_periods=40).median()
    return f


def rank_table_at(factors, date, cols, min_dollar_vol):
    """מחזיר (דירוגי אחוזון לכל גורם, ערכי גלם) או None אם אין מספיק מניות."""
    cols = [c for c in cols if c in factors["mom12_1"].columns]
    if not cols:
        return None
    raw = pd.DataFrame({k: factors[k].loc[date, cols] for k in FACTORS})
    raw["dollarvol"] = factors["dollarvol"].loc[date, cols]
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw[raw["dollarvol"] >= min_dollar_vol]
    if len(raw) < 40:
        return None
    ranks = pd.DataFrame(index=raw.index)
    for k in FACTORS:
        ranks[k] = raw[k].rank(pct=True, ascending=(k != "lowvol"))
    return ranks, raw


def normalize_weights(w):
    s = float(sum(w.values()))
    if s <= 0:
        w = BASE_WEIGHTS
        s = float(sum(w.values()))
    return {k: w[k] / s for k in FACTORS}


def score_series(ranks, weights):
    w = pd.Series(normalize_weights(weights))[FACTORS]
    return (ranks[FACTORS] * w).sum(axis=1) * 100


# ---------------------------------------------------------------------------
# לוח איזון ובניית טבלאות דירוג היסטוריות
# ---------------------------------------------------------------------------

def rebalance_pairs(index, min_signal_pos=260):
    """(תאריך אות, תאריך ביצוע): אות בסגירה של היום שלפני, ביצוע בפתיחה של
    יום המסחר הראשון של ינואר/אפריל/יולי/אוקטובר."""
    pairs, seen = [], set()
    for pos in range(1, len(index)):
        d = index[pos]
        if d.month in (1, 4, 7, 10):
            key = (d.year, d.month)
            if key not in seen:
                seen.add(key)
                if pos - 1 >= min_signal_pos:
                    pairs.append((index[pos - 1], d))
    return pairs


def build_rank_tables(factors, C, membership, signal_dates, min_dv):
    tables, cov = {}, []
    for d in signal_dates:
        members = members_on(membership, d)
        cols = [t for t in members if t in C.columns]
        with_price = int(C.loc[d, cols].notna().sum()) if cols else 0
        res = rank_table_at(factors, d, cols, min_dv)
        cov.append({
            "date": d, "members": len(members), "with_price": with_price,
            "eligible": 0 if res is None else len(res[0]),
        })
        if res is not None:
            tables[d] = res[0]
    return tables, pd.DataFrame(cov)


def prepare(P, SPY, membership, min_dv):
    cal = SPY["Close"]["SPY"].dropna().index
    O = P["Open"].reindex(cal)
    C = P["Close"].reindex(cal)
    V = P["Volume"].reindex(cal)
    spy_o = SPY["Open"]["SPY"].reindex(cal)
    spy_c = SPY["Close"]["SPY"].reindex(cal)
    factors = compute_factors(C, V)
    pairs = rebalance_pairs(cal)
    tables, cov = build_rank_tables(factors, C, membership, [s for s, _ in pairs], min_dv)
    pairs = [(s, e) for s, e in pairs if s in tables]
    return {"O": O, "C": C, "spy_o": spy_o, "spy_c": spy_c,
            "tables": tables, "cov": cov, "pairs": pairs}


# ---------------------------------------------------------------------------
# מנוע הבדיקה
# ---------------------------------------------------------------------------

def run_backtest(data, weights, top_n=30, keep_n=50, cost=0.001, start_cash=100000.0):
    O, C, tables, pairs = data["O"], data["C"], data["tables"], data["pairs"]
    cols = list(C.columns)
    cidx = {t: i for i, t in enumerate(cols)}
    dates = C.index
    pos_of = {d: i for i, d in enumerate(dates)}
    Cf = C.ffill().values
    Of = O.ffill().values
    Oraw = O.values
    exec_map = {pos_of[e]: s for s, e in pairs}
    first = pos_of[pairs[0][1]]

    cash, shares, traded = start_cash, {}, 0.0
    eq, log, npos = {}, [], []

    for p in range(first, len(dates)):
        if p in exec_map:
            tab = tables[exec_map[p]]
            score = score_series(tab, weights).sort_values(ascending=False)
            ranked = list(score.index)
            keep_zone = set(ranked[:keep_n])
            held = [cols[i] for i in shares]
            keep = [t for t in held if t in keep_zone][:top_n]
            keep_set = set(keep)
            fill = [t for t in ranked if t not in keep_set]
            target = keep + fill[: top_n - len(keep)]
            target = [t for t in target if np.isfinite(Oraw[p, cidx[t]]) and Oraw[p, cidx[t]] > 0]
            tset = set(target)

            for i in list(shares.keys()):
                if cols[i] not in tset:
                    value = shares[i] * Of[p, i]
                    cash += value * (1 - cost)
                    traded += value
                    del shares[i]

            if target:
                pv = cash + sum(shares[i] * Of[p, i] for i in shares)
                tv = pv / len(target)
                deltas = {}
                for t in target:
                    i = cidx[t]
                    deltas[i] = tv - shares.get(i, 0.0) * Of[p, i]
                for i, d in deltas.items():
                    if d < 0:
                        cash += -d * (1 - cost)
                        traded += -d
                        shares[i] = shares[i] + d / Of[p, i]
                for i, d in deltas.items():
                    if d > 0:
                        spend = min(d, cash)
                        if spend <= 0:
                            continue
                        cash -= spend
                        traded += spend
                        shares[i] = shares.get(i, 0.0) + spend * (1 - cost) / Of[p, i]
            log.append((p, [cols[i] for i in shares]))
            npos.append(len(shares))

        eq[dates[p]] = cash + sum(shares[i] * Cf[p, i] for i in shares)

    eq = pd.Series(eq)
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    turnover = (traded / 2) / eq.mean() / yrs
    return {"equity": eq, "turnover": turnover, "log": log,
            "avg_pos": float(np.mean(npos)) if npos else 0.0}


def spy_benchmark(data, start_pos_date, start_cash=100000.0):
    so, sc = data["spy_o"], data["spy_c"]
    sc = sc.loc[start_pos_date:]
    base = so.loc[start_pos_date]
    return sc / base * start_cash


def universe_benchmark(data, start_cash=100000.0):
    """כלל היקום הכשיר במשקל שווה, מאוזן רבעונית, בלי עלויות."""
    O, C, tables, pairs = data["O"], data["C"], data["tables"], data["pairs"]
    cols = list(C.columns)
    cidx = {t: i for i, t in enumerate(cols)}
    dates = C.index
    pos_of = {d: i for i, d in enumerate(dates)}
    Cf = C.ffill().values
    Of = O.ffill().values
    eq, val = {}, start_cash
    for k, (s, e) in enumerate(pairs):
        p0 = pos_of[e]
        last = k + 1 >= len(pairs)
        p1 = len(dates) if last else pos_of[pairs[k + 1][1]]
        idx = [cidx[t] for t in tables[s].index]
        o0 = Of[p0, idx]
        ok = np.isfinite(o0) & (o0 > 0)
        rel = Cf[p0:p1][:, idx][:, ok] / o0[ok]
        daily = val * np.nanmean(rel, axis=1)
        for d, v in zip(dates[p0:p1], daily):
            eq[d] = v
        if not last:
            nxt = Of[p1, idx][ok] / o0[ok]
            val = val * float(np.nanmean(nxt))
    return pd.Series(eq)


def hit_rate(data, log):
    O, spy_o = data["O"], data["spy_o"]
    cols = list(O.columns)
    cidx = {t: i for i, t in enumerate(cols)}
    Of = O.ffill().values
    So = spy_o.ffill().values
    wins = total = 0
    for (p0, names), (p1, _) in zip(log, log[1:]):
        spy_r = So[p1] / So[p0] - 1
        for t in names:
            i = cidx[t]
            r = Of[p1, i] / Of[p0, i] - 1
            if np.isfinite(r):
                total += 1
                wins += int(r > spy_r)
    return wins / total if total else float("nan")


def perf_stats(eq):
    eq = eq.dropna()
    if len(eq) < 30:
        return None
    r = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    down = r[r < 0].std() * np.sqrt(252)
    return {
        "total": eq.iloc[-1] / eq.iloc[0] - 1,
        "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
        "vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "sortino": r.mean() * 252 / down if down and down > 0 else np.nan,
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


def sensitivity_runs(data, base_weights, cost):
    variants = [
        ("בסיס (30 מניות, יציאה מחוץ ל-50)", base_weights, 30, 50, cost),
        ("20 מניות (יציאה מחוץ ל-33)", base_weights, 20, 33, cost),
        ("40 מניות (יציאה מחוץ ל-67)", base_weights, 40, 67, cost),
        ("יותר מומנטום", {"mom12_1": 45, "mom6_1": 25, "trend": 10, "lowvol": 10, "high52": 10}, 30, 50, cost),
        ("פחות מומנטום", {"mom12_1": 25, "mom6_1": 20, "trend": 20, "lowvol": 20, "high52": 15}, 30, 50, cost),
        ("משקולות שוות", {k: 20 for k in FACTORS}, 30, 50, cost),
        ("עלויות פי 3", base_weights, 30, 50, cost * 3),
    ]
    rows = []
    for name, w, n, kn, c in variants:
        res = run_backtest(data, w, top_n=n, keep_n=kn, cost=c)
        st_ = perf_stats(res["equity"])
        rows.append({"וריאציה": name, "cagr": st_["cagr"], "sharpe": st_["sharpe"],
                     "maxdd": st_["maxdd"], "turnover": res["turnover"]})
    return rows


# ---------------------------------------------------------------------------
# סורק (היום)
# ---------------------------------------------------------------------------

def scan_now(P, members, weights, min_dv):
    C, V = P["Close"], P["Volume"]
    counts = C.notna().sum(axis=1)
    recent = counts.tail(10)
    asof = recent[recent >= 0.8 * counts.max()].index[-1]
    factors = compute_factors(C, V)
    res = rank_table_at(factors, asof, [t for t in members if t in C.columns], min_dv)
    if res is None:
        return None, asof
    ranks, raw = res
    score = score_series(ranks, weights)
    df = raw.copy()
    df["score"] = score
    df = df.sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df, asof


def fmt_pct(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def fmt_num(x, d=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# ממשק
# ---------------------------------------------------------------------------

def weights_ui(key):
    with st.expander("משקולות הציון (שינוי = כוונון, ראה אזהרה בבדיקה האחורית)"):
        w = {}
        for k in FACTORS:
            w[k] = st.slider(FACTOR_LABELS[k], 0, 100, BASE_WEIGHTS[k], key=f"{key}_{k}")
    return w


def tab_scanner():
    st.write("מדרג את מניות ה-S&P 500 הנוכחיות לפי ציון 0-100 ומראה מה לקנות, להחזיק ולמכור באיזון הבא.")
    c1, c2, c3 = st.columns(3)
    top_n = c1.number_input("כמה מניות להחזיק", 10, 60, 30, key="sc_top")
    keep_n = c2.number_input("מוכרים רק מחוץ ל-Top", 10, 100, 50, key="sc_keep")
    min_dv = c3.number_input("מינימום מחזור יומי ממוצע ($ מיליון)", 0, 500, 20, key="sc_dv") * 1e6
    weights = weights_ui("sc")

    if st.button("הרץ סורק", type="primary", key="sc_run"):
        membership = load_membership()
        members = members_on(membership, datetime.now())
        start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
        P = load_prices(list(members), start, label="מוריד מחירים")
        if not P:
            st.error("לא הצלחתי להוריד מחירים. נסה שוב בעוד כמה דקות.")
            return
        df, asof = scan_now(P, members, weights, min_dv)
        if df is None:
            st.error("אין מספיק מניות כשירות לדירוג.")
            return
        st.session_state["scan_df"] = df
        st.session_state["scan_asof"] = asof
        st.session_state["scan_members"] = len(members)

    if "scan_df" not in st.session_state:
        return

    df = st.session_state["scan_df"]
    asof = st.session_state["scan_asof"]
    st.caption(f"נתונים עד {asof.date()} | {len(df)} מניות כשירות מתוך {st.session_state['scan_members']} חברות במדד")

    show = df.head(max(int(keep_n), int(top_n)) + 10).copy()
    show["status"] = np.where(show["rank"] <= top_n, "Top - לקנות/להחזיק",
                              np.where(show["rank"] <= keep_n, "אזור החזקה", ""))
    out = pd.DataFrame({
        "דירוג": show["rank"],
        "סימול": show.index,
        "ציון": show["score"].round(1),
        "מומנטום 12-1": (show["mom12_1"] * 100).round(1),
        "מומנטום 6-1": (show["mom6_1"] * 100).round(1),
        "מרחק מ-SMA200 %": (show["trend"] * 100).round(1),
        "תנודתיות יומית %": (show["lowvol"] * 100).round(2),
        "% מהשיא": (show["high52"] * 100).round(1),
        "סטטוס": show["status"],
    })
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button("ייצא ל-CSV", out.to_csv(index=False).encode("utf-8-sig"),
                       f"score_model_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    st.subheader("מה לעשות באיזון הבא")
    held_txt = st.text_input("הקלד את המניות שאתה מחזיק עכשיו (מופרדות בפסיק)", key="sc_held")
    if held_txt.strip():
        held = [t.strip().upper().replace(".", "-") for t in held_txt.split(",") if t.strip()]
        rank_of = df["rank"].to_dict()
        keep, sell = [], []
        for t in held:
            r = rank_of.get(t)
            if r is not None and r <= keep_n:
                keep.append((t, r))
            else:
                sell.append((t, r))
        slots = int(top_n) - len(keep)
        keep_names = {t for t, _ in keep}
        buys = [t for t in df.index if t not in keep_names][:max(slots, 0)]
        st.write("להחזיק: " + (", ".join(f"{t} (#{r})" for t, r in keep) or "-"))
        st.write("למכור: " + (", ".join(f"{t} ({'מחוץ לדירוג' if r is None else '#' + str(r)})" for t, r in sell) or "-"))
        st.write("לקנות: " + (", ".join(f"{t} (#{rank_of[t]})" for t in buys) or "-"))
        st.caption("המלצה מכנית לפי הכללים בלבד, לא ייעוץ השקעות. האיזון המתוכנן הוא ביום המסחר הראשון של ינואר, אפריל, יולי ואוקטובר.")


def tab_backtest():
    st.write("בודק אחורה את אותו מודל, עם יקום היסטורי, ביצוע בפתיחה של היום הבא ועלויות מסחר.")
    c1, c2, c3, c4 = st.columns(4)
    years = c1.slider("שנים", 5, 12, 10, key="bt_years")
    cost = c2.number_input("עלות לכל כיוון (%)", 0.0, 1.0, 0.10, 0.05, key="bt_cost") / 100
    min_dv = c3.number_input("מינימום מחזור ($ מיליון)", 0, 500, 20, key="bt_dv") * 1e6
    top_n = c4.number_input("מניות בתיק", 10, 60, 30, key="bt_top")
    weights = weights_ui("bt")
    st.warning(
        "המשקולות קבועות מראש ולא כוונו על הנתונים. אם תשנה אותן כדי לשפר את התוצאה "
        "אתה עושה Overfitting, והבדיקה מאבדת את המשמעות שלה."
    )

    if st.button("הרץ בדיקה אחורה", type="primary", key="bt_run"):
        membership = load_membership()
        start_dt = datetime.now() - timedelta(days=int(years * 365 + 400))
        start = start_dt.strftime("%Y-%m-%d")
        universe = tickers_ever(membership, start_dt)
        st.info(f"{len(universe)} סימולים היו חברים במדד בתקופה. ההורדה הראשונה עשויה לקחת כמה דקות.")
        P = load_prices(universe, start, label="מוריד מחירי מניות")
        S = load_prices(["SPY"], start, label="מוריד SPY")
        if not P or not S:
            st.error("ההורדה נכשלה. נסה שוב בעוד כמה דקות.")
            return
        with st.spinner("מחשב דירוגים וסימולציה..."):
            data = prepare(P, S, membership, min_dv)
            if not data["pairs"]:
                st.error("אין מספיק היסטוריה לבדיקה.")
                return
            res = run_backtest(data, weights, top_n=int(top_n), keep_n=int(round(top_n * 5 / 3)), cost=cost)
            eq = res["equity"]
            spy = spy_benchmark(data, eq.index[0]).reindex(eq.index).ffill()
            uni = universe_benchmark(data).reindex(eq.index).ffill()
            sens = sensitivity_runs(data, weights, cost)
            hr = hit_rate(data, res["log"])
        st.session_state["bt"] = {
            "eq": eq, "spy": spy, "uni": uni, "turnover": res["turnover"],
            "avg_pos": res["avg_pos"], "hit": hr, "sens": sens,
            "cov": data["cov"], "ever": len(universe), "with_data": P["Close"].shape[1],
        }

    if "bt" not in st.session_state:
        return
    b = st.session_state["bt"]
    eq, spy, uni = b["eq"], b["spy"], b["uni"]

    # כיסוי נתונים
    cov = b["cov"]
    cov = cov[cov["members"] > 0]
    ratio = cov["with_price"] / cov["members"]
    st.subheader("כיסוי נתונים (הטיית הישרדות שנותרה)")
    st.write(
        f"מתוך {b['ever']} סימולים שהיו חברים במדד, ל-{b['with_data']} יש מחירים ב-Yahoo. "
        f"בכל תאריך איזון היה מחיר ל-{ratio.mean():.0%} מחברות המדד בממוצע (מינימום {ratio.min():.0%})."
    )
    st.caption("החברות החסרות הן בעיקר כאלה שנמחקו, נרכשו או פשטו רגל. בלעדיהן התוצאה אופטימית יותר מהמציאות.")

    # מדדים
    s_m, s_s, s_u = perf_stats(eq), perf_stats(spy), perf_stats(uni)
    rows = [
        ("תשואה שנתית (CAGR)", "cagr", fmt_pct),
        ("תשואה כוללת", "total", fmt_pct),
        ("תנודתיות שנתית", "vol", fmt_pct),
        ("Sharpe", "sharpe", fmt_num),
        ("Sortino", "sortino", fmt_num),
        ("ירידה מקסימלית", "maxdd", fmt_pct),
    ]
    table = {"מדד": [r[0] for r in rows],
             "המודל": [r[2](s_m[r[1]]) for r in rows],
             "SPY": [r[2](s_s[r[1]]) for r in rows],
             "כל היקום במשקל שווה": [r[2](s_u[r[1]]) for r in rows]}
    table["מדד"] += ["Turnover שנתי", "אחוז מניות שניצחו את SPY (Hit Rate)", "ממוצע מניות בתיק"]
    table["המודל"] += [fmt_pct(b["turnover"], 0), fmt_pct(b["hit"], 1), fmt_num(b["avg_pos"], 1)]
    table["SPY"] += ["-", "-", "-"]
    table["כל היקום במשקל שווה"] += ["-", "-", "-"]
    st.subheader("תוצאות")
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.line_chart(pd.DataFrame({"המודל": eq, "SPY": spy, "כל היקום במשקל שווה": uni}))

    # פיצול תקופות
    st.subheader("פיצול תקופות")
    years_avail = sorted(set(eq.index.year))
    split_year = st.selectbox("תחילת התקופה השנייה", years_avail[1:], index=len(years_avail[1:]) // 2, key="bt_split")
    split = pd.Timestamp(f"{split_year}-01-01")
    prow = []
    for name, series in [("המודל", eq), ("SPY", spy), ("כל היקום במשקל שווה", uni)]:
        a, c = perf_stats(series[series.index < split]), perf_stats(series[series.index >= split])
        prow.append({
            "": name,
            f"עד {split_year - 1}: CAGR": fmt_pct(a["cagr"]) if a else "-",
            f"עד {split_year - 1}: Sharpe": fmt_num(a["sharpe"]) if a else "-",
            f"מ-{split_year}: CAGR": fmt_pct(c["cagr"]) if c else "-",
            f"מ-{split_year}: Sharpe": fmt_num(c["sharpe"]) if c else "-",
        })
    st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)

    # רגישות
    st.subheader("בדיקת רגישות")
    srows = [{"וריאציה": r["וריאציה"], "CAGR": fmt_pct(r["cagr"]), "Sharpe": fmt_num(r["sharpe"]),
              "ירידה מקסימלית": fmt_pct(r["maxdd"]), "Turnover": fmt_pct(r["turnover"], 0)} for r in b["sens"]]
    st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)

    # מסקנות אוטומטיות
    st.subheader("איך לקרוא את התוצאה")
    notes = []
    if s_m["cagr"] <= s_s["cagr"]:
        notes.append("המודל לא ניצח את SPY בתשואה שנתית, ובנוסף התוצאה מנופחת קצת בגלל הטיית ההישרדות.")
    else:
        notes.append(f"המודל ניצח את SPY ב-{(s_m['cagr'] - s_s['cagr']) * 100:.1f} נקודות אחוז בשנה, לפני התיקון להטיית הישרדות.")
    if s_u["cagr"] > s_s["cagr"] + 0.01:
        notes.append("גם כל היקום במשקל שווה ניצח את SPY. חלק מהיתרון בא מהרכב היקום והמשקל השווה, לא מהדירוג. היתרון האמיתי של הדירוג הוא ההפרש בין המודל לבין קו היקום.")
    if s_m["sharpe"] <= s_s["sharpe"]:
        notes.append("ה-Sharpe של המודל לא גבוה מזה של SPY, כלומר התשואה הנוספת (אם יש) לא מצדיקה את התנודתיות.")
    cagrs = [r["cagr"] for r in b["sens"][:6]]
    if max(cagrs) - min(cagrs) > 0.03:
        notes.append("שינויים קטנים בהגדרות מזיזים את התשואה השנתית ביותר מ-3 נקודות אחוז. זה סימן לשבירות.")
    if b["turnover"] > 1.5:
        notes.append("ה-Turnover גבוה מ-150% בשנה, והעלויות האמיתיות עלולות להיות גבוהות מההנחה.")
    for n in notes:
        st.write("- " + n)


def main():
    st.set_page_config(page_title="מודל דירוג 0-100", layout="wide")
    st.markdown(
        "<style>.stApp { direction: rtl; } .stMarkdown, .stCaption, label, p { text-align: right; }</style>",
        unsafe_allow_html=True,
    )
    st.title("מודל דירוג 0-100: סורק ובדיקה אחורה")
    st.caption("הציון מבוסס על נתוני מחיר בלבד. נתונים פונדמנטליים חינמיים אינם Point-in-Time, ולכן הם לא נכללים בכוונה.")
    tab1, tab2 = st.tabs(["סורק (היום)", "בדיקה אחורה"])
    with tab1:
        tab_scanner()
    with tab2:
        tab_backtest()


if __name__ == "__main__":
    main()
