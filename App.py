import time
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from yahoo_fin import stock_info as si

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="סורק שוק מלא", layout="wide")
st.title("🚀 סורק 3,000 מניות - מודל 11 האינדיקטורים")

# ─── קבועים ────────────────────────────────────────────────────────────────
MIN_PRICE        = 2.0
MIN_AVG_VOL      = 200_000
MIN_SCORE        = 7
BATCH_SIZE       = 100
SLEEP_BETWEEN    = 1.0   # שניות בין batches למניעת חסימה
DATA_PERIOD      = "300d" # מספיק ל-SMA200 + מרווח
STRONG_BUY_SCORE = 9


# ─── רשימת טיקרים ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_ticker_list() -> list[str]:
    try:
        tickers = si.tickers_nasdaq()
        logger.info(f"נטענו {len(tickers)} טיקרים מנאסד\"ק")
        return tickers
    except Exception as e:
        logger.error(f"שגיאה בטעינת טיקרים: {e}")
        return ["AAPL", "TSLA", "NVDA", "PLTR", "AMD", "MSFT", "AMZN"]


# ─── ניתוח batch ────────────────────────────────────────────────────────────
def analyze_batch(tickers: list[str]) -> list[dict]:
    if not tickers:
        return []

    try:
        raw = yf.download(
            tickers,
            period=DATA_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=True,
        )
    except Exception as e:
        logger.error(f"שגיאה בהורדת נתונים: {e}")
        return []

    results = []

    for ticker in tickers:
        try:
            # תמיכה ב-MultiIndex (מספר טיקרים) וב-flat DataFrame (טיקר בודד)
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna()
            else:
                df = raw.dropna()

            if len(df) < 60:
                continue

            last_price = float(df["Close"].iloc[-1])
            avg_vol    = float(df["Volume"].tail(20).mean())

            if last_price < MIN_PRICE or avg_vol < MIN_AVG_VOL:
                continue

            # ─── חישוב אינדיקטורים ──────────────────────────────────────
            df["RSI"]    = ta.rsi(df["Close"], length=14)
            df["SMA50"]  = ta.sma(df["Close"], length=50)
            df["SMA200"] = ta.sma(df["Close"], length=200)
            df["EMA20"]  = ta.ema(df["Close"], length=20)

            macd_df      = ta.macd(df["Close"])
            df["MACD"]   = macd_df.iloc[:, 0]   # MACD line
            df["MACD_S"] = macd_df.iloc[:, 1]   # Signal line

            bb_df        = ta.bbands(df["Close"], length=20)
            df["BBL"]    = bb_df.iloc[:, 0]
            df["BBU"]    = bb_df.iloc[:, 2]

            stoch_df     = ta.stoch(df["High"], df["Low"], df["Close"])
            df["STOCHk"] = stoch_df.iloc[:, 0]

            adx_df       = ta.adx(df["High"], df["Low"], df["Close"])
            df["ADX"]    = adx_df.iloc[:, 0]

            df.dropna(inplace=True)
            if len(df) < 2:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # ─── לוגיקת ניקוד (0-11) ────────────────────────────────────
            score = 0

            # מומנטום / oversold
            if last["RSI"] < 35:                          score += 1  # RSI נמוך – oversold
            if last["RSI"] > prev["RSI"]:                 score += 1  # RSI עולה
            if last["STOCHk"] < 25:                       score += 1  # סטוכסטיק oversold

            # טרנד
            if last["Close"] > last["SMA50"]:             score += 1  # מעל ממוצע 50
            if last["Close"] > last["SMA200"]:            score += 1  # מעל ממוצע 200
            if last["Close"] > last["EMA20"]:             score += 1  # מעל EMA 20

            # MACD
            if last["MACD"] > last["MACD_S"]:            score += 1  # MACD מעל signal
            if last["MACD"] > prev["MACD"]:               score += 1  # MACD עולה

            # ADX – עוצמת טרנד
            if last["ADX"] > 20:                          score += 1  # טרנד חזק

            # בולינגר – קרבה לרצועה תחתונה (ללא סתירה עם SMA50)
            bb_range = last["BBU"] - last["BBL"]
            bb_pos   = (last["Close"] - last["BBL"]) / bb_range if bb_range > 0 else 0.5
            if bb_pos < 0.3:                              score += 1  # בשליש התחתון של הבולינגר

            # פריצת ווליום
            if last["Volume"] > avg_vol * 1.5:            score += 1  # ווליום חריג

            if score < MIN_SCORE:
                continue

            pct_change = (last["Close"] - prev["Close"]) / prev["Close"] * 100

            results.append({
                "סימול":        ticker,
                "מחיר":         round(last_price, 2),
                "ציון (0-11)":  score,
                "מצב":          "💎 קנייה חזקה" if score >= STRONG_BUY_SCORE else "🚀 מומנטום חיובי",
                "RSI":          round(float(last["RSI"]), 1),
                "MACD":         round(float(last["MACD"]), 3),
                "ADX":          round(float(last["ADX"]), 1),
                "% שינוי יומי": round(pct_change, 2),
                "מחזור יומי":   int(last["Volume"]),
            })

        except Exception as e:
            logger.warning(f"שגיאה בניתוח {ticker}: {e}")
            continue

    return results


# ─── ממשק משתמש ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("ציון מינימלי", MIN_SCORE)
col2.metric("מחיר מינימלי", f"${MIN_PRICE}")
col3.metric("מחזור מינימלי", f"{MIN_AVG_VOL:,}")

st.divider()

if st.button("🔥 הרץ סריקה על כל השוק (3,000+ מניות)"):
    all_tickers = get_ticker_list()
    st.info(f"מתחיל סריקה על {len(all_tickers)} מניות. זה ייקח כ-2 דקות...")

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    all_results   = []
    total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch       = all_tickers[start : start + BATCH_SIZE]
        batch_res   = analyze_batch(batch)
        all_results.extend(batch_res)

        progress    = (i + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(
            f"batch {i+1}/{total_batches} | נמצאו עד כה: {len(all_results)} מניות"
        )

        if i < total_batches - 1:
            time.sleep(SLEEP_BETWEEN)

    status_text.empty()

    if all_results:
        df_final = (
            pd.DataFrame(all_results)
            .sort_values("ציון (0-11)", ascending=False)
            .reset_index(drop=True)
        )

        st.success(f"✅ נמצאו {len(df_final)} מניות שעומדות בקריטריונים!")

        # ─── סינון אינטראקטיבי ──────────────────────────────────────────
        st.subheader("סינון תוצאות")
        fc1, fc2 = st.columns(2)
        min_score_filter = fc1.slider("ציון מינימלי", MIN_SCORE, 11, MIN_SCORE)
        status_filter    = fc2.multiselect(
            "מצב", ["💎 קנייה חזקה", "🚀 מומנטום חיובי"],
            default=["💎 קנייה חזקה", "🚀 מומנטום חיובי"]
        )

        mask     = (df_final["ציון (0-11)"] >= min_score_filter) & \
                   (df_final["מצב"].isin(status_filter))
        filtered = df_final[mask]

        st.dataframe(
            filtered.style.background_gradient(subset=["ציון (0-11)"], cmap="RdYlGn"),
            use_container_width=True,
        )

        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 ייצא ל-CSV", csv, "תוצאות_סריקה.csv", "text/csv")
    else:
        st.warning("לא נמצאו מניות עם ציון 7 ומעלה כרגע. השוק כנראה במצב המתנה.")
