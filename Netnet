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
