import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from yahoo_fin import stock_info as si

st.set_page_config(page_title="סורק שוק מלא", layout="wide")

st.title('🚀 סורק 3,000 מניות - מודל 11 האינדיקטורים')

@st.cache_data
def get_huge_ticker_list():
    # משיכת כל מניות הנאסד"ק (מעל 3,000 סימולים)
    try:
        return si.tickers_nasdaq()
    except:
        return ['AAPL', 'TSLA', 'NVDA', 'PLTR', 'AMD', 'MSFT', 'AMZN']

def analyze_batch(tickers):
    if not tickers: return []
    # הורדה קבוצתית גדולה
    data = yf.download(tickers, period="150d", interval="1d", group_by='ticker', progress=False, threads=True)
    results = []
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < 50: continue
            
            # סינון ראשוני: מחיר מעל 2$ ומחזור מסחר ממוצע מעל 200,000 מניות
            last_price = float(df['Close'].iloc[-1])
            avg_vol = df['Volume'].tail(20).mean()
            if last_price < 2 or avg_vol < 200000: continue

            # חישוב 11 האינדיקטורים
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA50'] = ta.sma(df['Close'], length=50)
            df['SMA200'] = ta.sma(df['Close'], length=200)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd.iloc[:, 0]
            bbands = ta.bbands(df['Close'], length=20)
            df['BBL'] = bbands.iloc[:, 0]
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df['STOCHk'] = stoch.iloc[:, 0]
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx.iloc[:, 0]
            df['EMA20'] = ta.ema(df['Close'], length=20)
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # לוגיקת הניקוד המזוקקת (0-11)
            score = 0
            if last['RSI'] < 35: score += 1
            if last['Close'] > last['SMA50']: score += 1
            if last['MACD'] > 0: score += 1
            if last['Close'] < last['BBL'] * 1.02: score += 1
            if last['STOCHk'] < 25: score += 1
            if last['ADX'] > 20: score += 1
            if last['Close'] > last['SMA200']: score += 1
            if last['Close'] > last['EMA20']: score += 1
            if last['MACD'] > prev['MACD']: score += 1
            if last['RSI'] > prev['RSI']: score += 1
            if last['Volume'] > avg_vol * 1.5: score += 1 # פריצה בווליום

            if score >= 7:
                results.append({
                    "סימול": ticker,
                    "מחיר": round(last_price, 2),
                    "ציון (0-11)": score,
                    "RSI": round(float(last['RSI']), 1),
                    "מחזור יומי": f"{int(last['Volume']):,}",
                    "מצב": "💎 קנייה חזקה" if score >= 9 else "🚀 מומנטום חיובי"
                })
        except:
            continue
    return results

if st.button('🔥 הרץ סריקה על כל השוק (3,000+ מניות)'):
    all_tickers = get_huge_ticker_list()
    st.write(f"מתחיל סריקה על {len(all_tickers)} מניות. זה ייקח כ-2 דקות...")
    
    progress_bar = st.progress(0)
    all_results = []
    
    # חלוקה לקבוצות של 100 לביצועים מקסימליים
    batch_size = 100
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_res = analyze_batch(batch)
        all_results.extend(batch_res)
        progress_bar.progress(min((i + batch_size) / len(all_tickers), 1.0))
    
    if all_results:
        final_df = pd.DataFrame(all_results).sort_values(by="ציון (0-11)", ascending=False)
        st.success(f"נמצאו {len(all_results)} מניות שעומדות בקריטריונים!")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.info("לא נמצאו מניות עם ציון 7 ומעלה כרגע. השוק כנראה במצב המתנה.")
