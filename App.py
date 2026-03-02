import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from yahoo_fin import stock_info as si

st.set_page_config(page_title="סורק המניות המלא", layout="wide")

st.title('🔍 סורק השוק המלא - 11 אינדיקטורים')

# בחירת שוק
market = st.selectbox("בחר שוק לסריקה:", ["NASDAQ (טכנולוגיה וצמיחה)", "S&P 500 (גדולות)", "מניות ה-DOW"])

def analyze_stock(symbol):
    try:
        # משיכת נתונים
        df = yf.download(symbol, period="150d", interval="1d", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 11 האינדיקטורים
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
        
        # לוגיקת ניקוד
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
        if last['Volume'] > df['Volume'].tail(20).mean() * 1.3: score += 1

        if score >= 6: # מציג רק מניות עם פוטנציאל
            return {
                "סימול": symbol,
                "מחיר": round(float(last['Close']), 2),
                "ציון (0-11)": score,
                "RSI": round(float(last['RSI']), 1),
                "מצב": "🚀 קנייה חזקה" if score >= 8 else "מעקב"
            }
    except:
        return None

if st.button('🚀 הרץ סריקה חיה'):
    with st.spinner('טוען רשימת מניות...'):
        if "NASDAQ" in market: tickers = si.tickers_nasdaq()
        elif "S&P" in market: tickers = si.tickers_sp500()
        else: tickers = si.tickers_dow()
    
    # סורק 100 מניות ראשונות למהירות
    target = tickers[:100]
    st.write(f"סורק {len(target)} מניות...")
    
    results = []
    prog = st.progress(0)
    for i, t in enumerate(target):
        res = analyze_stock(t)
        if res: results.append(res)
        prog.progress((i + 1) / len(target))
        
    if results:
        st.table(pd.DataFrame(results).sort_values(by="ציון (0-11)", ascending=False))
    else:
        st.warning("לא נמצאו הזדמנויות כרגע.")
