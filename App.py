import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from yahoo_fin import stock_info as si

st.set_page_config(page_title="סורק השוק המלא", layout="wide")

st.title('🔍 סורק מניות ארה"ב - המודל המזוקק')

# בחירת שוק לסריקה
market_choice = st.selectbox("בחר איזה שוק לסרוק:", ["NASDAQ (טכנולוגיה וצמיחה)", "S&P 500 (החברות הגדולות)", "DOW JONES"])

def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period="150d", interval="1d", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 11 אינדיקטורים
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
        
        # לוגיקת ניקוד 11 נקודות
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

        return {
            "סימול": symbol,
            "מחיר": round(float(last['Close']), 2),
            "ציון (0-11)": score,
            "RSI": round(float(last['RSI']), 1),
            "מצב": "🔥 קנייה חזקה" if score >= 8 else "מעקב" if score >= 6 else "המתנה"
        }
    except:
        return None

if st.button('🚀 התחל סריקה חיה'):
    with st.spinner('מושך רשימת מניות מהשוק...'):
        if market_choice == "NASDAQ (טכנולוגיה וצמיחה)":
            tickers = si.tickers_nasdaq()
        elif market_choice == "S&P 500 (החברות הגדולות)":
            tickers = si.tickers_sp500()
        else:
            tickers = si.tickers_dow()

    # נסרוק את 150 המניות הראשונות ברשימה כדי לשמור על מהירות
    # אפשר להגדיל את המספר הזה בזהירות
    target_tickers = tickers[:150]
    
    st.write(f"סורק {len(target_tickers)} מניות מתוך רשימת {market_choice}...")
    progress_bar = st.progress(0)
    results = []
    
    for i, t in enumerate(target_tickers):
        res = analyze_stock(t)
        if res and res['ציון (0-11)'] >= 6: # מציג רק תוצאות מעניינות
            results.append(res)
        progress_bar.progress((i + 1) / len(target_tickers))
        
    if results:
        df_res = pd.DataFrame(results).sort_values(by="ציון (0-11)", ascending=False)
        st.table(df_res)
    else:
        st.info("לא נמצאו מניות שעומדות בקריטריונים כרגע.")
