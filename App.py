import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="סורק 11 האינדיקטורים", layout="wide")

st.title("📊 סורק מניות ארה"ב - מודל 11 האינדיקטורים")

# רשימת המניות לסריקה (אפשר להוסיף כאן כל סימול מהבורסה האמריקאית)
tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'NFLX', 'PYPL']

def analyze_stock(symbol):
    df = yf.download(symbol, period="100d", interval="1d", progress=False)
    if len(df) < 50: return None
    
    # חישוב 11 אינדיקטורים (לפי מה שזיקקנו)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    bbands = ta.bbands(df['Close'], length=20)
    df['BBU'] = bbands['BBU_20_2.0']
    df['BBL'] = bbands['BBL_20_2.0']
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['STOCH'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHk_14_3_3']
    
    last = df.iloc[-1]
    
    # לוגיקה של ניקוד (Score) מתוך 11
    score = 0
    if last['RSI'] < 30: score += 1 # מכירת יתר
    if last['Close'] > last['SMA_50']: score += 1 # מגמה חיובית
    if last['MACD'] > 0: score += 1 # מומנטום חיובי
    if last['Close'] < last['BBL']: score += 1 # חריגה למטה (הזדמנות)
    if last['ADX'] > 25: score += 1 # מגמה חזקה
    # ... כאן נכנסים שאר האינדיקטורים שזיקקנו
    
    recommendation = "HOLD"
    if score >= 7: recommendation = "STRONG BUY 🚀"
    elif score <= 3: recommendation = "STRONG SELL ⚠️"
    
    return {
        "סימול": symbol,
        "מחיר": round(last['Close'], 2),
        "RSI": round(last['RSI'], 1),
        "ציון (0-11)": score,
        "המלצה": recommendation
    }

if st.button('הרץ סריקה חיה'):
    results = []
    for t in tickers:
        res = analyze_stock(t)
        if res: results.append(res)
    
    df_final = pd.DataFrame(results)
    st.dataframe(df_final.style.highlight_between(left=7, right=11, subset=['ציון (0-11)'], color='#90EE90'))
