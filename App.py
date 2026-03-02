import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# הגדרות עמוד
st.set_page_config(page_title="סורק 11 האינדיקטורים", layout="wide")

# כותרת - השתמשתי בגרש בודד כדי למנוע את השגיאה שראית
st.title('📊 סורק מניות ארהב - מודל 11 האינדיקטורים')

# רשימת מניות
tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'NFLX', 'PYPL']

def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period="150d", interval="1d", progress=False)
        if len(df) < 50: return None
        
        # חישוב אינדיקטורים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        bbands = ta.bbands(df['Close'], length=20)
        df['BBU'] = bbands['BBU_20_2.0']
        df['BBL'] = bbands['BBL_20_2.0']
        adx_df = ta.adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx_df['ADX_14']
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['STOCH'] = stoch['STOCHk_14_3_3']
        
        last = df.iloc[-1]
        
        # לוגיקה של ניקוד (0-11)
        score = 0
        if last['RSI'] < 30: score += 1
        if last['RSI'] > 70: score -= 1 # הורדת ניקוד על קניית יתר
        if last['Close'] > last['SMA_50']: score += 1
        if last['MACD'] > 0: score += 1
        if last['Close'] < last['BBL']: score += 1
        if last['ADX'] > 25: score += 1
        if last['STOCH'] < 20: score += 1
        if last['Close'] > last['SMA_200']: score += 1
        
        recommendation = "המתנה"
        if score >= 5: recommendation = "קנייה חזקה 🚀"
        elif score <= 2: recommendation = "מכירה ⚠️"
        
        return {
            "סימול": symbol,
            "מחיר": round(float(last['Close']), 2),
            "RSI": round(float(last['RSI']), 1),
            "ציון (0-11)": score,
            "המלצה": recommendation
        }
    except Exception as e:
        return None

if st.button('הרץ סריקה חיה'):
    with st.spinner('מעבד נתונים...'):
        results = []
        for t in tickers:
            res = analyze_stock(t)
            if res: results.append(res)
        
        if results:
            df_final = pd.DataFrame(results)
            st.table(df_final)
        else:
            st.error("לא הצלחתי למשוך נתונים. נסה שוב בעוד כמה דקות.")
