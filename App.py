import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# הגדרות עמוד
st.set_page_config(page_title="סורק מניות", layout="wide")

st.title('📈 סורק 11 האינדיקטורים - חי')

# רשימת מניות מצומצמת לבדיקה ראשונית
tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META']

def analyze_stock(symbol):
    try:
        # משיכת נתונים בצורה פשוטה יותר
        df = yf.download(symbol, period="200d", interval="1d", progress=False)
        
        if df.empty or len(df) < 50:
            return None
            
        # תיקון למבנה הנתונים (מניעת שגיאת MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # חישוב אינדיקטורים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0] # לוקח את העמודה הראשונה של ה-MACD
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20)
        df['BBL'] = bbands.iloc[:, 0] # Lower band
        
        last_price = float(df['Close'].iloc[-1])
        last_rsi = float(df['RSI'].iloc[-1])
        
        # לוגיקה של ניקוד (Score)
        score = 0
        if last_rsi < 30: score += 2  # מכירת יתר - חזק
        if last_rsi > 70: score -= 2  # קניית יתר
        if last_price > float(df['SMA_50'].iloc[-1]): score += 1
        if float(df['MACD'].iloc[-1]) > 0: score += 1
        if last_price < float(df['BBL'].iloc[-1]): score += 2 # פריצה למטה
        
        res_text = "המתנה"
        if score >= 4: res_text = "קנייה 🟢"
        if score <= 0: res_text = "מכירה 🔴"

        return {
            "סימול": symbol,
            "מחיר": round(last_price, 2),
            "RSI": round(last_rsi, 1),
            "ציון": score,
            "שורה תחתונה": res_text
        }
    except Exception as e:
        return None

if st.button('🚀 הרץ סריקה עכשיו'):
    with st.spinner('מושך נתונים מהבורסה...'):
        results = []
        for t in tickers:
            res = analyze_stock(t)
            if res:
                results.append(res)
        
        if results:
            final_df = pd.DataFrame(results)
            st.table(final_df)
        else:
            st.warning("השרת של Yahoo Finance חסום זמנית או שאין נתונים. נסה ללחוץ שוב.")
