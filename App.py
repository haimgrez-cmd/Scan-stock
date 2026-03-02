import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="סורק המניות המלא", layout="wide")

st.title('🔍 סורק השוק המלא - 11 אינדיקטורים')

# פונקציות יציבות למשיכת רשימות מניות
@st.cache_data
def get_tickers(market):
    try:
        if market == "S&P 500 (גדולות)":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            return pd.read_html(url)[0]['Symbol'].tolist()
        elif market == "DOW JONES":
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            return pd.read_html(url)[1]['Symbol'].tolist()
        else: # NASDAQ - נשתמש ברשימה קבועה של הגדולות כדי למנוע קריסה
            return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'PEP', 'COST', 'ADBE', 'AZN', 'CSCO', 'AMD']
    except:
        return ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN'] # רשימת גיבוי

market_choice = st.selectbox("בחר שוק לסריקה:", ["S&P 500 (גדולות)", "DOW JONES", "NASDAQ (המובילות)"])

def analyze_stock(symbol):
    try:
        # החלפת נקודה במקף עבור סימולים כמו BRK.B
        symbol = symbol.replace('.', '-')
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

        if score >= 6:
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
    tickers = get_tickers(market_choice)
    # נסרוק עד 100 מניות ראשונות
    target = tickers[:100]
    
    st.write(f"סורק {len(target)} מניות מתוך {market_choice}...")
    prog = st.progress(0)
    results = []
    
    for i, t in enumerate(target):
        res = analyze_stock(t)
        if res: results.append(res)
        prog.progress((i + 1) / len(target))
        
    if results:
        st.table(pd.DataFrame(results).sort_values(by="ציון (0-11)", ascending=False))
    else:
        st.warning("לא נמצאו הזדמנויות כרגע.")
