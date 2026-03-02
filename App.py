import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="סורק שוק רחב", layout="wide")

st.title('🔍 סורק מניות ארה"ב המלא - 11 אינדיקטורים')
st.write("סורק מניות קטנות וגדולות (Russell 3000) לפי המודל המזוקק שלנו.")

@st.cache_data
def get_all_tickers():
    # משיכת רשימה רחבה מאוד של כ-3,000 מניות מארה"ב
    url = "https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7625d24ce021dd68adff358211033951/nasdaq-listed_csv.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

def analyze_stock(symbol):
    try:
        # משיכת נתונים (פחות ימים כדי להאיץ את הסריקה)
        df = yf.download(symbol, period="100d", interval="1d", progress=False)
        if df.empty or len(df) < 40: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # סינון ראשוני למניות קטנות: מחיר מעל 2$ ונפח מסחר מינימלי
        last_price = float(df['Close'].iloc[-1])
        avg_volume = df['Volume'].tail(10).mean()
        if last_price < 2 or avg_volume < 100000: return None

        # 11 האינדיקטורים המזוקקים
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
        
        prev = df.iloc[-2]
        last = df.iloc[-1]
        
        # לוגיקת הניקוד (0-11)
        score = 0
        if last['RSI'] < 35: score += 1
        if last['Close'] > last['SMA50']: score += 1
        if last['MACD'] > 0: score += 1
        if last['Close'] < last['BBL'] * 1.02: score += 1 # קרוב לרצועה תחתונה
        if last['STOCHk'] < 25: score += 1
        if last['ADX'] > 20: score += 1
        if last['Close'] > last['SMA200']: score += 1
        if last['Close'] > last['EMA20']: score += 1
        if last['MACD'] > prev['MACD']: score += 1
        if last['RSI'] > prev['RSI']: score += 1
        if last['Volume'] > df['Volume'].tail(20).mean() * 1.5: score += 1 # פריצה בנפח מסחר

        if score >= 8: # מציג רק "פצצות" פוטנציאליות
            return {
                "סימול": symbol,
                "מחיר": round(last_price, 2),
                "ציון (0-11)": score,
                "RSI": round(float(last['RSI']), 1),
                "ווליום": int(last['Volume']),
                "המלצה": "🔥 הזדמנות חזקה" if score >= 9 else "מעקב צמוד"
            }
    except:
        return None

if st.button('🚀 התחל סריקה רחבה (אלפי מניות)'):
    all_symbols = get_all_tickers()
    # נתחיל בסריקה של 200 הראשונות כדי לא לתקוע את השרת, אפשר להגדיל בהדרגה
    test_symbols = all_symbols[:300] 
    
    st.write(f"סורק {len(test_symbols)} מניות נבחרות מהשוק הרחב...")
    progress_bar = st.progress(0)
    results = []
    
    for i, t in enumerate(test_symbols):
        res = analyze_stock(t)
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(test_symbols))
        
    if results:
        final_df = pd.DataFrame(results).sort_values(by="ציון (0-11)", ascending=False)
        st.success(f"נמצאו {len(results)} מניות מעניינות!")
        st.dataframe(final_df)
    else:
        st.warning("לא נמצאו הזדמנויות ברגע זה. נסה שוב מאוחר יותר.")
