import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="סורק שוק מקצועי", layout="wide")

st.title('⚡ סורק שוק עוצמתי - 11 אינדיקטורים')

@st.cache_data
def get_broad_market():
    # משיכת רשימת S&P 500 המלאה מוויקיפדיה כמקור בסיס רחב
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]['Symbol'].tolist()

def run_heavy_scan(tickers):
    results = []
    # הורדת נתונים בקבוצות של 50 לשיפור מהירות דרמטי
    data = yf.download(tickers, period="150d", interval="1d", group_by='ticker', progress=False)
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < 50: continue
            
            # חישוב 11 האינדיקטורים המזוקקים שלנו
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

            if score >= 7: # מציג רק את הטופ של הטופ
                results.append({
                    "סימול": ticker,
                    "מחיר": round(float(last['Close']), 2),
                    "ציון (0-11)": score,
                    "RSI": round(float(last['RSI']), 1),
                    "ווליום חריג?": "✅" if last['Volume'] > df['Volume'].tail(20).mean() * 1.5 else "❌"
                })
        except:
            continue
    return results

if st.button('🚀 הרץ סריקה על כל ה-S&P 500'):
    all_tickers = get_broad_market()
    st.write(f"מתחיל סריקה עמוקה על {len(all_tickers)} מניות...")
    
    # מחלקים לנגלות של 50 כדי לא לחסום את ה-API
    all_results = []
    progress_bar = st.progress(0)
    
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        batch_results = run_heavy_scan(batch)
        all_results.extend(batch_results)
        progress_bar.progress(min((i + batch_size) / len(all_tickers), 1.0))
    
    if all_results:
        final_df = pd.DataFrame(all_results).sort_values(by="ציון (0-11)", ascending=False)
        st.success(f"סיום! נמצאו {len(all_results)} מניות עם פוטנציאל גבוה.")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.warning("אין מניות שעונות על הקריטריונים כרגע.")
