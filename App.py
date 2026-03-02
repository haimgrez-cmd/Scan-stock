import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="סורק שוק מקצועי", layout="wide")

st.title('⚡ סורק שוק עוצמתי - 11 אינדיקטורים')

# רשימה מובנית של כ-150 מניות מובילות (למניעת שגיאות משיכה)
STOCKS_TO_SCAN = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'PFE', 'PEP', 'KO',
    'BAC', 'COST', 'TMO', 'AVGO', 'CSCO', 'ACN', 'ABT', 'ADBE', 'MRK', 'DIS', 'DHR',
    'LIN', 'NEE', 'ORCL', 'VZ', 'TXN', 'HON', 'PM', 'MS', 'RTX', 'AMV', 'IBM', 'QCOM',
    'INTC', 'CAT', 'SBUX', 'LOW', 'AMD', 'SPGI', 'GS', 'PLD', 'NFLX', 'INTU', 'BLK',
    'T', 'GE', 'ISRG', 'MDLZ', 'GILD', 'BKNG', 'AXP', 'SYK', 'CVS', 'AMT', 'ADI',
    'DE', 'MO', 'TJX', 'MMC', 'LMT', 'CB', 'LRCX', 'ZTS', 'EL', 'CI', 'NOW', 'ADP',
    'BDX', 'C', 'VRTX', 'SLB', 'EW', 'BSX', 'REGN', 'ITW', 'HCA', 'HUM', 'TGT', 'WM',
    'DUK', 'PNC', 'FISV', 'MU', 'ATVI', 'ORLY', 'MCD', 'MMM', 'CL', 'SHW', 'CSX', 'NSC',
    'F', 'GM', 'USB', 'ELV', 'PYPL', 'PANW', 'SNPS', 'CDNS', 'MCHP', 'ON', 'MDB', 'SQ',
    'COIN', 'SHOP', 'NET', 'DDOG', 'U', 'DKNG', 'PLTR', 'SNOW', 'AFRM', 'RIVN', 'LCID'
]

def analyze_batch(tickers):
    # הורדה קבוצתית מהירה מאוד
    data = yf.download(tickers, period="150d", interval="1d", group_by='ticker', progress=False)
    results = []
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < 50: continue
            
            # ניקוי כותרות אם צריך
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

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
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # לוגיקת הניקוד (0-11)
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

            if score >= 7:
                results.append({
                    "סימול": ticker,
                    "מחיר": round(float(last['Close']), 2),
                    "ציון (0-11)": score,
                    "RSI": round(float(last['RSI']), 1),
                    "מצב": "💎 קנייה חזקה" if score >= 9 else "🚀 מעקב חיובי"
                })
        except:
            continue
    return results

if st.button('🚀 הרץ סריקה רחבה על השוק'):
    st.write(f"סורק {len(STOCKS_TO_SCAN)} מניות נבחרות (Large & Mid Caps)...")
    progress_bar = st.progress(0)
    
    # חלוקה לקבוצות למהירות שיא
    all_results = []
    batch_size = 30
    for i in range(0, len(STOCKS_TO_SCAN), batch_size):
        batch = STOCKS_TO_SCAN[i : i + batch_size]
        batch_res = analyze_batch(batch)
        all_results.extend(batch_res)
        progress_bar.progress(min((i + batch_size) / len(STOCKS_TO_SCAN), 1.0))
    
    if all_results:
        final_df = pd.DataFrame(all_results).sort_values(by="ציון (0-11)", ascending=False)
        st.success(f"הסריקה הושלמה! נמצאו {len(all_results)} הזדמנויות.")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.info("לא נמצאו מניות עם ציון גבוה כרגע. נסה שוב מאוחר יותר.")
