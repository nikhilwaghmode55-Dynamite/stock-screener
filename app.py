import streamlit as st
import pandas as pd
import pickle
import os
import yfinance as yf
from datetime import datetime, timedelta, time, timezone

st.set_page_config(page_title="Nifty 50 Signal Screener", layout="wide")
st.title("Nifty 50 — Technical Signal Screener")
st.caption("Predicts 5-day return direction using advanced momentum features, RSI, MACD, P/E and ROE")

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "HCLTECH.NS", "M&M.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HINDALCO.NS", "NESTLEIND.NS", 
    "SBILIFE.NS", "HDFCLIFE.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", 
    "TATACONSUM.NS", "ETERNAL.NS", "TRENT.NS", "JIOFIN.NS", 
    "MAXHEALTH.NS", "INDIGO.NS", "BEL.NS", "SHRIRAMFIN.NS", "TMPV.NS"
]

@st.cache_resource
def load_model():
    if not os.path.exists("classifier.pkl"):
        return None
    with open("classifier.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
if model is None:
    st.error("Model artifacts missing. Please ensure classifier.pkl is present.")
    st.stop()

@st.cache_data(ttl=3600)
def get_live_signals(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            return None
            
        info = stock.info
        close = df["Close"]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema_cross = (ema20 > ema50).astype(int)
        
        return_3d = close.pct_change(3)
        price_vs_ema20 = (close - ema20) / ema20
        volatility_10d = close.pct_change().rolling(10).std()
        
        last_market_date = df.index[-1].strftime("%Y-%m-%d")
        
        return {
            "Ticker": ticker,
            "Close": float(close.iloc[-1]),
            "RSI": float(rsi.dropna().iloc[-1]),
            "MACD_Hist": float(macd_hist.dropna().iloc[-1]),
            "EMA_Cross": int(ema_cross.iloc[-1]),
            "PE_Ratio": info.get("trailingPE", None),
            "ROE": info.get("returnOnEquity", None),
            "Return_3d": float(return_3d.dropna().iloc[-1]),
            "Price_vs_EMA20": float(price_vs_ema20.dropna().iloc[-1]),
            "Volatility_10d": float(volatility_10d.dropna().iloc[-1]),
            "Sector": info.get("sector", "Unknown"),
            "Last_Updated": last_market_date
        }
    except Exception:
        return None

ticker = st.selectbox("Select a stock", NIFTY_50)
data = get_live_signals(ticker)

IST = timezone(timedelta(hours=5, minutes=30))

def get_next_trading_day_str():
    now = datetime.now(IST)
    weekday = now.weekday()
    market_close_time = time(16, 0)
    
    if weekday < 5:
        if now.time() < market_close_time:
            next_day = now
        else:
            if weekday == 4:
                next_day = now + timedelta(days=3)
            else:
                next_day = now + timedelta(days=1)
    elif weekday == 5:
        next_day = now + timedelta(days=2)
    else:
        next_day = now + timedelta(days=1)
        
    return next_day.strftime("%d/%m/%Y")

next_update_date = get_next_trading_day_str()

if data is None:
    st.warning(f"No data available for {ticker}")
    st.stop()

last_updated = data.get("Last_Updated", "Unknown")
st.caption(f"Data last updated: {last_updated} | Next update: {next_update_date} at 4:00 PM IST (Post-Market Close)")

close_price = data.get("Close", None)
price_str = f"₹{close_price:.2f}" if pd.notna(close_price) else "N/A"

rsi_val = data.get("RSI", 50.0)
pe_val = data.get("PE_Ratio", None)
sector_val = data.get("Sector", "Unknown")
macd_hist = data.get("MACD_Hist", 0.0)
ema_cross = data.get("EMA_Cross", 0)

prediction_input = pd.DataFrame([[
    rsi_val,
    macd_hist,
    ema_cross,
    pe_val if pd.notna(pe_val) else 0.0,
    data.get("ROE") if pd.notna(data.get("ROE")) else 0.0,
    data.get("Return_3d") if pd.notna(data.get("Return_3d")) else 0.0,
    data.get("Price_vs_EMA20") if pd.notna(data.get("Price_vs_EMA20")) else 0.0,
    data.get("Volatility_10d") if pd.notna(data.get("Volatility_10d")) else 0.0
]], columns=["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE", "Return_3d", "Price_vs_EMA20", "Volatility_10d"])

prediction = model.predict(prediction_input)[0]
confidence = model.predict_proba(prediction_input)[0][prediction]
direction = "UP" if prediction == 1 else "DOWN"
color = "green" if prediction == 1 else "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current price", price_str)
col2.metric("RSI (14)", f"{rsi_val:.1f}")
col3.metric("P/E ratio", f"{pe_val:.1f}" if pd.notna(pe_val) else "N/A")
col4.metric("Sector", sector_val)

st.divider()

col5, col6 = st.columns(2)
with col5:
    st.subheader("5-day prediction")
    st.markdown(f"### :{color}[{direction}]")
    st.caption(f"Model confidence: {confidence:.1%}")
    st.caption("Baseline accuracy: 51.24% | Model Test Accuracy: 56.15%")

with col6:
    st.subheader("Signal summary")
    rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_signal = "Bullish" if macd_hist > 0 else "Bearish"
    ema_signal = "Uptrend" if ema_cross == 1 else "Downtrend"
    st.write(f"RSI: {rsi_val:.1f} — {rsi_signal}")
    st.write(f"MACD histogram: {'Positive' if macd_hist > 0 else 'Negative'} — {macd_signal}")
    st.write(f"EMA 20/50: {ema_signal}")