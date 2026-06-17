import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
from datetime import datetime, timedelta

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "HCLTECH.NS", "M&M.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BPCL.NS", "BRITANNIA.NS", "GRASIM.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "NESTLEIND.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "UPL.NS"
]

st.set_page_config(page_title="Nifty 50 Signal Screener", layout="wide")
st.title("Nifty 50 — Technical Signal Screener")
st.caption("Predicts 5-day return direction using RSI, MACD, EMA crossover, P/E and ROE")

with open("classifier.pkl", "rb") as f:
    model = pickle.load(f)

ticker = st.selectbox("Select a stock", NIFTY_50)

@st.cache_data(ttl=3600)
def get_data(ticker):
    import time
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            info = stock.info
            if df.empty:
                raise ValueError("Empty data returned")
            df["PE_Ratio"] = info.get("trailingPE", None)
            df["ROE"] = info.get("returnOnEquity", None)
            return df, info
        except Exception as e:
            if attempt < 2:
                time.sleep(2 + attempt * 2)
            else:
                st.error(f"Unable to fetch data for {ticker}. Yahoo Finance rate limit reached. Please try again in a few minutes.")
                st.stop()

df, info = get_data(ticker)

close = df["Close"]

df["RSI"] = compute_rsi(close)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = macd_line - signal_line
df["EMA20"] = close.ewm(span=20, adjust=False).mean()
df["EMA50"] = close.ewm(span=50, adjust=False).mean()
df["EMA_Cross"] = (df["EMA20"] > df["EMA50"]).astype(int)

latest = df.dropna().iloc[-1]

features = pd.DataFrame([[
    latest["RSI"],
    latest["MACD_Hist"],
    latest["EMA_Cross"],
    latest["PE_Ratio"],
    latest["ROE"]
]], columns=["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE"])
prediction = model.predict(features)[0]
confidence = model.predict_proba(features)[0][prediction]

direction = "UP" if prediction == 1 else "DOWN"
color = "green" if prediction == 1 else "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current price", f"₹{latest['Close']:.2f}")
col2.metric("RSI (14)", f"{latest['RSI']:.1f}")
col3.metric("P/E ratio", f"{latest['PE_Ratio']:.1f}" if latest['PE_Ratio'] else "N/A")
col4.metric("Sector", info.get("sector", "N/A"))

st.divider()

col5, col6 = st.columns(2)
with col5:
    st.subheader("5-day prediction")
    st.markdown(f"### :{color}[{direction}]")
    st.caption(f"Model confidence: {confidence:.1%}")
    st.caption("Baseline accuracy: 51.4% | Model accuracy: 55.96%")

with col6:
    st.subheader("Signal summary")
    rsi_val = latest['RSI']
    rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_signal = "Bullish" if latest['MACD_Hist'] > 0 else "Bearish"
    ema_signal = "Uptrend" if latest['EMA_Cross'] == 1 else "Downtrend"
    st.write(f"RSI: {rsi_val:.1f} — {rsi_signal}")
    st.write(f"MACD histogram: {'Positive' if latest['MACD_Hist'] > 0 else 'Negative'} — {macd_signal}")
    st.write(f"EMA 20/50: {ema_signal}")

st.divider()
st.subheader("Price chart — last 1 year")
chart_df = df[["Close", "EMA20", "EMA50"]].copy()
st.line_chart(chart_df)

st.subheader("Recent data")
display_cols = ["Close", "RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE"]
st.dataframe(df[display_cols].dropna().tail(10).round(2), width='stretch')