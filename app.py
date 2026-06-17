import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Nifty 50 Signal Screener", layout="wide")
st.title("Nifty 50 — Technical Signal Screener")
st.caption("Predicts 5-day return direction using RSI, MACD, EMA crossover, P/E and ROE")

with open("classifier.pkl", "rb") as f:
    model = pickle.load(f)

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

ticker = st.selectbox("Select a stock", NIFTY_50)

@st.cache_data(ttl=3600)
def load_precomputed():
    try:
        df = pd.read_csv("precomputed_signals.csv")
        return df
    except FileNotFoundError:
        return None

def get_ticker_data(df, ticker):
    row = df[df["Ticker"] == ticker]
    if row.empty:
        return None
    return row.iloc[0]

signals_df = load_precomputed()

if signals_df is None:
    st.error("Signal data not found. Please run precompute.py first.")
    st.stop()

last_updated = signals_df["Last_Updated"].iloc[0] if "Last_Updated" in signals_df.columns else "Unknown"
st.caption(f"Data last updated: {last_updated} | Next update: weekdays at 4:00 PM IST")

data = get_ticker_data(signals_df, ticker)

if data is None:
    st.warning(f"No data available for {ticker}")
    st.stop()

prediction_input = pd.DataFrame([[
    data["RSI"],
    data["MACD_Hist"],
    data["EMA_Cross"],
    data["PE_Ratio"],
    data["ROE"]
]], columns=["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE"])

prediction = model.predict(prediction_input)[0]
confidence = model.predict_proba(prediction_input)[0][prediction]
direction = "UP" if prediction == 1 else "DOWN"
color = "green" if prediction == 1 else "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current price", f"₹{data['Close']:.2f}")
col2.metric("RSI (14)", f"{data['RSI']:.1f}")
col3.metric("P/E ratio", f"{data['PE_Ratio']:.1f}" if pd.notna(data['PE_Ratio']) else "N/A")
col4.metric("Sector", data["Sector"])

st.divider()

col5, col6 = st.columns(2)
with col5:
    st.subheader("5-day prediction")
    st.markdown(f"### :{color}[{direction}]")
    st.caption(f"Model confidence: {confidence:.1%}")
    st.caption("Baseline accuracy: 51.4% | Model accuracy: 55.96%")

with col6:
    st.subheader("Signal summary")
    rsi_val = data["RSI"]
    rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_signal = "Bullish" if data["MACD_Hist"] > 0 else "Bearish"
    ema_signal = "Uptrend" if data["EMA_Cross"] == 1 else "Downtrend"
    st.write(f"RSI: {rsi_val:.1f} — {rsi_signal}")
    st.write(f"MACD histogram: {'Positive' if data['MACD_Hist'] > 0 else 'Negative'} — {macd_signal}")
    st.write(f"EMA 20/50: {ema_signal}")