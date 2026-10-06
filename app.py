import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Nifty 50 Signal Screener", layout="wide")
st.title("Nifty 50 — Technical Signal Screener")
st.caption("Predicts 5-day return direction using advanced momentum features, RSI, MACD, P/E and ROE")

# Load model and precomputed signals safely
@st.cache_resource
def load_artifacts():
    if not os.path.exists("classifier.pkl") or not os.path.exists("precomputed_signals.csv"):
        return None, None
    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)
    signals_df = pd.read_csv("precomputed_signals.csv")
    return model, signals_df

model, signals_df = load_artifacts()

if model is None or signals_df is None:
    st.error("Model artifacts or precomputed signals missing. Please run precompute.py / model.py.")
    st.stop()

# Standardize column names to avoid case mismatches
signals_df.columns = [col.strip() for col in signals_df.columns]
if "ticker" in signals_df.columns and "Ticker" not in signals_df.columns:
    signals_df.rename(columns={"ticker": "Ticker"}, inplace=True)
if "close" in signals_df.columns and "Close" not in signals_df.columns:
    signals_df.rename(columns={"close": "Close"}, inplace=True)

NIFTY_50 = signals_df["Ticker"].tolist() if "Ticker" in signals_df.columns else []

ticker = st.selectbox("Select a stock", NIFTY_50)

def get_ticker_data(df, ticker):
    row = df[df["Ticker"] == ticker]
    if row.empty:
        return None
    return row.iloc[0]

last_updated = signals_df["Last_Updated"].iloc[0] if "Last_Updated" in signals_df.columns else "Unknown"
st.caption(f"Data last updated: {last_updated} | Next update: weekdays at 4:00 PM IST (Post-Market Close)")

data = get_ticker_data(signals_df, ticker)
if data is None:
    st.warning(f"No data available for {ticker}")
    st.stop()

# Extract price safely (handling case variants)
close_price = data.get("Close", data.get("close", None))
price_str = f"₹{close_price:.2f}" if pd.notna(close_price) else "N/A"

rsi_val = data.get("RSI", data.get("rsi", 50.0))
pe_val = data.get("PE_Ratio", data.get("pe_ratio", None))
sector_val = data.get("Sector", data.get("sector", "Unknown"))
macd_hist = data.get("MACD_Hist", data.get("macd_hist", 0.0))
ema_cross = data.get("EMA_Cross", data.get("ema_cross", 0))

# Build prediction input matching the 8 trained features
prediction_input = pd.DataFrame([[
    rsi_val,
    macd_hist,
    ema_cross,
    pe_val if pd.notna(pe_val) else 0.0,
    data.get("ROE", 0.0) if pd.notna(data.get("ROE", 0.0)) else 0.0,
    data.get("Return_3d", 0.0) if pd.notna(data.get("Return_3d", 0.0)) else 0.0,
    data.get("Price_vs_EMA20", 0.0) if pd.notna(data.get("Price_vs_EMA20", 0.0)) else 0.0,
    data.get("Volatility_10d", 0.0) if pd.notna(data.get("Volatility_10d", 0.0)) else 0.0
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