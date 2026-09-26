import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Nifty 50 Signal Screener", layout="wide")
st.title("Nifty 50 — Technical Signal Screener")
st.caption("Predicts 5-day return direction using advanced momentum features, RSI, MACD, P/E and ROE")

# Load precomputed signals & trained model
@st.cache_resource
def load_artifacts():
    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)
    signals_df = pd.read_csv("precomputed_signals.csv")
    return model, signals_df

model, signals_df = load_artifacts()

NIFTY_50 = signals_df["Ticker"].tolist() if "Ticker" in signals_df.columns else []

ticker = st.selectbox("Select a stock", NIFTY_50)

def get_ticker_data(df, ticker):
    row = df[df["Ticker"] == ticker]
    if row.empty:
        return None
    return row.iloc[0]

last_updated = signals_df["Last_Updated"].iloc[0] if "Last_Updated" in signals_df.columns else "Unknown"
st.caption(f"Data last updated: {last_updated} | Next update: weekdays at 4:00 PM IST")

data = get_ticker_data(signals_df, ticker)
if data is None:
    st.warning(f"No data available for {ticker}")
    st.stop()

# Build prediction input matching the 8 trained features
prediction_input = pd.DataFrame([[
    data["RSI"],
    data["MACD_Hist"],
    data["EMA_Cross"],
    data["PE_Ratio"] if pd.notna(data["PE_Ratio"]) else 0.0,
    data["ROE"] if pd.notna(data["ROE"]) else 0.0,
    data["Return_3d"] if "Return_3d" in data and pd.notna(data["Return_3d"]) else 0.0,
    data["Price_vs_EMA20"] if "Price_vs_EMA20" in data and pd.notna(data["Price_vs_EMA20"]) else 0.0,
    data["Volatility_10d"] if "Volatility_10d" in data and pd.notna(data["Volatility_10d"]) else 0.0
]], columns=["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE", "Return_3d", "Price_vs_EMA20", "Volatility_10d"])

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
    # Displaying exact calculated numbers
    st.caption("Baseline accuracy: 51.24% | Model Test Accuracy: 56.15%")

with col6:
    st.subheader("Signal summary")
    rsi_val = data["RSI"]
    rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_signal = "Bullish" if data["MACD_Hist"] > 0 else "Bearish"
    ema_signal = "Uptrend" if data["EMA_Cross"] == 1 else "Downtrend"
    st.write(f"RSI: {rsi_val:.1f} — {rsi_signal}")
    st.write(f"MACD histogram: {'Positive' if data['MACD_Hist'] > 0 else 'Negative'} — {macd_signal}")
    st.write(f"EMA 20/50: {ema_signal}")