import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.copy()
        close = group["Close"]

        # Technical Indicators
        group["RSI"] = compute_rsi(close)
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        group["MACD_Hist"] = macd_line - signal_line

        group["EMA20"] = close.ewm(span=20, adjust=False).mean()
        group["EMA50"] = close.ewm(span=50, adjust=False).mean()
        group["EMA_Cross"] = (group["EMA20"] > group["EMA50"]).astype(int)

        # Advanced Predictive Momentum Features
        group["Return_3d"] = close.pct_change(3)
        group["Price_vs_EMA20"] = (close - group["EMA20"]) / group["EMA20"]
        group["Volatility_10d"] = close.pct_change().rolling(10).std()

        # Target: 5-day future return direction
        group["Return_5d"] = close.shift(-5) / close - 1
        group["Target"] = (group["Return_5d"] > 0).astype(int)

        results.append(group)

    final = pd.concat(results)
    features = ["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE", "Return_3d", "Price_vs_EMA20", "Volatility_10d"]
    final = final.dropna(subset=features + ["Target"])

    final.to_csv("features.csv", index=False)
    print(f"Done. {len(final)} rows saved with enhanced features.")

if __name__ == "__main__":
    df = pd.read_csv("raw_data.csv")
    compute_features(df)