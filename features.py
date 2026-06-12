import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features(df):
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.copy()

        close = group["Close"]

        group["RSI"] = compute_rsi(close)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        group["MACD_Hist"] = macd_line - signal_line

        group["EMA20"] = close.ewm(span=20, adjust=False).mean()
        group["EMA50"] = close.ewm(span=50, adjust=False).mean()
        group["EMA_Cross"] = (group["EMA20"] > group["EMA50"]).astype(int)

        group["Return_5d"] = close.shift(-5) / close - 1
        group["Target"] = (group["Return_5d"] > 0).astype(int)

        results.append(group)

    final = pd.concat(results)

    features = ["RSI", "MACD_Hist", "EMA_Cross", "PE_Ratio", "ROE"]
    final = final.dropna(subset=features + ["Target"])

    final.to_csv("features.csv", index=False)
    print(f"Done. {len(final)} rows saved to features.csv")
    print(f"Target distribution:\n{final['Target'].value_counts()}")

df = pd.read_csv("raw_data.csv")
compute_features(df)