import yfinance as yf
import pandas as pd
import pickle
import time

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

results = []

for ticker in NIFTY_50:
    print(f"Processing {ticker}...")
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            print(f"  Skipping {ticker} — no data")
            continue
        info = stock.info
        close = df["Close"]
        rsi = compute_rsi(close)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema_cross = (ema20 > ema50).astype(int)
        latest_close = float(close.iloc[-1])
        latest_rsi = float(rsi.dropna().iloc[-1])
        latest_macd = float(macd_hist.dropna().iloc[-1])
        latest_ema_cross = int(ema_cross.iloc[-1])
        pe = info.get("trailingPE", None)
        roe = info.get("returnOnEquity", None)
        sector = info.get("sector", "Unknown")
        results.append({
            "Ticker": ticker,
            "Close": round(latest_close, 2),
            "RSI": round(latest_rsi, 2),
            "MACD_Hist": round(latest_macd, 4),
            "EMA_Cross": latest_ema_cross,
            "PE_Ratio": round(pe, 2) if pe else None,
            "ROE": round(roe, 4) if roe else None,
            "Sector": sector,
            "Last_Updated": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
        print(f"  Done — RSI: {latest_rsi:.1f}, Close: ₹{latest_close:.2f}")
    except Exception as e:
        print(f"  Error for {ticker}: {e}")
        continue

output = pd.DataFrame(results)
output.to_csv("precomputed_signals.csv", index=False)
print(f"\nSaved {len(results)} stocks to precomputed_signals.csv")
print(output[["Ticker", "Close", "RSI", "MACD_Hist", "EMA_Cross"]].to_string())