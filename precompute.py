import yfinance as yf
import pandas as pd
import pickle
import time

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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

results = []

for ticker in NIFTY_50:
    print(f"Processing {ticker}...")
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
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
        
        return_3d = close.pct_change(3)
        price_vs_ema20 = (close - ema20) / ema20
        volatility_10d = close.pct_change().rolling(10).std()
        
        results.append({
            "Ticker": ticker,
            "Close": round(float(close.iloc[-1]), 2),
            "RSI": round(float(rsi.dropna().iloc[-1]), 2),
            "MACD_Hist": round(float(macd_hist.dropna().iloc[-1]), 4),
            "EMA_Cross": int(ema_cross.iloc[-1]),
            "PE_Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "ROE": round(info.get("returnOnEquity"), 4) if info.get("returnOnEquity") else None,
            "Return_3d": round(float(return_3d.dropna().iloc[-1]), 4),
            "Price_vs_EMA20": round(float(price_vs_ema20.dropna().iloc[-1]), 4),
            "Volatility_10d": round(float(volatility_10d.dropna().iloc[-1]), 4),
            "Sector": info.get("sector", "Unknown"),
            "Last_Updated": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    except Exception as e:
        print(f"Error for {ticker}: {e}")

output = pd.DataFrame(results)
output.to_csv("precomputed_signals.csv", index=False)
print(f"Saved {len(results)} stocks to precomputed_signals.csv")