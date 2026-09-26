import yfinance as yf
import pandas as pd

# Updated Real-Time Nifty 50 Ticker List
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

def fetch_all():
    all_data = []

    for ticker in NIFTY_50:
        print(f"Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        # Fetching historical data strictly from April 2025 to April 2026
        df = stock.history(start="2025-04-01", end="2026-04-30")

        if df.empty:
            print(f"  Skipping {ticker} — no data returned")
            continue

        df["Ticker"] = ticker

        info = stock.info
        df["PE_Ratio"] = info.get("trailingPE", None)
        df["ROE"] = info.get("returnOnEquity", None)
        df["Sector"] = info.get("sector", "Unknown")

        all_data.append(df)

    combined = pd.concat(all_data)
    combined.to_csv("raw_data.csv")
    print(f"\nDone. Saved {len(all_data)} stocks to raw_data.csv")

if __name__ == "__main__":
    fetch_all()