import yfinance as yf
import pandas as pd

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "HCLTECH.NS", "M&M.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BPCL.NS", "BRITANNIA.NS", "GRASIM.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "NESTLEIND.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "UPL.NS", "LTIM.NS"
]

def fetch_all():
    all_data = []

    for ticker in NIFTY_50:
        print(f"Fetching {ticker}...")

        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")

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

fetch_all()