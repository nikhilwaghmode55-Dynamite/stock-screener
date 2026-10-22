import yfinance as yf
ticker = yf.Ticker("RELIANCE.NS")
df = ticker.history(period="3d")
print(df[["Close"]])
print("Last available date on Yahoo Finance:", df.index[-1].strftime("%Y-%m-%d"))