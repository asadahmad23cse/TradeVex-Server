import yfinance as yf

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "LT.NS",
    "TATAMOTORS.NS", "MARUTI.NS", "ADANIENT.NS", "SUNPHARMA.NS", "DRREDDY.NS", "BAJAJ-AUTO.NS",
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN",
    "META", "AMD", "GOOGL", "COST", "LLY"
]

print("=== Ticker Verification ===")
failed = []
for t in tickers:
    try:
        data = yf.download(t, period="3d", progress=False)
        status = "OK" if len(data) > 0 else "NO DATA"
        if status == "NO DATA":
            failed.append(t)
    except Exception as e:
        status = f"ERROR: {e}"
        failed.append(t)
    print(f"  {t:20} -> {status}")

print(f"\nPassed: {len(tickers)-len(failed)}/21")
if failed:
    print(f"Failed: {failed}")
else:
    print("All tickers verified!")
