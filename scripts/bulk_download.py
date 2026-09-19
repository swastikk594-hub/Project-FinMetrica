import os
import sys
import logging
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Representative diverse universe across Tech, Semi, Finance, Health, Retail, Energy, Auto, Media, Crypto ETF, Indices
TICKERS = [
    # Mega Tech & Software
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'NOW',
    # Semis & Hardware
    'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TSM', 'ASML', 'MU',
    # Growth & AI / Cloud
    'PLTR', 'SNOW', 'CRWD', 'PANW', 'UBER', 'ABNB',
    # Autos & Clean Energy
    'TSLA', 'RIVN', 'ENPH', 'F', 'GM',
    # Financials & FinTech
    'JPM', 'BAC', 'GS', 'MS', 'C', 'V', 'MA', 'PYPL', 'SQ',
    # Healthcare & Pharma
    'JNJ', 'UNH', 'LLY', 'NVO', 'PFE', 'MRK', 'ABBV', 'TMO',
    # Consumer, Retail & Dining
    'WMT', 'COST', 'TGT', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS',
    # Industrials, Aerospace, Materials
    'CAT', 'DE', 'BA', 'GE', 'HON', 'LMT', 'RTX',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Benchmark ETFs
    'SPY', 'QQQ', 'DIA', 'IWM', 'SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'GLD', 'TLT', 'IBIT'
]

DATA_DIR = os.path.join(os.getcwd(), 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

def sanitize_and_save(ticker: str):
    csv_path = os.path.join(DATA_DIR, f"{ticker}_price.csv")
    try:
        logging.info(f"Fetching real market data for: {ticker}...")
        df = yf.download(ticker, start="2015-01-01", auto_adjust=False, progress=False)
        if df.empty:
            logging.warning(f"No price data found for {ticker}")
            return False
            
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure Adj Close and Close exist
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        elif 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna()
        df.to_csv(csv_path)
        logging.info(f"Successfully saved {len(df)} rows for {ticker} -> {csv_path}")
        return True
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return False

def main():
    unique_tickers = sorted(list(set(TICKERS)))
    print(f"Starting bulk download for {len(unique_tickers)} tickers...")
    success = 0
    for t in unique_tickers:
        if sanitize_and_save(t):
            success += 1
    print(f"\nDone! Successfully downloaded and cached {success}/{len(unique_tickers)} tickers into {DATA_DIR}")

if __name__ == '__main__':
    main()
