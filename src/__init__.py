"""
QuantFinance-EDU
================
A seven-module quantitative financial analysis system.

Usage
-----
  python -m src.cli --tickers AAPL MSFT GOOGL --start 2018-01-01
  
Or import as a library:
  from src.pipeline import QuantFinancePipeline
  pipeline = QuantFinancePipeline(tickers=["AAPL","MSFT"], start="2018-01-01")
  results = pipeline.run()
"""
