python -m src.cli research --track both --tickers AAPL MSFT GOOG AMZN META JNJ PFE UNH JPM BAC GS WFC XOM CVX COP PG KO HD MCD V --start 2005-01-01
cp research/results/FINDINGS.md research/results/determinism_check_run1.md
python -m src.cli research --track both --tickers AAPL MSFT GOOG AMZN META JNJ PFE UNH JPM BAC GS WFC XOM CVX COP PG KO HD MCD V --start 2005-01-01
cp research/results/FINDINGS.md research/results/determinism_check_run2.md
python scripts/diff_determinism.py
