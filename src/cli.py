
import os
import sys
import argparse
from datetime import datetime

def print_banner():
    banner = """
+-----------------------------------------------------------------------------+
|          ========================================================           |
|               QuantFinance: Institutional Research & Risk System            |
|          ========================================================           |
+-----------------------------------------------------------------------------+
    """
    print(banner.strip())

def main():
    parser = argparse.ArgumentParser(description='QuantFinance Master Pipeline Manager')

    # Parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--tickers', nargs='+', default=[], help='List of tickers to process')
    parent_parser.add_argument('--start', type=str, default='2000-01-01', help='Start date (YYYY-MM-DD)')
    parent_parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parent_parser.add_argument(
        '--verbosity', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
        help='Logging verbosity level'
    )
    parent_parser.add_argument('--seed', type=int, default=42, help='Global random seed for deterministic runs')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 'run' subcommand
    run_parser = subparsers.add_parser('run', parents=[parent_parser], help='Run the FinMetrica pipeline')
    run_parser.add_argument('--no-backtest', action='store_true', help='Flag to skip walk-forward backtest')
    run_parser.add_argument('--no-stress', action='store_true', help='Flag to skip crisis stress testing')
    run_parser.add_argument('--mode', type=str, choices=['research', 'execute', 'advise'], default='research',
                            help='Operating mode')
    run_parser.add_argument('--capital', type=float, default=100000.0, help='Total capital (AUM) in USD to deploy')
    run_parser.add_argument('--current-holdings', type=str, default='{}', help='JSON string of current ticker holdings')
    run_parser.add_argument('--output-dir', type=str, default='results/', help='Directory to save results JSON and audit trail')
    run_parser.add_argument('--alpaca-key', type=str, default=None, help='Alpaca Paper Trading API Key')
    run_parser.add_argument('--alpaca-secret', type=str, default=None, help='Alpaca Paper Trading Secret Key')
    run_parser.add_argument('--live-trade', action='store_true', help='If set, executes orders on Alpaca Paper Trading')
    run_parser.add_argument('--modules', type=int, nargs='+', default=None, help='Specific module numbers to run (1-9)')

    # 'research' subcommand
    research_parser = subparsers.add_parser('research', parents=[parent_parser], help='Run research experiments (Track 1 & 2)')
    research_parser.add_argument('--track', type=str, choices=['1', '2', 'both'], default='both',
                                 help='Which experiment track to run')
    research_parser.add_argument('--universe-alt', action='store_true', help='Run with the less hindsight-biased alternate universe')

    # Parse arguments
    args, unknown = parser.parse_known_args()
    
    if not args.command:
        # Backward compatibility
        args.command = 'run'
        args.no_backtest = False
        args.no_stress = False
        args.mode = 'research'
        args.capital = 100000.0
        args.current_holdings = '{}'
        args.output_dir = 'results/'
        args.alpaca_key = None
        args.alpaca_secret = None
        args.live_trade = False
        args.modules = None

    if not args.tickers and not getattr(args, 'universe_alt', False):
        parser.error("the following arguments are required: --tickers (unless --universe-alt is used)")
        
    import numpy as np
    import random
    seed = getattr(args, 'seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    # Also push to config cache so anything calling get_config() can access it if needed
    from src.data.config_loader import get_config
    config = get_config()
    if 'system' not in config:
        config['system'] = {}
    config['system']['random_seed'] = seed
    config['random_seed'] = seed

    print_banner()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.command == 'research':
        from research.experiments.generate_report import main as generate_report_main
        from research.data.loader import ResearchDataLoader
        from research.experiments.covariance_regularization_study import run_track1_study
        from research.experiments.full_system_benchmark import run_track2_study
        
        # Handle Universe switching
        universe_name = "Original Demo (Hindsight Biased)"
        output_dir = "research/results"
        
        tickers = args.tickers
        if getattr(args, 'universe_alt', False):
            # Substitute known winners like META, GOOG, AMZN, V 
            # with diverse large caps from mid-2000s that had varied outcomes
            # Reasonings:
            # GE: Mega-cap in 2000s, declined significantly
            # C (Citigroup): Huge in 2000s, crashed in 2008
            # INTC: Dominant tech in 2005, stagnated
            # CSCO: Huge tech, slower growth
            # AIG: Blew up in 2008
            tickers = ['AAPL', 'MSFT', 'GE', 'C', 'INTC', 'JNJ', 'PFE', 'UNH', 'JPM', 'BAC', 'GS', 'WFC', 'XOM', 'CVX', 'COP', 'PG', 'KO', 'HD', 'MCD', 'CSCO']
            universe_name = "Diversified Alternate (Less Hindsight Bias)"
            output_dir = "research/results/alt_universe"
            
        print(f"Running Research Experiments (Track: {getattr(args, 'track', 'both')}) on {universe_name}...")
        
        loader = ResearchDataLoader({})
        price_dict = loader.load_price_data(tickers, start=args.start, end=args.end)
        
        # Convert dictionary of individual price DataFrames into a single price matrix
        import pandas as pd
        prices = pd.DataFrame({ticker: df['Close'] for ticker, df in price_dict.items()})
        
        os.makedirs(output_dir, exist_ok=True)
        
        track = getattr(args, 'track', 'both')
        if track in ['1', 'both']:
            run_track1_study(prices, output_dir=output_dir)
        if track in ['2', 'both']:
            run_track2_study(prices, output_dir=output_dir, fast_mode=True)
            
        generate_report_main(force_rerun=False, fast_mode=True, universe_name=universe_name, start_date=args.start, tickers=tickers, output_dir=output_dir)
        print(f"Research experiments completed. View {output_dir}/FINDINGS.md for results.")
        return

    # Original 'run' functionality
    import json
    current_holdings = {}
    try:
        current_holdings = json.loads(getattr(args, 'current_holdings', '{}'))
    except json.JSONDecodeError:
        pass

    capital = getattr(args, 'capital', 100000.0)
    alpaca_broker = None

    if getattr(args, 'alpaca_key', None) and getattr(args, 'alpaca_secret', None):
        from src.execution.alpaca_broker import AlpacaBroker
        alpaca_broker = AlpacaBroker(args.alpaca_key, args.alpaca_secret)
        
        print(f"Connecting to Alpaca Paper Trading...")
        acc_info = alpaca_broker.get_account_info()
        if acc_info:
            print(f"Connected! Account Value: ${float(acc_info.portfolio_value):,.2f}")
        else:
            print("Failed to connect to Alpaca. Please check your credentials.")
            return

    # --- RUN PIPELINE ---
    from src.pipeline import QuantFinancePipeline
    
    pipeline = QuantFinancePipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end
    )

    modules_to_run = args.modules if getattr(args, 'modules', None) else list(range(1, 10))

    if args.no_backtest and 8 in modules_to_run:
        modules_to_run.remove(8)
    if args.no_stress and 9 in modules_to_run:
        modules_to_run.remove(9)

    pipeline_result = pipeline.run(
        modules=modules_to_run,
        run_backtest=not args.no_backtest,
        run_stress=not args.no_stress,
        mode=args.mode,
        capital=args.capital,
        current_holdings=current_holdings
    )
    pipeline.print_report(pipeline_result)
    pipeline.save_results(pipeline_result, args.output_dir)

    print(f"\nPipeline complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
