"""
Research Experiment Runner
==========================
Orchestrates the full research pipeline:
  raw data -> PIT store -> raw characteristics -> normalization -> factor scores
  -> Fama-MacBeth regression -> incremental R2 -> robustness diagnostics -> results

Usage:
  python -m research.experiments.run --config research/config/protocol.yaml
  python -m research.experiments.run --config research/config/protocol.yaml --normalization cross_sectional_zscore
"""
import argparse
import datetime
import logging
import os
import json
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Import subsystem components — class-based and registry-based APIs
from research.config.loader import load_protocol, get_normalizations, get_characteristics
from research.data.loader import ResearchDataLoader
from research.normalization import apply_normalization, get_normalizer
from research.factors import FACTOR_REGISTRY
from research.predictive.linear_fama_macbeth import FamaMacBeth
from research.robustness.incremental import RobustnessDiagnostics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Research Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment protocol YAML file"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        action="append",
        help="Optional: Run only specific normalization(s). Can be repeated."
    )
    return parser.parse_args()


def generate_experiment_id() -> str:
    """
    Generate a unique experiment ID based on the current timestamp.

    Returns
    -------
    str
        Timestamp in YYYYMMDD_HHMMSS format.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_forward_returns(
    price_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    as_of: pd.Timestamp,
    horizon_days: int
) -> pd.Series:
    """
    Compute forward excess returns for each ticker from as_of over horizon_days.

    r_{t, t+h} = ln(P_{t+h} / P_t)

    Parameters
    ----------
    price_data : Dict[str, pd.DataFrame]
        Price DataFrames keyed by ticker.
    tickers : List[str]
        Tickers to compute returns for.
    as_of : pd.Timestamp
        The current decision date.
    horizon_days : int
        Number of trading days forward.

    Returns
    -------
    pd.Series
        Forward log returns indexed by ticker.
    """
    fwd_returns = {}
    for ticker in tickers:
        if ticker not in price_data or price_data[ticker].empty:
            continue
        df = price_data[ticker]
        if as_of not in df.index:
            # Find nearest valid date
            valid = df.index[df.index <= as_of]
            if valid.empty:
                continue
            t0 = valid[-1]
        else:
            t0 = as_of

        future = df.index[df.index > t0]
        if len(future) < horizon_days:
            continue
        t1 = future[horizon_days - 1]

        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if close_col not in df.columns:
            continue

        p0 = df.loc[t0, close_col]
        p1 = df.loc[t1, close_col]
        if p0 > 0 and p1 > 0:
            fwd_returns[ticker] = float(np.log(p1 / p0))

    return pd.Series(fwd_returns, name='forward_return')


def run_single_normalization(
    config: Dict[str, Any],
    norm_name: str,
    norm_params: Dict[str, Any],
    data_loader: 'ResearchDataLoader',
    price_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    dates: List[pd.Timestamp],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run the full pipeline for one specific normalization method.

    Steps:
      1. For each date, compute raw characteristics for all tickers
      2. Apply the specified normalization cross-sectionally
      3. Compute forward returns
      4. Run Fama-MacBeth cross-sectional regression
      5. Compute incremental R² for each factor

    Parameters
    ----------
    config : Dict[str, Any]
        Experiment configuration dict.
    norm_name : str
        Name of the normalization method (registry key).
    norm_params : Dict[str, Any]
        Parameters for the normalization method.
    data_loader : ResearchDataLoader
        The initialised PIT-aware data loader.
    price_data : Dict[str, pd.DataFrame]
        Historical price data per ticker.
    tickers : List[str]
        List of target tickers.
    dates : List[pd.Timestamp]
        List of evaluation (rebalance) dates.
    output_dir : str
        Directory to save intermediate/final results.

    Returns
    -------
    Dict[str, Any]
        Summary of results including R², factor premia, incremental R², etc.
    """
    logger.info(f"Running pipeline with normalization: {norm_name}")

    pit_store = data_loader.get_pit_store()
    model_cfg = config.get('model', {})
    horizon_days = model_cfg.get('horizon_days', 21)
    characteristics = get_characteristics(config)
    factor_names = list(characteristics.keys())

    fm = FamaMacBeth(
        horizon_days=horizon_days,
        training_window=model_cfg.get('training_window', 252),
        rolling_step=model_cfg.get('rolling_step', 21),
    )

    # Collect per-date factor scores and forward returns
    factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    returns_by_date: Dict[pd.Timestamp, pd.Series] = {}

    for date in dates:
        # 1. Compute raw characteristics for each factor
        raw_dfs = {}
        for factor_name, factor_module in FACTOR_REGISTRY.items():
            if factor_name not in factor_names:
                continue
            try:
                raw_df = factor_module.compute_cross_section(
                    price_data, pit_store, tickers, date
                )
                raw_dfs[factor_name] = raw_df
            except Exception as e:
                logger.warning(f"Failed to compute {factor_name} at {date}: {e}")

        if not raw_dfs:
            continue

        # Build a single DataFrame of all raw characteristics
        all_chars_df = pd.DataFrame(index=tickers)
        for factor_name, raw_df in raw_dfs.items():
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                for col in raw_df.columns:
                    all_chars_df[col] = raw_df[col]
            elif isinstance(raw_df, pd.Series):
                all_chars_df[raw_df.name] = raw_df

        all_chars_df = all_chars_df.dropna(how='all')
        if all_chars_df.empty:
            continue

        # 2. Apply normalization cross-sectionally to ALL chars
        try:
            normed_chars = apply_normalization(all_chars_df, norm_name, **norm_params)
        except Exception as e:
            logger.warning(f"Normalization {norm_name} failed at {date}: {e}")
            continue

        # 3. Aggregate per factor by taking equal-weighted mean of normalized characteristics
        factor_scores = pd.DataFrame(index=normed_chars.index)
        for factor_name, chars in characteristics.items():
            available_chars = [c for c in chars if c in normed_chars.columns]
            if available_chars:
                factor_scores[factor_name] = normed_chars[available_chars].mean(axis=1)

        factor_scores_by_date[date] = factor_scores

        # Compute forward returns
        fwd_ret = compute_forward_returns(price_data, list(normed_chars.index), date, horizon_days)
        if not fwd_ret.empty:
            returns_by_date[date] = fwd_ret

    if not factor_scores_by_date:
        logger.warning(f"No valid dates for normalization {norm_name}")
        return {'normalization': norm_name, 'fama_macbeth': {}, 'incremental_r2': {}}

    # 4. Run Fama-MacBeth walk-forward
    valid_dates = sorted(set(factor_scores_by_date.keys()) & set(returns_by_date.keys()))
    fm_results = fm.walk_forward_fama_macbeth(valid_dates, factor_scores_by_date, returns_by_date)

    # 5. Compute OOS incremental R² using the new method
    try:
        oos_inc_df = fm.compute_all_oos_incremental(valid_dates, factor_scores_by_date, returns_by_date)
        incremental_r2 = oos_inc_df.to_dict(orient='index') if isinstance(oos_inc_df, pd.DataFrame) else oos_inc_df
    except Exception as e:
        logger.warning(f"Failed to compute OOS incremental R2: {e}")
        incremental_r2 = {}

    # Save intermediate CSVs
    norm_dir = os.path.join(output_dir, norm_name)
    os.makedirs(norm_dir, exist_ok=True)

    if isinstance(fm_results, pd.DataFrame) and not fm_results.empty:
        fm_results.to_csv(os.path.join(norm_dir, 'fama_macbeth.csv'), index=False)

    inc_df = pd.DataFrame(incremental_r2).T
    if not inc_df.empty:
        inc_df.to_csv(os.path.join(norm_dir, 'incremental_r2.csv'))

    results = {
        'normalization': norm_name,
        'fama_macbeth': fm_results.to_dict() if isinstance(fm_results, pd.DataFrame) else fm_results,
        'incremental_r2': incremental_r2,
    }

    logger.info(f"Completed pipeline for normalization: {norm_name}")
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save experiment results to CSV and a Markdown summary.

    Parameters
    ----------
    results : Dict[str, Any]
        Aggregated results across all normalizations.
    output_dir : str
        Directory to save results.
    """
    summary_path = os.path.join(output_dir, "summary.md")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Results Summary\n\n")
        f.write(f"Experiment ID: {results['experiment_id']}\n\n")

        for res in results["runs"]:
            f.write(f"## Normalization: {res['normalization']}\n\n")

            f.write("### OOS Incremental R²\n\n")
            f.write("| Factor | ΔR² | R²_full | R²_baseline |\n")
            f.write("|--------|-----|---------|-------------|\n")
            for factor, vals in res.get('incremental_r2', {}).items():
                if isinstance(vals, dict):
                    def _fmt(v):
                        try:
                            return f"{float(v):.6f}"
                        except (TypeError, ValueError):
                            return 'N/A'
                    f.write(f"| {factor} | {_fmt(vals.get('delta_r2'))} "
                            f"| {_fmt(vals.get('r2_oos_full'))} "
                            f"| {_fmt(vals.get('r2_oos_base'))} |\n")
            f.write("\n")

    logger.info(f"Results saved to {summary_path}")


def run_experiment(config: Dict[str, Any], selected_norms: Optional[List[str]] = None) -> None:
    """
    Run the full experiment across configured normalizations.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed configuration dict.
    selected_norms : Optional[List[str]], default None
        If provided, only run these specific normalizations.
    """
    exp_id = generate_experiment_id()
    output_dir = os.path.join("research", "results", exp_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting experiment {exp_id}. Output dir: {output_dir}")

    # Initialise data loader
    data_loader = ResearchDataLoader(config)

    sample_dates = config.get('sample_dates', {})
    start = sample_dates.get('start', '2005-01-01')
    end = sample_dates.get('end', '2023-12-31')

    # Determine universe tickers (placeholder — in production, loaded from config/data)
    universe_cfg = config.get('universe', {})
    # For now, use a placeholder list; in real usage, the data loader provides this
    tickers = data_loader.get_universe(pd.Timestamp(end))
    if not tickers:
        logger.warning("Universe is empty. Using placeholder tickers for demonstration.")
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

    # Load price data
    price_data = data_loader.load_price_data(tickers, start, end)

    # Load fundamental data into the PIT store (required by value/quality factors)
    data_loader.load_fundamental_data(tickers)

    # Generate rebalance dates
    freq = sample_dates.get('frequency', 'monthly')
    freq_map = {'monthly': 'BMS', 'weekly': 'W-MON', 'daily': 'B'}
    pd_freq = freq_map.get(freq, 'BMS')
    dates = list(pd.date_range(start, end, freq=pd_freq))

    # Determine which normalizations to run
    norm_specs = get_normalizations(config)
    if selected_norms:
        norm_specs = [n for n in norm_specs if n['name'] in selected_norms]

    all_results: Dict[str, Any] = {"experiment_id": exp_id, "runs": []}

    for norm_spec in norm_specs:
        norm_name = norm_spec['name']
        norm_params = norm_spec.get('params', {})
        try:
            res = run_single_normalization(
                config, norm_name, norm_params,
                data_loader, price_data, tickers, dates, output_dir
            )
            all_results["runs"].append(res)
        except Exception as e:
            logger.error(f"Failed pipeline for {norm_name}: {e}", exc_info=True)

    # Run robustness diagnostics comparing across normalizations
    logger.info("Running robustness diagnostics across normalizations...")
    try:
        diagnostics = RobustnessDiagnostics()

        # Collect incremental R2 DataFrames per normalization
        delta_r2_by_norm = {}
        for res in all_results["runs"]:
            norm = res['normalization']
            inc = res.get('incremental_r2', {})
            if inc:
                delta_r2_by_norm[norm] = pd.DataFrame(inc).T

        if len(delta_r2_by_norm) > 1:
            sign_stab = diagnostics.compute_sign_stability(delta_r2_by_norm)
            sign_stab.to_csv(os.path.join(output_dir, 'sign_stability.csv'))
            
            summary_stats = diagnostics.compute_summary_statistics(delta_r2_by_norm)
            summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
            logger.info("Robustness analysis complete (Sign Stability & Summary Statistics).")
    except Exception as e:
        logger.error(f"Robustness diagnostics failed: {e}", exc_info=True)

    save_results(all_results, output_dir)
    logger.info(f"Experiment {exp_id} completed successfully.")


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    # Load protocol config
    config = load_protocol(args.config)

    run_experiment(config, selected_norms=args.normalization)


if __name__ == "__main__":
    main()
