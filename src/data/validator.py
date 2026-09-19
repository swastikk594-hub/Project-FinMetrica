import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from statsmodels.tsa.stattools import adfuller
import logging
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """
    Report containing the results of data validation checks and an institutional DataQualityScore.
    """
    passed: bool
    data_quality_score: float = 100.0  # Scale 0 to 100
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    per_ticker: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DataValidator:
    """
    Validates financial data quality before analysis.

    Produces a DataQualityReport with an institutional DataQualityScore (0-100)
    that must pass minimum standards before quantitative modules run.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or {}
        self.max_missing_pct = self.config.get('data', {}).get('max_missing_pct', 0.05)
        self.outlier_z_threshold = self.config.get('data', {}).get('outlier_z_threshold', 5.0)

    def check_missing_data(self, df: pd.DataFrame, threshold: Optional[float] = None) -> dict:
        """Checks for missing data in the DataFrame."""
        thresh = threshold if threshold is not None else self.max_missing_pct
        missing_count = df.isna().sum().to_dict()
        total_rows = len(df)
        flags = {col: (count / total_rows) > thresh for col, count in missing_count.items() if total_rows > 0}
        
        return {
            'missing_count': missing_count,
            'missing_pct': {col: (count / total_rows) if total_rows > 0 else 0 for col, count in missing_count.items()},
            'flags': flags
        }

    def check_duplicate_dates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks for duplicate index entries (dates)."""
        duplicates = df.index[df.index.duplicated()].tolist()
        return {
            'has_duplicates': len(duplicates) > 0,
            'duplicate_dates': duplicates
        }

    def check_impossible_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks for impossible values like negative prices or zero volume on trading days."""
        impossible = {}
        
        for col in ['Close', 'Adj Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                negative_prices = (df[col] < 0).sum()
                if negative_prices > 0:
                    impossible[f'negative_{col.lower()}'] = int(negative_prices)
                
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            if zero_volume > 0:
                impossible['zero_volume'] = int(zero_volume)
                
        return {
            'has_impossible': len(impossible) > 0,
            'details': impossible
        }

    def check_outliers(self, returns_df: pd.DataFrame, z_thresh: Optional[float] = None) -> Dict[str, Any]:
        """Checks for outliers based on z-score."""
        z_lim = z_thresh if z_thresh is not None else self.outlier_z_threshold
        outliers = {}
        for col in returns_df.columns:
            series = returns_df[col].dropna()
            if len(series) < 2:
                continue
            std = series.std()
            if std == 0:
                continue
            z_scores = (series - series.mean()) / std
            outlier_count = (np.abs(z_scores) > z_lim).sum()
            if outlier_count > 0:
                outliers[col] = int(outlier_count)
                
        return {
            'has_outliers': len(outliers) > 0,
            'outliers': outliers
        }

    def check_timestamp_gaps(self, df: pd.DataFrame, max_gap_days: int = 5) -> Dict[str, Any]:
        """Detects suspiciously large gaps in trading calendar."""
        if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
            return {'has_gaps': False, 'gaps': {}}
            
        diffs = df.index.to_series().diff().dt.days
        gaps = diffs[diffs > max_gap_days]
        
        return {
            'has_gaps': len(gaps) > 0,
            'gaps': gaps.to_dict()
        }

    def check_return_stationarity(self, returns_df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
        """Performs ADF test for stationarity on returns."""
        results = {}
        for col in returns_df.columns:
            series = returns_df[col].dropna()
            if len(series) < 30:
                results[col] = {'stationary': False, 'p_value': None, 'reason': 'insufficient_data'}
                continue
                
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    adf_stat, p_val, _, _, _, _ = adfuller(series, autolag='AIC')
                results[col] = {
                    'stationary': bool(p_val < alpha),
                    'p_value': float(p_val)
                }
            except Exception as e:
                logger.warning(f"ADF test failed for {col}: {e}")
                results[col] = {'stationary': False, 'p_value': None, 'error': str(e)}
                
        return results

    def generate_quality_report(self, price_data: Dict[str, pd.DataFrame], fundamental_data: Dict[str, Dict[str, Any]]) -> DataQualityReport:
        """
        Runs all validation checks and computes a composite DataQualityScore (0-100).
        """
        report = DataQualityReport(passed=True)
        penalty = 0.0
        
        for ticker, df in price_data.items():
            if df.empty:
                report.errors.append(f"{ticker}: Empty price DataFrame")
                report.passed = False
                penalty += 25.0
                continue
                
            ticker_results = {}
            
            # Missing Data
            miss_res = self.check_missing_data(df)
            ticker_results['missing_data'] = miss_res
            if any(miss_res['flags'].values()):
                report.warnings.append(f"{ticker}: High missing data in columns {list(miss_res['flags'].keys())}")
                penalty += 5.0
                
            # Duplicates
            dup_res = self.check_duplicate_dates(df)
            ticker_results['duplicates'] = dup_res
            if dup_res['has_duplicates']:
                report.errors.append(f"{ticker}: Duplicate dates found: {dup_res['duplicate_dates']}")
                report.passed = False
                penalty += 20.0
                
            # Impossible Values
            imp_res = self.check_impossible_values(df)
            ticker_results['impossible_values'] = imp_res
            if imp_res['has_impossible']:
                report.errors.append(f"{ticker}: Impossible values detected {imp_res['details']}")
                report.passed = False
                penalty += 20.0
                
            # Gaps
            gap_res = self.check_timestamp_gaps(df)
            ticker_results['timestamp_gaps'] = gap_res
            if gap_res['has_gaps']:
                report.warnings.append(f"{ticker}: Large timestamp gaps detected")
                penalty += 2.0
                
            # Outliers & Stationarity
            col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else df.columns[0])
            returns = df[[col]].pct_change().dropna()
            returns.columns = [ticker]
            
            outlier_res = self.check_outliers(returns)
            ticker_results['outliers'] = outlier_res
            if outlier_res['has_outliers']:
                report.warnings.append(f"{ticker}: Extreme return outliers detected ({outlier_res['outliers']})")
                penalty += 1.0
                
            stat_res = self.check_return_stationarity(returns)
            ticker_results['stationarity'] = stat_res
            if not stat_res.get(ticker, {}).get('stationary', True):
                report.warnings.append(f"{ticker}: Returns may not be stationary")
                penalty += 3.0
                    
            report.per_ticker[ticker] = ticker_results
            
        for ticker, fund_dict in fundamental_data.items():
            if not fund_dict:
                report.warnings.append(f"{ticker}: Fundamental data is missing or empty")
                penalty += 5.0

        score = max(0.0, min(100.0, 100.0 - penalty))
        report.data_quality_score = score
        if score < 50.0:
            report.passed = False
                
        report.summary['total_tickers_checked'] = len(price_data)
        report.summary['data_quality_score'] = score
        report.summary['total_errors'] = len(report.errors)
        report.summary['total_warnings'] = len(report.warnings)
        
        return report

    def print_quality_report(self, report: DataQualityReport) -> None:
        """Prints a formatted report using the rich library or fallback console."""
        if Console is None:
            print(f"--- Data Quality Report [Score: {report.data_quality_score:.1f}/100] ---")
            print(f"Passed: {report.passed}")
            print(f"Errors ({len(report.errors)}):")
            for e in report.errors: print(f"  - {e}")
            print(f"Warnings ({len(report.warnings)}):")
            for w in report.warnings: print(f"  - {w}")
            return
            
        console = Console()
        status_color = 'green' if report.passed else 'red'
        console.print(f"\n[bold]Data Quality Report[/bold] [Score: [bold cyan]{report.data_quality_score:.1f}/100[/]]: [{status_color}]{'PASSED' if report.passed else 'FAILED'}[/]")
        
        if report.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in report.errors:
                console.print(f"  [red]x[/red] {error}")
                
        if report.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in report.warnings:
                console.print(f"  [yellow]![/yellow] {warning}")
                
        table = Table(title="Summary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in report.summary.items():
            val_str = f"{v:.1f}" if isinstance(v, float) else str(v)
            table.add_row(str(k).replace('_', ' ').title(), val_str)
            
        console.print(table)
