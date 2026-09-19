"""
QuantFinance Institutional Pipeline Orchestrator
================================================
Orchestrates end-to-end institutional quantitative investment research,
multi-module alpha generation, machine learning factor combination,
HMM macro regime detection, Student-t Copula tail risk, market impact execution,
walk-forward event backtesting, and formal strategy qualification.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from src.data.config_loader import get_config
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.point_in_time import PointInTimeStore
from src.data.universe import UniverseManager
from src.preprocessing.returns import log_returns
from src.modules.m1_fundamentals import FundamentalsModule
from src.modules.m2_valuation import ValuationModule
from src.modules.m3_timeseries import TimeSeriesModule
from src.modules.m4_factors import FactorModel
from src.modules.m5_risk import RiskModule
from src.modules.m6_macro import MacroModule
from src.modules.m7_optimisation import InstitutionalPortfolioOptimiser
from src.modules.m8_nlp import FinancialNLPEngine
from src.modules.m9_stat_arb import StatisticalArbitrageEngine
from src.models.meta_model import MLMetaModel
from src.models.hmm_regime import MacroHMMRegimeEngine
from src.risk.copula import CopulaTailRiskSimulator
from src.risk.evt import EVTTailEstimator
from src.risk.stress import StressTestingEngine
from src.backtesting.engine import EventDrivenBacktester
from src.reporting.qualification import StrategyQualificationEngine
from src.reporting.capacity import CapacityAnalysisEngine
from src.safety.pre_trade import PreTradeRiskEngine
from src.safety.kill_switch import KillSwitchManager
from src.execution.order_generator import ExecutionEngine
from src.advisory.horizon_engine import AdvisoryEngine
from src.safety.audit_logger import AuditLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantFinPipeline")


class QuantFinancePipeline:
    """
    Comprehensive Institutional Quantitative Research and Portfolio Construction Engine.
    """

    def __init__(
        self,
        tickers: List[str],
        start: str,
        end: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        self.tickers = [t.upper().strip() for t in tickers]
        self.start = start
        self.end = end or datetime.today().strftime('%Y-%m-%d')
        self.config = get_config(config_path)
        self.console = Console() if HAS_RICH else None

        # Data & Point-in-time Infrastructure
        self.fetcher = DataFetcher(self.config)
        self.validator = DataValidator(self.config)
        self.pit_store = PointInTimeStore()
        self.universe_mgr = UniverseManager(self.config)

        # Alpha Modules
        self.m1 = FundamentalsModule(self.config)
        self.m2 = ValuationModule(self.config)
        self.m3 = TimeSeriesModule(self.config)
        self.m4 = FactorModel(self.config)
        self.m5 = RiskModule(self.config)
        self.m6 = MacroModule(self.config)
        self.m7 = InstitutionalPortfolioOptimiser(self.config)
        self.m8 = FinancialNLPEngine(self.config)
        self.m9 = StatisticalArbitrageEngine(self.config)

        # Advanced Modeling & Risk Engines
        seed = self.config.get('random_seed', 42)
        self.meta_model = MLMetaModel(model_type="gbdt", random_state=seed)
        self.hmm_engine = MacroHMMRegimeEngine(n_states=3, random_state=seed)
        self.copula_engine = CopulaTailRiskSimulator(degrees_of_freedom=4.0, random_state=seed)
        self.evt_engine = EVTTailEstimator()
        self.stress_engine = StressTestingEngine(self.config)

        # Backtesting & Qualification
        self.backtester = EventDrivenBacktester(self.config)
        self.qualifier = StrategyQualificationEngine()
        self.capacity_engine = CapacityAnalysisEngine()

        # Operational Safety
        self.pre_trade = PreTradeRiskEngine()
        self.kill_switch = KillSwitchManager()
        self.audit_logger = AuditLogger()

        # Execution & Advisory
        self.execution_engine = ExecutionEngine(self.config)
        self.advisory_engine = AdvisoryEngine(self.config)

        # Data stores
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.fundamental_data: Dict[str, dict] = {}
        self.ff_factors: Optional[pd.DataFrame] = None
        self.macro_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None

    def _load_data(self) -> bool:
        """Loads and verifies all financial datasets with point-in-time registration."""
        logger.info(f"Loading market data for: {self.tickers}")
        self.price_data = self.fetcher.fetch_price_data(self.tickers, start=self.start, end=self.end)
        if not self.price_data:
            logger.error("No price data retrieved.")
            return False

        self.fundamental_data = self.fetcher.fetch_fundamental_data(self.tickers)
        self.ff_factors = self.fetcher.fetch_ff_factors(factor_set='4')

        macro_series = self.config.get('macro', {}).get('fred_series', {
            'FEDFUNDS': 'Fed Funds Rate',
            'T10Y2Y': 'Term Spread',
            'BAMLH0A0HYM2': 'HY Spread'
        })
        self.macro_data = self.fetcher.fetch_macro_data(macro_series)

        self.benchmark_data = self.fetcher.fetch_benchmark(
            ticker=self.config.get('benchmark_ticker', '^GSPC'),
            start=self.start,
            end=self.end
        )

        # Register data with Point-in-Time Store
        for t, df in self.price_data.items():
            self.pit_store.register_price_data(t, df)

        if self.macro_data is not None and not self.macro_data.empty:
            self.pit_store.register_macro_data(self.macro_data)

        # Validate
        report = self.validator.generate_quality_report(self.price_data, self.fundamental_data)
        self.validator.print_quality_report(report)
        return True

    @staticmethod
    def _reconstruct_fin_data(raw: dict) -> dict:
        """Helper to reconstruct DataFrame from raw JSON fundamental store."""
        reconstructed = {}
        for key in ('income_stmt', 'balance_sheet', 'cash_flow'):
            obj = raw.get(key)
            if not obj:
                reconstructed[key] = pd.DataFrame()
                continue
            try:
                df = pd.DataFrame(obj)
                df.columns = pd.to_datetime(df.columns, errors='coerce')
                df = df.sort_index(axis=1, ascending=False)
                reconstructed[key] = df
            except Exception:
                reconstructed[key] = pd.DataFrame()
        reconstructed['info'] = raw.get('info', {})
        return reconstructed

    def score_assets(
        self,
        price_data_window: Dict[str, pd.DataFrame],
        modules: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Scoring function for walk-forward event simulation."""
        scores = {}
        for t, df in price_data_window.items():
            if df.empty:
                continue
            fin_raw = self.fundamental_data.get(t, {})
            fin = self._reconstruct_fin_data(fin_raw)

            m1_s = self.m1.run(t, df, fin).normalised_score if (modules is None or 1 in modules) else 50.0
            m2_s = self.m2.run(t, df, fin).normalised_score if (modules is None or 2 in modules) else 50.0
            m3_s = self.m3.run(t, df).normalised_score if (modules is None or 3 in modules) else 50.0
            m4_s = self.m4.run(t, df, self.ff_factors).normalised_score if (modules is None or 4 in modules) else 50.0
            bench_df = self.benchmark_data if (self.benchmark_data is not None and not self.benchmark_data.empty) else pd.DataFrame()
            m5_res = self.m5.run(t, df, bench_df) if (modules is None or 5 in modules) else None
            m5_s = m5_res.normalised_score if m5_res else 50.0

            # Composite average with risk adjustment
            raw_comp = np.mean([m1_s, m2_s, m3_s, m4_s])
            risk_penalty = (100.0 - m5_s) / 100.0
            adjusted = raw_comp * (1.0 - 0.25 * risk_penalty)
            scores[t] = float(np.clip(adjusted, 0.0, 100.0))

        return scores

    def run(
        self,
        modules: Optional[List[int]] = None,
        run_backtest: bool = True,
        run_stress: bool = True,
        mode: str = 'research',
        capital: float = 100000.0,
        current_holdings: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """Executes full institutional pipeline."""
        if current_holdings is None:
            current_holdings = {}
        if not self._load_data():
            return {}

        results: Dict[str, Any] = {}

        bench_df = self.benchmark_data if (self.benchmark_data is not None and not self.benchmark_data.empty) else pd.DataFrame()

        # 1. Run Modules M1 to M6
        logger.info("Executing Fundamental, Valuation, Momentum, Factor, Risk, and Macro Modules...")
        results['m1'] = {t: self.m1.run(t, self.price_data[t], self._reconstruct_fin_data(self.fundamental_data.get(t, {}))) for t in self.tickers if t in self.price_data}
        results['m2'] = {t: self.m2.run(t, self.price_data[t], self._reconstruct_fin_data(self.fundamental_data.get(t, {}))) for t in self.tickers if t in self.price_data}
        results['m3'] = {t: self.m3.run(t, self.price_data[t]) for t in self.tickers if t in self.price_data}
        results['m4'] = {t: self.m4.run(t, self.price_data[t], self.ff_factors) for t in self.tickers if t in self.price_data}
        results['m5'] = {t: self.m5.run(t, self.price_data[t], bench_df) for t in self.tickers if t in self.price_data}
        results['m6'] = {t: self.m6.run(t, self.price_data[t], self.macro_data) for t in self.tickers if t in self.price_data}
        results['m8'] = {t: self.m8.run(t, self.price_data[t]) for t in self.tickers if t in self.price_data}
        results['m9'] = self.m9.scan_universe_pairs(self.price_data)

        # 2. Macro HMM Regime Engine
        logger.info("Running Macro Gaussian HMM Regime Engine...")
        if not bench_df.empty:
            bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else ('Close' if 'Close' in bench_df.columns else bench_df.columns[0])
            bm_rets = bench_df[bench_col].pct_change().dropna()
        else:
            # Fallback benchmark: Equal-weighted average of price data
            bm_rets = pd.DataFrame({
                t: self.price_data[t]['Close'].pct_change().dropna()
                for t in self.tickers if t in self.price_data
            }).mean(axis=1).dropna()

        hmm_feats = self.hmm_engine.prepare_features(bm_rets, self.macro_data if self.macro_data is not None else pd.DataFrame())
        self.hmm_engine.fit(hmm_feats)
        regime_probs = self.hmm_engine.predict_regime_probabilities(hmm_feats)
        results['regime_probs'] = regime_probs

        # 3. Machine Learning Meta-Model Factor Aggregation
        logger.info("Running ML Meta-Model Factor Aggregator...")
        feature_rows = []
        for t in self.tickers:
            if t in self.price_data:
                feature_rows.append({
                    'ticker': t,
                    'm1_score': results['m1'][t].normalised_score,
                    'm2_score': results['m2'][t].normalised_score,
                    'm3_score': results['m3'][t].normalised_score,
                    'm4_score': results['m4'][t].normalised_score,
                    'm5_score': results['m5'][t].normalised_score,
                    'm6_score': results['m6'][t].normalised_score,
                    'm8_score': results['m8'][t].normalised_score
                })
        feat_df = pd.DataFrame(feature_rows).set_index('ticker')
        ml_preds = self.meta_model.predict(feat_df)
        results['composite'] = ml_preds

        # 4. Institutional Portfolio Optimization (M7)
        logger.info("Running Institutional Portfolio Optimiser (Black-Litterman & HRP)...")
        opt_res = self.m7.run(self.tickers, self.price_data, ml_preds['composite_score'].to_dict())
        results['m7'] = opt_res

        # 5. Non-Linear Tail Risk (Student-t Copula & EVT)
        logger.info("Evaluating Tail Risk with Student-t Copula & EVT...")
        returns_df = pd.DataFrame({
            t: self.price_data[t]['Close'].pct_change().dropna()
            for t in self.tickers if t in self.price_data and not self.price_data[t].empty
        }).dropna()

        self.copula_engine.fit(returns_df)
        max_s_weights = pd.Series(opt_res['max_sharpe']['weights'])
        copula_risk = self.copula_engine.compute_portfolio_tail_risk(max_s_weights)
        results['copula_risk'] = copula_risk

        # 6. Historical Crisis Stress Testing
        if run_stress:
            logger.info("Executing Historical Scenario & Reverse Stress Testing...")
            betas = {t: results['m5'][t].metadata.get('metrics', {}).get('beta', 1.0) for t in self.tickers if t in results['m5']}
            stress_res = self.stress_engine.run_predefined_scenarios(max_s_weights, betas)
            rev_stress = self.stress_engine.reverse_stress_test(max_s_weights, betas)
            results['stress_testing'] = stress_res
            results['reverse_stress'] = rev_stress

        # 7. Walk-Forward Backtesting
        if run_backtest:
            logger.info("Running Walk-Forward Event Backtesting Engine...")
            bench = self.benchmark_data if self.benchmark_data is not None else pd.DataFrame()
            bt_results = self.backtester.run(
                price_data=self.price_data,
                score_function=lambda pw: self.score_assets(pw, modules=modules),
                benchmark_data=bench,
                strategy_name="QuantFinance Multi-Factor Strategy"
            )
            results['backtest'] = {
                'folds': bt_results,
                'aggregated': self.backtester.aggregate_fold_results(bt_results)
            }

            # 8. Strategy Qualification & Overfitting Analysis
            if bt_results:
                full_port_rets = pd.concat([fold.returns for fold in bt_results]).sort_index()
                full_bm_rets = pd.concat([fold.benchmark_returns for fold in bt_results]).sort_index()
                total_impact = sum(fold.total_market_impact_cost for fold in bt_results)
                agg_metrics = results['backtest']['aggregated']
                mdd = agg_metrics.get('max_drawdown', {}).get('mean', -0.20)

                qual_verdict = self.qualifier.evaluate_strategy(
                    returns=full_port_rets,
                    benchmark_returns=full_bm_rets,
                    total_market_impact_dollars=total_impact,
                    max_drawdown=mdd
                )
                results['qualification'] = qual_verdict

                # 9. AUM Capacity Curve
                cagr = agg_metrics.get('annualised_return', {}).get('mean', 0.15)
                vol = agg_metrics.get('volatility', {}).get('mean', 0.18)
                turnover = np.mean([f.avg_turnover for f in bt_results]) if bt_results else 1.0
                capacity_df = self.capacity_engine.generate_capacity_curve(cagr, vol, turnover)
                results['capacity_curve'] = capacity_df

        # Log decisions to audit trail
        dom_regime = regime_probs['dominant_regime'].iloc[-1] if not regime_probs.empty else "Normal"
        for t, w in max_s_weights.items():
            self.audit_logger.log_decision(
                timestamp=pd.Timestamp.now(),
                ticker=t,
                target_weight=w,
                previous_weight=0.0,
                expected_return=opt_res.get('expected_returns_bl', {}).get(t, 0.08),
                composite_score=float(ml_preds.loc[t, 'composite_score']) if t in ml_preds.index else 50.0,
                dominant_regime=dom_regime,
                rationale="Black-Litterman Max Sharpe Allocation"
            )

        # Execution & Advisory Routing
        latest_prices = {t: self.price_data[t]['Close'].iloc[-1] for t in self.tickers if t in self.price_data and not self.price_data[t].empty}
        
        if mode == 'execute':
            logger.info("Operating in EXECUTE Mode: Generating real-world order tickets...")
            # Approximate ADV from last 20 days
            adv_data = {}
            for t in self.tickers:
                if t in self.price_data and 'Volume' in self.price_data[t].columns:
                    adv_data[t] = self.price_data[t]['Volume'].iloc[-20:].mean()
            
            exec_res = self.execution_engine.generate_order_tickets(
                target_weights=max_s_weights.to_dict(),
                latest_prices=latest_prices,
                adv_data=adv_data,
                capital=capital,
                current_holdings=current_holdings
            )
            results['execution'] = exec_res
            
        elif mode == 'advise':
            logger.info("Operating in ADVISE Mode: Generating investment horizon advisory report...")
            multi_scores = {}
            risk_metrics = {}
            for t in self.tickers:
                multi_scores[t] = {
                    'm1_fundamentals': results['m1'][t].normalised_score if t in results.get('m1', {}) else 50.0,
                    'm2_valuation': results['m2'][t].normalised_score if t in results.get('m2', {}) else 50.0,
                    'm3_timeseries': results['m3'][t].normalised_score if t in results.get('m3', {}) else 50.0,
                    'm4_factors': results['m4'][t].normalised_score if t in results.get('m4', {}) else 50.0,
                    'm9_stat_arb': 50.0 # Simplify for now
                }
                rm = results.get('m5', {}).get(t)
                risk_metrics[t] = {
                    'cvar_95': rm.metadata.get('metrics', {}).get('cvar_95', 0.05) if rm else 0.05
                }
                
            adv_res = self.advisory_engine.generate_advisory_report(
                tickers=self.tickers,
                weights=max_s_weights.to_dict(),
                capital=capital,
                latest_prices=latest_prices,
                multi_factor_scores=multi_scores,
                risk_metrics=risk_metrics
            )
            results['advisory'] = adv_res

        return results

    def print_report(self, results: Dict[str, Any]) -> None:
        """Prints comprehensive institutional research & qualification report."""
        if not self.console:
            print("--- QuantFinance Institutional Report ---")
            print(f"Status: {results.get('qualification', {}).get('qualification_status', 'N/A')}")
            return

        self.console.rule("[bold cyan]QuantFinance Institutional Research & Portfolio Report[/bold cyan]")
        self.console.print()

        # Qualification Card
        qual = results.get('qualification')
        if qual:
            status = qual['qualification_status']
            color = 'green' if status == 'PASS' else ('yellow' if status == 'CONDITIONAL PASS' else 'red')
            panel_text = (
                f"[bold]Strategy Qualification Verdict:[/] [{color}]{status}[/]\n"
                f"[bold]Recommended Stage:[/] {qual['deployment_stage']}\n"
                f"[bold]Deflated Sharpe Ratio (DSR):[/] [cyan]{qual['metrics']['deflated_sharpe_ratio']:.1%}[/] "
                f"([yellow]Haircut Sharpe:[/] {qual['metrics']['haircut_sharpe']:.2f})\n"
                f"[bold]Information Ratio:[/] {qual['metrics']['information_ratio']:.2f} | "
                f"[bold]Max Drawdown:[/] {qual['metrics']['max_drawdown']:.1%}\n"
                f"[bold]Total Market Impact Cost:[/] ${qual['metrics']['total_impact_cost']:,.2f}"
            )
            self.console.print(Panel(panel_text, title="[bold]Institutional Qualification Scorecard[/bold]", border_style=color))
            self.console.print()

        # Multi-Module Score Table
        table = Table(title="Multi-Factor & Alpha Scores (0-100)", box=box.ROUNDED)
        table.add_column("Ticker", style="bold cyan")
        table.add_column("M1 Qual", justify="center")
        table.add_column("M2 Val", justify="center")
        table.add_column("M3 Mom", justify="center")
        table.add_column("M4 Fact", justify="center")
        table.add_column("M5 Risk", justify="center")
        table.add_column("M8 NLP", justify="center")
        table.add_column("ML Score", justify="center", style="bold green")

        for t in self.tickers:
            if t in results.get('m1', {}):
                m1_s = f"{results['m1'][t].normalised_score:.1f}"
                m2_s = f"{results['m2'][t].normalised_score:.1f}"
                m3_s = f"{results['m3'][t].normalised_score:.1f}"
                m4_s = f"{results['m4'][t].normalised_score:.1f}"
                m5_s = f"{results['m5'][t].normalised_score:.1f}"
                m8_s = f"{results['m8'][t].normalised_score:.1f}"
                ml_s = f"{results['composite'].loc[t, 'composite_score']:.1f}" if t in results['composite'].index else "50.0"
                table.add_row(t, m1_s, m2_s, m3_s, m4_s, m5_s, m8_s, ml_s)

        self.console.print(table)
        self.console.print()

        # Allocation Table
        m7 = results.get('m7', {})
        max_s = m7.get('max_sharpe', {})
        hrp = m7.get('hierarchical_risk_parity_weights', {})
        if max_s.get('weights'):
            w_table = Table(title="Portfolio Allocations", box=box.SIMPLE_HEAVY)
            w_table.add_column("Ticker", style="cyan")
            w_table.add_column("Black-Litterman / Max Sharpe", justify="right", style="yellow")
            w_table.add_column("Hierarchical Risk Parity (HRP)", justify="right", style="blue")
            for t in max_s['weights']:
                w_table.add_row(t, f"{max_s['weights'][t]:.1%}", f"{hrp.get(t, 0.0):.1%}")
            self.console.print(w_table)
            self.console.print(
                f"  Expected Return: [green]{max_s.get('expected_return', 0.0):.2%}[/]  "
                f"Volatility: [yellow]{max_s.get('expected_volatility', 0.0):.2%}[/]  "
                f"Sharpe: [cyan]{max_s.get('sharpe_ratio', 0.0):.2f}[/]\n"
            )

        # Copula Tail Risk
        cop = results.get('copula_risk')
        if cop:
            t_risk = cop['student_t_copula']
            g_risk = cop['gaussian_copula_baseline']
            self.console.print(
                f"[bold]Copula Tail-Risk Analysis (99% CVaR):[/bold]\n"
                f"  Student-t Copula CVaR: [red]{t_risk['cvar_99']:.2%}[/] (Joint Crash Prob: {t_risk['joint_crash_probability']:.1%})\n"
                f"  Gaussian Baseline CVaR: [yellow]{g_risk['cvar_99']:.2%}[/] (Joint Crash Prob: {g_risk['joint_crash_probability']:.1%})\n"
                f"  Tail Risk Underestimation Delta: [bold magenta]{cop['tail_risk_delta_cvar']:.2%}[/]\n"
            )

        if 'execution' in results:
            exec_res = results['execution']
            self.console.print(f"[bold cyan]EXECUTION TICKERS (AUM: ${exec_res['total_aum']:,.2f})[/bold cyan]")
            self.console.print(f"Residual Cash Buffer: [green]${exec_res['residual_cash']:,.2f}[/green]\n")
            
            t_table = Table(title="Generated Rebalance Order Tickets", box=box.ROUNDED)
            t_table.add_column("Action", style="bold")
            t_table.add_column("Ticker", style="cyan")
            t_table.add_column("Delta Shares", justify="right")
            t_table.add_column("Target Shares", justify="right")
            t_table.add_column("Trade Value ($)", justify="right")
            t_table.add_column("ADV %", justify="right")
            t_table.add_column("Warning", style="red")

            for tk in exec_res['tickets']:
                act_color = "green" if tk['action'] == "BUY" else "red"
                t_table.add_row(
                    f"[{act_color}]{tk['action']}[/]",
                    tk['ticker'],
                    str(tk['shares']),
                    str(tk['target_shares']),
                    f"${tk['est_trade_value']:,.2f}",
                    f"{tk['adv_participation_pct']:.1f}%",
                    tk['warning'] or ""
                )
            self.console.print(t_table)
            self.console.print()

        if 'advisory' in results:
            adv_res = results['advisory']
            self.console.print(f"[bold cyan]QUANTITATIVE ASSET ADVISOR & ALLOCATOR (AUM: ${adv_res['total_capital']:,.2f})[/bold cyan]\n")
            
            for t, data in adv_res['assets'].items():
                self.console.print(f"[bold yellow][{t}][/bold yellow] | Recommended Allocation: [green]{data['allocation_pct']:.1f}%[/green] (${data['allocation_usd']:,.2f} -> ~{data['target_shares']} shares)")
                self.console.print(f"  - Dominant Factor: [bold]{data['dominant_factor']}[/bold]")
                self.console.print(f"  - Recommended Investment Horizon: [cyan]{data['recommended_horizon']}[/cyan]")
                self.console.print(f"  - Risk Budget: 95% CVaR is {data['cvar_95']:.1f}%")
                self.console.print(f"  - [red]{data['invalidation_criteria']}[/red]\n")

        # Predefined Stress Testing
        stress = results.get('stress_testing')
        if stress:
            s_table = Table(title="Historical Crisis Stress Tests", box=box.ROUNDED)
            s_table.add_column("Crisis Scenario", style="cyan")
            s_table.add_column("Projected Loss", justify="right", style="red")
            s_table.add_column("Survival Status", justify="center")
            for sc_id, sc in stress.items():
                status_str = "[green]SURVIVES[/]" if sc['portfolio_survival'] else "[bold red]FAIL[/]"
                s_table.add_row(sc['scenario_name'], f"{sc['projected_portfolio_loss']:.1%}", status_str)
            self.console.print(s_table)
            self.console.print()

        # Disclaimer
        self.console.print(Panel(
            "[bold red]INSTITUTIONAL RESEARCH DISCLAIMER[/bold red]\n"
            "This platform is strictly for quantitative research, backtesting, and portfolio engineering.\n"
            "It does not constitute direct investment advice. Every model is a simplified approximation of reality.\n"
            "Past statistical performance and backtests do not guarantee future live returns.",
            border_style="red"
        ))

    def save_results(self, results: Dict[str, Any], output_dir: str = 'results/') -> None:
        """Serializes institutional results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        def _sanitize(item):
            if isinstance(item, dict):
                return {str(k): _sanitize(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [_sanitize(v) for v in item]
            elif isinstance(item, pd.Series):
                return {str(k): _sanitize(v) for k, v in item.items()}
            elif isinstance(item, pd.DataFrame):
                return {str(k): _sanitize(v) for k, v in item.to_dict().items()}
            elif isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, (np.floating, float)):
                return None if np.isnan(item) else float(item)
            elif isinstance(item, (np.integer, int)):
                return int(item)
            elif isinstance(item, (pd.Timestamp, datetime)):
                return str(item)
            elif hasattr(item, '__dict__'):
                return _sanitize(item.__dict__)
            return str(item)

        safe = {k: _sanitize(v) for k, v in results.items() if k != 'regime_probs'}
        out_path = os.path.join(output_dir, 'pipeline_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(safe, f, indent=2)
        logger.info(f"Institutional results saved -> {out_path}")
