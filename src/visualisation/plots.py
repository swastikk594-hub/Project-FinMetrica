import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from typing import Dict, List, Optional, Union

class Plots:
    """
    QuantFinance-EDU Visualisation Suite
    ====================================
    A collection of mathematically and financially meaningful plots.
    Each plot is designed to answer a specific analytical question.
    """

    def __init__(self):
        # Set consistent style
        sns.set_style("whitegrid")
        self.palette = sns.color_palette("viridis", as_cmap=False)
        self.primary_color = self.palette[0]
        self.secondary_color = self.palette[len(self.palette) // 2]
        self.accent_color = sns.color_palette("Reds")[4]
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.figsize'] = (10, 6)

    def _save_or_show(self, save_path: Optional[str] = None) -> None:
        """Helper to save or show plot and close."""
        plt.tight_layout()
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_return_distribution(self, returns: pd.Series, ticker: str, save_path: str = None) -> None:
        """
        Histogram of daily log returns with fitted normal distribution and VaR markers.
        
        Mathematical concept:
            Examines the normality assumption of returns and tail thickness.
            Values for skewness and kurtosis help identify non-normal behavior.
        """
        plt.figure()
        
        # Calculate stats
        mu, std = stats.norm.fit(returns.dropna())
        skew = stats.skew(returns.dropna())
        kurt = stats.kurtosis(returns.dropna())
        var_95 = np.percentile(returns.dropna(), 5)
        var_99 = np.percentile(returns.dropna(), 1)
        
        # Plot histogram
        sns.histplot(returns, stat="density", color=self.primary_color, alpha=0.6, bins=50)
        
        # Overlay normal distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\mu$={mu:.4f}, $\sigma$={std:.4f})')
        
        # Mark VaR
        plt.axvline(var_95, color=self.accent_color, linestyle='--', label=f'VaR 95%: {var_95:.4f}')
        plt.axvline(var_99, color='darkred', linestyle='-.', label=f'VaR 99%: {var_99:.4f}')
        
        plt.title(f"How fat are the tails of {ticker}'s return distribution?")
        plt.xlabel("Daily Returns")
        plt.ylabel("Density")
        
        # Annotate
        plt.annotate(f'Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}', 
                     xy=(0.05, 0.85), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.legend(loc='upper right')
        self._save_or_show(save_path)

    def plot_rolling_volatility(self, returns: pd.Series, ticker: str, windows: list = [21, 63, 252], save_path: str = None) -> None:
        """
        Rolling annualised volatility at multiple windows.
        
        Mathematical concept:
            Volatility clustering and time-varying risk.
            Annualisation factor is sqrt(252).
        """
        plt.figure()
        
        for i, window in enumerate(windows):
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            plt.plot(rolling_vol, label=f'{window}-Day Window', color=self.palette[i % len(self.palette)])
            
            # Shade high volatility periods for the longest window
            if window == max(windows):
                threshold = rolling_vol.quantile(0.75)
                plt.fill_between(rolling_vol.index, 0, rolling_vol, 
                                 where=(rolling_vol > threshold), 
                                 color=self.accent_color, alpha=0.2, label=f'> 75th pctl ({window}d)')
                plt.axhline(threshold, color='gray', linestyle=':', alpha=0.5)
        
        plt.title(f"How has {ticker}'s risk changed over time?")
        plt.xlabel("Date")
        plt.ylabel("Annualised Volatility")
        plt.legend()
        self._save_or_show(save_path)

    def plot_correlation_matrix(self, returns_df: pd.DataFrame, title: str = 'How correlated are the assets?', save_path: str = None) -> None:
        """
        Seaborn heatmap of asset return correlations.
        
        Mathematical concept:
            Pearson correlation matrix, symmetric, bounded [-1, 1].
        """
        plt.figure(figsize=(8, 6))
        corr = returns_df.corr()
        
        # Mask upper triangle for cleaner look
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, 
                    mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        plt.title(title)
        self._save_or_show(save_path)

    def plot_covariance_matrix(self, returns_df: pd.DataFrame, annualise: bool = True, save_path: str = None) -> None:
        """
        Heatmap of the covariance matrix.
        
        Mathematical concept:
            Covariance captures both correlation and individual volatilities.
            Used directly in Markowitz portfolio optimization.
        """
        plt.figure(figsize=(8, 6))
        cov = returns_df.cov()
        if annualise:
            cov = cov * 252
            
        sns.heatmap(cov, annot=True, cmap="viridis", fmt=".4f",
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        plt.title("What is the portfolio risk structure? (Annualised Covariance)")
        self._save_or_show(save_path)

    def plot_factor_loadings(self, factor_results: dict, ticker: str, save_path: str = None) -> None:
        """
        Bar chart of factor betas with confidence intervals.
        
        Mathematical concept:
            Multiple linear regression coefficients (betas) representing exposure to systematic risk factors.
        """
        plt.figure()
        
        factors = list(factor_results.keys())
        betas = [factor_results[f]['beta'] for f in factors]
        errors = [factor_results[f]['se'] * 1.96 for f in factors] # 95% CI
        
        x_pos = np.arange(len(factors))
        
        plt.bar(x_pos, betas, yerr=errors, align='center', alpha=0.8, color=self.primary_color, capsize=5)
        plt.xticks(x_pos, factors)
        plt.axhline(0, color='black', linewidth=1)
        
        plt.title(f"What factors drive {ticker}'s returns?")
        plt.ylabel("Factor Beta (with 95% CI)")
        self._save_or_show(save_path)

    def plot_rolling_beta(self, rolling_beta: pd.Series, ticker: str, save_path: str = None) -> None:
        """
        Time series of rolling beta with 95% CI band.
        
        Mathematical concept:
            Time-varying OLS slope parameter measuring changing market risk exposure.
        """
        plt.figure()
        
        # Simulate CI bands if not provided (assuming rolling_beta is just the series of estimates)
        # In a real scenario, standard errors would be calculated per window
        approx_se = rolling_beta.std() * 0.5 # Dummy approximation for visualization
        upper_bound = rolling_beta + 1.96 * approx_se
        lower_bound = rolling_beta - 1.96 * approx_se
        
        plt.plot(rolling_beta.index, rolling_beta, color=self.primary_color, label='Rolling Beta')
        plt.fill_between(rolling_beta.index, lower_bound, upper_bound, color=self.primary_color, alpha=0.2, label='95% Confidence Interval')
        plt.axhline(1.0, color='gray', linestyle='--', label='Market Beta = 1')
        
        plt.title(f"Is {ticker}'s market exposure stable?")
        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.legend()
        self._save_or_show(save_path)

    def plot_efficient_frontier(self, frontier_df: pd.DataFrame, optimal_portfolio: dict = None, save_path: str = None) -> None:
        """
        Scatter plot of return vs volatility for simulated portfolios forming the efficient frontier.
        
        Mathematical concept:
            Modern Portfolio Theory: Maximizing expected return for a given level of risk.
        """
        plt.figure()
        
        plt.scatter(frontier_df['volatility'], frontier_df['return'], 
                    c=frontier_df['sharpe'], cmap='viridis', marker='o', s=10, alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        
        if optimal_portfolio:
            if 'max_sharpe' in optimal_portfolio:
                plt.scatter(optimal_portfolio['max_sharpe']['volatility'], 
                            optimal_portfolio['max_sharpe']['return'], 
                            marker='*', color='red', s=200, label='Max Sharpe')
            if 'min_volatility' in optimal_portfolio:
                plt.scatter(optimal_portfolio['min_volatility']['volatility'], 
                            optimal_portfolio['min_volatility']['return'], 
                            marker='*', color='blue', s=200, label='Min Volatility')
            if 'equal_weight' in optimal_portfolio:
                plt.scatter(optimal_portfolio['equal_weight']['volatility'], 
                            optimal_portfolio['equal_weight']['return'], 
                            marker='X', color='black', s=100, label='Equal Weight')
        
        plt.title("Where is the optimal risk-return tradeoff?")
        plt.xlabel("Annualised Volatility (Risk)")
        plt.ylabel("Annualised Expected Return")
        plt.legend()
        self._save_or_show(save_path)

    def plot_drawdown(self, prices: pd.Series, ticker: str, save_path: str = None) -> None:
        """
        Price series and drawdown series.
        
        Mathematical concept:
            Drawdown measures the percentage decline from a historical peak.
            DD_t = (P_t - Max(P_0..t)) / Max(P_0..t)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        # Price panel
        ax1.plot(prices, color=self.primary_color)
        ax1.set_title(f"What is the worst historical loss from peak for {ticker}?")
        ax1.set_ylabel("Price")
        
        # Drawdown panel
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        
        ax2.fill_between(drawdown.index, drawdown, 0, color=self.accent_color, alpha=0.5)
        ax2.plot(drawdown, color='darkred', linewidth=1)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        
        # Mark max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.scatter(max_dd_date, max_dd, color='black', zorder=5)
        ax2.annotate(f'Max DD: {max_dd:.2%}', xy=(max_dd_date, max_dd), 
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        self._save_or_show(save_path)

    def plot_monte_carlo(self, simulated_paths: np.ndarray, percentiles: dict, horizon_days: int, title: str = '', save_path: str = None) -> None:
        """
        Monte Carlo simulation paths with percentile highlights.
        
        Mathematical concept:
            Stochastic processes (e.g., Geometric Brownian Motion) simulating future asset price paths.
        """
        plt.figure()
        
        # Plot a subset of paths (max 100)
        num_paths_to_plot = min(100, simulated_paths.shape[1])
        days = np.arange(horizon_days + 1)
        
        plt.plot(days, simulated_paths[:, :num_paths_to_plot], color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot percentiles
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        labels = ['5th', '25th', '50th', '75th', '95th']
        keys = ['p5', 'p25', 'p50', 'p75', 'p95']
        
        for i, key in enumerate(keys):
            if key in percentiles:
                plt.plot(days, percentiles[key], color=colors[i], linewidth=2, label=f'{labels[i]} Percentile')
        
        plt.title(f"SIMULATION, NOT PREDICTION\nWhat range of outcomes is consistent with historical statistics? {title}")
        plt.xlabel("Days into Future")
        plt.ylabel("Simulated Value")
        plt.legend()
        self._save_or_show(save_path)

    def plot_sensitivity(self, param_name: str, param_values: list, scores: list, baseline: float, save_path: str = None) -> None:
        """
        Line plot showing how a score changes with a parameter.
        
        Mathematical concept:
            Partial derivative/sensitivity of model output with respect to input parameter changes.
        """
        plt.figure()
        
        plt.plot(param_values, scores, marker='o', color=self.primary_color, linestyle='-')
        
        # Mark baseline
        baseline_score = np.interp(baseline, param_values, scores)
        plt.axvline(baseline, color=self.accent_color, linestyle='--', label=f'Baseline {param_name}={baseline}')
        plt.scatter(baseline, baseline_score, color=self.accent_color, s=100, zorder=5)
        
        plt.title(f"How sensitive is the score to {param_name}?")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.legend()
        self._save_or_show(save_path)

    def plot_module_scores_radar(self, module_scores: dict, ticker: str, save_path: str = None) -> None:
        """
        Radar/spider chart of M1-M6 scores.
        
        Mathematical concept:
            Multidimensional vector visualization mapping performance across orthogonal features.
        """
        # Ensure we have data
        if not module_scores:
            return
            
        labels = list(module_scores.keys())
        values = list(module_scores.values())
        
        # Number of variables
        N = len(labels)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], labels)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=8)
        plt.ylim(0, 100)
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=self.primary_color)
        
        # Fill area
        ax.fill(angles, values, self.primary_color, alpha=0.25)
        
        plt.title(f"What is the profile of {ticker} across all six dimensions?", y=1.1)
        self._save_or_show(save_path)

    def plot_regime_performance(self, regime_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Grouped bar chart of performance across regimes.
        
        Mathematical concept:
            Conditional expectation of performance given unobserved market regimes (e.g. from HMM).
        """
        plt.figure()
        
        # Assuming regime_df has columns ['Regime', 'Model_Sharpe', 'Benchmark_Sharpe']
        if 'Regime' in regime_df.columns:
            regime_df = regime_df.set_index('Regime')
            
        x = np.arange(len(regime_df.index))
        width = 0.35
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, regime_df['Model_Sharpe'], width, label='Model', color=self.primary_color)
        rects2 = ax.bar(x + width/2, regime_df['Benchmark_Sharpe'], width, label='Benchmark', color=self.secondary_color)
        
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title("Does the model perform consistently across market conditions?")
        ax.set_xticks(x)
        ax.set_xticklabels(regime_df.index)
        ax.legend()
        
        self._save_or_show(save_path)

    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Heatmap of models vs metrics.
        
        Mathematical concept:
            Model selection based on multiple criteria (Complexity vs Performance tradeoff).
        """
        plt.figure(figsize=(10, 6))
        
        # Standardize columns for better heatmap visualization
        normalized_df = (comparison_df - comparison_df.min()) / (comparison_df.max() - comparison_df.min())
        
        sns.heatmap(normalized_df, annot=comparison_df, cmap='YlGnBu', fmt='.2f', 
                    linewidths=1, cbar_kws={'label': 'Normalized Score (Higher is Better)'})
        
        plt.title("Does mathematical complexity add value?")
        self._save_or_show(save_path)

    def plot_backtesting_results(self, backtest_results: list, benchmark_returns: pd.Series, save_path: str = None) -> None:
        """
        Cumulative return series for each fold vs benchmark.
        
        Mathematical concept:
            Out-of-sample expected return estimation via k-fold cross-validation or walk-forward testing.
        """
        plt.figure()
        
        # Plot benchmark
        bench_cum = (1 + benchmark_returns).cumprod()
        plt.plot(bench_cum.index, bench_cum, color='black', linewidth=2, label='Benchmark')
        
        # Plot folds (assuming backtest_results is a list of dicts with 'returns' Series)
        for i, fold in enumerate(backtest_results):
            if 'returns' in fold:
                returns = fold['returns']
                fold_cum = (1 + returns).cumprod()
                # Align starting point to benchmark if needed or plot relative to its own start
                plt.plot(fold_cum.index, fold_cum, alpha=0.7, label=f'Fold {i+1}')
                
        plt.title("What was out-of-sample performance?")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        self._save_or_show(save_path)


    def plot_weight_stability(self, track1_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Grouped bar chart comparing weight stability (turnover) of Sample, Ledoit-Wolf, and HRP.
        """
        plt.figure()
        
        splits = track1_df['split']
        turnover_sample = track1_df['turnover_sample']
        turnover_lw = track1_df['turnover_lw']
        turnover_hrp = track1_df.get('turnover_hrp')
        
        x = np.arange(len(splits))
        
        if turnover_hrp is not None:
            width = 0.25
            plt.bar(x - width, turnover_sample, width, label='Sample Covariance', color=self.accent_color)
            plt.bar(x, turnover_lw, width, label='Ledoit-Wolf', color=self.primary_color)
            plt.bar(x + width, turnover_hrp, width, label='HRP', color=self.secondary_color)
        else:
            width = 0.35
            plt.bar(x - width/2, turnover_sample, width, label='Sample Covariance', color=self.accent_color)
            plt.bar(x + width/2, turnover_lw, width, label='Ledoit-Wolf', color=self.primary_color)
        
        plt.title("Does the method reduce weight instability?")
        plt.xlabel("Walk-Forward Split")
        plt.ylabel("Turnover (L1 Norm)")
        plt.xticks(x, splits)
        plt.legend()
        
        self._save_or_show(save_path)

    def plot_condition_number_timeseries(self, track1_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Line chart (log-y) comparing condition number of Sample vs Ledoit-Wolf over time.
        """
        plt.figure()
        
        plt.plot(track1_df['test_end'], track1_df['cond_sample'], marker='o', label='Sample Covariance', color=self.accent_color)
        plt.plot(track1_df['test_end'], track1_df['cond_lw'], marker='s', label='Ledoit-Wolf', color=self.primary_color)
        
        plt.yscale('log')
        plt.title("How ill-conditioned is the covariance matrix?")
        plt.xlabel("Date")
        plt.ylabel("Condition Number (Log Scale)")
        plt.xticks(rotation=45)
        plt.legend()
        
        self._save_or_show(save_path)

    def plot_sharpe_distribution(self, track2_paths_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Box plot showing distribution of out-of-sample Sharpe ratios across CPCV paths.
        """
        plt.figure()
        
        cols = ['sr_equal_weight', 'sr_naive_markowitz', 'sr_regularised_markowitz', 'sr_hrp']
        labels = ['Equal Weight', 'Naive Markowitz', 'Regularised Markowitz', 'HRP']
        
        data = [track2_paths_df[col].dropna() for col in cols]
        
        plt.boxplot(data, patch_artist=True, tick_labels=labels,
                    boxprops=dict(facecolor=self.primary_color, color='black'),
                    medianprops=dict(color='orange', linewidth=2))
        
        plt.title("OOS Sharpe Ratio Distribution across Combinatorial Paths")
        plt.ylabel("Annualised Sharpe Ratio")
        plt.axhline(0, color='gray', linestyle='--')
        plt.xticks(rotation=15)
        
        self._save_or_show(save_path)

    def plot_significance_heatmap(self, significance_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Heatmap of pairwise statistical significance (p-values).
        Expects a DataFrame where index and columns are method names, and values are p-values.
        """
        plt.figure(figsize=(8, 6))
        
        # We want small p-values to be prominent.
        sns.heatmap(significance_df, annot=True, cmap="YlOrRd_r", vmin=0, vmax=1,
                    square=True, linewidths=.5, cbar_kws={'label': 'p-value'})
        
        plt.title("Statistical Significance (p-values of difference)")
        self._save_or_show(save_path)
