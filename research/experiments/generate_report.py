"""
Report Generator
================
Runs Track 1 and Track 2 experiments (or loads existing results),
generates visualizations, and writes the final FINDINGS.md report.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from research.experiments.covariance_regularization_study import run_track1_study
from research.experiments.full_system_benchmark import run_track2_study
from src.visualisation.plots import Plots

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PURE FUNCTION: build_headline
# Extracted so it can be unit-tested independently of the rest of the report.
# ─────────────────────────────────────────────────────────────────────────────
def build_headline(sig_df: "pd.DataFrame") -> str:
    """
    Build the Executive Summary / Core Finding headline text from the
    full significance table.  Uses the BH-adjusted significance column
    as the primary criterion.

    Handles all realistic cases:
      - No pairs significant  → says so plainly
      - Exactly one pair sig  → names it specifically
      - Multiple pairs sig    → summarises the full pattern, never reports
                                only the single lowest-p pair when others
                                also cross the threshold

    Parameters
    ----------
    sig_df : pd.DataFrame
        Must contain columns:
          'Method A', 'Method B',
          'Sharpe A', 'Sharpe B',
          'Bootstrap p-value', 'Significant (5% BH)'

    Returns
    -------
    str
        Human-readable headline text.
    """
    import pandas as pd

    sig_pairs = sig_df[sig_df["Significant (5% BH)"] == True]

    # ── Case 1: Nothing significant ───────────────────────────────────────────
    if sig_pairs.empty:
        best_idx  = sig_df[["Sharpe A", "Sharpe B"]].stack().idxmax()
        best_sharpe = sig_df[["Sharpe A", "Sharpe B"]].stack().max()
        # Identify the method with the highest point-estimate Sharpe
        all_methods = list(set(sig_df["Method A"].tolist() + sig_df["Method B"].tolist()))
        method_sharpes = {}
        for _, row in sig_df.iterrows():
            method_sharpes[row["Method A"]] = row["Sharpe A"]
            method_sharpes[row["Method B"]] = row["Sharpe B"]
        best_method = max(method_sharpes, key=method_sharpes.get)
        best_sr     = method_sharpes[best_method]

        # Find EW p-value for context
        ew_row = sig_df[
            ((sig_df["Method A"] == "Equal Weight") & (sig_df["Method B"] == best_method)) |
            ((sig_df["Method B"] == "Equal Weight") & (sig_df["Method A"] == best_method))
        ]
        ew_pval_str = f" (p={ew_row.iloc[0]['Bootstrap p-value']:.4f})" if not ew_row.empty else ""

        return (
            f"There were **no statistically significant** differences among the methods "
            f"at the 5% level after Benjamini-Hochberg multiple-comparison correction. "
            f"The best point-estimate was {best_method} (Sharpe {best_sr:.2f}), "
            f"but this was not significantly better than any other method{ew_pval_str}."
        )

    # ── Determine winner/loser for each significant pair ─────────────────────
    # Build a directed "beats" graph: beats[winner].add(loser)
    all_methods = list(set(sig_df["Method A"].tolist() + sig_df["Method B"].tolist()))
    method_sharpes = {}
    for _, row in sig_df.iterrows():
        method_sharpes[row["Method A"]] = row["Sharpe A"]
        method_sharpes[row["Method B"]] = row["Sharpe B"]

    beats = {m: set() for m in all_methods}
    loses_to = {m: set() for m in all_methods}
    sig_p_values = {}  # (winner, loser) -> p-value

    for _, row in sig_pairs.iterrows():
        ma, mb = row["Method A"], row["Method B"]
        sa, sb = row["Sharpe A"], row["Sharpe B"]
        pval   = row["Bootstrap p-value"]
        if sa >= sb:
            winner, loser = ma, mb
        else:
            winner, loser = mb, ma
        beats[winner].add(loser)
        loses_to[loser].add(winner)
        sig_p_values[(winner, loser)] = pval

    # ── Case 2: Exactly one significant pair ─────────────────────────────────
    if len(sig_pairs) == 1:
        row    = sig_pairs.iloc[0]
        ma, mb = row["Method A"], row["Method B"]
        sa, sb = row["Sharpe A"], row["Sharpe B"]
        pval   = row["Bootstrap p-value"]
        winner = ma if sa >= sb else mb
        loser  = mb if sa >= sb else ma
        return (
            f"{winner} significantly outperforms {loser} "
            f"(BH-adjusted p={pval:.4f}). "
            f"No other pairwise comparison reaches statistical significance after correction."
        )

    # ── Case 3: Multiple significant pairs ───────────────────────────────────
    # Find methods that win at least one significant comparison
    winners = {m for m in all_methods if beats[m]}

    parts = []
    for winner in sorted(winners, key=lambda m: -method_sharpes.get(m, 0)):
        beaten = sorted(beats[winner])
        p_strs = [f"{loser} (BH p={sig_p_values[(winner, loser)]:.4f})" for loser in beaten]
        parts.append(f"{winner} significantly outperforms {', '.join(p_strs)}")

    # Check whether the best method beats the Equal Weight baseline
    best_method = max(method_sharpes, key=method_sharpes.get)
    ew_beaten   = "Equal Weight" in beats.get(best_method, set())

    # Assemble headline
    headline = ". ".join(parts) + "."

    if not ew_beaten:
        # Add explicit note that EW was NOT beaten
        ew_row = sig_df[
            ((sig_df["Method A"] == "Equal Weight") & (sig_df["Method B"] == best_method)) |
            ((sig_df["Method B"] == "Equal Weight") & (sig_df["Method A"] == best_method))
        ]
        if not ew_row.empty:
            ew_pval = ew_row.iloc[0]["Bootstrap p-value"]
            bh_pval = ew_row.iloc[0].get("BH p-value", ew_pval)
            headline += (
                f" However, no method — including {best_method} — "
                f"significantly beats Equal Weight after multiple-comparison correction "
                f"(BH p={bh_pval:.4f})."
            )

    return headline


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_findings_report(
    track1_df: "pd.DataFrame",
    track2_df: "pd.DataFrame",
    sig_df:    "pd.DataFrame",
    output_dir: str,
    run_params: dict
):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plots = Plots()

    # Track 1 figures
    t1_weight_path = os.path.join(fig_dir, "track1_weight_stability.png")
    plots.plot_weight_stability(track1_df, save_path=t1_weight_path)

    t1_cond_path = os.path.join(fig_dir, "track1_condition_number.png")
    plots.plot_condition_number_timeseries(track1_df, save_path=t1_cond_path)

    # Track 2 figures
    t2_dist_path = os.path.join(fig_dir, "track2_sharpe_dist.png")
    plots.plot_sharpe_distribution(track2_df, save_path=t2_dist_path)

    methods = ["Equal Weight", "Naive Markowitz", "Regularised Markowitz", "HRP"]
    n = len(methods)
    sig_mat = np.ones((n, n))
    for _, row in sig_df.iterrows():
        m1, m2 = row["Method A"], row["Method B"]
        if m1 in methods and m2 in methods:
            i, j = methods.index(m1), methods.index(m2)
            sig_mat[i, j] = row["Bootstrap p-value"]
            sig_mat[j, i] = row["Bootstrap p-value"]

    heatmap_df = pd.DataFrame(sig_mat, index=methods, columns=methods)
    t2_sig_path = os.path.join(fig_dir, "track2_significance.png")
    plots.plot_significance_heatmap(heatmap_df, save_path=t2_sig_path)

    avg_cond_sample = track1_df["cond_sample"].mean()
    avg_cond_lw     = track1_df["cond_lw"].mean()
    cond_reduction  = (1 - (avg_cond_lw / avg_cond_sample)) * 100 if avg_cond_sample > 0 else 0

    avg_turnover_sample = track1_df["turnover_sample"].mean()
    avg_turnover_lw     = track1_df["turnover_lw"].mean()
    
    if "turnover_hrp" in track1_df.columns:
        avg_turnover_hrp = track1_df["turnover_hrp"].mean()
        hrp_turnover_line = f"- HRP: {avg_turnover_hrp:.3f}\n"
        if avg_turnover_hrp < avg_turnover_lw and avg_turnover_hrp < avg_turnover_sample:
            hrp_turnover_sentence = f"\nFurthermore, Hierarchical Risk Parity (HRP) achieved the lowest average turnover ({avg_turnover_hrp:.3f}), empirically supporting the literature's claim that its hierarchical clustering approach produces more stable weight allocations across time than optimization approaches requiring matrix inversion."
        else:
            hrp_turnover_sentence = f"\nNote: While literature claims HRP produces more stable weight allocations, in this specific run its average turnover ({avg_turnover_hrp:.3f}) was NOT lower than the Markowitz methods. The 'more stable' claim is not empirically supported here."
    else:
        avg_turnover_hrp = None
        hrp_turnover_line = ""
        hrp_turnover_sentence = ""

    # ── Stability table ───────────────────────────────────────────────────────
    method_cols = {
        "Equal Weight":          "sr_equal_weight",
        "Naive Markowitz":       "sr_naive_markowitz",
        "Regularised Markowitz": "sr_regularised_markowitz",
        "HRP":                   "sr_hrp",
    }
    stability_data = []
    for name, col in method_cols.items():
        mean_val = track2_df[col].mean()
        std_val  = track2_df[col].std()
        cv_val   = std_val / abs(mean_val) if mean_val != 0 else float("nan")
        stability_data.append({"Method": name, "Mean OOS Sharpe": mean_val,
                                "Std Dev OOS Sharpe": std_val, "CV": cv_val})

    stability_df = pd.DataFrame(stability_data)

    cols = ["Method", "Mean OOS Sharpe", "Std Dev OOS Sharpe", "CV"]
    stab_header = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    stab_rows = [
        f"| {r['Method']} | {r['Mean OOS Sharpe']:.4f} | {r['Std Dev OOS Sharpe']:.4f} | {r['CV']:.4f} |"
        for _, r in stability_df.iterrows()
    ]
    stab_table_md = stab_header + "\n".join(stab_rows)

    hrp_row  = stability_df[stability_df["Method"] == "HRP"]
    hrp_std  = hrp_row["Std Dev OOS Sharpe"].values[0] if not hrp_row.empty else None
    min_std  = stability_df["Std Dev OOS Sharpe"].min()

    if hrp_std is not None and abs(hrp_std - min_std) < 1e-6:
        hrp_stability_sentence = (
            f"In this run, HRP indeed has the lowest cross-path Sharpe standard deviation "
            f"({hrp_std:.4f}) of all four methods. This supports a \'wins on consistency, "
            f"not average outperformance\' interpretation, directly connecting to Track 1 "
            f"(Ledoit-Wolf reduces covariance estimation variance; HRP structurally avoids "
            f"mean estimation variance — both are instability-reduction stories operating on "
            f"different parts of the estimation problem)."
        )
    else:
        hrp_std_str  = f"{hrp_std:.4f}" if hrp_std is not None else "N/A"
        best_std = min_std
        hrp_stability_sentence = (
        "In this run, HRP does NOT have the lowest cross-path Sharpe standard deviation "
        f"(its std is {hrp_std:.4f}, while the minimum was {best_std:.4f}). A pure 'wins on consistency' argument is therefore not supported by these numbers."
    ) if hrp_std > best_std else (
        "In this run, HRP does indeed have the lowest cross-path Sharpe standard deviation "
        f"({hrp_std:.4f}), strongly supporting the literature's claim that its hierarchical clustering "
        "approach produces more stable out-of-sample performance across different market regimes."
    )

    # --- Task B: Isolated Shrinkage Interpretation ---
    # Extract p-values from sig_df for Naive vs LW-Shrinkage-Only and LW-Shrinkage-Only vs Regularised
    p_naive_lw = 1.0
    p_lw_reg = 1.0
    for _, row in sig_df.iterrows():
        m1, m2 = row["Method A"], row["Method B"]
        if (m1 == "Naive Markowitz" and m2 == "LW-Shrinkage-Only Markowitz") or (m2 == "Naive Markowitz" and m1 == "LW-Shrinkage-Only Markowitz"):
            p_naive_lw = row["BH p-value"]
        if (m1 == "LW-Shrinkage-Only Markowitz" and m2 == "Regularised Markowitz") or (m2 == "LW-Shrinkage-Only Markowitz" and m1 == "Regularised Markowitz"):
            p_lw_reg = row["BH p-value"]
            
    if p_naive_lw < 0.05 and p_lw_reg >= 0.05:
        driver = "pure covariance shrinkage (Ledoit-Wolf)"
    elif p_lw_reg < 0.05 and p_naive_lw >= 0.05:
        driver = "the addition of Black-Litterman alpha views"
    elif p_naive_lw < 0.05 and p_lw_reg < 0.05:
        driver = "both covariance shrinkage and alpha views"
    else:
        driver = "neither modification alone (both yielded statistically indistinguishable differences)"
        
    shrinkage_interp = (
        f"\n\n**Isolation of Shrinkage vs. Alpha Views:** By introducing 'LW-Shrinkage-Only Markowitz' as a fifth method, "
        f"we isolated the two components of Regularised Markowitz. Comparing pure shrinkage against Naive Markowitz yielded "
        f"a p-value of {p_naive_lw:.4f}, while comparing Regularised Markowitz (which adds alpha views) against pure shrinkage "
        f"yielded a p-value of {p_lw_reg:.4f}. This indicates that {driver} is actually responsible for the "
        f"performance difference (if any) between Naive and Regularised Markowitz."
    )

    # ── Core Finding / Headline (Task B fix) ──────────────────────────────────
    strongest_finding = build_headline(sig_df)

    # ── DeMiguel et al. (2009) conditional framing (Task C) ──────────────────
    # Determine if any structured method significantly beats Equal Weight
    ew_beaten_by_any = any(
        (row["Method A"] == "Equal Weight" or row["Method B"] == "Equal Weight")
        and row["Significant (5% BH)"] == True
        for _, row in sig_df.iterrows()
    )

    if not ew_beaten_by_any:
        demiguel_section = (
            "This pattern — where no structured method significantly beats naive equal-weighting "
            "out-of-sample — is consistent with the finding in **DeMiguel, V., Garlappi, L., "
            "and Uppal, R. (2009), \'Optimal Versus Naive Diversification: How Inefficient is "
            "the 1/N Portfolio Strategy?\', Review of Financial Studies, 22(5), 1915–1953**, "
            "which reported that naive 1/N equal-weighting is difficult to beat across many "
            "'optimal' portfolio construction methods, because estimation error in the inputs "
            "erodes the theoretical benefits of optimization. "
            "**Note:** This citation is included as a qualitative framing consistent with the "
            "observed pattern — the user should verify specific claims against the source before "
            "treating them as confirmed."
        )
    else:
        demiguel_section = ""

    # ── DSR section (Task D: 4-decimal precision + sample-size note) ──────────
    dsr_path = os.path.join(output_dir, "track2_dsr.csv")
    if os.path.exists(dsr_path):
        dsr_df  = pd.read_csv(dsr_path)
        dsr_row = dsr_df.iloc[0]
        dsr_pct  = dsr_row["dsr_probability"] * 100.0
        sr_star  = dsr_row["expected_max_sharpe"]
        n_obs    = int(dsr_row["n_observations"])
        dsr_section = (
            f"### Overfitting Check: Deflated Sharpe Ratio\n"
            f"After adjusting for having tried {int(dsr_row['n_trials'])} methods, "
            f"the probability that {dsr_row['best_method']}'s true Sharpe ratio exceeds "
            f"the multiple-testing-adjusted benchmark of {sr_star:.6f} is **{dsr_pct:.4f}%**, "
            f"versus a naive (non-deflated) read of the raw Sharpe ratio.\n\n"
            f"*Sample-size note: n_observations = {n_obs}, representing the number of unique "
            f"calendar dates in the date-averaged ensemble return series (overlapping CPCV paths "
            f"are averaged per date before computing this, so this is NOT a naive pooled count). "
            f"Using the single representative fold size would give a more conservative estimate; "
            f"users should treat extremely high DSR values (>99%) with caution when "
            f"n_observations is large.*"
        )
    else:
        dsr_section = "*(Deflated Sharpe Ratio data not available for this run)*"

    # ── Significance table (Task B: all 8 columns) ────────────────────────────
    cols_sig = ["Method A", "Method B", "Sharpe A", "Sharpe B",
                "Raw p-val", "Bonferroni p", "BH p-val", "Sig (BH 5%)"]
    table_header = "| " + " | ".join(cols_sig) + " |\n| " + " | ".join(["---"] * len(cols_sig)) + " |\n"
    table_rows = []
    for _, row in sig_df.iterrows():
        row_vals = [
            str(row["Method A"]), str(row["Method B"]),
            f"{row['Sharpe A']:.4f}", f"{row['Sharpe B']:.4f}",
            f"{row['Bootstrap p-value']:.4f}", f"{row['Bonferroni p-value']:.4f}",
            f"{row['BH p-value']:.4f}", str(row["Significant (5% BH)"]),
        ]
        table_rows.append("| " + " | ".join(row_vals) + " |")
    sig_table_md = table_header + "\n".join(table_rows)

    # ── Assemble full report ──────────────────────────────────────────────────
    demiguel_block = f"\n{demiguel_section}\n" if demiguel_section else ""

    md_content = f"""# FinMetrica Research Findings: Covariance Regularization & System Benchmark
*Universe Name:* {run_params.get("universe_name", "Unknown")}

## Executive Summary
This report evaluates the empirical impact of advanced portfolio structuring (Ledoit-Wolf covariance shrinkage and Hierarchical Risk Parity) on portfolio stability and out-of-sample performance.

**Core Finding:** {strongest_finding}

* Regularization reduced the average condition number by **{cond_reduction:.1f}%** (from {avg_cond_sample:,.1f} to {avg_cond_lw:,.1f}).

## Track 1: Estimator Instability

Ledoit-Wolf shrinkage trades a small amount of bias for a large reduction in variance. This manifests as a better-conditioned covariance matrix.

![Condition Number](figures/track1_condition_number.png)

This improved conditioning directly impacts the stability of the optimizer's output weights.
Average turnover (L1 norm):
- Sample Covariance: {avg_turnover_sample:.3f}
- Ledoit-Wolf: {avg_turnover_lw:.3f}
{hrp_turnover_line}
{hrp_turnover_sentence}

![Weight Stability](figures/track1_weight_stability.png)

## Track 2: Full-System Benchmark (CPCV)

### Cross-Path Stability
{stab_table_md}

![Sharpe Distribution](figures/track2_sharpe_dist.png)

### Interpretation: Estimation Error and the Limits of Optimization

HRP is the only one of the four methods that does not use the expected-return vector (mu) at all — it allocates purely from the correlation and distance structure via recursive bisection. Both Markowitz variants (naive and regularized) use the noisy sample mean (mu_hat).

Ledoit-Wolf shrinkage in this experiment only regularizes the covariance matrix, not mu — so if mu estimation error is the dominant source of instability, Sigma-regularization alone would show little benefit. This pattern is consistent with **Chopra, V.K. and Ziemba, W.T. (1993), "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice," Journal of Portfolio Management**, which found that errors in mean estimates are typically far more damaging to portfolio choice than errors in the covariance matrix.
{demiguel_block}
{hrp_stability_sentence}

*Note: These are plausible explanations consistent with the observed patterns, not proven causal mechanisms.*

**Suggested Future Extension:** Introduce a fifth method (Ledoit-Wolf Sigma combined with a Minimum-Variance objective that ignores mu) to isolate whether "mu-avoidance" explains HRP's edge, versus something specific to HRP's clustering structure.

{dsr_section}

### Statistical Significance

The table below displays all pairwise Jobson-Korkie-Memmel and stationary-bootstrap tests for out-of-sample Sharpe equivalence, with Bonferroni and Benjamini-Hochberg (FDR) corrections applied.

{sig_table_md}

![Significance Heatmap](figures/track2_significance.png)

## Known Limitations

**Survivorship and Hindsight Bias**
This run was evaluated from `{run_params.get("start_date")}` using the following `{run_params.get("universe_name")}` universe:
`{", ".join(run_params.get("tickers", []))}`

If this list was selected using present-day knowledge of which companies became successful, applying it backward to a start date before that outcome was known represents a form of hindsight/survivorship bias. This likely inflates the *absolute* Sharpe ratios for all methods since they share the same biased universe.

While the *relative* comparison between methods is less affected since all are evaluated on the identical universe, this is an assumption. Evaluating against a point-in-time unbiased universe or a deliberately diverse alternate universe is recommended to verify these findings hold.

## Conclusion
{strongest_finding}
"""

    report_path = os.path.join(output_dir, "FINDINGS.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Report generated successfully at {report_path}")
    except PermissionError:
        report_path = os.path.join(output_dir, "FINDINGS_UPDATED.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.warning(f"FINDINGS.md was locked. Report saved to {report_path}")


def main(
    force_rerun:    bool  = False,
    fast_mode:      bool  = True,
    universe_name:  str   = "Demo",
    start_date:     str   = "2005-01-01",
    tickers:        list  = None,
    output_dir:     str   = "research/results",
):
    t1_path  = os.path.join(output_dir, "track1_estimator_instability.csv")
    t2_path  = os.path.join(output_dir, "track2_full_system_paths.csv")
    sig_path = os.path.join(output_dir, "track2_significance_tests.csv")

    if force_rerun or not os.path.exists(t1_path) or not os.path.exists(t2_path) or not os.path.exists(sig_path):
        logger.info(f"Running experiments for {universe_name} universe...")
        from research.data.loader import ResearchDataLoader
        loader = ResearchDataLoader({})
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "JNJ", "PFE", "UNH", "JPM", "BAC"]

        price_dict = loader.load_price_data(tickers, start=start_date, end="2023-12-31")
        prices = pd.DataFrame({ticker: df["Close"] for ticker, df in price_dict.items()})

        t1_df = run_track1_study(prices, output_dir=output_dir)
        t2_df = run_track2_study(prices, output_dir=output_dir, fast_mode=fast_mode)
    else:
        logger.info(f"Loading existing data for {universe_name}...")
        t1_df = pd.read_csv(t1_path)
        t2_df = pd.read_csv(t2_path)

    sig_df = pd.read_csv(sig_path)

    run_params = {
        "universe_name": universe_name,
        "start_date":    start_date,
        "tickers":       tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META", "JNJ", "PFE", "UNH", "JPM", "BAC"],
    }

    generate_findings_report(t1_df, t2_df, sig_df, output_dir, run_params)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fast",  action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(force_rerun=args.rerun, fast_mode=args.fast)
