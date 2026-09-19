import pandas as pd
import numpy as np
import ast
from typing import Callable, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LeakageValidator:
    """Validates the absence of forward-looking leakage in research implementations."""

    @staticmethod
    def test_cross_sectional_purity(normalize_fn: Callable[[pd.DataFrame], pd.DataFrame], df_date1: pd.DataFrame, df_date2: pd.DataFrame) -> bool:
        """
        Test that normalization at date 1 is unaffected by data from date 2.
        """
        try:
            result_a = normalize_fn(df_date1.copy())
            combined = pd.concat([df_date1, df_date2])
            result_combined = normalize_fn(combined.copy())
            result_b = result_combined.loc[df_date1.index]
            pd.testing.assert_frame_equal(result_a, result_b)
            return True
        except Exception as e:
            logger.error(f"Cross-sectional purity test failed: {e}")
            return False

    @staticmethod
    def test_look_ahead_mutation(factor_compute_fn: Callable, price_data: pd.DataFrame, pit_store: Any, ticker: str, as_of: pd.Timestamp) -> bool:
        """
        Test that modifying data AFTER as_of does not change factor scores AT as_of.
        """
        try:
            # 1. Compute factor scores using data through as_of
            df_before = pit_store.get_price_history(ticker, as_of)
            scores_before = factor_compute_fn(df_before)

            # 2. Add fake data after as_of to price_data
            fake_dates = pd.date_range(start=as_of + pd.Timedelta(days=1), periods=5)
            fake_data = pd.DataFrame(np.random.randn(5, len(price_data.columns)), index=fake_dates, columns=price_data.columns)
            pit_store.register_price_data(ticker, pd.concat([price_data, fake_data]))
            
            # 3. Recompute factor scores using data through as_of
            df_after = pit_store.get_price_history(ticker, as_of)
            scores_after = factor_compute_fn(df_after)

            # 4. Assert scores_before == scores_after
            pd.testing.assert_series_equal(scores_before, scores_after)
            return True
        except Exception as e:
            logger.error(f"Look-ahead mutation test failed: {e}")
            return False

    @staticmethod
    def test_treatment_equivalence(raw_characteristics_by_norm: Dict[str, pd.DataFrame]) -> bool:
        """
        Verify all normalization treatments received identical raw inputs.
        """
        try:
            items = list(raw_characteristics_by_norm.values())
            if not items:
                return True
            base_df = items[0]
            for df in items[1:]:
                pd.testing.assert_frame_equal(base_df, df)
            return True
        except Exception as e:
            logger.error(f"Treatment equivalence test failed: {e}")
            return False

    @staticmethod
    def test_no_execution_imports(research_dir: Path) -> bool:
        """
        Verify no execution related module imports in research dir.
        """
        forbidden_modules = ['src.execution']
        try:
            for py_file in research_dir.rglob('*.py'):
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(alias.name.startswith(f) for f in forbidden_modules):
                                return False
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(node.module.startswith(f) for f in forbidden_modules):
                            return False
            return True
        except Exception as e:
            logger.error(f"No execution imports test failed: {e}")
            return False

    @staticmethod
    def test_target_independence(factor_scores: pd.Series, forward_returns: pd.Series) -> bool:
        """
        Verify that factor scores do not contain information about forward returns beyond what's expected.
        """
        try:
            shifted_returns = forward_returns.shift(-1)
            common_idx = factor_scores.index.intersection(shifted_returns.dropna().index)
            if len(common_idx) < 2:
                return True
            corr = factor_scores.loc[common_idx].corr(shifted_returns.loc[common_idx])
            if abs(corr) > 0.5:
                logger.warning(f"Suspiciously high correlation: {corr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Target independence test failed: {e}")
            return False

    @classmethod
    def run_all_checks(cls, research_dir: Path) -> dict:
        """
        Run all validation checks and return summary.
        """
        results = {
            'no_execution_imports': cls.test_no_execution_imports(research_dir),
        }
        return results
