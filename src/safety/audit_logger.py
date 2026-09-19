"""
Institutional Decision Audit Trail Logger
==========================================
PURPOSE:
  Record immutable, auditable JSONL execution trails answering:
  "Why did the system hold this position at this exact timestamp?"
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logs every portfolio allocation decision with its complete model rationale.
    """

    def __init__(self, log_path: str = "results/audit_trail.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log_decision(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        target_weight: float,
        previous_weight: float,
        expected_return: float,
        composite_score: float,
        dominant_regime: str,
        model_version: str = "v2.0",
        rationale: str = ""
    ) -> None:
        """Appends a structured decision record."""
        record = {
            'timestamp': str(timestamp),
            'ticker': ticker,
            'target_weight': float(target_weight),
            'previous_weight': float(previous_weight),
            'weight_delta': float(target_weight - previous_weight),
            'expected_return': float(expected_return),
            'composite_score': float(composite_score),
            'dominant_regime': dominant_regime,
            'model_version': model_version,
            'rationale': rationale
        }

        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.debug(f"Audit log write failed: {e}")
