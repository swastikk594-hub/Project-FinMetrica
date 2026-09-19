"""
Module 8: Financial NLP Sentiment & Corporate Event Detection Engine
=====================================================================
PURPOSE:
  Extract quantitative sentiment and corporate event signals from unstructured
  financial texts (news, earnings call transcripts, 8-K filings) with strict
  point-in-time publication timestamp tagging.

MATHEMATICAL FOUNDATION:
  1. Financial Sentiment Distribution:
     Outputs softmax probability vector [P(Pos), P(Neu), P(Neg)].
     Raw Sentiment Score = P(Positive) - P(Negative) in [-1, +1].

  2. EWMA Sentiment Smoothing:
     S_t = λ * S_{t-1} + (1 - λ) * S_t^{daily},  with λ = 0.85 (halflife ~ 4.3 days)

  3. Sentiment Momentum:
     ΔS_t = S_t - S_{t-20}

  4. Event Classification:
     Rule & Lexicon matching for critical corporate events:
     - Earnings Surprises & Guidance Upgrades
     - Management Changes (CEO/CFO resignation)
     - Regulatory Investigations & Lawsuits
     - Dividend Cuts & Debt Downgrades
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)


# Financial Lexicon Keywords (Loughran-McDonald & FinBERT aligned)
POSITIVE_FINANCIAL_TERMS = {
    'record', 'profit', 'exceeded', 'surpassed', 'growth', 'outperformed',
    'expansion', 'strong', 'increased', 'dividend', 'buyback', 'raised',
    'upgrade', 'favorable', 'optimistic', 'gain', 'rebound', 'resilient'
}

NEGATIVE_FINANCIAL_TERMS = {
    'loss', 'dropped', 'declined', 'missed', 'recession', 'headwind',
    'downgrade', 'lawsuit', 'investigation', 'resignation', 'restructuring',
    'bankruptcy', 'default', 'impairment', 'deficit', 'slump', 'weakness'
}

EVENT_PATTERNS = {
    'earnings_beat': [r'beats? estimates?', r'revenue above expectations', r'record earnings'],
    'earnings_miss': [r'misses? estimates?', r'revenue below expectations', r'profit falls short'],
    'guidance_cut': [r'lowers? guidance', r'cuts? forecast', r'profit warning'],
    'management_change': [r'ceo resigns?', r'cfo steps down', r'executive departure'],
    'regulatory_probe': [r'sec investigation', r'regulatory inquiry', r'subpoena issued'],
    'capital_return': [r'share repurchase', r'stock buyback', r'dividend hike']
}


class FinancialNLPEngine:
    """
    Processes financial news and corporate disclosures to compute daily sentiment,
    EWMA sentiment, momentum, and detected event flags.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ewma_decay = self.config.get('nlp', {}).get('ewma_decay', 0.85)

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Computes financial sentiment probabilities from raw text.
        """
        if not text or not isinstance(text, str):
            return {'prob_pos': 0.33, 'prob_neu': 0.34, 'prob_neg': 0.33, 'sentiment_score': 0.0}

        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not tokens:
            return {'prob_pos': 0.33, 'prob_neu': 0.34, 'prob_neg': 0.33, 'sentiment_score': 0.0}

        pos_count = sum(1 for t in tokens if t in POSITIVE_FINANCIAL_TERMS)
        neg_count = sum(1 for t in tokens if t in NEGATIVE_FINANCIAL_TERMS)
        total_sentiment_words = pos_count + neg_count

        if total_sentiment_words == 0:
            return {'prob_pos': 0.15, 'prob_neu': 0.70, 'prob_neg': 0.15, 'sentiment_score': 0.0}

        p_pos = pos_count / (total_sentiment_words + 2.0)
        p_neg = neg_count / (total_sentiment_words + 2.0)
        p_neu = max(0.0, 1.0 - p_pos - p_neg)

        sentiment_score = p_pos - p_neg # Scale [-1.0, +1.0]

        return {
            'prob_pos': float(p_pos),
            'prob_neu': float(p_neu),
            'prob_neg': float(p_neg),
            'sentiment_score': float(sentiment_score)
        }

    def detect_events(self, text: str) -> List[str]:
        """Identifies specific corporate event categories."""
        if not text:
            return []
        detected = []
        text_lower = text.lower()
        for event_type, patterns in EVENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    detected.append(event_type)
                    break
        return detected

    def compute_daily_sentiment_series(
        self,
        news_records: List[Dict[str, Any]],
        index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Aggregates document-level sentiment into daily EWMA sentiment and momentum series.
        """
        daily_sentiment = pd.Series(0.0, index=index)
        events_by_date = {}

        for doc in news_records:
            t_pub = pd.to_datetime(doc.get('timestamp', doc.get('date')))
            # Find closest matching trading date at or before t_pub
            valid_dates = index[index <= t_pub]
            if not valid_dates.empty:
                dt = valid_dates[-1]
                scores = self.score_text(doc.get('text', doc.get('title', '')))
                daily_sentiment.loc[dt] += scores['sentiment_score']

                detected = self.detect_events(doc.get('text', ''))
                if detected:
                    events_by_date.setdefault(dt, []).extend(detected)

        # Clip daily sentiment
        daily_sentiment = daily_sentiment.clip(-1.0, 1.0)
        
        # Calculate EWMA
        ewma_sentiment = daily_sentiment.ewm(alpha=1.0 - self.ewma_decay).mean()
        # 20-Day Momentum
        sentiment_momentum = ewma_sentiment - ewma_sentiment.shift(20).fillna(0.0)

        df = pd.DataFrame({
            'daily_sentiment': daily_sentiment,
            'ewma_sentiment': ewma_sentiment,
            'sentiment_momentum': sentiment_momentum
        }, index=index)

        return df

    def run(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        news_records: Optional[List[Dict[str, Any]]] = None
    ) -> ModuleResult:
        """Executes full Financial NLP Module analysis."""
        logger.info(f"Running Financial NLP Module for {ticker}")
        news = news_records or []
        
        if price_data.empty:
            return ModuleResult(ticker=ticker, module_name='NLP', normalised_score=50.0)

        index = price_data.index
        sent_df = self.compute_daily_sentiment_series(news, index)
        
        latest_ewma = float(sent_df['ewma_sentiment'].iloc[-1]) if not sent_df.empty else 0.0
        latest_mom = float(sent_df['sentiment_momentum'].iloc[-1]) if not sent_df.empty else 0.0

        # Normalise to 0-100 scale: 0.0 sentiment -> 50 score
        norm_score = 50.0 + (latest_ewma * 35.0) + (latest_mom * 15.0)
        norm_score = float(np.clip(norm_score, 0.0, 100.0))

        return ModuleResult(
            ticker=ticker,
            module_name='NLP',
            normalised_score=norm_score,
            confidence=0.8 if news else 0.3,
            components={
                'latest_ewma_sentiment': latest_ewma,
                'sentiment_momentum': latest_mom,
                'news_items_count': len(news)
            },
            metadata={'sentiment_summary': f"EWMA: {latest_ewma:.2f}, Momentum: {latest_mom:.2f}"}
        )
