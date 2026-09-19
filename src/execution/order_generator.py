import pandas as pd
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Translates portfolio weights into actionable real-world execution tickets,
    applying minimum turnover thresholds and ADV liquidity limits.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_trade_pct = self.config.get('execution', {}).get('min_trade_pct_aum', 0.01) # 1% of AUM minimum trade size
        self.max_adv_pct = self.config.get('execution', {}).get('max_adv_participation', 0.05) # 5% ADV limit

    def generate_order_tickets(
        self,
        target_weights: Dict[str, float],
        latest_prices: Dict[str, float],
        adv_data: Dict[str, float],
        capital: float,
        current_holdings: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Creates actionable buy/sell tickets comparing target to current holdings.
        """
        current_holdings = current_holdings or {}
        
        logger.info(f"Generating execution tickets for AUM ${capital:,.2f}")
        
        tickets = []
        target_shares_dict = {}
        total_allocated = 0.0
        
        # 1. Convert Target Weights to Integer Shares
        for ticker, weight in target_weights.items():
            price = latest_prices.get(ticker, 1e-6)
            target_dollar = capital * weight
            target_shares = int(target_dollar / price)
            target_shares_dict[ticker] = target_shares
            total_allocated += target_shares * price
            
        residual_cash = capital - total_allocated
        
        # 2. Compute Deltas against Current Holdings
        all_tickers = set(list(target_weights.keys()) + list(current_holdings.keys()))
        
        for ticker in all_tickers:
            current_s = current_holdings.get(ticker, 0)
            target_s = target_shares_dict.get(ticker, 0)
            delta_s = target_s - current_s
            
            if delta_s == 0:
                continue
                
            price = latest_prices.get(ticker, 1.0)
            trade_value = abs(delta_s) * price
            
            # Turnover Filter: Skip micro trades
            if trade_value < (self.min_trade_pct * capital):
                logger.debug(f"Skipping micro-trade for {ticker}: Value ${trade_value:.2f} < Minimum ${(self.min_trade_pct * capital):.2f}")
                continue
                
            # Liquidity Filter: Check against Average Daily Volume
            adv = adv_data.get(ticker, 1_000_000)
            adv_value = adv * price
            adv_participation = trade_value / adv_value if adv_value > 0 else 1.0
            
            warning = None
            if adv_participation > self.max_adv_pct:
                warning = f"High Market Impact: Trade is {adv_participation*100:.1f}% of ADV (Limit {self.max_adv_pct*100:.1f}%)"
                
            tickets.append({
                'ticker': ticker,
                'action': 'BUY' if delta_s > 0 else 'SELL',
                'shares': abs(delta_s),
                'target_shares': target_s,
                'est_price': price,
                'est_trade_value': trade_value,
                'adv_participation_pct': adv_participation * 100,
                'warning': warning
            })
            
        # Sort tickets: Sells first (to free up cash), then Buys (largest to smallest)
        sells = sorted([t for t in tickets if t['action'] == 'SELL'], key=lambda x: x['est_trade_value'], reverse=True)
        buys = sorted([t for t in tickets if t['action'] == 'BUY'], key=lambda x: x['est_trade_value'], reverse=True)
        
        return {
            'residual_cash': residual_cash,
            'total_aum': capital,
            'tickets': sells + buys
        }
