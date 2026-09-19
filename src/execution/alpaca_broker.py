import requests
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AlpacaBroker:
    """
    Adapter for Alpaca Paper Trading REST API.
    """
    
    BASE_URL = "https://paper-api.alpaca.markets"
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "accept": "application/json"
        }
        
    def get_account_info(self) -> Dict[str, Any]:
        """Fetches account details (buying power, portfolio value)."""
        url = f"{self.BASE_URL}/v2/account"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch Alpaca account: {response.text}")
            return {}

    def get_current_holdings(self) -> Dict[str, int]:
        """Fetches current portfolio positions mapping ticker to qty."""
        url = f"{self.BASE_URL}/v2/positions"
        response = requests.get(url, headers=self.headers)
        
        holdings = {}
        if response.status_code == 200:
            positions = response.json()
            for pos in positions:
                holdings[pos['symbol']] = int(pos['qty'])
        else:
            logger.error(f"Failed to fetch Alpaca positions: {response.text}")
            
        return holdings
        
    def submit_orders(self, tickets: List[Dict[str, Any]]) -> None:
        """
        Takes the execution engine tickets and submits market orders.
        """
        url = f"{self.BASE_URL}/v2/orders"
        
        for ticket in tickets:
            if ticket['shares'] <= 0:
                continue
                
            payload = {
                "symbol": ticket['ticker'],
                "qty": str(ticket['shares']),
                "side": ticket['action'].lower(), # 'buy' or 'sell'
                "type": "market",
                "time_in_force": "day"
            }
            
            logger.info(f"Submitting Alpaca Order: {ticket['action']} {ticket['shares']} {ticket['ticker']}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code in [200, 201]:
                logger.info(f"Successfully placed order for {ticket['ticker']}")
            else:
                logger.error(f"Order failed for {ticket['ticker']}: {response.text}")
