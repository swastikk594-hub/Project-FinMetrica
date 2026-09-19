import requests
import json

headers = {
    'APCA-API-KEY-ID': 'YOUR_API_KEY_HERE',
    'APCA-API-SECRET-KEY': 'YOUR_SECRET_KEY_HERE'
}

acc_res = requests.get('https://paper-api.alpaca.markets/v2/account', headers=headers)
pos_res = requests.get('https://paper-api.alpaca.markets/v2/positions', headers=headers)

if acc_res.status_code == 200:
    acc = acc_res.json()
    equity = float(acc.get('equity', 0))
    cash = float(acc.get('cash', 0))
    last_equity = float(acc.get('last_equity', 0))
    print(f"Total Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
    print(f"Daily P&L: ${equity - last_equity:,.2f} ({((equity - last_equity) / last_equity)*100:.2f}%)")
    print("-" * 65)

if pos_res.status_code == 200:
    positions = pos_res.json()
    if not positions:
        print("No open positions found.")
    for p in positions:
        sym = p['symbol']
        qty = p['qty']
        entry = float(p['avg_entry_price'])
        cur = float(p['current_price'])
        pl = float(p['unrealized_pl'])
        plpc = float(p['unrealized_plpc']) * 100.0
        print(f"{sym:5s} | Qty: {qty:>4s} | Entry: ${entry:8.2f} | Now: ${cur:8.2f} | P/L: ${pl:8.2f} ({plpc:+.2f}%)")
else:
    print("Failed to fetch positions:", pos_res.text)
