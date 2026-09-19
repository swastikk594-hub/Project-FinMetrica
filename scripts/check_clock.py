import requests

headers = {
    'APCA-API-KEY-ID': 'YOUR_API_KEY_HERE',
    'APCA-API-SECRET-KEY': 'YOUR_SECRET_KEY_HERE'
}

orders = requests.get('https://paper-api.alpaca.markets/v2/orders?status=open', headers=headers).json()
print('Open orders count:', len(orders))
for o in orders:
    print(f"{o['side'].upper()} {o['qty']} {o['symbol']} -> Status: {o['status']}")

clock = requests.get('https://paper-api.alpaca.markets/v2/clock', headers=headers).json()
print('Market is open now:', clock['is_open'])
print('Next market open:', clock['next_open'])
