from decimal import Decimal
from binance.client import Client
from typing import List, Dict


def get_spot_positions(client: Client) -> (List[Dict], Decimal):
    """获取现货持仓及总价值"""
    account = client.get_account()
    balances = [b for b in account["balances"] if float(b["free"]) + float(b["locked"]) > 0]

    positions = []
    total_value = Decimal(0)

    for balance in balances:
        asset = balance["asset"]
        amount = Decimal(balance["free"]) + Decimal(balance["locked"])

        if asset == "USDT":
            value = amount
            symbol = "USDT"
        else:
            symbol = f"{asset}USDT"
            try:
                ticker = client.get_symbol_ticker(symbol=symbol)
                price = Decimal(ticker["price"])
                value = amount * price
            except:
                continue

        positions.append({
            "symbol": symbol,
            "type": "Spot",
            "amount": amount,
            "value": value
        })
        total_value += value

    return positions, total_value