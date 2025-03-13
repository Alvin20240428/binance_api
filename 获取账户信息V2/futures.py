import time
import hmac
import hashlib
import urllib.parse
import requests
from decimal import Decimal
from typing import List, Dict


def generate_signature(secret: str, data: dict) -> str:
    query_string = urllib.parse.urlencode(data)
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def get_futures_account(api_key: str, secret_key: str) -> dict:
    """获取统一账户资产信息"""
    BASE_URL = "https://papi.binance.com"
    ENDPOINT = "/papi/v1/um/account"

    timestamp = int(time.time() * 1000)
    params = {"timestamp": timestamp}
    params["signature"] = generate_signature(secret_key, params)

    response = requests.get(
        f"{BASE_URL}{ENDPOINT}",
        headers={"X-MBX-APIKEY": api_key},
        params=params
    )
    response.raise_for_status()
    return response.json()


def get_futures_positions(api_key: str, secret_key: str) -> (List[Dict], Decimal, Decimal):
    """获取合约持仓及USDT余额"""
    account_data = get_futures_account(api_key, secret_key)

    # 获取USDT余额
    usdt_balance = Decimal(0)
    for asset in account_data["assets"]:
        if asset["asset"] == "USDT":
            usdt_balance = Decimal(asset["crossWalletBalance"])
            break
    print(usdt_balance)

    # 处理持仓
    positions = []
    total_value = Decimal(0)

    for position in account_data["positions"]:
        if float(position["positionAmt"]) == 0:
            continue

        symbol = position["symbol"]
        amount = Decimal(position["positionAmt"])
        mark_price = get_mark_price(symbol)
        value = abs(amount) * mark_price

        positions.append({
            "symbol": position["symbol"],
            "amount": amount,
            "mark_price": mark_price,
            "value": value
        })
        total_value += value

    return positions, total_value, usdt_balance



def get_mark_price(symbol: str) -> Decimal:
    """获取合约标记价格"""
    BASE_URL = "https://fapi.binance.com"
    ENDPOINT = "/fapi/v1/ticker/price"

    response = requests.get(f"{BASE_URL}{ENDPOINT}?symbol={symbol}")
    response.raise_for_status()
    return Decimal(response.json()["price"])