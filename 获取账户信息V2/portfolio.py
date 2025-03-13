import time
import hmac
import hashlib
import urllib.parse
import requests
from decimal import Decimal
from typing import Tuple


def get_portfolio_balance(api_key: str, secret_key: str) -> Tuple[Decimal, Decimal]:
    """获取组合保证金账户的USDT余额"""
    BASE_URL = "https://papi.binance.com"
    ENDPOINT = "/papi/v1/account"

    timestamp = int(time.time() * 1000)
    params = {
        "timestamp": timestamp,
        "recvWindow": 5000
    }

    # 生成签名
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": api_key}

    try:
        response = requests.get(
            f"{BASE_URL}{ENDPOINT}",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        data = response.json()
        print(data)

        actual_equity = data.get("actualEquity")
        actual_equity_decimal = Decimal(actual_equity)



        return actual_equity_decimal

    except Exception as e:
        print(f"获取组合保证金数据失败: {str(e)}")
        return None