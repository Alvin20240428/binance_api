import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode
from abc import ABC, abstractmethod


class BinanceBaseManager(ABC):
    """交易所接口基类"""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self._get_base_url()

    @abstractmethod
    def _get_base_url(self) -> str:
        """由子类指定API域名"""
        pass

    def _generate_signature(self, params: dict) -> str:
        """安全增强的签名生成方法"""

        # 手动构建查询字符串（包含必要编码）
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])

        # 生成 HMAC-SHA256 签名
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()


        # 生成签名


    def _encode_value(self, value) -> str:
        """安全值编码方法"""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value).replace(" ", "%20")

    def _signed_request(self, method: str, endpoint: str, params=None) -> dict:
        """统一请求方法"""
        if params is None:
            params = {}

        params.update({
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000
        })
        params["signature"] = self._generate_signature(params)

        url = f"{self.base_url}{endpoint}"
        response = requests.request(
            method,
            url,
            headers={"X-MBX-APIKEY": self.api_key},
            params=params
        )
        response.raise_for_status()
        return response.json()
