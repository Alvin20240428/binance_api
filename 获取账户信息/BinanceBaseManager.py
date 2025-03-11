import requests
import hmac
import hashlib
import time
from tenacity import retry, stop_after_attempt, wait_exponential
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
        # 手动构建查询字符串（包含必要编码）
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        # 生成 HMAC-SHA256 签名
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _encode_value(self, value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return requests.utils.quote(str(value))

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

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def make_request():

            response = requests.request(
                method,
                url,
                headers={"X-MBX-APIKEY": self.api_key},
                params=params
            )
            response.raise_for_status()
            return response.json()

        return make_request()
