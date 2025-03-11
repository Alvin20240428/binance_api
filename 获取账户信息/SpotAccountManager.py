from binance import Client
from binance.exceptions import BinanceAPIException
from BinanceBaseManager import BinanceBaseManager


class SpotAccountManager(BinanceBaseManager):
    """基于python-binance库的现货账户管理"""

    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        self.client = Client(api_key, api_secret)

    def _get_base_url(self) -> str:
        """指定现货API的域名"""
        return "https://api.binance.com"

    def get_balances(self) -> dict:
        """获取现货账户资产"""
        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            self._handle_api_error(e)

    def get_spot_positions(self) -> list:
        """获取有效持仓"""
        try:
            balances = self.get_balances().get("balances", [])
            return [
                b for b in balances
                if float(b.get("free", 0)) + float(b.get("locked", 0)) > 0
            ]
        except Exception as e:
            raise RuntimeError(f"过滤持仓失败: {str(e)}")

    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET') -> dict:
        """创建现货订单（当前未调用，预留交易接口）"""
        return self.client.create_order(
            symbol=symbol,
            side=side.upper(),
            type=order_type,
            quantity=quantity
        )

    def _handle_api_error(self, e: BinanceAPIException):
        """统一错误处理"""
        raise RuntimeError(f"Binance API Error: {e.status_code} - {e.message}")