from BinanceBaseManager import BinanceBaseManager


class UnifiedAccountManager(BinanceBaseManager):
    """统一账户管理"""

    def _get_base_url(self) -> str:
        return "https://papi.binance.com"

    def get_um_positions(self) -> list:
        """获取合约持仓"""
        response = self._signed_request("GET", "/papi/v1/um/positionRisk")
        if isinstance(response, dict):
            return response.get("positions", [])
        elif isinstance(response, list):
            return response
        else:
            raise ValueError("API 返回数据格式异常")


    def get_balance_summary(self) -> dict:
        """获取统一账户资产总览"""
        return self._signed_request("GET", "/papi/v1/balance")