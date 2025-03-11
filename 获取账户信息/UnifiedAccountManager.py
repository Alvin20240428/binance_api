from BinanceBaseManager import BinanceBaseManager
import time


class UnifiedAccountManager(BinanceBaseManager):
    """统一账户管理"""

    def _get_base_url(self) -> str:
        return "https://papi.binance.com"

    def get_um_positions(self) -> list:
        """获取合约持仓"""
        response = self._signed_request("GET", "/papi/v1/um/positionRisk")
        # 验证响应结构：若返回的是字典，需提取列表字段
        if isinstance(response, dict):
            return response.get("positions", [])  # 提取 positions 列表
        elif isinstance(response, list):
            return response
        else:
            raise ValueError("API 返回数据格式异常")

    def get_cm_positions(self) -> list:
        """获取币本位合约持仓"""
        return self._signed_request(
            "GET", "/papi/v1/um/positionRisk"
        )["positions"]

    def get_balance_summary(self) -> dict:
        """获取统一账户资产总览"""
        return self._signed_request("GET", "/papi/v1/balance")