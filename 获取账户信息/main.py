from SpotAccountManager import SpotAccountManager
from UnifiedAccountManager import UnifiedAccountManager
from PortfolioFormatter import PortfolioFormatter
import os
from dotenv import load_dotenv


class BinancePortfolio:
    """全账户资产管理入口"""

    def __init__(self, spot_key: str, spot_secret: str,
                 unified_key: str, unified_secret: str):
        self.spot = SpotAccountManager(spot_key, spot_secret)
        self.unified = UnifiedAccountManager(unified_key, unified_secret)

    def get_full_portfolio(self) -> str:
        """获取完整资产报告"""
        spot_data = self.spot.get_spot_positions()
        unified_data = self.unified.get_um_positions()

        formatted_spot = PortfolioFormatter.format_spot_assets(spot_data)
        filtered_unified = [p for p in unified_data if float(p.get('positionAmt', 0)) != 0]
        formatted_unified = PortfolioFormatter.format_unified_positions(filtered_unified)

        # 合并结果
        return f"{formatted_spot}\n{formatted_unified}"

def main():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    spot = SpotAccountManager(api_key=api_key, api_secret=api_secret)
    unified = UnifiedAccountManager(api_key=api_key, api_secret=api_secret)
    try:
        # 获取并展示现货资产
        spot_assets = spot.get_spot_positions()
        print(PortfolioFormatter.format_spot_assets(spot_assets))

        # 获取并展示统一账户持仓
        unified_positions = unified.get_um_positions()
        print(PortfolioFormatter.format_unified_positions(unified_positions))

    except Exception as e:
        print(f"操作失败: {str(e)}")
# 使用示例
if __name__ == "__main__":
    load_dotenv()
    main()