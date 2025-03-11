from datetime import datetime
from typing import List, Dict


class PortfolioFormatter:
    """专业级资产格式化工具"""

    @staticmethod
    def format_spot_assets(spot_data: list) -> str:
        """现货资产表格（兼容空数据）"""
        output = ["\n🔵 现 货 资 产"]
        if not spot_data:
            output.append("当前无现货持仓")
            return '\n'.join(output)

        output.append("{:<8} | {:<14} | {:<14}".format("资产", "可用余额", "冻结金额"))
        output.append("-" * 45)
        for asset in spot_data:
            try:
                asset_name = asset.get("asset", "N/A")
                free = float(asset.get("free", 0))
                locked = float(asset.get("locked", 0))
                output.append("{:<8} | {:>14.4f} | {:>14.4f}".format(
                    asset_name, free, locked
                ))
            except Exception as e:
                output.append(f"数据错误: {str(e)}")
        return '\n'.join(output)

    @staticmethod
    def format_unified_positions(unified_data: list) -> str:
        """统一账户持仓格式化（增强健壮性）"""
        output = ["\n🟢 统 一 账 户"]

        # 验证输入数据格式
        if not isinstance(unified_data, list):
            return "⚠️ 持仓数据格式错误: 非列表类型"

        # 过滤有效持仓并验证元素类型
        valid_positions = []
        for pos in unified_data:
            if not isinstance(pos, dict):
                continue  # 跳过非字典元素
            try:
                position_amt = float(pos.get("positionAmt", 0))
                if position_amt != 0:
                    valid_positions.append(pos)
            except ValueError:
                continue  # 跳过数值转换失败的数据

        # 无持仓时的提示
        if not valid_positions:
            output.append("当前无有效合约持仓")
            return '\n'.join(output)

        # 生成表格
        try:
            rows = []
            for pos in valid_positions:
                symbol = pos.get("symbol", "未知")
                position_amt = float(pos.get("positionAmt", 0))
                unrealized_profit = float(pos.get("unRealizedProfit", 0))
                liquidation_price = float(pos.get("liquidationPrice", 0))

                rows.append((
                    symbol,
                    "空" if position_amt < 0 else "多",
                    f"{abs(position_amt):.4f}",
                    f"{unrealized_profit:+.2f}",
                    f"{liquidation_price:.4f}"
                ))

            output.append(PortfolioFormatter._generate_table(
                headers=["产品", "方向", "数量", "盈亏", "强平价格"],
                rows=rows
            ))
        except Exception as e:
            output.append(f"表格生成失败: {str(e)}")

        return '\n'.join(output)

    @staticmethod
    def _generate_table(headers: list, rows: list) -> str:
        """生成专业表格"""
        col_width = [10, 10, 12, 12, 15]
        sep_line = "+" + "+".join(["-" * w for w in col_width]) + "+"
        table = [sep_line]
        table.append("|" + "|".join(f"{h:^{col_width[i]}}" for i, h in enumerate(headers)) + "|")
        table.append(sep_line)
        for row in rows:
            table.append("|" + "|".join(f"{str(c):^{col_width[i]}}" for i, c in enumerate(row)) + "|")
            table.append(sep_line)
        return '\n'.join(table)


    @staticmethod
    def _get_asset_value(asset: str, amount: float) -> float:
        """资产估值计算（需实现价格获取逻辑）"""
        # 此处可接入实时行情接口
        return amount * 1.0  # 示例估值