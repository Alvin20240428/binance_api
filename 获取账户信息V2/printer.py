from tabulate import tabulate
from decimal import Decimal


def print_full_portfolio(spot_data, futures_data, portfolio_usdt: Decimal):
    """打印包含所有账户的完整报表"""
    spot_positions, spot_total = spot_data
    futures_positions, futures_total, futures_usdt = futures_data

    # 现货表格
    print("\n=== 现货账户 ===")
    spot_table = [
        [p["symbol"], f"{p['amount']:.4f}", f"${p['value']:.2f}"]
        for p in spot_positions
    ]
    spot_table.append(["持仓总价值", "",f"{spot_total:.2f}", ""])
    print(tabulate(spot_table, headers=["Symbol", "Amount", "Value"], tablefmt="pretty"))

    # 合约账户表格
    print("\n=== 合约账户 ===")
    futures_table = [
        [p["symbol"], f"{p['amount']:.4f}", f"${p['value']:.2f}"]
        for p in futures_positions
    ]
    futures_table.extend([
        ["持仓总价值", "", f"${futures_total:.2f}"],
        ["可用保证金", "",f"${futures_usdt:.2f}", ""]
    ])
    print(tabulate(futures_table, headers=["Symbol", "Amount", "Value"], tablefmt="pretty"))

    # 组合保证金账户
    if portfolio_usdt > 0:
        print("\n=== 组合保证金账户 ===")
        portfolio_table = [
            ["保证金余额", f"${portfolio_usdt:.2f}", ""]
        ]
        print(tabulate(portfolio_table, headers=["Type", "Balance", ""], tablefmt="pretty"))
    else:
        print("\n[提示] 组合保证金账户数据获取失败或余额为0")

    #计算账户总余额
    total = spot_total + portfolio_usdt
    print(f"\n总资产估值: ${total:.2f}")