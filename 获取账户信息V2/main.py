import os
from dotenv import load_dotenv
from binance import Client
from spot import get_spot_positions
from decimal import Decimal
from futures import get_futures_positions
from portfolio import get_portfolio_balance
from printer import print_full_portfolio

def load_config() -> tuple:
    load_dotenv()
    return (os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"))
def main():
    API_KEY, SECRET_KEY = load_config()
    try:
        client = Client(API_KEY, SECRET_KEY)

        # 获取数据
        spot_data = get_spot_positions(client)
        futures_data = get_futures_positions(API_KEY, SECRET_KEY)
        portfolio_usdt = get_portfolio_balance(API_KEY, SECRET_KEY) or Decimal(0)

        # 打印分开的表格
        print_full_portfolio(
            spot_data=spot_data,
            futures_data=futures_data,
            portfolio_usdt=portfolio_usdt
        )

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()