import sys
import os
import csv
import pprint
import pandas as pd
import pickle
from datetime import datetime, timezone,timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
import MyTools as mt
from tmp_lib import get_start_end_time
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor





# 配置常量
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10
RETRY_WAIT_BASE = 2
BAR_TIME_INTERVAL = "1h"
FUNDING_RATE_PERIODS = 1000
BASE_URL = "https://api.binance.com"
FUTURE_URL = "https://fapi.binance.com"
FAILED_SYMBOLS_FILE = "failed_symbols.log"
GLOBAL_SAVE_FILENAME = "../历史数据/saved_data5.pkl"
DEBUG_SAVE_FILENAME = "../debug_saved_data.pkl"


# 定义重试装饰器
def http_retry_decorator():
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_WAIT_BASE, max=30),
        retry=(
                retry_if_exception_type(Timeout) |
                retry_if_exception_type(ConnectionError) |
                retry_if_exception_type(HTTPError)
        ),
        reraise=True
    )

class RequestSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.failed_symbols = []  # 用于记录失败交易对

    def log_failed_symbol(self, symbol, error, retry_count):
        """记录失败交易对信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.failed_symbols.append({
            "symbol": symbol,
            "error": str(error),
            "retries": retry_count,
            "timestamp": timestamp
        })

    @http_retry_decorator()
    def get_with_retry(self, url, params=None):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # 检查响应状态码
            return response
        except (Timeout, ConnectionError, HTTPError) as e:
            # 这里可以添加记录日志的逻辑，如调用log_failed_symbol方法
            raise

def is_in_time_range(dt):
    """修正后的时间窗口验证函数"""
    dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    hour = dt.hour
    minute = dt.minute
    valid_windows = [
        (23, 55, 59), (0, 0, 5),
        (7, 55, 59), (8, 0, 5),
        (15, 55, 59), (16, 0, 5)
    ]
    return any(
        (hour == target_hour) and (min_start <= minute <= min_end)
        for target_hour, min_start, min_end in valid_windows
    )


def get_current_hour_timestamp():
    now_utc = datetime.now(timezone.utc)
    now_beijing = now_utc + timedelta(hours=8)
    current_hour = now_beijing.replace(minute=0, second=0, microsecond=0)
    return int(current_hour.timestamp() * 1000)


def save_hourly_timestamp(timestamp):
    """保存当前小时的时间戳到文件"""
    with open("../历史数据/latest_hourly_timestamp.txt", "w") as f:
        f.write(str(timestamp))


def load_hourly_timestamp():
    """加载本地保存的时间戳"""
    try:
        with open("../历史数据/latest_hourly_timestamp.txt", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None


def is_data_expired():
    """检查是否需要重新获取数据"""
    saved_timestamp = load_hourly_timestamp()
    current_hour_timestamp = get_current_hour_timestamp()

    # 如果本地无记录或当前时间超过保存的小时，则需更新
    return (saved_timestamp is None) or (current_hour_timestamp > saved_timestamp)


def get_common_symbols():
    BASE_URL = "https://api.binance.com"
    FUTURE_URL = "https://fapi.binance.com"
    # Get spot symbols
    spot_info = requests.get(f"{BASE_URL}/api/v3/exchangeInfo").json()
    spot_symbols = {
        s["symbol"].replace("-", "").replace("/", "")
        for s in spot_info["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    }

    url = f"{FUTURE_URL}/fapi/v1/premiumIndex"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['dt'] = pd.to_datetime(df['nextFundingTime'], unit='ms')
        df['is_in_time_range'] = df['dt'].apply(is_in_time_range)
        df_8h = df[df['is_in_time_range']]
        future_symbols = {
            symbol.replace("-", "").replace("/", "")
            for symbol in df_8h['symbol'].tolist()
        }
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 期货交易对请求失败，状态码: {response.status_code}")
        future_symbols = set()

    common_symbols = sorted(list(spot_symbols & future_symbols))
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 共找到 {len(common_symbols)} 个有效交易对")

    return common_symbols

def fetch_klines(symbol, is_future=False):
    url = f"{FUTURE_URL}/fapi/v1/klines" if is_future else f"{BASE_URL}/api/v3/klines"
    start_time, end_time = get_start_end_time()

    params = {"symbol": symbol, "interval": BAR_TIME_INTERVAL, "startTime": start_time,
              "endTime": end_time,"limit":FUNDING_RATE_PERIODS}
    requester = RequestSession()
    try:
        response = requester.get_with_retry(url, params=params)
        data = response.json()
        if not data:
            print(f"警告: {symbol} 返回空数据")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("h")
        df["timestamp"] = df["open_time"].astype(np.int64) // 1000  # 毫秒转秒
        return df[["timestamp", "close", "volume"]]

    except Exception as e:
        print(f"获取 {symbol} K线失败: {str(e)}")
        return pd.DataFrame()

def fetch_raw_funding(symbol):
    url = f"{FUTURE_URL}/fapi/v1/fundingRate"
    start_time, end_time = get_start_end_time()
    params = {
        "symbol": symbol,
        "limit": FUNDING_RATE_PERIODS,
        "startTime": start_time,
        "endTime": end_time,

    }

    requester = RequestSession()
    try:
        response = requester.get_with_retry(url, params=params)
        data = response.json()
        if not data:
            print(f"警告: {symbol} 资金费率返回空数据")
            return pd.Series()

        df = pd.DataFrame(data)
        if 'fundingRate' not in df.columns:
            print(f"错误: {symbol} 数据缺少资金费率列")
            return pd.Series()

        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["timestamp"] = df["fundingTime"].astype(np.int64) // 10 ** 9
        df = df.set_index("timestamp")["fundingRate"].astype(float)
        return df
    except Exception as e:
        print(f"{symbol} 资金费率获取失败: {str(e)}")
        return pd.Series()

def calculate_future_vwap_60min(symbol):
    try:
        future_df = fetch_klines(symbol, is_future=True)
        if future_df.empty:
            return pd.Series()
        # 优化部分：直接筛选非零交易量数据
        non_zero_volume_df = future_df.query("volume != 0")
        if non_zero_volume_df.empty:
            return pd.Series()
        # 正确计算VWAP
        numerator = (non_zero_volume_df["close"] * non_zero_volume_df["volume"]).sum()
        denominator = non_zero_volume_df["volume"].sum()
        if denominator == 0:
            return pd.Series()
        vwap_value = numerator / denominator
        non_zero_volume_df["vwap"] = vwap_value
        return non_zero_volume_df.set_index("timestamp")["vwap"]
    except (ValueError, TypeError) as e:
        print(f"{symbol} VWAP计算失败: {str(e)}")
        return pd.Series()

def calculate_spot_vwap_60min(symbol):
    try:
        spot_df = fetch_klines(symbol, is_future=False)
        if spot_df.empty:
            return pd.Series()
        # 优化部分：直接筛选非零交易量数据
        non_zero_volume_df = spot_df.query("volume != 0")
        if non_zero_volume_df.empty:
            return pd.Series()
        # 正确计算VWAP
        numerator = (non_zero_volume_df["close"] * non_zero_volume_df["volume"]).sum()
        denominator = non_zero_volume_df["volume"].sum()
        if denominator == 0:
            return pd.Series()
        vwap_value = numerator / denominator
        non_zero_volume_df["vwap"] = vwap_value
        return non_zero_volume_df.set_index("timestamp")["vwap"]
    except (ValueError, TypeError) as e:
        print(f"{symbol} 现货VWAP计算失败: {str(e)}")
        return pd.Series()

def calculate_basis(symbol, debug=False):
    try:
        spot_df = fetch_klines(symbol, is_future=False)
        future_df = fetch_klines(symbol, is_future=True)

        spot_df = spot_df.set_index("timestamp")
        future_df = future_df.set_index("timestamp")
        spot_df.index = spot_df.index.astype(str)
        future_df.index = future_df.index.astype(str)

        merged = pd.merge(
            spot_df,
            future_df,
            left_index=True,
            right_index=True,
            suffixes=("_spot", "_future"),
            how="inner"
        )

        if merged.empty:
            print(f"警告: {symbol} 时间戳未对齐，跳过")
            return pd.Series()

        basis_series = (merged["close_future"] / merged["close_spot"]) - 1
        basis_series.name = symbol
        if debug:
            print(f"[DEBUG] spot_df.index.dtype = {spot_df.index.dtype}")
            print(f"[DEBUG] future_df.index.dtype = {future_df.index.dtype}")
            print(f"[DEBUG] spot_df.index[:5] = {spot_df.index[:5].tolist()}")
            print(f"[DEBUG] future_df.index[:5] = {future_df.index[:5].tolist()}")
        return basis_series
    except Exception as e:
        print(f"{symbol} 基差计算失败: {str(e)}")
        return pd.Series()

def fetch_single_symbol_data(symbol):
    try:
        # 获取期货、现货、资金费率数据
        future_df = fetch_klines(symbol, is_future=True)
        spot_df = fetch_klines(symbol, is_future=False)
        funding_series = fetch_raw_funding(symbol)

        if future_df.empty or spot_df.empty:
            print(f"跳过数据缺失的品种: {symbol}")
            return None

        # 设置时间戳为索引
        future_df = future_df.set_index("timestamp")
        spot_df = spot_df.set_index("timestamp")

        # 合并期货和现货数据的时间戳
        merged = future_df.join(spot_df, how="inner", lsuffix="_future", rsuffix="_spot")
        if merged.empty:
            print(f"跳过时间戳未对齐的品种: {symbol}")
            return None

        # 合并资金费率数据时间戳
        if not funding_series.empty:
            merged = merged.join(funding_series.to_frame(name="funding"), how="left")

        return future_df, spot_df, funding_series
    except Exception as e:
        print(f"处理 {symbol} 时出现异常: {e}")
        return None


def build_bar_dict(common_symbols=None, debug=False):
    # 允许从外部传入交易对列表
    if common_symbols is None:
        common_symbols = get_common_symbols()

    if debug:
        common_symbols = common_symbols[:3]
        print(f"[DEBUG] 调试模式启用，处理品种: {common_symbols}")


    # 存储期货、现货和资金费率数据的列表
    future_dfs = []
    spot_dfs = []
    funding_series_list = []

    valid_symbols = []

    with ThreadPoolExecutor() as executor:
        # 并行执行数据获取任务
        results = list(executor.map(fetch_single_symbol_data, common_symbols))

    for symbol, result in zip(common_symbols, results):
        if result is None:
            continue

        future_df, spot_df, funding_series = result

        # 存储有效数据
        future_dfs.append(future_df)
        spot_dfs.append(spot_df)
        funding_series_list.append(funding_series)
        valid_symbols.append(symbol)

    if not valid_symbols:
        print("无有效数据")
        return None

    # 找到所有数据的时间戳交集
    all_timestamps = set(future_dfs[0].index)
    for df in future_dfs[1:]:
        all_timestamps &= set(df.index)

    all_timestamps = sorted(list(all_timestamps))

    # 用于存储各列数据的列表
    pvwap_60min_columns = []
    spot_pvwap_60min_columns = []
    sp_factor_columns = []
    funding_columns = []

    # 填充数据
    for i, symbol in enumerate(valid_symbols):
        future_df = future_dfs[i].copy()
        spot_df = spot_dfs[i].copy()
        funding_series = funding_series_list[i]

        # --- 期货VWAP ---
        future_vwap = (future_df["close"] * future_df["volume"]) / future_df["volume"]
        pvwap_60min_columns.append(future_vwap.reindex(all_timestamps).rename(symbol))

        # --- 现货VWAP ---
        spot_vwap = (spot_df["close"] * spot_df["volume"]) / spot_df["volume"]
        spot_pvwap_60min_columns.append(spot_vwap.reindex(all_timestamps).rename(symbol))

        # --- 基差 ---
        basis = (future_df["close"] / spot_df["close"]) - 1
        sp_factor_columns.append(basis.reindex(all_timestamps).rename(symbol))

        # --- 资金费率（先ffill再bfill） ---
        funding = funding_series.reindex(all_timestamps)
        funding_columns.append(funding.rename(symbol))

    # 使用 pd.concat 一次性合并所有列
    pvwap_60min_df = pd.concat(pvwap_60min_columns, axis=1)
    spot_pvwap_60min_df = pd.concat(spot_pvwap_60min_columns, axis=1)
    sp_factor_df = pd.concat(sp_factor_columns, axis=1)
    funding_df = pd.concat(funding_columns, axis=1)
    funding_df.index = pd.to_datetime(funding_df.index, unit='s')


    # 按每八个小时分组计算 ewma21
    selected_times = funding_df[funding_df.index.hour.isin([0, 8, 16])]
    # 检测 NaN 并报警
    nan_mask = selected_times.isna().any(axis=1)
    if nan_mask.any():
        nan_indices = selected_times.index[nan_mask].tolist()
        print(f"⚠️ 警告：在以下时间点检测到 NaN 值：{nan_indices}")

    # 前向填充 NaN（用前一个有效值填充）
    selected_times = selected_times.ffill()
    # ----------------------------------------

    selected_times_with_prefix = selected_times.copy()
    selected_times_with_prefix.columns = ["FUT_" + col for col in selected_times.columns]

    # 计算 EWMA
    ewma21_df = selected_times_with_prefix.ewm(halflife=21, adjust=False).mean()

    def convert_timestamp_index(df):
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime("%Y-%m-%d %H%M")
        else:
            df.index = pd.to_datetime(df.index, unit='s').strftime("%Y-%m-%d %H%M")
        return df

    pvwap_60min_df = convert_timestamp_index(pvwap_60min_df)
    spot_pvwap_60min_df = convert_timestamp_index(spot_pvwap_60min_df)
    sp_factor_df = convert_timestamp_index(sp_factor_df)
    funding_df = convert_timestamp_index(funding_df)
    ewma21_df = convert_timestamp_index(ewma21_df)


    bar_dict = {
        "research_prod": {
            "basic::pvwap_60min": pvwap_60min_df,
            "basic::spot_pvwap_60min": spot_pvwap_60min_df
        },
        "indicators": {
            "sp_factor": sp_factor_df,
            "funding": funding_df,
            "ewma21":ewma21_df
        }
    }

    return bar_dict

'''
def check_symbol_exists(target_symbol="AUDIOUSDT"):
    """检查指定交易对是否存在于有效列表中"""
    # 统一格式化交易对名称（与get_common_symbols()中的处理一致）
    formatted_symbol = target_symbol.replace("-", "").replace("/", "").upper()

    # 获取当前有效交易对列表
    common_symbols = get_common_symbols()

    # 创建包含原始符号和格式化符号的集合
    formatted_symbols = {s.upper() for s in common_symbols}

    exists = formatted_symbol in formatted_symbols
    print(f"交易对 {target_symbol} {'存在' if exists else '不存在'}于有效列表")
    return exists
'''
def save_data(bar_dict, debug=False, file_path=None):
    latest_timestamp = bar_dict["research_prod"]["basic::pvwap_60min"].index[-1]
    with open("../历史数据/latest_timestamp.txt", "w") as f:
        f.write(str(latest_timestamp))

    if file_path is None:
        file_path = DEBUG_SAVE_FILENAME if debug else GLOBAL_SAVE_FILENAME

    with open(file_path, 'wb') as f:
        pickle.dump(bar_dict, f)
    print(f"[{'DEBUG' if debug else 'INFO'}] 数据已保存到 {file_path}")

def load_data(file_path):
    """加载数据并返回数据和最新时间戳"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # 从数据中提取时间戳
        latest_timestamp = data["research_prod"]["basic::pvwap_60min"].index[-1]
        return data, latest_timestamp
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
        return None, None
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {str(e)}")
        return None, None

def main():
    start_time = time.time()
    #params = mt.get_parsers()
    #debug = params.debug
    debug = False
    USE_SAVED_DATA = True
    SAVED_DATA_PATH = GLOBAL_SAVE_FILENAME if not debug else DEBUG_SAVE_FILENAME

    # ------------------ 第一步：加载或初始化交易对列表 ------------------
    file_path = '../um_cols.csv'
    try:
        # 从 CSV 文件加载交易对列表
        if os.path.getsize(file_path) == 0:
            raise FileNotFoundError(f"文件 {file_path} 为空")
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            common_symbols = next(reader)  # 读取第一行
        print(f"从 um_cols.csv 加载 {len(common_symbols)} 个交易对")
    except FileNotFoundError as e:
        # 如果文件不存在或为空，获取新列表并保存
        print(f"错误: {e}")
        common_symbols = get_common_symbols()
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(common_symbols)
        print(f"生成新交易对列表并保存到 um_cols.csv")
    except Exception as e:
        print(f"发生未知错误: {e}")
        common_symbols = []

    # ------------------ 第二步：检查本地数据和时间戳 ------------------
    bar_dict = None
    saved_timestamp = None

    # 尝试加载本地数据
    if USE_SAVED_DATA and os.path.exists(SAVED_DATA_PATH):
        bar_dict, saved_timestamp = load_data(SAVED_DATA_PATH)
        if bar_dict is not None:
            print(f"成功加载本地数据，最新时间戳: {saved_timestamp}")

    # ------------------ 第三步：判断是否需要重新构建数据 ------------------
    # 获取当前时间戳范围
    if USE_SAVED_DATA and not is_data_expired():
        # 直接使用本地数据
        bar_dict = load_data(SAVED_DATA_PATH)
        print("时间戳未过期，直接使用本地数据")
    else:
        # 重新构建数据
        print("时间戳已过期或本地数据无效，重新构建数据")
        bar_dict = build_bar_dict(common_symbols=common_symbols, debug=debug)
        save_data(bar_dict, file_path=GLOBAL_SAVE_FILENAME)
        # 保存当前小时的时间戳
        save_hourly_timestamp(get_current_hour_timestamp())

    # ------------------ 第四步：确保 bar_dict 有效 ------------------
    if bar_dict is None:
        print("错误: 无法加载或构建数据")
        return

    # ------------------ 后续处理（保持不变） ------------------
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行完成，总耗时: {elapsed_time} 秒")
    pprint.pprint(bar_dict)


if __name__ == '__main__':
    main()