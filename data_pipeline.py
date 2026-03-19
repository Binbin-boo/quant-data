"""
Production-Grade Financial Data Pipeline
=========================================
项目 1：构建高可用数据管道

学习目标：
- 理解金融数据的特殊性（复权、拆股、节假日）
- 掌握时间序列数据清洗的正确方法
- 实现幂等性写入（Upsert 逻辑）
- 使用 logging 替代 print，构建可观测系统
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

from storage import get_storage, StorageBackend

# ============================================================
# 第一步：配置 Logging（放弃 print，拥抱可观测性）
#
# 学习要点：
#   - logging.INFO  → 正常流程日志（数据获取成功等）
#   - logging.WARNING → 数据质量问题（发现离群值等）
#   - logging.ERROR  → 流程中断（API 失败、文件写入失败等）
#
# 注意：
#   - 日志同时写入文件和控制台（双 Handler）
#   - %(asctime)s 让每条日志都带时间戳，排查问题时救命
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("data_pipeline")


# ============================================================
# 第二步：数据获取层（Data Fetching Layer）
#
# 学习要点：
#   - auto_adjust=False → 拿到原始 OHLCV + Adj Close 两组数据
#     如果 auto_adjust=True，yfinance 会把 Close 直接替换为复权价，
#     你会丢失原始价格，无法做价格展示或部分策略回测
#   - 为什么用 Ticker.history() 而不是 yf.download()？
#     前者返回的 DataFrame index 是 DatetimeIndex，
#     后者在多 ticker 时返回 MultiIndex，处理更复杂
# ============================================================

def fetch_raw_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Yahoo Finance 获取原始 OHLCV 数据。

    Args:
        ticker_symbol: 股票代码，如 'AAPL'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date:   结束日期，格式 'YYYY-MM-DD'

    Returns:
        原始 DataFrame，列包含 Open/High/Low/Close/Adj Close/Volume

    Raises:
        ValueError: 当 API 返回空数据时
        ConnectionError: 当网络请求失败时
    """
    logger.info(f"[{ticker_symbol}] 开始获取数据，范围：{start_date} → {end_date}")

    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=False,   # 保留原始 Close 和 Adj Close
        )
    except Exception as e:
        logger.error(f"[{ticker_symbol}] API 请求失败: {e}")
        raise ConnectionError(f"无法获取 {ticker_symbol} 的数据") from e

    if df.empty:
        logger.error(f"[{ticker_symbol}] API 返回空数据，请检查 ticker 代码或日期范围")
        raise ValueError(f"No data fetched for {ticker_symbol}")

    # yfinance 有时返回带时区的 index，统一去掉时区信息方便后续处理
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info(f"[{ticker_symbol}] 获取成功，共 {len(df)} 行，"
                f"日期范围 {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ============================================================
# 第三步：数据清洗层（Data Cleaning Layer）
#
# 这是整个 Pipeline 的核心，分为 4 个子步骤
# ============================================================

def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    学习要点：
      yfinance 返回的列名有时带 MultiIndex 或多余列（Dividends, Stock Splits）
      我们明确选择需要的列，并统一命名规范（snake_case）
    """
    needed = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    # 只取需要的列，如果某列不存在会抛出 KeyError，提前发现数据源变化
    df = df[needed].copy()
    df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    return df


def handle_missing_values(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    处理缺失值的金融正确方法。

    学习要点（关键！）：
      1. 价格用 ffill（前向填充）→ 用上一个已知价格填补
         ❌ 绝对不能用线性插值：插值用了未来的价格，引入 Look-ahead Bias
         ❌ 绝对不能用 bfill（后向填充）：同样引入未来信息
      2. 成交量缺失填 0 → 那天没交易，成交量就是 0
      3. 先 ffill 再检查头部 NaN → 如果第一行就是 NaN，ffill 无能为力

    注意：
      节假日通常不会出现在 DataFrame 里（index 直接跳过），
      ffill 主要处理 API 偶尔返回的空格行
    """
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']

    # 记录清洗前的缺失情况
    missing_before = df[price_cols].isna().sum().sum()
    if missing_before > 0:
        logger.warning(f"[{ticker}] 价格列发现 {missing_before} 个缺失值，执行前向填充")

    df[price_cols] = df[price_cols].ffill()
    df['volume'] = df['volume'].fillna(0)

    # 如果第一行就是 NaN（序列起始点就缺失），ffill 无法修复，直接删除
    remaining_na = df[price_cols].isna().sum().sum()
    if remaining_na > 0:
        logger.warning(f"[{ticker}] ffill 后仍有 {remaining_na} 个缺失值（序列起始位置），删除这些行")
        df = df.dropna(subset=price_cols)

    return df


def detect_and_fix_outliers(df: pd.DataFrame, ticker: str, threshold: float = 0.20) -> pd.DataFrame:
    """
    使用移动窗口 Z-score 检测价格离群值。

    学习要点：
      为什么不直接用全局 Z-score？
        因为股票价格是非平稳时间序列（Nonstationary），
        全局均值和标准差毫无意义。NVDA 2020 年的均值和 2024 年差了 10 倍。
        所以改用"日收益率"来检测离群点，再用移动窗口捕捉局部异常。

      MAD vs Z-score：
        MAD（中位数绝对偏差）对极端值更鲁棒，
        Z-score 在正态分布假设下更简单。
        这里用基于日收益率的简单阈值方法（单日涨跌幅 > threshold 视为可疑）

    注意：
      真实的离群值检测要复杂得多，需要结合成交量、
      跨市场数据比对等。这里是 MVP 版本。
    """
    # 计算临时日收益率用于离群值检测
    temp_returns = df['adj_close'].pct_change().abs()

    # 找出超过阈值的索引（默认单日涨跌幅超过 20%）
    outlier_mask = temp_returns > threshold
    outlier_count = outlier_mask.sum()

    if outlier_count > 0:
        outlier_dates = df.index[outlier_mask].tolist()
        logger.warning(
            f"[{ticker}] 检测到 {outlier_count} 个可疑离群点（单日涨跌幅 > {threshold*100:.0f}%），"
            f"日期: {[str(d.date()) for d in outlier_dates[:5]]}"
            f"{'...' if outlier_count > 5 else ''}"
        )
        # 将离群价格标记为 NaN，然后再次用 ffill 修复
        price_cols = ['open', 'high', 'low', 'close', 'adj_close']
        df.loc[outlier_mask, price_cols] = np.nan
        df[price_cols] = df[price_cols].ffill()
        logger.info(f"[{ticker}] 离群值已用前向填充修复")

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算金融特征（Financial Feature Engineering）。

    学习要点：
      1. Daily Return：最基础的金融信号，后续所有风险计算都基于此
      2. Log Return（对数收益率）：
         公式：ln(P_t / P_{t-1})
         优点：在时间上可以累加（Simple Return 不能），方便计算多日累积收益
         数学性质更好（近似正态分布）
      3. 成交量 Z-score（相对于 20 日均量的偏差）：
         捕捉成交量异常放大，通常是重要事件（财报、新闻）的信号

    注意：
      这里故意保留 NaN（pct_change 产生的第一行）
      让调用方决定如何处理，不在此处 dropna，保持函数职责单一
    """
    df = df.copy()

    # 简单日收益率（用于 PnL 计算）
    df['daily_return'] = df['adj_close'].pct_change()

    # 对数收益率（用于统计分析、风险模型）
    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

    # 20 日滚动波动率（年化）
    df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)

    # 成交量相对于 20 日均量的比值（Volume Ratio）
    df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()

    return df


def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    清洗流水线的总入口，按顺序调用各个清洗步骤。

    设计原则：
      每个清洗函数只做一件事（单一职责），
      方便独立测试和单独替换某一步的实现。
    """
    logger.info(f"[{ticker}] 开始数据清洗流程")

    df = select_and_rename_columns(df)
    df = handle_missing_values(df, ticker)
    df = detect_and_fix_outliers(df, ticker)
    df = compute_features(df)

    # 删除因 pct_change / rolling 产生的首行 NaN
    df = df.dropna(subset=['daily_return'])

    logger.info(f"[{ticker}] 清洗完成，最终数据 {len(df)} 行")
    return df


# ============================================================
# 第四步：存储层（Storage Layer）—— 幂等性 Upsert
#
# 学习要点：
#   幂等性（Idempotency）：无论运行多少次，结果相同
#   实现方法：先读取已有数据，合并后去重，再写回
#   去重依据：index（日期），新数据覆盖旧数据（以最新拉取为准）
#
# 注意：
#   Parquet 不支持真正的 Upsert（不像数据库），
#   所以我们手动实现"读-合并-去重-写"的逻辑
# ============================================================

DATA_DIR = Path("market_data")
DATA_DIR.mkdir(exist_ok=True)

# 存储 key 的命名规范：{目录}/{ticker}_historical.parquet
# 本地时对应文件路径，云端时对应 S3/R2 对象 key
def _storage_key(ticker: str) -> str:
    return f"market_data/{ticker}_historical.parquet"


def upsert_parquet(df: pd.DataFrame, ticker: str, storage: StorageBackend = None) -> str:
    """
    将新数据 Upsert 进已有的 Parquet 文件（本地或云端）。
    如果文件不存在则直接写入，如果存在则合并去重。

    学习要点 - 存储抽象的效果：
      本地开发：storage=LocalStorage → 读写 market_data/ 目录
      云端生产：storage=S3Storage   → 读写 R2/S3 Bucket
      调用方（run_pipeline）完全不感知差异。
    """
    if storage is None:
        storage = get_storage()

    key = _storage_key(ticker)

    if storage.exists(key):
        logger.info(f"[{ticker}] 检测到已有数据，执行 Upsert 合并")
        existing_df = storage.read_parquet(key)

        combined = pd.concat([existing_df, df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        rows_added = len(combined) - len(existing_df)
        logger.info(f"[{ticker}] Upsert 完成：新增 {rows_added} 行，总计 {len(combined)} 行")
        df_to_write = combined
    else:
        logger.info(f"[{ticker}] 首次写入")
        df_to_write = df

    storage.write_parquet(df_to_write, key)
    return key


# ============================================================
# 第五步：数据质量报告（Data Quality Report）
#
# 学习要点：
#   生产系统不只是"跑完不报错"就行，
#   还需要主动输出数据质量指标，方便监控和排查
# ============================================================

def print_quality_report(df: pd.DataFrame, ticker: str) -> None:
    """打印数据质量摘要报告。"""
    report = {
        "股票代码": ticker,
        "数据行数": len(df),
        "日期范围": f"{df.index[0].date()} → {df.index[-1].date()}",
        "缺失值总数": df.isna().sum().sum(),
        "平均日收益率": f"{df['daily_return'].mean():.4%}",
        "年化波动率": f"{df['daily_return'].std() * np.sqrt(252):.4%}",
        "最大单日跌幅": f"{df['daily_return'].min():.4%}",
        "最大单日涨幅": f"{df['daily_return'].max():.4%}",
        "Adj Close 最新": f"${df['adj_close'].iloc[-1]:.2f}",
    }

    logger.info(f"[{ticker}] 数据质量报告：")
    for key, value in report.items():
        logger.info(f"  {key}: {value}")


# ============================================================
# 主流程（Main Orchestrator）
# ============================================================

def run_pipeline(
    tickers: list[str],
    start_date: str,
    end_date: str,
    storage: StorageBackend = None,
) -> dict[str, pd.DataFrame]:
    """
    运行完整的数据管道。

    Args:
        storage: 存储后端实例。不传则自动根据环境变量选择。

    Returns:
        dict，key 为 ticker，value 为清洗后的 DataFrame
    """
    if storage is None:
        storage = get_storage()

    results = {}
    failed = []

    for ticker in tickers:
        try:
            # Fetch → Clean → Upsert（存储后端由 storage 决定）
            raw_df = fetch_raw_data(ticker, start_date, end_date)
            clean_df = clean_data(raw_df, ticker)
            upsert_parquet(clean_df, ticker, storage)
            print_quality_report(clean_df, ticker)
            results[ticker] = clean_df

        except (ValueError, ConnectionError) as e:
            logger.error(f"[{ticker}] Pipeline 失败: {e}")
            failed.append(ticker)
            # 注意：这里选择 continue 而不是 raise
            # 一支股票失败不应该中断其他股票的处理
            continue

    if failed:
        logger.error(f"以下股票处理失败: {failed}")
    else:
        logger.info(f"所有 {len(tickers)} 支股票处理完成")

    return results


if __name__ == "__main__":
    target_stocks = ['AAPL', 'MSFT', 'NVDA']

    data = run_pipeline(
        tickers=target_stocks,
        start_date="2021-01-01",
        end_date="2026-03-18",
    )

    # 简单验证：打印每支股票的最后 3 行
    for ticker, df in data.items():
        print(f"\n{'='*50}")
        print(f"  {ticker} - 最新数据预览")
        print(f"{'='*50}")
        print(df[['adj_close', 'daily_return', 'log_return', 'volatility_20d']].tail(3).to_string())
