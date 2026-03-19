"""
数据验证模块（Data Validation）
================================
在写入数据库/文件之前，通过一系列断言检查数据质量。
这是生产系统的"防御性编程"实践。

学习要点：
  - 断言驱动的数据验证 vs 异常驱动的错误处理
  - 什么是数据契约（Data Contract）
  - 如何发现"静默错误"（Silent Corruption）

运行方式：
  python validate_data.py
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("validator")

DATA_DIR = Path("market_data")


class DataValidationError(Exception):
    """数据验证失败时抛出，区别于程序 Bug 导致的 Exception。"""
    pass


def validate_ohlcv_integrity(df: pd.DataFrame, ticker: str) -> None:
    """
    验证 OHLCV 数据的金融逻辑完整性。

    学习要点 - 什么是 OHLCV 的金融约束？
      1. High >= Low（最高价 >= 最低价）
      2. High >= Open 且 High >= Close
      3. Low  <= Open 且 Low  <= Close
      4. 所有价格 > 0（负价格只在原油期货历史上出现过一次）
      5. 成交量 >= 0

    为什么这些验证很重要？
      如果 High < Low，你的 K 线图会画反，
      部分策略会产生错误信号（如基于 High-Low 的波动率计算）。
    """
    errors = []

    # 检查价格非负
    for col in ['open', 'high', 'low', 'close', 'adj_close']:
        if (df[col] <= 0).any():
            bad_dates = df.index[df[col] <= 0].tolist()
            errors.append(f"{col} 列存在非正价格，日期: {bad_dates[:3]}")

    # 检查 High >= Low
    if (df['high'] < df['low']).any():
        bad_count = (df['high'] < df['low']).sum()
        errors.append(f"{bad_count} 行的 High < Low（K线倒置）")

    # 检查 High >= Open, Close
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        errors.append("存在 High < Open 或 High < Close 的行")

    # 检查 Low <= Open, Close
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        errors.append("存在 Low > Open 或 Low > Close 的行")

    # 检查成交量非负
    if (df['volume'] < 0).any():
        errors.append("成交量存在负值")

    if errors:
        for err in errors:
            logger.error(f"[{ticker}] 数据完整性错误: {err}")
        raise DataValidationError(f"[{ticker}] OHLCV 完整性验证失败: {errors}")

    logger.info(f"[{ticker}] OHLCV 完整性验证通过")


def validate_no_future_leakage(df: pd.DataFrame, ticker: str) -> None:
    """
    验证数据中没有未来日期（Look-ahead Bias 的一种来源）。

    学习要点 - Look-ahead Bias（未来函数偏差）：
      这是量化策略回测中最常见、最致命的错误。
      如果你的训练数据包含"测试期间"才能知道的信息，
      你的策略回测结果就是虚假的（看起来很好，实盘一塌糊涂）。

      这里的验证只是基础：确保数据集中没有超出 end_date 的数据。
      更深层的 Look-ahead Bias（如用未来数据标准化）需要在策略层防范。
    """
    today = pd.Timestamp.today().normalize()
    future_mask = df.index > today

    if future_mask.any():
        future_dates = df.index[future_mask].tolist()
        logger.warning(
            f"[{ticker}] 数据中存在未来日期（可能是 API 返回了预估数据）: {future_dates}"
        )
        # 这里选择 warning 而不是 error，因为某些 API 会返回当天未收盘数据


def validate_return_distribution(df: pd.DataFrame, ticker: str) -> None:
    """
    验证收益率分布是否在合理范围内。

    学习要点 - 收益率的"正常范围"：
      单日收益率超过 ±50% 的情况极其罕见（几乎只有 IPO 第一天或濒临退市股票）。
      对于 AAPL/MSFT/NVDA 这样的大市值股票，单日 ±20% 就已经非常极端。
      如果你看到 ±100%，几乎可以断定是数据错误。

      Kurtosis（峰度）：股票收益率的尾部比正态分布更厚（Fat Tails）。
      如果 Kurtosis 异常大（> 50），说明有极端离群值没被清洗掉。
    """
    returns = df['daily_return'].dropna()

    # 检查极端收益率
    extreme_threshold = 0.5  # 单日超过 50% 视为可疑
    extreme_mask = returns.abs() > extreme_threshold

    if extreme_mask.any():
        extreme_returns = returns[extreme_mask]
        logger.warning(
            f"[{ticker}] 发现 {len(extreme_returns)} 个单日收益率超过 ±{extreme_threshold*100:.0f}%：\n"
            f"{extreme_returns.to_string()}"
        )

    # 检查峰度
    kurtosis = returns.kurtosis()
    if kurtosis > 50:
        logger.warning(
            f"[{ticker}] 收益率峰度异常高（{kurtosis:.2f}），可能存在未清洗的离群值"
        )
    else:
        logger.info(f"[{ticker}] 收益率峰度: {kurtosis:.2f}（正常范围：5-20）")


def run_all_validations(ticker: str) -> bool:
    """
    对已保存的 Parquet 文件运行所有验证。

    Returns:
        True 表示全部通过，False 表示有 warning（但未抛出 error）
    """
    file_path = DATA_DIR / f"{ticker}_historical.parquet"

    if not file_path.exists():
        logger.error(f"[{ticker}] 文件不存在: {file_path}，请先运行 data_pipeline.py")
        return False

    df = pd.read_parquet(file_path)
    logger.info(f"[{ticker}] 开始数据验证，共 {len(df)} 行")

    try:
        validate_ohlcv_integrity(df, ticker)
        validate_no_future_leakage(df, ticker)
        validate_return_distribution(df, ticker)
        logger.info(f"[{ticker}] 全部验证通过 ✓")
        return True

    except DataValidationError as e:
        logger.error(f"[{ticker}] 验证失败: {e}")
        return False


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'NVDA']
    all_passed = all(run_all_validations(t) for t in tickers)

    if all_passed:
        print("\n所有数据验证通过，可以进行下一步分析。")
    else:
        print("\n部分数据验证失败，请检查日志文件 logs/pipeline.log")
