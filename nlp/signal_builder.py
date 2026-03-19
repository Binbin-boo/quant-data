"""
第四层：信号整合器（Sentiment + Price → Trading Signal）
=========================================================
将 FinBERT 输出的情绪分数与历史价格数据按日期对齐，
构建最终的"黄金层"特征数据集，供回测引擎使用。

学习目标：
  - 理解"日期对齐"（Date Alignment）问题及其金融含义
  - 学习事件驱动数据的前向传播（Event Forward-Fill）
  - 掌握 Pandas 的 merge_asof 做时间序列近似合并
  - 理解 Publication Bias（发布偏差）和如何避免它

运行：
  python nlp/signal_builder.py
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("signal_builder")

BASE_DIR = Path(__file__).parent
SIGNAL_DIR = BASE_DIR / "sentiment_signals"
PRICE_DIR = BASE_DIR.parent / "market_data"
OUTPUT_DIR = BASE_DIR / "sentiment_signals"


# ============================================================
# 核心概念：日期对齐与发布偏差
#
# 问题：财报在"申报日期"发布，但市场在"下一个交易日"才能反应。
#
# 时间线示例（AAPL 2023 Q4 财报）：
#   2024-01-25（盘后）：Apple 发布 Q4 财报 + 电话会议
#   2024-01-26（开盘）：市场消化信息，股价做出反应
#
# 错误做法（引入 Look-ahead Bias）：
#   把 2024-01-25 的情绪分数对应到 2024-01-25 的价格 ❌
#   实际上当天收盘前，这份财报还没发布！
#
# 正确做法：
#   把 2024-01-25 的情绪分数对应到 2024-01-26（下一个交易日）✅
#   这才是你"实际上能用这个信息交易"的最早时间点。
#
# 这就是"Publication Bias 发布偏差"的典型案例。
# ============================================================

def load_sentiment_signals(ticker: str) -> pd.DataFrame:
    """加载 FinBERT 输出的情绪信号。"""
    signal_path = SIGNAL_DIR / f"{ticker}_sentiment.json"

    if not signal_path.exists():
        logger.warning(f"[{ticker}] 未找到情绪信号文件，请先运行 finbert_scorer.py")
        return pd.DataFrame()

    data = json.loads(signal_path.read_text(encoding='utf-8'))
    df = pd.DataFrame(data)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df = df.sort_values('filing_date').reset_index(drop=True)
    return df


def load_price_data(ticker: str) -> pd.DataFrame:
    """加载项目 1 生成的价格 Parquet 文件。"""
    price_path = PRICE_DIR / f"{ticker}_historical.parquet"

    if not price_path.exists():
        logger.error(f"[{ticker}] 未找到价格数据，请先运行 data_pipeline.py")
        return pd.DataFrame()

    df = pd.read_parquet(price_path)
    df.index = pd.to_datetime(df.index)
    return df


def align_sentiment_to_price(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    将情绪信号与价格数据按日期对齐。

    学习要点 - pd.merge_asof 的工作原理：
      这是处理"不同频率时间序列对齐"的神器。
      普通 merge 需要 key 完全匹配，
      merge_asof 允许 key 近似匹配（取最近的过去值）。

      参数 direction='backward' 表示：
        对于价格表中的每一个交易日 T，
        在情绪表中找最近的 filing_date <= T 的记录。
        这自然实现了"只用过去已发布的信息"，避免 Look-ahead Bias。

    注意 - 申报日期 vs 有效日期（T+1 处理）：
      财报通常在盘后发布，所以我们把有效日期设为申报日期的 T+1。
      这样 merge_asof 找到的永远是"你昨天收盘后才能看到的"信息。
    """
    if sentiment_df.empty:
        logger.warning(f"[{ticker}] 情绪数据为空，跳过对齐")
        return price_df

    # T+1 处理：情绪信号在下一个交易日才生效
    sentiment_aligned = sentiment_df.copy()
    sentiment_aligned['effective_date'] = sentiment_aligned['filing_date'] + pd.Timedelta(days=1)

    # 对齐：price_df 的 index 是交易日，sentiment 的 effective_date 是信号生效日
    price_reset = price_df.reset_index()
    price_reset.columns.values[0] = 'date'

    sentiment_for_merge = sentiment_aligned[[
        'effective_date', 'weighted_score', 'positive_ratio',
        'negative_ratio', 'sentiment_label', 'sentiment_magnitude'
    ]].rename(columns={'effective_date': 'date'})

    # merge_asof：对每个交易日，找最近的（不超过该日的）情绪信号
    merged = pd.merge_asof(
        price_reset.sort_values('date'),
        sentiment_for_merge.sort_values('date'),
        on='date',
        direction='backward',  # 只用过去的信号
    )

    merged = merged.set_index('date')
    merged.index.name = price_df.index.name  # 保持原始 index 名称

    filled_count = merged['weighted_score'].notna().sum()
    total_count = len(merged)
    logger.info(
        f"[{ticker}] 情绪对齐完成：{total_count} 个交易日中，"
        f"{filled_count} 个有情绪信号覆盖（{filled_count/total_count:.1%}）"
    )

    return merged


def add_derived_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    在对齐后的数据上增加派生特征，供策略使用。

    学习要点 - 特征工程的几个思路：
      1. 情绪动量（Sentiment Momentum）：
         当前情绪分数 - 上一期情绪分数
         捕捉"情绪是否在改善"比"情绪的绝对水平"更重要

      2. 情绪-价格背离（Sentiment-Price Divergence）：
         情绪乐观但股价下跌 → 可能的买入信号（市场过于悲观）
         情绪悲观但股价上涨 → 可能的卖出信号（市场过于乐观）

      3. 滚动情绪均值（Rolling Sentiment Mean）：
         平滑掉单次财报的噪音，捕捉更稳定的情绪趋势
    """
    df = df.copy()

    if 'weighted_score' not in df.columns or df['weighted_score'].isna().all():
        return df

    # 1. 情绪动量（当前分数 - 上一期分数）
    df['sentiment_momentum'] = df['weighted_score'].diff()

    # 2. 情绪滚动均值（最近 60 个交易日，约 3 个月）
    df['sentiment_rolling_60d'] = df['weighted_score'].rolling(window=60, min_periods=1).mean()

    # 3. 情绪 Z-score（相对于过去 252 天的标准化分数）
    rolling_mean = df['weighted_score'].rolling(window=252, min_periods=30).mean()
    rolling_std  = df['weighted_score'].rolling(window=252, min_periods=30).std()
    df['sentiment_zscore'] = (df['weighted_score'] - rolling_mean) / (rolling_std + 1e-8)

    # 4. 情绪-价格背离信号（需要 daily_return 列）
    if 'daily_return' in df.columns:
        # 简单背离：情绪正但收益负，或情绪负但收益正
        price_direction = np.sign(df['daily_return'])
        sentiment_direction = np.sign(df['weighted_score'])
        df['sentiment_price_divergence'] = (price_direction != sentiment_direction).astype(int)

    return df


def build_final_dataset(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    构建最终的黄金层数据集：价格 + 情绪信号 + 派生特征。
    """
    results = {}

    for ticker in tickers:
        logger.info(f"[{ticker}] 开始构建黄金层数据集...")

        price_df = load_price_data(ticker)
        if price_df.empty:
            continue

        sentiment_df = load_sentiment_signals(ticker)

        # 对齐
        aligned_df = align_sentiment_to_price(price_df, sentiment_df, ticker)

        # 派生特征
        final_df = add_derived_features(aligned_df, ticker)

        # 保存为 Parquet（黄金层）
        output_path = OUTPUT_DIR / f"{ticker}_gold.parquet"
        final_df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        logger.info(f"[{ticker}] 黄金层数据已保存至 {output_path}")

        results[ticker] = final_df

    return results


def print_signal_preview(df: pd.DataFrame, ticker: str) -> None:
    """打印信号数据预览，展示情绪+价格的对齐效果。"""
    preview_cols = ['adj_close', 'daily_return', 'weighted_score',
                    'sentiment_label', 'sentiment_momentum']
    available_cols = [c for c in preview_cols if c in df.columns]

    # 只展示有情绪信号的行
    has_signal = df['weighted_score'].notna() if 'weighted_score' in df.columns else pd.Series(False, index=df.index)

    print(f"\n{'='*70}")
    print(f"  {ticker} - 情绪+价格对齐预览（有情绪信号的日期）")
    print(f"{'='*70}")
    if has_signal.any():
        print(df.loc[has_signal, available_cols].tail(10).to_string())
    else:
        print("（当前无情绪信号，仅显示价格数据最后 5 行）")
        print(df[available_cols[:2]].tail(5).to_string())


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'NVDA']
    final_datasets = build_final_dataset(tickers)

    for ticker, df in final_datasets.items():
        print_signal_preview(df, ticker)

    print("\n黄金层数据集构建完成！")
    print(f"文件保存在: {OUTPUT_DIR}")
    print("下一步：将这些数据输入回测引擎（项目 2/3）")
