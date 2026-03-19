"""
第一层：SEC EDGAR 文件下载器
==============================
从美国证监会官方数据库下载 10-K（年报）和 10-Q（季报）。

学习目标：
  - 理解 SEC EDGAR 数据库的结构和访问方式
  - 掌握 sec-edgar-downloader 库的使用
  - 理解文件系统的"青铜层/黄金层"分层架构
  - 学习 User-Agent 规范和 API 礼仪（Rate Limiting）

运行：
  python nlp/sec_downloader.py
"""

import logging
import time
from pathlib import Path
from datetime import datetime

from sec_edgar_downloader import Downloader

# ============================================================
# 路径配置
#
# 学习要点 - 数据湖（Data Lake）分层架构：
#   Bronze（青铜层）：原始数据，保持原样，不做任何修改
#                    就算下游处理出 bug，你还能从这里重新开始
#   Silver（白银层）：清洗后的结构化数据
#   Gold（黄金层）：最终可用于模型训练的特征数据
#
#   本项目的对应关系：
#   Bronze = raw_filings/   (HTML/TXT 原始文件)
#   Silver = clean_text/    (纯文本，去掉 HTML 标签)
#   Gold   = sentiment_signals/ (FinBERT 输出的情绪得分 + 价格对齐)
# ============================================================

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_filings"
RAW_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "logs" / "nlp_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sec_downloader")


# ============================================================
# SEC EDGAR 下载器初始化
#
# 学习要点 - User-Agent 规范：
#   SEC EDGAR 要求每个请求都携带合法的 User-Agent，格式为：
#   "公司名 联系邮箱"
#   如果你不填或者填假的，SEC 会封你的 IP。
#   这是一个"API 礼仪"（API Etiquette）的典型例子：
#   免费使用公共 API 的代价是遵守使用规范。
#
#   SEC 的限流规则：每秒不超过 10 个请求（我们远低于这个）
# ============================================================

def create_downloader() -> Downloader:
    """
    创建 SEC EDGAR 下载器实例。
    请替换为你自己的名字和邮箱（SEC 要求，不会发广告）。

    注意 - v5 API 变更：
      save_path 参数从构造函数移到了 .get() 的 download_folder 参数。
    """
    return Downloader(
        company_name="QuantResearch",
        email_address="your_email@example.com",  # 替换为你的邮箱（任意邮箱即可，SEC 不验证）
        download_folder=str(RAW_DIR),
    )


def download_filings(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 5,
    downloader: Downloader = None,
) -> int:
    """
    下载指定股票的 SEC 财报文件。

    Args:
        ticker:    股票代码，如 'AAPL'
        form_type: 文件类型，'10-K'（年报）或 '10-Q'（季报）
        limit:     最多下载几份（最新的 N 份）
        downloader: Downloader 实例（可复用）

    Returns:
        成功下载的文件数量

    学习要点 - 10-K vs 10-Q：
      10-K：年报，每年提交一次，包含完整财务信息 + 管理层讨论
            最重要的 NLP 分析对象（信息量最大，措辞最谨慎）
      10-Q：季报，每季度提交三次（第四季度合并进 10-K），
            时效性更强，适合追踪季度情绪变化
      8-K： 重大事件报告（如 CEO 离职、并购），实时触发，
            可作为"事件驱动策略"的信号源

    学习要点 - 为什么只下载最新 5 份而不是全部？
      情绪分析的时效性：2010 年的管理层措辞和现在的行业语境完全不同，
      用太老的数据训练/评分会引入噪音。
      实际上，5 份 10-K = 5 年数据，对于情绪趋势分析已经足够。
    """
    if downloader is None:
        downloader = create_downloader()

    logger.info(f"[{ticker}] 开始下载 {form_type}，最新 {limit} 份...")

    try:
        downloader.get(form_type, ticker, limit=limit)

        # 验证文件是否真的下载了
        # v5 在 download_folder 下多加了一层 sec-edgar-filings/ 子目录
        filing_dir = RAW_DIR / "sec-edgar-filings" / ticker / form_type
        if filing_dir.exists():
            downloaded = list(filing_dir.glob("**/*.htm")) + \
                         list(filing_dir.glob("**/*.html")) + \
                         list(filing_dir.glob("**/full-submission.txt"))
            count = len(downloaded)
            logger.info(f"[{ticker}] {form_type} 下载完成，共 {count} 个文件，路径: {filing_dir}")
            return count
        else:
            logger.warning(f"[{ticker}] 下载后未发现文件目录，可能没有 {form_type} 数据")
            return 0

    except Exception as e:
        logger.error(f"[{ticker}] 下载失败: {e}")
        return 0


def batch_download(
    tickers: list[str],
    form_types: list[str] = ["10-K", "10-Q"],
    limit_per_type: int = 5,
) -> dict:
    """
    批量下载多支股票的多种文件类型。

    学习要点 - Rate Limiting 礼貌爬取：
      即使 SEC 允许每秒 10 个请求，我们也在每支股票之间睡眠 2 秒。
      原因：
        1. 避免触发 IP 封锁（SEC 有时会对高频请求 ban IP）
        2. 减少服务器负担（公共资源，大家共享）
        3. 生产系统稳定性 > 速度，不要贪图快而触发不可预知的错误
    """
    dl = create_downloader()
    results = {}

    for ticker in tickers:
        results[ticker] = {}
        for form_type in form_types:
            count = download_filings(ticker, form_type, limit_per_type, dl)
            results[ticker][form_type] = count
            # 礼貌性等待，避免触发限流
            time.sleep(2)

    # 打印下载摘要
    logger.info("\n下载摘要：")
    for ticker, forms in results.items():
        for form, count in forms.items():
            logger.info(f"  {ticker} {form}: {count} 个文件")

    return results


def list_downloaded_filings(ticker: str) -> list[Path]:
    """
    列出已下载的文件清单。

    学习要点 - Path.glob() 的递归匹配：
      "**/*.htm" 中的 ** 表示匹配任意深度的子目录
      这是处理 SEC 文件目录结构的必要技巧，
      因为 sec-edgar-downloader 会创建多层嵌套目录：
      raw_filings/AAPL/10-K/0000320193-24-000123/primary-document.htm
    """
    ticker_dir = RAW_DIR / "sec-edgar-filings" / ticker
    if not ticker_dir.exists():
        logger.warning(f"[{ticker}] 尚未下载任何文件，请先运行 batch_download()")
        return []

    files = []
    for pattern in ["**/*.htm", "**/*.html", "**/full-submission.txt"]:
        files.extend(ticker_dir.glob(pattern))

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # 按修改时间排序
    return files


if __name__ == "__main__":
    target_stocks = ['AAPL', 'MSFT', 'NVDA']

    logger.info("="*60)
    logger.info("SEC EDGAR 批量下载开始")
    logger.info("="*60)

    results = batch_download(
        tickers=target_stocks,
        form_types=["10-K"],   # 先只下年报，节省时间
        limit_per_type=3,      # 每支股票最新 3 份
    )

    # 列出所有下载的文件
    for ticker in target_stocks:
        files = list_downloaded_filings(ticker)
        print(f"\n{ticker} 已下载文件：")
        for f in files[:5]:
            print(f"  {f.relative_to(RAW_DIR)}")
