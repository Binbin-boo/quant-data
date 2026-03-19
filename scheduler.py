"""
自动调度模块（Scheduler）
==========================
实现每日自动更新数据，只拉取增量（最近 5 天），
通过 Upsert 合并进历史文件，避免全量重新下载。

学习要点：
  - 增量更新（Incremental Update）：只取最新几天的数据
  - 幂等性保证：重复运行不会产生重复数据（由 upsert_parquet 保障）
  - 使用 schedule 库做本地调度（生产环境替换为 crontab 或 Airflow）

运行方式：
  python scheduler.py          # 启动后台调度（持续运行）
  python scheduler.py --once   # 立刻运行一次更新，然后退出
"""

import sys
import logging
from datetime import datetime, timedelta

import schedule
import time

from data_pipeline import run_pipeline

logger = logging.getLogger("scheduler")

TARGET_STOCKS = ['AAPL', 'MSFT', 'NVDA']

# 增量更新：只拉最近 7 天（覆盖周末 + 保险起见多拉几天）
INCREMENTAL_DAYS = 7


def daily_update_job():
    """
    每日增量更新任务。

    学习要点 - 为什么只拉 7 天而不是全量？
      全量拉取（几年数据）每次都要花几十秒，
      增量拉取只需要几秒，且通过 Upsert 保证数据完整性。
      这是生产系统的标准做法。

    学习要点 - 为什么用 timedelta 计算日期而不是 hardcode？
      hardcode 日期的脚本只能用一次，
      动态计算才能让脚本永远有效。
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=INCREMENTAL_DAYS)).strftime('%Y-%m-%d')

    logger.info(f"[Scheduler] 开始每日增量更新，范围: {start_date} → {end_date}")

    run_pipeline(
        tickers=TARGET_STOCKS,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"[Scheduler] 增量更新完成，下次运行: {schedule.next_run()}")


if __name__ == "__main__":
    # --once 参数：立刻运行一次，方便手动触发或 crontab 调用
    if "--once" in sys.argv:
        logger.info("单次运行模式")
        daily_update_job()
        sys.exit(0)

    # 常驻调度模式：
    # 美东时间 16:30（美股收盘后 30 分钟）= UTC 21:30
    # 如果你在中国标准时间（CST = UTC+8），就是次日 05:30
    # 根据你的系统时区调整这里的时间
    schedule.every().monday.at("16:30").do(daily_update_job)
    schedule.every().tuesday.at("16:30").do(daily_update_job)
    schedule.every().wednesday.at("16:30").do(daily_update_job)
    schedule.every().thursday.at("16:30").do(daily_update_job)
    schedule.every().friday.at("16:30").do(daily_update_job)

    logger.info("调度器已启动，等待下次运行时间...")
    logger.info(f"下次运行: {schedule.next_run()}")

    # 主循环：每分钟检查一次是否到了运行时间
    while True:
        schedule.run_pending()
        time.sleep(60)
