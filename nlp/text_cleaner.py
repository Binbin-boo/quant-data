"""
第二层：文本清洗器（Bronze → Silver）
=======================================
将 SEC 下载的原始 HTML/TXT 文件清洗为纯净的可分析文本。
重点提取最有情绪分析价值的"MD&A"（管理层讨论与分析）章节。

学习目标：
  - 掌握 BeautifulSoup 解析 HTML 的核心技巧
  - 学习金融文档的结构（MD&A 章节的位置和特征）
  - 理解 NLP 预处理：去噪、分句、过滤
  - 学习正则表达式在文本清洗中的应用

运行：
  python nlp/text_cleaner.py
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString

logger = logging.getLogger("text_cleaner")

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_filings"
CLEAN_DIR = BASE_DIR / "clean_text"
CLEAN_DIR.mkdir(exist_ok=True)


# ============================================================
# 第一步：HTML → 纯文本
#
# 学习要点 - BeautifulSoup 的核心操作：
#   soup.get_text()  → 提取所有文本（包括脚本、样式，噪音大）
#   soup.find()      → 找第一个匹配的标签
#   soup.find_all()  → 找所有匹配的标签
#   tag.decompose()  → 从 DOM 树中删除该标签（及其子节点）
#   tag.get_text(separator=" ") → 提取该标签内的文本
#
# 注意 - SEC 文件的特殊性：
#   SEC 的 HTML 非常"古老"，用了大量已废弃的 HTML 标签
#   （<FONT>、<TABLE> 做布局），不遵循现代语义化 HTML。
#   所以我们用"暴力清洗"策略：先删除所有已知噪音标签，
#   再提取剩余文本。
# ============================================================

# 这些标签对 NLP 没有任何价值，直接删除
NOISE_TAGS = [
    'script', 'style', 'meta', 'link', 'head',
    'table',   # SEC 大量用 table 做排版，但表格数据不适合 NLP
    'img', 'svg', 'figure',
    'ix:nonfraction', 'ix:nonnumeric',  # XBRL 内联标记
]


def html_to_clean_text(html_content: str) -> str:
    """
    将 HTML 内容转换为清洁的纯文本。

    学习要点 - 为什么不直接用 re.sub('<.*?>', '', html)？
      正则表达式无法正确处理嵌套标签、属性值中含有 > 的情况。
      BeautifulSoup 用真正的 HTML 解析器，语义更准确。
      原则：用专业工具做专业事，正则只用于简单的字符串模式。
    """
    # lxml 解析器速度最快；如果没装 lxml，退回用 html.parser
    try:
        soup = BeautifulSoup(html_content, 'lxml')
    except Exception:
        soup = BeautifulSoup(html_content, 'html.parser')

    # 删除所有噪音标签
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 提取文本，段落间用换行分隔
    text = soup.get_text(separator='\n', strip=True)

    # 进一步清理：合并多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 删除只含数字/特殊符号的行（财务表格残留）
    lines = text.split('\n')
    cleaned_lines = [
        line for line in lines
        if len(line.strip()) > 20              # 过短的行通常是碎片
        and not re.match(r'^[\d\s\$\%\,\.\-\(\)]+$', line.strip())  # 纯数字行
    ]

    return '\n'.join(cleaned_lines)


# ============================================================
# 第二步：提取 MD&A 章节
#
# 学习要点 - 为什么只要 MD&A 而不是全文？
#   一份 10-K 动辄 200-400 页，其中：
#   - 财务报表：数字表格，NLP 无法分析
#   - 法律条款：标准化文本，年年相似，情绪信号很弱
#   - 风险因素 (Item 1A)：重要！但很多是固定的免责声明
#   - MD&A (Item 7)：最有价值！管理层用"人话"解释业绩，
#     主观性强，情绪信号最浓。
#
# 提取策略：基于 SEC 的标准章节标题做正则匹配
# ============================================================

# MD&A 章节的各种可能标题格式（SEC 没有严格统一格式）
MDA_PATTERNS = [
    r"ITEM\s+7\.?\s+MANAGEMENT.?S\s+DISCUSSION",
    r"Item\s+7\.?\s+Management.?s\s+Discussion",
    r"MANAGEMENT.S DISCUSSION AND ANALYSIS",
]

# MD&A 结束的标志：下一个章节的标题
MDA_END_PATTERNS = [
    r"ITEM\s+7A",
    r"Item\s+7A",
    r"QUANTITATIVE AND QUALITATIVE DISCLOSURES",
]


def extract_mda_section(full_text: str) -> Optional[str]:
    """
    从 10-K 全文中定位并提取 MD&A 章节。

    学习要点 - 基于锚点的文本切割（Anchor-based Extraction）：
      我们用"章节开始标题"作为起点锚，
      用"下一章节标题"作为终点锚，
      切割出中间的内容。

      这比训练一个分类器来定位章节要简单得多，
      且对于格式相对固定的 SEC 文件非常有效。

    注意 - re.IGNORECASE 和 re.DOTALL 的区别：
      IGNORECASE：匹配时忽略大小写
      DOTALL：让 "." 匹配包括换行符在内的所有字符（默认 . 不匹配 \n）
    """
    # 找到 MD&A 开始位置
    start_pos = None
    for pattern in MDA_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        logger.warning("未找到 MD&A 章节标题，尝试返回全文的中间部分")
        # 降级策略：返回文档的 40%-70% 部分（经验上 MD&A 在这个区间）
        total_len = len(full_text)
        return full_text[int(total_len * 0.4): int(total_len * 0.7)]

    # 找到 MD&A 结束位置（下一章节的开始）
    end_pos = len(full_text)
    search_text = full_text[start_pos + 100:]  # 跳过标题本身，从后续内容搜索
    for pattern in MDA_END_PATTERNS:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            end_pos = start_pos + 100 + match.start()
            break

    mda_text = full_text[start_pos:end_pos]
    logger.info(f"  MD&A 章节提取成功：{len(mda_text):,} 字符（原文 {len(full_text):,} 字符的 "
                f"{len(mda_text)/len(full_text)*100:.1f}%）")
    return mda_text


# ============================================================
# 第三步：句子切分（Sentence Segmentation）
#
# 学习要点 - NLP 预处理的基本单元：句子 vs 段落
#   FinBERT 的输入有长度限制（512 tokens ≈ 300-400 个英文单词）
#   如果直接把整个 MD&A 扔进去，会被截断，丢失大量信息。
#   正确做法：以句子为单位分析情绪，再对全章节做加权平均。
#
#   为什么不用 NLTK 的 sent_tokenize？
#   对于金融文本，NLTK 对缩写（"e.g."、"vs."）的处理不好，
#   会产生很多假断句。我们用简单规则代替，够用就好。
# ============================================================

def split_into_sentences(text: str, min_length: int = 30) -> list[str]:
    """
    将文本分割为句子列表。

    Args:
        text:       输入文本
        min_length: 最短句子长度（过滤掉碎片）

    学习要点 - 为什么过滤短句？
      MD&A 里大量的章节标题、页眉页脚会被分割成短句：
        "Overview"、"2023"、"Table of Contents"
      这些对情绪分析毫无价值，还会稀释有效样本。
    """
    # 按常见句子结束符切割
    # (?<=[.!?])    →  lookbehind：前面必须是句号/感叹号/问号
    # \s+           →  后面必须有空白符（防止切割 "3.5%" 这类数字）
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # 过滤太短的句子和含有过多数字的句子（表格残留）
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < min_length:
            continue
        # 如果句子里数字占比超过 40%，跳过（这是财务表格的特征）
        digit_ratio = sum(c.isdigit() for c in sent) / len(sent)
        if digit_ratio > 0.4:
            continue
        filtered.append(sent)

    return filtered


# ============================================================
# 第四步：整合为完整的清洗流水线
# ============================================================

def process_filing(filing_path: Path, ticker: str, filing_date: str) -> Optional[dict]:
    """
    处理单个 SEC 文件，输出结构化的清洁文本数据。

    Returns:
        dict 包含：
          - ticker:       股票代码
          - filing_date:  申报日期
          - form_type:    文件类型（10-K/10-Q）
          - full_text:    完整清洁文本
          - mda_text:     MD&A 章节文本
          - sentences:    分句后的列表
          - char_count:   字符数（用于质量评估）
    """
    logger.info(f"[{ticker}] 处理文件: {filing_path.name}")

    try:
        # 读取文件，尝试多种编码（SEC 文件有时用 latin-1 或 utf-8）
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                raw_content = filing_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"[{ticker}] 无法解码文件: {filing_path}")
            return None

        # HTML 还是纯文本？
        if '<html' in raw_content.lower() or '<body' in raw_content.lower():
            clean_text = html_to_clean_text(raw_content)
        else:
            # 纯文本文件：只需基本清理
            clean_text = re.sub(r'\n{3,}', '\n\n', raw_content)

        if len(clean_text) < 1000:
            logger.warning(f"[{ticker}] 清洗后文本过短（{len(clean_text)} 字符），文件可能有问题")
            return None

        # 提取 MD&A
        mda_text = extract_mda_section(clean_text)

        # 分句
        sentences = split_into_sentences(mda_text or clean_text)

        # 判断 form_type（从路径推断）
        form_type = "10-K" if "10-K" in str(filing_path) else "10-Q"

        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "form_type": form_type,
            "source_file": str(filing_path.relative_to(BASE_DIR)),
            "char_count": len(clean_text),
            "mda_char_count": len(mda_text) if mda_text else 0,
            "sentence_count": len(sentences),
            "mda_text": mda_text,
            "sentences": sentences,
        }

    except Exception as e:
        logger.error(f"[{ticker}] 文件处理失败: {e}", exc_info=True)
        return None


def process_all_filings(tickers: list[str]) -> list[dict]:
    """
    批量处理所有已下载的 SEC 文件。
    清洗结果保存为 JSON 到 clean_text/ 目录。
    """
    all_results = []

    for ticker in tickers:
        ticker_raw_dir = RAW_DIR / "sec-edgar-filings" / ticker
        if not ticker_raw_dir.exists():
            logger.warning(f"[{ticker}] 未找到原始文件，请先运行 sec_downloader.py")
            continue

        # 找到所有文件（v5 下载的是 full-submission.txt，兼容旧版 .htm）
        html_files = list(ticker_raw_dir.glob("**/*.htm")) + \
                     list(ticker_raw_dir.glob("**/*.html")) + \
                     list(ticker_raw_dir.glob("**/full-submission.txt"))

        logger.info(f"[{ticker}] 发现 {len(html_files)} 个文件，开始处理...")

        ticker_results = []
        for html_file in html_files:
            # 从目录名推断申报日期（sec-edgar-downloader 用 accession number 命名）
            # 格式：AAPL/10-K/0000320193-24-000123/primary-document.htm
            # 这里用文件修改时间作为近似日期
            import time as time_module
            mtime = html_file.stat().st_mtime
            from datetime import datetime
            filing_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

            result = process_filing(html_file, ticker, filing_date)
            if result:
                ticker_results.append(result)

        # 保存到 clean_text/ 目录
        if ticker_results:
            output_path = CLEAN_DIR / f"{ticker}_clean_filings.json"
            # 保存时不包含 sentences 列表（太大），单独保存句子
            summary_results = [{k: v for k, v in r.items() if k != 'sentences'}
                               for r in ticker_results]
            output_path.write_text(
                json.dumps(summary_results, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            # 单独保存句子文件（供 FinBERT 分析用）
            sentences_path = CLEAN_DIR / f"{ticker}_sentences.json"
            sentences_data = [
                {"ticker": r["ticker"], "filing_date": r["filing_date"],
                 "form_type": r["form_type"], "sentences": r["sentences"]}
                for r in ticker_results
            ]
            sentences_path.write_text(
                json.dumps(sentences_data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            logger.info(f"[{ticker}] 清洗完成，保存至 {output_path}")
            all_results.extend(ticker_results)

    return all_results


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'NVDA']
    results = process_all_filings(tickers)

    print(f"\n清洗完成，共处理 {len(results)} 份文件")
    for r in results:
        print(f"  [{r['ticker']}] {r['form_type']} {r['filing_date']} "
              f"- MD&A {r['mda_char_count']:,} 字符，{r['sentence_count']} 句")
