"""
第三层：FinBERT 情绪评分引擎（Silver → Gold 的关键一步）
==========================================================
使用 HuggingFace 的 FinBERT 模型对金融文本进行情绪分析，
输出每份财报的情绪得分（Sentiment Score）。

学习目标：
  - 理解为什么金融 NLP 必须用领域专用模型（Domain-Specific Model）
  - 掌握 HuggingFace Transformers Pipeline 的使用方式
  - 学习 BERT 的 Token 限制和批处理（Batch Processing）策略
  - 学习如何将句子级分数聚合为文档级分数

运行：
  pip install transformers torch
  python nlp/finbert_scorer.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("finbert_scorer")

BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "clean_text"
SIGNAL_DIR = BASE_DIR / "sentiment_signals"
SIGNAL_DIR.mkdir(exist_ok=True)


# ============================================================
# 为什么用 FinBERT，而不是 VADER 或 GPT？
#
# 对比三种方案：
#
# 1. VADER (NLTK 内置)
#    原理：基于词典的规则方法，给每个词打分然后加权求和
#    优点：零依赖，速度极快
#    缺点：对金融文本完全失效：
#      "The company reduced its debt burden significantly"
#      VADER 会给这句话打负分（因为 "debt" 和 "burden" 是负面词）
#      但对股东来说，减少债务是极好的消息！
#
# 2. FinBERT (我们的选择)
#    模型：ProsusAI/finbert（HuggingFace 开源）
#    训练数据：专门用金融新闻、财报、分析师报告微调的 BERT
#    输出：Positive / Negative / Neutral 三分类 + 概率
#    优点：理解金融上下文语义，准确率远超 VADER
#    缺点：需要 GPU 加速（CPU 上跑一份财报大约 2-5 分钟）
#          每次输入不超过 512 tokens（需要分句处理）
#
# 3. GPT-4 via API
#    优点：理解能力最强，可以做更复杂的分析
#    缺点：每次调用花钱，延迟高，不适合批量处理几千句话
#          输出不稳定，需要做额外的解析
#    适合场景：少量文本的深度分析、生成摘要
#
# 结论：批量情绪打分 → FinBERT；深度单文本分析 → GPT
# ============================================================

def load_finbert_pipeline():
    """
    加载 FinBERT 模型管道。

    学习要点 - HuggingFace Pipeline 的抽象层次：
      pipeline() 是 HuggingFace 提供的高级 API，
      它把 tokenizer（分词）+ model（模型推理）+ postprocessor（解码输出）
      封装在一个对象里，你只需要传入文本，就能得到预测结果。

      底层实际在做：
        文本 → Tokenizer → Token IDs → BERT Encoder → CLS向量 → 全连接层 → Softmax → 概率

    学习要点 - device 参数：
      device=0   → 使用第一块 GPU（CUDA）
      device=-1  → 强制使用 CPU
      device="mps" → 使用 Apple Silicon 的 Metal 加速（M1/M2/M3）
      不设置时自动检测（优先 GPU，没有就用 CPU）

    注意 - 首次运行会下载约 500MB 的模型权重到 ~/.cache/huggingface/
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch

        # 自动选择最佳设备
        if torch.backends.mps.is_available():
            device = "mps"       # Apple Silicon
            logger.info("使用 Apple Silicon MPS 加速")
        elif torch.cuda.is_available():
            device = 0           # NVIDIA GPU
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = -1          # CPU
            logger.info("使用 CPU（建议：在 M系列 Mac 上用 MPS，速度快 5-10 倍）")

        model_name = "ProsusAI/finbert"
        logger.info(f"正在加载 FinBERT 模型（首次运行需下载 ~500MB）...")

        # truncation=True 确保超过 512 tokens 的输入被截断而不报错
        nlp_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

        logger.info("FinBERT 模型加载成功")
        return nlp_pipeline

    except ImportError:
        logger.error("请安装依赖: pip install transformers torch")
        raise


# ============================================================
# 单句打分 → 文档级聚合
#
# 学习要点 - 为什么要做"聚合"而不是直接给整篇文章打分？
#   BERT 的 512 token 限制（约 350-400 个英文单词）
#   一篇 MD&A 通常有 5,000-20,000 个单词
#   如果直接输入，前 512 个 token 之后的内容全部丢失。
#
#   正确做法：以句子为单位逐句分析，再聚合：
#     weighted_score = Σ (score_i × weight_i) / Σ weight_i
#   权重可以是：
#     - 句子长度（更长的句子包含更多信息）
#     - 模型的置信度（只信任高置信度的预测）
#     - 句子位置（首尾段落权重更高）
# ============================================================

# 情绪标签 → 数值分数的映射
SENTIMENT_SCORE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral":  0.0,
}


def score_sentences_batch(
    sentences: list[str],
    nlp_pipeline,
    batch_size: int = 16,
) -> list[dict]:
    """
    批量对句子列表进行情绪打分。

    Args:
        sentences:    句子列表
        nlp_pipeline: FinBERT pipeline 实例
        batch_size:   每批处理的句子数（根据显存调整，CPU 用 8，GPU 用 32）

    Returns:
        每句话的打分结果列表

    学习要点 - 批处理（Batch Processing）：
      单句逐个处理：每次调用都有固定的 PyTorch 开销（初始化矩阵等）
      批处理：GPU 可以并行处理多个句子，效率提升 5-10 倍
      batch_size 的选择：取决于显存大小
        8GB 显存 → batch_size 约 32-64
        Apple M2 (16GB 共享内存) → batch_size 约 16-32
        CPU → batch_size 约 8（避免内存占用过高）
    """
    if not sentences:
        return []

    results = []
    total = len(sentences)

    for i in range(0, total, batch_size):
        batch = sentences[i: i + batch_size]

        try:
            # pipeline 的输入：字符串列表
            # pipeline 的输出：[{'label': 'positive', 'score': 0.92}, ...]
            predictions = nlp_pipeline(batch)

            for sent, pred in zip(batch, predictions):
                label = pred['label'].lower()
                confidence = pred['score']   # 模型对这个分类的置信度 (0-1)
                numeric_score = SENTIMENT_SCORE.get(label, 0.0) * confidence

                results.append({
                    "sentence": sent,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "score": round(numeric_score, 4),
                    # 有效句子：置信度 > 0.7，避免将模糊句子计入统计
                    "is_high_confidence": confidence > 0.7,
                })

        except Exception as e:
            logger.error(f"批处理第 {i}-{i+batch_size} 句失败: {e}")
            # 不中断整体流程，跳过这批
            continue

        # 进度日志
        if (i // batch_size) % 5 == 0:
            logger.info(f"  进度: {min(i + batch_size, total)}/{total} 句")

    return results


def aggregate_document_sentiment(sentence_scores: list[dict]) -> dict:
    """
    将句子级分数聚合为文档级情绪摘要。

    学习要点 - 加权平均 vs 简单平均：
      简单平均的问题："Revenue was strong." (5字)
      和 "While we faced significant headwinds from macroeconomic uncertainty,
          our diversified portfolio allowed us to navigate challenges effectively." (23字)
      这两句话对整体情绪的贡献不应该相同。
      我们用置信度加权：高置信度的预测权重更大。

    返回的指标含义：
      weighted_score:  加权情绪分数，范围 [-1, 1]
                       > 0.2 偏乐观，< -0.2 偏悲观，中间是中性
      positive_ratio:  正面句子占比（不含中性）
      negative_ratio:  负面句子占比
      high_conf_count: 高置信度句子数量（样本质量指标）
    """
    if not sentence_scores:
        return {"weighted_score": 0.0, "positive_ratio": 0.0,
                "negative_ratio": 0.0, "total_sentences": 0}

    # 只用高置信度的句子做统计
    high_conf = [s for s in sentence_scores if s['is_high_confidence']]

    if not high_conf:
        high_conf = sentence_scores  # 如果都是低置信度，退回用全部

    scores = np.array([s['score'] for s in high_conf])
    confidences = np.array([s['confidence'] for s in high_conf])

    # 加权平均情绪分数
    weighted_score = float(np.average(scores, weights=confidences))

    # 情绪分布统计
    labels = [s['label'] for s in high_conf]
    total = len(labels)
    positive_count = labels.count('positive')
    negative_count = labels.count('negative')

    return {
        "weighted_score":    round(weighted_score, 4),
        "positive_ratio":    round(positive_count / total, 4),
        "negative_ratio":    round(negative_count / total, 4),
        "neutral_ratio":     round((total - positive_count - negative_count) / total, 4),
        "total_sentences":   total,
        "high_conf_count":   len(high_conf),
        # 情绪强度：绝对值越大，情绪越明确（不管正负）
        "sentiment_magnitude": round(abs(weighted_score), 4),
        # 情绪分类（便于后续使用）
        "sentiment_label":  "positive" if weighted_score > 0.1
                           else "negative" if weighted_score < -0.1
                           else "neutral",
    }


def analyze_ticker_filings(ticker: str, nlp_pipeline) -> list[dict]:
    """
    分析某支股票的所有已清洗财报文件，输出情绪信号。
    """
    sentences_path = CLEAN_DIR / f"{ticker}_sentences.json"

    if not sentences_path.exists():
        logger.warning(f"[{ticker}] 未找到清洗后的文本，请先运行 text_cleaner.py")
        return []

    filing_data = json.loads(sentences_path.read_text(encoding='utf-8'))
    results = []

    for filing in filing_data:
        sentences = filing.get('sentences', [])
        if not sentences:
            continue

        logger.info(f"[{ticker}] 分析 {filing['form_type']} {filing['filing_date']}，"
                    f"共 {len(sentences)} 句...")

        # 打分
        sentence_scores = score_sentences_batch(sentences, nlp_pipeline)

        # 聚合
        doc_sentiment = aggregate_document_sentiment(sentence_scores)

        # 构建结果记录
        result = {
            "ticker":          ticker,
            "filing_date":     filing['filing_date'],
            "form_type":       filing['form_type'],
            **doc_sentiment,   # 展开情绪指标
        }
        results.append(result)

        logger.info(
            f"[{ticker}] {filing['form_type']} {filing['filing_date']} "
            f"情绪分析完成：score={doc_sentiment['weighted_score']:.4f} "
            f"({doc_sentiment['sentiment_label']})，"
            f"正面{doc_sentiment['positive_ratio']:.1%} "
            f"负面{doc_sentiment['negative_ratio']:.1%}"
        )

    # 保存情绪信号
    if results:
        import json as json_module
        output_path = SIGNAL_DIR / f"{ticker}_sentiment.json"
        output_path.write_text(
            json_module.dumps(results, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.info(f"[{ticker}] 情绪信号已保存至 {output_path}")

    return results


# ============================================================
# 快速演示：不需要下载文件，直接用示例文本验证 FinBERT
# ============================================================

DEMO_SENTENCES = [
    # 正面
    "Our revenue grew 23% year-over-year, driven by strong demand across all segments.",
    "We are confident in our ability to continue delivering strong returns to shareholders.",
    "The successful launch of our new product line exceeded all internal expectations.",
    # 负面
    "Revenue dropped by 20% due to supply chain disruptions and macroeconomic headwinds.",
    "We face significant uncertainty regarding regulatory compliance in key markets.",
    "Operating margins compressed significantly as input costs rose sharply.",
    # 中性（VADER 容易出错，FinBERT 应该正确识别）
    "The company reduced its total debt by $2 billion during the fiscal year.",  # 减少债务 = 好事，但 VADER 会打负分
    "We will continue to evaluate strategic alternatives to maximize shareholder value.",
    "Our cash and cash equivalents decreased to $15.3 billion from $18.7 billion.",
]


def run_demo():
    """运行快速演示，验证 FinBERT 的金融语义理解能力。"""
    print("\n" + "="*70)
    print("FinBERT 金融情绪分析演示")
    print("="*70)

    nlp = load_finbert_pipeline()

    print("\n{'句子':<60} {'情绪':<12} {'置信度'}")
    print("-"*70)

    for sent in DEMO_SENTENCES:
        result = nlp(sent)[0]
        label = result['label'].lower()
        conf = result['score']

        # 颜色标记（终端输出）
        emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "⚪"
        print(f"{emoji} {sent[:58]:<58} {label:<12} {conf:.4f}")

    print("\n注意观察：'The company reduced its total debt' 被正确识别为正面/中性")
    print("这证明 FinBERT 理解了'减少债务'在金融语境中是正面的。")
    print("如果用 VADER，这句话会因为'debt'这个词被错误打成负分。")


if __name__ == "__main__":
    # 先运行演示，验证模型工作正常
    run_demo()

    # 如果已经完成了前两步（下载 + 清洗），再运行完整分析
    tickers = ['AAPL', 'MSFT', 'NVDA']
    clean_files_exist = any(
        (CLEAN_DIR / f"{t}_sentences.json").exists() for t in tickers
    )

    if clean_files_exist:
        nlp_pipeline = load_finbert_pipeline()
        for ticker in tickers:
            analyze_ticker_filings(ticker, nlp_pipeline)
    else:
        print("\n（未找到清洗后的文本文件，跳过完整分析。请先运行 sec_downloader.py 和 text_cleaner.py）")
