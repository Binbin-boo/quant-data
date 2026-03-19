"""
项目 2：全球科技股实时监控看板
================================
使用 Streamlit + Plotly 构建专业级交互式量化看板。

学习目标：
  - 掌握 Streamlit 的响应式组件和状态管理
  - 理解 MACD / RSI 的计算原理和交易信号含义
  - 学习 Plotly 的 make_subplots 多图联动布局
  - 体验"支撑位、背离、超买超卖"等核心技术分析概念

运行：
  pip install streamlit plotly
  streamlit run app.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

from storage import get_storage

# ============================================================
# Streamlit 页面配置
#
# 学习要点 - st.set_page_config() 必须是第一个 Streamlit 调用：
#   layout="wide"  → 使用浏览器全宽（量化看板必须），否则默认窄列
#   page_icon      → 浏览器标签页的图标
#   initial_sidebar_state → 控制左侧边栏默认展开/收起
# ============================================================
st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "market_data"

# 存储后端：本地开发自动用 LocalStorage，云端自动用 S3Storage
# 与 data_pipeline.py 的 get_storage() 共用同一套逻辑
_storage = get_storage(base_dir=str(Path(__file__).parent))

# ============================================================
# 自定义 CSS：黑色主题 + 专业字体
#
# 学习要点 - st.markdown(unsafe_allow_html=True)：
#   Streamlit 本身的主题系统功能有限，
#   通过注入 CSS 可以完全自定义外观。
#   这是生产级 Streamlit 应用的标准技巧。
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3548;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #8892a4; margin-top: 4px; }
    .positive { color: #00d4aa; }
    .negative { color: #ff4b6e; }
    .neutral  { color: #ffd700; }
    div[data-testid="stSidebar"] { background-color: #161b2a; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 第一步：数据加载层（带 Streamlit 缓存）
#
# 学习要点 - @st.cache_data 装饰器：
#   Streamlit 的执行模型是"每次用户操作都重新执行整个脚本"。
#   如果不缓存，每次滑动日期滑块都会重新读 Parquet 文件，体验极差。
#   @st.cache_data 把函数结果缓存到内存中，
#   只有当参数（ticker）变化时才重新执行。
#   ttl=3600 表示缓存 1 小时后自动失效（适合日线数据更新频率）。
# ============================================================

@st.cache_data(ttl=3600)
def load_stock_data(ticker: str) -> pd.DataFrame:
    """加载并返回指定股票的历史数据（本地或云端，由 _storage 自动决定）。"""
    df = _storage.read_parquet(f"market_data/{ticker}_historical.parquet")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df


# ============================================================
# 第二步：技术指标计算
#
# 这是本看板的"金融核心"，你会在这里真正理解指标的数学本质
# ============================================================

def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 MACD（移动平均收敛散度）。

    学习要点 - MACD 的三条线：
      1. MACD Line（快线）= EMA(12) - EMA(26)
         捕捉短期动量和长期趋势的差距
      2. Signal Line（信号线）= EMA(MACD Line, 9)
         MACD 的平滑版，用来产生买卖信号
      3. Histogram（柱状图）= MACD - Signal
         两线之差，正值→柱子在零轴上方（看多），负值→看空

    交易信号（入门级）：
      ✅ 金叉：MACD 线从下方穿越信号线 → 买入信号
      ❌ 死叉：MACD 线从上方穿越信号线 → 卖出信号
      ⚠️ 背离：价格新高但 MACD 不新高（顶背离）→ 趋势可能反转

    为什么用 EMA（指数移动平均）而不是 SMA（简单移动平均）？
      EMA 对近期数据赋予更高权重，反应更灵敏。
      SMA 把所有历史数据等权重处理，对旧数据过于迟钝。
      计算公式：EMA_t = α × P_t + (1-α) × EMA_{t-1}，其中 α = 2/(N+1)
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 RSI（相对强弱指数）。

    学习要点 - RSI 的数学原理：
      RSI = 100 - 100 / (1 + RS)
      RS = 过去 N 天平均涨幅 / 过去 N 天平均跌幅

      RSI 的含义：衡量"最近 N 天内，上涨的力度占总波动的比例"
        RSI = 100：每天都在涨（极度超买，泡沫信号）
        RSI = 0  ：每天都在跌（极度超卖）
        RSI = 50 ：涨跌力度相当

    交易信号（入门级）：
      RSI > 70：超买（Overbought）→ 短期可能回调，注意止盈
      RSI < 30：超卖（Oversold）  → 短期可能反弹，注意抄底窗口
      RSI = 50 穿越：趋势确认信号

    注意 - RSI 的局限性（非常重要！）：
      在强趋势行情中，RSI 可以在 70 以上持续运行几个月。
      比如 NVDA 在 2023 年 AI 行情中，RSI 长期超买但股价一直涨。
      所以 RSI 只能作为参考，不能单独作为交易决策依据。
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)   # 只保留涨幅（跌幅设为 0）
    loss  = (-delta).clip(lower=0) # 只保留跌幅的绝对值

    # 用 EMA 计算平均涨跌幅（Wilder's Smoothing，比简单平均更准确）
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.inf)  # 防止除以 0
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带（Bollinger Bands）。

    学习要点 - 布林带的含义：
      中轨 = 20 日 SMA（趋势中枢）
      上轨 = 中轨 + 2σ（价格高估区域）
      下轨 = 中轨 - 2σ（价格低估区域）

      统计意义：正态分布下，约 95% 的价格落在上下轨之间。
      当价格触及上轨 → 可能过热；触及下轨 → 可能超卖。
      布林带收窄（带宽小）→ 低波动，即将爆发的前兆。
    """
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ============================================================
# 第三步：主图表绘制
#
# 学习要点 - Plotly make_subplots：
#   量化看板需要多个图表共享同一个 X 轴（时间轴），
#   这样拖动/缩放时所有子图同步联动。
#   make_subplots(rows=3, shared_xaxes=True) 实现这个效果。
#
#   subplot 的 row_heights 控制各子图的相对高度比例：
#   主 K 线图要占最大空间，MACD/RSI 占次要空间。
# ============================================================

def build_chart(df: pd.DataFrame, ticker: str, show_bb: bool = True) -> go.Figure:
    """
    构建包含 K 线 + 成交量 + MACD + RSI 的四合一图表。
    """
    # 计算指标
    macd_line, signal_line, histogram = compute_macd(df['adj_close'])
    rsi = compute_rsi(df['adj_close'])
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(df['adj_close'])

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.20, 0.15],
        vertical_spacing=0.02,
        subplot_titles=("", "成交量", "MACD (12,26,9)", "RSI (14)"),
    )

    # ── 子图1：K 线图 + 布林带 ──────────────────────────────
    # 学习要点 - Candlestick（蜡烛图/K线）的四要素：
    #   Open（开盘价）、High（最高价）、Low（最低价）、Close（收盘价）
    #   绿色蜡烛（阳线）：收盘 > 开盘，当天上涨
    #   红色蜡烛（阴线）：收盘 < 开盘，当天下跌
    #   上下影线：当天最高/最低价与开盘/收盘的差距，代表价格的"探测范围"
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'],   close=df['close'],
        name="K线",
        increasing_line_color='#00d4aa',  # 涨：绿色
        decreasing_line_color='#ff4b6e',  # 跌：红色
        increasing_fillcolor='#00d4aa',
        decreasing_fillcolor='#ff4b6e',
    ), row=1, col=1)

    if show_bb:
        # 布林上轨
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_upper,
            name="布林上轨", line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dot'),
            showlegend=False,
        ), row=1, col=1)
        # 布林中轨（20日均线）
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_mid,
            name="MA20", line=dict(color='rgba(255,255,255,0.4)', width=1),
            showlegend=False,
        ), row=1, col=1)
        # 布林下轨（填充带状区域）
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lower,
            name="布林下轨",
            line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dot'),
            fill='tonexty',  # 填充到上方曲线（即布林中轨），形成带状区域
            fillcolor='rgba(255,165,0,0.05)',
            showlegend=False,
        ), row=1, col=1)

    # ── 子图2：成交量柱状图 ──────────────────────────────────
    # 学习要点 - 成交量颜色跟随价格涨跌：
    #   量价配合是技术分析的核心：放量上涨 > 缩量上涨（信号更强）
    volume_colors = [
        '#00d4aa' if c >= o else '#ff4b6e'
        for c, o in zip(df['close'], df['open'])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        name="成交量",
        marker_color=volume_colors,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    # ── 子图3：MACD ──────────────────────────────────────────
    # 柱状图颜色：正值绿（多头强），负值红（空头强）
    hist_colors = ['#00d4aa' if v >= 0 else '#ff4b6e' for v in histogram]
    fig.add_trace(go.Bar(
        x=df.index, y=histogram,
        name="MACD 柱", marker_color=hist_colors, opacity=0.8,
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_line,
        name="MACD", line=dict(color='#4da6ff', width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=signal_line,
        name="信号线", line=dict(color='#ff9500', width=1.5),
    ), row=3, col=1)
    # 零轴参考线
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=3, col=1)

    # ── 子图4：RSI ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        name="RSI", line=dict(color='#c084fc', width=1.5),
        fill='tozeroy', fillcolor='rgba(192,132,252,0.05)',
    ), row=4, col=1)
    # 超买线（70）
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,75,110,0.5)",
                  annotation_text="超买 70", annotation_position="left",
                  row=4, col=1)
    # 超卖线（30）
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,212,170,0.5)",
                  annotation_text="超卖 30", annotation_position="left",
                  row=4, col=1)
    # 50 中轴
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", row=4, col=1)

    # ── 全局布局设置 ─────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> 技术分析看板",
            font=dict(size=20, color='#e0e6f0'),
            x=0.02,
        ),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b2a',
        font=dict(color='#8892a4', size=11),
        legend=dict(
            orientation="h", x=0.01, y=0.99,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,  # 关掉 K 线下方的默认滑块（我们用 Streamlit 控件代替）
        height=850,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # 所有子图的网格线统一设置（暗色系）
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.05)',
            showgrid=True, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.05)',
            showgrid=True, row=i, col=1,
        )

    return fig


# ============================================================
# 第四步：Streamlit 应用主体
#
# 学习要点 - Streamlit 的执行模型：
#   Streamlit 是"响应式（Reactive）"框架。
#   每次用户改变任何控件（选择框、滑块、按钮），
#   整个 Python 脚本从头到尾重新执行一遍。
#   和 React 的状态驱动 UI 是同一个设计思想，只是用 Python 实现。
#
#   因此你不需要写事件监听回调（unlike Dash），
#   只需要写"如果控件值是 X，则展示 Y"的声明式逻辑。
# ============================================================

def render_metrics_row(df: pd.DataFrame, ticker: str):
    """
    渲染顶部指标卡片行（最新价、日涨跌幅、年初至今收益等）。
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price     = latest['adj_close']
    day_ret   = latest['daily_return']
    ytd_start = df[df.index.year == df.index[-1].year].iloc[0]['adj_close']
    ytd_ret   = (price - ytd_start) / ytd_start
    vol_20d   = latest.get('volatility_20d', df['daily_return'].std() * np.sqrt(252))
    rsi_val   = compute_rsi(df['adj_close']).iloc[-1]

    day_color  = "positive" if day_ret >= 0 else "negative"
    ytd_color  = "positive" if ytd_ret >= 0 else "negative"
    rsi_color  = "negative" if rsi_val > 70 else "positive" if rsi_val < 30 else "neutral"

    cols = st.columns(5)
    metrics = [
        ("最新价（复权）",    f"${price:.2f}",              "neutral"),
        ("日涨跌幅",          f"{day_ret:+.2%}",            day_color),
        ("年初至今（YTD）",   f"{ytd_ret:+.2%}",            ytd_color),
        ("年化波动率（20d）", f"{vol_20d:.1%}",              "neutral"),
        ("RSI (14)",          f"{rsi_val:.1f}",              rsi_color),
    ]

    for col, (label, value, color) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar() -> tuple[str, tuple, bool]:
    """
    渲染侧边栏控件，返回用户选择的参数。

    学习要点 - Streamlit 侧边栏（Sidebar）的设计原则：
      把"配置类"控件（选哪支股票、选什么时间段）放侧边栏，
      把"展示类"内容（图表、数据表格）放主区域。
      这是 dashboard 设计的通用 UI 范式。
    """
    with st.sidebar:
        st.markdown("## 📊 控制面板")
        st.markdown("---")

        # 股票选择
        available_tickers = [
            p.stem.replace('_historical', '')
            for p in DATA_DIR.glob('*_historical.parquet')
        ]
        ticker = st.selectbox("选择股票", sorted(available_tickers), index=0)

        st.markdown("---")

        # 日期范围
        st.markdown("### 时间范围")
        range_options = {
            "近 3 个月":  90,
            "近 6 个月":  180,
            "近 1 年":    365,
            "近 2 年":    730,
            "近 5 年":    1825,
            "全部数据":   9999,
        }
        selected_range = st.selectbox("快速选择", list(range_options.keys()), index=2)
        days = range_options[selected_range]
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)
        date_range = (start_date, end_date)

        st.markdown("---")

        # 显示选项
        st.markdown("### 显示设置")
        show_bb = st.checkbox("布林带", value=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#8892a4; font-size:0.75rem;">
        📖 使用指南：<br>
        • 鼠标滚轮缩放图表<br>
        • 拖拽平移时间轴<br>
        • 双击重置视图<br>
        • 点击图例隐藏/显示<br>
        </div>
        """, unsafe_allow_html=True)

    return ticker, date_range, show_bb


def main():
    # 侧边栏控件
    ticker, date_range, show_bb = render_sidebar()

    # 加载数据
    df = load_stock_data(ticker)
    if df.empty:
        st.error(f"未找到 {ticker} 的数据，请先运行 data_pipeline.py")
        return

    # 按日期筛选
    mask = (df.index >= pd.Timestamp(date_range[0])) & \
           (df.index <= pd.Timestamp(date_range[1]))
    df_view = df[mask].copy()

    if len(df_view) < 30:
        st.warning("数据点不足 30 个，请选择更长的时间范围")
        return

    # 标题行
    st.markdown(f"# {ticker} · 量化监控看板")
    st.markdown(f"数据范围：{df_view.index[0].date()} → {df_view.index[-1].date()}，共 {len(df_view)} 个交易日")

    # 指标卡片行
    render_metrics_row(df_view, ticker)
    st.markdown("<br>", unsafe_allow_html=True)

    # 主图表
    fig = build_chart(df_view, ticker, show_bb=show_bb)
    # use_container_width=True 让图表宽度自适应浏览器窗口
    st.plotly_chart(fig, use_container_width=True)

    # ── 数据洞察区 ───────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 收益率分布")
        returns = df_view['daily_return'].dropna()

        # 用 Plotly 画收益率直方图
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=60,
            name="日收益率",
            marker_color='#4da6ff',
            opacity=0.75,
        ))
        # 正态分布拟合曲线（可视化"肥尾"现象）
        x_range = np.linspace(returns.min(), returns.max(), 200)
        from scipy.stats import norm
        mu, sigma = returns.mean(), returns.std()
        normal_pdf = norm.pdf(x_range, mu, sigma) * len(returns) * (returns.max()-returns.min()) / 60
        hist_fig.add_trace(go.Scatter(
            x=x_range, y=normal_pdf,
            name="正态分布拟合",
            line=dict(color='#ff9500', width=2),
        ))
        hist_fig.update_layout(
            paper_bgcolor='#0e1117', plot_bgcolor='#161b2a',
            font=dict(color='#8892a4'),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        st.caption(
            "注意：真实股票收益率的尾部比正态分布更厚（肥尾，Fat Tails）。"
            "橙色曲线是理论正态分布，蓝色柱子是实际分布——两侧极端值比理论更多。"
            "这是金融风险模型不能简单用正态分布的根本原因。"
        )

    with col2:
        st.markdown("### 滚动统计（近期数据）")
        recent = df_view.tail(20)[['adj_close', 'daily_return', 'volume']].copy()
        recent.index = recent.index.date
        recent.columns = ['复权收盘价', '日收益率', '成交量']
        recent['日收益率'] = recent['日收益率'].map(lambda x: f"{x:+.2%}")
        recent['成交量'] = recent['成交量'].map(lambda x: f"{x/1e6:.1f}M")
        st.dataframe(
            recent.sort_index(ascending=False),
            use_container_width=True,
            height=300,
        )


if __name__ == "__main__":
    main()
