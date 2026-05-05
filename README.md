# Teddy Project
# EDQS: 事件驱动型量化选股系统
# (Event-Driven Quantitative Selection System)

EDQS 是一套端到端的量化研究与交易辅助平台。系统通过大语言模型（LLM）对海量财经新闻进行语义解构，结合严苛的财务与技术面风控模型，自动化挖掘具备周度爆发潜力的股票组合。

## 核心流程图

1. **智能感知**：实时抓取东方财富快讯，通过 LLM (DeepSeek/Gemini) 结构化提取事件强度与情绪。
2. **知识映射**：基于概念板块词表，将新闻事件精准映射至具体的 A 股标的。
3. **风控过滤**：执行财务红线审查（剔除ST/亏损）与技术面共振评估（动量/波动率校验）。
4. **组合配置**：生成 Top3 动态权重组合，模拟周二买入、周五卖出的周频轮动策略。
5. **绩效评估**：自动化回测并生成净值曲线及夏普比率、最大回撤等核心指标。

## 项目结构说明

```text
├── run_by_dates.ps1          # 主流程驱动脚本 (入口)
├── em_historical_selector.py # 数据抓取与事件抽取核心逻辑
├── llm_api_filter_date.py    # 多 LLM 适配器与容错封装
├── graph_matcher.py          # 事件-股票知识图谱映射模块
├── RiskFilter.py             # 财务与技术面风控过滤引擎
├── rolling_weekly_backtest.py# 周频回测与权重分配模块
├── MetricsPlotter.py         # 可视化报告生成工具
└── data/                     # (本地数据目录，建议放置概念板块.csv等)

## Quick Start (Windows)

### 1) Create a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Prepare required data files
Ensure these files exist at the project root:
- 日线数据.csv
- 概念板块.csv

### 4) Configure LLM API key (at least one)
```powershell
$env:DEEPSEEK_API_KEY="your_key"
# or
$env:ZHIPU_API_KEY="your_key"
# or
$env:GOOGLE_API_KEY="your_key"
```
Optional: force provider
```powershell
$env:LLM_PROVIDER="deepseek"  # deepseek | zhipu | gemini | auto
```

### 5) Run the main pipeline
```powershell
.\run_by_dates.ps1 -StartDate 2026-04-12 -EndDate 2026-04-20
```

### 6) Check outputs
Results are written to the result/ folder.

## Notes
- If you change dates frequently, keep the virtual environment activated.
- The script will download news data online; make sure your network can access the sources.
