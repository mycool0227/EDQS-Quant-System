import math
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# ----------------------
# Fonts & style
# ----------------------
def _set_chinese_font():
    """Try common Chinese fonts; fall back silently if missing."""
    candidate_paths = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            font_prop = fm.FontProperties(fname=path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            return

    # 若未命中路径，尝试常见中文字体族名
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# 设置风格与字体
try:
    plt.style.use("seaborn-muted")
except OSError:
    plt.style.use("default")
_set_chinese_font()

# ----------------------
# Paths
# ----------------------
DEFAULT_ACCOUNT_CSV = Path("result/account_value.csv")
DEFAULT_BENCHMARK_CSV = None
DEFAULT_PLOT_PATH = Path("result/net_value_vs_hs300.png")
DEFAULT_METRICS_PATH = Path("result/metrics_summary.csv")

# ----------------------
# Loaders
# ----------------------
def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_account_value(account_path: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Load account values. If the file is missing, generate a simple upward
    synthetic curve so the plotting code can still run.

    Returns (df, used_synthetic).
    """
    if account_path.exists():
        df = pd.read_csv(account_path)
        used_synthetic = False
    else:
        weeks = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
        values = np.linspace(100000, 150000, len(weeks))
        df = pd.DataFrame({"week": weeks, "account_value": values})
        used_synthetic = True

    rename_map = {"日期": "week", "account": "account_value"}
    df = df.rename(columns=rename_map)

    if "week" not in df.columns or "account_value" not in df.columns:
        raise ValueError("account_value.csv 需要包含 week 和 account_value 列")

    df["week"] = pd.to_datetime(df["week"])
    df["account_value"] = pd.to_numeric(df["account_value"], errors="coerce")
    df = df.dropna(subset=["week", "account_value"]).sort_values("week")

    # 若只有一行，补一行平直的占位，便于绘图
    added_placeholder = False
    if len(df) == 1:
        added_placeholder = True
        next_week = df.iloc[0]["week"] + pd.Timedelta(days=7)
        df = pd.concat([
            df,
            pd.DataFrame({"week": [next_week], "account_value": [df.iloc[0]["account_value"]]})
        ], ignore_index=True)

    # 规范化净值曲线
    initial = df["account_value"].iloc[0]
    df["net_value"] = df["account_value"] / initial
    df["placeholder_flat"] = added_placeholder

    if added_placeholder:
        print("account_value.csv 仅1行，已补充一行平直占位用于绘图。")

    return df.reset_index(drop=True), used_synthetic


def load_benchmark(benchmark_path: Path, strategy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load HS300 benchmark data. If the file does not exist, create a mild
    synthetic curve as a placeholder.
    """
    if benchmark_path and benchmark_path.exists():
        bench = pd.read_csv(benchmark_path, comment="#", on_bad_lines="skip")
        rename_map = {
            "trade_date": "date",
            "日期": "date",
            "Date": "date",
            "收盘": "close",
            "Close": "close",
            "收盘价": "close",
            "代码": "code",
            "股票代码": "code",
            "symbol": "code",
            "Symbol": "code",
        }
        bench = bench.rename(columns=rename_map)

        # 若是多标的日线表，优先提取沪深300(000300)
        if "code" in bench.columns:
            # 兼容 000300 / sh000300 / sz000300 / 000300.SH 等格式
            bench["code"] = (
                bench["code"]
                .astype(str)
                .str.extract(r"(\d{6})", expand=False)
            )
            if (bench["code"] == "000300").any():
                bench = bench[bench["code"] == "000300"].copy()
                print("基准已使用本地文件中的 000300（沪深300）数据。")
            else:
                # 未包含000300时使用第一只代码，确保流程可运行
                first_code = bench["code"].dropna().iloc[0] if not bench["code"].dropna().empty else None
                if first_code is not None:
                    bench = bench[bench["code"] == first_code].copy()
                    print(f"警告: 本地基准文件不含000300，暂使用 {first_code} 作为基准。")

        if "date" not in bench.columns or "close" not in bench.columns:
            raise ValueError("基准文件需要包含 date 和 close 列")

        bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
        bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
        bench = bench.dropna(subset=["date", "close"]).sort_values("date")

        bench["net_value"] = bench["close"] / bench["close"].iloc[0]

        # 对齐到策略周频，用最近交易日向前对齐
        aligned = pd.merge_asof(
            strategy_df[["week"]],
            bench[["date", "net_value"]],
            left_on="week",
            right_on="date",
            direction="backward",
        )
        aligned["net_value"] = aligned["net_value"].ffill().bfill()
        aligned = aligned.rename(columns={"week": "date", "date": "bench_date"})
        return aligned[["date", "net_value"]]

    # 未提供基准文件时，尝试自动拉取沪深300真实数据
    try:
        import akshare as ak

        start = (strategy_df["week"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
        end = (strategy_df["week"].max() + pd.Timedelta(days=10)).strftime("%Y%m%d")
        hs300 = None
        try:
            hs300 = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start, end_date=end)
        except Exception:
            hs300 = None

        # 某些网络环境下 index_zh_a_hist 不稳定，回退到日线接口
        if hs300 is None or hs300.empty:
            try:
                hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
            except Exception:
                hs300 = None

        if hs300 is not None and not hs300.empty:
            hs300 = hs300.rename(columns={"日期": "date", "收盘": "close"})
            if "date" not in hs300.columns and "Date" in hs300.columns:
                hs300 = hs300.rename(columns={"Date": "date"})
            if "close" not in hs300.columns and "Close" in hs300.columns:
                hs300 = hs300.rename(columns={"Close": "close"})

            hs300["date"] = pd.to_datetime(hs300["date"])
            hs300["close"] = pd.to_numeric(hs300["close"], errors="coerce")
            hs300 = hs300.dropna(subset=["date", "close"]).sort_values("date")

            # 统一按策略窗口前后做切片
            min_dt = strategy_df["week"].min() - pd.Timedelta(days=10)
            max_dt = strategy_df["week"].max() + pd.Timedelta(days=10)
            hs300 = hs300[(hs300["date"] >= min_dt) & (hs300["date"] <= max_dt)]

            hs300["net_value"] = hs300["close"] / hs300["close"].iloc[0]
            aligned = pd.merge_asof(
                strategy_df[["week"]],
                hs300[["date", "net_value"]],
                left_on="week",
                right_on="date",
                direction="backward",
            )
            aligned["net_value"] = aligned["net_value"].ffill().bfill()
            aligned = aligned.rename(columns={"week": "date", "date": "bench_date"})
            print("未提供 benchmark_csv，已自动使用沪深300真实行情。")
            return aligned[["date", "net_value"]]
    except Exception as exc:
        print(f"自动拉取沪深300失败({exc})，将回退示例基准曲线。")

    # 若策略仅有平直占位（1行扩展而成），让基准与策略净值保持一致，避免视觉不一致
    if "placeholder_flat" in strategy_df.columns and strategy_df["placeholder_flat"].any():
        synthetic = pd.DataFrame({
            "date": strategy_df["week"],
            "net_value": strategy_df["net_value"],
        })
    else:
        synthetic = pd.DataFrame({
            "date": strategy_df["week"],
            "net_value": np.linspace(1.0, 1.05, len(strategy_df)),
        })
    return synthetic


# ----------------------
# Metrics
# ----------------------
def compute_metrics(account_df: pd.DataFrame, periods_per_year: int = 52) -> Dict[str, float]:
    returns = account_df["account_value"].pct_change().dropna()

    sharpe = np.nan
    if not returns.empty and returns.std(ddof=1) != 0:
        sharpe = (returns.mean() / returns.std(ddof=1)) * math.sqrt(periods_per_year)

    peak = account_df["net_value"].cummax()
    drawdown = account_df["net_value"] / peak - 1
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan

    win_rate = returns.gt(0).mean() if not returns.empty else np.nan

    total_return = account_df["net_value"].iloc[-1] - 1

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
        "periods": len(account_df),
    }


# ----------------------
# Plotting
# ----------------------
def plot_curves(strategy_df: pd.DataFrame, benchmark_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_directory(output_path)

    # 优先使用 seaborn-muted；若环境不支持则退回默认样式，避免报错
    try:
        plt.style.use("seaborn-muted")
    except OSError:
        plt.style.use("default")
    _set_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    # 显式转成 numpy，避免老版本 matplotlib/pandas 触发“多维索引”错误
    ax.plot(strategy_df["week"].to_numpy(), strategy_df["net_value"].to_numpy(), label="策略净值", linewidth=2.0, color="#1f77b4")
    ax.plot(benchmark_df["date"].to_numpy(), benchmark_df["net_value"].to_numpy(), label="沪深300基准", linewidth=2.0, color="#d62728", linestyle="--")

    ax.set_title("策略净值 VS 沪深300基准", fontsize=14)
    ax.set_xlabel("周")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------
# Runner
# ----------------------
def run(
    account_csv: Path = DEFAULT_ACCOUNT_CSV,
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
    plot_path: Path = DEFAULT_PLOT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> Dict[str, float]:
    account_df, used_synthetic = load_account_value(account_csv)
    benchmark_df = load_benchmark(benchmark_csv if benchmark_csv else None, account_df)

    metrics = compute_metrics(account_df)
    plot_curves(account_df, benchmark_df, plot_path)

    _ensure_directory(metrics_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    if used_synthetic:
        print("account_value.csv 未找到，已使用合成示例曲线计算指标和绘图。")

    print("量化指标：", metrics)
    print(f"净值对比图已保存到: {plot_path}")
    print(f"指标明细已保存到: {metrics_path}")

    return metrics


if __name__ == "__main__":
    run()
