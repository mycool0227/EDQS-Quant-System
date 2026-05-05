import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from RiskFilter import RiskFilter


BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "result"
DATA_DB_DIR = BASE_DIR / "data" / "db"
DATA_EVENTS_DIR = BASE_DIR / "data" / "events"


def _enforce_weight_bounds(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Clamp weights into [min_weight, max_weight] and renormalize to sum 1."""
    n = len(weights)
    if n == 0:
        return weights

    if min_weight * n > 1.0 + 1e-12:
        raise ValueError("min_weight 过大，无法满足权重和为1。")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight 过小，无法满足权重和为1。")

    w = np.array(weights, dtype=float)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / w.sum()

    for _ in range(10):
        w = np.clip(w, min_weight, max_weight)
        diff = 1.0 - w.sum()
        if abs(diff) < 1e-10:
            break

        if diff > 0:
            free = np.where(w < max_weight - 1e-12)[0]
            if len(free) == 0:
                break
            add = diff / len(free)
            w[free] += add
        else:
            free = np.where(w > min_weight + 1e-12)[0]
            if len(free) == 0:
                break
            sub = (-diff) / len(free)
            w[free] -= sub

    w = np.clip(w, min_weight, max_weight)
    w = w / w.sum()
    return w


def compute_dynamic_weights(
    top_df: pd.DataFrame,
    score_beta: float = 2.0,
    mix_lambda: float = 0.6,
    min_weight: float = 0.15,
    max_weight: float = 0.55,
) -> pd.DataFrame:
    """Compute per-stock dynamic weights from composite score and inverse volatility."""
    if top_df is None or len(top_df) == 0:
        return pd.DataFrame(columns=list(top_df.columns) + ["weight", "score_weight", "risk_weight"]) if top_df is not None else pd.DataFrame()

    df = top_df.copy()
    if "composite_score" not in df.columns:
        df["composite_score"] = pd.to_numeric(df.get("strength", 0), errors="coerce").fillna(0)
    if "vol_20d" not in df.columns:
        df["vol_20d"] = 0.0

    scores = pd.to_numeric(df["composite_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vols = pd.to_numeric(df["vol_20d"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if len(scores) == 1:
        df["score_weight"] = 1.0
        df["risk_weight"] = 1.0
        df["weight"] = 1.0
        return df

    score_std = scores.std()
    if score_std < 1e-12:
        z = np.zeros_like(scores)
    else:
        z = (scores - scores.mean()) / score_std

    logits = score_beta * z
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    score_weight = exp_logits / exp_logits.sum()

    safe_vols = np.maximum(vols, 1e-6)
    inv_vol = 1.0 / safe_vols
    risk_weight = inv_vol / inv_vol.sum()

    mix_lambda = float(np.clip(mix_lambda, 0.0, 1.0))
    raw = mix_lambda * score_weight + (1.0 - mix_lambda) * risk_weight
    final_weight = _enforce_weight_bounds(raw, min_weight=min_weight, max_weight=max_weight)

    df["score_weight"] = score_weight
    df["risk_weight"] = risk_weight
    df["weight"] = final_weight
    return df


def load_price_data(price_file: Path) -> pd.DataFrame:
    df = pd.read_csv(price_file, encoding="utf-8")
    col_map = {
        "代码": "code",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
    }
    df = df.rename(columns=col_map)

    required = {"code", "date", "open", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"日线数据缺少必需列: {required}")

    df["code"] = df["code"].astype(str).str.strip().str.split(".").str[0].str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    return df


def get_trade_weeks(json_file: Path, price_df: pd.DataFrame, max_weeks: int) -> list[pd.Timestamp]:
    events = json.loads(json_file.read_text(encoding="utf-8"))

    event_dates = pd.to_datetime([e.get("date", "") for e in events], errors="coerce")
    event_dates = event_dates[pd.notna(event_dates)]
    if len(event_dates) == 0:
        return []

    event_weeks = (event_dates - pd.to_timedelta(event_dates.weekday, unit="D")).normalize()
    trade_weeks = sorted({w + pd.Timedelta(days=7) for w in event_weeks})

    max_price_date = price_df["date"].max().normalize()
    eligible = [w for w in trade_weeks if (w + pd.Timedelta(days=4)) <= max_price_date]

    if max_weeks and max_weeks > 0:
        eligible = eligible[-max_weeks:]

    return eligible


def fetch_week_return(price_df: pd.DataFrame, code: str, week_start: pd.Timestamp):
    buy_date = week_start + pd.Timedelta(days=1)
    sell_date = week_start + pd.Timedelta(days=4)

    stock_df = price_df[price_df["code"] == code]
    if stock_df.empty:
        return None

    buy_row = stock_df[stock_df["date"].dt.normalize() == buy_date]
    sell_row = stock_df[stock_df["date"].dt.normalize() == sell_date]

    if buy_row.empty or sell_row.empty:
        return None

    tuesday_open = float(buy_row.iloc[0]["open"])
    friday_close = float(sell_row.iloc[0]["close"])
    if tuesday_open == 0:
        weekly_return = 0.0
    else:
        weekly_return = (friday_close - tuesday_open) / tuesday_open

    return {
        "buy_date": buy_date,
        "sell_date": sell_date,
        "tuesday_open": tuesday_open,
        "friday_close": friday_close,
        "weekly_return": weekly_return,
    }


def run_rolling_backtest(
    max_weeks: int,
    initial_capital: float,
    db_path: Path,
    json_file: Path,
    price_file: Path,
    score_beta: float,
    mix_lambda: float,
    min_weight: float,
    max_weight: float,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_price_data(price_file)
    weeks = get_trade_weeks(json_file, price_df, max_weeks)

    if not weeks:
        raise ValueError("未找到可回测的交易周。")

    account_value = float(initial_capital)
    weekly_rows = []
    picks_rows = []

    print(f"准备回测交易周数量: {len(weeks)}")

    for i, week_start in enumerate(weeks, 1):
        print("-" * 70)
        print(f"[{i}/{len(weeks)}] 交易周: {week_start.date()} ~ {(week_start + pd.Timedelta(days=4)).date()}")

        rf = RiskFilter(
            db_path=str(db_path),
            json_file=str(json_file),
            price_data_file=str(price_file),
            trade_week_start=week_start,
        )
        top3 = rf.run()

        if top3 is None or len(top3) == 0:
            print("该周无可交易Top3，跳过。")
            continue

        top3 = top3.head(3).copy()
        top3["code"] = top3["code"].astype(str).str.zfill(6)
        top3 = compute_dynamic_weights(
            top3,
            score_beta=score_beta,
            mix_lambda=mix_lambda,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        per_stock = []
        skip_week = False
        for _, row in top3.iterrows():
            info = fetch_week_return(price_df, row["code"], week_start)
            if info is None:
                print(f"  股票 {row['code']} 缺少周二或周五行情，整周跳过。")
                skip_week = True
                break
            per_stock.append((row, info))

        if skip_week:
            continue

        week_return = 0.0
        week_start_capital = account_value
        for row, info in per_stock:
            weight = float(row.get("weight", 0.0))
            week_return += weight * info["weekly_return"]
            picks_rows.append(
                {
                    "week_start": week_start.date(),
                    "code": row["code"],
                    "buy_date": info["buy_date"].date(),
                    "sell_date": info["sell_date"].date(),
                    "tuesday_open": round(info["tuesday_open"], 6),
                    "friday_close": round(info["friday_close"], 6),
                    "weekly_return": info["weekly_return"],
                    "weight": weight,
                    "score_weight": float(row.get("score_weight", 0.0)),
                    "risk_weight": float(row.get("risk_weight", 0.0)),
                    "allocated_amount": round(week_start_capital * weight, 2),
                    "composite_score": round(float(row.get("composite_score", 0.0)), 6),
                    "vol_20d": round(float(row.get("vol_20d", 0.0)), 6),
                }
            )

        account_value = account_value * (1.0 + week_return)
        weekly_rows.append(
            {
                "week": week_start.date(),
                "account_value": account_value,
                "weekly_return": week_return,
            }
        )

        print(f"  周收益: {week_return:.4%} | 账户净值: {account_value:.2f}")
        print("  动态权重:")
        for _, row in top3.iterrows():
            print(
                f"    {row['code']}: {float(row['weight']):.2%} "
                f"(score={float(row.get('score_weight', 0.0)):.2%}, risk={float(row.get('risk_weight', 0.0)):.2%})"
            )

    if not weekly_rows:
        raise ValueError("没有生成有效的周度回测结果。")

    weekly_df = pd.DataFrame(weekly_rows)
    picks_df = pd.DataFrame(picks_rows)

    weekly_path = RESULT_DIR / "rolling_account_value.csv"
    picks_path = RESULT_DIR / "rolling_weekly_picks.csv"

    weekly_df.to_csv(weekly_path, index=False, encoding="utf-8-sig")
    picks_df.to_csv(picks_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print(f"滚动回测完成，周数: {len(weekly_df)}")
    print(f"结果文件: {weekly_path}")
    print(f"结果文件: {picks_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="按周滚动: 前一周事件选股, 次周周二买入周五卖出")
    parser.add_argument("--max-weeks", type=int, default=12, help="最多回测多少个交易周")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--db-path", default=str(DATA_DB_DIR / "structured_events.db"), help="结构化数据库路径")
    parser.add_argument("--json-file", default=str(DATA_EVENTS_DIR / "events_history.json"), help="事件JSON路径")
    parser.add_argument("--price-file", default=str(BASE_DIR / "日线数据.csv"), help="日线CSV路径")
    parser.add_argument("--score-beta", type=float, default=1.0, help="分数softmax温度系数，越大越偏向高分股")
    parser.add_argument("--mix-lambda", type=float, default=1.0, help="分数权重与风险权重融合系数[0,1]")
    parser.add_argument("--min-weight", type=float, default=0.10, help="单票最小仓位")
    parser.add_argument("--max-weight", type=float, default=0.60, help="单票最大仓位")
    return parser.parse_args()


def main():
    args = parse_args()
    run_rolling_backtest(
        max_weeks=args.max_weeks,
        initial_capital=args.initial_capital,
        db_path=Path(args.db_path),
        json_file=Path(args.json_file),
        price_file=Path(args.price_file),
        score_beta=args.score_beta,
        mix_lambda=args.mix_lambda,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )


if __name__ == "__main__":
    main()
