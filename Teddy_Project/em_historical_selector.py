import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
import requests

from RiskFilter import RiskFilter
from graph_matcher import match_stocks
from llm_api_filter_date import InsufficientBalanceError, call_llm
from rolling_weekly_backtest import compute_dynamic_weights, fetch_week_return


BASE_DIR = Path(__file__).resolve().parent
EVENTS_FILE = BASE_DIR / "data" / "events" / "events_em_history.json"
RESULT_DIR = BASE_DIR / "result"
DEFAULT_CAPITAL = 100000.0
DEFAULT_DB_PATH = BASE_DIR / "data" / "db" / "structured_events.db"
DEFAULT_PRICE_FILE = BASE_DIR / "日线数据.csv"
LLM_CACHE_FILE = BASE_DIR / "data" / "events" / "llm_event_cache.json"


def _write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

EM_URL = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
EM_PARAMS = {
    "client": "web",
    "biz": "web_724",
    "fastColumn": "102",
    "sortEnd": "",
    "pageSize": "200",
    "req_trace": "1710315450384",
}

POSITIVE_KEYWORDS = [
    "中标", "订单", "合同", "签约", "突破", "创新高", "并购", "收购", "重组",
    "政策", "补贴", "利好", "业绩预增", "扭亏", "增持", "回购", "扩产", "投产", "获批",
]

NEGATIVE_HINTS = ["利空", "减持", "业绩预减", "亏损", "下调", "暴雷", "处罚", "跌停"]

NOISE_HINTS = [
    "早评", "午评", "收评", "盘面", "解盘", "复盘", "股吧", "情绪", "短线",
    "龙虎榜", "异动", "拉升", "冲高", "回落", "震荡", "题材", "连板",
]

KEYWORD_WEIGHTS = {
    "中标": 2.0,
    "订单": 1.8,
    "合同": 1.8,
    "签约": 1.6,
    "突破": 1.4,
    "创新高": 1.4,
    "并购": 2.0,
    "收购": 2.0,
    "重组": 2.2,
    "政策": 1.6,
    "补贴": 1.6,
    "利好": 1.8,
    "业绩预增": 2.0,
    "扭亏": 2.1,
    "增持": 1.8,
    "回购": 1.8,
    "扩产": 1.6,
    "投产": 1.6,
    "获批": 2.0,
}


def _extract_keywords(text: str) -> List[str]:
    kws = [k for k in POSITIVE_KEYWORDS if k in text]
    return kws[:5]


def _prefilter_daily_news(day_df: pd.DataFrame) -> pd.DataFrame:
    """温和预过滤：优先保留利好/硬事件，剔除明显盘面噪音。"""
    if day_df.empty:
        return day_df

    df = day_df.copy().reset_index(drop=True)
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    summary = df.get("summary", pd.Series("", index=df.index)).fillna("").astype(str)
    text = (title + " " + summary).str.strip()

    positive_mask = text.str.contains("|".join(re.escape(k) for k in POSITIVE_KEYWORDS), case=False, regex=True, na=False)
    noise_mask = text.str.contains("|".join(re.escape(k) for k in NOISE_HINTS), case=False, regex=True, na=False)
    negative_mask = text.str.contains("|".join(re.escape(k) for k in NEGATIVE_HINTS), case=False, regex=True, na=False)
    source_mask = df.get("stockList", pd.Series("", index=df.index)).fillna("").astype(str).str.len() > 2

    # 先强保留高价值信号，避免误杀真正的利好。
    force_keep = positive_mask | source_mask | negative_mask
    keep_ratio = float(os.getenv("LLM_PREFILTER_KEEP_RATIO", "0.50"))
    keep_ratio = max(0.30, min(0.90, keep_ratio))

    target_count = max(120, int(round(len(df) * keep_ratio)))
    kept = df[force_keep].copy()

    # 先丢掉明显噪音，再按时间补齐到目标数量。
    remainder = df[~force_keep & ~noise_mask].copy()
    if len(kept) < target_count and not remainder.empty:
        remainder = remainder.sort_values("showTime", ascending=False)
        kept = pd.concat([kept, remainder.head(target_count - len(kept))], ignore_index=True)

    if len(kept) > target_count:
        # 如果强保留超过目标，则优先保留更像硬新闻的条目。
        kept = kept.sort_values("showTime", ascending=False).head(target_count).reset_index(drop=True)
    else:
        kept = kept.sort_values("showTime", ascending=False).reset_index(drop=True)

    kept = kept.drop_duplicates(subset=[c for c in ["code", "showTime"] if c in kept.columns], keep="first")
    return kept


def _calc_strength(text: str, kws: List[str]) -> int:
    score = 4.5
    score += min(len(kws), 3) * 0.8
    score += sum(KEYWORD_WEIGHTS.get(k, 1.0) for k in kws)
    if any(n in text for n in NEGATIVE_HINTS):
        score -= 2.0
    return int(max(1, min(10, round(score))))


def _infer_level_1(text: str) -> str:
    if any(k in text for k in ["政策", "补贴", "监管", "办法", "通知"]):
        return "宏观政策类事件"
    if any(k in text for k in ["并购", "收购", "重组", "回购", "增持", "业绩"]):
        return "公司类事件"
    if any(k in text for k in ["中标", "订单", "合同", "签约", "投产", "扩产", "获批"]):
        return "行业景气事件"
    return "公司类事件"


def _infer_cycle(text: str) -> str:
    if any(k in text for k in ["政策", "并购", "重组", "扩产", "投产"]):
        return "中期型"
    if any(k in text for k in ["获批", "业绩预增", "回购"]):
        return "长尾型"
    return "脉冲型"


def _extract_a_codes(stock_list: List[str]) -> List[str]:
    codes = []
    for item in stock_list or []:
        m = re.search(r"\.(\d{6})$", str(item))
        if not m:
            continue
        code = m.group(1)
        if code[0] in {"0", "3", "6", "8"}:
            codes.append(code)
    return sorted(set(codes))


def _is_a_share_code(code: str) -> bool:
    c = str(code).strip().zfill(6)
    return c.startswith(("000", "001", "002", "003", "004", "300", "301", "600", "601", "603", "605", "688", "689", "8"))


def fetch_em_news_window(start_dt: datetime, end_dt: datetime, max_pages: int = 120, retry: int = 4) -> pd.DataFrame:
    rows: List[Dict] = []
    cursor = ""

    for page in range(1, max_pages + 1):
        params = dict(EM_PARAMS)
        params["sortEnd"] = cursor

        payload = None
        for _ in range(retry):
            try:
                resp = requests.get(EM_URL, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception:
                time.sleep(1.5)

        if payload is None:
            print(f"警告: 第 {page} 页连续请求失败，终止翻页")
            break

        page_rows = payload.get("data", {}).get("fastNewsList", [])
        if not page_rows:
            print(f"翻页结束: 第 {page} 页无数据")
            break

        rows.extend(page_rows)

        oldest = pd.to_datetime(page_rows[-1].get("showTime", ""), errors="coerce")
        if pd.notna(oldest) and oldest <= pd.Timestamp(start_dt):
            print(f"已回溯到目标起点: 第 {page} 页最旧时间 {oldest}")
            break

        cursor = str(page_rows[-1].get("realSort", ""))
        if not cursor:
            print("翻页结束: 缺少 realSort 游标")
            break

        if page % 10 == 0:
            newest = pd.to_datetime(page_rows[0].get("showTime", ""), errors="coerce")
            print(f"已抓取到第 {page} 页, 时间范围示例: {newest} ~ {oldest}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["showTime"] = pd.to_datetime(df["showTime"], errors="coerce")
    df = df.dropna(subset=["showTime"])
    df = df[(df["showTime"] >= pd.Timestamp(start_dt)) & (df["showTime"] <= pd.Timestamp(end_dt))]
    df = df.sort_values("showTime").drop_duplicates(subset=["code", "showTime"], keep="first")
    df = df.reset_index(drop=True)
    return df


def _to_tx_symbol(code: str) -> Optional[str]:
    c = str(code).strip().zfill(6)
    if c.startswith("6"):
        return f"sh{c}"
    if c.startswith(("0", "3")):
        return f"sz{c}"
    if c.startswith(("8", "4")):
        return f"bj{c}"
    return None


def _fetch_week_return_online(code: str, week_start: pd.Timestamp):
    symbol = _to_tx_symbol(code)
    if not symbol:
        return None

    buy_date = (week_start + pd.Timedelta(days=1)).normalize()
    sell_date = (week_start + pd.Timedelta(days=4)).normalize()
    start = (buy_date - pd.Timedelta(days=3)).strftime("%Y%m%d")
    end = (sell_date + pd.Timedelta(days=3)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust='')
        if df is None or df.empty:
            return None

        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d["open"] = pd.to_numeric(d.get("open", pd.NA), errors="coerce")
        d["close"] = pd.to_numeric(d.get("close", pd.NA), errors="coerce")
        d = d.dropna(subset=["date", "open", "close"])

        buy_row = d[d["date"] == buy_date]
        sell_row = d[d["date"] == sell_date]
        if buy_row.empty or sell_row.empty:
            return None

        tuesday_open = float(buy_row.iloc[0]["open"])
        friday_close = float(sell_row.iloc[0]["close"])
        if tuesday_open <= 0:
            return None

        weekly_return = (friday_close - tuesday_open) / tuesday_open
        return {
            "buy_date": buy_date,
            "sell_date": sell_date,
            "tuesday_open": tuesday_open,
            "friday_close": friday_close,
            "weekly_return": weekly_return,
        }
    except Exception:
        return None


def fetch_week_return_with_fallback(price_df: pd.DataFrame, code: str, week_start: pd.Timestamp):
    info = fetch_week_return(price_df, code, week_start)
    if info is not None:
        info["price_source"] = "local"
        return info

    online = _fetch_week_return_online(code, week_start)
    if online is not None:
        online["price_source"] = "online"
        return online

    return None


def build_events(df: pd.DataFrame, max_events_per_day: int = 120) -> List[Dict]:
    if df.empty:
        return []

    df = df.copy()
    df["day"] = df["showTime"].dt.date

    events_by_day: Dict = {}

    for _, row in df.iterrows():
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        text = f"{title} {summary}".strip()
        kws = _extract_keywords(text)
        if not kws:
            continue

        related_codes = _extract_a_codes(row.get("stockList") or [])

        event = {
            "date": row["showTime"].strftime("%Y-%m-%d %H:%M:%S"),
            "event_level_1": _infer_level_1(text),
            "event_level_2": (title[:40] if title else kws[0]),
            "sentiment": "positive" if not any(n in text for n in NEGATIVE_HINTS) else "neutral",
            "impact_cycle": _infer_cycle(text),
            "predictability": "突发型",
            "strength": _calc_strength(text, kws),
            "related_concepts": kws[:3],
            "related_companies": related_codes,
        }
        events_by_day.setdefault(row["day"], []).append(event)

    final_events: List[Dict] = []
    for day in sorted(events_by_day.keys()):
        day_events = events_by_day[day]
        uniq = {}
        for e in day_events:
            key = e["event_level_2"]
            old = uniq.get(key)
            if old is None or e["strength"] > old["strength"]:
                uniq[key] = e
        dedup = list(uniq.values())
        dedup.sort(key=lambda x: x["strength"], reverse=True)
        final_events.extend(dedup[:max_events_per_day])

    return final_events


def _normalize_llm_event(raw_event: Dict, row: pd.Series) -> Dict:
    title = str(row.get("title") or "").strip()
    ts = pd.to_datetime(row.get("showTime"), errors="coerce")
    related_codes = _extract_a_codes(row.get("stockList") or [])

    e = raw_event if isinstance(raw_event, dict) else {}
    sentiment = str(e.get("sentiment", "neutral")).strip().lower()
    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "neutral"

    level_2 = str(e.get("event_level_2") or e.get("event_title") or title[:40] or "LLM事件").strip()
    if not level_2:
        level_2 = "LLM事件"

    try:
        strength = float(e.get("strength", 0))
    except Exception:
        strength = 0.0

    llm_concepts = e.get("related_concepts", [])
    if not isinstance(llm_concepts, list):
        llm_concepts = []
    llm_concepts = [str(x).strip() for x in llm_concepts if str(x).strip()][:3]

    llm_companies = e.get("related_companies", [])
    if not isinstance(llm_companies, list):
        llm_companies = []
    llm_codes = []
    for x in llm_companies:
        code = re.sub(r"\D", "", str(x or "")).zfill(6)
        if len(code) == 6 and code.isdigit() and _is_a_share_code(code):
            llm_codes.append(code)

    if not llm_codes:
        llm_codes = related_codes

    event_date = ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "date": event_date,
        "event_level_1": str(e.get("event_level_1") or "公司类事件").strip(),
        "event_level_2": level_2[:60],
        "sentiment": sentiment,
        "impact_cycle": str(e.get("impact_cycle") or "脉冲型").strip(),
        "predictability": str(e.get("predictability") or "突发型").strip(),
        "strength": strength,
        "related_concepts": llm_concepts,
        "related_companies": sorted(set(llm_codes)),
    }


def build_events_with_llm(
    df: pd.DataFrame,
    batch_size: int = 20,
    delay_sec: float = 0.15,
    llm_provider: str = "auto",
    raw_artifact_path: Optional[Path] = None,
) -> List[Dict]:
    """使用 llm_api_filter_date.call_gemini 进行事件抽取（不走本地规则打分）。"""
    if df.empty:
        return []

    work_df = df.copy().sort_values("showTime").reset_index(drop=True)
    print(f"LLM输入条数(新闻窗口原始): {len(work_df)}")

    max_items = int(os.getenv("LLM_MAX_ITEMS", "0"))
    if max_items > 0 and len(work_df) > max_items:
        work_df = work_df.tail(max_items).reset_index(drop=True)
        print(f"LLM处理条数过多，已截断到最近 {max_items} 条")

    if work_df.empty:
        return []

    total_items = int(len(work_df))
    total_non_empty = 0
    truncated_items = 0
    llm_attempted = 0
    fallback_defaults = 0
    cache_hits = 0
    circuit_trips = 0
    consecutive_failures = 0

    circuit_fail_threshold = max(1, int(os.getenv("LLM_CIRCUIT_FAILS", "12")))
    circuit_cooldown_sec = max(1.0, float(os.getenv("LLM_CIRCUIT_COOLDOWN_SEC", "25")))
    enable_provider_fallback = os.getenv("LLM_ENABLE_PROVIDER_FALLBACK", "1").strip() != "0"

    enable_cache = os.getenv("LLM_CACHE_ENABLED", "1").strip() != "0"
    llm_text_max_chars = max(200, int(os.getenv("LLM_TEXT_MAX_CHARS", "600")))
    llm_batch_char_budget = max(800, int(os.getenv("LLM_BATCH_CHAR_BUDGET", "4500")))
    llm_cache: Dict[str, Dict] = {}
    llm_cache_dirty = False
    raw_batches: List[Dict] = []

    if enable_cache:
        try:
            if LLM_CACHE_FILE.exists():
                loaded = json.loads(LLM_CACHE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    llm_cache = loaded
        except Exception:
            llm_cache = {}

    def _persist_llm_cache() -> None:
        nonlocal llm_cache_dirty
        if not enable_cache or not llm_cache_dirty:
            return
        try:
            LLM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            LLM_CACHE_FILE.write_text(json.dumps(llm_cache, ensure_ascii=False), encoding="utf-8")
            llm_cache_dirty = False
        except Exception:
            pass

    def _llm_cache_key(payload_item: Dict[str, str]) -> str:
        base = f"{llm_provider}|{payload_item.get('date','')}|{payload_item.get('text','')}"
        return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()

    def _call_with_fallback(payload_items: List[Dict[str, str]]) -> Optional[List[Dict]]:
        nonlocal consecutive_failures
        try:
            parsed = call_llm(payload_items, provider=llm_provider)
        except InsufficientBalanceError:
            if enable_provider_fallback and str(llm_provider).lower() == "deepseek":
                parsed = call_llm(payload_items, provider="gemini")
                if parsed:
                    consecutive_failures = 0
                    return parsed
            raise

        if parsed:
            consecutive_failures = 0
            return parsed

        if enable_provider_fallback and str(llm_provider).lower() == "deepseek":
            parsed = call_llm(payload_items, provider="gemini")
            if parsed:
                consecutive_failures = 0
                return parsed

        consecutive_failures += 1
        return None

    def _safe_extract_batch(batch_payload: List[Dict[str, str]], batch_rows: List[int]) -> List[Dict]:
        nonlocal fallback_defaults, circuit_trips, llm_cache_dirty
        out_events: List[Dict] = []
        if consecutive_failures >= circuit_fail_threshold:
            circuit_trips += 1
            print(
                f"LLM熔断触发: 连续失败={consecutive_failures}，冷却 {circuit_cooldown_sec:.1f}s 后继续"
            )
            time.sleep(circuit_cooldown_sec)

        parsed = _call_with_fallback(batch_payload)
        raw_batches.append({
            "batch_size": len(batch_payload),
            "input_items": batch_payload,
            "raw_output": parsed,
        })

        if isinstance(parsed, list) and len(parsed) == len(batch_payload):
            for item, e, ridx in zip(batch_payload, parsed, batch_rows):
                normalized = _normalize_llm_event(e, work_df.iloc[ridx])
                out_events.append(normalized)
                if enable_cache:
                    llm_cache[_llm_cache_key(item)] = normalized
                    llm_cache_dirty = True
            return out_events

        # 兼容模型返回条数不一致或整批不可解析：逐条降级重试，保证每条都被处理。
        if isinstance(parsed, list) and len(parsed) > 0:
            for item, e, ridx in zip(batch_payload, parsed, batch_rows):
                normalized = _normalize_llm_event(e, work_df.iloc[ridx])
                out_events.append(normalized)
                if enable_cache:
                    llm_cache[_llm_cache_key(item)] = normalized
                    llm_cache_dirty = True
            start_idx = len(out_events)
        else:
            start_idx = 0

        if start_idx < len(batch_payload):
            for item, ridx in zip(batch_payload[start_idx:], batch_rows[start_idx:]):
                try:
                    if consecutive_failures >= circuit_fail_threshold:
                        circuit_trips += 1
                        print(
                            f"LLM熔断触发: 连续失败={consecutive_failures}，冷却 {circuit_cooldown_sec:.1f}s 后继续"
                        )
                        time.sleep(circuit_cooldown_sec)
                    single = _call_with_fallback([item])
                except InsufficientBalanceError:
                    raise
                if isinstance(single, list) and len(single) > 0:
                    normalized = _normalize_llm_event(single[0], work_df.iloc[ridx])
                    out_events.append(normalized)
                    if enable_cache:
                        llm_cache[_llm_cache_key(item)] = normalized
                        llm_cache_dirty = True
                else:
                    fallback_defaults += 1
                    normalized = _normalize_llm_event({}, work_df.iloc[ridx])
                    out_events.append(normalized)
                    if enable_cache:
                        llm_cache[_llm_cache_key(item)] = normalized
                        llm_cache_dirty = True
                time.sleep(min(0.3, max(delay_sec / 3.0, 0.05)))

        return out_events

    batch_items: List[Dict[str, str]] = []
    row_idx_batch: List[int] = []
    batch_chars = 0
    events: List[Dict] = []

    for idx, row in work_df.iterrows():
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        text = f"{title} {summary}".strip()
        if not text:
            continue
        if len(text) > llm_text_max_chars:
            text = text[:llm_text_max_chars]
            truncated_items += 1
        total_non_empty += 1
        payload_item = {"text": text, "date": str(pd.to_datetime(row.get("showTime"), errors="coerce"))}
        if enable_cache:
            k = _llm_cache_key(payload_item)
            cached = llm_cache.get(k)
            if isinstance(cached, dict):
                events.append(_normalize_llm_event(cached, row))
                cache_hits += 1
                continue

        # 自适应切批：同时控制“条数”和“文本总长度”，避免大批量长文本请求超时。
        item_chars = len(payload_item["text"])
        if batch_items and (
            len(batch_items) >= batch_size
            or (batch_chars + item_chars) > llm_batch_char_budget
        ):
            events.extend(_safe_extract_batch(batch_items, row_idx_batch))
            llm_attempted += len(batch_items)
            batch_items = []
            row_idx_batch = []
            batch_chars = 0
            time.sleep(delay_sec)
            _persist_llm_cache()

        batch_items.append(payload_item)
        row_idx_batch.append(idx)
        batch_chars += item_chars

    if batch_items:
        events.extend(_safe_extract_batch(batch_items, row_idx_batch))
        llm_attempted += len(batch_items)
        _persist_llm_cache()

    print(
        f"LLM分批统计: 窗口总条数={total_items}, 有效文本={total_non_empty}, 已送LLM={llm_attempted}, "
        f"缓存命中={cache_hits}, 截断条数={truncated_items}, 默认回填={fallback_defaults}, 熔断次数={circuit_trips}"
    )

    _persist_llm_cache()

    if raw_artifact_path is not None:
        try:
            raw_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(raw_artifact_path, {
                "window_start": str(work_df["showTime"].min()) if "showTime" in work_df.columns and len(work_df) > 0 else None,
                "window_end": str(work_df["showTime"].max()) if "showTime" in work_df.columns and len(work_df) > 0 else None,
                "provider": llm_provider,
                "batch_size": batch_size,
                "total_items": total_items,
                "valid_items": total_non_empty,
                "truncated_items": truncated_items,
                "fallback_defaults": fallback_defaults,
                "raw_batches": raw_batches,
            })
        except Exception:
            pass

    dedup = {}
    for e in events:
        key = str(e.get("event_level_2", "")).strip() or str(e.get("date", ""))
        old = dedup.get(key)
        if old is None or float(e.get("strength", 0)) > float(old.get("strength", 0)):
            dedup[key] = e

    out = sorted(dedup.values(), key=lambda x: (float(x.get("strength", 0)), x.get("date", "")), reverse=True)
    return out


def rebuild_structured_db(events: List[Dict], db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS event_stocks")
    cur.execute("DROP TABLE IF EXISTS events")

    cur.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            news_title TEXT,
            event_category TEXT,
            event_sentiment TEXT,
            keywords TEXT,
            publish_time TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE event_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_title TEXT,
            event_id INTEGER,
            stock_code TEXT,
            stock_name TEXT,
            match_reason TEXT,
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
        """
    )

    source_map_count = 0
    keyword_map_count = 0
    events_with_source = 0
    events_with_keyword = 0

    for i, event in enumerate(events, 1):
        title = event.get("event_level_2", "")
        keywords = event.get("related_concepts", []) or []

        cur.execute(
            """
            INSERT INTO events (id, news_title, event_category, event_sentiment, keywords, publish_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                i,
                title,
                event.get("event_level_1", ""),
                event.get("sentiment", "neutral"),
                json.dumps(keywords, ensure_ascii=False),
                event.get("date", ""),
            ),
        )

        inserted_codes = set()
        source_added_this_event = 0
        keyword_added_this_event = 0

        for code in event.get("related_companies", []) or []:
            code = str(code).strip().zfill(6)
            if len(code) == 6 and code.isdigit() and _is_a_share_code(code) and code not in inserted_codes:
                cur.execute(
                    """
                    INSERT INTO event_stocks (news_title, event_id, stock_code, stock_name, match_reason)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (title, i, code, "", "source_stock_list"),
                )
                inserted_codes.add(code)
                source_added_this_event += 1

        try:
            matched = match_stocks(
                keywords[:3],
                require_concept_match=True,
                min_keyword_hits=2,
            ) if keywords else []
            match_reason = "keyword_match_strict"

            if not matched and keywords:
                matched = match_stocks(
                    keywords[:3],
                    require_concept_match=True,
                    min_keyword_hits=1,
                )
                match_reason = "keyword_match_relaxed"

            if not matched and keywords:
                matched = match_stocks(
                    keywords[:3],
                    require_concept_match=False,
                    min_keyword_hits=1,
                )
                match_reason = "keyword_match_name_fallback"

            for s in matched:
                code = str((s or {}).get("code", "")).strip()
                code = re.sub(r"\D", "", code).zfill(6)
                if len(code) == 6 and code.isdigit() and _is_a_share_code(code) and code not in inserted_codes:
                    cur.execute(
                        """
                        INSERT INTO event_stocks (news_title, event_id, stock_code, stock_name, match_reason)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                            (title, i, code, str((s or {}).get("name", "")), match_reason),
                    )
                    inserted_codes.add(code)
                    keyword_added_this_event += 1
        except Exception:
            pass

        source_map_count += source_added_this_event
        keyword_map_count += keyword_added_this_event
        if source_added_this_event > 0:
            events_with_source += 1
        if keyword_added_this_event > 0:
            events_with_keyword += 1

    conn.commit()
    conn.close()
    print(
        "映射统计: "
        f"source映射 {source_map_count} 条 / 覆盖事件 {events_with_source}; "
        f"keyword映射 {keyword_map_count} 条 / 覆盖事件 {events_with_keyword}"
    )


def run_pipeline(
    start_date: str,
    end_date: str,
    capital: float = DEFAULT_CAPITAL,
    db_path: Path = DEFAULT_DB_PATH,
    price_file: Path = DEFAULT_PRICE_FILE,
    event_source: str = "llm",
    llm_provider: str = "deepseek",
) -> None:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    print("=" * 70)
    print("东方财富可回溯历史快讯: 抓取 + 选股 + 动态分配")
    print("=" * 70)
    print(f"新闻窗口: {start_dt} ~ {end_dt}")

    em_max_pages = int(os.getenv("EM_MAX_PAGES", "120"))
    raw_df = fetch_em_news_window(start_dt=start_dt, end_dt=end_dt, max_pages=em_max_pages)
    print(f"窗口内快讯条数: {len(raw_df)}")
    if raw_df.empty:
        raise ValueError("目标窗口内未抓到新闻数据。")

    if event_source == "llm":
        print(f"事件抽取方式: LLM ({llm_provider})")
        llm_batch_size = int(os.getenv("LLM_BATCH_SIZE", "20"))
        llm_delay_sec = float(os.getenv("LLM_DELAY_SEC", "0.15"))
        llm_last_day_max_items = max(0, int(os.getenv("LLM_LAST_DAY_MAX_ITEMS", "0")))
        llm_last_day_keywords = [
            x.strip() for x in str(os.getenv("LLM_LAST_DAY_KEYWORDS", "")).split(",") if x.strip()
        ]
        llm_last_day_keep_latest = os.getenv("LLM_LAST_DAY_KEEP_LATEST", "1").strip() != "0"
        llm_raw_file = EVENTS_FILE.parent / f"llm_events_raw.{start_date}_{end_date}.json"
        llm_normalized_file = EVENTS_FILE.parent / f"llm_events_normalized.{start_date}_{end_date}.json"

        def _dedup_events(all_events: List[Dict]) -> List[Dict]:
            dedup = {}
            for e in all_events:
                key = str(e.get("event_level_2", "")).strip() or str(e.get("date", ""))
                old = dedup.get(key)
                if old is None or float(e.get("strength", 0)) > float(old.get("strength", 0)):
                    dedup[key] = e
            return sorted(dedup.values(), key=lambda x: (float(x.get("strength", 0)), x.get("date", "")), reverse=True)

        partial_file = EVENTS_FILE.parent / f"events_em_history.partial.{start_date}_{end_date}.json"
        partial_state = {
            "window_start": start_date,
            "window_end": end_date,
            "provider": llm_provider,
            "processed_days": [],
            "events": [],
        }

        if partial_file.exists():
            try:
                old_state = json.loads(partial_file.read_text(encoding="utf-8"))
                if (
                    old_state.get("window_start") == start_date
                    and old_state.get("window_end") == end_date
                    and old_state.get("provider") == llm_provider
                ):
                    partial_state = old_state
                    print(
                        f"检测到断点文件，已续跑: 已处理 {len(partial_state.get('processed_days', []))} 天, "
                        f"累计事件 {len(partial_state.get('events', []))}"
                    )
            except Exception:
                pass

        work_df = raw_df.copy()
        work_df["day"] = work_df["showTime"].dt.strftime("%Y-%m-%d")
        all_days = sorted([d for d in work_df["day"].dropna().unique().tolist() if str(d).strip()])

        processed_days = set(str(x) for x in partial_state.get("processed_days", []))
        events = list(partial_state.get("events", []))

        for i, day in enumerate(all_days, 1):
            if day in processed_days:
                print(f"LLM按日处理 [{i}/{len(all_days)}] {day}: 已完成，跳过")
                continue

            day_df = work_df[work_df["day"] == day].drop(columns=["day"], errors="ignore").reset_index(drop=True)
            original_count = len(day_df)

            if os.getenv("LLM_PREFILTER_ENABLED", "1").strip() != "0" and len(day_df) > 0:
                pref_df = _prefilter_daily_news(day_df)
                if len(pref_df) < len(day_df):
                    print(f"日内预过滤: {day} {len(day_df)} -> {len(pref_df)} 条")
                day_df = pref_df

            if i == len(all_days) and (llm_last_day_max_items > 0 or llm_last_day_keywords):
                filtered_df = day_df

                if llm_last_day_keywords:
                    pattern = "|".join(re.escape(k) for k in llm_last_day_keywords)
                    text_series = (
                        filtered_df.get("title", pd.Series("", index=filtered_df.index)).fillna("").astype(str)
                        + " "
                        + filtered_df.get("summary", pd.Series("", index=filtered_df.index)).fillna("").astype(str)
                    )
                    keyword_mask = text_series.str.contains(pattern, case=False, regex=True, na=False)
                    matched_count = int(keyword_mask.sum())
                    print(
                        f"最后一天应急过滤(关键词): 关键词数={len(llm_last_day_keywords)}, 命中={matched_count}/{len(filtered_df)}"
                    )
                    if matched_count > 0:
                        filtered_df = filtered_df[keyword_mask].reset_index(drop=True)

                if llm_last_day_max_items > 0 and len(filtered_df) > llm_last_day_max_items:
                    if llm_last_day_keep_latest and "showTime" in filtered_df.columns:
                        filtered_df = filtered_df.sort_values("showTime").tail(llm_last_day_max_items).reset_index(drop=True)
                    else:
                        filtered_df = filtered_df.head(llm_last_day_max_items).reset_index(drop=True)
                    print(f"最后一天应急过滤(上限): 已限制到 {len(filtered_df)} 条")

                if len(filtered_df) > 0:
                    day_df = filtered_df
                    if len(day_df) != original_count:
                        print(f"最后一天应急过滤完成: {original_count} -> {len(day_df)} 条")
                else:
                    print("最后一天应急过滤后为空，已回退到原始数据，避免任务中断")

            print(f"LLM按日处理 [{i}/{len(all_days)}] {day}: 新闻 {len(day_df)} 条")
            try:
                day_events = build_events_with_llm(
                    day_df,
                    batch_size=llm_batch_size,
                    delay_sec=llm_delay_sec,
                    llm_provider=llm_provider,
                    raw_artifact_path=llm_raw_file,
                )
            except InsufficientBalanceError as e:
                print(f"检测到余额不足，已立即停止，避免默认回填污染: {e}")
                raise
            events.extend(day_events)
            processed_days.add(day)

            partial_state["processed_days"] = sorted(processed_days)
            partial_state["events"] = events
            _write_json_atomic(partial_file, partial_state)

        events = _dedup_events(events)
        print(f"构建事件条数(LLM抽取后): {len(events)}")

        try:
            _write_json_atomic(llm_normalized_file, {
                "window_start": start_date,
                "window_end": end_date,
                "provider": llm_provider,
                "events": events,
            })
            print(f"LLM归一化快照输出: {llm_normalized_file}")
            print(f"LLM原始批次输出: {llm_raw_file}")
        except Exception as e:
            print(f"LLM快照写入失败: {e}")

        if partial_file.exists():
            try:
                partial_file.unlink()
            except Exception:
                pass
    else:
        print("事件抽取方式: 本地规则")
        events = build_events(raw_df, max_events_per_day=120)
        print(f"构建事件条数(关键词过滤后): {len(events)}")

    if not events:
        raise ValueError("窗口内新闻未匹配到可用利好关键词，无法继续。")

    _write_json_atomic(EVENTS_FILE, events)
    print(f"事件文件输出: {EVENTS_FILE}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    rebuild_structured_db(events, db_path=db_path)
    print(f"结构化数据库重建完成: {db_path}")

    end_day = pd.Timestamp(end_dt.date())
    trade_week_start = end_day - pd.Timedelta(days=end_day.weekday())
    alloc_file = RESULT_DIR / f"em_weekly_recommendation_{trade_week_start.date()}.csv"
    rf = RiskFilter(
        db_path=str(db_path),
        json_file=str(EVENTS_FILE),
        price_data_file=str(price_file),
        trade_week_start=trade_week_start,
        event_window_start=start_dt.strftime("%Y-%m-%d"),
        event_window_end=end_dt.strftime("%Y-%m-%d"),
    )
    candidates = rf.run()
    if candidates is None or len(candidates) == 0:
        raise ValueError("风险过滤后无可交易股票。")

    # 兜底保护: 最终结果禁止 ST 股票。
    candidates = candidates.copy()
    candidates["code"] = candidates["code"].astype(str).str.zfill(6)
    candidates["name"] = candidates["name"].fillna("").astype(str)

    def _is_st_stock_name(name: str) -> bool:
        s = str(name or "").upper().replace(" ", "")
        return "ST" in s

    non_st = candidates[~candidates["name"].apply(_is_st_stock_name)].copy()
    removed_st = len(candidates) - len(non_st)
    if removed_st > 0:
        print(f"最终兜底过滤: 已剔除ST股票 {removed_st} 只")

    if non_st.empty:
        raise ValueError("最终兜底过滤后无可交易股票（全部为ST）。")

    top3 = non_st.head(3).copy()
    if len(top3) < 3:
        print(f"警告: 剔除ST后仅剩 {len(top3)} 只股票，少于3只。")

    weight_score_beta = float(os.getenv("RW_SCORE_BETA", "1.0"))
    weight_mix_lambda = float(os.getenv("RW_MIX_LAMBDA", "0.5"))
    weight_mix_lambda = max(0.0, min(1.0, weight_mix_lambda))
    weight_min = float(os.getenv("RW_MIN_WEIGHT", "0.10"))
    weight_max = float(os.getenv("RW_MAX_WEIGHT", "0.40"))
    if weight_min > weight_max:
        weight_min, weight_max = weight_max, weight_min

    weighted = compute_dynamic_weights(
        top3,
        score_beta=weight_score_beta,
        mix_lambda=weight_mix_lambda,
        min_weight=weight_min,
        max_weight=weight_max,
    ).copy()

    weighted["trade_week_start"] = trade_week_start.date()
    weighted["buy_date"] = (trade_week_start + pd.Timedelta(days=1)).date()
    weighted["sell_date"] = (trade_week_start + pd.Timedelta(days=4)).date()
    weighted["allocated_amount"] = weighted["weight"] * float(capital)
    weighted["selection_reason"] = weighted.apply(
        lambda r: (
            f"由事件《{str(r.get('event_title', '')).strip()}》触发; "
            f"强度={float(r.get('strength', 0)):.2f}; "
            f"综合分={float(r.get('composite_score', 0)):.2f}"
        ),
        axis=1,
    )

    price_df = pd.read_csv(price_file, encoding="utf-8")
    price_df = price_df.rename(columns={"代码": "code", "日期": "date", "开盘": "open", "收盘": "close"})
    price_df["code"] = price_df["code"].astype(str).str.split(".").str[0].str.zfill(6)
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    price_df["open"] = pd.to_numeric(price_df["open"], errors="coerce")
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df = price_df.dropna(subset=["date", "open", "close"])

    week_return = 0.0
    for idx, row in weighted.iterrows():
        info = fetch_week_return_with_fallback(price_df, row["code"], trade_week_start)
        if info is None:
            weighted.loc[idx, "weekly_return"] = None
            weighted.loc[idx, "price_source"] = "missing"
            continue
        r = float(info["weekly_return"])
        weighted.loc[idx, "weekly_return"] = r
        weighted.loc[idx, "tuesday_open"] = float(info["tuesday_open"])
        weighted.loc[idx, "friday_close"] = float(info["friday_close"])
        weighted.loc[idx, "price_source"] = info.get("price_source", "local")
        week_return += float(row["weight"]) * r

    weighted["final_capital"] = float(capital) * (1.0 + week_return)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "trade_week_start", "buy_date", "sell_date", "code", "name", "event_title", "event_date",
        "selection_reason",
        "strength", "composite_score", "vol_20d", "score_weight", "risk_weight", "weight",
        "allocated_amount", "tuesday_open", "friday_close", "weekly_return", "price_source", "final_capital",
    ]
    for c in keep_cols:
        if c not in weighted.columns:
            weighted[c] = None
    out_df = weighted[keep_cols].sort_values("weight", ascending=False).reset_index(drop=True)
    out_df.to_csv(alloc_file, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Top3 动态分配结果")
    print("=" * 70)
    for i, row in out_df.iterrows():
        print(f"{i+1}. {row['code']} {row['name']} | 仓位 {float(row['weight']):.2%} | 金额 {float(row['allocated_amount']):.2f}")
    print(
        f"组合周收益({(trade_week_start + pd.Timedelta(days=1)).date()}买入-"
        f"{(trade_week_start + pd.Timedelta(days=4)).date()}卖出): {week_return:.2%}"
    )
    print(f"期末资金: {float(capital) * (1.0 + week_return):.2f}")
    print(f"输出文件: {alloc_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输入起止日期后，自动抓新闻+调用LLM+筛选Top3并动态分配")
    parser.add_argument("--start-date", type=str, required=True, help="新闻窗口开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="新闻窗口结束日期 YYYY-MM-DD")
    parser.add_argument("--llm-provider", type=str, default="auto", choices=["auto", "deepseek", "zhipu", "gemini"], help="LLM提供方，默认auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        capital=DEFAULT_CAPITAL,
        db_path=DEFAULT_DB_PATH,
        price_file=DEFAULT_PRICE_FILE,
        event_source="llm",
        llm_provider=args.llm_provider,
    )


if __name__ == "__main__":
    main()
