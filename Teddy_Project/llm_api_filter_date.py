import json
import os
import re
import sqlite3
import time
import ast
from typing import Dict, List, Optional

from google import genai
import requests
from tqdm import tqdm


class InsufficientBalanceError(RuntimeError):
    """Raised when the upstream LLM provider reports insufficient balance."""

# Single key config kept from existing file behavior.
API_KEY = "  "
MODEL = "gemini-2.5-flash-lite"
# 在这里直接粘贴你的 DeepSeek API Key
DEEPSEEK_API_KEY_HARDCODE = "输入apikey"


def _deepseek_runtime_config() -> Dict[str, object]:
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", "") or DEEPSEEK_API_KEY_HARDCODE,
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "connect_timeout": int(os.getenv("DEEPSEEK_CONNECT_TIMEOUT", "15")),
        "read_timeout": int(os.getenv("DEEPSEEK_READ_TIMEOUT", "90")),
        "max_retries": int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
    }


def _zhipu_runtime_config() -> Dict[str, object]:
    return {
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn"),
        "model": os.getenv("ZHIPU_MODEL", "glm-4-flash"),
        "connect_timeout": int(os.getenv("ZHIPU_CONNECT_TIMEOUT", "15")),
        "read_timeout": int(os.getenv("ZHIPU_READ_TIMEOUT", "90")),
        "max_retries": int(os.getenv("ZHIPU_MAX_RETRIES", "3")),
    }

KEYWORDS = [
    "涨停", "跌停", "中标", "订单", "合同", "签约",
    "突破", "创新高", "并购", "收购", "重组",
    "政策", "补贴", "利好", "利空",
    "业绩预增", "业绩预减", "扭亏",
    "增持", "减持", "回购",
    "停牌", "复牌", "解禁",
    "扩产", "投产", "融资",
]
KEYWORD_PATTERN = re.compile("|".join(KEYWORDS))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "db", "master_data.db")
TABLE_NAME = "market_news"
TEXT_COLUMN = "content"
DATE_COLUMN = "publish_time"

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "events", "llm_events_date.json")
BATCH_SIZE = 50
DELAY = 8

client = genai.Client(api_key=API_KEY)

PROMPT_TEMPLATE = """
Role
你是一位资深的A股量化金融分析师和文本数据挖掘专家。你的任务是阅读金融新闻/公告，提取核心事件，并严格按照【主办方事件分类体系】与【A股长板量化打分模型】提取结构化信息，输出合法的 JSON 格式。
Background Knowledge (主办方分类体系与打分矩阵)
一、 主办方标准分类字典（强制匹配，严禁自造词汇！）
1.	【一级分类】与【二级分类】映射关系：
•	宏观政策类事件: [货币政策, 财政政策, 国家发展规划, 国际贸易政策, 领导人重要讲话]
•	行业监管类事件: [行业新规, 环保/安全检查, 行业标准制定, 反垄断/监管处罚]
•	公司类事件: [财报发布, 重大资产重组, 管理层变动, 股票回购/分红, 产品发布/召回, 负面舆情, 重大合同签署, 在手订单类, 产品价格类]
•	行业景气事件: [原材料价格波动, 供应链变化, 技术突破, 行业龙头动作]
•	地缘/国际事件: [战争/冲突, 外交制裁/反制, 国际组织决议, 贸易协定签署]
•	宏观数据发布: [经济数据, 就业数据, 消费/投资/出口数据]
•	自然灾害/卫生: [疫情爆发, 自然灾害, 重大事故]
•	无实质事件/市场噪音: [散户评论, 盘面描述] (注：此为系统防噪选项，处理主观股评时使用)
2.	【影响周期】仅限：[脉冲型, 中期型, 长尾型]
3.	【可预测性】仅限：[突发型, 预披露型]
4.	【行业属性】仅限：[宏观, 多行业, 周期类, 新能源类, 科技类, 消费类, 医药类, 军工类]
二、 核心打分公式（长板模型）
最终总分 = 方向 × (加权基础分 + 长板增强分) × 确定性系数 × 位置系数 × 行业系数
1.	预期差 (权重0.4, 1-5分): 5(完全意外)；4(明显超预期)；3(超预期有预热)；2(略超预期)；1(符合预期)。
2.	深度 (权重0.4, 1-5分): 5(利润影响>30%)；4(15-30%)；3(5-15%)；2(<5%)；1(无实质影响)。
3.	持续性 (权重0.2, 1-5分): 5(>3年)；4(1-3年)；3(6-12个月)；2(1-6个月)；1(数天)。
加权基础分 = (预期差×0.4 + 深度×0.4 + 持续性×0.2)
4.	长板增强分 = 0.15 × max(预期差, 深度)
5.	乘数系数:
•	方向: 利好(+1), 利空(-1)
•	确定性: 1.0(已落地), 0.8(政策明确), 0.6(预期), 0.3(传闻)
•	位置: 未提及默认 1.0；明确提及低位(利好1.2, 利空0.5)；高位(利好0.6, 利空1.3)
•	行业阶段与景气: 综合乘数，默认 1.0。景气上行可取 1.05~1.15，衰退期取 0.85~0.98。
Task & Rules
从文本中提取信息，遵循约束填充 JSON：
1. event_title: 凝练核心事件（15字内）。
2. event_level_1, event_level_2, impact_cycle, predictability, industry_attribute: 必须且只能从上述【主办方标准分类字典】中精准摘取一词，绝不可更改一字！
   - 【例外】：如果股评文本中夹带了真实的“中标合同/重组/发财报”，忽略主观废话，直接提取该客观事实的对应分类。
3. sentiment: 评估短期影响，仅限：[positive, negative, neutral]。
   - 🚨 【极性强制判定】：遇到“大股东减持/清仓”、“业绩转亏/下修”、“立案调查/被罚”、“地缘制裁/法案点名”，必须判为 negative！遇到“注销式回购”、“主营产品提价/爆单”、“核心技术量产”，必须判为 positive！
4. **score_breakdown**: 写出各分数、系数的打分理由及最终计算公式。
    【证据优先】：只有出现可核验硬信息（金额、同比比例、产能规模、合同期限、公告主体）时，预期差/深度才可打到4-5分；无量化证据时二者最高3分。
  【确定性保守】：预告/快报/媒体转述/二手摘要默认不高于0.8；仅公司公告、交易所公告、监管文件等一手来源才可取1.0。
  【噪音抑制】：若主要是盘面描述、情绪渲染、题材联想，优先判为 neutral，strength 应接近 0。
5. strength: 根据公式算出的【带有正负号的绝对值数字】（保留2位小数）。
  【负分铁律】：若 sentiment 为 negative，最终 strength 必须为负数（如 -8.50）！若是市场噪音或 neutral，分值强制为 0。
  【上限约束】：strength 默认限制在 [-7.50, 7.50]。只有在“高确定性(=1.0)+量化硬证据+持续性>=4”同时满足时，才允许超过 7.50。
6. related_concepts / related_companies: 最多3个A股概念及提及的客观关联公司 []。
Output Format constraints
绝对要求：只能输出纯 JSON，不能包含 markdown 代码块标记 (如 ```json)。
当输入是单条新闻时，输出一个 JSON 对象；当输入包含多条新闻时，必须输出 JSON 数组，且数组长度必须与输入新闻条数完全一致。
Few-Shot Examples
Example 1 (标准成长期重大利好)
输入文本: "今日，宁德时代正式发布神行超充电池，全球首款采用磷酸铁锂材料并可实现大规模量产的4C超充电池。这标志着动力电池技术迈入新阶段。"
输出:
{
"event_title": "宁德时代首发4C神行超充电池",
"event_level_1": "行业景气事件",
"event_level_2": "技术突破",
"sentiment": "positive",
"impact_cycle": "长尾型",
"predictability": "突发型",
"industry_attribute": "新能源类",
"score_breakdown": "方向: 利好(+1)。预期差:4(明显超预期), 深度:5(技术量产且有明确产品路径), 持续性:4(1-3年)。加权基础分=4*0.4+5*0.4+4*0.2=4.4。长板增强=0.15*max(4,5)=0.75。确定性:1.0(官方发布)。位置:1.0(未提及)。行业:景气上行1.12。总分=1*(4.4+0.75)*1.0*1.0*1.12=5.77分。",
"strength": 5.77,
"related_concepts": ["固态电池", "动力电池", "新能源汽车"],
"related_companies": ["宁德时代"]
}
Example 2 (沙里淘金/合同签署)
输入文本: "周末消息面平静，主线还是电网。重点看这几只：1. 顺钠股份看能否封板。2. 三星医疗：海外订单超9亿，直接催化涨停，基本面扎实。操作上去弱留强。"
输出:
{
"event_title": "三星医疗获超9亿海外订单",
"event_level_1": "公司类事件",
"event_level_2": "重大合同签署",
"sentiment": "positive",
"impact_cycle": "脉冲型",
"predictability": "突发型",
"industry_attribute": "科技类",
"score_breakdown": "方向: 利好(+1)。股评中包含客观硬事件（9亿订单），提取该事实。预期差:4(明显超预期), 深度:4(大额合同), 持续性:2(短期脉冲)。加权基础分=40.4+40.4+20.2=3.6。长板增强=0.3max(4,4)=1.2。确定性:0.9(媒体实证)。位置:1.0(未提及)。行业:1.0(平稳)。总分=1*(3.6+1.2)0.91.0*1.0=4.32分。",
"strength": 4.32,
"related_concepts": ["智能电网", "出海"],
"related_companies": ["三星医疗"]
}
Input
{text}
"""


def call_gemini(batch_items: List[Dict[str, str]]) -> Optional[List[Dict]]:
    news_text = "\n\n".join(
        [f"新闻{i + 1}：{item['text']}" for i, item in enumerate(batch_items)]
    )
    prompt = PROMPT_TEMPLATE.replace("{text}", news_text)

    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        text = (response.text or "").strip().replace("```json", "").replace("```", "")
        return _parse_llm_json_list(text)
    except Exception as e:
        print("API调用失败:", e)
        return None
    except BaseException as e:
        print("API调用中断，已按失败批次处理:", e)
        return None


def _parse_llm_json_list(text: str) -> Optional[List[Dict]]:
    if not text:
        return None
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    l = cleaned.find("[")
    r = cleaned.rfind("]")
    if l >= 0 and r > l:
        candidate = cleaned[l:r + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

    # 兼容场景1: 模型返回多个JSON对象顺序拼接，而不是数组。
    try:
        decoder = json.JSONDecoder()
        seq: List[Dict] = []
        s = cleaned
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                break
            obj, end = decoder.raw_decode(s, i)
            if isinstance(obj, dict):
                seq.append(obj)
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        seq.append(it)
            i = end
        if seq:
            return seq
    except Exception:
        pass

    # 兼容场景2: 模型使用Python字面量风格（单引号、True/False/None）。
    try:
        lit = ast.literal_eval(cleaned)
        if isinstance(lit, dict):
            return [lit]
        if isinstance(lit, list):
            out = [x for x in lit if isinstance(x, dict)]
            if out:
                return out
    except Exception:
        pass

    # 兼容场景3: 返回被包在常见字段中。
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for k in ("data", "result", "results", "items", "events"):
                v = obj.get(k)
                if isinstance(v, list):
                    out = [x for x in v if isinstance(x, dict)]
                    if out:
                        return out
                if isinstance(v, dict):
                    return [v]
    except Exception:
        pass
    return None


def call_deepseek(batch_items: List[Dict[str, str]]) -> Optional[List[Dict]]:
    cfg = _deepseek_runtime_config()
    api_key = str(cfg["api_key"])
    base_url = str(cfg["base_url"])
    model = str(cfg["model"])
    connect_timeout = int(cfg["connect_timeout"])
    read_timeout = int(cfg["read_timeout"])
    max_retries = int(cfg["max_retries"])

    if not api_key:
        print("DeepSeek调用失败: 未设置 DEEPSEEK_API_KEY")
        return None

    news_text = "\n\n".join(
        [f"新闻{i + 1}：{item['text']}" for i, item in enumerate(batch_items)]
    )
    prompt = PROMPT_TEMPLATE.replace("{text}", news_text)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是A股事件抽取助手，只返回纯JSON。单条新闻返回JSON对象；多条新闻返回JSON数组，且数量与输入一致。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            )
            parsed = _parse_llm_json_list(text)
            if parsed:
                return parsed
            print(f"DeepSeek返回内容无法解析为JSON列表，第 {attempt} 次重试")
        except requests.HTTPError as e:
            body = ""
            try:
                body = (e.response.text or "")[:500]
            except Exception:
                body = ""
            body_lower = body.lower()
            if (
                "balance is insufficient" in body_lower
                or "insufficient balance" in body_lower
                or '"code":30001' in body_lower
                or "payment required" in body_lower
            ):
                raise InsufficientBalanceError(f"DeepSeek余额不足: {body}")
            print(f"DeepSeek HTTP错误(第 {attempt} 次): {e}; body={body}")
        except requests.RequestException as e:
            print(f"DeepSeek网络错误(第 {attempt} 次): {e}")
        except Exception as e:
            print(f"DeepSeek调用异常(第 {attempt} 次): {e}")
        except BaseException as e:
            print(f"DeepSeek调用中断(第 {attempt} 次): {type(e).__name__}: {e!r}")
            return None

        if attempt < max_retries:
            time.sleep(min(2 * attempt, 6))

    return None


def call_zhipu(batch_items: List[Dict[str, str]]) -> Optional[List[Dict]]:
    cfg = _zhipu_runtime_config()
    api_key = str(cfg["api_key"])
    base_url = str(cfg["base_url"])
    model = str(cfg["model"])
    connect_timeout = int(cfg["connect_timeout"])
    read_timeout = int(cfg["read_timeout"])
    max_retries = int(cfg["max_retries"])

    if not api_key:
        print("智谱调用失败: 未设置 ZHIPU_API_KEY")
        return None

    news_text = "\n\n".join(
        [f"新闻{i + 1}：{item['text']}" for i, item in enumerate(batch_items)]
    )
    prompt = PROMPT_TEMPLATE.replace("{text}", news_text)

    url = base_url.rstrip("/") + "/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是A股事件抽取助手，只返回纯JSON。单条新闻返回JSON对象；多条新闻返回JSON数组，且数量与输入一致。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            )
            parsed = _parse_llm_json_list(text)
            if parsed:
                return parsed
            print(f"智谱返回内容无法解析为JSON列表，第 {attempt} 次重试")
        except requests.HTTPError as e:
            body = ""
            try:
                body = (e.response.text or "")[:500]
            except Exception:
                body = ""
            is_rate_limited = False
            status_code = 0
            try:
                status_code = int((e.response.status_code or 0))
            except Exception:
                status_code = 0
            body_lower = body.lower()
            if status_code == 429 or '"code":"1302"' in body_lower or "速率限制" in body:
                is_rate_limited = True

            if is_rate_limited:
                wait_sec = min(8 * attempt, 60)
                print(f"智谱限流(第 {attempt} 次): 等待 {wait_sec}s 后重试")
                if attempt < max_retries:
                    time.sleep(wait_sec)
                    continue

            print(f"智谱 HTTP错误(第 {attempt} 次): {e}; body={body}")
        except requests.RequestException as e:
            print(f"智谱网络错误(第 {attempt} 次): {e}")
        except Exception as e:
            print(f"智谱调用异常(第 {attempt} 次): {e}")
        except BaseException as e:
            print(f"智谱调用中断(第 {attempt} 次): {type(e).__name__}: {e!r}")
            return None

        if attempt < max_retries:
            time.sleep(min(2 * attempt, 6))

    return None


def call_llm(batch_items: List[Dict[str, str]], provider: str = "auto") -> Optional[List[Dict]]:
    p = str(provider or "auto").strip().lower()
    if p == "gemini":
        return call_gemini(batch_items)
    if p == "deepseek":
        return call_deepseek(batch_items)
    if p == "zhipu":
        return call_zhipu(batch_items)

    # auto: 优先 DeepSeek，失败后回退智谱和 Gemini
    deepseek_result = call_deepseek(batch_items)
    if deepseek_result:
        return deepseek_result
    zhipu_result = call_zhipu(batch_items)
    if zhipu_result:
        return zhipu_result
    return call_gemini(batch_items)


def load_texts() -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = f"""
    SELECT {TEXT_COLUMN}, {DATE_COLUMN}
    FROM {TABLE_NAME}
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    data: List[Dict[str, str]] = []
    for text, publish_time in rows:
        if not text:
            continue
        if KEYWORD_PATTERN.search(text):
            data.append({"text": text, "date": publish_time})

    print("原始新闻数量:", len(rows))
    print("过滤后新闻数量:", len(data))
    return data


def _fallback_event(item: Dict[str, str]) -> Dict:
    return {
        "date": item.get("date", ""),
        "event_level_1": "无实质事件/市场噪音",
        "event_level_2": "解析失败",
        "sentiment": "neutral",
        "impact_cycle": "脉冲型",
        "predictability": "突发型",
        "strength": 0,
        "related_concepts": [],
        "related_companies": [],
    }


def load_and_extract_events(limit: int = None, output_file: str = OUTPUT_FILE, provider: str = "auto") -> List[Dict]:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = load_texts()
    if limit is not None:
        data = data[:limit]

    results: List[Dict] = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    if limit is not None and len(results) > limit:
        results = results[:limit]

    done = len(results)
    print("已完成:", done)
    data = data[done:]

    batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    for batch in tqdm(batches):
        event_list = call_llm(batch, provider=provider)

        if event_list and isinstance(event_list, list):
            for event, item in zip(event_list, batch):
                if not isinstance(event, dict):
                    results.append(_fallback_event(item))
                    continue
                event["date"] = item.get("date", "")
                results.append(event)
            if len(event_list) < len(batch):
                for item in batch[len(event_list):]:
                    results.append(_fallback_event(item))
        else:
            for item in batch:
                results.append(_fallback_event(item))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(DELAY)

    return results


def main() -> None:
    results = load_and_extract_events(limit=None, output_file=OUTPUT_FILE, provider="auto")
    print(f"提取完成，共 {len(results)} 条事件")


if __name__ == "__main__":
    main()
