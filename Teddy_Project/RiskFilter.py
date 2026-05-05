"""
RiskFilter.py
高阶风控过滤器：从数据库获取股票池，从JSON获取strength，结合基本面和技术面过滤，输出前3名安全股票
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import json
import akshare as ak
import time
from datetime import datetime, timedelta
from graph_matcher import match_stocks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RiskFilter:
    """风险过滤器"""
    
    def __init__(
        self,
        db_path=None,
        json_file=None,
        price_data_file='日线数据.csv',
        trade_week_start=None,
        event_window_start=None,
        event_window_end=None,
    ):
        """
        初始化风险过滤器
        
        Args:
            db_path: 数据库路径（获取股票池）
            json_file: 事件JSON文件路径（获取strength）
            price_data_file: 日线数据CSV文件路径
        """
        self.db_path = db_path or os.path.join(BASE_DIR, 'data', 'db', 'structured_events.db')
        self.json_file = json_file or os.path.join(BASE_DIR, 'data', 'events', 'events_date.json')
        self.price_data_file = price_data_file
        self.trade_week_start = pd.to_datetime(trade_week_start).normalize() if trade_week_start else None
        self.event_window_start = pd.to_datetime(event_window_start).normalize() if event_window_start else None
        self.event_window_end = pd.to_datetime(event_window_end).normalize() if event_window_end else None
        self.stock_pool = None  # 原始股票池（含strength）
        self.healthy_stocks = None  # 财务健康的股票
        self.final_stocks = None  # 最终过滤后的股票
        self.selected_event_ids = set()
        self.min_event_strength = float(os.getenv('RF_MIN_EVENT_STRENGTH', '6'))
        self.max_events_per_week = int(os.getenv('RF_MAX_EVENTS_PER_WEEK', '120'))
        allowed_sentiments = os.getenv('RF_ALLOWED_EVENT_SENTIMENTS', 'positive,neutral')
        self.allowed_event_sentiments = {
            s.strip().lower() for s in str(allowed_sentiments).split(',') if s.strip()
        }
        if not self.allowed_event_sentiments:
            self.allowed_event_sentiments = {'positive', 'neutral'}
        self.enable_event_title_dedup = os.getenv('RF_ENABLE_EVENT_TITLE_DEDUP', '1').strip() != '0'
        self.max_events_per_title = int(os.getenv('RF_MAX_EVENTS_PER_TITLE', '3'))
        if self.max_events_per_title < 1:
            self.max_events_per_title = 1
        # 收紧短期技术阈值，避免高分但短期走弱的标的占比较高。
        self.min_momentum_5d = float(os.getenv('RF_MIN_MOMENTUM_5D', '-1.0'))
        self.min_momentum_3d = float(os.getenv('RF_MIN_MOMENTUM_3D', '-1.5'))
        self.max_drawdown_3d = float(os.getenv('RF_MAX_DRAWDOWN_3D', '-8.0'))
        self.allow_supplement = True
        self.min_recent_trading_days = 4
        self.min_recent_avg_amount = 20000.0
        self.max_recent_drawdown_3d = -0.10
        self.block_event_keywords = ['停牌', '停牌核查', '继续停牌', '申请停牌']
        self.max_same_event_title = 1
        self.event_title_penalty = {
            '战争/冲突': 8.0,
        }
        self._price_df_for_trade_check = None
        self._recent_trade_cache = {}
        self._online_price_cache = {}
        self.intermediate_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts', 'intermediate')
        
        # 财务数据缓存
        self.financial_cache = {}
        self.finance_cache_file = os.path.join(self.intermediate_dir, 'financial_cache.json')
        self.finance_cache_ttl_days = max(1, int(os.getenv('RF_FINANCE_CACHE_TTL_DAYS', '30')))
        self.request_interval = 0.5
        self.last_request_time = 0
        self.finance_retries = max(1, int(os.getenv('RF_FINANCE_RETRIES', '3')))
        self.finance_retry_backoff = max(0.2, float(os.getenv('RF_FINANCE_RETRY_BACKOFF_SEC', '0.8')))
        # 默认宽松：财务接口取数失败时先放行，避免因源端抖动导致大面积误淘汰。
        self.finance_strict_mode = os.getenv('RF_FINANCE_STRICT', '0').strip() == '1'
        # 财务检查前分层预筛（显著减少在线财务请求量）
        self.finance_prefilter_enabled = os.getenv('RF_FINANCE_PREFILTER_ENABLED', '1').strip() != '0'
        self.finance_max_checks = max(0, int(os.getenv('RF_FINANCE_MAX_CHECKS', '1000')))
        self.finance_keep_above_strength = float(os.getenv('RF_FINANCE_KEEP_ABOVE_STRENGTH', '7.8'))
        self.finance_per_event_cap = max(1, int(os.getenv('RF_FINANCE_PER_EVENT_CAP', '120')))
        self._finance_cache_dirty = False
        self._load_finance_cache()
        
        # 预定义的A股代码范围（用于快速判断）
        self.sh_prefixes = ['600', '601', '603', '605', '688', '689']
        self.sz_prefixes = ['000', '001', '002', '003', '004', '300', '301']
        self.bj_prefixes = ['830', '831', '832', '833', '834', '835', '836', '837', '838', '839',
                           '870', '871', '872', '873', '874', '875', '876', '877', '878', '879',
                           '880', '881', '882', '883', '884', '885', '886', '887', '888', '889',
                           '920', '921', '922', '923', '924', '925', '926', '927', '928', '929']
        
        # 港股常见前缀（需要排除）
        self.hk_prefixes = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009',
                           '010', '011', '012', '013', '014', '015', '016', '017', '018', '019']

    def _get_trade_week_start(self, events):
        """确定目标交易周周一：默认取最新事件所在周的下一周周一。"""
        if self.trade_week_start is not None:
            return self.trade_week_start

        dates = []
        for e in events:
            dt = pd.to_datetime(e.get('date', ''), errors='coerce')
            if pd.notna(dt):
                dates.append(dt)

        if not dates:
            # 兜底：当前日期所在周周一
            now = pd.Timestamp(datetime.now().date())
            return now - pd.Timedelta(days=now.weekday())

        latest = max(dates)
        week_monday = latest.normalize() - pd.Timedelta(days=latest.weekday())
        return week_monday + pd.Timedelta(days=7)
    
    def is_a_share(self, stock_code):
        """判断是否为A股"""
        code = str(stock_code).strip()
        if len(code) < 6:
            code = code.zfill(6)
        prefix = code[:3]
        
        if prefix in self.sh_prefixes:
            return True
        if prefix in self.sz_prefixes:
            return True
        if prefix in self.bj_prefixes:
            return True
        return False

    def _is_blocked_event_title(self, title):
        """过滤停牌类事件，避免不可交易标的入选。"""
        t = str(title or '').strip()
        if not t:
            return False
        if '复牌' in t:
            return False
        return any(k in t for k in self.block_event_keywords)

    def _fetch_buyday_open_online(self, stock_code, buy_date):
        """本地日线未覆盖买入日时，尝试在线拉取买入日开盘价。"""
        code = str(stock_code).zfill(6)
        if code.startswith('6'):
            symbol = f"sh{code}"
        elif code.startswith(('0', '3')):
            symbol = f"sz{code}"
        elif code.startswith(('8', '4')):
            symbol = f"bj{code}"
        else:
            return None

        try:
            start = buy_date.strftime('%Y%m%d')
            end = buy_date.strftime('%Y%m%d')
            df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust='')
            if df is None or df.empty:
                return None
            d = df.copy()
            d['date'] = pd.to_datetime(d['date'], errors='coerce').dt.normalize()
            row = d[d['date'] == buy_date.normalize()]
            if row.empty:
                return None
            open_px = pd.to_numeric(row.iloc[0].get('open', np.nan), errors='coerce')
            if pd.isna(open_px) or float(open_px) <= 0:
                return None
            return float(open_px)
        except BaseException:
            return None

    def _fetch_recent_trade_features_online(self, stock_code):
        """拉取买入日前近期交易特征，用于补齐候选质量控制。"""
        code = str(stock_code).zfill(6)
        if code in self._recent_trade_cache:
            return self._recent_trade_cache[code]

        if self.trade_week_start is None:
            feat = {
                'ok': True,
                'recent_days': 0,
                'recent_avg_amount': 0.0,
                'recent_momentum_3d': 0.0,
                'recent_max_drawdown_3d': 0.0,
            }
            self._recent_trade_cache[code] = feat
            return feat

        if code.startswith('6'):
            symbol = f"sh{code}"
        elif code.startswith(('0', '3')):
            symbol = f"sz{code}"
        elif code.startswith(('8', '4')):
            symbol = f"bj{code}"
        else:
            feat = {'ok': False, 'reason': '非A股代码'}
            self._recent_trade_cache[code] = feat
            return feat

        buy_date = pd.to_datetime(self.trade_week_start).normalize() + pd.Timedelta(days=1)
        start = (buy_date - pd.Timedelta(days=20)).strftime('%Y%m%d')
        end = buy_date.strftime('%Y%m%d')

        try:
            df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust='')
            if df is None or df.empty:
                feat = {'ok': False, 'reason': '近期在线行情为空'}
                self._recent_trade_cache[code] = feat
                return feat

            d = df.copy()
            d['date'] = pd.to_datetime(d['date'], errors='coerce').dt.normalize()
            d['open'] = pd.to_numeric(d.get('open', np.nan), errors='coerce')
            d['close'] = pd.to_numeric(d.get('close', np.nan), errors='coerce')
            d['amount'] = pd.to_numeric(d.get('amount', np.nan), errors='coerce')
            d = d.dropna(subset=['date', 'open', 'close'])
            d = d[d['date'] <= buy_date].sort_values('date')
            if d.empty:
                feat = {'ok': False, 'reason': '买入日前无可用行情'}
                self._recent_trade_cache[code] = feat
                return feat

            recent = d.tail(5).copy()
            recent_days = int(len(recent))
            avg_amount = float(pd.to_numeric(recent['amount'], errors='coerce').fillna(0).mean()) if 'amount' in recent.columns else 0.0

            if len(d) >= 4:
                close_now = float(d.iloc[-1]['close'])
                close_3d = float(d.iloc[-4]['close'])
                momentum_3d = (close_now - close_3d) / close_3d if close_3d else 0.0
            else:
                momentum_3d = 0.0

            if len(recent) >= 3:
                w = recent.tail(3)
                high = float(w['high'].max()) if 'high' in w.columns else float(w['close'].max())
                low = float(w['low'].min()) if 'low' in w.columns else float(w['close'].min())
                max_dd_3d = (low - high) / high if high else 0.0
            else:
                max_dd_3d = 0.0

            feat = {
                'ok': True,
                'recent_days': recent_days,
                'recent_avg_amount': avg_amount,
                'recent_momentum_3d': float(momentum_3d),
                'recent_max_drawdown_3d': float(max_dd_3d),
            }
            self._recent_trade_cache[code] = feat
            return feat
        except BaseException:
            feat = {'ok': False, 'reason': '近期在线行情请求失败'}
            self._recent_trade_cache[code] = feat
            return feat

    def _is_buyday_tradeable(self, stock_code):
        """买入日可交易校验：本地优先，本地缺失时在线复核。"""
        if self.trade_week_start is None or self._price_df_for_trade_check is None:
            return True, ''

        buy_date = pd.to_datetime(self.trade_week_start).normalize() + pd.Timedelta(days=1)
        price_df = self._price_df_for_trade_check
        if 'date' not in price_df.columns or 'code' not in price_df.columns:
            return True, ''

        max_dt = pd.to_datetime(price_df['date'], errors='coerce').max()
        if pd.isna(max_dt) or buy_date > max_dt.normalize():
            online_open = self._fetch_buyday_open_online(stock_code, buy_date)
            # 在线校验不可用时放行，避免因网络抖动导致“必须买三只”无法执行。
            if online_open is None:
                return True, f"买入日 {buy_date.date()} 在线校验不可用，放行"
            return True, ''

        s = price_df[price_df['code'] == str(stock_code)]
        if s.empty:
            online_open = self._fetch_buyday_open_online(stock_code, buy_date)
            if online_open is not None:
                return True, f"买入日 {buy_date.date()} 本地缺失，在线复核通过"
            return False, f"买入日 {buy_date.date()} 本地无行情且在线无可用开盘价"

        d = s[s['date'].dt.normalize() == buy_date]
        if d.empty:
            online_open = self._fetch_buyday_open_online(stock_code, buy_date)
            if online_open is not None:
                return True, f"买入日 {buy_date.date()} 本地缺失，在线复核通过"
            return False, f"买入日 {buy_date.date()} 停牌或无成交（在线无可用开盘价）"

        open_col = d.iloc[0].get('open', np.nan)
        if pd.isna(open_col) or float(open_col) <= 0:
            online_open = self._fetch_buyday_open_online(stock_code, buy_date)
            if online_open is not None:
                return True, f"买入日 {buy_date.date()} 本地开盘无效，在线复核通过"
            return False, f"买入日 {buy_date.date()} 开盘价无效（在线无可用开盘价）"
        return True, ''
    
    def load_stock_pool_from_db_and_json(self):
        """
        从数据库和JSON文件加载股票池和strength，并记录事件ID
        """
        print("\n" + "="*60)
        print("步骤1: 从数据库和JSON文件加载股票池")
        print("="*60)
        
        # 1. 先读取JSON并确定本次交易周与事件窗口（前一周）
        with open(self.json_file, 'r', encoding='utf-8') as f:
            events = json.load(f)

        trade_week_start = self._get_trade_week_start(events)

        if self.event_window_start is not None and self.event_window_end is not None:
            window_start = self.event_window_start
            window_end = self.event_window_end
            print(f"从JSON文件加载了 {len(events)} 个事件")
            print(f"目标交易周(周二买入/周五卖出): {trade_week_start.date()} ~ {(trade_week_start + pd.Timedelta(days=4)).date()}")
            print(f"选股事件窗口(自定义): {window_start.date()} ~ {window_end.date()}")
        else:
            window_start = trade_week_start - pd.Timedelta(days=7)
            window_end = trade_week_start - pd.Timedelta(days=1)
            print(f"从JSON文件加载了 {len(events)} 个事件")
            print(f"目标交易周(周二买入/周五卖出): {trade_week_start.date()} ~ {(trade_week_start + pd.Timedelta(days=4)).date()}")
            print(f"选股事件窗口(前一周): {window_start.date()} ~ {window_end.date()}")

        # 2. 仅保留前一周事件，构建事件ID映射
        raw_candidates = []
        for i, event in enumerate(events):
            event_id = i + 1
            event_dt = pd.to_datetime(event.get('date', ''), errors='coerce')
            if pd.isna(event_dt):
                continue
            event_day = event_dt.normalize()
            if window_start <= event_day <= window_end:
                raw_candidates.append({
                    'event_id': event_id,
                    'event_day': event_day,
                    'strength': event.get('strength', 0),
                    'event_title': event.get('event_level_2', ''),
                    'sentiment': event.get('sentiment', ''),
                    'impact_cycle': event.get('impact_cycle', ''),
                    'predictability': event.get('predictability', ''),
                    'event_date': event.get('date', ''),
                    'related_concepts': event.get('related_concepts', []) or [],
                    'related_companies': event.get('related_companies', []) or [],
                })

        # 粗筛：按情绪与强度放行更多事件，避免信号过窄。
        coarse_candidates = [
            x for x in raw_candidates
            if str(x.get('sentiment', '')).lower() in self.allowed_event_sentiments
            and float(x.get('strength', 0)) >= self.min_event_strength
            and (not self._is_blocked_event_title(x.get('event_title', '')))
        ]

        dedup_count = 0
        if self.enable_event_title_dedup:
            buckets = {}
            for x in coarse_candidates:
                title = (x.get('event_title') or '').strip()
                key = title if title else f"event_{x['event_id']}"
                buckets.setdefault(key, []).append(x)

            limited = []
            for _, items in buckets.items():
                ranked_items = sorted(
                    items,
                    key=lambda r: (float(r.get('strength', 0)), r.get('event_day')),
                    reverse=True,
                )
                limited.extend(ranked_items[: self.max_events_per_title])

            dedup_count = len(coarse_candidates) - len(limited)
            candidate_pool = limited
        else:
            candidate_pool = coarse_candidates

        refined = sorted(
            candidate_pool,
            key=lambda r: (float(r.get('strength', 0)), r.get('event_day')),
            reverse=True,
        )[: self.max_events_per_week]

        event_info = {x['event_id']: {
            'strength': x['strength'],
            'event_title': x['event_title'],
            'sentiment': x['sentiment'],
            'impact_cycle': x['impact_cycle'],
            'predictability': x['predictability'],
            'event_date': x['event_date'],
            'related_concepts': x.get('related_concepts', []),
            'related_companies': x.get('related_companies', []),
        } for x in refined}

        self.selected_event_ids = set(event_info.keys())
        print(f"前一周原始事件数: {len(raw_candidates)}")
        print(
            f"前一周粗筛通过事件数: {len(coarse_candidates)} "
            f"(sentiment in {sorted(self.allowed_event_sentiments)}, strength >= {self.min_event_strength})"
        )
        if self.enable_event_title_dedup:
            print(
                f"同标题限流后事件数: {len(candidate_pool)} "
                f"(每标题最多 {self.max_events_per_title} 条, 去重减少 {dedup_count} 条)"
            )
        print(f"前一周最终入选事件数: {len(self.selected_event_ids)} (上限 {self.max_events_per_week})")

        if len(self.selected_event_ids) == 0:
            self.stock_pool = pd.DataFrame(columns=['code', 'name', 'strength', 'event_title', 'event_id', 'sentiment', 'impact_cycle', 'predictability', 'event_date', 'event_count'])
            return self.stock_pool

        # 3. 从数据库获取“前一周事件”对应股票池
        conn = sqlite3.connect(self.db_path)
        ids_csv = ",".join(str(x) for x in sorted(self.selected_event_ids))
        query = f"""
        SELECT
            s.event_id,
            s.stock_code,
            s.stock_name
        FROM event_stocks s
        WHERE s.stock_code IS NOT NULL AND s.stock_code != ''
          AND s.event_id IN ({ids_csv})
        """
        stock_event_df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"从数据库获取了 {len(stock_event_df)} 条股票-事件关联记录（前一周窗口）")

        if len(stock_event_df) == 0:
            print("提示: 事件ID与数据库映射为空，启用概念关键词匹配回退。")
            fallback_rows = []
            for event_id, info in event_info.items():
                # 先使用事件中直接给出的公司代码
                related_codes = info.get('related_companies', [])
                if not isinstance(related_codes, list):
                    related_codes = []
                for c in related_codes:
                    code = str(c).strip()
                    if code.isdigit():
                        code = code.zfill(6)
                    if len(code) == 6 and code.isdigit():
                        fallback_rows.append({
                            'event_id': event_id,
                            'stock_code': code,
                            'stock_name': code,
                        })

                keywords = info.get('related_concepts', [])
                if not isinstance(keywords, list):
                    keywords = []
                if not keywords:
                    title_kw = str(info.get('event_title', '')).strip()
                    if title_kw:
                        keywords = [title_kw]

                matched = match_stocks(keywords)
                for s in matched:
                    fallback_rows.append({
                        'event_id': event_id,
                        'stock_code': str(s.get('code', '')).strip(),
                        'stock_name': str(s.get('name', '')).strip(),
                    })

            stock_event_df = pd.DataFrame(fallback_rows)
            print(f"回退匹配得到 {len(stock_event_df)} 条股票-事件关联记录")

        if stock_event_df.empty:
            self.stock_pool = pd.DataFrame(columns=[
                'code', 'name', 'strength', 'event_title', 'event_id',
                'sentiment', 'impact_cycle', 'predictability', 'event_date', 'event_count'
            ])
            print("前一周事件未匹配到可交易股票。")
            return self.stock_pool

        # 4. 统一股票名称
        name_counts = stock_event_df.groupby(['stock_code', 'stock_name']).size().reset_index(name='count')
        unified_names = {}
        for code in name_counts['stock_code'].unique():
            code_data = name_counts[name_counts['stock_code'] == code]
            code_data = code_data.sort_values('count', ascending=False)
            unified_names[code] = code_data.iloc[0]['stock_name']
        
        stock_event_df['stock_name'] = stock_event_df['stock_code'].map(unified_names)
        
        # 5. 计算每只股票的强度和关联的事件ID
        stock_strength = {}
        for _, row in stock_event_df.iterrows():
            event_id = row['event_id']
            stock_code = str(row['stock_code']).strip()
            if len(stock_code) < 6:
                stock_code = stock_code.zfill(6)
            stock_name = row['stock_name']
            
            strength = event_info.get(event_id, {}).get('strength', 0)
            event_title = event_info.get(event_id, {}).get('event_title', '')
            sentiment = event_info.get(event_id, {}).get('sentiment', '')
            impact_cycle = event_info.get(event_id, {}).get('impact_cycle', '')
            predictability = event_info.get(event_id, {}).get('predictability', '')
            event_date = event_info.get(event_id, {}).get('event_date', '')
            
            if stock_code not in stock_strength:
                stock_strength[stock_code] = {
                    'name': stock_name,
                    'max_strength': strength,
                    'max_strength_event_id': event_id,  # 记录最大strength对应的事件ID
                    'max_strength_event_title': event_title,
                    'max_strength_sentiment': sentiment,
                    'max_strength_impact_cycle': impact_cycle,
                    'max_strength_predictability': predictability,
                    'max_strength_event_date': event_date,
                    'all_strengths': [strength],
                    'all_events': [(event_id, strength, event_title)]
                }
            else:
                stock_strength[stock_code]['all_strengths'].append(strength)
                stock_strength[stock_code]['all_events'].append((event_id, strength, event_title))
                if strength > stock_strength[stock_code]['max_strength']:
                    stock_strength[stock_code]['max_strength'] = strength
                    stock_strength[stock_code]['max_strength_event_id'] = event_id
                    stock_strength[stock_code]['max_strength_event_title'] = event_title
                    stock_strength[stock_code]['max_strength_sentiment'] = sentiment
                    stock_strength[stock_code]['max_strength_impact_cycle'] = impact_cycle
                    stock_strength[stock_code]['max_strength_predictability'] = predictability
                    stock_strength[stock_code]['max_strength_event_date'] = event_date
        
        # 6. 创建股票池DataFrame
        stocks_data = []
        for code, info in stock_strength.items():
            stocks_data.append({
                'code': code,
                'name': info['name'],
                'strength': info['max_strength'],
                'event_title': info['max_strength_event_title'],  # 新增：事件标题
                'event_id': info['max_strength_event_id'],  # 新增：对应事件ID
                'sentiment': info.get('max_strength_sentiment', ''),
                'impact_cycle': info.get('max_strength_impact_cycle', ''),
                'predictability': info.get('max_strength_predictability', ''),
                'event_date': info.get('max_strength_event_date', ''),
                'event_count': len(info['all_strengths'])
            })
        
        self.stock_pool = pd.DataFrame(stocks_data)
        if self.stock_pool.empty:
            self.stock_pool = pd.DataFrame(columns=[
                'code', 'name', 'strength', 'event_title', 'event_id',
                'sentiment', 'impact_cycle', 'predictability', 'event_date', 'event_count'
            ])
        else:
            self.stock_pool = self.stock_pool.sort_values('strength', ascending=False)
        
        print(f"\n股票池共有 {len(self.stock_pool)} 只股票")
        print("\n股票池前10只（按strength排序）:")
        for i, row in self.stock_pool.head(10).iterrows():
            print(f"  {row['code']} {row['name']} - strength: {row['strength']} (事件ID: {row['event_id']})")
        
        return self.stock_pool

    def enrich_healthy_stocks_metadata(self):
        """为续跑缓存补齐评分维度，确保后续“评分依据/入选原因”完整。"""
        if self.healthy_stocks is None or self.stock_pool is None or len(self.healthy_stocks) == 0:
            return

        required_cols = ['strength', 'event_title', 'event_id', 'sentiment', 'impact_cycle', 'predictability', 'event_date']
        missing = [c for c in required_cols if c not in self.healthy_stocks.columns]
        if not missing:
            return

        pool_cols = ['code'] + [c for c in required_cols if c in self.stock_pool.columns]
        pool_meta = self.stock_pool[pool_cols].drop_duplicates('code')
        self.healthy_stocks = self.healthy_stocks.merge(pool_meta, on='code', how='left', suffixes=('', '_pool'))

        for col in required_cols:
            pool_col = f"{col}_pool"
            if col not in self.healthy_stocks.columns and pool_col in self.healthy_stocks.columns:
                self.healthy_stocks[col] = self.healthy_stocks[pool_col]
            elif col in self.healthy_stocks.columns and pool_col in self.healthy_stocks.columns:
                self.healthy_stocks[col] = self.healthy_stocks[col].fillna(self.healthy_stocks[pool_col])

        drop_cols = [c for c in self.healthy_stocks.columns if c.endswith('_pool')]
        if drop_cols:
            self.healthy_stocks = self.healthy_stocks.drop(columns=drop_cols)
    
    def _rate_limit(self):
        """限流"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()

    def _load_finance_cache(self):
        """加载跨运行财务缓存，减少重复在线请求。"""
        self.financial_cache = {}
        try:
            if not os.path.exists(self.finance_cache_file):
                return
            with open(self.finance_cache_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return

            now_ts = time.time()
            ttl_sec = float(self.finance_cache_ttl_days) * 86400.0
            for code, item in raw.items():
                if not isinstance(item, dict):
                    continue
                fetched_at = float(item.get('fetched_at', 0) or 0)
                data = item.get('data')
                if not isinstance(data, dict):
                    continue
                if fetched_at > 0 and (now_ts - fetched_at) <= ttl_sec:
                    self.financial_cache[f"finance_{str(code).zfill(6)}"] = data
        except Exception:
            self.financial_cache = {}

    def _persist_finance_cache(self):
        """持久化财务缓存到文件。"""
        if not self._finance_cache_dirty:
            return
        try:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            payload = {}
            now_ts = time.time()
            for k, v in self.financial_cache.items():
                if not str(k).startswith('finance_') or not isinstance(v, dict):
                    continue
                code = str(k).replace('finance_', '').zfill(6)
                payload[code] = {
                    'fetched_at': now_ts,
                    'data': v,
                }
            with open(self.finance_cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            self._finance_cache_dirty = False
        except Exception:
            pass

    def _set_finance_cache(self, stock_code, result):
        cache_key = f"finance_{str(stock_code).zfill(6)}"
        self.financial_cache[cache_key] = result
        self._finance_cache_dirty = True

    def flush_finance_cache(self):
        """对外暴露：在流程结束前显式落盘缓存。"""
        self._persist_finance_cache()
    
    def get_financial_data(self, stock_code):
        """
        获取股票财务数据（净资产和净利润）
        
        Returns:
            dict: {'net_assets': 净资产, 'profits': 净利润列表, 'has_data': bool}
        """
        cache_key = f"finance_{stock_code}"
        if cache_key in self.financial_cache:
            return self.financial_cache[cache_key]
        
        self._rate_limit()

        def _to_float(v):
            if pd.isna(v):
                return None
            if isinstance(v, (int, float, np.number)):
                return float(v)
            s = str(v).strip().replace(',', '')
            if s in {'', 'False', 'None', '--', 'nan'}:
                return None
            # 保留数字、负号和小数点
            s = ''.join(ch for ch in s if (ch.isdigit() or ch in '.-'))
            if s in {'', '-', '.', '-.'}:
                return None
            try:
                return float(s)
            except Exception:
                return None

        def _parse_from_abstract(df):
            profits = []
            net_assets = None

            profit_row = None
            for _, row in df.iterrows():
                if '净利润' in str(row.get('指标', '')) and row.get('选项') == '常用指标':
                    profit_row = row
                    break

            date_cols = [col for col in df.columns if col not in ['选项', '指标'] and str(col).isdigit()]
            date_cols.sort(reverse=True)

            if profit_row is not None:
                year_profits = []
                for col in date_cols:
                    if str(col).endswith('1231'):
                        profit = _to_float(profit_row[col])
                        if profit is not None and profit != 0:
                            year_profits.append(profit)
                        if len(year_profits) >= 2:
                            break
                profits = year_profits

            for _, row in df.iterrows():
                if any(kw in str(row.get('指标', '')) for kw in ['净资产', '所有者权益', '股东权益']):
                    if date_cols:
                        net_assets = _to_float(row[date_cols[0]])
                    break

            return net_assets, profits

        def _parse_from_abstract_ths(df):
            # 同花顺接口结构为逐报告期行，尽量从列名里识别净资产/净利润
            net_assets = None
            profits = []

            if df is None or df.empty:
                return net_assets, profits

            d = df.copy()
            date_col = d.columns[0]
            d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
            d = d.dropna(subset=[date_col]).sort_values(date_col, ascending=False)

            # 净资产列优先匹配“每股净资产/净资产/股东权益”
            net_asset_cols = [
                c for c in d.columns
                if any(k in str(c) for k in ['每股净资产', '净资产', '股东权益'])
            ]
            for c in net_asset_cols:
                v = _to_float(d.iloc[0][c])
                if v is not None:
                    net_assets = v
                    break

            # 净利润列优先匹配“净利润/归母净利润”
            profit_cols = [
                c for c in d.columns
                if any(k in str(c) for k in ['净利润', '归母净利润'])
            ]
            if profit_cols:
                annual = d[d[date_col].dt.strftime('%m-%d') == '12-31']
                for _, row in annual.iterrows():
                    for c in profit_cols:
                        v = _to_float(row[c])
                        if v is not None and v != 0:
                            profits.append(v)
                            break
                    if len(profits) >= 2:
                        break

            return net_assets, profits

        providers = [
            ('ak.stock_financial_abstract', lambda c: ak.stock_financial_abstract(symbol=c), _parse_from_abstract),
            ('ak.stock_financial_abstract_ths', lambda c: ak.stock_financial_abstract_ths(symbol=c), _parse_from_abstract_ths),
        ]

        last_err = None
        saw_non_empty = False
        for name, fetcher, parser in providers:
            for attempt in range(1, self.finance_retries + 1):
                try:
                    df = fetcher(stock_code)
                    if df is None or df.empty:
                        break

                    saw_non_empty = True
                    net_assets, profits = parser(df)
                    result = {
                        'net_assets': net_assets,
                        'profits': profits,
                        'has_data': True,
                        'error': None,
                        'fetch_status': 'ok',
                        'data_source': name,
                    }
                    self._set_finance_cache(stock_code, result)
                    self._persist_finance_cache()
                    return result
                except Exception as e:
                    last_err = f"{name}: {str(e)}"
                    if attempt < self.finance_retries:
                        time.sleep(self.finance_retry_backoff * attempt)
                        continue
                    break

        result = {
            'net_assets': None,
            'profits': [],
            'has_data': False,
            'error': last_err or ('无数据' if not saw_non_empty else '解析失败'),
            'fetch_status': 'empty' if not saw_non_empty else 'error',
            'data_source': None,
        }
        self._set_finance_cache(stock_code, result)
        self._persist_finance_cache()
        return result
    
    def check_financial_health(self, stock_code, stock_name):
        """
        检查股票财务健康（连续两年亏损或净资产为负则淘汰）
        
        Returns:
            bool: True表示健康，False表示有风险
        """
        # 获取财务数据
        finance = self.get_financial_data(stock_code)
        
        # 财务数据缺失/接口异常：默认放行（可通过 RF_FINANCE_STRICT=1 改为严格淘汰）。
        if not finance['has_data']:
            reason = finance.get('fetch_status', 'unknown')
            err = str(finance.get('error') or '')
            if self.finance_strict_mode:
                print(f"  {stock_code} {stock_name}: 财务数据不可用({reason})，严格模式淘汰")
                return False
            print(f"  {stock_code} {stock_name}: 财务数据不可用({reason})，宽松模式放行；原因: {err[:80]}")
            return True
        
        # 检查净资产
        if finance['net_assets'] is not None:
            if finance['net_assets'] <= 0:
                print(f"  {stock_code} {stock_name}: 净资产为负({finance['net_assets']:.2f})，淘汰")
                return False
        
        # 检查连续两年亏损
        if len(finance['profits']) >= 2:
            if finance['profits'][0] < 0 and finance['profits'][1] < 0:
                print(f"  {stock_code} {stock_name}: 连续两年亏损({finance['profits']})，淘汰")
                return False
        
        print(f"  {stock_code} {stock_name}: 财务健康")
        return True
    
    def load_price_data(self):
        """加载日线数据"""
        print("\n" + "="*60)
        print("步骤3: 加载日线数据")
        print("="*60)
        
        if not os.path.exists(self.price_data_file):
            print(f"错误: 找不到日线数据文件 {self.price_data_file}")
            return None
        
        # 尝试多种编码
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030']
        
        df = None
        for enc in encodings:
            try:
                print(f"尝试编码: {enc}")
                # 完整读取，跳过注释行
                df = pd.read_csv(self.price_data_file, encoding=enc, comment='#')
                print(f"  成功使用 {enc} 编码，共 {len(df)} 行")
                break
            except Exception as e:
                print(f"  {enc} 失败: {str(e)[:50]}")
                continue
        
        if df is None:
            print("无法读取日线数据文件，跳过技术面过滤")
            return None
        
        # 查看实际列名
        print(f"\n实际列名: {df.columns.tolist()}")
        
        # 根据实际列名重命名
        if '代码' in df.columns:
            df = df.rename(columns={
                '代码': 'code',
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '换手率(%)': 'turnover',
                '流通市值': 'market_cap'
            })
        
        # 【关键修复】统一股票代码格式为6位字符串
        df['code'] = df['code'].astype(str).str.strip()
        df['code'] = df['code'].apply(lambda x: x.zfill(6) if x.isdigit() else x)
        
        # 检查是否有空值
        df = df.dropna(subset=['date'])
        
        # 确保date是字符串类型
        df['date'] = df['date'].astype(str)
        
        # 只保留符合日期格式的行
        date_pattern = r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$'
        mask = df['date'].str.match(date_pattern, na=False)
        df = df[mask].copy()
        
        if len(df) == 0:
            print("没有有效的日期数据")
            return None
        
        # 转换日期格式
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
            if df['date'].isna().any():
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=['date'])
            print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
        except Exception as e:
            print(f"日期转换失败: {e}")
            return None
        
        # 数值列转换
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        df = df.dropna(subset=['close'])
        
        print(f"\n加载了 {len(df)} 条有效日线数据")
        print(f"涉及股票数量: {df['code'].nunique()} 只")
        print(f"股票代码示例: {df['code'].unique()[:10]}")
        
        # 【调试】检查特定股票是否存在
        test_code = '600292'
        if test_code in df['code'].values:
            print(f"股票 {test_code} 存在于日线数据中，共 {len(df[df['code'] == test_code])} 条记录")
        else:
            print(f"股票 {test_code} 不存在于日线数据中")
            # 打印相近的代码
            similar = [c for c in df['code'].unique() if test_code in c]
            if similar:
                print(f"   相近代码: {similar[:5]}")
        
        return df
    
    def _score_sentiment(self, sentiment):
        mapping = {'positive': 2.0, 'neutral': 0.0, 'negative': -2.0}
        return mapping.get(str(sentiment).strip().lower(), 0.0)

    def _score_predictability(self, predictability):
        mapping = {'预披露型': 1.0, '突发型': 0.0}
        return mapping.get(str(predictability).strip(), 0.0)

    def _score_impact_cycle(self, impact_cycle):
        mapping = {'中期型': 0.8, '长尾型': 0.6, '脉冲型': 0.2}
        return mapping.get(str(impact_cycle).strip(), 0.0)

    def calculate_event_aligned_indicators(self, stock_code, event_date, price_df):
        """以事件日期为锚点计算技术指标，避免使用事件后的未来数据。"""
        stock_data = price_df[price_df['code'] == stock_code].sort_values('date').reset_index(drop=True)
        if stock_data.empty:
            return None

        event_dt = pd.to_datetime(event_date, errors='coerce')
        if pd.isna(event_dt):
            anchor_idx = len(stock_data) - 1
        else:
            candidates = stock_data.index[stock_data['date'] <= event_dt]
            if len(candidates) == 0:
                return None
            anchor_idx = int(candidates.max())

        if anchor_idx < 20 or anchor_idx < 5:
            return None

        anchor_close = float(stock_data.loc[anchor_idx, 'close'])
        close_20d_ago = float(stock_data.loc[anchor_idx - 20, 'close'])
        close_5d_ago = float(stock_data.loc[anchor_idx - 5, 'close'])
        close_3d_ago = float(stock_data.loc[anchor_idx - 3, 'close'])
        ma_20 = float(stock_data.loc[anchor_idx - 19:anchor_idx, 'close'].mean())

        if close_20d_ago == 0 or close_5d_ago == 0 or close_3d_ago == 0 or ma_20 == 0:
            return None

        return_20d = (anchor_close - close_20d_ago) / close_20d_ago * 100
        bias_20d = (anchor_close - ma_20) / ma_20 * 100
        momentum_5d = (anchor_close - close_5d_ago) / close_5d_ago * 100
        momentum_3d = (anchor_close - close_3d_ago) / close_3d_ago * 100

        window_3d = stock_data.loc[anchor_idx - 2:anchor_idx, :]
        high_3d = float(window_3d['high'].max()) if 'high' in window_3d.columns else float(window_3d['close'].max())
        low_3d = float(window_3d['low'].min()) if 'low' in window_3d.columns else float(window_3d['close'].min())
        drawdown_3d = (low_3d - high_3d) / high_3d * 100 if high_3d else 0.0

        pct = stock_data.loc[anchor_idx - 19:anchor_idx, 'close'].pct_change().dropna()
        vol_20d = float(pct.std(ddof=1) * 100) if not pct.empty else 0.0

        return {
            'return_20d': return_20d,
            'bias_20d': bias_20d,
            'momentum_5d': momentum_5d,
            'momentum_3d': momentum_3d,
            'drawdown_3d': drawdown_3d,
            'vol_20d': vol_20d,
            'anchor_date': stock_data.loc[anchor_idx, 'date'],
        }

    def _fetch_stock_history_online(self, stock_code, event_date=None, lookback_days=120):
        """在线抓取单票日线，作为本地数据缺失时的回退。"""
        code = str(stock_code).zfill(6)
        event_dt = pd.to_datetime(event_date, errors='coerce') if event_date is not None else pd.NaT
        if pd.isna(event_dt):
            end_dt = pd.Timestamp(datetime.now().date())
        else:
            end_dt = event_dt.normalize()

        cache_key = f"{code}_{end_dt.strftime('%Y%m%d')}"
        if cache_key in self._online_price_cache:
            return self._online_price_cache[cache_key]

        if code.startswith('6'):
            symbol = f"sh{code}"
        elif code.startswith(('0', '3')):
            symbol = f"sz{code}"
        elif code.startswith(('8', '4')):
            symbol = f"bj{code}"
        else:
            self._online_price_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_dt.strftime('%Y%m%d'),
                end_date=end_dt.strftime('%Y%m%d'),
                adjust=''
            )
            if df is None or df.empty:
                self._online_price_cache[cache_key] = pd.DataFrame()
                return pd.DataFrame()

            d = df.copy()
            d['date'] = pd.to_datetime(d['date'], errors='coerce')
            d['open'] = pd.to_numeric(d.get('open', np.nan), errors='coerce')
            d['high'] = pd.to_numeric(d.get('high', np.nan), errors='coerce')
            d['low'] = pd.to_numeric(d.get('low', np.nan), errors='coerce')
            d['close'] = pd.to_numeric(d.get('close', np.nan), errors='coerce')
            d['amount'] = pd.to_numeric(d.get('amount', np.nan), errors='coerce')
            d['code'] = code
            d = d.dropna(subset=['date', 'close']).sort_values('date').reset_index(drop=True)
            self._online_price_cache[cache_key] = d
            return d
        except BaseException:
            self._online_price_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    def _get_price_data_with_fallback(self, stock_code, event_date, price_df):
        """本地优先，缺失或事件时点覆盖不足时改为在线抓取。"""
        code = str(stock_code).zfill(6)
        local = pd.DataFrame()
        if price_df is not None and len(price_df) > 0:
            local = price_df[price_df['code'] == code].copy()

        event_dt = pd.to_datetime(event_date, errors='coerce') if event_date is not None else pd.NaT
        need_online = local.empty
        if not need_online and pd.notna(event_dt):
            local_max = pd.to_datetime(local['date'], errors='coerce').max()
            if pd.isna(local_max) or local_max.normalize() < event_dt.normalize():
                need_online = True

        if need_online:
            online = self._fetch_stock_history_online(code, event_date=event_dt)
            if online is not None and len(online) > 0:
                return online, 'online'

        return local, 'local'

    def _build_composite_score(self, row):
        strength_part = float(row['strength']) * 10.0
        sentiment_part = self._score_sentiment(row.get('sentiment', ''))
        predict_part = self._score_predictability(row.get('predictability', ''))
        cycle_part = self._score_impact_cycle(row.get('impact_cycle', ''))

        tech_part = (
            -0.25 * abs(float(row['bias_20d']))
            -0.10 * max(float(row['return_20d']), 0)
            +0.30 * float(row['momentum_5d'])
            -0.20 * float(row['vol_20d'])
        )

        event_title = str(row.get('event_title', '')).strip()
        event_penalty = 0.0
        for k, v in self.event_title_penalty.items():
            if k in event_title:
                event_penalty = max(event_penalty, float(v))

        return strength_part + sentiment_part + predict_part + cycle_part + tech_part - event_penalty

    def _pick_diversified_top(self, ranked_df, top_n=3):
        """优先按综合分选股，同时限制同一事件标题过度集中。"""
        if ranked_df is None or len(ranked_df) == 0:
            return pd.DataFrame()

        ranked = ranked_df.sort_values(
            ['composite_score', 'strength', 'bias_20d'],
            ascending=[False, False, True]
        ).copy()

        chosen = []
        title_counts = {}
        for _, row in ranked.iterrows():
            title = str(row.get('event_title', '')).strip() or '__EMPTY__'
            if title_counts.get(title, 0) >= self.max_same_event_title:
                continue
            chosen.append(row)
            title_counts[title] = title_counts.get(title, 0) + 1
            if len(chosen) >= top_n:
                break

        if len(chosen) < top_n:
            chosen_codes = {str(r.get('code', '')).strip() for r in chosen}
            for _, row in ranked.iterrows():
                code = str(row.get('code', '')).strip()
                if code in chosen_codes:
                    continue
                chosen.append(row)
                chosen_codes.add(code)
                if len(chosen) >= top_n:
                    break

        return pd.DataFrame(chosen)
    
    def filter_by_financial(self):
        """步骤2: 财务过滤"""
        print("\n" + "="*60)
        print("步骤2: 财务过滤（检查连续两年亏损、净资产为负、ST股票）")
        print("="*60)
        
        healthy_list = []
        total_a_share = 0
        total_checked = 0
        st_count = 0
        
        # 只保留A股
        a_share_stocks = self.stock_pool[self.stock_pool['code'].apply(self.is_a_share)].copy()
        if not a_share_stocks.empty:
            a_share_stocks = a_share_stocks.sort_values('strength', ascending=False).reset_index(drop=True)

        # 分层预筛：优先保留高强度，再保证事件多样性，最后按强度补齐上限。
        if self.finance_prefilter_enabled and self.finance_max_checks > 0 and len(a_share_stocks) > self.finance_max_checks:
            strong_df = a_share_stocks[a_share_stocks['strength'] >= self.finance_keep_above_strength]

            diverse_blocks = []
            if 'event_title' in a_share_stocks.columns:
                for _, grp in a_share_stocks.groupby('event_title', dropna=False):
                    g = grp.sort_values('strength', ascending=False).head(self.finance_per_event_cap)
                    diverse_blocks.append(g)

            if diverse_blocks:
                diverse_df = pd.concat(diverse_blocks, axis=0).drop_duplicates(subset=['code'])
            else:
                diverse_df = a_share_stocks.head(self.finance_max_checks)

            selected = pd.concat([strong_df, diverse_df], axis=0).drop_duplicates(subset=['code'])

            if len(selected) < self.finance_max_checks:
                remain = a_share_stocks[~a_share_stocks['code'].isin(set(selected['code']))]
                pad = remain.head(self.finance_max_checks - len(selected))
                selected = pd.concat([selected, pad], axis=0)

            selected = selected.sort_values('strength', ascending=False).head(self.finance_max_checks).reset_index(drop=True)

            print(
                f"财务预筛: A股候选 {len(a_share_stocks)} -> {len(selected)} "
                f"(max_checks={self.finance_max_checks}, keep_above={self.finance_keep_above_strength}, per_event_cap={self.finance_per_event_cap})"
            )
            a_share_stocks = selected
        
        for idx, row in a_share_stocks.iterrows():
            stock_code = row['code']
            stock_name = row['name']
            strength = row['strength']
            event_title = row['event_title']
            event_id = row['event_id']

            total_a_share += 1
            total_checked += 1
            
            # 先检查是否为ST股票
            if self.is_st_stock(stock_name):
                st_count += 1
                print(f"\n[{total_checked}/{len(a_share_stocks)}] 检查: {stock_code} {stock_name} - ST股票，直接淘汰")
                continue
            
            print(f"\n[{total_checked}/{len(a_share_stocks)}] 检查: {stock_code} {stock_name}", end="")
            
            if self.check_financial_health(stock_code, stock_name):
                healthy_list.append({
                    'code': stock_code,
                    'name': stock_name,
                    'strength': strength,
                    'event_title': event_title,
                    'event_id': event_id,
                    'sentiment': row.get('sentiment', ''),
                    'impact_cycle': row.get('impact_cycle', ''),
                    'predictability': row.get('predictability', ''),
                    'event_date': row.get('event_date', ''),
                })
            
            # 每10只打印分隔线
            if total_checked % 10 == 0:
                print("-" * 40)
        
        self.healthy_stocks = pd.DataFrame(healthy_list)
        print(f"\n\nA股总数: {total_a_share}")
        print(f"ST股票淘汰: {st_count}")
        print(f"通过财务过滤: {len(self.healthy_stocks)} 只")
        
        # 保存健康股票列表
        os.makedirs(self.intermediate_dir, exist_ok=True)
        healthy_path = os.path.join(self.intermediate_dir, 'healthy_stocks.csv')
        self.healthy_stocks.to_csv(healthy_path, index=False, encoding='utf-8-sig')
        
        return self.healthy_stocks
    
    def filter_by_technical(self, price_df):
        """步骤4: 技术面过滤"""
        print("\n" + "="*60)
        print("步骤4: 技术面过滤（20日涨幅>30%或乖离率>15%淘汰）")
        print("="*60)
        
        if price_df is None:
            print("无法加载日线数据，跳过技术面过滤")
            self.final_stocks = self.healthy_stocks.copy()
            self.final_stocks['return_20d'] = 0
            self.final_stocks['bias_20d'] = 0
            return self.final_stocks
        
        final_list = []
        
        for idx, row in self.healthy_stocks.iterrows():
            stock_code = row['code']
            stock_name = row['name']
            strength = row['strength']
            event_title = row['event_title']
            event_id = row['event_id']
            event_date = row.get('event_date', '')
           
            
            print(f"\n[{idx+1}/{len(self.healthy_stocks)}] 检查: {stock_code} {stock_name}", end="")

            if self._is_blocked_event_title(event_title):
                print("  事件标题含停牌关键词，淘汰")
                continue
            
            tradeable, reason = self._is_buyday_tradeable(stock_code)
            if not tradeable:
                print(f"  {reason}，淘汰")
                continue

            stock_price_df, source = self._get_price_data_with_fallback(stock_code, event_date, price_df)
            if stock_price_df is None or len(stock_price_df) == 0:
                print("  本地与在线日线均不可用，淘汰")
                continue
            
            indicators = self.calculate_event_aligned_indicators(str(stock_code).zfill(6), event_date, stock_price_df)
            if indicators is None:
                print(f"  事件时点前数据不足({source})，淘汰")
                continue

            return_20d = indicators['return_20d']
            bias_20d = indicators['bias_20d']
            momentum_5d = indicators['momentum_5d']
            momentum_3d = indicators.get('momentum_3d', 0.0)
            drawdown_3d = indicators.get('drawdown_3d', 0.0)
            vol_20d = indicators['vol_20d']
            anchor_date = indicators['anchor_date']

            print(
                f"  数据源: {source} | 锚点: {anchor_date.date()} | "
                f"20日涨幅: {return_20d:.2f}%, 乖离率: {bias_20d:.2f}%, "
                f"5日动量: {momentum_5d:.2f}%, 3日动量: {momentum_3d:.2f}%, 3日回撤: {drawdown_3d:.2f}%"
            )

            if momentum_5d < self.min_momentum_5d:
                print("  5日动量为负，淘汰")
                continue

            if momentum_3d < self.min_momentum_3d:
                print("  近3日动量过弱，淘汰")
                continue

            if drawdown_3d < self.max_drawdown_3d:
                print("  近3日回撤过大，淘汰")
                continue
            
            if return_20d > 30:
                print(f"  涨幅过高，淘汰")
                continue
            
            if bias_20d > 15:
                print(f"  乖离率过高，淘汰")
                continue
            
            print(f"  通过")
            final_list.append({
                'code': stock_code,
                'name': stock_name,
                'strength': strength,
                'event_title': event_title,
                'event_id': event_id,
                'sentiment': row.get('sentiment', ''),
                'impact_cycle': row.get('impact_cycle', ''),
                'predictability': row.get('predictability', ''),
                'event_date': event_date,
                'return_20d': return_20d,
                'bias_20d': bias_20d,
                'momentum_5d': momentum_5d,
                'momentum_3d': momentum_3d,
                'drawdown_3d': drawdown_3d,
                'vol_20d': vol_20d,
            })
        
        self.final_stocks = pd.DataFrame(final_list)
        print(f"\n\n通过技术过滤: {len(self.final_stocks)} 只")
        
        return self.final_stocks

    def _build_ranked_df(self, df):
        """统一补齐评分所需字段并计算综合评分。"""
        if df is None or len(df) == 0:
            return pd.DataFrame()

        ranked = df.copy()
        defaults = {
            'return_20d': 0.0,
            'bias_20d': 0.0,
            'momentum_5d': 0.0,
            'momentum_3d': 0.0,
            'drawdown_3d': 0.0,
            'vol_20d': 0.0,
            'recent_days': 0,
            'recent_avg_amount': 0.0,
            'recent_momentum_3d': -1.0,
            'recent_max_drawdown_3d': -1.0,
            'sentiment': '',
            'impact_cycle': '',
            'predictability': '',
            'event_date': '',
            'event_title': '',
            'event_id': 0,
            'strength': 0,
        }
        for col, val in defaults.items():
            if col not in ranked.columns:
                ranked[col] = val

        ranked['composite_score'] = ranked.apply(self._build_composite_score, axis=1)
        return ranked

    def _supplement_to_minimum(self, ranked_df, minimum=3):
        """当技术过滤不足3只时，用财务健康池/股票池按综合评分补齐。"""
        if ranked_df is None:
            ranked_df = pd.DataFrame()

        selected = ranked_df.copy()
        selected_codes = set(selected['code'].astype(str).tolist()) if len(selected) > 0 else set()

        def _refine_candidates(candidates_df, strict=True):
            if candidates_df is None or len(candidates_df) == 0:
                return pd.DataFrame()
            c = candidates_df.copy()
            if 'event_title' in c.columns:
                c = c[~c['event_title'].apply(self._is_blocked_event_title)]
            if len(c) == 0:
                return c

            c = c[c['code'].apply(lambda x: self._is_buyday_tradeable(x)[0])]
            if len(c) == 0:
                return c

            features = c['code'].astype(str).apply(self._fetch_recent_trade_features_online)
            c['recent_days'] = features.apply(lambda f: int(f.get('recent_days', 0)) if isinstance(f, dict) else 0)
            c['recent_avg_amount'] = features.apply(lambda f: float(f.get('recent_avg_amount', 0.0)) if isinstance(f, dict) else 0.0)
            c['recent_momentum_3d'] = features.apply(lambda f: float(f.get('recent_momentum_3d', -1.0)) if isinstance(f, dict) else -1.0)
            c['recent_max_drawdown_3d'] = features.apply(lambda f: float(f.get('recent_max_drawdown_3d', -1.0)) if isinstance(f, dict) else -1.0)
            c['recent_ok'] = features.apply(lambda f: bool(f.get('ok', False)) if isinstance(f, dict) else False)

            if strict:
                c = c[c['recent_ok']]
                c = c[c['recent_days'] >= self.min_recent_trading_days]
                c = c[c['recent_avg_amount'] >= self.min_recent_avg_amount]
                c = c[c['recent_max_drawdown_3d'] >= self.max_recent_drawdown_3d]
            return c

        if len(selected) < minimum and self.healthy_stocks is not None and len(self.healthy_stocks) > 0:
            candidates = self.healthy_stocks.copy()
            candidates['code'] = candidates['code'].astype(str)
            candidates = candidates[~candidates['code'].isin(selected_codes)]
            candidates = _refine_candidates(candidates, strict=True)
            ranked_candidates = self._build_ranked_df(candidates)
            if len(ranked_candidates) > 0:
                ranked_candidates = ranked_candidates.sort_values(
                    ['recent_momentum_3d', 'recent_max_drawdown_3d', 'recent_avg_amount', 'composite_score', 'strength', 'bias_20d'],
                    ascending=[False, False, False, False, False, True]
                )
                need = minimum - len(selected)
                if need > 0:
                    selected = pd.concat([selected, ranked_candidates.head(need)], ignore_index=True)
                    selected_codes = set(selected['code'].astype(str).tolist())

            # 严格条件不足时，回退到温和补齐，优先保证事件相关股票数量。
            if len(selected) < minimum:
                relaxed = self.healthy_stocks.copy()
                relaxed['code'] = relaxed['code'].astype(str)
                relaxed = relaxed[~relaxed['code'].isin(selected_codes)]
                relaxed = _refine_candidates(relaxed, strict=False)
                ranked_relaxed = self._build_ranked_df(relaxed)
                if len(ranked_relaxed) > 0:
                    ranked_relaxed = ranked_relaxed.sort_values(
                        ['composite_score', 'strength', 'bias_20d'],
                        ascending=[False, False, True]
                    )
                    need = minimum - len(selected)
                    if need > 0:
                        selected = pd.concat([selected, ranked_relaxed.head(need)], ignore_index=True)
                        selected_codes = set(selected['code'].astype(str).tolist())

        if len(selected) < minimum and self.stock_pool is not None and len(self.stock_pool) > 0:
            candidates = self.stock_pool.copy()
            candidates['code'] = candidates['code'].astype(str)
            candidates = candidates[~candidates['code'].isin(selected_codes)]
            candidates = _refine_candidates(candidates, strict=True)
            ranked_candidates = self._build_ranked_df(candidates)
            if len(ranked_candidates) > 0:
                ranked_candidates = ranked_candidates.sort_values(
                    ['recent_momentum_3d', 'recent_max_drawdown_3d', 'recent_avg_amount', 'composite_score', 'strength', 'bias_20d'],
                    ascending=[False, False, False, False, False, True]
                )
                need = minimum - len(selected)
                if need > 0:
                    selected = pd.concat([selected, ranked_candidates.head(need)], ignore_index=True)
                    selected_codes = set(selected['code'].astype(str).tolist())

            if len(selected) < minimum:
                relaxed = self.stock_pool.copy()
                relaxed['code'] = relaxed['code'].astype(str)
                relaxed = relaxed[~relaxed['code'].isin(selected_codes)]
                relaxed = _refine_candidates(relaxed, strict=False)
                ranked_relaxed = self._build_ranked_df(relaxed)
                if len(ranked_relaxed) > 0:
                    ranked_relaxed = ranked_relaxed.sort_values(
                        ['composite_score', 'strength', 'bias_20d'],
                        ascending=[False, False, True]
                    )
                    need = minimum - len(selected)
                    if need > 0:
                        selected = pd.concat([selected, ranked_relaxed.head(need)], ignore_index=True)
                        selected_codes = set(selected['code'].astype(str).tolist())

        # 最终兜底：若事件映射太窄导致不足3只，用稳定大盘股补齐，保证输出数量。
        if len(selected) < minimum:
            fallback_codes = ['600000', '000001', '601318', '600519', '000333', '002415']
            fallback_rows = []
            for code in fallback_codes:
                if code in selected_codes:
                    continue
                fallback_rows.append({
                    'code': code,
                    'name': code,
                    'strength': 0,
                    'event_title': '兜底补齐',
                    'event_id': 0,
                    'sentiment': 'neutral',
                    'impact_cycle': '',
                    'predictability': '',
                    'event_date': '',
                    'return_20d': 0.0,
                    'bias_20d': 0.0,
                    'momentum_5d': 0.0,
                    'vol_20d': 0.0,
                })
                if len(selected) + len(fallback_rows) >= minimum:
                    break

            if fallback_rows:
                fallback_df = self._build_ranked_df(pd.DataFrame(fallback_rows))
                selected = pd.concat([selected, fallback_df], ignore_index=True)

        return selected.head(minimum)
    
    def select_top3(self):
        """步骤5: 综合评分排序，取前3名"""
        print("\n" + "="*60)
        print("步骤5: 按综合评分排序，取前3名")
        print("="*60)
        
        if self.final_stocks is None:
            print("没有通过所有过滤的股票")
            return None

        ranked = self._build_ranked_df(self.final_stocks)
        if ranked is None or len(ranked) == 0 or 'composite_score' not in ranked.columns:
            if self.allow_supplement:
                print("通过技术过滤结果为空，直接启动补齐机制至至少3只。")
                top3 = self._supplement_to_minimum(pd.DataFrame(), minimum=3)
            else:
                print("通过技术过滤结果为空，且当前为宁缺毋滥模式，不补齐。")
                top3 = pd.DataFrame()
        else:
            top3 = self._pick_diversified_top(ranked, top_n=3)

        if len(top3) < 3 and self.allow_supplement:
            print(f"通过技术过滤仅 {len(top3)} 只，启动补齐机制至至少3只。")
            top3 = self._supplement_to_minimum(top3, minimum=3)
        elif len(top3) < 3:
            print(f"通过技术过滤仅 {len(top3)} 只，当前为宁缺毋滥模式，不补齐。")

        if len(top3) == 0:
            print("没有可用于补齐的股票")
            return None
        
        print("\n最终选出的 TOP 3 股票:")
        for i, (idx, row) in enumerate(top3.iterrows(), 1):
            print(f"  {i}. {row['code']} {row['name']} - strength: {row['strength']}")
            print(f"     综合评分: {row['composite_score']:.2f}")
            print(f"     20日涨幅: {row['return_20d']:.2f}%, 乖离率: {row['bias_20d']:.2f}%")
        
        return top3
    
    def run(self):
        """运行完整的风控流程"""
        print("="*60)
        print("高阶风控过滤器 RiskFilter")
        print("="*60)
        
        # 1. 加载股票池（从数据库和JSON获取strength）
        self.load_stock_pool_from_db_and_json()
        
        # 2. 财务过滤（支持从已有 healthy_stocks.csv 续跑）
        healthy_cache_file = os.path.join(self.intermediate_dir, 'healthy_stocks.csv')
        use_cache = self.trade_week_start is None
        if self.stock_pool is None or len(self.stock_pool) == 0:
            print("当前周无可用股票池，跳过缓存续跑并清空健康股票列表。")
            self.healthy_stocks = pd.DataFrame(columns=['code','name','strength','event_title','event_id','sentiment','impact_cycle','predictability','event_date'])
        elif use_cache and os.path.exists(healthy_cache_file):
            try:
                self.healthy_stocks = pd.read_csv(healthy_cache_file, dtype={'code': str})
                if self.stock_pool is not None and len(self.stock_pool) > 0:
                    valid_codes = set(self.stock_pool['code'].astype(str).tolist())
                    self.healthy_stocks = self.healthy_stocks[
                        self.healthy_stocks['code'].astype(str).isin(valid_codes)
                    ].copy()
                self.enrich_healthy_stocks_metadata()
                print(f"\n检测到 {healthy_cache_file}，已加载 {len(self.healthy_stocks)} 只健康股票，跳过财务过滤")
                if self.healthy_stocks is None or len(self.healthy_stocks) == 0:
                    print("缓存健康股票与本周股票池无交集，重新执行财务过滤。")
                    self.filter_by_financial()
            except Exception as e:
                print(f"读取 {healthy_cache_file} 失败，重新执行财务过滤: {e}")
                self.filter_by_financial()
        else:
            if not use_cache:
                print("滚动按周模式：不复用 artifacts/intermediate/healthy_stocks.csv，按当周股票池重新执行财务过滤。")
            self.filter_by_financial()
        
        # 3. 加载日线数据
        price_df = self.load_price_data()
        self._price_df_for_trade_check = price_df
        
        # 4. 技术面过滤
        self.filter_by_technical(price_df)
        
        # 5. 选前3名
        top3 = self.select_top3()
        
        # 6. 保存结果
        self.save_result(top3)
        
        return top3
    
    def is_st_stock(self, stock_name):
        """
        判断是否为ST股票
        ST股票通常名称中包含 ST、*ST、SST 等字样
        """
        if not stock_name:
            return False
        
        st_patterns = ['ST', '*ST', 'SST', 'ST*']
        stock_name_upper = str(stock_name).upper()
        
        for pattern in st_patterns:
            if pattern in stock_name_upper:
                return True
        return False
    
    def save_result(self, top3, output_file=None):
        """保存结果，包含事件ID"""
        if top3 is None or len(top3) == 0:
            print("没有符合条件的股票，不保存文件")
            return

        def _build_score_ref(row):
            return (
                f"事件强度={row['strength']}; 情绪={row.get('sentiment', '')}; "
                f"影响周期={row.get('impact_cycle', '')}; 可预测性={row.get('predictability', '')}"
            )

        def _build_reason(row):
            return (
                f"由事件《{row['event_title']}》(ID={row['event_id']})触发，强度{row['strength']}分；"
                f"通过财务与ST过滤；20日涨幅{row['return_20d']:.2f}%<=30%，"
                f"20日乖离率{row['bias_20d']:.2f}%<=15%。"
            )

        def _build_rank_reason(row):
            return (
                f"综合分={row['composite_score']:.2f}（强度主分+情绪/周期/可预测性+技术面修正）；"
                f"技术面: 5日动量{row['momentum_5d']:.2f}%, 20日波动率{row['vol_20d']:.2f}%"
            )

        top3 = top3.copy()
        top3['score_ref'] = top3.apply(_build_score_ref, axis=1)
        top3['selection_reason'] = top3.apply(_build_reason, axis=1)
        top3['rank_reason'] = top3.apply(_build_rank_reason, axis=1)
        
        # 创建新的DataFrame，使用中文列名
        result_df = pd.DataFrame({
            '股票代码': top3['code'],
            '股票名称': top3['name'],
            '强度': top3['strength'],
            '事件标题': top3['event_title'],
            '对应事件ID': top3['event_id'],
            '事件日期': top3['event_date'],
            '评分依据参考': top3['score_ref'],
            '入选原因': top3['selection_reason'],
            '综合评分': top3['composite_score'].round(2),
            '排序依据': top3['rank_reason'],
            '5日动量(%)': top3['momentum_5d'].round(2),
            '20日波动率(%)': top3['vol_20d'].round(2),
            '20日涨幅(%)': top3['return_20d'].round(2),
            '20日乖离率(%)': top3['bias_20d'].round(2)
        })
        
        # 保存
        if output_file is None:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            output_file = os.path.join(self.intermediate_dir, 'safe_top3_stocks.csv')
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        
        # 打印简洁版
        print("\n" + "="*70)
        print("本周推荐买入的3只股票:")
        print("="*70)
        for i, (idx, row) in enumerate(result_df.iterrows(), 1):
            print(f"{i}. {row['股票代码']} {row['股票名称']}")
            print(f"   强度: {row['强度']}")
            print(f"   综合评分: {row['综合评分']:.2f}")
            print(f"   对应事件ID: {row['事件标题']} - {row['对应事件ID']}")
            print(f"   20日涨幅: {row['20日涨幅(%)']:.2f}%")
            print(f"   20日乖离率: {row['20日乖离率(%)']:.2f}%")
            print()


def main():
    """主函数"""
    # 创建风险过滤器
    filter = RiskFilter(
        db_path=os.path.join(BASE_DIR, 'data', 'db', 'structured_events.db'),
        json_file=os.path.join(BASE_DIR, 'data', 'events', 'events_history.json'),
        price_data_file='日线数据.csv'
    )
    
    # 运行过滤
    top3 = filter.run()
    
    print("\n" + "="*60)
    print("风控过滤完成！")
    print("="*60)


if __name__ == "__main__":
    main()