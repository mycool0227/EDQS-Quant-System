import pandas as pd
from pathlib import Path
from rolling_weekly_backtest import compute_dynamic_weights
from em_historical_selector import fetch_week_return_with_fallback, DEFAULT_PRICE_FILE

capital = 102830.01
trade_week_start = pd.Timestamp('2025-12-15')

in_file = Path(r'./result/top3_from_healthy_new_rules_2025-12-15.csv')
out_file = Path(r'./result/em_weekly_recommendation_2025-12-15_from_healthy_cap10283001.csv')

if not in_file.exists():
    raise FileNotFoundError(str(in_file))

raw = pd.read_csv(in_file, encoding='utf-8-sig')
if raw is None or len(raw) == 0:
    raise ValueError('top3 file empty')

# Column mapping from Chinese export
col_code = '股票代码' if '股票代码' in raw.columns else 'code'
col_name = '股票名称' if '股票名称' in raw.columns else 'name'
col_strength = '强度' if '强度' in raw.columns else 'strength'
col_event_title = '事件标题' if '事件标题' in raw.columns else 'event_title'
col_event_date = '事件日期' if '事件日期' in raw.columns else 'event_date'
col_score = '综合评分' if '综合评分' in raw.columns else 'composite_score'
col_vol = '20日波动率(%)' if '20日波动率(%)' in raw.columns else ('vol_20d' if 'vol_20d' in raw.columns else None)

if col_vol is None:
    # Fallback: detect a volatility-like column
    for c in raw.columns:
        if '波动率' in str(c):
            col_vol = c
            break
if col_vol is None:
    raise ValueError('volatility column not found')

sel = pd.DataFrame({
    'code': raw[col_code].astype(str).str.zfill(6),
    'name': raw[col_name].astype(str),
    'strength': pd.to_numeric(raw[col_strength], errors='coerce').fillna(0.0),
    'event_title': raw[col_event_title].astype(str),
    'event_date': raw[col_event_date].astype(str),
    'composite_score': pd.to_numeric(raw[col_score], errors='coerce').fillna(0.0),
    'vol_20d': pd.to_numeric(raw[col_vol], errors='coerce').fillna(0.0),
}).head(3).copy()

weighted = compute_dynamic_weights(
    sel,
    score_beta=1.0,
    mix_lambda=0.5,
    min_weight=0.10,
    max_weight=0.40,
).copy()

weighted['trade_week_start'] = trade_week_start.date()
weighted['buy_date'] = (trade_week_start + pd.Timedelta(days=1)).date()
weighted['sell_date'] = (trade_week_start + pd.Timedelta(days=4)).date()
weighted['allocated_amount'] = weighted['weight'] * float(capital)

price_df = pd.read_csv(DEFAULT_PRICE_FILE, encoding='utf-8')
price_df = price_df.rename(columns={'代码': 'code', '日期': 'date', '开盘': 'open', '收盘': 'close'})
price_df['code'] = price_df['code'].astype(str).str.split('.').str[0].str.zfill(6)
price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
price_df['open'] = pd.to_numeric(price_df['open'], errors='coerce')
price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
price_df = price_df.dropna(subset=['date', 'open', 'close'])

week_return = 0.0
for idx, row in weighted.iterrows():
    info = fetch_week_return_with_fallback(price_df, row['code'], trade_week_start)
    if info is None:
        weighted.loc[idx, 'weekly_return'] = None
        weighted.loc[idx, 'price_source'] = 'missing'
        continue
    r = float(info['weekly_return'])
    weighted.loc[idx, 'weekly_return'] = r
    weighted.loc[idx, 'tuesday_open'] = float(info['tuesday_open'])
    weighted.loc[idx, 'friday_close'] = float(info['friday_close'])
    weighted.loc[idx, 'price_source'] = info.get('price_source', 'local')
    week_return += float(row['weight']) * r

weighted['final_capital'] = float(capital) * (1.0 + week_return)
weighted['pnl'] = weighted['allocated_amount'] * pd.to_numeric(weighted['weekly_return'], errors='coerce').fillna(0.0)

keep = [
    'trade_week_start','buy_date','sell_date','code','name','event_title','event_date',
    'strength','composite_score','vol_20d','score_weight','risk_weight','weight',
    'allocated_amount','tuesday_open','friday_close','weekly_return','price_source','pnl','final_capital'
]
for c in keep:
    if c not in weighted.columns:
        weighted[c] = None

out = weighted[keep].sort_values('weight', ascending=False).reset_index(drop=True)
out_file.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_file, index=False, encoding='utf-8-sig')

print('output=', out_file)
print('init_capital=', f'{capital:.2f}')
print('portfolio_return=', f'{week_return:.6f}')
print('final_capital=', f'{float(capital) * (1.0 + week_return):.2f}')
print(out[['code','name','weight','weekly_return','pnl']].to_string(index=False))
