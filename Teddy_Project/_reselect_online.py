import pandas as pd
from pathlib import Path
from RiskFilter import RiskFilter
from em_historical_selector import compute_dynamic_weights, _fetch_week_return_online

base = Path('.')
healthy_path = base / 'artifacts' / 'intermediate' / 'healthy_stocks.csv'
out_path = base / 'result' / 'em_weekly_recommendation_2025-12-08_online.csv'

hs = pd.read_csv(healthy_path, dtype={'code': str})
for c in ['code','name','strength','event_title','event_id','sentiment','impact_cycle','predictability','event_date']:
    if c not in hs.columns:
        hs[c] = '' if c not in ['strength','event_id'] else 0

rf = RiskFilter(trade_week_start='2025-12-08', event_window_start='2025-12-01', event_window_end='2025-12-08')
rf.allow_supplement = True
rf.healthy_stocks = hs.copy()
empty_local = pd.DataFrame(columns=['code', 'date', 'open', 'close'])
rf._price_df_for_trade_check = empty_local
rf.filter_by_technical(empty_local)
top3 = rf.select_top3()

if top3 is None or len(top3) == 0:
    raise SystemExit('No top3 generated.')

top3 = top3.copy()
top3['code'] = top3['code'].astype(str).str.zfill(6)
top3['name'] = top3['name'].fillna('').astype(str)
non_st = top3[~top3['name'].str.upper().str.replace(' ', '', regex=False).str.contains('ST')].copy()
top3 = non_st.head(3).copy()

weighted = compute_dynamic_weights(top3, score_beta=1.0, mix_lambda=1.0, min_weight=0.10, max_weight=0.40).copy()
trade_week_start = pd.Timestamp('2025-12-08')
weighted['trade_week_start'] = trade_week_start.date()
weighted['buy_date'] = (trade_week_start + pd.Timedelta(days=1)).date()
weighted['sell_date'] = (trade_week_start + pd.Timedelta(days=4)).date()
weighted['allocated_amount'] = weighted['weight'] * 100000.0
weighted['selection_reason'] = weighted.apply(lambda r: f"由事件《{str(r.get('event_title', '')).strip()}》触发; 强度={float(r.get('strength', 0)):.2f}; 综合分={float(r.get('composite_score', 0)):.2f}", axis=1)

week_return = 0.0
for idx, row in weighted.iterrows():
    info = _fetch_week_return_online(row['code'], trade_week_start)
    if info is None:
        weighted.loc[idx, 'weekly_return'] = None
        weighted.loc[idx, 'price_source'] = 'online-missing'
        continue
    r = float(info['weekly_return'])
    weighted.loc[idx, 'weekly_return'] = r
    weighted.loc[idx, 'tuesday_open'] = float(info['tuesday_open'])
    weighted.loc[idx, 'friday_close'] = float(info['friday_close'])
    weighted.loc[idx, 'price_source'] = 'online'
    week_return += float(row['weight']) * r

weighted['final_capital'] = 100000.0 * (1.0 + week_return)
keep_cols = ['trade_week_start','buy_date','sell_date','code','name','event_title','event_date','selection_reason','strength','composite_score','vol_20d','score_weight','risk_weight','weight','allocated_amount','tuesday_open','friday_close','weekly_return','price_source','final_capital']
for c in keep_cols:
    if c not in weighted.columns:
        weighted[c] = None
weighted[keep_cols].sort_values('weight', ascending=False).reset_index(drop=True).to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'Online reselection output: {out_path}')
print(weighted[['code','name','weight','strength','composite_score','weekly_return','price_source']].to_string(index=False))
print(f'portfolio_week_return={week_return:.4%}, final_capital={100000.0*(1.0+week_return):.2f}')
