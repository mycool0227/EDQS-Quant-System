import pandas as pd
from RiskFilter import RiskFilter

healthy_path = r'./artifacts/intermediate/healthy_stocks.csv'
out_path = r'./result/top3_from_healthy_new_rules_2025-12-15.csv'

rf = RiskFilter(price_data_file='日线数据.csv', trade_week_start='2025-12-15')
rf.healthy_stocks = pd.read_csv(healthy_path, dtype={'code': str})
rf.healthy_stocks['code'] = rf.healthy_stocks['code'].astype(str).str.zfill(6)

price_df = rf.load_price_data()
rf._price_df_for_trade_check = price_df
rf.filter_by_technical(price_df)
top3 = rf.select_top3()
rf.save_result(top3, output_file=out_path)

print('healthy_rows=', len(rf.healthy_stocks))
print('final_after_technical=', 0 if rf.final_stocks is None else len(rf.final_stocks))
print('top3_rows=', 0 if top3 is None else len(top3))
print('output=', out_path)
