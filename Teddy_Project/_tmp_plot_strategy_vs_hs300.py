import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from em_historical_selector import DEFAULT_PRICE_FILE


def fetch_hs300_from_eastmoney(start_date: str, end_date: str) -> pd.DataFrame:
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': '1.000300',
        'klt': '101',
        'fqt': '0',
        'beg': start_date.replace('-', ''),
        'end': end_date.replace('-', ''),
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    data = (j.get('data') or {}).get('klines') or []
    rows = []
    for line in data:
        parts = line.split(',')
        if len(parts) >= 3:
            rows.append((parts[0], parts[2]))
    df = pd.DataFrame(rows, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().sort_values('date')


def main():
    weekly_files = [
        Path('result/em_weekly_recommendation_2025-12-08.csv'),
        Path('result/em_weekly_recommendation_2025-12-15_before_tighten.csv'),
        Path('result/pnl_buy_2025-12-23_sell_2025-12-25.csv'),
    ]

    price = pd.read_csv(DEFAULT_PRICE_FILE)
    price_df = pd.DataFrame({
        'code': price.iloc[:, 0].astype(str).str.zfill(6),
        'date': pd.to_datetime(price.iloc[:, 1], errors='coerce').dt.normalize(),
        'close': pd.to_numeric(price.iloc[:, 5], errors='coerce'),
    }).dropna(subset=['date', 'close'])

    all_starts, all_ends = [], []
    for f in weekly_files:
        d = pd.read_csv(f)
        if d.empty:
            continue
        all_starts.append(pd.to_datetime(d.iloc[0]['trade_week_start']).strftime('%Y-%m-%d'))
        all_ends.append(pd.to_datetime(d.iloc[0]['sell_date']).strftime('%Y-%m-%d'))

    hs300 = fetch_hs300_from_eastmoney(min(all_starts), max(all_ends))
    if hs300.empty:
        raise RuntimeError('东方财富接口未返回沪深300数据')

    saved = []
    for f in weekly_files:
        df = pd.read_csv(f)
        if df.empty:
            continue

        row0 = df.iloc[0]
        week_start = pd.to_datetime(row0['trade_week_start']).normalize()
        buy_date = pd.to_datetime(row0['buy_date']).normalize()
        sell_date = pd.to_datetime(row0['sell_date']).normalize()

        bench = hs300[(hs300['date'] >= week_start) & (hs300['date'] <= sell_date)].copy()
        if bench.empty:
            continue
        bench['hs300_nav'] = bench['close'] / float(bench.iloc[0]['close'])

        initial_capital = float(df['allocated_amount'].sum())
        holdings = []
        for _, r in df.iterrows():
            code = str(r['code']).zfill(6)
            alloc = float(r['allocated_amount'])
            buy_open = pd.to_numeric(r.get('tuesday_open', np.nan), errors='coerce')
            if pd.isna(buy_open) or float(buy_open) <= 0:
                continue
            shares = alloc / float(buy_open)
            s = price_df[(price_df['code'] == code) & (price_df['date'] >= buy_date) & (price_df['date'] <= sell_date)][['date', 'close']].copy().sort_values('date')
            holdings.append((shares, s))

        nav_rows = []
        for d in bench['date']:
            if d < buy_date:
                nav_rows.append((d, 1.0))
                continue
            total_value = 0.0
            for shares, s in holdings:
                if s.empty:
                    continue
                sub = s[s['date'] <= d]
                if sub.empty:
                    continue
                total_value += shares * float(sub.iloc[-1]['close'])
            nav_rows.append((d, total_value / initial_capital if total_value > 0 else np.nan))

        strat = pd.DataFrame(nav_rows, columns=['date', 'strategy_nav'])
        plot_df = bench[['date', 'hs300_nav']].merge(strat, on='date', how='left')
        plot_df['strategy_nav'] = plot_df['strategy_nav'].ffill().fillna(1.0)

        plt.figure(figsize=(10, 5.2))
        plt.plot(plot_df['date'], plot_df['strategy_nav'], marker='o', linewidth=2.2, label='Strategy NAV')
        plt.plot(plot_df['date'], plot_df['hs300_nav'], marker='o', linewidth=2.0, label='HS300 NAV')
        plt.title(f'Strategy vs HS300 | {week_start.date()} to {sell_date.date()}')
        plt.xlabel('Date')
        plt.ylabel('Normalized NAV')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()

        out_file = Path('result') / f'strategy_vs_hs300_{week_start.date()}_to_{sell_date.date()}.png'
        plt.savefig(out_file, dpi=180)
        plt.close()
        saved.append(out_file.as_posix())

    print('saved_plots:')
    for s in saved:
        print(s)


if __name__ == '__main__':
    main()
