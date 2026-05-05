import pandas as pd
import os


def _is_a_share_code(code):
    c = str(code).strip().zfill(6)
    return c.startswith(('000', '001', '002', '003', '004', '300', '301', '600', '601', '603', '605', '688', '689', '8'))

# ==========================================
# 配置与映射
# ==========================================

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 统一股票匹配数据源：由 概念板块.xlsx 转换得到的 CSV
CONCEPT_CSV_PATH = os.path.join(BASE_DIR, '概念板块.csv')

# ==========================================
# 核心函数
# ==========================================

def match_stocks(keywords_list, require_concept_match=False, min_keyword_hits=1):
    """
    根据输入的关键词列表，匹配相关的股票代码和名称，并记录每个股票匹配到的关键词。
    
    Returns:
        list: 匹配到的股票信息列表，每个元素为字典格式：
        [
            {
                'code': '002085', 
                'name': '万丰奥威',
                'matched_keywords': ['低空经济', '飞行汽车']  # 该股票具体匹配到的关键词
            },
            ...
        ]
    """
    if not isinstance(keywords_list, list):
        print("Error: keywords_list must be a list")
        return []
    
    # 使用字典存储股票信息，key为股票代码，value为股票信息和匹配到的关键词集合
    stocks_dict = {}  # {code: {'name': name, 'matched_keywords': set()}}
    
    # 使用单一概念板块数据源
    if not os.path.exists(CONCEPT_CSV_PATH):
        print(f"Warning: Concept CSV not found: {CONCEPT_CSV_PATH}")
        return []

    # 尝试不同编码读取CSV
    df = None
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'gb2312'):
        try:
            df = pd.read_csv(CONCEPT_CSV_PATH, dtype={'股票代码': str}, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading concept csv ({encoding}): {e}")
            return []

    if df is None or df.empty:
        return []

    required_cols = {'概念名称', '股票代码', '股票名称'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Error: Missing required columns in {CONCEPT_CSV_PATH}")
        return []

    # 清洗基础字段
    df = df[['概念名称', '股票代码', '股票名称']].copy()
    df['概念名称'] = df['概念名称'].astype(str).str.strip()
    df['股票名称'] = df['股票名称'].astype(str).str.strip()
    df['股票代码'] = (
        df['股票代码']
        .astype(str)
        .str.replace('.0', '', regex=False)
        .str.strip()
        .str.zfill(6)
    )
    df = df.dropna(subset=['股票代码'])
    df = df.drop_duplicates(subset=['概念名称', '股票代码'])
    
    for keyword in keywords_list:
        keyword = str(keyword).strip()
        if not keyword:
            continue
            
        # 概念优先匹配：先命中概念名称，再在需要时用股票名称补充。
        try:
            concept_mask = df['概念名称'].str.contains(keyword, na=False, case=False)
            name_mask = df['股票名称'].str.contains(keyword, na=False, case=False)

            subset = df[concept_mask].copy()
            if subset.empty and not require_concept_match:
                subset = df[name_mask].copy()
            elif not require_concept_match:
                extra = df[name_mask & (~concept_mask)].copy()
                if not extra.empty:
                    subset = pd.concat([subset, extra], ignore_index=True)

            if subset.empty:
                continue

            for _, row in subset.iterrows():
                code = str(row['股票代码']).strip().zfill(6)
                if not code:
                    continue
                if not _is_a_share_code(code):
                    continue
                if code not in stocks_dict:
                    stocks_dict[code] = {
                        'name': str(row.get('股票名称', '')).strip(),
                        'matched_keywords': set()
                    }
                stocks_dict[code]['matched_keywords'].add(keyword)
        except Exception as e:
            print(f"Error matching keyword {keyword}: {e}")
            continue
    
    # 在返回结果前
    result = []
    min_hits = max(1, int(min_keyword_hits))
    for code, info in stocks_dict.items():
        if len(info['matched_keywords']) < min_hits:
            continue
        # 格式化为6位
        formatted_code = str(code).zfill(6)
        result.append({
            'code': formatted_code,
            'name': info['name'],
            'matched_keywords': list(info['matched_keywords'])
        })
    return result


def save_results_to_csv(stock_codes, output_file='matched_stocks.csv'):
    """
    保存匹配到的股票代码到 CSV 文件
    """
    if not stock_codes:
        print("未匹配到股票，跳过保存。")
        return

    # 转换为DataFrame
    df = pd.DataFrame(stock_codes)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(df)} 个股票代码至: {output_file}")


# ==========================================
# 测试入口
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("graph_matcher.py 单独测试")
    print("=" * 60)
    
    # 测试关键词
    test_keywords = ['低空经济', '中标', '无人机']
    print(f"测试关键词: {test_keywords}")
    
    result = match_stocks(test_keywords)
    print(f"匹配到 {len(result)} 只股票")
    print("前5只股票信息:")
    for i, stock in enumerate(result[:5]):
        print(f"  {i+1}. {stock['code']} - {stock['name']}")
    
    # 可选：保存结果
    save_choice = input("\n是否保存结果到文件? (y/n): ")
    if save_choice.lower() == 'y':
        save_results_to_csv(result, 'test_match_results.csv')