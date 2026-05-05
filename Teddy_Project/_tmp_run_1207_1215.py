from em_historical_selector import run_pipeline, DEFAULT_DB_PATH, DEFAULT_PRICE_FILE
run_pipeline(
    start_date='2025-12-07',
    end_date='2025-12-15',
    capital=102830.01,
    db_path=DEFAULT_DB_PATH,
    price_file=DEFAULT_PRICE_FILE,
    event_source='llm',
    llm_provider='deepseek',
)
