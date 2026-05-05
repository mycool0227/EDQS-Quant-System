# Teddy Project

## Quick Start (Windows)

### 1) Create a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Prepare required data files
Ensure these files exist at the project root:
- 日线数据.csv
- 概念板块.csv

### 4) Configure LLM API key (at least one)
```powershell
$env:DEEPSEEK_API_KEY="your_key"
# or
$env:ZHIPU_API_KEY="your_key"
# or
$env:GOOGLE_API_KEY="your_key"
```
Optional: force provider
```powershell
$env:LLM_PROVIDER="deepseek"  # deepseek | zhipu | gemini | auto
```

### 5) Run the main pipeline
```powershell
.\run_by_dates.ps1 -StartDate 2026-04-12 -EndDate 2026-04-20
```

### 6) Check outputs
Results are written to the result/ folder.

## Notes
- If you change dates frequently, keep the virtual environment activated.
- The script will download news data online; make sure your network can access the sources.
