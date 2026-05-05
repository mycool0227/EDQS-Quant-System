param(
    [string]$StartDate,
    [string]$EndDate
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "em_historical_selector.py"

# LLM batching defaults:
# Always process all news (no truncation), but allow override from env.
if (-not $env:LLM_MAX_ITEMS) { $env:LLM_MAX_ITEMS = "0" }
if (-not $env:LLM_BATCH_SIZE) { $env:LLM_BATCH_SIZE = "1" }
if (-not $env:LLM_DELAY_SEC) { $env:LLM_DELAY_SEC = "1.2" }
if (-not $env:DEEPSEEK_MAX_RETRIES) { $env:DEEPSEEK_MAX_RETRIES = "6" }
if (-not $env:DEEPSEEK_CONNECT_TIMEOUT) { $env:DEEPSEEK_CONNECT_TIMEOUT = "30" }
if (-not $env:DEEPSEEK_READ_TIMEOUT) { $env:DEEPSEEK_READ_TIMEOUT = "180" }
$llmProvider = if ($env:LLM_PROVIDER) { $env:LLM_PROVIDER } else { "auto" }

if (-not (Test-Path $pythonExe)) {
    Write-Host "Python not found: $pythonExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $scriptPath)) {
    Write-Host "Script not found: $scriptPath" -ForegroundColor Red
    exit 1
}

if (-not $StartDate) {
    $StartDate = Read-Host "Input start date (YYYY-MM-DD)"
}
if (-not $EndDate) {
    $EndDate = Read-Host "Input end date (YYYY-MM-DD)"
}

$pattern = '^\d{4}-\d{2}-\d{2}$'
if ($StartDate -notmatch $pattern -or $EndDate -notmatch $pattern) {
    Write-Host "Invalid date format. Use YYYY-MM-DD." -ForegroundColor Red
    exit 1
}

try {
    $startObj = [DateTime]::ParseExact($StartDate, "yyyy-MM-dd", $null)
    $endObj = [DateTime]::ParseExact($EndDate, "yyyy-MM-dd", $null)
    if ($startObj -gt $endObj) {
        Write-Host "Start date cannot be later than end date." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Date parse failed. Check input values." -ForegroundColor Red
    exit 1
}

Write-Host "Running: $StartDate -> $EndDate" -ForegroundColor Cyan
Write-Host "LLM batching: max_items=$env:LLM_MAX_ITEMS, batch_size=$env:LLM_BATCH_SIZE, delay_sec=$env:LLM_DELAY_SEC" -ForegroundColor DarkCyan
Write-Host "LLM provider: $llmProvider (DeepSeek异常时自动回退)" -ForegroundColor DarkCyan
& $pythonExe $scriptPath --start-date $StartDate --end-date $EndDate --llm-provider $llmProvider

if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Done. Check files under result/." -ForegroundColor Green
