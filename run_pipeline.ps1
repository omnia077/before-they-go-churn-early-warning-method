# ============================================================
# run_pipeline.ps1  —  One-command pipeline runner (Windows)
#
# Usage:  .\run_pipeline.ps1
#
# Prerequisites (see README for install steps):
#   - Java 11+ on PATH
#   - Python 3.12 with requirements.txt installed
#   - winutils.exe + hadoop.dll at C:\hadoop\bin\
# ============================================================

param([switch]$SkipGeneration)

Set-StrictMode -Version Latest

# ---- Required environment variables for PySpark on Windows ----
$env:HADOOP_HOME            = "C:\hadoop"
$env:PATH                   = "C:\hadoop\bin;" + $env:PATH
$env:PYSPARK_PYTHON         = "python"
$env:PYSPARK_DRIVER_PYTHON  = "python"
$env:PYTHONUTF8             = "1"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$SRC  = Join-Path $ROOT "src"
$LINE = "=" * 60

# ---- Prerequisite check ----
if (-not (Test-Path "C:\hadoop\bin\winutils.exe")) {
    Write-Host "[ERROR] C:\hadoop\bin\winutils.exe not found." -ForegroundColor Red
    Write-Host "        See the 'Windows Setup' section in README.md." -ForegroundColor Red
    exit 1
}

# ---- Phase runner ----
function Invoke-Phase ([string]$Label, [string]$Dir, [string]$Script) {
    Write-Host ""
    Write-Host $LINE                -ForegroundColor DarkGray
    Write-Host "  $Label"           -ForegroundColor Cyan
    Write-Host $LINE                -ForegroundColor DarkGray
    Push-Location $Dir
    python $Script
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Write-Host ""
        Write-Host "[FAIL] $Label — exit code $code" -ForegroundColor Red
        exit 1
    }
    Write-Host "[PASS] $Label" -ForegroundColor Green
}

# ---- Pipeline ----
if (-not $SkipGeneration) {
    Invoke-Phase "Phase 1 — Data Generation"               (Join-Path $SRC "01_generate_data")  "run_phase1.py"
} else {
    Write-Host "Skipping Phase 1 - using existing data in src\data" -ForegroundColor Yellow
}
Invoke-Phase "Phase 2 — ETL / Raw Layer"               (Join-Path $SRC "02_raw_ingest")          "run_phase2.py"
Invoke-Phase "Phase 4 — Spark Classification + Iceberg" (Join-Path $SRC "04_classify_spark")       "iceberg_writer.py"
Invoke-Phase "Phase 5 — Star Schema"                   (Join-Path $SRC "05_star_schema")  "phase5_star_schema_builder.py"
Invoke-Phase "Phase 6 — Analytics Mart"                (Join-Path $SRC "06_analytics_mart")    "build_mart.py"
Invoke-Phase "Phase 7 — Pipeline Monitor"              (Join-Path $SRC "07_monitoring")   "pipeline_monitor.py"

Write-Host ""
Write-Host $LINE                              -ForegroundColor Green
Write-Host "  PIPELINE COMPLETE — all phases passed" -ForegroundColor Green
Write-Host $LINE                              -ForegroundColor Green
