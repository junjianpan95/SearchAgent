# ============================================================
#  SearchAgent Web Demo - Start (PowerShell, foreground)
#  Usage:  .\start.ps1          (default port 8000)
#          .\start.ps1 -Port 9000
#  If blocked by execution policy, run:
#     powershell -ExecutionPolicy Bypass -File .\start.ps1
# ============================================================
param([int]$Port = 8000)

$Root = $PSScriptRoot
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "============================================================"
Write-Host " SearchAgent Web Demo"
Write-Host " URL         : http://127.0.0.1:$Port"
Write-Host " Interpreter : $py"
Write-Host " Stop        : press Ctrl+C"
Write-Host "============================================================"

& $py -m http.server $Port --bind 127.0.0.1 --directory (Join-Path $Root "web")
