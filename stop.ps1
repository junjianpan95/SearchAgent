# ============================================================
#  SearchAgent Web Demo - Stop (PowerShell)
#  Kills whatever is listening on the given port.
#  Usage:  .\stop.ps1          (default port 8000)
#          .\stop.ps1 -Port 9000
# ============================================================
param([int]$Port = 8000)

$found = $false
netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING" | ForEach-Object {
    $procId = ($_ -split '\s+')[-1]
    Write-Host "[SearchAgent] Stopping PID $procId on port $Port"
    taskkill /F /PID $procId 2>$null | Out-Null
    $found = $true
}
if (-not $found) { Write-Host "[SearchAgent] No service listening on port $Port" }
