@echo off
REM ============================================================
REM  SearchAgent Web Demo - Start in BACKGROUND (minimized window)
REM  Stop: run stop.bat
REM
REM  Usage:  start_bg.bat           (default port 8000)
REM          start_bg.bat 9000      (custom port)
REM ============================================================
setlocal
cd /d "%~dp0"

set PORT=8000
if not "%~1"=="" set PORT=%~1

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [SearchAgent] Starting web demo in background on port %PORT% ...
start "SearchAgent-Web" /min cmd /c ""%PY%" -m http.server %PORT% --bind 127.0.0.1 --directory "%~dp0web""

REM Give the server a moment, then probe it
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%' -UseBasicParsing -TimeoutSec 3).StatusCode } catch { 'unreachable' }"

echo [SearchAgent] URL    : http://127.0.0.1:%PORT%
echo [SearchAgent] Stop   : stop.bat %PORT%
endlocal
