@echo off
REM ============================================================
REM  SearchAgent Web Demo - Stop (Windows CMD / double-click)
REM  Usage:  stop.bat          (default port 8000)
REM          stop.bat 9000     (custom port)
REM ============================================================
setlocal enabledelayedexpansion
set PORT=8000
if not "%~1"=="" set PORT=%~1

set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo [SearchAgent] Stopping PID %%a on port %PORT%
    taskkill /F /PID %%a >nul 2>&1
    set FOUND=1
)

if "!FOUND!"=="0" echo [SearchAgent] No service listening on port %PORT%
endlocal
