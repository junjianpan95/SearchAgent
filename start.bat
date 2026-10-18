@echo off
REM ============================================================
REM  SearchAgent Web Demo - Start (Windows CMD / double-click)
REM  Runs in the foreground. Keep this window open.
REM  Stop: press Ctrl+C in this window, or just close it.
REM
REM  Usage:  start.bat           (default port 8000)
REM          start.bat 9000      (custom port)
REM ============================================================
setlocal
cd /d "%~dp0"

set PORT=8000
if not "%~1"=="" set PORT=%~1

REM Use the project venv interpreter; fall back to system python
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo ============================================================
echo  SearchAgent Web Demo
echo  URL         : http://127.0.0.1:%PORT%
echo  Interpreter : %PY%
echo  Stop        : press Ctrl+C in this window
echo ============================================================
echo.

"%PY%" -m http.server %PORT% --bind 127.0.0.1 --directory "%~dp0web"

endlocal
