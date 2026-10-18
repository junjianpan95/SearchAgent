@echo off
chcp 65001 >nul
REM ============================================================
REM  SearchAgent 一键启动（前端 + 后端）
REM  开两个独立窗口：后端(8001) + 前端(8000)
REM  停止：分别关闭那两个窗口即可
REM ============================================================
setlocal
cd /d "%~dp0"

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [SearchAgent] 正在启动，请稍候...

REM 后端走 backend.bat（前台窗口，能看日志，关窗口即停）
start "SearchAgent-Backend(8001)" cmd /k "%~dp0backend.bat"

REM 前端走独立窗口（前台 http.server）
start "SearchAgent-Web(8000)" cmd /k ""%PY%" -m http.server 8000 --bind 127.0.0.1 --directory "%~dp0web""

timeout /t 4 /nobreak >nul

echo.
echo ============================================================
echo  已启动两个窗口！
echo   前端页面 : http://127.0.0.1:8000
echo   后端接口 : http://127.0.0.1:8001/api/health
echo   停止方式 : 分别关闭那两个弹出的窗口
echo ============================================================
echo.
pause
endlocal
