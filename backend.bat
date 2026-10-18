@echo off
chcp 65001 >nul
REM ============================================================
REM  SearchAgent 在线问答后端 - Start (Windows CMD)
REM  启动 Flask 后端，提供 /api/ask 在线问答接口
REM  默认端口 8001
REM
REM  用法:
REM    backend.bat            (规则模式，无需模型，秒启动)
REM    backend.bat --model    (加载 Qwen3-1.7B 真实模型推理)
REM ============================================================
setlocal
cd /d "%~dp0"

set PORT=8001
set MODE=--rule
if "%~1"=="--model" set MODE=

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo ============================================================
echo  SearchAgent 在线问答后端
echo  URL    : http://127.0.0.1:%PORT%
echo  Mode   : %MODE% (空=加载真实模型)
echo  Stop   : 关闭本窗口 或 Ctrl+C
echo ============================================================

"%PY%" scripts\backend.py %MODE% --port %PORT%
set ERR=%ERRORLEVEL%

echo.
echo [后端已退出] 退出码 %ERR%
if not "%ERR%"=="0" (
    echo 常见原因：端口 %PORT% 已被占用，或依赖缺失。
    echo 排查：netstat -ano ^| findstr :%PORT%  查看占用进程
)
pause
endlocal
