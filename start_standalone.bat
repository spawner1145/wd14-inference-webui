@echo off
setlocal

cd /d "%~dp0"

echo Starting wd14-inference-webui...
echo URL: http://127.0.0.1:7860
echo.

python main.py
if errorlevel 1 (
    echo.
    echo Python launcher failed. Trying "py"...
    py main.py
)

echo.
pause
