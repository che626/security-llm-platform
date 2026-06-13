@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo Security LLM Platform - setup
echo ========================================
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .\.venv\Scripts\activate

echo.
echo [3/4] Installing core dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Creating runtime directories...
if not exist "outputs" mkdir "outputs"
if not exist "outputs\benchmark" mkdir "outputs\benchmark"
if not exist "outputs\search" mkdir "outputs\search"
if not exist "logs" mkdir "logs"

echo.
echo ========================================
echo Setup complete.
echo ========================================
echo.
echo Start frontend:
echo   start.bat
echo.
echo Optional backend:
echo   scripts\start_backend.bat
echo.
echo Demo accounts:
echo   admin / Admin#2026
echo   analyst / Analyst#2026
echo   researcher / Research#2026
echo.
echo Optional ML stack:
echo   pip install -r requirements-ml.txt
echo.
pause
