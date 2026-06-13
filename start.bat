@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .\.venv\Scripts\activate.bat
)

echo ========================================
echo Security LLM Platform - frontend
echo ========================================
echo.
echo URL: http://localhost:8501
echo.
echo Demo accounts:
echo   admin / Admin#2026
echo   analyst / Analyst#2026
echo   researcher / Research#2026
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

streamlit run streamlit_app.py --server.headless true --browser.gatherUsageStats false
