@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0\.."

if exist ".venv\Scripts\activate.bat" (
    call .\.venv\Scripts\activate.bat
)

streamlit run streamlit_app.py --server.headless true --browser.gatherUsageStats false
