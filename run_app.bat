@echo off
setlocal
title E-Commerce Customer Loyalty Analytics - Streamlit Dashboard

echo ======================================================================
echo   E-Commerce Customer Loyalty Analytics Dashboard
echo ======================================================================
echo.

:: 1. Navigate to project root directory
cd /d "%~dp0"
echo [OK] Working Directory: %CD%

:: 2. Check Python availability
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ or add Python to your system PATH.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -V 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python Environment: %PYTHON_VERSION%

:: 3. Check Streamlit module availability
python -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Streamlit is not installed in the active Python environment.
    echo Run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 4. Verify .env presence (informational only)
if exist ".env" (
    echo [OK] Configuration: .env loaded
) else (
    echo [INFO] .env not found. Dashboard will run in deterministic local fallback mode.
)

echo.
echo Launching dashboard at http://localhost:8501 ...
echo Press Ctrl+C in this terminal to stop the server when finished.
echo.

python -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with error code: %ERRORLEVEL%
    pause
)
