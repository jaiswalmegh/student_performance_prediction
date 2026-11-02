
@echo off
title Student Performance AI Studio

echo.
echo  ==============================================
echo       STUDENT PERFORMANCE AI STUDIO
echo  ==============================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...

python --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not added to PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo Python detected successfully.
echo.

echo [2/3] Activating virtual environment if available...

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo Virtual environment activated.
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    echo Virtual environment activated.
) else (
    echo No virtual environment found. Using installed Python packages.
)

echo.
echo [3/3] Launching Student Performance AI Studio...
echo.

python -m streamlit run main.py

if errorlevel 1 (
    echo.
    echo Application stopped due to an error.
    pause
)

    