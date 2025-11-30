@echo off
REM Life Planner Startup Script for Windows

echo ====================================
echo Life Planner Application Startup
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [!] Virtual environment not found. Creating...
    python -m venv venv
    echo [+] Virtual environment created
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate

REM Check if .env exists
if not exist ".env" (
    echo [!] .env file not found!
    echo [!] Copying .env.template to .env
    copy .env.template .env
    echo.
    echo [!] IMPORTANT: Edit .env file with your configuration before continuing!
    echo [!] Press any key to exit and configure .env...
    pause >nul
    exit /b 1
)

REM Install/update dependencies
echo [*] Checking dependencies...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Failed to install dependencies
    pause
    exit /b 1
)
echo [+] Dependencies installed

REM Create logs directory
if not exist "logs\" mkdir logs

REM Check if database is initialized
python -c "from app import app, db; import os; app.app_context().push(); exit(0 if os.path.exists('life_planner.db') or db.engine.url.database else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Database not initialized
    echo [*] Run 'python init_db.py' to set up the database
    pause
    exit /b 1
)

echo.
echo ====================================
echo Starting Life Planner Application
echo ====================================
echo.
echo [*] Access at: http://localhost:5000
echo [*] Admin: onlinediscountsllc@gmail.com / admin8587037321
echo [*] Press Ctrl+C to stop
echo.

REM Start the application
python app.py

pause
