@echo off
REM Life Planner - One-Click Setup and Run (Windows)
REM This script does EVERYTHING automatically

echo ====================================================================
echo LIFE PLANNER - AUTOMATIC SETUP AND LAUNCH
echo ====================================================================
echo.
echo This will automatically:
echo   1. Create folder structure
echo   2. Install all dependencies
echo   3. Configure the application
echo   4. Initialize the database
echo   5. Start the server
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

cls

REM ============================================================================
REM STEP 1: Check Python
REM ============================================================================

echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM ============================================================================
REM STEP 2: Create/Activate Virtual Environment
REM ============================================================================

echo [2/7] Setting up virtual environment...
if not exist "venv\" (
    echo Creating new virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Virtual environment activated
echo.

REM ============================================================================
REM STEP 3: Upgrade pip
REM ============================================================================

echo [3/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Pip upgraded
echo.

REM ============================================================================
REM STEP 4: Install Dependencies
REM ============================================================================

echo [4/7] Installing dependencies (this may take a few minutes)...
echo Installing core packages...

REM Install packages one by one for better error handling
python -m pip install Flask==3.0.0 --quiet
python -m pip install Flask-SQLAlchemy==3.1.1 --quiet
python -m pip install Flask-CORS==4.0.0 --quiet
python -m pip install python-dotenv==1.0.0 --quiet
python -m pip install Werkzeug==3.0.1 --quiet
python -m pip install SQLAlchemy==2.0.23 --quiet

echo Installing optional packages...
python -m pip install Flask-JWT-Extended Flask-Mail bcrypt stripe Pillow requests email-validator --quiet 2>nul

echo Installing ML packages (may take longer)...
python -m pip install numpy scikit-learn --quiet 2>nul

echo Dependencies installed (some optional packages may have been skipped)
echo.

REM ============================================================================
REM STEP 5: Run Setup Script
REM ============================================================================

echo [5/7] Running self-healing setup...

if exist "setup_and_fix.py" (
    python setup_and_fix.py
) else (
    echo Setup script not found, creating folders manually...
    if not exist "models\" mkdir models
    if not exist "backend\" mkdir backend
    if not exist "templates\" mkdir templates
    if not exist "logs\" mkdir logs
    if not exist "static\" mkdir static
    
    REM Move files if needed
    if exist "database.py" (
        if not exist "models\database.py" move database.py models\ >nul 2>&1
    )
    if exist "life_planning_core.py" (
        if not exist "backend\life_planning_core.py" move life_planning_core.py backend\ >nul 2>&1
    )
    if exist "gpu_extensions.py" (
        if not exist "backend\gpu_extensions.py" move gpu_extensions.py backend\ >nul 2>&1
    )
    if exist "index.html" (
        if not exist "templates\index.html" move index.html templates\ >nul 2>&1
    )
    
    REM Create __init__.py files
    type nul > models\__init__.py
    type nul > backend\__init__.py
)
echo.

REM ============================================================================
REM STEP 6: Create .env if missing
REM ============================================================================

echo [6/7] Checking configuration...

if not exist ".env" (
    echo Creating .env file...
    (
        echo # Life Planner Configuration
        echo SECRET_KEY=auto-generated-change-for-production
        echo JWT_SECRET_KEY=auto-generated-jwt-key
        echo DEBUG=True
        echo DATABASE_URL=sqlite:///life_planner.db
        echo ADMIN_EMAIL=onlinediscountsllc@gmail.com
        echo ADMIN_PASSWORD=admin8587037321
        echo GOFUNDME_URL=https://gofund.me/8d9303d27
        echo SUBSCRIPTION_PRICE=20.00
        echo TRIAL_DAYS=7
        echo USE_GPU=False
        echo RATELIMIT_STORAGE_URL=memory://
        echo CORS_ORIGINS=http://localhost:5000
        echo LOG_LEVEL=INFO
        echo.
        echo # Add your Stripe keys here:
        echo STRIPE_SECRET_KEY=sk_test_YOUR_KEY
        echo STRIPE_PUBLISHABLE_KEY=pk_test_YOUR_KEY
        echo STRIPE_PRICE_ID=price_YOUR_PRICE_ID
        echo.
        echo # Add your Gmail settings here:
        echo MAIL_USERNAME=your-email@gmail.com
        echo MAIL_PASSWORD=your-app-password
    ) > .env
    echo Created .env file (edit to add Stripe/Gmail credentials)
) else (
    echo .env file exists
)
echo.

REM ============================================================================
REM STEP 7: Initialize Database
REM ============================================================================

echo [7/7] Initializing database...

if exist "init_db.py" (
    REM Check if database already exists
    if exist "life_planner.db" (
        echo Database already exists
    ) else (
        echo Creating database...
        echo 1| python init_db.py
    )
) else (
    echo Database initialization script not found
    echo Database will be created on first run
)
echo.

REM ============================================================================
REM FINAL: Start the application
REM ============================================================================

cls
echo ====================================================================
echo SETUP COMPLETE! STARTING LIFE PLANNER...
echo ====================================================================
echo.
echo Application will start in a moment...
echo.
echo Once started:
echo   - Open browser: http://localhost:5000
echo   - Admin email: onlinediscountsllc@gmail.com
echo   - Admin password: admin8587037321
echo.
echo IMPORTANT:
echo   - Edit .env file to add Stripe and Gmail credentials
echo   - Change admin password after first login
echo.
echo Press Ctrl+C to stop the server
echo.
echo ====================================================================
echo.

timeout /t 3 >nul

REM Check which app file to use
if exist "app_refactored.py" (
    echo Starting with refactored app...
    python app_refactored.py
) else if exist "app.py" (
    echo Starting with standard app...
    python app.py
) else (
    echo ERROR: No app.py file found!
    echo Please ensure app.py or app_refactored.py exists
    pause
    exit /b 1
)

REM If we get here, the app stopped
echo.
echo Application stopped.
pause
