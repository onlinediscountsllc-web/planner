@echo off
REM ============================================
REM Life Fractal Intelligence v10.0
REM Git Push Script for Windows
REM ============================================

echo.
echo ========================================
echo  Life Fractal Intelligence v10.0
echo  Git Push Script
echo ========================================
echo.

REM Check if we're in a git repository
git status >nul 2>&1
if errorlevel 1 (
    echo ERROR: Not in a git repository!
    echo Please navigate to your planner folder first.
    pause
    exit /b 1
)

echo Current directory: %cd%
echo.

REM Show status
echo Checking git status...
git status

echo.
echo ========================================
echo Files to be committed:
echo ========================================
echo  - life_fractal_v10.py (main app)
echo  - requirements.txt (dependencies)
echo  - Procfile (process config)
echo  - render.yaml (Render settings)
echo  - runtime.txt (Python version)
echo  - README.md (documentation)
echo  - .gitignore (ignore rules)
echo ========================================
echo.

REM Stage all files
echo Staging all files...
git add -A

echo.
echo Files staged. Here's what will be committed:
git status --short

echo.
set /p confirm="Ready to commit and push? (y/n): "
if /i not "%confirm%"=="y" (
    echo Aborted.
    pause
    exit /b 0
)

REM Commit
echo.
echo Committing changes...
git commit -m "v10.0: Complete rewrite with all features working - Fixed deployment"

REM Push
echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ========================================
echo  DONE! Changes pushed to GitHub.
echo ========================================
echo.
echo Next steps:
echo  1. Go to Render dashboard
echo  2. Check deployment status
echo  3. Wait for build to complete
echo  4. Test at: https://planner-1-pyd9.onrender.com
echo.
echo If auto-deploy is off, click "Manual Deploy"
echo.

pause
