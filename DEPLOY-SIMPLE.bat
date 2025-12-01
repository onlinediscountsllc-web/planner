@echo off
REM LIFE FRACTAL INTELLIGENCE - SIMPLE DEPLOYMENT
REM Double-click this file to deploy everything!

echo.
echo ========================================================================
echo          LIFE FRACTAL INTELLIGENCE - DEPLOYMENT
echo               Enhanced Features v2.0.0
echo ========================================================================
echo.
echo.

echo This will deploy your enhanced Life Fractal Intelligence!
echo.
echo What it does:
echo   1. Patches your code with new features
echo   2. Deploys to Render.com
echo   3. Tests the deployment
echo.

set /p confirm="Ready to deploy? (y/n): "
if /i not "%confirm%"=="y" (
    echo.
    echo Deployment cancelled.
    echo.
    pause
    exit /b 0
)

echo.
echo ========================================================================
echo Starting deployment...
echo ========================================================================
echo.

REM Run the PowerShell script with execution policy bypass
PowerShell -ExecutionPolicy Bypass -File "%~dp0ONE-CLICK-DEPLOY.ps1"

if %errorlevel% equ 0 (
    echo.
    echo ========================================================================
    echo SUCCESS! Deployment completed successfully!
    echo ========================================================================
    echo.
    echo Check your Render dashboard: https://dashboard.render.com/
    echo.
) else (
    echo.
    echo ========================================================================
    echo Deployment encountered an error. Check the output above.
    echo ========================================================================
    echo.
)

pause
