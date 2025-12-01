@echo off
REM 🔧 FIX POWERSHELL EXECUTION POLICY
REM Run this ONCE to enable PowerShell scripts

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                               ║
echo ║        🔧 POWERSHELL EXECUTION POLICY FIX 🔧                  ║
echo ║                                                               ║
echo ║    This will enable PowerShell scripts to run                 ║
echo ║                                                               ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.
echo.

echo This script will:
echo   1. Unblock all PowerShell scripts in this directory
echo   2. Set execution policy to RemoteSigned (safe for local scripts)
echo.
echo This is safe and standard for development.
echo.

set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo Step 1: Unblocking PowerShell files...
echo ═══════════════════════════════════════════════════════════════
echo.

PowerShell -Command "Get-ChildItem -Path . -Filter *.ps1 | Unblock-File"
if %errorlevel% equ 0 (
    echo ✅ PowerShell files unblocked
) else (
    echo ⚠️  Could not unblock files ^(may need admin^)
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo Step 2: Setting execution policy...
echo ═══════════════════════════════════════════════════════════════
echo.

PowerShell -Command "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"
if %errorlevel% equ 0 (
    echo ✅ Execution policy set to RemoteSigned
) else (
    echo ⚠️  Could not set execution policy
    echo    Try running as Administrator
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo ✅ FIX COMPLETE!
echo ═══════════════════════════════════════════════════════════════
echo.
echo You can now run:
echo   .\ONE-CLICK-DEPLOY.ps1
echo.
echo Or simply double-click: DEPLOY.bat
echo.

pause
