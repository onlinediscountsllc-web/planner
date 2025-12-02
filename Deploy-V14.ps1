# ============================================================
# Life Fractal v14.0 - ULTIMATE UNIFIED ENGINE
# PowerShell Deployment Script
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Life Fractal v14.0 - ULTIMATE UNIFIED ENGINE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location -Path "C:\Users\Luke\Desktop\planner"
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Yellow

# Step 1: Backup
Write-Host ""
Write-Host "[1/5] Backing up current app.py..." -ForegroundColor Green
if (Test-Path "app.py") {
    $backupName = "app_backup_v13_$(Get-Date -Format 'yyyyMMdd_HHmmss').py"
    Copy-Item "app.py" -Destination $backupName
    Write-Host "Backup created: $backupName" -ForegroundColor Gray
}

# Step 2: Copy v14
Write-Host ""
Write-Host "[2/5] Copying v14 to app.py..." -ForegroundColor Green
if (Test-Path "life_fractal_v14_ultimate.py") {
    Copy-Item "life_fractal_v14_ultimate.py" -Destination "app.py" -Force
    Write-Host "app.py updated with v14" -ForegroundColor Gray
} else {
    Write-Host "ERROR: life_fractal_v14_ultimate.py not found!" -ForegroundColor Red
    Write-Host "Please download it from Claude first." -ForegroundColor Yellow
    exit 1
}

# Step 3: Update requirements
Write-Host ""
Write-Host "[3/5] Ensuring requirements.txt is correct..." -ForegroundColor Green
@"
Flask>=3.0.0
Flask-Cors>=4.0.0
Werkzeug>=3.0.0
gunicorn>=21.0.0
numpy>=2.0.0
Pillow>=10.0.0
requests>=2.31.0
python-dotenv>=1.0.0
"@ | Out-File -FilePath "requirements.txt" -Encoding ASCII -Force
Write-Host "requirements.txt updated" -ForegroundColor Gray

# Step 4: Git add and commit
Write-Host ""
Write-Host "[4/5] Staging and committing..." -ForegroundColor Green
git add app.py requirements.txt life_fractal_v14_ultimate.py
git status
git commit -m "v14.0 - ULTIMATE UNIFIED ENGINE: All features integrated, zero placeholders"

# Step 5: Push
Write-Host ""
Write-Host "[5/5] Pushing to GitHub (triggers Render auto-deploy)..." -ForegroundColor Green
git push origin main

# Done
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Render will auto-deploy. Check:" -ForegroundColor Yellow
Write-Host "  Dashboard: https://dashboard.render.com"
Write-Host "  Live site: https://planner-1-pyd9.onrender.com"
Write-Host ""
Write-Host "v14.0 UNIFIED FEATURES:" -ForegroundColor Cyan
Write-Host "  ✅ 13 Life Domains (with Law of Attraction)"
Write-Host "  ✅ 18 Task Types (full effect vectors)"
Write-Host "  ✅ 39 Spillover Effects"
Write-Host "  ✅ 19 Life Milestones"
Write-Host "  ✅ 8 Virtual Pet Species"
Write-Host "  ✅ Bellman Optimization (Q-values)"
Write-Host "  ✅ Flow State Calculation"
Write-Host "  ✅ Fractal Math (Dimension, Hurst, Lyapunov)"
Write-Host "  ✅ Compound Growth Projections"
Write-Host "  ✅ Fibonacci Scheduling"
Write-Host "  ✅ Golden Ratio Allocation"
Write-Host "  ✅ Habit Formation Tracking"
Write-Host "  ✅ Journal System"
Write-Host "  ✅ Goals System"
Write-Host "  ✅ Energy/Spoon Management"
Write-Host "  ✅ Full Responsive Frontend"
Write-Host "  ✅ 50+ API Endpoints"
Write-Host "  ✅ Zero Placeholders"
Write-Host ""
Write-Host "Sacred Math:" -ForegroundColor Magenta
Write-Host "  phi = 1.618033988749895"
Write-Host "  Golden Angle = 137.5077640500378 deg"
Write-Host ""

Read-Host "Press Enter to exit"
