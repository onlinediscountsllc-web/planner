# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 ONE-CLICK DEPLOYMENT - v13.1 Bugfix Patch
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🔧 DEPLOYING v13.1 BUGFIX PATCH" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Fixes:" -ForegroundColor White
Write-Host "  ✅ Particle Swarm endpoint (400 → 200)" -ForegroundColor Green
Write-Host "  ✅ Binaural Beat audio (404 → WAV download)" -ForegroundColor Green
Write-Host ""

# Navigate to planner directory
cd C:\Users\Luke\Desktop\planner

# Check if fixed file exists
if (Test-Path "life_fractal_v13_FIXED.py") {
    Write-Host "[1/4] Found v13.1 FIXED file ✅" -ForegroundColor Green
} else {
    Write-Host "[1/4] ERROR: life_fractal_v13_FIXED.py not found!" -ForegroundColor Red
    Write-Host "      Please download it from Claude first." -ForegroundColor Yellow
    exit 1
}

# Copy to app.py
Write-Host "[2/4] Copying to app.py..." -ForegroundColor Yellow
Copy-Item life_fractal_v13_FIXED.py -Destination app.py -Force
Write-Host "      Done ✅" -ForegroundColor Green

# Git operations
Write-Host "[3/4] Committing to git..." -ForegroundColor Yellow
git add app.py
git commit -m "v13.1: Fix particle swarm 400 error and binaural audio 404 error"
Write-Host "      Done ✅" -ForegroundColor Green

# Push to GitHub
Write-Host "[4/4] Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
Write-Host "      Done ✅" -ForegroundColor Green

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✨ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Render is now rebuilding... (~2 minutes)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test after deployment:" -ForegroundColor White
Write-Host "  curl 'https://planner-1-pyd9.onrender.com/api/math/particle-swarm?energy=0.7'" -ForegroundColor Yellow
Write-Host "  curl 'https://planner-1-pyd9.onrender.com/api/audio/binaural/focus?duration=10.0' -o focus.wav" -ForegroundColor Yellow
Write-Host ""
Write-Host "Dashboard: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host ""
