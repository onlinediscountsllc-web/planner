# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 COMPLETE FIX + DEPLOY v14.0
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Red
Write-Host "🚨 FIXING RENDER CONFIGURATION ISSUE" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Red
Write-Host ""

Write-Host "Problem Found:" -ForegroundColor Red
Write-Host "  Render is running: 'gunicorn life_fractal_v10:app'" -ForegroundColor Red
Write-Host "  Should be running: 'gunicorn app:app'" -ForegroundColor Red
Write-Host ""

# Navigate
cd C:\Users\Luke\Desktop\planner

# Create Procfile to fix permanently
Write-Host "[1/5] Creating Procfile to fix start command..." -ForegroundColor Yellow
"web: gunicorn app:app" | Out-File -FilePath "Procfile" -Encoding ASCII -NoNewline
Write-Host "      Done ✅" -ForegroundColor Green

# Verify app.py exists
Write-Host "[2/5] Verifying app.py exists..." -ForegroundColor Yellow
if (Test-Path "app.py") {
    $firstLine = Get-Content "app.py" -First 1
    Write-Host "      Found: $firstLine" -ForegroundColor Green
} else {
    Write-Host "      ERROR: app.py not found!" -ForegroundColor Red
    Write-Host "      Deploy v14.0 first!" -ForegroundColor Red
    exit 1
}

# Commit
Write-Host "[3/5] Committing Procfile..." -ForegroundColor Yellow
git add Procfile
git commit -m "Fix: Add Procfile with correct start command (app:app)"
Write-Host "      Done ✅" -ForegroundColor Green

# Push
Write-Host "[4/5] Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
Write-Host "      Done ✅" -ForegroundColor Green

# Wait for rebuild
Write-Host "[5/5] Waiting for Render to rebuild..." -ForegroundColor Yellow
Write-Host "      This takes ~3 minutes..." -ForegroundColor Gray
Write-Host ""

# Show manual step
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "⚠️  MANUAL STEP REQUIRED (One Time Only)" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "If this doesn't work, also update Render Dashboard:" -ForegroundColor White
Write-Host "  1. Go to: https://dashboard.render.com" -ForegroundColor Yellow
Write-Host "  2. Select 'planner' service" -ForegroundColor Yellow
Write-Host "  3. Click 'Settings' tab" -ForegroundColor Yellow
Write-Host "  4. Find 'Start Command'" -ForegroundColor Yellow
Write-Host "  5. Change to: gunicorn app:app" -ForegroundColor Yellow
Write-Host "  6. Click 'Save Changes'" -ForegroundColor Yellow
Write-Host ""

# Test commands
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🧪 TEST COMMANDS (Run after 3 minutes)" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "# Health check (should show version 14.0)" -ForegroundColor White
Write-Host "curl 'https://planner-1-pyd9.onrender.com/api/health'" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Particle swarm (should work now)" -ForegroundColor White
Write-Host "curl 'https://planner-1-pyd9.onrender.com/api/math/particle-swarm?energy=0.7&wellness=0.7'" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Binaural audio (should work now)" -ForegroundColor White
Write-Host "curl 'https://planner-1-pyd9.onrender.com/api/audio/binaural/focus?duration=5.0' -o test.wav" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Virtual pet (NEW in v14)" -ForegroundColor White
Write-Host "curl 'https://planner-1-pyd9.onrender.com/api/pet'" -ForegroundColor Yellow
Write-Host ""

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ Fix deployed! Wait 3 minutes then test above commands" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
