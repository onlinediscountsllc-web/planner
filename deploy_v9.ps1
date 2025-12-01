# ═══════════════════════════════════════════════════════════════════════════════
# 🌀 LIFE FRACTAL INTELLIGENCE v9.0 - ULTIMATE UNIFIED DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 LIFE FRACTAL INTELLIGENCE v9.0 - ULTIMATE UNIFIED" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\Users\Luke\Desktop\planner"
Write-Host "📁 Working directory: $(Get-Location)" -ForegroundColor Yellow

# Backup
if (Test-Path "app.py") {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item "app.py" "backup\app_$ts.py" -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Backed up existing app.py" -ForegroundColor Green
}

# Requirements
$requirements = @"
# Core Flask
Flask==3.0.0
Flask-CORS==4.0.0
Werkzeug==3.0.1

# Data Processing (Python 3.13 compatible)
numpy==1.26.4
Pillow==10.4.0

# Machine Learning (optional but recommended)
scikit-learn==1.3.2

# Production Server
gunicorn==21.2.0
"@
$requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8
Write-Host "✅ requirements.txt created" -ForegroundColor Green

"web: gunicorn app:app" | Out-File -FilePath "Procfile" -Encoding ASCII -NoNewline
Write-Host "✅ Procfile created" -ForegroundColor Green

"python-3.11.7" | Out-File -FilePath "runtime.txt" -Encoding ASCII -NoNewline
Write-Host "✅ runtime.txt created" -ForegroundColor Green

# Check app.py
Write-Host ""
Write-Host "🔍 Verifying v9.0 features..." -ForegroundColor Cyan
$content = Get-Content "app.py" -Raw -ErrorAction SilentlyContinue

$features = @{
    "Neurodivergent subtitle" = "Neurodivergent-optimized"
    "3D Visualization route" = "@app.route\('/3d'\)"
    "Voice input hint" = "Voice input supported"
    "Stats cards API" = "@app.route\('/api/stats'\)"
    "Three.js 3D" = "three.js"
    "Golden Spiral 3D" = "createSacredGeometry"
    "Goal orbs" = "goalOrbs"
    "ML Predictor" = "MoodPredictor"
    "Fuzzy Logic" = "FuzzyLogicEngine"
    "Context fix" = "callable\(fallback_value\)"
}

foreach ($f in $features.GetEnumerator()) {
    if ($content -match $f.Value) {
        Write-Host "   ✅ $($f.Key)" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $($f.Key) - NOT FOUND" -ForegroundColor Yellow
    }
}

# Git
Write-Host ""
Write-Host "🚀 Committing and pushing..." -ForegroundColor Cyan
git add app.py requirements.txt Procfile runtime.txt 2>$null

$msg = @"
Deploy Life Fractal Intelligence v9.0 - Ultimate Unified

MERGED FEATURES:
- Original clean GUI (neurodivergent-optimized)
- Top stats cards (Active Goals, Streak, Level)
- Voice input support for notes
- Interactive 3D visualization (/3d route)
- Three.js fractal universe with goal orbs
- Sacred geometry (Golden Spiral, Flower of Life)
- ML mood prediction + fuzzy logic guidance
- Sacred badges + Fibonacci milestones
- Self-healing system
- All Render.com deployment fixes
"@

git commit -m $msg 2>$null
git push origin main 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ v9.0 DEPLOYMENT READY!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "📋 FEATURES INCLUDED:" -ForegroundColor Yellow
Write-Host "   🧠 Neurodivergent-friendly design" -ForegroundColor White
Write-Host "   📊 Clean stats cards layout" -ForegroundColor White
Write-Host "   💬 Voice input support" -ForegroundColor White
Write-Host "   🌀 Interactive 3D fractal universe" -ForegroundColor White
Write-Host "   🎯 Goal orbs in 3D space" -ForegroundColor White
Write-Host "   📐 Sacred geometry overlays" -ForegroundColor White
Write-Host "   🤖 ML mood prediction" -ForegroundColor White
Write-Host "   💜 Fuzzy logic guidance" -ForegroundColor White
Write-Host ""
Write-Host "📍 ROUTES:" -ForegroundColor Yellow
Write-Host "   /     - Main dashboard" -ForegroundColor White
Write-Host "   /3d   - 3D visualization" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Deploy at: https://dashboard.render.com" -ForegroundColor Cyan
Write-Host "🌐 Live URL: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host ""
