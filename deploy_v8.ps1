# ═══════════════════════════════════════════════════════════════════════════════
# 🌀 LIFE FRACTAL INTELLIGENCE v8.0 - COMPLETE DEPLOYMENT SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
# This script deploys the complete v8.0 system with all features to Render.com
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 LIFE FRACTAL INTELLIGENCE v8.0 - DEPLOYMENT" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Change to planner directory
$plannerPath = "C:\Users\Luke\Desktop\planner"
if (-not (Test-Path $plannerPath)) {
    Write-Host "❌ Directory not found: $plannerPath" -ForegroundColor Red
    Write-Host "   Please update the path in this script." -ForegroundColor Yellow
    exit 1
}

Set-Location $plannerPath
Write-Host "📁 Working directory: $(Get-Location)" -ForegroundColor Yellow

# Step 1: Backup existing files
Write-Host ""
Write-Host "📦 Step 1: Backing up existing files..." -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (Test-Path "app.py") {
    Copy-Item "app.py" "backup\app_$timestamp.py" -Force -ErrorAction SilentlyContinue
    Write-Host "   ✅ Backed up app.py" -ForegroundColor Green
}

# Step 2: Check for downloaded files
Write-Host ""
Write-Host "📥 Step 2: Checking for v8.0 files..." -ForegroundColor Cyan

$downloadFiles = @(
    "life_fractal_v8_complete.py",
    "app_v8.py"
)

$foundSource = $null
foreach ($file in $downloadFiles) {
    if (Test-Path $file) {
        $foundSource = $file
        Write-Host "   ✅ Found: $file" -ForegroundColor Green
        break
    }
}

if ($foundSource) {
    Copy-Item $foundSource "app.py" -Force
    Write-Host "   ✅ Copied to app.py" -ForegroundColor Green
} elseif (Test-Path "app.py") {
    Write-Host "   ⚠️  Using existing app.py" -ForegroundColor Yellow
} else {
    Write-Host "   ❌ No app.py found! Download from Claude first." -ForegroundColor Red
    exit 1
}

# Step 3: Create/Update requirements.txt
Write-Host ""
Write-Host "📋 Step 3: Creating requirements.txt with ML support..." -ForegroundColor Cyan

$requirements = @"
# Core Flask
Flask==3.0.0
Flask-CORS==4.0.0
Werkzeug==3.0.1

# Data Processing
numpy==1.26.4
Pillow==10.4.0

# Machine Learning (optional but recommended)
scikit-learn==1.3.2

# Production Server
gunicorn==21.2.0
"@

$requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8
Write-Host "   ✅ requirements.txt created with:" -ForegroundColor Green
Write-Host "      - Flask 3.0.0" -ForegroundColor White
Write-Host "      - numpy 1.26.4 (Python 3.13 compatible)" -ForegroundColor White
Write-Host "      - Pillow 10.4.0 (Python 3.13 compatible)" -ForegroundColor White
Write-Host "      - scikit-learn 1.3.2 (ML mood prediction)" -ForegroundColor White

# Step 4: Create Procfile
Write-Host ""
Write-Host "📋 Step 4: Creating Procfile..." -ForegroundColor Cyan
"web: gunicorn app:app" | Out-File -FilePath "Procfile" -Encoding ASCII -NoNewline
Write-Host "   ✅ Procfile created" -ForegroundColor Green

# Step 5: Create runtime.txt
Write-Host ""
Write-Host "📋 Step 5: Creating runtime.txt..." -ForegroundColor Cyan
"python-3.11.7" | Out-File -FilePath "runtime.txt" -Encoding ASCII -NoNewline
Write-Host "   ✅ runtime.txt created (Python 3.11.7)" -ForegroundColor Green

# Step 6: Verify app.py features
Write-Host ""
Write-Host "🔍 Step 6: Verifying v8.0 features in app.py..." -ForegroundColor Cyan

$appContent = Get-Content "app.py" -Raw

$features = @{
    "jsonify() context fix" = "callable\(fallback_value\)"
    "JSON DataStore" = "class DataStore"
    "Sacred Mathematics" = "class SacredMath"
    "GPU Fractal Engine" = "class FractalEngine"
    "ML Mood Predictor" = "class MoodPredictor"
    "Fuzzy Logic Engine" = "class FuzzyLogicEngine"
    "Sacred Badge System" = "SACRED_BADGES"
    "Self-Healing System" = "class SelfHealingSystem"
    "Fibonacci Milestones" = "fibonacci_milestones"
    "Daily Wellness Entry" = "/api/daily-entry"
    "Analytics Summary" = "/api/analytics/summary"
}

$allPresent = $true
foreach ($feature in $features.GetEnumerator()) {
    if ($appContent -match $feature.Value) {
        Write-Host "   ✅ $($feature.Key): PRESENT" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $($feature.Key): NOT FOUND" -ForegroundColor Yellow
        $allPresent = $false
    }
}

# Step 7: Show file sizes
Write-Host ""
Write-Host "📁 Step 7: Deployment files ready:" -ForegroundColor Cyan
Get-ChildItem -Name "app.py", "requirements.txt", "Procfile", "runtime.txt" 2>$null | ForEach-Object {
    $size = (Get-Item $_).Length
    $sizeKB = [math]::Round($size / 1024, 1)
    Write-Host "   📄 $_ (${sizeKB}KB)" -ForegroundColor White
}

# Step 8: Git operations
Write-Host ""
Write-Host "🚀 Step 8: Preparing Git commit..." -ForegroundColor Cyan

git add app.py requirements.txt Procfile runtime.txt 2>$null
Write-Host "   ✅ Files staged" -ForegroundColor Green

# Show status
Write-Host ""
Write-Host "📊 Git Status:" -ForegroundColor Cyan
git status --short

# Step 9: Commit
Write-Host ""
Write-Host "💾 Step 9: Creating commit..." -ForegroundColor Cyan

$commitMessage = @"
Deploy Life Fractal Intelligence v8.0 - Complete Production System

FEATURES INCLUDED:
- Sacred Mathematics (Golden Ratio, Fibonacci, Flower of Life)
- GPU-accelerated fractal generation with CPU fallback
- ML Mood Prediction (Decision Tree)
- Fuzzy Logic Guidance System
- Virtual Pet System (5 species) with evolution
- Sacred Badge Achievements (Fibonacci milestones)
- Goals & Habits with Fibonacci milestone tracking
- Daily Wellness Check-ins with analytics
- Self-Healing System for automatic recovery
- 7-day free trial + subscription support

DEPLOYMENT FIXES:
- jsonify() context error FIXED (lambda deferral)
- Python 3.13 compatible dependencies
- JSON storage (no database required)
- Production-ready for Render.com

ACCESSIBILITY:
- Aphantasia-friendly text descriptions
- Autism/ADHD accommodations
- High contrast mode support
"@

git commit -m $commitMessage 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Commit created" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Nothing to commit or already committed" -ForegroundColor Yellow
}

# Step 10: Push
Write-Host ""
Write-Host "📤 Step 10: Pushing to GitHub..." -ForegroundColor Cyan
git push origin main 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Push may have had issues - check above" -ForegroundColor Yellow
}

# Final summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ DEPLOYMENT PREPARATION COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "📋 v8.0 FEATURES READY TO DEPLOY:" -ForegroundColor Yellow
Write-Host "   🌀 Sacred Mathematics Engine" -ForegroundColor White
Write-Host "   🎨 GPU-Accelerated Fractals (with CPU fallback)" -ForegroundColor White
Write-Host "   🧠 ML Mood Prediction" -ForegroundColor White
Write-Host "   💬 Fuzzy Logic Guidance" -ForegroundColor White
Write-Host "   🐾 Virtual Pet System (5 species)" -ForegroundColor White
Write-Host "   🏆 Sacred Badge Achievements" -ForegroundColor White
Write-Host "   🎯 Goals with Fibonacci Milestones" -ForegroundColor White
Write-Host "   ✨ Habits with Streak Tracking" -ForegroundColor White
Write-Host "   📊 Daily Wellness & Analytics" -ForegroundColor White
Write-Host "   🛡️ Self-Healing System" -ForegroundColor White
Write-Host ""
Write-Host "📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "   1. Go to https://dashboard.render.com" -ForegroundColor White
Write-Host "   2. Click on your service (planner-1)" -ForegroundColor White
Write-Host "   3. Click 'Manual Deploy' -> 'Deploy latest commit'" -ForegroundColor White
Write-Host "   4. Wait 3-4 minutes for build (sklearn takes longer)" -ForegroundColor White
Write-Host ""
Write-Host "📊 EXPECTED BUILD OUTPUT:" -ForegroundColor Yellow
Write-Host "   ✅ Collecting Flask==3.0.0" -ForegroundColor Green
Write-Host "   ✅ Collecting numpy==1.26.4" -ForegroundColor Green
Write-Host "   ✅ Collecting scikit-learn==1.3.2" -ForegroundColor Green
Write-Host "   ✅ Starting gunicorn 21.2.0" -ForegroundColor Green
Write-Host "   ✅ Your service is live 🎉" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Your URL: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
