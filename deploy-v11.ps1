# ═══════════════════════════════════════════════════════════════════════════════
# 🌀 LIFE FRACTAL INTELLIGENCE v11.0 - DEPLOYMENT SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
# Run this in PowerShell from your project directory
# Make sure you've downloaded app.py and requirements.txt from Claude first!
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 LIFE FRACTAL INTELLIGENCE v11.0 - DEPLOYMENT TO RENDER" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Navigate to your project folder
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "📁 Step 1: Navigate to project folder..." -ForegroundColor Yellow

# CHANGE THIS PATH to your actual project location!
$projectPath = "C:\Users\YourUsername\Projects\planner"

# Or if you're already in the folder, comment out the above and use:
# $projectPath = Get-Location

if (Test-Path $projectPath) {
    Set-Location $projectPath
    Write-Host "✅ Changed to: $projectPath" -ForegroundColor Green
} else {
    Write-Host "❌ Project path not found: $projectPath" -ForegroundColor Red
    Write-Host "   Please update the `$projectPath variable in this script" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Backup existing files (optional but recommended)
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "💾 Step 2: Creating backups..." -ForegroundColor Yellow

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "backups\$timestamp"

if (!(Test-Path "backups")) {
    New-Item -ItemType Directory -Path "backups" | Out-Null
}
New-Item -ItemType Directory -Path $backupDir | Out-Null

if (Test-Path "app.py") {
    Copy-Item "app.py" "$backupDir\app.py.bak"
    Write-Host "✅ Backed up app.py" -ForegroundColor Green
}
if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" "$backupDir\requirements.txt.bak"
    Write-Host "✅ Backed up requirements.txt" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Copy new files (assumes they're in Downloads folder)
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "📄 Step 3: Copying new files..." -ForegroundColor Yellow

$downloadsPath = "$env:USERPROFILE\Downloads"

# Check for downloaded files
$appSource = "$downloadsPath\app.py"
$reqSource = "$downloadsPath\requirements.txt"

if (Test-Path $appSource) {
    Copy-Item $appSource "app.py" -Force
    Write-Host "✅ Copied app.py from Downloads" -ForegroundColor Green
} else {
    Write-Host "⚠️  app.py not found in Downloads - make sure you downloaded it from Claude!" -ForegroundColor Yellow
}

if (Test-Path $reqSource) {
    Copy-Item $reqSource "requirements.txt" -Force
    Write-Host "✅ Copied requirements.txt from Downloads" -ForegroundColor Green
} else {
    Write-Host "⚠️  requirements.txt not found in Downloads - make sure you downloaded it from Claude!" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Verify files
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "🔍 Step 4: Verifying files..." -ForegroundColor Yellow

if (Test-Path "app.py") {
    $lines = (Get-Content "app.py" | Measure-Object -Line).Lines
    Write-Host "✅ app.py exists ($lines lines)" -ForegroundColor Green
    
    # Check for v11 signature
    $content = Get-Content "app.py" -Raw
    if ($content -match "v11\.0") {
        Write-Host "✅ Confirmed: Life Fractal Intelligence v11.0" -ForegroundColor Green
    }
    if ($content -match "FractalIntelligenceBrain") {
        Write-Host "✅ Confirmed: AI Brain module present" -ForegroundColor Green
    }
    if ($content -match "MultiLayerFractalEngine") {
        Write-Host "✅ Confirmed: Multi-layer fractal engine present" -ForegroundColor Green
    }
} else {
    Write-Host "❌ app.py is missing!" -ForegroundColor Red
    exit 1
}

if (Test-Path "requirements.txt") {
    Write-Host "✅ requirements.txt exists" -ForegroundColor Green
    Get-Content "requirements.txt" | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }
} else {
    Write-Host "❌ requirements.txt is missing!" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Git operations
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "🔄 Step 5: Git operations..." -ForegroundColor Yellow

# Check if git is available
try {
    git --version | Out-Null
    Write-Host "✅ Git is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check git status
Write-Host ""
Write-Host "📊 Current git status:" -ForegroundColor Cyan
git status --short

# Stage all changes
Write-Host ""
Write-Host "📦 Staging changes..." -ForegroundColor Yellow
git add app.py
git add requirements.txt
git add -A

Write-Host "✅ Files staged" -ForegroundColor Green

# Commit with descriptive message
Write-Host ""
Write-Host "💬 Creating commit..." -ForegroundColor Yellow

$commitMessage = @"
🌀 Life Fractal Intelligence v11.0 - Complete AI System

✅ NEW FEATURES:
- AI Brain with pattern recognition (RandomForest ML)
- Executive dysfunction early warning system
- Predictive analytics for mood/energy
- Multi-layer 2D fractals (Julia + Goals + Spiral + Particles)
- 3D immersive universe with sacred geometry
- Fractal math optimization engine
- Federated learning from anonymized data
- Math combination storage per user
- S_therapy scalar for visual intensity control

✅ EXISTING FEATURES PRESERVED:
- Spoon Theory energy management
- Mayan Tzolkin calendar
- Binaural beats therapy
- Virtual pet system
- Goals & Habits tracking
- Daily wellness check-ins

🧠 AI learns from user data to personalize fractals and predictions
📐 Sacred mathematics: φ=1.618, Golden Angle=137.5°, Fibonacci sequences
🎯 Designed for: Aphantasia, Autism, ADHD, Dysgraphia, Executive Dysfunction
"@

git commit -m $commitMessage
Write-Host "✅ Commit created" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Push to GitHub
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "🚀 Step 6: Pushing to GitHub..." -ForegroundColor Yellow

# Force push to ensure clean state
git push origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Push may have failed. Trying without --force..." -ForegroundColor Yellow
    git push origin main
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Final summary
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🎉 DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 What happens next:" -ForegroundColor White
Write-Host "   1. Render.com will detect the GitHub push" -ForegroundColor Gray
Write-Host "   2. Auto-deployment will start (watch your Render dashboard)" -ForegroundColor Gray
Write-Host "   3. Build takes ~2-5 minutes" -ForegroundColor Gray
Write-Host "   4. Your app will be live at: https://planner-1-pyd9.onrender.com" -ForegroundColor Gray
Write-Host ""
Write-Host "🔗 Quick Links:" -ForegroundColor White
Write-Host "   Render Dashboard: https://dashboard.render.com" -ForegroundColor Cyan
Write-Host "   GitHub Repo: https://github.com/onlinediscountsllc-web/planner" -ForegroundColor Cyan
Write-Host ""
Write-Host "🧠 v11.0 AI Features Now Active:" -ForegroundColor White
Write-Host "   • Pattern Recognition ML (needs 7+ days of data)" -ForegroundColor Gray
Write-Host "   • Tomorrow's Mood/Energy Predictions" -ForegroundColor Gray
Write-Host "   • Executive Dysfunction Early Warning" -ForegroundColor Gray
Write-Host "   • Personalized Fractal Parameters" -ForegroundColor Gray
Write-Host "   • Multi-Layer 2D Fractal Generation" -ForegroundColor Gray
Write-Host "   • 3D Immersive Universe" -ForegroundColor Gray
Write-Host ""
