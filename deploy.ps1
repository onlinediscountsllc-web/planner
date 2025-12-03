# ================================================
# LIFE FRACTAL v8 - DEPLOYMENT SCRIPT
# Run this in PowerShell from your project folder
# ================================================

Write-Host "🌀 LIFE FRACTAL INTELLIGENCE v8.0 DEPLOYMENT" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Step 1: Check if we're in a git repo
if (-not (Test-Path ".git")) {
    Write-Host "❌ Not a git repository. Initializing..." -ForegroundColor Yellow
    git init
    git remote add origin https://github.com/onlinediscountsllc-web/planner.git
}

# Step 2: Show current status
Write-Host "`n📁 Current files:" -ForegroundColor Green
git status --short

# Step 3: Add all files
Write-Host "`n➕ Adding all files..." -ForegroundColor Green
git add -A

# Step 4: Commit
$commitMsg = "Deploy Life Fractal v8.0 - Full 3D Visualization System"
Write-Host "💾 Committing: $commitMsg" -ForegroundColor Green
git commit -m $commitMsg

# Step 5: Push to GitHub
Write-Host "`n🚀 Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

# If main doesn't exist, try master
if ($LASTEXITCODE -ne 0) {
    Write-Host "Trying 'master' branch instead..." -ForegroundColor Yellow
    git push -u origin master
}

Write-Host "`n✅ DONE! Now go to Render.com to deploy." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
