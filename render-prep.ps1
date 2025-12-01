# Life Fractal Intelligence - Render Deployment Prep
# Run this in your planner directory

Write-Host "🌀 LIFE FRACTAL - RENDER DEPLOYMENT PREP" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ Error: app.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from your planner directory" -ForegroundColor Yellow
    pause
    exit
}

Write-Host "✅ Found app.py" -ForegroundColor Green

# Configure Git
Write-Host ""
Write-Host "Configuring Git..." -ForegroundColor Yellow
git config --global user.email "onlinediscountsllc@gmail.com"
git config --global user.name "Luke"
Write-Host "✅ Git configured" -ForegroundColor Green

# Initialize Git repo
Write-Host ""
Write-Host "Initializing Git repository..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git already initialized" -ForegroundColor Green
}

# Add all files
Write-Host ""
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .
git status
Write-Host "✅ Files added" -ForegroundColor Green

# Commit
Write-Host ""
Write-Host "Creating commit..." -ForegroundColor Yellow
git commit -m "Life Fractal Intelligence - Ready for Render"
Write-Host "✅ Committed" -ForegroundColor Green

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "✅ READY FOR RENDER!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/new" -ForegroundColor White
Write-Host "2. Create a new repository (e.g., 'life-fractal-app')" -ForegroundColor White
Write-Host "3. Copy the remote URL" -ForegroundColor White
Write-Host "4. Run these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR-USERNAME/life-fractal-app.git" -ForegroundColor Yellow
Write-Host "   git branch -M main" -ForegroundColor Yellow
Write-Host "   git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "5. Then go to https://render.com and connect your repo!" -ForegroundColor White
Write-Host ""

pause
