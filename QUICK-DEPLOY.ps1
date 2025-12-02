# =====================================================================
# QUICK DEPLOY - Life Fractal Intelligence v8.0
# =====================================================================
# Ultra-simple one-click deployment script
# =====================================================================

Write-Host "`n=== LIFE FRACTAL v8.0 - QUICK DEPLOY ===" -ForegroundColor Cyan
Write-Host "Deploying to GitHub → Render...`n" -ForegroundColor Yellow

# Add all changes
Write-Host "[1/4] Staging changes..." -ForegroundColor Cyan
git add .

# Commit
Write-Host "[2/4] Committing..." -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "Deploy Life Fractal v8.0 - $timestamp"

# Get current branch
$branch = git rev-parse --abbrev-ref HEAD

# Push
Write-Host "[3/4] Pushing to GitHub ($branch)..." -ForegroundColor Cyan
git push origin $branch

# Done
Write-Host "[4/4] Complete!" -ForegroundColor Green
Write-Host "`nDeployment initiated! Render will auto-deploy from GitHub." -ForegroundColor Green
Write-Host "`nMonitor at: https://dashboard.render.com" -ForegroundColor Yellow
Write-Host "Deployed URL: https://planner-1-pyd9.onrender.com" -ForegroundColor Yellow

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
