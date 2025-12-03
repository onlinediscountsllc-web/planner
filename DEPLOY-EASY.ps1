# ═══════════════════════════════════════════════════════════
# 🎮 COVER FACE - ONE-CLICK DEPLOY & PLAY
# ═══════════════════════════════════════════════════════════

Write-Host @"

    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║     🎮 COVER FACE - ONE-CLICK DEPLOY            ║
    ║                                                  ║
    ║     Deploying your 3D life planning game...     ║
    ║                                                  ║
    ╔══════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

Write-Host ""

# Step 1: Stage all changes
Write-Host "[1/4] 📦 Adding files..." -ForegroundColor Yellow
git add .
Write-Host "      ✓ Files staged" -ForegroundColor Green

# Step 2: Commit
Write-Host "[2/4] 💾 Committing..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "Deploy COVER FACE game - $timestamp" 2>&1 | Out-Null
Write-Host "      ✓ Changes committed" -ForegroundColor Green

# Step 3: Push to GitHub
Write-Host "[3/4] 🚀 Pushing to GitHub..." -ForegroundColor Yellow
git push origin main 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "      ✓ Pushed successfully!" -ForegroundColor Green
} else {
    # Try 'master' branch if 'main' fails
    git push origin master 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✓ Pushed successfully!" -ForegroundColor Green
    } else {
        Write-Host "      ✗ Push failed - check your connection" -ForegroundColor Red
        Write-Host ""
        Write-Host "Try running manually:" -ForegroundColor Yellow
        Write-Host "  git push origin main" -ForegroundColor White
        pause
        exit 1
    }
}

# Step 4: Done!
Write-Host "[4/4] ✨ Deployment initiated!" -ForegroundColor Yellow
Write-Host "      ✓ Render is building your app..." -ForegroundColor Green

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🎉 DEPLOYMENT IN PROGRESS!" -ForegroundColor Green
Write-Host ""
Write-Host "  Your game will be live in 5-10 minutes at:" -ForegroundColor White
Write-Host ""
Write-Host "  🎮 Game:      https://planner-1-pyd9.onrender.com/game" -ForegroundColor Cyan
Write-Host "  📊 Dashboard: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host "  📈 Monitor:   https://dashboard.render.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎮 GAME CONTROLS:" -ForegroundColor Yellow
Write-Host "   WASD      - Move your character" -ForegroundColor White
Write-Host "   Mouse     - Look around" -ForegroundColor White
Write-Host "   Spacebar  - Jump" -ForegroundColor White
Write-Host "   Click     - Interact with goal orbs" -ForegroundColor White
Write-Host "   C         - Capture screenshot" -ForegroundColor White
Write-Host ""

# Ask to open browser
Write-Host "Open Render dashboard to monitor? (y/n): " -NoNewline -ForegroundColor Yellow
$response = Read-Host

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "Opening Render dashboard..." -ForegroundColor Green
    Start-Process "https://dashboard.render.com"
    Write-Host ""
    Write-Host "🎯 Look for: 'Deploy live for [commit-id]'" -ForegroundColor Yellow
    Write-Host "   When you see this, your game is READY!" -ForegroundColor Green
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏰ NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Wait 5-10 minutes for deployment" -ForegroundColor White
Write-Host "  2. Go to: https://planner-1-pyd9.onrender.com/game" -ForegroundColor White
Write-Host "  3. Play your game!" -ForegroundColor White
Write-Host ""
Write-Host "💡 First time? Set environment variables in Render:" -ForegroundColor Yellow
Write-Host "   - SECRET_KEY (generate with: python -c ""import secrets; print(secrets.token_hex(32))"")" -ForegroundColor White
Write-Host "   - SMTP_PASSWORD (Gmail App Password)" -ForegroundColor White
Write-Host ""
Write-Host "Need help? onlinediscountsllc@gmail.com" -ForegroundColor Cyan
Write-Host "GoFundMe: https://gofund.me/8d9303d27" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Keep window open
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
