# 🎮 COVER FACE - INSTANT DEPLOY
# ================================
# Just double-click this file!

Clear-Host

Write-Host ""
Write-Host "  🎮 DEPLOYING COVER FACE..." -ForegroundColor Cyan
Write-Host ""

git add .
git commit -m "Deploy COVER FACE - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push

Write-Host ""
Write-Host "  ✅ DONE! Deploying to Render..." -ForegroundColor Green
Write-Host ""
Write-Host "  🎮 Play in 10 min: https://planner-1-pyd9.onrender.com/game" -ForegroundColor Yellow
Write-Host ""
Write-Host "  📈 Monitor: https://dashboard.render.com" -ForegroundColor Gray
Write-Host ""

timeout /t 15
