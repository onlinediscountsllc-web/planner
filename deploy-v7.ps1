# Life Fractal v7 - Clean Deploy Script
# Run this in PowerShell

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  CLEAN DEPLOY v7 - FIX GIT ISSUE" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\Users\Luke\Desktop\planner"

Write-Host "Removing nested git repos..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "life-fractal-render" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "render-deployment" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Staging only essential files..." -ForegroundColor Yellow
git add app.py requirements.txt render.yaml runtime.txt

Write-Host ""
Write-Host "Committing..." -ForegroundColor Yellow
git commit -m "v7 Production - Email fixed, Nordic design, ML foundations"

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  DEPLOYED! Check Render now." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your app: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host ""

Pause
