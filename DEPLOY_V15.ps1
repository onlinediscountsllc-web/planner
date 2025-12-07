# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 DEPLOY v15.0 ULTIMATE INTERACTIVE - ONE COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌟 DEPLOYING v15.0 ULTIMATE INTERACTIVE" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Features:" -ForegroundColor White
Write-Host "  ✅ Voice conversations (Whisper + Ollama)" -ForegroundColor Green
Write-Host "  ✅ AI pet assistant" -ForegroundColor Green
Write-Host "  ✅ Congratulations animations + sounds" -ForegroundColor Green
Write-Host "  ✅ Plain English reports" -ForegroundColor Green
Write-Host "  ✅ Swedish design" -ForegroundColor Green
Write-Host "  ✅ Interactive clickable orbs" -ForegroundColor Green
Write-Host ""

# Navigate to project
cd C:\Users\Luke\Desktop\planner

# Backup
Write-Host "[1/4] Backing up current version..." -ForegroundColor Yellow
Copy-Item app.py app_backup_v14.py -Force
Write-Host "      Done ✅" -ForegroundColor Green

# Deploy
Write-Host "[2/4] Deploying v15.0..." -ForegroundColor Yellow
if (Test-Path "life_fractal_v15_ultimate_interactive.py") {
    Copy-Item life_fractal_v15_ultimate_interactive.py -Destination app.py -Force
    Write-Host "      Done ✅" -ForegroundColor Green
} else {
    Write-Host "      ERROR: v15 file not found!" -ForegroundColor Red
    exit 1
}

# Commit
Write-Host "[3/4] Committing to git..." -ForegroundColor Yellow
git add app.py
git commit -m "v15.0 Ultimate Interactive: Voice, AI, Animations, Reports"
Write-Host "      Done ✅" -ForegroundColor Green

# Push
Write-Host "[4/4] Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
Write-Host "      Done ✅" -ForegroundColor Green

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✨ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Render is rebuilding... (~3 minutes)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test after deployment:" -ForegroundColor White
Write-Host "  curl 'https://planner-1-pyd9.onrender.com/api/health'" -ForegroundColor Yellow
Write-Host "  curl 'https://planner-1-pyd9.onrender.com/api/audio/celebration' -o celebrate.wav" -ForegroundColor Yellow
Write-Host "  curl 'https://planner-1-pyd9.onrender.com/api/reports/progress'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Dashboard: https://planner-1-pyd9.onrender.com" -ForegroundColor Cyan
Write-Host ""

# Optional: Set up Ollama
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "📝 OPTIONAL: Set up AI Assistant (Ollama)" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "For voice & AI features, install Ollama:" -ForegroundColor White
Write-Host "  1. Download: https://ollama.ai" -ForegroundColor Yellow
Write-Host "  2. Install Llama 3.1: ollama pull llama3.1" -ForegroundColor Yellow
Write-Host "  3. Start server: ollama serve" -ForegroundColor Yellow
Write-Host "  4. Add to Render environment: OLLAMA_API_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "Without Ollama, assistant will use fallback responses" -ForegroundColor Gray
Write-Host ""
