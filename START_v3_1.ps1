# START THE COMPLETE v3.1 SYSTEM
# This is the CORRECT file with zero placeholders!

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host "  LIFE FRACTAL INTELLIGENCE v3.1 COMPLETE - STARTING..." -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host ""

# Check if file exists
if (-Not (Test-Path "life_fractal_complete_v3_1.py")) {
    Write-Host "ERROR: life_fractal_complete_v3_1.py not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please copy it from:" -ForegroundColor Yellow
    Write-Host "  C:\Users\onlin\AppData\Local\Temp\gradio\life_fractal_complete_v3_1.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To your current directory:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)" -ForegroundColor Cyan
    Write-Host ""
    pause
    exit 1
}

Write-Host "Starting the COMPLETE v3.1 system..." -ForegroundColor Green
Write-Host "  - No database needed (in-memory)" -ForegroundColor Gray
Write-Host "  - GPU optimized (3-5x faster)" -ForegroundColor Gray
Write-Host "  - Audio reactive ready" -ForegroundColor Gray
Write-Host "  - Zero placeholders" -ForegroundColor Gray
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host ""

# Run the correct file
py life_fractal_complete_v3_1.py
