# AUTOMATED SETUP AND RUN - ONE COMMAND DOES EVERYTHING!
# Just run this script and it will install dependencies and start the server

param(
    [switch]$SkipInstall = $false
)

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  LIFE FRACTAL INTELLIGENCE v3.1 - AUTOMATED SETUP" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = py --version 2>&1
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.10+" -ForegroundColor Red
    pause
    exit 1
}

# Check if file exists
Write-Host "[2/4] Checking for life_fractal_complete_v3_1.py..." -ForegroundColor Yellow
if (-Not (Test-Path "life_fractal_complete_v3_1.py")) {
    Write-Host "  ✗ File not found in current directory!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please download it from:" -ForegroundColor Yellow
    Write-Host "  /mnt/user-data/outputs/life_fractal_complete_v3_1.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Or copy from your Gradio temp folder to:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)" -ForegroundColor Cyan
    Write-Host ""
    pause
    exit 1
}
Write-Host "  ✓ Found life_fractal_complete_v3_1.py" -ForegroundColor Green

# Install dependencies
if (-Not $SkipInstall) {
    Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
    Write-Host "  This may take 2-5 minutes (downloading ~2GB for PyTorch)..." -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "  Installing core libraries..." -ForegroundColor Cyan
    py -m pip install flask flask-cors numpy pillow scikit-learn --break-system-packages --quiet
    
    Write-Host "  Installing PyTorch (GPU support)..." -ForegroundColor Cyan
    py -m pip install torch torchvision torchaudio --break-system-packages --index-url https://download.pytorch.org/whl/cu118 --quiet
    
    Write-Host "  Installing audio libraries..." -ForegroundColor Cyan
    py -m pip install librosa soundfile --break-system-packages --quiet
    
    Write-Host "  Installing MIDI library..." -ForegroundColor Cyan
    py -m pip install mido --break-system-packages --quiet
    
    Write-Host ""
    Write-Host "  ✓ All dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "[3/4] Skipping dependency installation..." -ForegroundColor Yellow
}

# Start server
Write-Host "[4/4] Starting server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  SERVER STARTING - Open http://localhost:5000 in your browser" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "  Login Credentials:" -ForegroundColor Yellow
Write-Host "    Email:    onlinediscountsllc@gmail.com" -ForegroundColor Cyan
Write-Host "    Password: admin8587037321" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press CTRL+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Run the server
py life_fractal_complete_v3_1.py
