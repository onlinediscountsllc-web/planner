# Life Fractal Intelligence - INSTANT LAUNCHER
# Just double-click this file or run: .\START.ps1

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 LIFE FRACTAL INTELLIGENCE - STARTING..." -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Quick dependency check
Write-Host "`n📦 Checking dependencies..." -ForegroundColor Yellow

$missingDeps = @()

try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) { $missingDeps += "Flask" }
} catch { $missingDeps += "Flask" }

try {
    python -c "import numpy" 2>$null
    if ($LASTEXITCODE -ne 0) { $missingDeps += "numpy" }
} catch { $missingDeps += "numpy" }

try {
    python -c "import PIL" 2>$null
    if ($LASTEXITCODE -ne 0) { $missingDeps += "Pillow" }
} catch { $missingDeps += "Pillow" }

if ($missingDeps.Count -gt 0) {
    Write-Host "⚠️  Missing dependencies detected!" -ForegroundColor Yellow
    Write-Host "Installing: $($missingDeps -join ', ')" -ForegroundColor Gray
    
    pip install Flask Flask-CORS numpy Pillow scikit-learn --break-system-packages
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Installation failed. Please run manually:" -ForegroundColor Red
        Write-Host "   pip install Flask Flask-CORS numpy Pillow --break-system-packages" -ForegroundColor White
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "✅ All dependencies found!" -ForegroundColor Green
}

# Create data directory
if (-not (Test-Path "data")) {
    Write-Host "`n📁 Creating data directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "data" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/fractals" | Out-Null
    Write-Host "✅ Data directory created" -ForegroundColor Green
}

# Create minimal .env if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "`n🔐 Creating .env configuration..." -ForegroundColor Yellow
    $secretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
    
    @"
SECRET_KEY=$secretKey
PORT=5000
FLASK_ENV=development
DATA_DIR=./data
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ Configuration created" -ForegroundColor Green
}

Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🚀 LAUNCHING LIFE FRACTAL INTELLIGENCE..." -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

Write-Host "`n📡 Server will start on: http://localhost:5000" -ForegroundColor White
Write-Host "🌐 Open index.html in your browser for the dashboard" -ForegroundColor White
Write-Host "📚 Press Ctrl+C to stop the server`n" -ForegroundColor Gray

# Start the server
python life_fractal_complete.py
