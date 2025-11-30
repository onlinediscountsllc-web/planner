# Life Fractal Intelligence - Quick Deploy Script
# ═══════════════════════════════════════════════════════════════════════════════════════
# This script automates the deployment process
# ═══════════════════════════════════════════════════════════════════════════════════════

Write-Host "🌀 Life Fractal Intelligence - Deployment Script" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check Python version
Write-Host "`n📋 Checking Python installation..." -ForegroundColor Yellow
python --version

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment (optional but recommended)
$createVenv = Read-Host "`n🔧 Create virtual environment? (y/n)"
if ($createVenv -eq 'y') {
    Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
}

# Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
pip install Flask Flask-CORS Werkzeug numpy Pillow scikit-learn gunicorn gevent python-dotenv cryptography pytest pytest-flask --break-system-packages

# Check for GPU
Write-Host "`n🎨 Checking for GPU support..." -ForegroundColor Yellow
$hasGPU = Read-Host "Do you have an NVIDIA GPU? (y/n)"

if ($hasGPU -eq 'y') {
    Write-Host "📥 Installing GPU acceleration packages..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes..." -ForegroundColor Gray
    
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
    
    $testGPU = python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1
    Write-Host $testGPU -ForegroundColor Green
}

# Create data directory
Write-Host "`n📁 Creating data directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "data/fractals" | Out-Null
Write-Host "✅ Data directory created" -ForegroundColor Green

# Create .env file
Write-Host "`n🔐 Setting up environment variables..." -ForegroundColor Yellow

if (Test-Path ".env") {
    $overwrite = Read-Host ".env file exists. Overwrite? (y/n)"
    if ($overwrite -ne 'y') {
        Write-Host "⏭️ Skipping .env creation" -ForegroundColor Gray
    } else {
        Remove-Item ".env"
    }
}

if (-not (Test-Path ".env")) {
    $secretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
    
    @"
# Life Fractal Intelligence - Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════════════

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=$secretKey

# Stripe Configuration (Get from https://dashboard.stripe.com)
STRIPE_SECRET_KEY=sk_test_your_test_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_your_test_key_here
STRIPE_PRICE_ID=price_your_price_id_here

# GoFundMe Campaign
GOFUNDME_CAMPAIGN_URL=https://gofundme.com/your-campaign

# Data Storage
DATA_DIR=./data

# Server Configuration
PORT=5000
FLASK_ENV=development

# Optional: Sentry Error Tracking
# SENTRY_DSN=https://your-sentry-dsn
"@ | Out-File -FilePath ".env" -Encoding UTF8
    
    Write-Host "✅ .env file created with random SECRET_KEY" -ForegroundColor Green
    Write-Host "⚠️  Remember to update Stripe keys before going live!" -ForegroundColor Yellow
}

# Run tests
Write-Host "`n🧪 Running health check..." -ForegroundColor Yellow
$testCode = @"
import sys
sys.path.insert(0, '.')

# Quick import test
try:
    import flask
    import numpy as np
    from PIL import Image
    print('✅ Core dependencies OK')
    
    try:
        import sklearn
        print('✅ Machine Learning (scikit-learn) OK')
    except ImportError:
        print('⚠️  scikit-learn not available (optional)')
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f'✅ GPU Acceleration OK - {torch.cuda.get_device_name(0)}')
        else:
            print('ℹ️  GPU not available - will use CPU mode')
    except ImportError:
        print('ℹ️  PyTorch not available - will use CPU mode')
    
    print('\n🎉 All checks passed!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"@

$testCode | python

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Dependency check failed. Please review errors above." -ForegroundColor Red
    exit 1
}

# Ask to start server
Write-Host "`n═══════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Update .env file with your Stripe keys" -ForegroundColor White
Write-Host "2. Run the application:" -ForegroundColor White
Write-Host "   python life_fractal_complete.py" -ForegroundColor Cyan
Write-Host "`nOr for production:" -ForegroundColor White
Write-Host "   gunicorn -w 4 -b 0.0.0.0:5000 life_fractal_complete:app" -ForegroundColor Cyan
Write-Host "`n═══════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

$startNow = Read-Host "`nStart the server now? (y/n)"

if ($startNow -eq 'y') {
    Write-Host "`n🚀 Starting Life Fractal Intelligence..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Gray
    python life_fractal_complete.py
}
