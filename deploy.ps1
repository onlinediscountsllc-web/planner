# ═══════════════════════════════════════════════════════════════
# HEROKU DEPLOYMENT SCRIPT - PowerShell
# ═══════════════════════════════════════════════════════════════

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 LIFE FRACTAL INTELLIGENCE - HEROKU DEPLOYMENT" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if Heroku CLI is installed
Write-Host "📋 Checking Heroku CLI..." -ForegroundColor Yellow
$herokuInstalled = Get-Command heroku -ErrorAction SilentlyContinue

if (-not $herokuInstalled) {
    Write-Host "❌ Heroku CLI not found. Installing..." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Heroku CLI from: https://devcenter.heroku.com/articles/heroku-cli" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing, run this script again." -ForegroundColor Yellow
    pause
    exit
}

Write-Host "✅ Heroku CLI found" -ForegroundColor Green
Write-Host ""

# Login to Heroku
Write-Host "🔐 Logging into Heroku..." -ForegroundColor Yellow
Write-Host "(This will open a browser window)" -ForegroundColor Gray
heroku login

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Heroku login failed" -ForegroundColor Red
    pause
    exit
}

Write-Host "✅ Logged in successfully" -ForegroundColor Green
Write-Host ""

# Get app name
Write-Host "📝 Enter your Heroku app name:" -ForegroundColor Yellow
Write-Host "(Must be unique, lowercase, use dashes for spaces)" -ForegroundColor Gray
Write-Host "Example: life-fractal-john-2024" -ForegroundColor Gray
$appName = Read-Host "App name"

if ([string]::IsNullOrWhiteSpace($appName)) {
    Write-Host "❌ App name cannot be empty" -ForegroundColor Red
    pause
    exit
}

Write-Host ""
Write-Host "🚀 Creating Heroku app: $appName..." -ForegroundColor Yellow
heroku create $appName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create app. App name might be taken." -ForegroundColor Red
    Write-Host "Try a different name." -ForegroundColor Yellow
    pause
    exit
}

Write-Host "✅ App created successfully" -ForegroundColor Green
Write-Host ""

# Add PostgreSQL database
Write-Host "🗄️  Adding PostgreSQL database (free tier)..." -ForegroundColor Yellow
heroku addons:create heroku-postgresql:essential-0 -a $appName

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: Could not add database automatically" -ForegroundColor Yellow
    Write-Host "You may need to add it manually from Heroku dashboard" -ForegroundColor Yellow
}

Write-Host ""

# Configure environment variables
Write-Host "⚙️  Configuring environment variables..." -ForegroundColor Yellow

# Generate secret key
$secretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
heroku config:set SECRET_KEY=$secretKey -a $appName

heroku config:set ENVIRONMENT=production -a $appName
heroku config:set APP_URL=https://$appName.herokuapp.com -a $appName

Write-Host ""
Write-Host "📧 Email Configuration (Optional - for email verification)" -ForegroundColor Yellow
Write-Host "Press Enter to skip, or enter your SMTP details:" -ForegroundColor Gray
Write-Host ""

$smtpServer = Read-Host "SMTP Server (e.g., smtp.gmail.com)"
if (-not [string]::IsNullOrWhiteSpace($smtpServer)) {
    $smtpPort = Read-Host "SMTP Port (default: 587)"
    if ([string]::IsNullOrWhiteSpace($smtpPort)) { $smtpPort = "587" }
    
    $smtpUsername = Read-Host "SMTP Username (your email)"
    $smtpPassword = Read-Host "SMTP Password" -AsSecureString
    $smtpPasswordPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($smtpPassword)
    )
    $fromEmail = Read-Host "From Email Address"
    
    heroku config:set SMTP_SERVER=$smtpServer -a $appName
    heroku config:set SMTP_PORT=$smtpPort -a $appName
    heroku config:set SMTP_USERNAME=$smtpUsername -a $appName
    heroku config:set SMTP_PASSWORD=$smtpPasswordPlain -a $appName
    heroku config:set FROM_EMAIL=$fromEmail -a $appName
    
    Write-Host "✅ Email configured" -ForegroundColor Green
} else {
    Write-Host "⚠️  Email skipped - verification emails will be logged only" -ForegroundColor Yellow
}

Write-Host ""

# Initialize git repository
Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow

if (-not (Test-Path ".git")) {
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git already initialized" -ForegroundColor Green
}

# Add Heroku remote if not exists
$remotes = git remote
if ($remotes -notcontains "heroku") {
    git remote add heroku https://git.heroku.com/$appName.git
    Write-Host "✅ Heroku remote added" -ForegroundColor Green
}

Write-Host ""

# Deploy to Heroku
Write-Host "🚀 Deploying to Heroku..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

git add .
git commit -m "Initial deployment" 2>$null
git push heroku master -f

if ($LASTEXITCODE -ne 0) {
    # Try main branch
    git push heroku main -f
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Your app is live at:" -ForegroundColor Yellow
Write-Host "https://$appName.herokuapp.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Visit your app URL above" -ForegroundColor White
Write-Host "2. Create an account" -ForegroundColor White
Write-Host "3. Check your email for verification" -ForegroundColor White
Write-Host "4. Start using Life Fractal Intelligence!" -ForegroundColor White
Write-Host ""
Write-Host "📊 View logs:" -ForegroundColor Yellow
Write-Host "   heroku logs --tail -a $appName" -ForegroundColor Gray
Write-Host ""
Write-Host "⚙️  Open Heroku dashboard:" -ForegroundColor Yellow
Write-Host "   heroku dashboard -a $appName" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 Add Stripe for payments:" -ForegroundColor Yellow
Write-Host "   heroku config:set STRIPE_SECRET_KEY=sk_test_... -a $appName" -ForegroundColor Gray
Write-Host "   heroku config:set STRIPE_PUBLISHABLE_KEY=pk_test_... -a $appName" -ForegroundColor Gray
Write-Host ""

# Ask if user wants to open the app
$openApp = Read-Host "Open app in browser now? (y/n)"
if ($openApp -eq "y" -or $openApp -eq "Y") {
    Start-Process "https://$appName.herokuapp.com"
}

Write-Host ""
Write-Host "Deployment complete! ✨" -ForegroundColor Green
Write-Host ""
pause
