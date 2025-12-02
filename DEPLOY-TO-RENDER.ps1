# =====================================================================
# LIFE FRACTAL INTELLIGENCE v8.0 - POWERSHELL DEPLOYMENT SCRIPT
# =====================================================================
# This script will:
# 1. Check Git status
# 2. Add all changes
# 3. Commit with message
# 4. Push to GitHub
# 5. Trigger Render deployment
# =====================================================================

# Colors for output
$ErrorColor = "Red"
$SuccessColor = "Green"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor $InfoColor
    Write-Host "  $Message" -ForegroundColor $InfoColor
    Write-Host "========================================" -ForegroundColor $InfoColor
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $SuccessColor
}

function Write-Error-Message {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $InfoColor
}

function Write-Warning-Message {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor
}

# =====================================================================
# HEADER
# =====================================================================

Clear-Host
Write-Host @"

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   LIFE FRACTAL INTELLIGENCE v8.0 - DEPLOY TO RENDER          ║
║                                                               ║
║   Automated PowerShell Deployment Script                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor $InfoColor

# =====================================================================
# STEP 1: CHECK PREREQUISITES
# =====================================================================

Write-Step "STEP 1: Checking Prerequisites"

# Check if Git is installed
Write-Info "Checking Git installation..."
try {
    $gitVersion = git --version
    Write-Success "Git installed: $gitVersion"
} catch {
    Write-Error-Message "Git is not installed or not in PATH!"
    Write-Host "`nPlease install Git from: https://git-scm.com/download/win" -ForegroundColor $WarningColor
    pause
    exit 1
}

# Check if we're in a Git repository
Write-Info "Checking if current directory is a Git repository..."
if (-not (Test-Path ".git")) {
    Write-Error-Message "Not a Git repository!"
    Write-Host "`nPlease run this script from your project root directory." -ForegroundColor $WarningColor
    Write-Host "Or initialize Git with: git init" -ForegroundColor $WarningColor
    pause
    exit 1
}
Write-Success "Git repository detected"

# Check for required files
Write-Info "Checking for required v8.0 files..."
$requiredFiles = @(
    "secure_auth_module.py",
    "life_fractal_v8_secure.py",
    "requirements.txt",
    "test_bugs.py"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Success "Found: $file"
    } else {
        Write-Error-Message "Missing: $file"
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`nMissing required files. Please copy them first!" -ForegroundColor $ErrorColor
    pause
    exit 1
}

# =====================================================================
# STEP 2: CHECK GIT STATUS
# =====================================================================

Write-Step "STEP 2: Checking Git Status"

$gitStatus = git status --short
if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Warning-Message "No changes detected to commit"
    Write-Host "`nYour repository is already up to date." -ForegroundColor $WarningColor
    Write-Host "Do you want to force push anyway? (y/n): " -NoNewline -ForegroundColor $WarningColor
    $forcePush = Read-Host
    if ($forcePush -ne "y" -and $forcePush -ne "Y") {
        Write-Info "Deployment cancelled"
        pause
        exit 0
    }
} else {
    Write-Info "Changes detected:"
    Write-Host $gitStatus -ForegroundColor $InfoColor
}

# =====================================================================
# STEP 3: GET COMMIT MESSAGE
# =====================================================================

Write-Step "STEP 3: Commit Message"

Write-Host "`nEnter commit message (or press Enter for default): " -ForegroundColor $InfoColor -NoNewline
$commitMessage = Read-Host

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Deploy Life Fractal Intelligence v8.0 with secure authentication"
}

Write-Success "Commit message: $commitMessage"

# =====================================================================
# STEP 4: ADD ALL CHANGES
# =====================================================================

Write-Step "STEP 4: Adding All Changes to Git"

Write-Info "Running: git add ."
try {
    git add . 2>&1 | Out-Null
    Write-Success "All changes staged"
} catch {
    Write-Error-Message "Failed to stage changes: $_"
    pause
    exit 1
}

# =====================================================================
# STEP 5: COMMIT CHANGES
# =====================================================================

Write-Step "STEP 5: Committing Changes"

Write-Info "Running: git commit -m `"$commitMessage`""
try {
    $commitOutput = git commit -m "$commitMessage" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Changes committed successfully"
        Write-Host $commitOutput -ForegroundColor $InfoColor
    } else {
        # Check if it's just "nothing to commit"
        if ($commitOutput -like "*nothing to commit*") {
            Write-Warning-Message "Nothing to commit (working tree clean)"
        } else {
            Write-Error-Message "Commit failed: $commitOutput"
            pause
            exit 1
        }
    }
} catch {
    Write-Error-Message "Failed to commit: $_"
    pause
    exit 1
}

# =====================================================================
# STEP 6: PUSH TO GITHUB
# =====================================================================

Write-Step "STEP 6: Pushing to GitHub"

# Get current branch
$currentBranch = git rev-parse --abbrev-ref HEAD 2>&1
Write-Info "Current branch: $currentBranch"

Write-Info "Running: git push origin $currentBranch"
Write-Host "`nPushing to GitHub... This may take a moment..." -ForegroundColor $InfoColor

try {
    $pushOutput = git push origin $currentBranch 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Successfully pushed to GitHub!"
        Write-Host $pushOutput -ForegroundColor $InfoColor
    } else {
        Write-Error-Message "Push failed!"
        Write-Host $pushOutput -ForegroundColor $ErrorColor
        Write-Host "`nPossible solutions:" -ForegroundColor $WarningColor
        Write-Host "1. Check your internet connection" -ForegroundColor $WarningColor
        Write-Host "2. Verify GitHub credentials" -ForegroundColor $WarningColor
        Write-Host "3. Make sure you have push access to the repository" -ForegroundColor $WarningColor
        Write-Host "4. Try: git push -u origin $currentBranch" -ForegroundColor $WarningColor
        pause
        exit 1
    }
} catch {
    Write-Error-Message "Failed to push: $_"
    pause
    exit 1
}

# =====================================================================
# STEP 7: RENDER DEPLOYMENT
# =====================================================================

Write-Step "STEP 7: Render Deployment"

Write-Host @"

Your code has been pushed to GitHub!

Render will automatically detect the changes and start deploying.

"@ -ForegroundColor $SuccessColor

Write-Host "Next steps:" -ForegroundColor $InfoColor
Write-Host "1. Go to: https://dashboard.render.com" -ForegroundColor $InfoColor
Write-Host "2. Find your service (planner-1-pyd9)" -ForegroundColor $InfoColor
Write-Host "3. Check the 'Events' tab to see deployment progress" -ForegroundColor $InfoColor
Write-Host "4. Wait 5-10 minutes for deployment to complete" -ForegroundColor $InfoColor

Write-Host "`nWould you like to open Render dashboard now? (y/n): " -NoNewline -ForegroundColor $InfoColor
$openDashboard = Read-Host

if ($openDashboard -eq "y" -or $openDashboard -eq "Y") {
    Write-Info "Opening Render dashboard..."
    Start-Process "https://dashboard.render.com"
}

# =====================================================================
# STEP 8: POST-DEPLOYMENT CHECKLIST
# =====================================================================

Write-Step "STEP 8: Post-Deployment Checklist"

Write-Host @"

After deployment completes, verify these items:

REQUIRED ENVIRONMENT VARIABLES IN RENDER:
  [ ] SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
  [ ] PORT=8080
  [ ] DEBUG=False
  [ ] SMTP_HOST=smtp.gmail.com
  [ ] SMTP_PORT=587
  [ ] SMTP_USER=onlinediscountsllc@gmail.com
  [ ] SMTP_PASSWORD=<your-gmail-app-password>

TESTING:
  [ ] Health check: https://planner-1-pyd9.onrender.com/health
  [ ] Run: python test_bugs.py https://planner-1-pyd9.onrender.com
  [ ] Register test account
  [ ] Check email delivery
  [ ] Verify GoFundMe links

"@ -ForegroundColor $InfoColor

# =====================================================================
# STEP 9: GENERATE SECRET KEY
# =====================================================================

Write-Step "STEP 9: Secret Key Generation (Optional)"

Write-Host "`nWould you like to generate a new SECRET_KEY? (y/n): " -NoNewline -ForegroundColor $InfoColor
$generateKey = Read-Host

if ($generateKey -eq "y" -or $generateKey -eq "Y") {
    Write-Info "Generating secure SECRET_KEY..."
    
    # Generate using Python if available, otherwise use PowerShell
    try {
        $secretKey = python -c "import secrets; print(secrets.token_hex(32))" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Your SECRET_KEY:"
            Write-Host "`n  $secretKey`n" -ForegroundColor $SuccessColor
            Write-Host "Add this to Render environment variables!" -ForegroundColor $InfoColor
            
            # Copy to clipboard if possible
            try {
                Set-Clipboard -Value $secretKey
                Write-Success "SECRET_KEY copied to clipboard!"
            } catch {
                Write-Info "Copy the key manually"
            }
        } else {
            # Fallback to PowerShell method
            $bytes = New-Object byte[] 32
            [Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($bytes)
            $secretKey = [BitConverter]::ToString($bytes).Replace('-', '').ToLower()
            Write-Success "Your SECRET_KEY:"
            Write-Host "`n  $secretKey`n" -ForegroundColor $SuccessColor
            Write-Host "Add this to Render environment variables!" -ForegroundColor $InfoColor
        }
    } catch {
        Write-Error-Message "Failed to generate key: $_"
    }
}

# =====================================================================
# COMPLETION
# =====================================================================

Write-Host @"

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   DEPLOYMENT TO GITHUB COMPLETE!                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor $SuccessColor

Write-Host "Summary:" -ForegroundColor $SuccessColor
Write-Host "  - All changes committed to Git" -ForegroundColor $InfoColor
Write-Host "  - Code pushed to GitHub" -ForegroundColor $InfoColor
Write-Host "  - Render will auto-deploy from GitHub" -ForegroundColor $InfoColor
Write-Host "`nMonitor deployment at: https://dashboard.render.com" -ForegroundColor $InfoColor
Write-Host "`nGitHub Repository: onlinediscountsllc-web/planner" -ForegroundColor $InfoColor
Write-Host "Deployed URL: https://planner-1-pyd9.onrender.com" -ForegroundColor $InfoColor
Write-Host "GoFundMe: https://gofund.me/8d9303d27" -ForegroundColor $InfoColor

Write-Host "`n" -NoNewline
pause
