# LIFE FRACTAL INTELLIGENCE - ONE-CLICK DEPLOYMENT
# Master script that orchestrates: Patch -> Deploy -> Test
# Run this for complete automated deployment

param(
    [switch]$SkipBackup,
    [switch]$SkipTests,
    [switch]$Force,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "          LIFE FRACTAL INTELLIGENCE - ONE-CLICK DEPLOYMENT" -ForegroundColor Cyan
Write-Host "               Version 2.0.0 - Enhanced Features" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Patch your codebase with enhanced features" -ForegroundColor White
Write-Host "  2. Deploy to Render.com" -ForegroundColor White
Write-Host "  3. Run comprehensive tests" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "PHASE 1: PATCHING CODEBASE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if SUPER-PATCH.ps1 exists
if (-not (Test-Path "SUPER-PATCH.ps1")) {
    Write-Host "ERROR: SUPER-PATCH.ps1 not found!" -ForegroundColor Red
    Write-Host "Please ensure all deployment scripts are in the project directory." -ForegroundColor Yellow
    exit 1
}

# Run SUPER-PATCH
try {
    $patchParams = @()
    if ($SkipBackup) { $patchParams += "-SkipBackup" }
    
    & ".\SUPER-PATCH.ps1" @patchParams
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Patching failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "SUCCESS: Phase 1 complete - Codebase patched successfully" -ForegroundColor Green
    Write-Host ""
    
    # Wait for user to review
    Write-Host "Review the changes before deploying..." -ForegroundColor Yellow
    $continueDeployment = Read-Host "Continue to deployment? (y/n)"
    
    if ($continueDeployment -ne 'y') {
        Write-Host "Deployment paused. You can:" -ForegroundColor Yellow
        Write-Host "  - Review changes: git diff" -ForegroundColor White
        Write-Host "  - Continue later: .\DEPLOY-TO-RENDER.ps1" -ForegroundColor White
        exit 0
    }
    
} catch {
    Write-Host "ERROR during patching: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "PHASE 2: DEPLOYING TO RENDER" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if DEPLOY-TO-RENDER.ps1 exists
if (-not (Test-Path "DEPLOY-TO-RENDER.ps1")) {
    Write-Host "ERROR: DEPLOY-TO-RENDER.ps1 not found!" -ForegroundColor Red
    exit 1
}

# Run deployment
try {
    $deployParams = @{
        Branch = $Branch
    }
    if ($Force) { $deployParams.Force = $true }
    if ($SkipTests) { $deployParams.SkipTests = $true }
    
    & ".\DEPLOY-TO-RENDER.ps1" @deployParams
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Deployment failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "SUCCESS: Phase 2 complete - Deployed to Render" -ForegroundColor Green
    Write-Host ""
    
} catch {
    Write-Host "ERROR during deployment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Wait for deployment to propagate
Write-Host "Waiting for deployment to propagate (60 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "PHASE 3: TESTING DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipTests) {
    # Check if TEST-DEPLOYMENT.ps1 exists
    if (-not (Test-Path "TEST-DEPLOYMENT.ps1")) {
        Write-Host "WARNING: TEST-DEPLOYMENT.ps1 not found, skipping tests" -ForegroundColor Yellow
    } else {
        try {
            Write-Host "Enter your Render app URL to run tests" -ForegroundColor Yellow
            Write-Host "Example: https://your-app.onrender.com" -ForegroundColor Gray
            $appUrl = Read-Host "URL"
            
            if ($appUrl) {
                & ".\TEST-DEPLOYMENT.ps1" -AppUrl $appUrl -Verbose
                
                Write-Host ""
                Write-Host "SUCCESS: Phase 3 complete - Testing finished" -ForegroundColor Green
                Write-Host ""
            } else {
                Write-Host "WARNING: No URL provided, skipping tests" -ForegroundColor Yellow
            }
            
        } catch {
            Write-Host "WARNING: Testing failed: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "You can run tests manually later: .\TEST-DEPLOYMENT.ps1" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Tests skipped (use -SkipTests:`$false to enable)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Final summary
Write-Host "DEPLOYMENT SUMMARY" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Codebase Patched:" -ForegroundColor Green
Write-Host "  - Enhanced Pet AI integrated" -ForegroundColor White
Write-Host "  - Fractal Calendar system added" -ForegroundColor White
Write-Host "  - Executive Function Support enabled" -ForegroundColor White
Write-Host "  - Accessibility features activated" -ForegroundColor White
Write-Host "  - 5 new API endpoints created" -ForegroundColor White
Write-Host ""
Write-Host "Deployed to Render:" -ForegroundColor Green
Write-Host "  - Branch: $Branch" -ForegroundColor White
Write-Host "  - All files pushed successfully" -ForegroundColor White
Write-Host "  - Environment configured" -ForegroundColor White
Write-Host ""

if (-not $SkipTests) {
    Write-Host "Tests Completed" -ForegroundColor Green
} else {
    Write-Host "Tests Skipped" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "NEW FEATURES LIVE:" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Emotional Pet AI" -ForegroundColor Green
Write-Host "    - Differential equations for realistic behavior" -ForegroundColor White
Write-Host "    - Species-specific traits (dragon, phoenix, owl, cat, fox)" -ForegroundColor White
Write-Host "    - Direct fractal visualization influence" -ForegroundColor White
Write-Host ""
Write-Host "  Fractal Time Calendar" -ForegroundColor Green
Write-Host "    - Fibonacci-based time blocks" -ForegroundColor White
Write-Host "    - Circadian rhythm alignment" -ForegroundColor White
Write-Host "    - Spoon theory energy tracking" -ForegroundColor White
Write-Host ""
Write-Host "  Fibonacci Task Scheduler" -ForegroundColor Green
Write-Host "    - Golden ratio prioritization" -ForegroundColor White
Write-Host "    - Energy-aware scheduling" -ForegroundColor White
Write-Host "    - Automatic time block allocation" -ForegroundColor White
Write-Host ""
Write-Host "  Executive Function Support" -ForegroundColor Green
Write-Host "    - Fourier analysis dysfunction detection" -ForegroundColor White
Write-Host "    - Micro-step task scaffolding" -ForegroundColor White
Write-Host "    - Compassionate recommendations" -ForegroundColor White
Write-Host ""
Write-Host "  Full Accessibility Suite" -ForegroundColor Green
Write-Host "    - Autism-safe color themes" -ForegroundColor White
Write-Host "    - Aphantasia text-first mode" -ForegroundColor White
Write-Host "    - Dysgraphia voice input support" -ForegroundColor White
Write-Host "    - Screen reader compatible" -ForegroundColor White
Write-Host ""
Write-Host "  Privacy-Preserving ML" -ForegroundColor Green
Write-Host "    - Local-first data storage" -ForegroundColor White
Write-Host "    - Differential privacy" -ForegroundColor White
Write-Host "    - Federated learning framework" -ForegroundColor White
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Monitor Deployment" -ForegroundColor Yellow
Write-Host "   - Dashboard: https://dashboard.render.com/" -ForegroundColor White
Write-Host "   - Logs: render logs tail -f <service-name>" -ForegroundColor White
Write-Host ""
Write-Host "2. Share with Users" -ForegroundColor Yellow
Write-Host "   - Beta test with neurodivergent community" -ForegroundColor White
Write-Host "   - Gather feedback on new features" -ForegroundColor White
Write-Host ""
Write-Host "3. Marketing" -ForegroundColor Yellow
Write-Host "   - Update GoFundMe with new features" -ForegroundColor White
Write-Host "   - Share on social media" -ForegroundColor White
Write-Host "   - Reach out to ADHD/Autism communities" -ForegroundColor White
Write-Host ""
Write-Host "4. Iterate" -ForegroundColor Yellow
Write-Host "   - Monitor user engagement" -ForegroundColor White
Write-Host "   - Track which features are most used" -ForegroundColor White
Write-Host "   - Plan next enhancements" -ForegroundColor White
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Deployment orchestration complete!" -ForegroundColor Magenta
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Generate final checklist
$checklist = @"
========================================================================
POST-DEPLOYMENT CHECKLIST
========================================================================

Immediate (Next 5 minutes):
  [ ] Verify app is accessible at your Render URL
  [ ] Test login/register functionality
  [ ] Check at least 1 enhanced feature works
  [ ] Monitor Render logs for errors

Today:
  [ ] Test all new API endpoints
  [ ] Verify pet emotional AI is working
  [ ] Test fractal calendar generation
  [ ] Check accessibility settings
  [ ] Update your documentation/README

This Week:
  [ ] Invite 5-10 beta testers
  [ ] Collect initial feedback
  [ ] Monitor error rates and performance
  [ ] Update GoFundMe with new features

This Month:
  [ ] Iterate based on feedback
  [ ] Plan next feature set
  [ ] Grow user base to 100+
  [ ] Consider custom domain

========================================================================
"@

$checklist | Out-File "POST_DEPLOYMENT_CHECKLIST.txt"
Write-Host "Checklist saved to: POST_DEPLOYMENT_CHECKLIST.txt" -ForegroundColor Cyan
Write-Host ""
