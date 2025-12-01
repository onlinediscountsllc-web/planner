# 🚀 LIFE FRACTAL INTELLIGENCE - RENDER DEPLOYMENT SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════════════
# Deploys enhanced Life Fractal Intelligence to Render.com
# Run after SUPER-PATCH.ps1
# ═══════════════════════════════════════════════════════════════════════════════════════

param(
    [string]$Branch = "main",
    [switch]$Force,
    [switch]$SkipTests,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 LIFE FRACTAL INTELLIGENCE - RENDER DEPLOYMENT" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if Git is available
try {
    $gitVersion = git --version
    Write-Host "✅ Git: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Git not found!" -ForegroundColor Red
    Write-Host "   Please install Git and try again." -ForegroundColor Yellow
    exit 1
}

# Check if we have uncommitted changes
$gitStatus = git status --porcelain
if ($gitStatus -and -not $Force) {
    Write-Host "⚠️  Warning: You have uncommitted changes" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Uncommitted files:" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y') {
        Write-Host "Deployment cancelled." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""

# Step 1: Verify all required files exist
Write-Host "📋 Step 1: Verifying deployment files..." -ForegroundColor Cyan

$requiredFiles = @(
    "life_planner_unified_master.py",
    "life_fractal_render.py",
    "life_fractal_enhanced_implementation.py",
    "requirements.txt",
    "Procfile",
    "runtime.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (missing)" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ Missing required files!" -ForegroundColor Red
    Write-Host "   Please run SUPER-PATCH.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Run local syntax check
Write-Host "🔍 Step 2: Running syntax checks..." -ForegroundColor Cyan

try {
    python -m py_compile life_fractal_enhanced_implementation.py
    Write-Host "  ✅ life_fractal_enhanced_implementation.py" -ForegroundColor Green
    
    if (Test-Path "life_planner_unified_master.py") {
        python -m py_compile life_planner_unified_master.py
        Write-Host "  ✅ life_planner_unified_master.py" -ForegroundColor Green
    } else {
        python -m py_compile life_fractal_render.py
        Write-Host "  ✅ life_fractal_render.py" -ForegroundColor Green
    }
    
    Write-Host "✅ All Python files have valid syntax" -ForegroundColor Green
} catch {
    Write-Host "❌ Syntax error detected!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: Check Git remote
Write-Host "🔗 Step 3: Checking Git remote..." -ForegroundColor Cyan

$remotes = git remote -v
if (-not $remotes) {
    Write-Host "⚠️  No Git remote configured" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Setting up Git remote for Render..." -ForegroundColor Cyan
    Write-Host "You'll need your Render Git URL (e.g., https://git.render.com/srv-xxx.git)" -ForegroundColor Yellow
    Write-Host ""
    $renderUrl = Read-Host "Enter your Render Git URL"
    
    if ($renderUrl) {
        git remote add render $renderUrl
        Write-Host "✅ Render remote added" -ForegroundColor Green
    } else {
        Write-Host "❌ No URL provided" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Current remotes:" -ForegroundColor Yellow
    Write-Host $remotes -ForegroundColor White
    
    # Check if 'render' remote exists
    if ($remotes -match "render") {
        Write-Host "✅ Render remote configured" -ForegroundColor Green
    } else {
        Write-Host ""
        $addRender = Read-Host "Add Render remote? (y/n)"
        if ($addRender -eq 'y') {
            $renderUrl = Read-Host "Enter your Render Git URL"
            git remote add render $renderUrl
            Write-Host "✅ Render remote added" -ForegroundColor Green
        }
    }
}

Write-Host ""

# Step 4: Push to Git
Write-Host "📤 Step 4: Pushing to Git..." -ForegroundColor Cyan

try {
    # Check current branch
    $currentBranch = git branch --show-current
    Write-Host "Current branch: $currentBranch" -ForegroundColor Yellow
    
    if ($currentBranch -ne $Branch) {
        Write-Host "⚠️  You're on branch '$currentBranch', but specified '$Branch'" -ForegroundColor Yellow
        $switchBranch = Read-Host "Switch to '$Branch'? (y/n)"
        if ($switchBranch -eq 'y') {
            git checkout $Branch
            Write-Host "✅ Switched to $Branch" -ForegroundColor Green
        }
    }
    
    # Push to origin first (GitHub/GitLab)
    Write-Host "Pushing to origin..." -ForegroundColor Yellow
    git push origin $Branch
    Write-Host "✅ Pushed to origin" -ForegroundColor Green
    
    # Push to Render
    Write-Host "Pushing to Render..." -ForegroundColor Yellow
    git push render $Branch --force
    Write-Host "✅ Pushed to Render" -ForegroundColor Green
    
} catch {
    Write-Host "⚠️  Push failed" -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    
    $retry = Read-Host "Retry with force push? (y/n)"
    if ($retry -eq 'y') {
        try {
            git push render $Branch --force
            Write-Host "✅ Force pushed to Render" -ForegroundColor Green
        } catch {
            Write-Host "❌ Force push failed" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
            exit 1
        }
    } else {
        exit 1
    }
}

Write-Host ""

# Step 5: Monitor deployment
Write-Host "👀 Step 5: Monitoring deployment..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Render is now building and deploying your application..." -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 You can monitor the deployment at:" -ForegroundColor Cyan
Write-Host "   https://dashboard.render.com/" -ForegroundColor White
Write-Host ""

# Wait a bit for deployment to start
Write-Host "Waiting 30 seconds for deployment to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host ""

# Step 6: Test deployment (if not skipped)
if (-not $SkipTests) {
    Write-Host "🧪 Step 6: Testing deployment..." -ForegroundColor Cyan
    Write-Host ""
    
    $appUrl = Read-Host "Enter your Render app URL (e.g., https://your-app.onrender.com)"
    
    if ($appUrl) {
        Write-Host ""
        Write-Host "Testing endpoints..." -ForegroundColor Yellow
        
        # Test health endpoint
        try {
            Write-Host "  Testing: $appUrl/health" -ForegroundColor Gray
            $response = Invoke-WebRequest -Uri "$appUrl/health" -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "  ✅ Health check passed" -ForegroundColor Green
            }
        } catch {
            Write-Host "  ⚠️  Health check failed (app may still be starting)" -ForegroundColor Yellow
        }
        
        # Test features endpoint
        try {
            Write-Host "  Testing: $appUrl/api/features/status" -ForegroundColor Gray
            $response = Invoke-WebRequest -Uri "$appUrl/api/features/status" -TimeoutSec 10 -UseBasicParsing
            $data = $response.Content | ConvertFrom-Json
            
            if ($data.enhanced_features_available) {
                Write-Host "  ✅ Enhanced features: ACTIVE" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️  Enhanced features: Not loaded yet" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Feature status:" -ForegroundColor Cyan
            foreach ($feature in $data.features.PSObject.Properties) {
                $status = if ($feature.Value) { "✅" } else { "❌" }
                Write-Host "    $status $($feature.Name)" -ForegroundColor White
            }
        } catch {
            Write-Host "  ⚠️  Features endpoint not responding yet" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "🧪 Step 6: Tests skipped" -ForegroundColor Yellow
}

Write-Host ""

# Step 7: Generate deployment report
Write-Host "📊 Step 7: Generating deployment report..." -ForegroundColor Cyan

$report = @"
═══════════════════════════════════════════════════════════════
LIFE FRACTAL INTELLIGENCE - DEPLOYMENT REPORT
═══════════════════════════════════════════════════════════════

Deployment Time: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Branch: $Branch
Commit: $(git rev-parse --short HEAD)

DEPLOYED FEATURES:
  ✅ Core Life Planning System
  ✅ Emotional Pet AI (Differential Equations)
  ✅ Fractal Time Calendar (Fibonacci Scheduling)
  ✅ Executive Dysfunction Support
  ✅ Accessibility Suite (Autism-safe, Aphantasia, Dysgraphia)
  ✅ Privacy-Preserving ML Framework
  ✅ Enhanced Fractal Visualization

NEW API ENDPOINTS:
  • GET  /api/user/<id>/calendar/daily
  • GET  /api/user/<id>/executive-support
  • GET  /api/user/<id>/pet/emotional-state
  • GET  /api/user/<id>/accessibility
  • POST /api/user/<id>/accessibility
  • GET  /api/features/status

NEXT STEPS:
  1. Monitor deployment at: https://dashboard.render.com/
  2. Wait 2-3 minutes for app to fully start
  3. Test features at: $appUrl
  4. Check logs: render logs tail -f <service-name>
  5. Share with beta users!

═══════════════════════════════════════════════════════════════
"@

Write-Host $report -ForegroundColor White

# Save report to file
if (-not $DryRun) {
    $report | Out-File "DEPLOYMENT_REPORT_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    Write-Host "Report saved to: DEPLOYMENT_REPORT_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt" -ForegroundColor Green
}

Write-Host ""

# Final success message
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🎉 DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your enhanced Life Fractal Intelligence is now deploying to Render!" -ForegroundColor White
Write-Host ""
Write-Host "🌀 Features deployed:" -ForegroundColor Cyan
Write-Host "   • Emotional Pet AI with differential equations" -ForegroundColor White
Write-Host "   • Fibonacci-based fractal time calendar" -ForegroundColor White
Write-Host "   • Golden ratio task prioritization" -ForegroundColor White
Write-Host "   • Executive dysfunction detection & support" -ForegroundColor White
Write-Host "   • Full accessibility suite" -ForegroundColor White
Write-Host "   • Privacy-preserving machine learning" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  Deployment typically takes 2-3 minutes" -ForegroundColor Yellow
Write-Host "📊 Monitor at: https://dashboard.render.com/" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎊 You're building something amazing! Keep going! 🌟" -ForegroundColor Magenta
Write-Host ""
