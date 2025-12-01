# 🧪 LIFE FRACTAL INTELLIGENCE - COMPREHENSIVE TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════════════
# Tests all enhanced features after deployment
# Run after deployment to verify everything works
# ═══════════════════════════════════════════════════════════════════════════════════════

param(
    [Parameter(Mandatory=$false)]
    [string]$AppUrl = "",
    [switch]$Verbose,
    [switch]$CreateTestData
)

$ErrorActionPreference = "Stop"

Write-Host "🧪 LIFE FRACTAL INTELLIGENCE - COMPREHENSIVE TEST SUITE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Get app URL if not provided
if (-not $AppUrl) {
    Write-Host "Enter your Render app URL" -ForegroundColor Yellow
    Write-Host "Example: https://your-app.onrender.com" -ForegroundColor Gray
    $AppUrl = Read-Host "URL"
    
    if (-not $AppUrl) {
        Write-Host "❌ No URL provided" -ForegroundColor Red
        exit 1
    }
}

# Remove trailing slash
$AppUrl = $AppUrl.TrimEnd('/')

Write-Host "Testing app at: $AppUrl" -ForegroundColor Cyan
Write-Host ""

# Test results
$testResults = @{
    Passed = 0
    Failed = 0
    Warnings = 0
    Tests = @()
}

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Endpoint,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null,
        [scriptblock]$Validator = $null
    )
    
    Write-Host "Testing: $Name" -ForegroundColor Yellow
    Write-Host "  Endpoint: $Endpoint" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = "$AppUrl$Endpoint"
            Method = $Method
            Headers = $Headers
            TimeoutSec = 30
            UseBasicParsing = $true
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @params
        
        if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 201) {
            $data = $response.Content | ConvertFrom-Json
            
            # Run custom validator if provided
            if ($Validator) {
                $validationResult = & $Validator $data
                if ($validationResult.Success) {
                    Write-Host "  ✅ PASSED" -ForegroundColor Green
                    if ($validationResult.Message) {
                        Write-Host "     $($validationResult.Message)" -ForegroundColor Gray
                    }
                    $script:testResults.Passed++
                    $script:testResults.Tests += @{
                        Name = $Name
                        Status = "PASSED"
                        Message = $validationResult.Message
                    }
                    return $data
                } else {
                    Write-Host "  ❌ FAILED: $($validationResult.Message)" -ForegroundColor Red
                    $script:testResults.Failed++
                    $script:testResults.Tests += @{
                        Name = $Name
                        Status = "FAILED"
                        Message = $validationResult.Message
                    }
                    return $null
                }
            } else {
                Write-Host "  ✅ PASSED" -ForegroundColor Green
                $script:testResults.Passed++
                $script:testResults.Tests += @{
                    Name = $Name
                    Status = "PASSED"
                }
                return $data
            }
        } else {
            Write-Host "  ⚠️  WARNING: Status $($response.StatusCode)" -ForegroundColor Yellow
            $script:testResults.Warnings++
            $script:testResults.Tests += @{
                Name = $Name
                Status = "WARNING"
                Message = "HTTP $($response.StatusCode)"
            }
            return $null
        }
    } catch {
        Write-Host "  ❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $script:testResults.Failed++
        $script:testResults.Tests += @{
            Name = $Name
            Status = "FAILED"
            Message = $_.Exception.Message
        }
        return $null
    }
}

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PHASE 1: CORE SYSTEM TESTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Test-Endpoint `
    -Name "Health Check" `
    -Endpoint "/health" `
    -Validator {
        param($data)
        if ($data.status -eq "healthy" -or $data.status -eq "ok") {
            return @{ Success = $true; Message = "App is healthy" }
        } else {
            return @{ Success = $false; Message = "App reported unhealthy status" }
        }
    }

Write-Host ""

# Test 2: Features Status
$featuresData = Test-Endpoint `
    -Name "Features Status" `
    -Endpoint "/api/features/status" `
    -Validator {
        param($data)
        if ($data.enhanced_features_available) {
            return @{ Success = $true; Message = "Enhanced features are active" }
        } else {
            return @{ Success = $false; Message = "Enhanced features not loaded" }
        }
    }

if ($featuresData -and $Verbose) {
    Write-Host "  Feature Details:" -ForegroundColor Cyan
    foreach ($feature in $featuresData.features.PSObject.Properties) {
        $status = if ($feature.Value) { "✅" } else { "❌" }
        Write-Host "    $status $($feature.Name)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PHASE 2: AUTHENTICATION TESTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Test 3: User Registration
$testEmail = "test_$(Get-Random)@example.com"
$testPassword = "TestPassword123!"

$registerData = Test-Endpoint `
    -Name "User Registration" `
    -Endpoint "/api/register" `
    -Method "POST" `
    -Body (@{
        email = $testEmail
        password = $testPassword
    } | ConvertTo-Json) `
    -Validator {
        param($data)
        if ($data.token) {
            return @{ Success = $true; Message = "User registered successfully" }
        } else {
            return @{ Success = $false; Message = "No token received" }
        }
    }

$authToken = if ($registerData) { $registerData.token } else { $null }
$userId = if ($registerData) { $registerData.user_id } else { $null }

Write-Host ""

# Test 4: User Login
if ($authToken) {
    Test-Endpoint `
        -Name "User Login" `
        -Endpoint "/api/login" `
        -Method "POST" `
        -Body (@{
            email = $testEmail
            password = $testPassword
        } | ConvertTo-Json) `
        -Validator {
            param($data)
            if ($data.token) {
                return @{ Success = $true; Message = "Login successful" }
            } else {
                return @{ Success = $false; Message = "Login failed" }
            }
        }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PHASE 3: ENHANCED FEATURES TESTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($authToken -and $userId) {
    $authHeaders = @{
        "Authorization" = "Bearer $authToken"
        "Content-Type" = "application/json"
    }
    
    # Test 5: Fractal Calendar
    $calendarData = Test-Endpoint `
        -Name "Fractal Calendar Generation" `
        -Endpoint "/api/user/$userId/calendar/daily" `
        -Headers $authHeaders `
        -Validator {
            param($data)
            if ($data.time_blocks -and $data.time_blocks.Count -gt 0) {
                $blockCount = $data.time_blocks.Count
                $totalSpoons = $data.total_available_spoons
                return @{ 
                    Success = $true
                    Message = "Generated $blockCount Fibonacci time blocks ($totalSpoons total spoons)"
                }
            } else {
                return @{ Success = $false; Message = "No time blocks generated" }
            }
        }
    
    if ($calendarData -and $Verbose) {
        Write-Host "  Calendar Details:" -ForegroundColor Cyan
        foreach ($block in $calendarData.time_blocks) {
            Write-Host "    $($block.start_time)-$($block.end_time): $($block.energy_phase) ($($block.spoon_capacity) spoons)" -ForegroundColor White
        }
    }
    
    Write-Host ""
    
    # Test 6: Executive Function Support
    Test-Endpoint `
        -Name "Executive Dysfunction Detection" `
        -Endpoint "/api/user/$userId/executive-support" `
        -Headers $authHeaders `
        -Validator {
            param($data)
            if ($data.PSObject.Properties['severity']) {
                return @{ 
                    Success = $true
                    Message = "Severity: $($data.severity), Score: $([math]::Round($data.score, 3))"
                }
            } else {
                return @{ Success = $false; Message = "No analysis data returned" }
            }
        }
    
    Write-Host ""
    
    # Test 7: Pet Emotional State
    Test-Endpoint `
        -Name "Pet Emotional State (if pet exists)" `
        -Endpoint "/api/user/$userId/pet/emotional-state" `
        -Headers $authHeaders `
        -Validator {
            param($data)
            if ($data.emotional_state) {
                $mood = [math]::Round($data.emotional_state.mood, 1)
                $energy = [math]::Round($data.emotional_state.energy, 1)
                return @{ 
                    Success = $true
                    Message = "Mood: $mood, Energy: $energy, Species: $($data.species)"
                }
            } else {
                return @{ Success = $false; Message = "No pet emotional state" }
            }
        }
    
    Write-Host ""
    
    # Test 8: Accessibility Settings
    $accessibilityData = Test-Endpoint `
        -Name "Accessibility Settings (GET)" `
        -Endpoint "/api/user/$userId/accessibility" `
        -Headers $authHeaders `
        -Validator {
            param($data)
            if ($data.PSObject.Properties['color_theme']) {
                return @{ 
                    Success = $true
                    Message = "Theme: $($data.color_theme), Contrast: $($data.contrast_level)"
                }
            } else {
                return @{ Success = $false; Message = "No accessibility settings" }
            }
        }
    
    Write-Host ""
    
    # Test 9: Update Accessibility Settings
    Test-Endpoint `
        -Name "Accessibility Settings (POST)" `
        -Endpoint "/api/user/$userId/accessibility" `
        -Method "POST" `
        -Headers $authHeaders `
        -Body (@{
            color_theme = "calm"
            contrast_level = "high"
            reduced_motion = $true
        } | ConvertTo-Json) `
        -Validator {
            param($data)
            if ($data.success -and $data.settings) {
                return @{ Success = $true; Message = "Settings updated successfully" }
            } else {
                return @{ Success = $false; Message = "Failed to update settings" }
            }
        }
    
} else {
    Write-Host "⚠️  Skipping enhanced features tests (no auth token)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PHASE 4: CORE FUNCTIONALITY TESTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($authToken -and $userId) {
    # Test 10: Dashboard
    Test-Endpoint `
        -Name "User Dashboard" `
        -Endpoint "/api/user/$userId/dashboard" `
        -Headers $authHeaders `
        -Validator {
            param($data)
            if ($data.user -and $data.stats) {
                return @{ Success = $true; Message = "Dashboard loaded successfully" }
            } else {
                return @{ Success = $false; Message = "Incomplete dashboard data" }
            }
        }
    
    Write-Host ""
    
    # Test 11: Create Goal
    $goalData = Test-Endpoint `
        -Name "Create Goal" `
        -Endpoint "/api/user/$userId/goals" `
        -Method "POST" `
        -Headers $authHeaders `
        -Body (@{
            title = "Test Goal - Enhanced Features"
            description = "Testing new mathematical goal system"
            category = "testing"
            priority = 1
        } | ConvertTo-Json) `
        -Validator {
            param($data)
            if ($data.success -and $data.goal) {
                return @{ Success = $true; Message = "Goal created: $($data.goal.title)" }
            } else {
                return @{ Success = $false; Message = "Failed to create goal" }
            }
        }
    
    Write-Host ""
    
    # Test 12: Create Habit
    Test-Endpoint `
        -Name "Create Habit" `
        -Endpoint "/api/user/$userId/habits" `
        -Method "POST" `
        -Headers $authHeaders `
        -Body (@{
            name = "Test Habit - Fibonacci Tracking"
            category = "wellness"
        } | ConvertTo-Json) `
        -Validator {
            param($data)
            if ($data.success -and $data.habit) {
                return @{ Success = $true; Message = "Habit created: $($data.habit.name)" }
            } else {
                return @{ Success = $false; Message = "Failed to create habit" }
            }
        }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$total = $testResults.Passed + $testResults.Failed + $testResults.Warnings
$passRate = if ($total -gt 0) { [math]::Round(($testResults.Passed / $total) * 100, 1) } else { 0 }

Write-Host "Total Tests: $total" -ForegroundColor White
Write-Host "✅ Passed: $($testResults.Passed)" -ForegroundColor Green
Write-Host "❌ Failed: $($testResults.Failed)" -ForegroundColor Red
Write-Host "⚠️  Warnings: $($testResults.Warnings)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Pass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 80) { "Green" } elseif ($passRate -ge 60) { "Yellow" } else { "Red" })
Write-Host ""

# Detailed results
if ($Verbose -or $testResults.Failed -gt 0) {
    Write-Host "DETAILED RESULTS:" -ForegroundColor Cyan
    Write-Host ""
    
    foreach ($test in $testResults.Tests) {
        $statusColor = switch ($test.Status) {
            "PASSED" { "Green" }
            "FAILED" { "Red" }
            "WARNING" { "Yellow" }
        }
        
        $icon = switch ($test.Status) {
            "PASSED" { "✅" }
            "FAILED" { "❌" }
            "WARNING" { "⚠️ " }
        }
        
        Write-Host "$icon $($test.Name)" -ForegroundColor $statusColor
        if ($test.Message) {
            Write-Host "   $($test.Message)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "RECOMMENDATIONS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($testResults.Failed -eq 0) {
    Write-Host "🎉 All tests passed! Your deployment is successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Share your app URL with beta users" -ForegroundColor White
    Write-Host "  2. Monitor Render logs for any issues" -ForegroundColor White
    Write-Host "  3. Collect user feedback" -ForegroundColor White
    Write-Host "  4. Iterate on features" -ForegroundColor White
} elseif ($testResults.Failed -le 2) {
    Write-Host "⚠️  Minor issues detected" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommendations:" -ForegroundColor Cyan
    Write-Host "  1. Review failed tests above" -ForegroundColor White
    Write-Host "  2. Check Render logs: render logs tail -f" -ForegroundColor White
    Write-Host "  3. Verify environment variables are set" -ForegroundColor White
    Write-Host "  4. Re-run tests after fixes" -ForegroundColor White
} else {
    Write-Host "❌ Multiple test failures detected" -ForegroundColor Red
    Write-Host ""
    Write-Host "Action required:" -ForegroundColor Cyan
    Write-Host "  1. Check Render deployment logs immediately" -ForegroundColor White
    Write-Host "  2. Verify all files were deployed correctly" -ForegroundColor White
    Write-Host "  3. Check Python dependencies in requirements.txt" -ForegroundColor White
    Write-Host "  4. Verify environment variables" -ForegroundColor White
    Write-Host "  5. Consider rolling back if critical features broken" -ForegroundColor White
}

Write-Host ""

# Save report
$reportPath = "TEST_REPORT_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$reportContent = @"
═══════════════════════════════════════════════════════════════
LIFE FRACTAL INTELLIGENCE - TEST REPORT
═══════════════════════════════════════════════════════════════

Test Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
App URL: $AppUrl

SUMMARY:
  Total Tests: $total
  Passed: $($testResults.Passed)
  Failed: $($testResults.Failed)
  Warnings: $($testResults.Warnings)
  Pass Rate: $passRate%

DETAILED RESULTS:
$(foreach ($test in $testResults.Tests) {
"  $($test.Status): $($test.Name)"
if ($test.Message) { "    $($test.Message)" }
""
} | Out-String)

═══════════════════════════════════════════════════════════════
"@

$reportContent | Out-File $reportPath
Write-Host "📊 Test report saved: $reportPath" -ForegroundColor Cyan

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🌀 Testing complete! Keep building something amazing! ✨" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
