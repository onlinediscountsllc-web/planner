# POWERSHELL DEPLOYMENT SCRIPTS - INSTRUCTIONS
# ============================================

## Two Scripts Available

### 1. DEPLOY-TO-RENDER.ps1 (Full Featured)
**What it does:**
- Checks prerequisites (Git installed, in Git repo)
- Verifies all required files are present
- Shows Git status
- Prompts for commit message
- Adds all changes to Git
- Commits with your message
- Pushes to GitHub
- Provides post-deployment checklist
- Can generate SECRET_KEY
- Opens Render dashboard

**When to use:** First time deploying or when you want full control

---

### 2. QUICK-DEPLOY.ps1 (One-Click)
**What it does:**
- Adds all changes
- Commits with timestamp
- Pushes to GitHub
- Shows completion message

**When to use:** Quick updates after initial setup

---

## STEP-BY-STEP: First Time Deployment

### Prerequisites

1. **Install Git for Windows** (if not already installed)
   - Download from: https://git-scm.com/download/win
   - Install with default settings
   - Restart PowerShell after installation

2. **Copy v8.0 Files to Your Repo**
   ```
   Copy these files to your planner folder:
   - secure_auth_module.py
   - life_fractal_v8_secure.py
   - requirements.txt
   - test_bugs.py
   - setup_local.py
   - deploy_to_render.py
   - DEPLOY-TO-RENDER.ps1
   - QUICK-DEPLOY.ps1
   ```

### Method 1: Using DEPLOY-TO-RENDER.ps1 (Recommended First Time)

1. **Open PowerShell**
   - Press `Windows + X`
   - Select "Windows PowerShell" or "Terminal"

2. **Navigate to Your Project**
   ```powershell
   cd C:\path\to\your\planner
   ```

3. **Enable Script Execution** (if needed)
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   - Type `Y` and press Enter if prompted

4. **Run the Deployment Script**
   ```powershell
   .\DEPLOY-TO-RENDER.ps1
   ```

5. **Follow the Prompts**
   - Review changes detected
   - Enter commit message (or press Enter for default)
   - Script will add, commit, and push to GitHub
   - Choose whether to open Render dashboard
   - Optionally generate SECRET_KEY

6. **Monitor Deployment in Render**
   - Go to https://dashboard.render.com
   - Find your service: planner-1-pyd9
   - Click "Events" tab
   - Wait 5-10 minutes for deployment

7. **Set Environment Variables in Render** (CRITICAL!)
   ```
   SECRET_KEY=<generated-secret-key>
   PORT=8080
   DEBUG=False
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=onlinediscountsllc@gmail.com
   SMTP_PASSWORD=<your-gmail-app-password>
   ```

8. **Test Deployment**
   ```powershell
   python test_bugs.py https://planner-1-pyd9.onrender.com
   ```

---

### Method 2: Using QUICK-DEPLOY.ps1 (For Updates)

**After initial setup is complete, use this for quick updates:**

1. **Open PowerShell in your project folder**
   - Navigate to folder in File Explorer
   - Type `powershell` in the address bar
   - Press Enter

2. **Run Quick Deploy**
   ```powershell
   .\QUICK-DEPLOY.ps1
   ```

3. **Done!**
   - Changes committed and pushed
   - Render auto-deploys

---

## Troubleshooting

### "Script cannot be loaded" Error

**Problem:** PowerShell execution policy blocks scripts

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Git is not recognized" Error

**Problem:** Git not installed or not in PATH

**Solutions:**
1. Install Git from: https://git-scm.com/download/win
2. Restart PowerShell
3. Or use full path: `C:\Program Files\Git\bin\git.exe`

### "Not a Git Repository" Error

**Problem:** Script not run from project root

**Solution:**
1. Navigate to correct folder:
   ```powershell
   cd C:\path\to\planner
   ```
2. Or initialize Git:
   ```powershell
   git init
   git remote add origin https://github.com/onlinediscountsllc-web/planner.git
   ```

### "Authentication Failed" Error

**Problem:** GitHub credentials not configured

**Solution:**
1. Configure Git credentials:
   ```powershell
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   ```

2. Or use GitHub Desktop for easier authentication

### "Push Rejected" Error

**Problem:** Remote has changes you don't have

**Solution:**
```powershell
git pull origin main --rebase
.\DEPLOY-TO-RENDER.ps1
```

### Render Not Auto-Deploying

**Problem:** Render not connected to GitHub properly

**Solution:**
1. Go to Render dashboard
2. Click your service
3. Settings → Build & Deploy
4. Verify "Auto-Deploy" is set to "Yes"
5. Verify branch is correct (usually "main")
6. Click "Manual Deploy" → "Deploy latest commit"

---

## PowerShell Tips

### Run as Administrator (if needed)
1. Right-click PowerShell
2. Select "Run as Administrator"

### Check Git Status
```powershell
git status
```

### View Recent Commits
```powershell
git log --oneline -5
```

### Force Push (use carefully!)
```powershell
git push origin main --force
```

### Undo Last Commit (keep changes)
```powershell
git reset --soft HEAD~1
```

---

## What the Scripts Do Behind the Scenes

### DEPLOY-TO-RENDER.ps1

```powershell
# 1. Check Git is installed
git --version

# 2. Check we're in a Git repo
Test-Path ".git"

# 3. Check for required files
Test-Path "secure_auth_module.py"
Test-Path "life_fractal_v8_secure.py"
# ... etc

# 4. Show what changed
git status --short

# 5. Stage all changes
git add .

# 6. Commit changes
git commit -m "Your message here"

# 7. Push to GitHub
git push origin main
```

### QUICK-DEPLOY.ps1

```powershell
# 1. Stage everything
git add .

# 2. Commit with timestamp
git commit -m "Deploy Life Fractal v8.0 - 2025-01-15 14:30"

# 3. Push to current branch
git push origin $branch
```

---

## After Successful Deployment

### 1. Verify Deployment
- Check Render dashboard shows "Live"
- Health check: https://planner-1-pyd9.onrender.com/health

### 2. Test Functionality
```powershell
python test_bugs.py https://planner-1-pyd9.onrender.com
```

### 3. Register Test Account
- Go to your app
- Register with test email
- Verify welcome email arrives
- Check GoFundMe link appears

### 4. Monitor Logs
- Render dashboard → Your service → Logs
- Watch for errors or warnings

---

## Environment Variables Reference

### Required in Render

```bash
SECRET_KEY=<64-char-hex-string>
PORT=8080
DEBUG=False
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=onlinediscountsllc@gmail.com
SMTP_PASSWORD=<16-char-gmail-app-password>
```

### Generate SECRET_KEY

**In PowerShell:**
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

**Or use the deployment script's built-in generator**

### Get Gmail App Password

1. Google Account → Security
2. Enable 2-Factor Authentication
3. App Passwords → Mail → Generate
4. Copy 16-character password

---

## Quick Reference Commands

```powershell
# Navigate to project
cd C:\path\to\planner

# Full deployment
.\DEPLOY-TO-RENDER.ps1

# Quick deployment
.\QUICK-DEPLOY.ps1

# Check status
git status

# View logs
git log --oneline -10

# Test deployment
python test_bugs.py https://planner-1-pyd9.onrender.com

# Generate secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Support

- **Email:** onlinediscountsllc@gmail.com
- **GoFundMe:** https://gofund.me/8d9303d27
- **Render Dashboard:** https://dashboard.render.com
- **Deployed URL:** https://planner-1-pyd9.onrender.com

---

## Next Steps After Deployment

1. ✅ Deployment successful
2. ✅ Tests passing
3. ✅ Email delivery working
4. → Share GoFundMe with users
5. → Monitor user registrations
6. → Track trial conversions
7. → Gather feedback

---

**Happy Deploying!** 🚀

*Life Fractal Intelligence v8.0 - PowerShell Edition*
