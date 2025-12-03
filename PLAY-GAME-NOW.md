# 🎮 COVER FACE - QUICK START GUIDE
## Get Your 3D Life Planning Game Running NOW!

---

## ⚡ FASTEST WAY (30 Seconds)

### Just Run This:

```powershell
.\DEPLOY-NOW.ps1
```

**That's it!** Wait 10 minutes, then go to:
👉 **https://planner-1-pyd9.onrender.com/game**

---

## 🎯 THREE DEPLOYMENT OPTIONS

### 🚀 Option 1: INSTANT (Easiest)
**Best for:** Quick deployments, no questions asked

```powershell
.\DEPLOY-NOW.ps1
```

**What it does:**
- Adds all files
- Commits with timestamp
- Pushes to GitHub
- Done in 10 seconds!

---

### 🎮 Option 2: ONE-CLICK (Recommended)
**Best for:** First time deployment, want to see progress

```powershell
.\DEPLOY-EASY.ps1
```

**What it does:**
- Shows you each step
- Progress indicators
- Opens Render dashboard
- Game instructions included

---

### 🔧 Option 3: FULL-FEATURED (Advanced)
**Best for:** Want full control, see all details

```powershell
.\DEPLOY-TO-RENDER.ps1
```

**What it does:**
- Checks everything
- Asks for commit message
- Generates SECRET_KEY
- Full post-deploy checklist
- Detailed instructions

---

## 📋 STEP-BY-STEP FOR BEGINNERS

### Step 1: Open PowerShell
- Press `Windows + X`
- Click "Windows PowerShell"

### Step 2: Navigate to Your Folder
```powershell
cd C:\Users\Luke\Desktop\planner
```

### Step 3: Run Deployment
```powershell
.\DEPLOY-EASY.ps1
```

### Step 4: Wait (5-10 minutes)
- Watch Render dashboard
- Wait for "Deploy live"

### Step 5: Play Your Game!
Go to: **https://planner-1-pyd9.onrender.com/game**

---

## 🎮 GAME CONTROLS

Once the game loads:

**Movement:**
- `W` - Move forward
- `A` - Move left
- `S` - Move backward
- `D` - Move right
- `Shift` - Sprint (run faster)
- `Spacebar` - Jump

**Camera:**
- Move mouse - Look around
- Click canvas - Lock mouse (for smooth camera)
- `Esc` - Unlock mouse

**Interactions:**
- Click goal orbs - Work on goals
- `C` - Capture screenshot
- `1-6` - Switch regions (future)

---

## ⚙️ FIRST TIME SETUP (Required)

### You Need to Set Environment Variables in Render:

1. **Go to:** https://dashboard.render.com
2. **Click:** Your service (planner-1-pyd9)
3. **Click:** Environment (left sidebar)
4. **Add these variables:**

```
SECRET_KEY=<generate-below>
PORT=8080
DEBUG=False
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=onlinediscountsllc@gmail.com
SMTP_PASSWORD=<gmail-app-password>
```

### Generate SECRET_KEY:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output and paste as `SECRET_KEY`

### Get Gmail App Password:
1. Go to: https://myaccount.google.com
2. Security → 2-Step Verification (enable)
3. Security → App passwords
4. Generate: Mail → Other → "Life Fractal"
5. Copy 16-character password
6. Paste as `SMTP_PASSWORD`

---

## 🐛 TROUBLESHOOTING

### Problem: "Script cannot be loaded"
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: "Git is not recognized"
**Solution:**
1. Install Git: https://git-scm.com/download/win
2. Restart PowerShell

### Problem: "Push failed"
**Solution:**
```powershell
# Check your branch name
git branch

# Push to correct branch
git push origin main
# OR
git push origin master
```

### Problem: Game doesn't load
**Solutions:**
1. Wait 10 minutes (deployment takes time)
2. Check Render dashboard shows "Live"
3. Try Ctrl+F5 (hard refresh)
4. Check environment variables are set
5. Try different browser (Chrome/Edge best)

### Problem: Black screen in game
**Solutions:**
1. Wait 5-10 seconds for loading
2. Click on the canvas
3. Check browser console (F12) for errors

### Problem: Can't move character
**Solutions:**
1. Click on game canvas first
2. Make sure mouse is locked (click canvas)
3. Check WASD keys aren't stuck

---

## 📊 DEPLOYMENT TIMELINE

```
00:00  ➜  Run deployment script
00:10  ➜  Files pushed to GitHub
00:30  ➜  Render detects changes
02:00  ➜  Render starts building
05:00  ➜  Building dependencies
08:00  ➜  Starting application
10:00  ➜  LIVE! Game is ready! 🎮
```

---

## ✅ SUCCESS CHECKLIST

When game is working, you should see:

**In the game:**
- [ ] 3D world renders (green terrain)
- [ ] Pink character sphere visible
- [ ] Colored goal orbs floating
- [ ] HUD in top-left (Level, XP, Energy)
- [ ] Goals panel on right side

**Controls work:**
- [ ] WASD moves character
- [ ] Mouse rotates camera
- [ ] Can click goal orbs
- [ ] Screenshot button works

---

## 🎯 WHAT FILES YOU HAVE

**Core Game Files:**
- `cover_face_game_v1.py` - Main 3D game
- `audio_system.py` - Binaural beats
- `secure_auth_module.py` - Login system
- `life_fractal_v8_secure.py` - Backend API

**Deployment Scripts:**
- `DEPLOY-NOW.ps1` ⚡ Fastest (10 seconds)
- `DEPLOY-EASY.ps1` 🎮 Easy (shows progress)
- `DEPLOY-TO-RENDER.ps1` 🔧 Full (all options)

**Testing:**
- `test_bugs.py` - 15 automated tests
- `requirements.txt` - Dependencies

---

## 📱 QUICK REFERENCE

**Deploy:**
```powershell
.\DEPLOY-NOW.ps1
```

**Game URL:**
```
https://planner-1-pyd9.onrender.com/game
```

**Dashboard URL:**
```
https://planner-1-pyd9.onrender.com
```

**Monitor Deployment:**
```
https://dashboard.render.com
```

---

## 🎉 WHAT EACH VERSION GIVES YOU

### Current Deployment (v8.0):
✅ Secure login with CAPTCHA
✅ Email notifications
✅ Trial management
✅ Traditional dashboard

### After COVER FACE Deploy:
✅ Everything above PLUS:
✅ 3D open world game
✅ Animated pet characters
✅ Goal orbs in 3D space
✅ Binaural beats audio
✅ Fractal terrain
✅ No-typing gameplay
✅ Screenshot capture

---

## 💡 PRO TIPS

**Tip 1: Bookmark These URLs**
- Game: https://planner-1-pyd9.onrender.com/game
- Render: https://dashboard.render.com

**Tip 2: Use DEPLOY-NOW.ps1 for Quick Updates**
After first deployment, just run:
```powershell
.\DEPLOY-NOW.ps1
```
Done in 10 seconds!

**Tip 3: Test Locally First (Optional)**
```powershell
python cover_face_game_v1.py
# Then visit: http://localhost:8080/game
```

**Tip 4: Monitor Deployment**
Watch Render dashboard for:
- "Building" → Building app
- "Live" → Ready to use!

**Tip 5: Browser Performance**
- Best: Chrome, Edge
- Good: Firefox
- Limited: Safari

---

## 📞 SUPPORT

**Email:** onlinediscountsllc@gmail.com
**GoFundMe:** https://gofund.me/8d9303d27
**Render Dashboard:** https://dashboard.render.com

---

## 🚀 READY? LET'S GO!

### The Absolute Simplest Path:

1. Open PowerShell
2. `cd C:\Users\Luke\Desktop\planner`
3. `.\DEPLOY-NOW.ps1`
4. Wait 10 minutes
5. Go to: https://planner-1-pyd9.onrender.com/game
6. **PLAY!** 🎮

---

## 🎊 CONGRATULATIONS!

You're about to experience:
- 🎮 Your life goals in a 3D world
- 🐱 Animated pet character
- 🏔️ Fractal terrain generated by sacred math
- 🎵 Binaural beats for focus
- 🎯 No-typing gameplay
- 📸 Beautiful screenshot captures
- ✨ All your fractal mathematics visualized!

**Built specifically for neurodivergent minds like yours!**

---

*"Your Life. Your World. Your Game."*

**COVER FACE v1.0** - December 2025 🐱🐉🦊
