# 🚀 QUICKSTART - Deploy in 5 Minutes

## What You're Getting

✅ Self-healing production app  
✅ Email verification system  
✅ PostgreSQL database  
✅ Complete life planning platform  
✅ Ready for users immediately  

---

## Step 1: Prerequisites (2 minutes)

### Install Heroku CLI

**Windows:**
Download and run: https://cli-assets.heroku.com/heroku-x64.exe

**Mac:**
```bash
brew tap heroku/brew && brew install heroku
```

**Linux:**
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

### Verify Installation
```powershell
heroku --version
```

---

## Step 2: Deploy (3 minutes)

### Option A: Automated (Easiest)

1. Extract this folder
2. Right-click folder → "Open PowerShell window here"
3. Run:
   ```powershell
   .\deploy.ps1
   ```
4. Follow the prompts
5. Done!

### Option B: Manual

```powershell
# Login
heroku login

# Create app (replace YOUR-APP-NAME)
heroku create YOUR-APP-NAME

# Add database
heroku addons:create heroku-postgresql:essential-0

# Deploy
git init
git add .
git commit -m "Deploy"
git push heroku master

# Open app
heroku open
```

---

## Step 3: Test (30 seconds)

1. Visit: `https://YOUR-APP-NAME.herokuapp.com`
2. Click "Register"
3. Create account
4. Check email (if configured)
5. Login and explore!

---

## What's Working Now

✅ User registration with email verification  
✅ Secure login/logout  
✅ Goal tracking  
✅ Habit tracking  
✅ Virtual pet system  
✅ Fractal generation  
✅ Self-healing error recovery  
✅ Health monitoring  
✅ 7-day free trial  

---

## Email Setup (Optional)

### For Gmail:

1. Enable 2FA on Gmail
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Configure:
   ```powershell
   heroku config:set SMTP_SERVER=smtp.gmail.com
   heroku config:set SMTP_PORT=587
   heroku config:set SMTP_USERNAME=your@gmail.com
   heroku config:set SMTP_PASSWORD=16-char-app-password
   heroku config:set FROM_EMAIL=noreply@yourdomain.com
   ```

### Without Email:

- App works fine without email
- Verification tokens logged to Heroku logs
- View with: `heroku logs --tail`

---

## Common Commands

```powershell
# View logs (real-time)
heroku logs --tail

# Restart app
heroku restart

# Open app
heroku open

# Check health
curl https://YOUR-APP-NAME.herokuapp.com/health
```

---

## Costs

**Free Tier:** $0/month  
- 1000 dyno hours/month (enough for testing)
- Sleeps after 30 min inactivity

**Basic Tier:** $7/month  
- Never sleeps
- Perfect for production

**Database:** Free with credit card verification

---

## Next Steps

1. ✅ Deploy (done!)
2. Test all features
3. Configure email (optional)
4. Add custom domain
5. Set up Stripe for payments
6. Invite beta users

---

## Need Help?

```powershell
# View logs
heroku logs --tail

# Check status
heroku ps

# Open dashboard
heroku dashboard
```

**Full docs:** See README.md

---

## That's It!

Your production app is live at:
**https://YOUR-APP-NAME.herokuapp.com**

**Time to go live: 5 minutes ✨**
