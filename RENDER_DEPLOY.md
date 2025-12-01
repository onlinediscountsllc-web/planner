# 🚀 RENDER DEPLOYMENT GUIDE - Life Fractal Intelligence

Render is EASIER than Heroku with a better free tier!

## ⚡ SUPER QUICK METHOD (5 Minutes)

### Step 1: Copy Files to Your Planner Folder

Copy these 3 files into `C:\Users\Luke\Desktop\planner\`:

1. **render.yaml** - I created this for you
2. **requirements.txt** - Make sure you have this
3. **app.py** - Your main application (you already have this)

### Step 2: Push to GitHub

```powershell
# Configure Git (if you haven't)
git config --global user.email "onlinediscountsllc@gmail.com"
git config --global user.name "Luke"

# Initialize repo
git init
git add .
git commit -m "Life Fractal Intelligence - Ready for Render"

# Create repo on GitHub (go to github.com/new)
# Then connect it:
git remote add origin https://github.com/YOUR-USERNAME/life-fractal-app.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with GitHub
4. Click **"New +"** → **"Web Service"**
5. Select your **life-fractal-app** repo
6. Render auto-detects everything from render.yaml!
7. Click **"Create Web Service"**
8. Wait 3-5 minutes
9. **Your app is LIVE!**

---

## 📋 DETAILED STEPS

### Step 1: Prepare Your Files

Make sure you have these files in `C:\Users\Luke\Desktop\planner\`:

```
planner/
├── app.py                  # Your Flask app
├── requirements.txt        # Python dependencies
├── render.yaml            # Render configuration (download below)
├── .gitignore             # Git ignore file
└── Procfile               # Optional (Render can use render.yaml)
```

**Download render.yaml:**
[Download render.yaml](computer:///mnt/user-data/outputs/render.yaml)

Save it to: `C:\Users\Luke\Desktop\planner\render.yaml`

### Step 2: Check Your Files

**requirements.txt should have:**
```
Flask==3.0.0
Flask-Cors==4.0.0
Werkzeug==3.0.1
gunicorn==21.2.0
psycopg2-binary==2.9.9
PyJWT==2.8.0
Flask-Limiter==3.5.0
numpy==1.26.2
Pillow==10.1.0
scikit-learn==1.3.2
stripe==7.8.0
python-dotenv==1.0.0
```

**app.py should have at the end:**
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

### Step 3: Create GitHub Repository

**Option A: Using GitHub Desktop (Easiest)**
1. Download GitHub Desktop: https://desktop.github.com
2. Install and sign in
3. File → Add Local Repository → Browse to your planner folder
4. Click "Publish repository"
5. Name it "life-fractal-app"
6. Click "Publish Repository"

**Option B: Using Git Command Line**
```powershell
cd C:\Users\Luke\Desktop\planner

# Configure Git
git config --global user.email "onlinediscountsllc@gmail.com"
git config --global user.name "Luke"

# Initialize
git init
git add .
git commit -m "Initial commit - Life Fractal Intelligence"

# Create repo on GitHub.com first, then:
git remote add origin https://github.com/YOUR-USERNAME/life-fractal-app.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy to Render

1. **Go to Render**: https://render.com
2. **Sign Up**: Use your GitHub account (easiest)
3. **New Web Service**: Click "New +" → "Web Service"
4. **Connect Repo**: Authorize GitHub → Select "life-fractal-app"
5. **Auto-Configure**: Render reads render.yaml automatically!
6. **Review Settings**:
   - Name: life-fractal-intelligence
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
7. **Create**: Click "Create Web Service"
8. **Wait**: 3-5 minutes for deployment
9. **Success**: Your app is live!

---

## 🆓 RENDER FREE TIER

What you get for FREE:
- ✅ 750 hours/month (enough for 24/7)
- ✅ PostgreSQL database (FREE - 1GB storage)
- ✅ Automatic HTTPS
- ✅ Auto-deploy from GitHub
- ✅ Custom domains
- ✅ 90GB bandwidth/month

**Better than Heroku free tier!**

---

## 🔧 RENDER vs HEROKU

| Feature | Render | Heroku |
|---------|--------|--------|
| Free Tier | 750 hrs/month | Deprecated |
| Database | Free 1GB | $5/month minimum |
| Setup | Easier (render.yaml) | More complex |
| Build Time | Faster | Slower |
| GitHub Integration | Automatic | Manual |

**Render is the better choice!**

---

## 📊 AFTER DEPLOYMENT

Your app will be at:
```
https://life-fractal-intelligence.onrender.com
```

**What works immediately:**
- ✅ User registration & login
- ✅ Goals & habits tracking
- ✅ Daily wellness check-ins
- ✅ Virtual pet system
- ✅ Fractal generation
- ✅ PostgreSQL database
- ✅ HTTPS security

---

## 🔐 ENVIRONMENT VARIABLES

Render automatically sets:
- `SECRET_KEY` - Auto-generated
- `DATABASE_URL` - Auto-configured
- `PORT` - Auto-set

**To add Stripe (optional):**
1. Go to your service on Render
2. Environment tab
3. Add:
   - `STRIPE_SECRET_KEY` = your_stripe_key
   - `STRIPE_PUBLISHABLE_KEY` = your_publishable_key

---

## 🔄 UPDATE YOUR APP

When you make changes:

```powershell
cd C:\Users\Luke\Desktop\planner
git add .
git commit -m "Updated feature X"
git push
```

**Render auto-deploys!** No manual steps needed.

---

## 📝 TROUBLESHOOTING

### Build Fails
Check build logs on Render dashboard

### App Won't Start
Check logs: Dashboard → Logs tab

### Database Connection Error
Make sure render.yaml includes database configuration

### Import Errors
Add missing packages to requirements.txt

---

## 🎯 QUICK START CHECKLIST

- [ ] Download render.yaml
- [ ] Save to planner folder
- [ ] Have requirements.txt ready
- [ ] Create GitHub account
- [ ] Push code to GitHub
- [ ] Create Render account
- [ ] Connect GitHub repo
- [ ] Deploy web service
- [ ] App is live!

---

## 💡 PRO TIPS

1. **Use render.yaml** - Simplifies deployment
2. **Enable auto-deploy** - Updates automatically
3. **Check logs regularly** - Catch issues early
4. **Use environment variables** - For secrets
5. **Add health checks** - Monitors app health

---

## 🆘 NEED HELP?

- Render Docs: https://render.com/docs
- Render Discord: https://discord.gg/render
- GitHub Help: https://docs.github.com

---

## 🚀 READY TO DEPLOY?

**Simplest Path:**
1. Copy render.yaml to your planner folder
2. Push to GitHub (use GitHub Desktop if easier)
3. Connect to Render
4. Auto-deploy!

**Your app will be live in 10 minutes!**

---

## 📥 DOWNLOADS

- [render.yaml](computer:///mnt/user-data/outputs/render.yaml)
- GitHub Desktop: https://desktop.github.com
- Git: https://git-scm.com/download/win

---

**Render is MUCH easier than Heroku. You've got this!** 🌀
