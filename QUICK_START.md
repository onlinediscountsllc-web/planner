# 🚀 QUICK START - Life Fractal Intelligence v7.0

## Run Locally (30 seconds)

```bash
# Install dependencies
pip install Flask Flask-Cors Werkzeug gunicorn numpy Pillow

# Run the app
python app.py

# Open browser: http://localhost:5000
```

## Deploy to Render.com (5 minutes)

### Option 1: Using render.yaml (Recommended)

1. Create a new GitHub repository
2. Upload these files: `app.py`, `requirements.txt`, `render.yaml`
3. Go to https://render.com
4. Click "New" → "Blueprint"
5. Connect your GitHub repo
6. Render will auto-deploy!

### Option 2: Manual Setup

1. Go to https://render.com
2. Click "New" → "Web Service"
3. Connect GitHub repo with `app.py` and `requirements.txt`
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click "Create Web Service"

## Test It Works

After deployment, check:
- `https://your-app.onrender.com/api/health` → Should return `{"status": "healthy"}`
- `https://your-app.onrender.com/api/system/status` → Shows all features

## Features Included

✅ Emotional Pet AI (8 species with differential equations)
✅ Spoon Theory Energy Management
✅ Fractal Time Calendar (Fibonacci blocks)
✅ Fibonacci Task Scheduler (Golden ratio priority)
✅ Executive Function Support (Pattern detection)
✅ 2D/3D Fractal Visualization
✅ Mayan Calendar Integration
✅ Full Accessibility (5 color palettes)
✅ Complete Authentication
✅ Stripe Payment Ready

## Your Existing Deployment

Your current live site: https://planner-1-pyd9.onrender.com
Stripe payment link: https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00
GoFundMe: https://gofund.me/8d9303d27

## Update Existing Deployment

To update your live site:

1. Replace the code in your GitHub repo with the new `app.py`
2. Render will auto-deploy (or click "Manual Deploy" in dashboard)

---

🌀 **2,400+ lines of production code - ZERO placeholders!** 🌀
