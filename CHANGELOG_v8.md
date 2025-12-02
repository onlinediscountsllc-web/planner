# 🌀 Life Fractal Intelligence v8.0 - Complete Production Build

## For brains like mine - built with love for the neurodivergent community

---

## ✅ What's Fixed in v8.0

### 1. Neurodiversity Accessibility (∞ Symbol, Not Wheelchair)
- **Changed icon** from wheelchair ♿ to the **neurodiversity infinity symbol ∞**
- Rainbow gradient styling on the symbol to represent neurodiversity
- Full accessibility customization panel with **17 different settings**:
  - Visual: Reduced motion, high contrast, larger text, dyslexia font, calm colors
  - Navigation: Keyboard navigation, focus indicators, simplified layout
  - Executive function: Time blindness helpers, task chunking, gentle reminders, sensory breaks, auto-save
- **All settings save to the database** and persist across sessions

### 2. Complete User Management System
- **Secure registration** with email validation and password requirements
- **Password hashing** using PBKDF2-SHA256 (industry standard)
- **Session management** with 30-day persistence
- **Password reset** functionality with secure tokens
- User data stored in SQLite database (persistent across restarts)
- 7-day free trial system
- All user preferences and data saved securely

### 3. Working 3D Fractal Visualization
- **Three.js** powered interactive 3D view
- **Goal orbs** positioned using golden angle mathematics
- **Sacred geometry overlays** (icosahedron, dodecahedron)
- **Interactive controls**:
  - Mouse drag to rotate
  - Scroll to zoom
  - Right-click to pan
  - Space to pause/resume auto-rotation
- Mood and energy parameters affect the fractal appearance
- Real-time fractal regeneration

### 4. Pet Companion System
- All stats display correctly (no more NaN values)
- **8 pet species**: cat, dragon, phoenix, owl, fox, bunny, turtle, butterfly
- Dynamic emoji based on mood
- Feed, play, rest interactions work properly
- Experience and leveling system
- Mood affected by user check-ins

### 5. Spoon Theory Energy Tracking
- Spoons display in header
- Tasks cost spoons
- Completing tasks deducts spoons and rewards pet XP

### 6. Sacred/Mayan Calendar
- Accurate Tzolkin calculation
- Shows today's day sign and meaning

---

## 🚀 Deployment to Render.com

### Step 1: Update Your Repository
Replace your current `life_planner_unified_master.py` (or main file) with `life_fractal_complete_v8.py`.

Rename it if needed:
```bash
mv life_fractal_complete_v8.py app.py
```

### Step 2: Update requirements.txt
Use the provided `requirements.txt`:
```
Flask>=3.0.0
flask-cors>=4.0.0
Werkzeug>=3.0.0
numpy>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
gunicorn>=21.0.0
```

### Step 3: Create/Update render.yaml
```yaml
services:
  - type: web
    name: life-fractal-intelligence
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
      - key: SECRET_KEY
        generateValue: true
      - key: PRODUCTION
        value: true
```

### Step 4: Push and Deploy
```bash
git add .
git commit -m "Upgrade to v8.0 - Complete fixes"
git push origin main
```

---

## 📂 File Structure
```
your-repo/
├── app.py                    # Main application (life_fractal_complete_v8.py)
├── requirements.txt          # Dependencies
├── render.yaml               # Render deployment config
└── life_fractal.db           # SQLite database (auto-created)
```

---

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open in browser
http://localhost:5000
```

---

## 🌟 Features Summary

| Feature | Status |
|---------|--------|
| User Registration | ✅ Working |
| User Login | ✅ Working |
| Password Reset | ✅ Working |
| Session Persistence | ✅ Working |
| Accessibility Settings | ✅ All 17 options work |
| Neurodiversity Symbol | ✅ Rainbow infinity ∞ |
| Pet System | ✅ All stats display correctly |
| Pet Interactions | ✅ Feed, Play, Rest work |
| 3D Visualization | ✅ Interactive Three.js |
| Goal Orbs | ✅ Golden angle positioning |
| Sacred Geometry | ✅ Toggle-able overlays |
| Mayan Calendar | ✅ Accurate calculation |
| Spoon Tracking | ✅ Displays and tracks |
| Daily Check-in | ✅ Saves to database |
| Goals System | ✅ Create and track |
| Tasks System | ✅ Create and complete |

---

## 💡 For Neurodivergent Users

This app is designed specifically for how our brains work:

- **Aphantasia-friendly**: Everything is visual and text-based - no need to "picture" anything
- **ADHD-friendly**: Spoon theory tracks energy, gentle reminders, task chunking
- **Autism-friendly**: Predictable layouts, reduced sensory options, clear structure
- **Dyslexia-friendly**: OpenDyslexic font option, larger text option
- **Shame-free**: No guilt trips, celebrates all progress, adapts to your energy

---

## 🆘 Troubleshooting

### "NaN" values in pet stats
This is fixed in v8.0. If you see this, make sure you're running the new version.

### "undefined Spoons" in header
This is fixed in v8.0. The session now properly loads user data.

### 3D view not working
Make sure JavaScript is enabled. The 3D view uses Three.js loaded from CDN.

### Accessibility settings not saving
This is fixed in v8.0. Settings are now saved to the SQLite database.

---

## 🎁 Support Development

- GoFundMe: https://gofund.me/8d9303d27
- Stripe: $20/month subscription

---

Built with 💜 for the neurodivergent community
