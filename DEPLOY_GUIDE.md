# Life Fractal Intelligence v15.0 Ultimate
## Complete Deployment Guide for Render.com

---

## What's Fixed (All Critical Gaps Resolved)

| Issue | Status | Solution |
|-------|--------|----------|
| No Database | ✅ FIXED | SQLite with 8 tables, automatic schema creation |
| No Forgot Password | ✅ FIXED | Token-based reset with `/api/auth/forgot-password` |
| No Stripe | ✅ FIXED | Payment link integration, subscription status tracking |
| No Frontend | ✅ FIXED | Complete Nordic design dashboard, accessible UI |
| Insecure Auth | ✅ FIXED | HMAC-SHA256 JWT tokens (no pyjwt dependency) |

---

## Mathematical Foundations (Occam's Razor Applied)

All math is **pure Python** - no scipy, sklearn, or heavy dependencies:

- **Golden Ratio (φ)**: 1.618033988749895
- **Golden Angle**: 137.5077640500378°
- **Fibonacci Sequence**: Pre-computed for efficiency
- **Logistic Map**: Chaos theory for fractal variation
- **Pythagorean Means**: Arithmetic, geometric, harmonic
- **Spoon Theory Cost**: Fibonacci-weighted energy calculation

---

## Accessibility Features (Neurodivergent-First)

### Autism
- Predictable patterns and layouts
- Muted, sensory-safe colors
- Clear visual hierarchy
- No surprise animations

### Aphantasia
- External visualization (fractals show what you can't imagine)
- Concrete progress bars
- Text descriptions available

### Dysgraphia
- Large touch targets (48px minimum)
- Slider-based inputs
- Minimal typing required

### ADHD / Executive Dysfunction
- Spoon Theory energy tracking
- Task breakdown support
- Visual gamification (pet companion)

---

## Files to Deploy

```
app.py           - Main application (2,400+ lines)
requirements.txt - Minimal dependencies (7 packages)
```

---

## Quick Deploy to Render.com

### Step 1: Prepare Files

1. Download `app.py` and `requirements.txt`
2. Create a new folder: `life-fractal`
3. Put both files in the folder

### Step 2: Initialize Git

```bash
cd life-fractal
git init
git add .
git commit -m "Life Fractal v15.0 Ultimate - All fixes"
```

### Step 3: Push to GitHub

```bash
# Create repo on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/life-fractal.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Render

1. Go to https://dashboard.render.com
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name**: `life-fractal` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

### Step 5: Set Environment Variables

In Render Dashboard → Environment:

```
SECRET_KEY=your-random-32-char-string-here
JWT_SECRET=another-random-32-char-string
STRIPE_PAYMENT_LINK=https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00
GOFUNDME_URL=https://gofund.me/8d9303d27
```

### Step 6: Deploy!

Click **Deploy** and wait 2-3 minutes.

Your app will be live at: `https://life-fractal.onrender.com`

---

## API Endpoints Reference

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account with 7-day trial |
| POST | `/api/auth/login` | Login, receive JWT token |
| POST | `/api/auth/logout` | Logout (discard token) |
| POST | `/api/auth/forgot-password` | Request password reset |
| POST | `/api/auth/reset-password` | Reset with token |

### User & Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/user/profile` | Get user profile |
| GET | `/api/dashboard` | Get complete dashboard data |

### Daily Check-in
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/daily/today` | Get today's entry |
| POST | `/api/daily/today` | Save/update today's entry |

### Goals
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/goals` | List all goals |
| POST | `/api/goals` | Create goal |
| POST | `/api/goals/<id>/progress` | Update progress |

### Habits
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/habits` | List all habits |
| POST | `/api/habits` | Create habit |
| POST | `/api/habits/<id>/complete` | Mark complete |

### Virtual Pet
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/pet` | Get pet status |
| POST | `/api/pet/feed` | Feed pet |
| POST | `/api/pet/play` | Play with pet |

### Visualization
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/visualization/params` | Get fractal parameters |
| GET | `/api/visualization/fractal` | Get PNG image |
| GET | `/api/visualization/fractal/base64` | Get base64 image |

### Subscription
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/subscription/status` | Check subscription |
| POST | `/api/subscription/checkout` | Get payment URL |
| POST | `/api/subscription/activate` | Activate subscription |

### Google Calendar (OAuth Ready)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/calendar/connect` | Start OAuth flow |
| POST | `/api/calendar/sync` | Sync tasks |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/sacred-math` | Get math constants |

---

## Database Schema

8 tables, automatically created on first run:

1. **users** - Account info, subscription status
2. **password_resets** - Reset tokens
3. **pets** - Virtual pet state
4. **goals** - User goals
5. **habits** - User habits
6. **daily_entries** - Daily check-ins
7. **tasks** - Calendar tasks
8. **calendar_sync** - Google Calendar integration

---

## Updating Your Existing Render Deployment

If you already have `planner-1-pyd9.onrender.com`:

```bash
# In your local planner folder:
git add .
git commit -m "Upgrade to Life Fractal v15.0 Ultimate"
git push origin main
```

Render will auto-deploy within 2-3 minutes.

---

## Testing the Deployment

1. **Health Check**: `GET /api/health`
2. **Register**: `POST /api/auth/register`
3. **Login**: `POST /api/auth/login`
4. **Dashboard**: `GET /api/dashboard` (with Bearer token)
5. **Fractal**: `GET /api/visualization/fractal`

---

## Security Features

- **JWT Authentication**: HMAC-SHA256 signed tokens
- **Password Hashing**: Werkzeug pbkdf2:sha256
- **Token Expiration**: 24 hours (configurable)
- **Reset Token Expiration**: 1 hour
- **CORS Protection**: Flask-CORS enabled
- **Input Validation**: Email, password length checks

---

## What's NOT Included (Phase 2)

- Actual Google Calendar API calls (OAuth flow ready)
- Stripe webhook for automatic subscription activation
- Email sending for password reset
- 3D visualization (can add later)
- Binaural audio generation

These are placeholder-ready but require additional API keys/services.

---

## Commit Message for Git

```
Life Fractal Intelligence v15.0 Ultimate

FIXES:
- SQLite database (8 tables, persistent storage)
- JWT HMAC-SHA256 authentication (no pyjwt dependency)
- Forgot password with token reset
- Stripe payment link integration
- Complete Nordic design frontend
- Full accessibility (autism, dysgraphia, aphantasia)

FEATURES:
- Sacred mathematics (golden ratio, Fibonacci, chaos theory)
- Wellness-mapped fractal visualization
- Virtual pet with 5 species
- Goal/habit tracking with streaks
- Spoon Theory energy management
- Google Calendar OAuth ready

DEPENDENCIES: Flask, NumPy, Pillow (7 packages total)
```

---

## Support

- **Email**: onlinediscountsllc@gmail.com
- **GoFundMe**: https://gofund.me/8d9303d27

---

**Your Life Fractal Intelligence is ready for production!** 🌀
