# 🌀 LIFE FRACTAL INTELLIGENCE v8.0

## VISUALIZATION-FIRST LIFE PLANNING

This is NOT just another organizer. This is a **visual thinking tool** for neurodivergent brains that need to SEE their life to understand it.

---

## 🌟 CORE FEATURES

### 🌌 3D FRACTAL UNIVERSE (`/universe`)
The heart of the app. Your goals become glowing orbs positioned in 3D space using sacred geometry:
- **Golden Angle Positioning** - Each goal orb is placed at 137.5° intervals
- **Fibonacci Connections** - Goals connected by Fibonacci-sequence relationships
- **Sacred Geometry Overlays** - Golden spiral, Flower of Life, Icosahedron wireframes
- **Interactive Camera** - Click and drag to rotate, scroll to zoom
- **Goal Focus** - Click any goal orb to fly camera to it

### 🎨 ART THERAPY STUDIO (`/studio`)
Create beautiful, shareable art from your life data:
- **Poster Generation** - High-res printable posters with your fractal and goals
- **Desktop Wallpapers** - HD/2K/4K fractal wallpapers
- **Video Export** - Animated fractal videos (connects to HuggingFace for GPU rendering)
- **Meditation Visuals** - Calming sacred geometry animations
- **Share Links** - Unique shareable URLs for your art

### 📊 PROGRESS TIMELINE (`/timeline`)
Watch how your fractal evolves over time:
- **Fractal Snapshots** - Visual history of your life fractal
- **Wellness Trends** - Chart showing wellness score over time
- **Insights** - Goals completed, best streaks, energy averages, wellness peaks

### 📋 LIFE DASHBOARD (`/app`)
The complete life planning interface:
- **13 Life State Metrics** - Health, Skills, Finances, Relationships, Career, Mood, Energy, Purpose, Creativity, Spirituality, Belief, Focus, Gratitude
- **Spoon Energy System** - Track energy using spoon theory
- **Virtual Pet Companion** - Feed and play with your pet
- **Recommended Tasks** - Daily habits with streak tracking
- **Sacred Mathematics** - Live display of φ, golden angle, Fibonacci

---

## 🚀 DEPLOYMENT

### Render.com Deployment

1. Push to GitHub repository (onlinediscountsllc-web/planner)

2. In Render dashboard:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn life_fractal_complete_v8:app`
   - **Environment Variables**:
     - `PORT`: 10000 (or auto)
     - `SECRET_KEY`: (generate a secure key)
     - `DATABASE_PATH`: /opt/render/project/src/life_fractal.db

3. Deploy!

### Local Testing

```bash
pip install -r requirements.txt
python life_fractal_complete_v8.py
# Open http://localhost:5000
```

---

## 📐 SACRED MATHEMATICS

The app uses these sacred mathematical principles:

| Constant | Value | Usage |
|----------|-------|-------|
| φ (Golden Ratio) | 1.618033988749895 | Goal orb sizing, fractal generation |
| φ⁻¹ (Discount Factor) | 0.618033988749895 | Color cycling, opacity curves |
| Golden Angle | 137.5077640500378° | Goal orb positioning in 3D |
| Fibonacci | 1,1,2,3,5,8,13,21... | Inter-goal connections |

---

## 🧠 NEURODIVERGENT-FRIENDLY DESIGN

- **Text-First**: All visual elements have text alternatives
- **Predictable Layouts**: Consistent navigation and structure
- **Shame-Free**: No judgment for progress levels
- **Spoon Theory**: Energy tracking that understands limited capacity
- **External Visualization**: For aphantasia - see your life when you can't picture it mentally

---

## 🔗 API ENDPOINTS

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/register` | POST | Create new user |
| `/api/login` | POST | Authenticate |
| `/api/me` | GET | Current user info |
| `/api/dashboard` | GET | Full dashboard data |
| `/api/goals` | GET/POST | List/create goals |
| `/api/tasks/{id}/complete` | POST | Complete a task |
| `/api/pet/feed` | POST | Feed your pet |
| `/api/pet/play` | POST | Play with pet |
| `/api/timeline` | GET | Progress timeline data |
| `/api/art/generate` | POST | Generate art exports |
| `/share/{token}` | GET | View shared art |

---

## 💡 WHAT MAKES THIS DIFFERENT

Other organizers give you lists and checkboxes.

**Life Fractal gives you a UNIVERSE.**

Your goals aren't tasks to check off - they're **glowing orbs floating in a 3D fractal space**, connected by sacred geometry, evolving as you grow. You don't just track progress - you **watch your life transform** into living art.

For brains that need to see things to understand them, this is the difference between reading about a place and actually BEING there.

---

Built with 💜 for neurodivergent minds.
