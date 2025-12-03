# 🌀 Life Fractal Intelligence v10.0

**Your Life • Visualized as Living Fractal Art**

A neurodivergent-focused life planning application that combines sacred mathematics, fractal geometry, and virtual pet companions. Designed for individuals with autism, ADHD, aphantasia, dysgraphia, and executive dysfunction.

![Version](https://img.shields.io/badge/version-10.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### 🎯 Goal Management
- Create and track short, medium, and long-term goals
- Visual progress tracking with percentage completion
- Priority levels and categorization

### ✅ Habit Tracking
- Daily habit completion
- Streak counting and personal records
- Frequency customization

### 📊 Daily Wellness
- Mood, energy, and stress tracking
- Sleep quality monitoring
- Journal entries with sentiment analysis
- Automatic wellness score calculation

### 🎨 Fractal Visualization
- 2D Mandelbrot fractals based on your metrics
- 3D Mandelbulb-inspired visualizations
- Sacred geometry overlays (Golden Ratio φ, Fibonacci)
- Colors that respond to your emotional state

### 🐾 Virtual Pet System
- 5 unique species: Cat, Dragon, Phoenix, Owl, Fox
- Pet mood reflects your wellness
- Feed, play, and care for your companion
- Level up system with experience points

### ♿ Accessibility First
- Text-first interface design
- Keyboard navigation support
- Screen reader friendly
- Optional visualizations (not required)
- Designed for aphantasia compatibility

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/onlinediscountsllc-web/planner.git
cd planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python life_fractal_v10.py
```

Visit `http://localhost:5000` in your browser.

### Production Deployment (Render)

1. Connect your GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn life_fractal_v10:app`
4. Deploy!

## 📁 Project Structure

```
planner/
├── life_fractal_v10.py    # Main application
├── requirements.txt       # Python dependencies
├── Procfile              # Process configuration
├── render.yaml           # Render.com settings
├── runtime.txt           # Python version
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Get current user

### Goals
- `GET /api/goals` - List all goals
- `POST /api/goals` - Create goal
- `PUT /api/goals/<id>/progress` - Update progress
- `DELETE /api/goals/<id>` - Delete goal

### Habits
- `GET /api/habits` - List habits
- `POST /api/habits` - Create habit
- `POST /api/habits/<id>/complete` - Mark complete

### Daily Check-in
- `GET /api/daily/today` - Get today's entry
- `POST /api/daily/checkin` - Submit check-in
- `GET /api/daily/history` - Get history

### Visualization
- `GET /api/visualization/fractal/2d` - 2D fractal image
- `GET /api/visualization/fractal/3d` - 3D fractal image
- `GET /api/visualization/fractal-base64/<mode>` - Base64 encoded

### Pet
- `GET /api/pet/status` - Get pet status
- `POST /api/pet/feed` - Feed pet
- `POST /api/pet/play` - Play with pet
- `POST /api/pet/rest` - Rest pet

### System
- `GET /api/health` - Health check
- `GET /api/dashboard` - Dashboard data
- `GET /api/sacred-math` - Sacred math constants

## 🧮 Sacred Mathematics

The application uses ancient mathematical principles:

- **Golden Ratio (φ):** 1.618033988749895
- **Golden Angle:** 137.5077640500378°
- **Fibonacci Sequence:** 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...
- **Platonic Solids:** Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron

## 🛠️ Technology Stack

- **Backend:** Flask 3.0, Python 3.11+
- **Database:** SQLite (self-healing)
- **Visualization:** NumPy, Pillow
- **ML:** scikit-learn (optional)
- **Server:** Gunicorn
- **Frontend:** Vanilla JS, CSS3

## 🌐 Live Demo

**Production URL:** https://planner-1-pyd9.onrender.com

## 💝 Support Development

- **GoFundMe:** https://gofund.me/8d9303d27
- **Email:** onlinediscountsllc@gmail.com

## 📜 License

MIT License - Feel free to use and modify!

---

*Built with 💜 for neurodivergent minds*
