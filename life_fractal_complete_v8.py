"""
🌀 LIFE FRACTAL INTELLIGENCE v8.0 - COMPLETE VISUALIZATION SYSTEM
═══════════════════════════════════════════════════════════════════════════════════════

This is a VISUALIZATION-FIRST life planning tool for neurodivergent brains.
The 3D fractal universe is the CORE feature - not an afterthought.

Features:
✅ 3D FRACTAL UNIVERSE - Interactive WebGL visualization with goal orbs
✅ ART THERAPY STUDIO - Generate posters, printouts, animated videos
✅ PROGRESS TIMELINE - See your fractal evolve over time
✅ SACRED GEOMETRY - Golden ratio, Fibonacci, Platonic solids
✅ VIRTUAL PET COMPANION - Differential equation emotional behavior
✅ SPOON ENERGY SYSTEM - Neurodivergent-friendly energy tracking
✅ EXPORT & SHARE - HD images, videos, shareable links

Built for brains that need to SEE things to understand them.
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import secrets
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
import base64
import sqlite3

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ML (optional)
try:
    from sklearn.tree import DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# GPU Support (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None


# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

DB_PATH = os.getenv('DATABASE_PATH', 'life_fractal.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with all tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            spoons INTEGER DEFAULT 12,
            max_spoons INTEGER DEFAULT 12,
            created_at TEXT,
            subscription_status TEXT DEFAULT 'trial',
            trial_end_date TEXT
        )
    ''')
    
    # Life state metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS life_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            health REAL DEFAULT 50,
            skills REAL DEFAULT 50,
            finances REAL DEFAULT 50,
            relationships REAL DEFAULT 50,
            career REAL DEFAULT 50,
            mood REAL DEFAULT 50,
            energy REAL DEFAULT 50,
            purpose REAL DEFAULT 50,
            creativity REAL DEFAULT 50,
            spirituality REAL DEFAULT 50,
            belief REAL DEFAULT 50,
            focus REAL DEFAULT 50,
            gratitude REAL DEFAULT 50,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Goals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'general',
            priority INTEGER DEFAULT 3,
            progress REAL DEFAULT 0,
            is_completed INTEGER DEFAULT 0,
            target_date TEXT,
            created_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Tasks (Habits)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT DEFAULT 'wellness',
            flow_percent REAL DEFAULT 1.0,
            duration_minutes INTEGER DEFAULT 10,
            spoon_cost INTEGER DEFAULT 1,
            streak INTEGER DEFAULT 0,
            last_completed TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Pet companion
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            name TEXT DEFAULT 'Buddy',
            species TEXT DEFAULT 'phoenix',
            happiness REAL DEFAULT 78,
            fed REAL DEFAULT 68,
            mood TEXT DEFAULT 'happy',
            level INTEGER DEFAULT 1,
            experience INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Fractal snapshots (for progress timeline)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fractal_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            wellness_score REAL,
            fractal_params TEXT,
            thumbnail_base64 TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Art exports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS art_exports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at TEXT,
            export_type TEXT,
            title TEXT,
            settings TEXT,
            file_path TEXT,
            share_token TEXT UNIQUE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized")


# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'life-fractal-v8-secret-key')

init_db()


# ═══════════════════════════════════════════════════════════════════════════════════════
# 2D FRACTAL GENERATOR (for static exports)
# ═══════════════════════════════════════════════════════════════════════════════════════

class FractalGenerator:
    """Generate 2D fractal images for exports"""
    
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
    
    def generate_mandelbrot(self, max_iter=256, zoom=1.0, center=(-0.5, 0),
                           wellness_influence=0.5) -> np.ndarray:
        """Generate Mandelbrot with wellness influence on colors"""
        x = np.linspace(-2.5/zoom + center[0], 2.5/zoom + center[0], self.width)
        y = np.linspace(-2.5/zoom + center[1], 2.5/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def generate_julia(self, c_real=-0.7, c_imag=0.27015, max_iter=256) -> np.ndarray:
        """Generate Julia set"""
        x = np.linspace(-2, 2, self.width)
        y = np.linspace(-2, 2, self.height)
        X, Y = np.meshgrid(x, y)
        
        z = X + 1j * Y
        c = complex(c_real, c_imag)
        iterations = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c
            iterations[mask] = i
        
        return iterations
    
    def apply_wellness_colors(self, fractal_data: np.ndarray, 
                              wellness: float, mood_hue: float = 0.6) -> Image.Image:
        """Apply colors based on wellness and mood"""
        norm = (fractal_data - fractal_data.min()) / (fractal_data.max() - fractal_data.min() + 1e-10)
        
        # Create RGB image
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Hue based on mood and wellness
        base_hue = mood_hue * wellness
        
        for i in range(self.height):
            for j in range(self.width):
                v = norm[i, j]
                if v > 0.95:  # Inside set
                    img_array[i, j] = [10, 5, 20]
                else:
                    # Smooth coloring with sacred geometry influence
                    hue = (base_hue + v * PHI_INVERSE) % 1.0
                    sat = 0.7 + 0.3 * wellness
                    val = 0.3 + 0.7 * v
                    
                    # HSV to RGB
                    r, g, b = self._hsv_to_rgb(hue, sat, val)
                    img_array[i, j] = [int(r*255), int(g*255), int(b*255)]
        
        return Image.fromarray(img_array)
    
    def _hsv_to_rgb(self, h, s, v):
        if s == 0:
            return v, v, v
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i = i % 6
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q
    
    def create_poster(self, wellness_data: dict, goals: list, 
                      title: str = "My Life Fractal") -> Image.Image:
        """Create a shareable art poster"""
        # Create larger canvas for poster
        poster_width = 1200
        poster_height = 1600
        poster = Image.new('RGB', (poster_width, poster_height), (15, 10, 30))
        draw = ImageDraw.Draw(poster)
        
        # Generate fractal
        wellness_avg = sum(wellness_data.values()) / len(wellness_data) / 100 if wellness_data else 0.5
        fractal = self.generate_mandelbrot(max_iter=200, wellness_influence=wellness_avg)
        fractal_img = self.apply_wellness_colors(fractal, wellness_avg)
        fractal_img = fractal_img.resize((1000, 1000), Image.Resampling.LANCZOS)
        
        # Paste fractal centered
        poster.paste(fractal_img, (100, 300))
        
        # Add title
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Draw title
        draw.text((poster_width//2, 100), title, fill=(240, 196, 32), 
                  font=title_font, anchor="mm")
        
        # Draw subtitle with sacred math
        subtitle = f"φ = {PHI:.6f} | Wellness: {wellness_avg*100:.0f}%"
        draw.text((poster_width//2, 160), subtitle, fill=(150, 150, 180),
                  font=text_font, anchor="mm")
        
        # Draw goal summary at bottom
        if goals:
            completed = sum(1 for g in goals if g.get('is_completed'))
            goal_text = f"{completed}/{len(goals)} Goals Achieved"
            draw.text((poster_width//2, 1400), goal_text, fill=(100, 200, 120),
                      font=text_font, anchor="mm")
        
        # Add date
        draw.text((poster_width//2, 1500), datetime.now().strftime("%B %d, %Y"),
                  fill=(100, 100, 120), font=text_font, anchor="mm")
        
        return poster
    
    def to_base64(self, img: Image.Image, format='PNG') -> str:
        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


fractal_gen = FractalGenerator()


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_user_from_token(token):
    """Verify auth token and return user_id"""
    if not token:
        return None
    try:
        # Simple token format: user_id:hash
        parts = token.split(':')
        if len(parts) != 2:
            return None
        user_id = parts[0]
        # Verify user exists
        conn = get_db()
        user = conn.execute('SELECT id FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        return user_id if user else None
    except:
        return None


def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user_id = get_user_from_token(token)
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(user_id, *args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN PAGES - HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════════════

MAIN_CSS = '''
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg-dark: #0a0a1a;
    --bg-card: #12122a;
    --accent-purple: #667eea;
    --accent-violet: #764ba2;
    --accent-gold: #f0c420;
    --accent-green: #48c774;
    --accent-coral: #ff6b6b;
    --text-light: #e8e8f0;
    --text-dim: #8888aa;
}
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg-dark);
    color: var(--text-light);
    min-height: 100vh;
}
.gradient-bg {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-violet));
}
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 30px;
    background: rgba(10, 10, 26, 0.95);
    border-bottom: 1px solid rgba(102, 126, 234, 0.3);
    position: sticky;
    top: 0;
    z-index: 1000;
}
.nav-logo {
    font-size: 1.4em;
    font-weight: bold;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.nav-links { display: flex; gap: 25px; }
.nav-links a {
    color: var(--text-dim);
    text-decoration: none;
    transition: color 0.3s;
    font-size: 0.95em;
}
.nav-links a:hover, .nav-links a.active { color: var(--accent-gold); }
.nav-right { display: flex; align-items: center; gap: 20px; }
.spoon-badge {
    background: linear-gradient(135deg, var(--accent-gold), #e8b400);
    color: #1a1a2e;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
}
.btn {
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s;
    text-decoration: none;
    display: inline-block;
}
.btn-primary { background: linear-gradient(135deg, var(--accent-purple), var(--accent-violet)); color: white; }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
.btn-secondary { background: transparent; border: 2px solid var(--accent-purple); color: var(--accent-purple); }
.btn-gold { background: linear-gradient(135deg, var(--accent-gold), #e8b400); color: #1a1a2e; }
.container { max-width: 1400px; margin: 0 auto; padding: 30px; }
.grid { display: grid; gap: 25px; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(102, 126, 234, 0.2);
}
.card-title {
    font-size: 1.2em;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.metric-row {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 12px;
}
.metric-label { min-width: 120px; color: var(--text-dim); }
.metric-bar {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.metric-value { min-width: 50px; text-align: right; font-weight: 600; }
@media (max-width: 900px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    .nav { flex-wrap: wrap; gap: 15px; }
}
'''

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence</title>
    <style>
''' + MAIN_CSS + '''
        .hero-section {
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
        }
        .hero-title { font-size: 3em; margin-bottom: 15px; }
        .hero-subtitle { color: var(--text-dim); font-size: 1.2em; margin-bottom: 30px; }
        .feature-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-top: 40px; }
        .feature-card {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 35px;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s;
            cursor: pointer;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--accent-gold);
            box-shadow: 0 10px 40px rgba(240, 196, 32, 0.15);
        }
        .feature-icon { font-size: 3em; margin-bottom: 20px; }
        .feature-title { font-size: 1.4em; margin-bottom: 10px; }
        .feature-desc { color: var(--text-dim); line-height: 1.6; }
        .login-panel {
            max-width: 400px;
            margin: 40px auto;
            background: var(--bg-card);
            border-radius: 20px;
            padding: 40px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        .login-panel input {
            width: 100%;
            padding: 14px;
            margin-bottom: 15px;
            border-radius: 10px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            background: rgba(255,255,255,0.05);
            color: white;
            font-size: 1em;
        }
        .login-panel input::placeholder { color: var(--text-dim); }
        .login-panel .btn { width: 100%; margin-top: 10px; padding: 14px; }
        .tab-switch {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
        }
        .tab-btn {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: 2px solid var(--accent-purple);
            border-radius: 10px;
            color: var(--accent-purple);
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab-btn.active {
            background: var(--accent-purple);
            color: white;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">🌀 Life Fractal Intelligence</div>
        <div class="nav-links">
            <a href="/" class="active">Home</a>
            <a href="/universe">3D Universe</a>
            <a href="/studio">Art Studio</a>
            <a href="/timeline">Timeline</a>
        </div>
        <div class="nav-right" id="navRight">
            <!-- Dynamic content based on auth -->
        </div>
    </nav>

    <div id="heroSection" class="hero-section">
        <h1 class="hero-title">🌀 Your Life, Visualized</h1>
        <p class="hero-subtitle">Transform your goals and progress into living fractal art.<br>Built for brains that need to SEE things to understand them.</p>
        
        <div class="feature-cards container">
            <div class="feature-card" onclick="location.href='/universe'">
                <div class="feature-icon">🌌</div>
                <div class="feature-title">3D Fractal Universe</div>
                <div class="feature-desc">Explore your life as an interactive 3D world. Goals become glowing orbs positioned by sacred geometry.</div>
            </div>
            <div class="feature-card" onclick="location.href='/studio'">
                <div class="feature-icon">🎨</div>
                <div class="feature-title">Art Therapy Studio</div>
                <div class="feature-desc">Create beautiful posters, printable art, and animated videos of your life fractal to share.</div>
            </div>
            <div class="feature-card" onclick="location.href='/timeline'">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Progress Timeline</div>
                <div class="feature-desc">Watch your fractal evolve over time. See how your life patterns shift and grow.</div>
            </div>
            <div class="feature-card" onclick="location.href='/app'">
                <div class="feature-icon">📋</div>
                <div class="feature-title">Life Dashboard</div>
                <div class="feature-desc">Track wellness, habits, goals, and your virtual pet companion with spoon-based energy.</div>
            </div>
        </div>
    </div>

    <div id="loginPanel" class="login-panel" style="display: none;">
        <div class="tab-switch">
            <button class="tab-btn active" onclick="showTab('login')">Login</button>
            <button class="tab-btn" onclick="showTab('register')">Register</button>
        </div>
        <form id="loginForm">
            <input type="email" id="loginEmail" placeholder="Email" required>
            <input type="password" id="loginPassword" placeholder="Password" required>
            <button type="submit" class="btn btn-primary">Enter Your Universe</button>
        </form>
        <form id="registerForm" style="display: none;">
            <input type="text" id="regFirstName" placeholder="First Name" required>
            <input type="text" id="regLastName" placeholder="Last Name" required>
            <input type="email" id="regEmail" placeholder="Email" required>
            <input type="password" id="regPassword" placeholder="Password" required>
            <button type="submit" class="btn btn-gold">Create Your Universe</button>
        </form>
        <p id="authError" style="color: var(--accent-coral); margin-top: 15px; text-align: center;"></p>
    </div>

    <script>
        const API = '';
        let authToken = localStorage.getItem('fractal_token');
        let currentUser = null;

        async function checkAuth() {
            if (!authToken) {
                showLoginPanel();
                return;
            }
            try {
                const res = await fetch(API + '/api/me', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                if (res.ok) {
                    currentUser = await res.json();
                    showAuthenticatedUI();
                } else {
                    localStorage.removeItem('fractal_token');
                    showLoginPanel();
                }
            } catch (e) {
                showLoginPanel();
            }
        }

        function showLoginPanel() {
            document.getElementById('heroSection').style.display = 'none';
            document.getElementById('loginPanel').style.display = 'block';
            document.getElementById('navRight').innerHTML = '';
        }

        function showAuthenticatedUI() {
            document.getElementById('heroSection').style.display = 'block';
            document.getElementById('loginPanel').style.display = 'none';
            document.getElementById('navRight').innerHTML = `
                <span class="spoon-badge">🥄 ${currentUser.spoons || 12} spoons</span>
                <span>Hi, ${currentUser.first_name || 'Friend'}</span>
                <button class="btn btn-secondary" onclick="logout()">Logout</button>
            `;
        }

        function showTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('loginForm').style.display = tab === 'login' ? 'block' : 'none';
            document.getElementById('registerForm').style.display = tab === 'register' ? 'block' : 'none';
        }

        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const res = await fetch(API + '/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: document.getElementById('loginEmail').value,
                        password: document.getElementById('loginPassword').value
                    })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    authToken = data.token;
                    localStorage.setItem('fractal_token', authToken);
                    checkAuth();
                } else {
                    document.getElementById('authError').textContent = data.error || 'Login failed';
                }
            } catch (e) {
                document.getElementById('authError').textContent = 'Connection error';
            }
        });

        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const res = await fetch(API + '/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: document.getElementById('regEmail').value,
                        password: document.getElementById('regPassword').value,
                        first_name: document.getElementById('regFirstName').value,
                        last_name: document.getElementById('regLastName').value
                    })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    authToken = data.token;
                    localStorage.setItem('fractal_token', authToken);
                    checkAuth();
                } else {
                    document.getElementById('authError').textContent = data.error || 'Registration failed';
                }
            } catch (e) {
                document.getElementById('authError').textContent = 'Connection error';
            }
        });

        function logout() {
            localStorage.removeItem('fractal_token');
            authToken = null;
            currentUser = null;
            showLoginPanel();
        }

        checkAuth();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3D FRACTAL UNIVERSE - THE MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

UNIVERSE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌌 3D Fractal Universe - Life Fractal Intelligence</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #000; 
            color: #fff; 
            overflow: hidden;
        }
        #canvas-container { width: 100vw; height: 100vh; position: relative; }
        
        /* Control Panel */
        #control-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(10, 10, 30, 0.92);
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #667eea;
            max-width: 320px;
            z-index: 1000;
            backdrop-filter: blur(15px);
        }
        #control-panel h1 {
            font-size: 1.4em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #667eea, #f0c420);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .control-section {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .control-section:last-child { border-bottom: none; }
        .control-section h3 {
            font-size: 0.9em;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            margin-bottom: 10px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-gold {
            background: linear-gradient(135deg, #f0c420, #e8b400);
            color: #1a1a2e;
        }
        .btn-outline {
            background: transparent;
            border: 2px solid #667eea;
            color: #667eea;
        }
        .btn:hover { transform: scale(1.02); }
        
        .slider-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        .slider-row label {
            min-width: 100px;
            font-size: 0.85em;
            color: #aaa;
        }
        .slider-row input[type="range"] {
            flex: 1;
            accent-color: #667eea;
        }
        .slider-row .value {
            min-width: 45px;
            text-align: right;
            font-size: 0.9em;
            color: #f0c420;
        }
        
        .sacred-math {
            font-size: 0.85em;
            color: #888;
            line-height: 1.8;
        }
        .sacred-math span { color: #f0c420; font-family: monospace; }
        
        /* Goals Panel */
        #goals-panel {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(10, 10, 30, 0.92);
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #48c774;
            max-width: 380px;
            max-height: 70vh;
            overflow-y: auto;
            z-index: 1000;
        }
        #goals-panel h2 {
            font-size: 1.2em;
            margin-bottom: 20px;
            color: #48c774;
        }
        .goal-item {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 4px solid #667eea;
            cursor: pointer;
            transition: all 0.3s;
        }
        .goal-item:hover {
            background: rgba(102, 126, 234, 0.15);
            transform: translateX(5px);
        }
        .goal-item.selected {
            border-left-color: #f0c420;
            background: rgba(240, 196, 32, 0.1);
        }
        .goal-title { font-weight: 600; margin-bottom: 8px; }
        .goal-meta { font-size: 0.85em; color: #888; margin-bottom: 8px; }
        .goal-progress {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        .goal-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #48c774);
            transition: width 0.5s;
        }
        
        /* Stats Bar */
        #stats-bar {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 10, 30, 0.92);
            padding: 15px 40px;
            border-radius: 30px;
            border: 2px solid #f0c420;
            display: flex;
            gap: 50px;
            z-index: 1000;
        }
        .stat { text-align: center; }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #f0c420;
        }
        .stat-label { font-size: 0.8em; color: #888; }
        
        /* Toggle Button */
        #toggle-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(10, 10, 30, 0.8);
            border: 2px solid #667eea;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            z-index: 1001;
            display: none;
        }
        
        /* Loading */
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 2000;
        }
        .spinner {
            width: 80px;
            height: 80px;
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Tooltip */
        #tooltip {
            position: absolute;
            background: rgba(10, 10, 30, 0.95);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #667eea;
            pointer-events: none;
            display: none;
            z-index: 2000;
            max-width: 280px;
        }
        #tooltip h4 { color: #667eea; margin-bottom: 10px; }
        #tooltip .progress-text { color: #f0c420; font-size: 1.2em; }
        
        /* Back Button */
        #back-link {
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div style="font-size: 1.2em;">Generating Your Fractal Universe...</div>
        <div style="color: #888; margin-top: 10px;">Positioning goals by golden angle</div>
    </div>
    
    <div id="canvas-container"></div>
    
    <div id="control-panel">
        <h1>🌀 Fractal Universe</h1>
        
        <div class="control-section">
            <h3>🎛️ Controls</h3>
            <button class="btn btn-primary" onclick="regenerateFractal()">
                🔄 Regenerate
            </button>
            <button class="btn btn-gold" onclick="addGoalOrb()">
                ✨ Add Goal Orb
            </button>
            <button class="btn btn-outline" onclick="toggleSacredGeometry()">
                📐 Toggle Sacred Geometry
            </button>
            <button class="btn btn-outline" onclick="toggleAutoRotate()">
                🔁 Auto-Rotate
            </button>
        </div>
        
        <div class="control-section">
            <h3>🎚️ Parameters</h3>
            <div class="slider-row">
                <label>Mood Influence</label>
                <input type="range" id="moodSlider" min="0" max="100" value="50" oninput="updateParam('mood', this.value)">
                <span class="value" id="moodValue">50</span>
            </div>
            <div class="slider-row">
                <label>Energy Flow</label>
                <input type="range" id="energySlider" min="0" max="100" value="50" oninput="updateParam('energy', this.value)">
                <span class="value" id="energyValue">50</span>
            </div>
            <div class="slider-row">
                <label>Complexity</label>
                <input type="range" id="complexitySlider" min="500" max="5000" value="2000" oninput="updateParam('complexity', this.value)">
                <span class="value" id="complexityValue">2000</span>
            </div>
        </div>
        
        <div class="control-section">
            <h3>📐 Sacred Mathematics</h3>
            <div class="sacred-math">
                φ (Golden Ratio) = <span id="phi">1.618034</span><br>
                Golden Angle = <span id="goldenAngle">137.51°</span><br>
                Fibonacci: <span>1, 1, 2, 3, 5, 8, 13...</span><br>
                Frame: <span id="frameCount">0</span>
            </div>
        </div>
        
        <div class="control-section">
            <button class="btn btn-outline" onclick="resetCamera()">🎯 Reset Camera</button>
        </div>
    </div>
    
    <div id="goals-panel">
        <h2>🎯 Goals in Space</h2>
        <div id="goals-list">
            <div style="color: #888;">Loading goals...</div>
        </div>
    </div>
    
    <div id="stats-bar">
        <div class="stat">
            <div class="stat-value" id="totalGoals">0</div>
            <div class="stat-label">Total Goals</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="completedGoals">0</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="avgProgress">0%</div>
            <div class="stat-label">Progress</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="wellnessScore">--</div>
            <div class="stat-label">Wellness</div>
        </div>
    </div>
    
    <a id="back-link" href="/" class="btn btn-outline">← Dashboard</a>
    
    <div id="tooltip"></div>
    
    <script>
        // Sacred Mathematics Constants
        const PHI = (1 + Math.sqrt(5)) / 2;
        const PHI_INV = PHI - 1;
        const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
        const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        
        // Three.js globals
        let scene, camera, renderer;
        let fractalParticles, goalOrbs = [], connectionLines = [], sacredGeometry = [];
        let raycaster, mouse;
        let autoRotate = true;
        let showSacred = true;
        let frameCount = 0;
        let animationSpeed = 1.0;
        let userData = null;
        let selectedGoal = null;
        
        // Camera controls
        let isDragging = false;
        let previousMouse = { x: 0, y: 0 };
        let cameraDistance = 20;
        let cameraTheta = 0;
        let cameraPhi = Math.PI / 4;
        
        const authToken = localStorage.getItem('fractal_token');
        
        function init() {
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050510);
            scene.fog = new THREE.FogExp2(0x050510, 0.012);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 8, 20);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            document.getElementById('canvas-container').appendChild(renderer.domElement);
            
            // Raycaster for interaction
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
            scene.add(ambientLight);
            
            const purpleLight = new THREE.PointLight(0x667eea, 2.5, 100);
            purpleLight.position.set(15, 15, 15);
            scene.add(purpleLight);
            
            const violetLight = new THREE.PointLight(0x764ba2, 2.5, 100);
            violetLight.position.set(-15, -10, 15);
            scene.add(violetLight);
            
            const goldLight = new THREE.PointLight(0xf0c420, 2, 80);
            goldLight.position.set(0, 20, -10);
            scene.add(goldLight);
            
            // Create fractal
            createFractalParticles();
            createSacredGeometry();
            createStarfield();
            
            // Event listeners
            window.addEventListener('resize', onWindowResize);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            renderer.domElement.addEventListener('click', onClick);
            setupCameraControls();
            
            // Load user data
            loadUserData();
            
            // Start animation
            animate();
            
            // Hide loading
            setTimeout(() => {
                document.getElementById('loading').style.display = 'none';
            }, 2000);
        }
        
        function createFractalParticles() {
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            const complexity = parseInt(document.getElementById('complexitySlider').value) || 2000;
            
            // Create 3D Mandelbulb-like structure
            const power = 8;
            const iterations = 6;
            const scale = 4;
            const resolution = Math.sqrt(complexity);
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const theta = (i / resolution) * Math.PI;
                    const phi_angle = (j / resolution) * Math.PI * 2;
                    
                    let x = Math.sin(theta) * Math.cos(phi_angle);
                    let y = Math.sin(theta) * Math.sin(phi_angle);
                    let z = Math.cos(theta);
                    
                    // Mandelbulb iteration
                    let r = 1;
                    for (let iter = 0; iter < iterations; iter++) {
                        r = Math.sqrt(x*x + y*y + z*z);
                        if (r > 2) break;
                        let theta_p = Math.acos(z / r);
                        let phi_p = Math.atan2(y, x);
                        let zr = Math.pow(r, power);
                        theta_p *= power;
                        phi_p *= power;
                        x = zr * Math.sin(theta_p) * Math.cos(phi_p);
                        y = zr * Math.sin(theta_p) * Math.sin(phi_p);
                        z = zr * Math.cos(theta_p);
                    }
                    
                    // Apply golden ratio influence
                    const finalR = scale * (1 + 0.4 * Math.sin(theta * PHI * 3) * Math.cos(phi_angle * PHI * 2));
                    vertices.push(
                        finalR * Math.sin(theta) * Math.cos(phi_angle),
                        finalR * Math.cos(theta),
                        finalR * Math.sin(theta) * Math.sin(phi_angle)
                    );
                    
                    // Color based on position and PHI
                    const hue = (i / resolution * PHI_INV + j / resolution) % 1;
                    const color = new THREE.Color().setHSL(hue * 0.4 + 0.55, 0.85, 0.55 + 0.25 * Math.sin(theta * 4));
                    colors.push(color.r, color.g, color.b);
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({
                size: 0.06,
                vertexColors: true,
                transparent: true,
                opacity: 0.85,
                blending: THREE.AdditiveBlending
            });
            
            if (fractalParticles) scene.remove(fractalParticles);
            fractalParticles = new THREE.Points(geometry, material);
            scene.add(fractalParticles);
        }
        
        function createSacredGeometry() {
            // Golden Spiral
            const spiralGeometry = new THREE.BufferGeometry();
            const spiralVertices = [];
            for (let i = 0; i < 800; i++) {
                const angle = i * 0.1;
                const r = 0.15 * Math.pow(PHI, 2 * angle / Math.PI);
                if (r > 10) break;
                spiralVertices.push(
                    r * Math.cos(angle),
                    r * Math.sin(angle) * 0.4,
                    r * Math.sin(angle)
                );
            }
            spiralGeometry.setAttribute('position', new THREE.Float32BufferAttribute(spiralVertices, 3));
            const spiral = new THREE.Line(spiralGeometry, new THREE.LineBasicMaterial({
                color: 0xf0c420,
                transparent: true,
                opacity: 0.5
            }));
            spiral.name = 'golden-spiral';
            scene.add(spiral);
            sacredGeometry.push(spiral);
            
            // Flower of Life circles
            const flowerGroup = new THREE.Group();
            flowerGroup.name = 'flower-of-life';
            const circleGeometry = new THREE.CircleGeometry(2.5, 64);
            const edges = new THREE.EdgesGeometry(circleGeometry);
            const circleMaterial = new THREE.LineBasicMaterial({
                color: 0x667eea,
                transparent: true,
                opacity: 0.25
            });
            
            // Central circle + 6 surrounding
            const positions = [[0, 0]];
            for (let i = 0; i < 6; i++) {
                positions.push([2.5 * Math.cos(i * Math.PI / 3), 2.5 * Math.sin(i * Math.PI / 3)]);
            }
            positions.forEach(([x, z]) => {
                const circle = new THREE.LineSegments(edges.clone(), circleMaterial);
                circle.position.set(x, 0, z);
                circle.rotation.x = Math.PI / 2;
                flowerGroup.add(circle);
            });
            scene.add(flowerGroup);
            sacredGeometry.push(flowerGroup);
            
            // Icosahedron wireframe
            const icoGeometry = new THREE.IcosahedronGeometry(6, 0);
            const icoEdges = new THREE.EdgesGeometry(icoGeometry);
            const icosahedron = new THREE.LineSegments(icoEdges, new THREE.LineBasicMaterial({
                color: 0x764ba2,
                transparent: true,
                opacity: 0.2
            }));
            icosahedron.name = 'icosahedron';
            scene.add(icosahedron);
            sacredGeometry.push(icosahedron);
        }
        
        function createStarfield() {
            const starsGeometry = new THREE.BufferGeometry();
            const starVertices = [];
            const starColors = [];
            
            for (let i = 0; i < 4000; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 60 + Math.random() * 60;
                
                starVertices.push(
                    r * Math.sin(phi) * Math.cos(theta),
                    r * Math.sin(phi) * Math.sin(theta),
                    r * Math.cos(phi)
                );
                
                const brightness = 0.3 + Math.random() * 0.7;
                const tint = Math.random();
                if (tint < 0.1) {
                    starColors.push(brightness * 0.8, brightness * 0.8, brightness);
                } else if (tint < 0.2) {
                    starColors.push(brightness, brightness * 0.9, brightness * 0.7);
                } else {
                    starColors.push(brightness, brightness, brightness);
                }
            }
            
            starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
            starsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));
            
            scene.add(new THREE.Points(starsGeometry, new THREE.PointsMaterial({
                size: 0.25,
                vertexColors: true,
                transparent: true,
                opacity: 0.9
            })));
        }
        
        function createGoalOrbs(goals) {
            // Remove existing orbs
            goalOrbs.forEach(orb => scene.remove(orb));
            goalOrbs = [];
            connectionLines.forEach(line => scene.remove(line));
            connectionLines = [];
            
            if (!goals || goals.length === 0) return;
            
            goals.forEach((goal, index) => {
                // Position using GOLDEN ANGLE - this is the key sacred geometry positioning
                const goldenAngle = index * GOLDEN_ANGLE;
                const radius = 5 + (goal.priority || 3) * 0.6;
                const heightFactor = (goal.progress / 100);
                const height = heightFactor * 5 - 2.5 + Math.sin(index * PHI) * 2;
                
                const x = radius * Math.cos(goldenAngle);
                const z = radius * Math.sin(goldenAngle);
                const y = height;
                
                // Size based on progress
                const size = 0.35 + (goal.progress / 100) * 0.5;
                const geometry = new THREE.SphereGeometry(size, 32, 32);
                
                // Color based on status
                let color;
                if (goal.is_completed || goal.progress >= 100) {
                    color = new THREE.Color(0x48c774); // Green - completed
                } else if (goal.progress >= 70) {
                    color = new THREE.Color(0x3298dc); // Blue - almost there
                } else if (goal.progress >= 40) {
                    color = new THREE.Color(0xf0c420); // Gold - in progress
                } else {
                    color = new THREE.Color(0xff6b6b); // Coral - just started
                }
                
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.4,
                    transparent: true,
                    opacity: 0.92
                });
                
                const orb = new THREE.Mesh(geometry, material);
                orb.position.set(x, y, z);
                orb.userData = { goal: goal, index: index, baseY: y };
                
                // Glow effect
                const glowGeometry = new THREE.SphereGeometry(size * 1.8, 16, 16);
                const glowMaterial = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.12
                });
                orb.add(new THREE.Mesh(glowGeometry, glowMaterial));
                
                scene.add(orb);
                goalOrbs.push(orb);
                
                // Connection line to center
                const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    orb.position
                ]);
                const line = new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.25
                }));
                scene.add(line);
                connectionLines.push(line);
            });
            
            // Create Fibonacci connections between orbs
            for (let i = 0; i < goalOrbs.length; i++) {
                for (let j = i + 1; j < goalOrbs.length; j++) {
                    const diff = j - i;
                    if (FIBONACCI.includes(diff)) {
                        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                            goalOrbs[i].position,
                            goalOrbs[j].position
                        ]);
                        const line = new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({
                            color: 0x667eea,
                            transparent: true,
                            opacity: 0.15
                        }));
                        scene.add(line);
                        connectionLines.push(line);
                    }
                }
            }
        }
        
        function updateGoalsPanel(goals) {
            const container = document.getElementById('goals-list');
            if (!goals || goals.length === 0) {
                container.innerHTML = '<div style="color: #888;">No goals yet. Add some to see them in 3D space!</div>';
                return;
            }
            
            let html = '';
            goals.forEach((goal, index) => {
                const progressColor = goal.progress >= 70 ? '#48c774' : (goal.progress >= 40 ? '#f0c420' : '#ff6b6b');
                const icon = goal.is_completed ? '✅' : '🎯';
                html += `
                    <div class="goal-item" data-index="${index}" onclick="focusGoal(${index})">
                        <div class="goal-title">${icon} ${goal.title}</div>
                        <div class="goal-meta">Priority: ${goal.priority || 3} | ${goal.category || 'general'}</div>
                        <div class="goal-progress">
                            <div class="goal-progress-fill" style="width: ${goal.progress}%; background: ${progressColor};"></div>
                        </div>
                        <div style="font-size: 0.8em; color: #888; margin-top: 5px;">${goal.progress?.toFixed(1) || 0}%</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
        
        function focusGoal(index) {
            // Highlight in panel
            document.querySelectorAll('.goal-item').forEach(el => el.classList.remove('selected'));
            document.querySelector(`.goal-item[data-index="${index}"]`)?.classList.add('selected');
            
            // Animate camera to goal
            if (goalOrbs[index]) {
                selectedGoal = goalOrbs[index];
                const targetPos = goalOrbs[index].position.clone();
                const cameraTarget = targetPos.clone().add(new THREE.Vector3(4, 3, 4));
                
                const startPos = camera.position.clone();
                let progress = 0;
                
                function animateCamera() {
                    progress += 0.025;
                    if (progress < 1) {
                        camera.position.lerpVectors(startPos, cameraTarget, progress);
                        camera.lookAt(targetPos);
                        requestAnimationFrame(animateCamera);
                    }
                }
                animateCamera();
            }
        }
        
        function setupCameraControls() {
            const canvas = renderer.domElement;
            
            canvas.addEventListener('mousedown', (e) => {
                if (e.button === 0) {
                    isDragging = true;
                    previousMouse = { x: e.clientX, y: e.clientY };
                }
            });
            
            canvas.addEventListener('mouseup', () => isDragging = false);
            canvas.addEventListener('mouseleave', () => isDragging = false);
            
            canvas.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                cameraTheta -= (e.clientX - previousMouse.x) * 0.005;
                cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, 
                    cameraPhi + (e.clientY - previousMouse.y) * 0.005));
                
                previousMouse = { x: e.clientX, y: e.clientY };
                autoRotate = false;
            });
            
            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                cameraDistance = Math.max(8, Math.min(60, cameraDistance + e.deltaY * 0.03));
            });
        }
        
        async function loadUserData() {
            if (!authToken) {
                document.getElementById('goals-list').innerHTML = '<div style="color: #ff6b6b;">Please log in to see your goals</div>';
                return;
            }
            
            try {
                const res = await fetch('/api/dashboard', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                
                if (!res.ok) throw new Error('API error');
                
                userData = await res.json();
                
                // Update stats
                const goals = userData.goals || [];
                document.getElementById('totalGoals').textContent = goals.length;
                document.getElementById('completedGoals').textContent = goals.filter(g => g.is_completed || g.progress >= 100).length;
                
                const avgProgress = goals.length ? 
                    (goals.reduce((sum, g) => sum + (g.progress || 0), 0) / goals.length).toFixed(0) : 0;
                document.getElementById('avgProgress').textContent = avgProgress + '%';
                
                const wellness = userData.life_state ? 
                    Object.values(userData.life_state).reduce((a, b) => a + b, 0) / Object.keys(userData.life_state).length : 50;
                document.getElementById('wellnessScore').textContent = wellness.toFixed(0);
                
                // Create goal orbs in 3D
                createGoalOrbs(goals);
                updateGoalsPanel(goals);
                
            } catch (e) {
                console.error('Error loading data:', e);
                document.getElementById('goals-list').innerHTML = '<div style="color: #ff6b6b;">Error loading data</div>';
            }
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        function onMouseMove(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }
        
        function onClick(event) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(goalOrbs);
            
            if (intersects.length > 0) {
                const orb = intersects[0].object;
                const goal = orb.userData.goal;
                focusGoal(orb.userData.index);
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            document.getElementById('frameCount').textContent = frameCount;
            
            // Auto-rotate camera
            if (autoRotate) {
                cameraTheta += 0.002 * animationSpeed;
            }
            
            // Update camera position
            camera.position.x = cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);
            camera.position.y = cameraDistance * Math.cos(cameraPhi);
            camera.position.z = cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            camera.lookAt(0, 0, 0);
            
            // Animate fractal
            if (fractalParticles) {
                fractalParticles.rotation.y += 0.001 * animationSpeed;
                fractalParticles.rotation.x = Math.sin(frameCount * 0.001) * 0.1;
            }
            
            // Animate goal orbs (gentle floating)
            goalOrbs.forEach((orb, i) => {
                const baseY = orb.userData.baseY;
                orb.position.y = baseY + Math.sin(frameCount * 0.02 + i * PHI) * 0.15;
                orb.rotation.y += 0.01;
            });
            
            // Animate sacred geometry
            sacredGeometry.forEach((geo, i) => {
                if (geo.name === 'golden-spiral') {
                    geo.rotation.y += 0.003;
                } else if (geo.name === 'icosahedron') {
                    geo.rotation.x += 0.001;
                    geo.rotation.y += 0.002;
                }
            });
            
            renderer.render(scene, camera);
        }
        
        // Control functions
        function regenerateFractal() {
            createFractalParticles();
        }
        
        function addGoalOrb() {
            const title = prompt('Goal title:');
            if (!title) return;
            
            // Add goal via API
            fetch('/api/goals', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken
                },
                body: JSON.stringify({ title: title, priority: 3, progress: 0 })
            }).then(() => loadUserData());
        }
        
        function toggleSacredGeometry() {
            showSacred = !showSacred;
            sacredGeometry.forEach(geo => {
                geo.visible = showSacred;
            });
        }
        
        function toggleAutoRotate() {
            autoRotate = !autoRotate;
        }
        
        function updateParam(param, value) {
            document.getElementById(param + 'Value').textContent = value;
            if (param === 'complexity') {
                createFractalParticles();
            }
        }
        
        function resetCamera() {
            cameraDistance = 20;
            cameraTheta = 0;
            cameraPhi = Math.PI / 4;
            autoRotate = true;
        }
        
        // Initialize
        document.getElementById('phi').textContent = PHI.toFixed(6);
        document.getElementById('goldenAngle').textContent = (GOLDEN_ANGLE * 180 / Math.PI).toFixed(2) + '°';
        
        init();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# ART THERAPY STUDIO
# ═══════════════════════════════════════════════════════════════════════════════════════

STUDIO_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 Art Therapy Studio - Life Fractal Intelligence</title>
    <style>
''' + MAIN_CSS + '''
        .studio-header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.15) 0%, transparent 100%);
        }
        .studio-header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .studio-header p { color: var(--text-dim); font-size: 1.1em; }
        
        .export-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }
        .export-card {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s;
        }
        .export-card:hover {
            border-color: var(--accent-gold);
            transform: translateY(-5px);
        }
        .export-icon { font-size: 4em; margin-bottom: 20px; }
        .export-title { font-size: 1.4em; margin-bottom: 10px; }
        .export-desc { color: var(--text-dim); margin-bottom: 20px; line-height: 1.6; }
        
        .preview-section {
            margin-top: 50px;
            padding: 40px;
            background: var(--bg-card);
            border-radius: 20px;
            text-align: center;
        }
        .preview-section h2 { margin-bottom: 30px; }
        #previewCanvas {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }
        
        .settings-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
            padding: 25px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
        }
        .setting-group label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-dim);
        }
        .setting-group select, .setting-group input {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            background: rgba(255,255,255,0.05);
            color: white;
        }
        
        .share-section {
            margin-top: 40px;
            padding: 30px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 20px;
            text-align: center;
        }
        .share-link {
            display: flex;
            gap: 10px;
            max-width: 500px;
            margin: 20px auto 0;
        }
        .share-link input {
            flex: 1;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid var(--accent-purple);
            background: rgba(0,0,0,0.3);
            color: white;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">🌀 Life Fractal Intelligence</div>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/universe">3D Universe</a>
            <a href="/studio" class="active">Art Studio</a>
            <a href="/timeline">Timeline</a>
        </div>
        <div class="nav-right">
            <a href="/app" class="btn btn-secondary">Dashboard</a>
        </div>
    </nav>
    
    <div class="studio-header">
        <h1>🎨 Art Therapy Studio</h1>
        <p>Transform your life fractal into beautiful, shareable art</p>
    </div>
    
    <div class="container">
        <div class="export-options">
            <div class="export-card" onclick="generatePoster()">
                <div class="export-icon">🖼️</div>
                <div class="export-title">Poster Print</div>
                <div class="export-desc">High-resolution poster perfect for printing and framing. Includes your wellness data and goals.</div>
                <button class="btn btn-primary">Generate Poster</button>
            </div>
            
            <div class="export-card" onclick="generateWallpaper()">
                <div class="export-icon">🖥️</div>
                <div class="export-title">Desktop Wallpaper</div>
                <div class="export-desc">Beautiful fractal wallpaper for your computer. Available in multiple resolutions.</div>
                <button class="btn btn-primary">Generate Wallpaper</button>
            </div>
            
            <div class="export-card" onclick="generateVideo()">
                <div class="export-icon">🎬</div>
                <div class="export-title">Animated Video</div>
                <div class="export-desc">Watch your fractal evolve in a mesmerizing animation. Perfect for meditation or sharing.</div>
                <button class="btn btn-gold">Generate Video</button>
            </div>
            
            <div class="export-card" onclick="generateMeditation()">
                <div class="export-icon">🧘</div>
                <div class="export-title">Meditation Visual</div>
                <div class="export-desc">Calming fractal animation with sacred geometry overlays for mindfulness practice.</div>
                <button class="btn btn-primary">Generate</button>
            </div>
        </div>
        
        <div class="preview-section">
            <h2>✨ Preview</h2>
            
            <div class="settings-panel">
                <div class="setting-group">
                    <label>Fractal Type</label>
                    <select id="fractalType">
                        <option value="mandelbrot">Mandelbrot Set</option>
                        <option value="julia">Julia Set</option>
                        <option value="burning_ship">Burning Ship</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Color Theme</label>
                    <select id="colorTheme">
                        <option value="cosmic">Cosmic Purple</option>
                        <option value="ocean">Ocean Blue</option>
                        <option value="forest">Forest Green</option>
                        <option value="sunset">Sunset Gold</option>
                        <option value="custom">From Wellness</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Resolution</label>
                    <select id="resolution">
                        <option value="1080">1920x1080 (HD)</option>
                        <option value="1440">2560x1440 (2K)</option>
                        <option value="2160">3840x2160 (4K)</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Include Goals</label>
                    <select id="includeGoals">
                        <option value="yes">Yes - Show as Orbs</option>
                        <option value="no">No - Just Fractal</option>
                    </select>
                </div>
            </div>
            
            <img id="previewCanvas" src="" alt="Preview will appear here" style="display: none;">
            <div id="previewPlaceholder" style="padding: 80px; background: rgba(0,0,0,0.3); border-radius: 15px; color: var(--text-dim);">
                Click any export option to generate a preview
            </div>
            
            <div style="margin-top: 30px;">
                <button class="btn btn-gold" onclick="downloadArt()" id="downloadBtn" style="display: none;">
                    ⬇️ Download High Resolution
                </button>
            </div>
        </div>
        
        <div class="share-section">
            <h2>📤 Share Your Art</h2>
            <p style="color: var(--text-dim); margin-bottom: 20px;">Get a shareable link to your fractal art</p>
            <div class="share-link">
                <input type="text" id="shareUrl" placeholder="Generate art first..." readonly>
                <button class="btn btn-primary" onclick="copyShareLink()">📋 Copy</button>
            </div>
        </div>
    </div>
    
    <script>
        const authToken = localStorage.getItem('fractal_token');
        let currentArt = null;
        
        async function generatePoster() {
            await generateArt('poster');
        }
        
        async function generateWallpaper() {
            await generateArt('wallpaper');
        }
        
        async function generateVideo() {
            alert('Video generation requires GPU processing. This feature connects to HuggingFace Spaces for heavy computation.');
            await generateArt('video_frame');
        }
        
        async function generateMeditation() {
            await generateArt('meditation');
        }
        
        async function generateArt(type) {
            const settings = {
                type: type,
                fractal_type: document.getElementById('fractalType').value,
                color_theme: document.getElementById('colorTheme').value,
                resolution: document.getElementById('resolution').value,
                include_goals: document.getElementById('includeGoals').value === 'yes'
            };
            
            document.getElementById('previewPlaceholder').innerHTML = '⏳ Generating your art...';
            
            try {
                const res = await fetch('/api/art/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + authToken
                    },
                    body: JSON.stringify(settings)
                });
                
                const data = await res.json();
                
                if (data.image_base64) {
                    currentArt = data;
                    const img = document.getElementById('previewCanvas');
                    img.src = 'data:image/png;base64,' + data.image_base64;
                    img.style.display = 'block';
                    document.getElementById('previewPlaceholder').style.display = 'none';
                    document.getElementById('downloadBtn').style.display = 'inline-block';
                    
                    if (data.share_token) {
                        document.getElementById('shareUrl').value = 
                            window.location.origin + '/share/' + data.share_token;
                    }
                } else {
                    document.getElementById('previewPlaceholder').innerHTML = 
                        '❌ Error generating art: ' + (data.error || 'Unknown error');
                }
            } catch (e) {
                document.getElementById('previewPlaceholder').innerHTML = '❌ Connection error';
            }
        }
        
        function downloadArt() {
            if (!currentArt || !currentArt.image_base64) return;
            
            const link = document.createElement('a');
            link.href = 'data:image/png;base64,' + currentArt.image_base64;
            link.download = 'life-fractal-' + Date.now() + '.png';
            link.click();
        }
        
        function copyShareLink() {
            const input = document.getElementById('shareUrl');
            input.select();
            document.execCommand('copy');
            alert('Link copied to clipboard!');
        }
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROGRESS TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════════════

TIMELINE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Progress Timeline - Life Fractal Intelligence</title>
    <style>
''' + MAIN_CSS + '''
        .timeline-header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(180deg, rgba(72, 199, 116, 0.1) 0%, transparent 100%);
        }
        .timeline-header h1 { font-size: 2.5em; margin-bottom: 10px; }
        
        .timeline-container {
            display: flex;
            overflow-x: auto;
            padding: 40px 20px;
            gap: 30px;
        }
        .timeline-item {
            min-width: 280px;
            background: var(--bg-card);
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            border: 2px solid rgba(102, 126, 234, 0.2);
            flex-shrink: 0;
        }
        .timeline-date {
            color: var(--accent-gold);
            font-weight: 600;
            margin-bottom: 15px;
        }
        .timeline-fractal {
            width: 200px;
            height: 200px;
            border-radius: 15px;
            margin: 0 auto 15px;
            background: rgba(0,0,0,0.3);
        }
        .timeline-wellness {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-green);
        }
        .timeline-label { color: var(--text-dim); font-size: 0.9em; }
        
        .chart-section {
            margin-top: 50px;
            padding: 40px;
            background: var(--bg-card);
            border-radius: 20px;
        }
        .chart-section h2 { margin-bottom: 30px; text-align: center; }
        #wellnessChart {
            width: 100%;
            height: 300px;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
        }
        
        .insights-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }
        .insight-card {
            background: var(--bg-card);
            padding: 25px;
            border-radius: 15px;
            border-left: 4px solid var(--accent-purple);
        }
        .insight-title { color: var(--accent-gold); margin-bottom: 10px; }
        .insight-value { font-size: 2em; font-weight: bold; }
        .insight-desc { color: var(--text-dim); margin-top: 10px; }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">🌀 Life Fractal Intelligence</div>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/universe">3D Universe</a>
            <a href="/studio">Art Studio</a>
            <a href="/timeline" class="active">Timeline</a>
        </div>
        <div class="nav-right">
            <a href="/app" class="btn btn-secondary">Dashboard</a>
        </div>
    </nav>
    
    <div class="timeline-header">
        <h1>📊 Your Progress Timeline</h1>
        <p style="color: var(--text-dim);">Watch how your life fractal has evolved over time</p>
    </div>
    
    <div class="container">
        <h2 style="margin-bottom: 20px;">🌀 Fractal Evolution</h2>
        <div class="timeline-container" id="timelineContainer">
            <!-- Timeline items will be loaded here -->
            <div style="color: var(--text-dim); padding: 40px;">Loading your timeline...</div>
        </div>
        
        <div class="chart-section">
            <h2>📈 Wellness Trends</h2>
            <canvas id="wellnessChart"></canvas>
        </div>
        
        <div class="insights-section">
            <div class="insight-card">
                <div class="insight-title">🎯 Goals Completed</div>
                <div class="insight-value" id="goalsCompleted">--</div>
                <div class="insight-desc">Total goals achieved this month</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">📈 Best Streak</div>
                <div class="insight-value" id="bestStreak">--</div>
                <div class="insight-desc">Longest consecutive active days</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">⚡ Energy Average</div>
                <div class="insight-value" id="avgEnergy">--</div>
                <div class="insight-desc">Average spoons per day</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">🌟 Wellness Peak</div>
                <div class="insight-value" id="wellnessPeak">--</div>
                <div class="insight-desc">Highest wellness score achieved</div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const authToken = localStorage.getItem('fractal_token');
        
        async function loadTimeline() {
            try {
                const res = await fetch('/api/timeline', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                const data = await res.json();
                
                renderTimeline(data.snapshots || []);
                renderChart(data.wellness_history || []);
                renderInsights(data.insights || {});
            } catch (e) {
                console.error('Error loading timeline:', e);
            }
        }
        
        function renderTimeline(snapshots) {
            const container = document.getElementById('timelineContainer');
            
            if (snapshots.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 60px; color: var(--text-dim);">
                        <div style="font-size: 4em; margin-bottom: 20px;">🌱</div>
                        <div>Your timeline will grow as you use the app.</div>
                        <div>Each day creates a new snapshot of your fractal!</div>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = snapshots.map(snap => `
                <div class="timeline-item">
                    <div class="timeline-date">${new Date(snap.date).toLocaleDateString('en-US', { 
                        month: 'short', day: 'numeric', year: 'numeric' 
                    })}</div>
                    <img class="timeline-fractal" 
                         src="${snap.thumbnail_base64 ? 'data:image/png;base64,' + snap.thumbnail_base64 : ''}" 
                         alt="Fractal snapshot"
                         onerror="this.src=''; this.style.background='linear-gradient(135deg, #667eea, #764ba2)';">
                    <div class="timeline-wellness">${snap.wellness_score?.toFixed(0) || '--'}%</div>
                    <div class="timeline-label">Wellness Score</div>
                </div>
            `).join('');
        }
        
        function renderChart(history) {
            const ctx = document.getElementById('wellnessChart').getContext('2d');
            
            // Generate sample data if empty
            const labels = history.length > 0 ? 
                history.map(h => new Date(h.date).toLocaleDateString()) :
                ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
            
            const data = history.length > 0 ?
                history.map(h => h.wellness) :
                [50, 52, 48, 55, 53, 58, 60];
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Wellness Score',
                        data: data,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#f0c420',
                        pointBorderColor: '#f0c420',
                        pointRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { 
                            min: 0, 
                            max: 100,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#888' }
                        },
                        x: { 
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#888' }
                        }
                    }
                }
            });
        }
        
        function renderInsights(insights) {
            document.getElementById('goalsCompleted').textContent = insights.goals_completed || 0;
            document.getElementById('bestStreak').textContent = (insights.best_streak || 0) + ' days';
            document.getElementById('avgEnergy').textContent = (insights.avg_energy || 10).toFixed(1);
            document.getElementById('wellnessPeak').textContent = (insights.wellness_peak || 50).toFixed(0) + '%';
        }
        
        loadTimeline();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# LIFE DASHBOARD (The full app)
# ═══════════════════════════════════════════════════════════════════════════════════════

APP_DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📋 Dashboard - Life Fractal Intelligence</title>
    <style>
''' + MAIN_CSS + '''
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 25px;
            margin-top: 20px;
        }
        @media (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 800px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }
        
        .life-state-card .metric-fill.health { background: linear-gradient(90deg, #ff6b6b, #ee5a5a); }
        .life-state-card .metric-fill.skills { background: linear-gradient(90deg, #3498db, #2980b9); }
        .life-state-card .metric-fill.finances { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .life-state-card .metric-fill.relationships { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .life-state-card .metric-fill.career { background: linear-gradient(90deg, #9b59b6, #8e44ad); }
        .life-state-card .metric-fill.mood { background: linear-gradient(90deg, #1abc9c, #16a085); }
        .life-state-card .metric-fill.energy { background: linear-gradient(90deg, #f39c12, #d68910); }
        .life-state-card .metric-fill.purpose { background: linear-gradient(90deg, #e67e22, #d35400); }
        .life-state-card .metric-fill.creativity { background: linear-gradient(90deg, #fd79a8, #e84393); }
        .life-state-card .metric-fill.spirituality { background: linear-gradient(90deg, #a29bfe, #6c5ce7); }
        .life-state-card .metric-fill.belief { background: linear-gradient(90deg, #55a3ff, #0984e3); }
        .life-state-card .metric-fill.focus { background: linear-gradient(90deg, #00b894, #00a085); }
        .life-state-card .metric-fill.gratitude { background: linear-gradient(90deg, #fdcb6e, #f9ca24); }
        
        .task-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .task-item:hover { background: rgba(102, 126, 234, 0.1); }
        .task-icon { font-size: 1.5em; margin-right: 15px; }
        .task-info { flex: 1; }
        .task-title { font-weight: 600; margin-bottom: 3px; }
        .task-meta { font-size: 0.85em; color: var(--text-dim); }
        .task-streak {
            background: var(--accent-gold);
            color: #1a1a2e;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .pet-card { text-align: center; }
        .pet-avatar {
            font-size: 5em;
            margin: 20px 0;
            animation: pet-bounce 2s ease-in-out infinite;
        }
        @keyframes pet-bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .pet-name { font-size: 1.3em; font-weight: 600; margin-bottom: 5px; }
        .pet-mood { color: var(--text-dim); margin-bottom: 20px; }
        .pet-stats { display: flex; justify-content: center; gap: 30px; margin-bottom: 20px; }
        .pet-stat { text-align: center; }
        .pet-stat-icon { font-size: 1.5em; }
        .pet-stat-value { font-size: 1.2em; font-weight: 600; }
        .pet-buttons { display: flex; gap: 10px; justify-content: center; }
        
        .sacred-math-card pre {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.9em;
            color: var(--accent-gold);
            overflow-x: auto;
        }
        
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .quick-action {
            padding: 20px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        .quick-action:hover {
            background: rgba(102, 126, 234, 0.2);
            transform: translateY(-3px);
        }
        .quick-action-icon { font-size: 2em; margin-bottom: 10px; }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">🌀 Life Fractal v8</div>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/universe">3D Universe</a>
            <a href="/studio">Art Studio</a>
            <a href="/timeline">Timeline</a>
        </div>
        <div class="nav-right">
            <span class="spoon-badge" id="spoonBadge">🥄 -- spoons</span>
            <span id="userName">Loading...</span>
            <button class="btn btn-secondary" onclick="logout()">Logout</button>
        </div>
    </nav>
    
    <div class="container">
        <div class="quick-actions">
            <div class="quick-action" onclick="location.href='/universe'">
                <div class="quick-action-icon">🌌</div>
                <div>Enter 3D Universe</div>
            </div>
            <div class="quick-action" onclick="location.href='/studio'">
                <div class="quick-action-icon">🎨</div>
                <div>Create Art</div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <!-- Life State -->
            <div class="card life-state-card">
                <div class="card-title">📊 Life State</div>
                <div id="lifeStateMetrics">Loading...</div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center; color: var(--text-dim);">
                    Average: <span id="avgScore">--</span>% | Balance: <span id="balanceScore">--</span>%
                </div>
            </div>
            
            <!-- Recommended Tasks -->
            <div class="card">
                <div class="card-title">✅ Recommended Tasks</div>
                <div id="tasksList">Loading...</div>
            </div>
            
            <!-- Pet Companion -->
            <div class="card pet-card">
                <div class="card-title">🐾 Companion</div>
                <div class="pet-avatar" id="petAvatar">🔥</div>
                <div class="pet-name" id="petName">Buddy</div>
                <div class="pet-mood" id="petMood">Mood: happy</div>
                <div class="pet-stats">
                    <div class="pet-stat">
                        <div class="pet-stat-icon">❤️</div>
                        <div class="pet-stat-value" id="petHappy">78%</div>
                        <div style="font-size: 0.8em; color: var(--text-dim);">Happy</div>
                    </div>
                    <div class="pet-stat">
                        <div class="pet-stat-icon">🍖</div>
                        <div class="pet-stat-value" id="petFed">68%</div>
                        <div style="font-size: 0.8em; color: var(--text-dim);">Fed</div>
                    </div>
                </div>
                <div class="pet-buttons">
                    <button class="btn btn-primary" onclick="feedPet()">Feed 🍖</button>
                    <button class="btn btn-secondary" onclick="playPet()">Play 🎾</button>
                </div>
            </div>
            
            <!-- Sacred Mathematics -->
            <div class="card sacred-math-card" style="grid-column: span 2;">
                <div class="card-title">📐 Sacred Mathematics</div>
                <pre id="sacredMath">
φ (Golden Ratio) = 1.618033988749895
Golden Angle = 137.5077640500378°
(Discount Factor) = 0.618033988749895
Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34...
Flow Optimal Ratio = φ = 1.618 Habit
Formation = 66 days average</pre>
            </div>
        </div>
    </div>
    
    <script>
        const authToken = localStorage.getItem('fractal_token');
        
        const PET_EMOJIS = {
            'phoenix': '🔥',
            'dragon': '🐉',
            'owl': '🦉',
            'fox': '🦊',
            'cat': '🐱'
        };
        
        const TASK_ICONS = {
            'gratitude': '🙏',
            'sleep': '😴',
            'affirmations': '💬',
            'meditation': '🧘',
            'nature': '🌳',
            'journaling': '📓',
            'exercise': '💪',
            'default': '✨'
        };
        
        async function loadDashboard() {
            if (!authToken) {
                location.href = '/';
                return;
            }
            
            try {
                const res = await fetch('/api/dashboard', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                
                if (!res.ok) {
                    if (res.status === 401) {
                        localStorage.removeItem('fractal_token');
                        location.href = '/';
                    }
                    return;
                }
                
                const data = await res.json();
                renderDashboard(data);
            } catch (e) {
                console.error('Error loading dashboard:', e);
            }
        }
        
        function renderDashboard(data) {
            // User info
            document.getElementById('userName').textContent = 'Hi, ' + (data.user?.first_name || 'Friend');
            document.getElementById('spoonBadge').textContent = '🥄 ' + (data.user?.spoons || 12) + ' spoons';
            
            // Life state metrics
            const lifeState = data.life_state || {};
            const metrics = ['health', 'skills', 'finances', 'relationships', 'career', 
                           'mood', 'energy', 'purpose', 'creativity', 'spirituality', 
                           'belief', 'focus', 'gratitude'];
            
            const metricIcons = {
                health: '❤️', skills: '🎯', finances: '💰', relationships: '💕',
                career: '💼', mood: '😊', energy: '⚡', purpose: '🎯',
                creativity: '🎨', spirituality: '🙏', belief: '✨', focus: '🔍', gratitude: '💝'
            };
            
            let metricsHtml = '';
            let total = 0;
            let count = 0;
            
            metrics.forEach(m => {
                const value = lifeState[m] || 50;
                total += value;
                count++;
                metricsHtml += `
                    <div class="metric-row">
                        <span class="metric-label">${metricIcons[m] || ''} ${m.charAt(0).toUpperCase() + m.slice(1)}</span>
                        <div class="metric-bar">
                            <div class="metric-fill ${m}" style="width: ${value}%;"></div>
                        </div>
                        <span class="metric-value">${value.toFixed(0)}%</span>
                    </div>
                `;
            });
            
            document.getElementById('lifeStateMetrics').innerHTML = metricsHtml;
            document.getElementById('avgScore').textContent = (total / count).toFixed(0);
            
            // Calculate balance (how even the scores are)
            const avg = total / count;
            const variance = metrics.reduce((sum, m) => sum + Math.pow((lifeState[m] || 50) - avg, 2), 0) / count;
            const balance = Math.max(0, 100 - Math.sqrt(variance));
            document.getElementById('balanceScore').textContent = balance.toFixed(0);
            
            // Tasks
            const tasks = data.tasks || [];
            if (tasks.length === 0) {
                document.getElementById('tasksList').innerHTML = '<div style="color: var(--text-dim);">No tasks yet!</div>';
            } else {
                document.getElementById('tasksList').innerHTML = tasks.map(t => `
                    <div class="task-item" onclick="completeTask(${t.id})">
                        <div class="task-icon">${TASK_ICONS[t.category] || TASK_ICONS.default}</div>
                        <div class="task-info">
                            <div class="task-title">${t.title}</div>
                            <div class="task-meta">Flow: ${t.flow_percent || 1}% | ${t.duration_minutes || 10}min</div>
                        </div>
                        ${t.streak > 0 ? `<div class="task-streak">${t.streak}🔥</div>` : ''}
                    </div>
                `).join('');
            }
            
            // Pet
            const pet = data.pet || {};
            document.getElementById('petAvatar').textContent = PET_EMOJIS[pet.species] || '🔥';
            document.getElementById('petName').textContent = pet.name || 'Buddy';
            document.getElementById('petMood').textContent = 'Mood: ' + (pet.mood || 'happy');
            document.getElementById('petHappy').textContent = (pet.happiness || 78).toFixed(0) + '%';
            document.getElementById('petFed').textContent = (pet.fed || 68).toFixed(0) + '%';
        }
        
        async function completeTask(taskId) {
            try {
                await fetch('/api/tasks/' + taskId + '/complete', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                loadDashboard();
            } catch (e) {
                console.error('Error completing task:', e);
            }
        }
        
        async function feedPet() {
            try {
                await fetch('/api/pet/feed', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                loadDashboard();
            } catch (e) {
                console.error('Error feeding pet:', e);
            }
        }
        
        async function playPet() {
            try {
                await fetch('/api/pet/play', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                loadDashboard();
            } catch (e) {
                console.error('Error playing with pet:', e);
            }
        }
        
        function logout() {
            localStorage.removeItem('fractal_token');
            location.href = '/';
        }
        
        loadDashboard();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template_string(DASHBOARD_HTML)

@app.route('/universe')
def universe():
    return render_template_string(UNIVERSE_HTML)

@app.route('/studio')
def studio():
    return render_template_string(STUDIO_HTML)

@app.route('/timeline')
def timeline():
    return render_template_string(TIMELINE_HTML)

@app.route('/app')
def app_dashboard():
    return render_template_string(APP_DASHBOARD_HTML)


# Auth Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    first_name = data.get('first_name', '')
    last_name = data.get('last_name', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    conn = get_db()
    existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'Email already registered'}), 400
    
    user_id = secrets.token_hex(8)
    password_hash = generate_password_hash(password)
    now = datetime.now(timezone.utc).isoformat()
    trial_end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    
    conn.execute('''
        INSERT INTO users (id, email, password_hash, first_name, last_name, created_at, trial_end_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, email, password_hash, first_name, last_name, now, trial_end))
    
    # Create initial life state
    conn.execute('''
        INSERT INTO life_state (user_id, date) VALUES (?, ?)
    ''', (user_id, now))
    
    # Create pet
    conn.execute('''
        INSERT INTO pets (user_id, name, species) VALUES (?, ?, ?)
    ''', (user_id, 'Buddy', 'phoenix'))
    
    # Add sample tasks
    sample_tasks = [
        ('Gratitude Practice', 'gratitude', 2, 10),
        ('Quality Sleep', 'sleep', 1, 480),
        ('Affirmations', 'affirmations', 1, 5),
        ('Meditation', 'meditation', 6, 20),
        ('Time in Nature', 'nature', 2, 30),
        ('Journaling', 'journaling', 6, 15),
    ]
    for title, cat, flow, dur in sample_tasks:
        conn.execute('''
            INSERT INTO tasks (user_id, title, category, flow_percent, duration_minutes)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, title, cat, flow, dur))
    
    conn.commit()
    conn.close()
    
    token = f"{user_id}:{secrets.token_hex(16)}"
    return jsonify({'token': token, 'user_id': user_id})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = f"{user['id']}:{secrets.token_hex(16)}"
    return jsonify({'token': token, 'user_id': user['id']})


@app.route('/api/me')
@require_auth
def get_me(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user['id'],
        'email': user['email'],
        'first_name': user['first_name'],
        'last_name': user['last_name'],
        'spoons': user['spoons'],
        'max_spoons': user['max_spoons']
    })


@app.route('/api/dashboard')
@require_auth
def get_dashboard(user_id):
    conn = get_db()
    
    # User
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    # Life state
    life_state = conn.execute('''
        SELECT * FROM life_state WHERE user_id = ? ORDER BY date DESC LIMIT 1
    ''', (user_id,)).fetchone()
    
    # Goals
    goals = conn.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,)).fetchall()
    
    # Tasks
    tasks = conn.execute('SELECT * FROM tasks WHERE user_id = ?', (user_id,)).fetchall()
    
    # Pet
    pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,)).fetchone()
    
    conn.close()
    
    return jsonify({
        'user': dict(user) if user else {},
        'life_state': dict(life_state) if life_state else {},
        'goals': [dict(g) for g in goals],
        'tasks': [dict(t) for t in tasks],
        'pet': dict(pet) if pet else {}
    })


@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def handle_goals(user_id):
    conn = get_db()
    
    if request.method == 'POST':
        data = request.get_json()
        conn.execute('''
            INSERT INTO goals (user_id, title, description, category, priority, progress, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, data.get('title'), data.get('description'), 
              data.get('category', 'general'), data.get('priority', 3),
              data.get('progress', 0), datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    
    goals = conn.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return jsonify([dict(g) for g in goals])


@app.route('/api/tasks/<int:task_id>/complete', methods=['POST'])
@require_auth
def complete_task(user_id, task_id):
    conn = get_db()
    conn.execute('''
        UPDATE tasks SET streak = streak + 1, last_completed = ? WHERE id = ? AND user_id = ?
    ''', (datetime.now(timezone.utc).isoformat(), task_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet(user_id):
    conn = get_db()
    conn.execute('''
        UPDATE pets SET fed = MIN(100, fed + 15), happiness = MIN(100, happiness + 5) WHERE user_id = ?
    ''', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_pet(user_id):
    conn = get_db()
    conn.execute('''
        UPDATE pets SET happiness = MIN(100, happiness + 10), fed = MAX(0, fed - 5) WHERE user_id = ?
    ''', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/timeline')
@require_auth
def get_timeline(user_id):
    conn = get_db()
    
    snapshots = conn.execute('''
        SELECT * FROM fractal_snapshots WHERE user_id = ? ORDER BY date DESC LIMIT 30
    ''', (user_id,)).fetchall()
    
    # Get wellness history from life_state
    history = conn.execute('''
        SELECT date, (health + skills + finances + relationships + career + mood + energy + 
               purpose + creativity + spirituality + belief + focus + gratitude) / 13.0 as wellness
        FROM life_state WHERE user_id = ? ORDER BY date DESC LIMIT 30
    ''', (user_id,)).fetchall()
    
    # Get insights
    goals = conn.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,)).fetchall()
    tasks = conn.execute('SELECT * FROM tasks WHERE user_id = ?', (user_id,)).fetchall()
    
    conn.close()
    
    insights = {
        'goals_completed': sum(1 for g in goals if g['is_completed']),
        'best_streak': max((t['streak'] for t in tasks), default=0),
        'avg_energy': 10,
        'wellness_peak': max((h['wellness'] for h in history), default=50) if history else 50
    }
    
    return jsonify({
        'snapshots': [dict(s) for s in snapshots],
        'wellness_history': [dict(h) for h in history],
        'insights': insights
    })


@app.route('/api/art/generate', methods=['POST'])
@require_auth
def generate_art(user_id):
    data = request.get_json()
    art_type = data.get('type', 'poster')
    
    conn = get_db()
    life_state = conn.execute('''
        SELECT * FROM life_state WHERE user_id = ? ORDER BY date DESC LIMIT 1
    ''', (user_id,)).fetchone()
    goals = conn.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    
    wellness_data = dict(life_state) if life_state else {}
    goals_list = [dict(g) for g in goals]
    
    # Generate the art
    if art_type == 'poster':
        img = fractal_gen.create_poster(wellness_data, goals_list, "My Life Fractal")
    else:
        # Generate fractal
        wellness_avg = sum(v for k, v in wellness_data.items() if isinstance(v, (int, float))) / max(1, len([v for v in wellness_data.values() if isinstance(v, (int, float))])) / 100
        fractal_data = fractal_gen.generate_mandelbrot(max_iter=200)
        img = fractal_gen.apply_wellness_colors(fractal_data, wellness_avg)
    
    # Save export record
    share_token = secrets.token_urlsafe(16)
    conn = get_db()
    conn.execute('''
        INSERT INTO art_exports (user_id, created_at, export_type, share_token)
        VALUES (?, ?, ?, ?)
    ''', (user_id, datetime.now(timezone.utc).isoformat(), art_type, share_token))
    conn.commit()
    conn.close()
    
    return jsonify({
        'image_base64': fractal_gen.to_base64(img),
        'share_token': share_token
    })


@app.route('/share/<token>')
def view_shared_art(token):
    conn = get_db()
    export = conn.execute('SELECT * FROM art_exports WHERE share_token = ?', (token,)).fetchone()
    conn.close()
    
    if not export:
        return "Art not found", 404
    
    # Return a simple page showing the shared art
    return f'''<!DOCTYPE html>
<html><head><title>Shared Life Fractal</title>
<style>body{{background:#0a0a1a;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}}
img{{max-width:90%;border-radius:20px;box-shadow:0 10px 60px rgba(102,126,234,0.3);}}</style></head>
<body><div style="text-align:center;color:white;">
<h1 style="color:#667eea;">🌀 Life Fractal</h1>
<p>Shared on {export["created_at"][:10]}</p>
<p><a href="/" style="color:#f0c420;">Create your own →</a></p>
</div></body></html>'''


@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '8.0',
        'gpu_available': GPU_AVAILABLE,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("=" * 70)
    print("🌀 LIFE FRACTAL INTELLIGENCE v8.0")
    print("=" * 70)
    print(f"✅ GPU: {'Available - ' + GPU_NAME if GPU_AVAILABLE else 'Using CPU'}")
    print(f"✅ Database: {DB_PATH}")
    print(f"✅ Server starting on port {port}")
    print()
    print("🌌 VISUALIZATION-FIRST LIFE PLANNING")
    print("   • 3D Fractal Universe: /universe")
    print("   • Art Therapy Studio: /studio")
    print("   • Progress Timeline: /timeline")
    print("   • Life Dashboard: /app")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=False)
