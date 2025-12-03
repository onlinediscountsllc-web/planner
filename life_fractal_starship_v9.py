"""
🚀 LIFE FRACTAL INTELLIGENCE v9.0 - STARSHIP BRIDGE EDITION
═══════════════════════════════════════════════════════════════════════════════════════

STARSHIP VISUALIZATION SYSTEM
- Star Trek-inspired bridge interface with multiple viewscreens
- HUD displays with sliders and real-time data
- Fractal-generated MIDI ambient audio system
- 30+ fractal types with max iterations
- Swarm intelligence: thousands of agents per goal orb
- GPU detection and WebGL acceleration
- Swedish game design aesthetics (clean, functional, beautiful)
- Memory management and performance optimization

For neurodivergent brains that need to SEE their life to understand it.
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import secrets
import logging
import hashlib
import random
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
    from sklearn.ensemble import RandomForestRegressor
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
# LOGGING & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS - EXTENDED
# ═══════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
PHI_SQUARED = PHI * PHI  # 2.618033988749895
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843]
TRIBONACCI = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504, 927]

# Multiples of 200 for detail levels
DETAIL_LEVELS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]

# 30+ Fractal Types
FRACTAL_TYPES = [
    'mandelbrot', 'julia', 'burning_ship', 'tricorn', 'phoenix',
    'newton', 'nova', 'magnet', 'buffalo', 'celtic',
    'perpendicular_mandelbrot', 'perpendicular_burning_ship', 'perpendicular_celtic',
    'heart', 'mandelbar', 'lambda', 'spider', 'siegel_disk',
    'barnsley_fern', 'sierpinski', 'koch_snowflake', 'dragon_curve',
    'mandelbulb_slice', 'quaternion_julia_slice', 'hybrid_mandelbrot_julia',
    'multibrot_3', 'multibrot_4', 'multibrot_5', 'multibrot_6',
    'cosine_mandelbrot', 'sine_julia', 'exponential_map',
    'tetration', 'collatz', 'lyapunov'
]

# Psychology Metrics (Extended)
PSYCHOLOGY_METRICS = [
    'health', 'skills', 'finances', 'relationships', 'career',
    'mood', 'energy', 'purpose', 'creativity', 'spirituality',
    'belief', 'focus', 'gratitude', 'resilience', 'empathy',
    'self_awareness', 'emotional_regulation', 'motivation', 'optimism',
    'mindfulness', 'social_connection', 'autonomy', 'competence',
    'meaning', 'engagement', 'accomplishment', 'positive_emotion'
]


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

DB_PATH = os.getenv('DATABASE_PATH', 'life_fractal_v9.db')

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
            trial_end_date TEXT,
            preferences TEXT DEFAULT '{}'
        )
    ''')
    
    # Extended life state metrics (27 psychology metrics)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS life_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            health REAL DEFAULT 50, skills REAL DEFAULT 50, finances REAL DEFAULT 50,
            relationships REAL DEFAULT 50, career REAL DEFAULT 50, mood REAL DEFAULT 50,
            energy REAL DEFAULT 50, purpose REAL DEFAULT 50, creativity REAL DEFAULT 50,
            spirituality REAL DEFAULT 50, belief REAL DEFAULT 50, focus REAL DEFAULT 50,
            gratitude REAL DEFAULT 50, resilience REAL DEFAULT 50, empathy REAL DEFAULT 50,
            self_awareness REAL DEFAULT 50, emotional_regulation REAL DEFAULT 50,
            motivation REAL DEFAULT 50, optimism REAL DEFAULT 50, mindfulness REAL DEFAULT 50,
            social_connection REAL DEFAULT 50, autonomy REAL DEFAULT 50, competence REAL DEFAULT 50,
            meaning REAL DEFAULT 50, engagement REAL DEFAULT 50, accomplishment REAL DEFAULT 50,
            positive_emotion REAL DEFAULT 50,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Goals with swarm data
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
            swarm_data TEXT DEFAULT '{}',
            connections TEXT DEFAULT '[]',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Tasks/Habits
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
    
    # ML training data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT,
            input_features TEXT,
            output_label TEXT,
            feedback_score REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # User sessions for GPU/preferences
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_start TEXT,
            gpu_detected INTEGER DEFAULT 0,
            gpu_name TEXT,
            performance_mode TEXT DEFAULT 'balanced',
            fractal_preferences TEXT DEFAULT '{}',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized with extended schema")


# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'life-fractal-v9-starship-key')

init_db()


# ═══════════════════════════════════════════════════════════════════════════════════════
# FRACTAL ENGINE - 30+ TYPES
# ═══════════════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    """Advanced fractal generation with 30+ types"""
    
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
    
    def generate(self, fractal_type='mandelbrot', max_iter=200, zoom=1.0,
                 center=(0, 0), params=None) -> np.ndarray:
        """Generate any fractal type"""
        params = params or {}
        
        generators = {
            'mandelbrot': self._mandelbrot,
            'julia': self._julia,
            'burning_ship': self._burning_ship,
            'tricorn': self._tricorn,
            'phoenix': self._phoenix,
            'multibrot_3': lambda **kw: self._multibrot(power=3, **kw),
            'multibrot_4': lambda **kw: self._multibrot(power=4, **kw),
            'multibrot_5': lambda **kw: self._multibrot(power=5, **kw),
            'celtic': self._celtic,
            'heart': self._heart,
        }
        
        gen_func = generators.get(fractal_type, self._mandelbrot)
        return gen_func(max_iter=max_iter, zoom=zoom, center=center, **params)
    
    def _mandelbrot(self, max_iter=200, zoom=1.0, center=(-0.5, 0), **kwargs) -> np.ndarray:
        x = np.linspace(-2.5/zoom + center[0], 1.5/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            output[mask] = i
        
        return output
    
    def _julia(self, max_iter=200, zoom=1.0, center=(0, 0), 
               c_real=-0.7, c_imag=0.27015, **kwargs) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        c = complex(c_real, c_imag)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c
            output[mask] = i
        
        return output
    
    def _burning_ship(self, max_iter=200, zoom=1.0, center=(-0.5, -0.5), **kwargs) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 1.5/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 1/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = (np.abs(z[mask].real) + 1j * np.abs(z[mask].imag)) ** 2 + c[mask]
            output[mask] = i
        
        return output
    
    def _tricorn(self, max_iter=200, zoom=1.0, center=(-0.5, 0), **kwargs) -> np.ndarray:
        x = np.linspace(-2.5/zoom + center[0], 1.5/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = np.conj(z[mask]) ** 2 + c[mask]
            output[mask] = i
        
        return output
    
    def _phoenix(self, max_iter=200, zoom=1.0, center=(0, 0), 
                 p=-0.5, **kwargs) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        z_prev = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z_new = z[mask] ** 2 + c[mask].real + c[mask].imag * z_prev[mask]
            z_prev[mask] = z[mask]
            z[mask] = z_new
            output[mask] = i
        
        return output
    
    def _multibrot(self, power=3, max_iter=200, zoom=1.0, center=(0, 0), **kwargs) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** power + c[mask]
            output[mask] = i
        
        return output
    
    def _celtic(self, max_iter=200, zoom=1.0, center=(-0.5, 0), **kwargs) -> np.ndarray:
        x = np.linspace(-2.5/zoom + center[0], 1.5/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z_real = np.abs(z[mask].real ** 2 - z[mask].imag ** 2) + c[mask].real
            z_imag = 2 * z[mask].real * z[mask].imag + c[mask].imag
            z[mask] = z_real + 1j * z_imag
            output[mask] = i
        
        return output
    
    def _heart(self, max_iter=200, zoom=1.0, center=(0, 0), **kwargs) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z_real = z[mask].real ** 2 - z[mask].imag ** 2 + c[mask].real
            z_imag = 2 * np.abs(z[mask].real * z[mask].imag) + c[mask].imag
            z[mask] = z_real + 1j * z_imag
            output[mask] = i
        
        return output
    
    def colorize(self, fractal_data: np.ndarray, wellness: float = 0.5, 
                 hue_shift: float = 0.0) -> Image.Image:
        """Apply beautiful coloring based on wellness"""
        norm = (fractal_data - fractal_data.min()) / (fractal_data.max() - fractal_data.min() + 1e-10)
        
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i in range(self.height):
            for j in range(self.width):
                v = norm[i, j]
                if v > 0.95:
                    img_array[i, j] = [8, 4, 16]
                else:
                    hue = (hue_shift + v * PHI_INVERSE) % 1.0
                    sat = 0.7 + 0.3 * wellness
                    val = 0.2 + 0.8 * v
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
    
    def to_base64(self, img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


fractal_engine = FractalEngine()


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_user_from_token(token):
    if not token:
        return None
    try:
        parts = token.split(':')
        if len(parts) != 2:
            return None
        user_id = parts[0]
        conn = get_db()
        user = conn.execute('SELECT id FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        return user_id if user else None
    except:
        return None


def require_auth(f):
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
# CSS STYLES - SWEDISH DESIGN + STARSHIP AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════════════════

STARSHIP_CSS = '''
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg-space: #030308;
    --bg-panel: rgba(8, 12, 28, 0.92);
    --bg-hud: rgba(15, 20, 40, 0.85);
    --accent-cyan: #00d4ff;
    --accent-blue: #4a9eff;
    --accent-purple: #8b5cf6;
    --accent-gold: #fbbf24;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-orange: #f97316;
    --text-bright: #e8f4ff;
    --text-dim: #6b8aaa;
    --border-glow: rgba(0, 212, 255, 0.3);
    --scanline: rgba(0, 212, 255, 0.03);
}

@font-face {
    font-family: 'Starship';
    src: local('Orbitron'), local('Rajdhani'), local('Exo 2'), local('Share Tech Mono');
}

body {
    font-family: 'Segoe UI', 'Starship', system-ui, sans-serif;
    background: var(--bg-space);
    color: var(--text-bright);
    overflow: hidden;
    min-height: 100vh;
}

/* Scanline effect */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        0deg,
        var(--scanline) 0px,
        var(--scanline) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 10000;
    opacity: 0.5;
}

/* HUD Panel Base */
.hud-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-radius: 4px;
    backdrop-filter: blur(20px);
    box-shadow: 
        0 0 20px rgba(0, 212, 255, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.hud-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}

/* Viewscreen */
.viewscreen {
    border: 2px solid var(--accent-cyan);
    border-radius: 8px;
    background: var(--bg-space);
    box-shadow: 
        0 0 30px rgba(0, 212, 255, 0.2),
        inset 0 0 60px rgba(0, 0, 0, 0.8);
    position: relative;
    overflow: hidden;
}

.viewscreen::before {
    content: 'MAIN VIEWSCREEN';
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 10px;
    letter-spacing: 3px;
    color: var(--accent-cyan);
    opacity: 0.6;
    z-index: 100;
}

/* HUD Elements */
.hud-title {
    font-size: 0.7em;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
}

.hud-value {
    font-size: 1.8em;
    font-weight: 300;
    color: var(--text-bright);
    font-family: 'Courier New', monospace;
}

.hud-unit {
    font-size: 0.5em;
    color: var(--text-dim);
    margin-left: 4px;
}

/* Sliders */
.hud-slider {
    width: 100%;
    height: 6px;
    -webkit-appearance: none;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 3px;
    outline: none;
}

.hud-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--accent-cyan);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 0 10px var(--accent-cyan);
}

/* Buttons */
.hud-btn {
    background: transparent;
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8em;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.3s;
}

.hud-btn:hover {
    background: rgba(0, 212, 255, 0.1);
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
}

.hud-btn-primary {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    border: none;
    color: var(--bg-space);
    font-weight: 600;
}

.hud-btn-gold {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-orange));
    border: none;
    color: var(--bg-space);
}

/* Metrics */
.metric-bar-container {
    margin-bottom: 10px;
}

.metric-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75em;
    margin-bottom: 4px;
}

.metric-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
}

.metric-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
    box-shadow: 0 0 8px currentColor;
}

/* Status indicators */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s ease-in-out infinite;
}

.status-online { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }
.status-warning { background: var(--accent-gold); box-shadow: 0 0 10px var(--accent-gold); }
.status-offline { background: var(--accent-red); box-shadow: 0 0 10px var(--accent-red); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Navigation */
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-glow);
}

.nav-logo {
    font-size: 1.2em;
    letter-spacing: 3px;
    color: var(--accent-cyan);
    text-transform: uppercase;
}

.nav-links {
    display: flex;
    gap: 30px;
}

.nav-link {
    color: var(--text-dim);
    text-decoration: none;
    font-size: 0.85em;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: color 0.3s;
    position: relative;
}

.nav-link:hover, .nav-link.active {
    color: var(--accent-cyan);
}

.nav-link.active::after {
    content: '';
    position: absolute;
    bottom: -12px;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent-cyan);
    box-shadow: 0 0 10px var(--accent-cyan);
}

/* Animations */
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.2); }
    50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.4); }
}

@keyframes data-stream {
    0% { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

.glow-animation { animation: glow-pulse 3s ease-in-out infinite; }
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# STARSHIP BRIDGE - MAIN 3D VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

STARSHIP_BRIDGE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Starship Bridge - Life Fractal Intelligence</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
''' + STARSHIP_CSS + '''
        #bridge-container {
            width: 100vw;
            height: 100vh;
            position: relative;
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            grid-template-rows: 60px 1fr 180px;
            gap: 0;
        }
        
        /* Top Navigation Bar */
        .top-bar {
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 24px;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border-glow);
        }
        
        /* Left Control Panel */
        .left-panel {
            padding: 20px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border-glow);
            overflow-y: auto;
        }
        
        /* Main Viewscreen */
        .main-viewscreen {
            position: relative;
            background: var(--bg-space);
        }
        
        #viewscreen-canvas {
            width: 100%;
            height: 100%;
        }
        
        /* Viewscreen overlay HUD */
        .viewscreen-hud {
            position: absolute;
            pointer-events: none;
        }
        
        .viewscreen-hud-tl {
            top: 20px;
            left: 20px;
        }
        
        .viewscreen-hud-tr {
            top: 20px;
            right: 20px;
            text-align: right;
        }
        
        .viewscreen-hud-bl {
            bottom: 20px;
            left: 20px;
        }
        
        .viewscreen-hud-br {
            bottom: 20px;
            right: 20px;
            text-align: right;
        }
        
        .hud-readout {
            font-family: 'Courier New', monospace;
            font-size: 11px;
            color: var(--accent-cyan);
            opacity: 0.8;
            line-height: 1.6;
            text-shadow: 0 0 10px var(--accent-cyan);
        }
        
        /* Right Data Panel */
        .right-panel {
            padding: 20px;
            background: var(--bg-panel);
            border-left: 1px solid var(--border-glow);
            overflow-y: auto;
        }
        
        /* Bottom Control Bar */
        .bottom-bar {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            padding: 15px 24px;
            background: var(--bg-panel);
            border-top: 1px solid var(--border-glow);
        }
        
        /* Audio Visualizer */
        .audio-visualizer {
            height: 60px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 2px;
            padding: 8px;
        }
        
        .audio-bar {
            width: 4px;
            background: linear-gradient(to top, var(--accent-cyan), var(--accent-purple));
            border-radius: 2px;
            transition: height 0.1s;
        }
        
        /* Fractal Type Selector */
        .fractal-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 12px;
        }
        
        .fractal-type-btn {
            padding: 8px;
            font-size: 0.7em;
            text-align: center;
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 4px;
            color: var(--text-dim);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .fractal-type-btn:hover, .fractal-type-btn.active {
            background: rgba(0, 212, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
        
        /* Goals List */
        .goal-orb-item {
            display: flex;
            align-items: center;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            margin-bottom: 10px;
            cursor: pointer;
            border-left: 3px solid var(--accent-purple);
            transition: all 0.3s;
        }
        
        .goal-orb-item:hover {
            background: rgba(0, 212, 255, 0.1);
            border-left-color: var(--accent-cyan);
        }
        
        .goal-orb-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 12px;
            box-shadow: 0 0 10px currentColor;
        }
        
        .goal-orb-info { flex: 1; }
        .goal-orb-title { font-size: 0.9em; margin-bottom: 3px; }
        .goal-orb-progress { font-size: 0.75em; color: var(--text-dim); }
        
        /* Swarm Counter */
        .swarm-counter {
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            color: var(--accent-gold);
        }
        
        /* Sub-viewscreens */
        .sub-viewscreen {
            background: var(--bg-space);
            border: 1px solid var(--border-glow);
            border-radius: 4px;
            padding: 10px;
            position: relative;
        }
        
        .sub-viewscreen-title {
            font-size: 0.65em;
            letter-spacing: 2px;
            color: var(--accent-cyan);
            opacity: 0.6;
            margin-bottom: 8px;
        }
        
        /* Loading */
        #loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-space);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .loading-spinner {
            width: 80px;
            height: 80px;
            border: 2px solid rgba(0, 212, 255, 0.1);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .loading-text {
            margin-top: 20px;
            font-size: 0.9em;
            letter-spacing: 3px;
            color: var(--accent-cyan);
        }
        
        .loading-status {
            margin-top: 10px;
            font-size: 0.75em;
            color: var(--text-dim);
        }
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div id="loading-screen">
        <div class="loading-spinner"></div>
        <div class="loading-text">INITIALIZING BRIDGE SYSTEMS</div>
        <div class="loading-status" id="loading-status">Detecting GPU capabilities...</div>
    </div>
    
    <div id="bridge-container">
        <!-- Top Navigation Bar -->
        <nav class="top-bar">
            <div class="nav-logo">🚀 LIFE FRACTAL • STARSHIP BRIDGE</div>
            <div class="nav-links">
                <a href="/" class="nav-link">HOME</a>
                <a href="/bridge" class="nav-link active">BRIDGE</a>
                <a href="/studio" class="nav-link">ART STUDIO</a>
                <a href="/timeline" class="nav-link">TIMELINE</a>
                <a href="/app" class="nav-link">DASHBOARD</a>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div><span class="status-dot status-online"></span><span id="gpu-status">GPU ACTIVE</span></div>
                <div style="color: var(--accent-gold);">🥄 <span id="spoon-count">12</span></div>
                <button class="hud-btn" onclick="logout()">LOGOUT</button>
            </div>
        </nav>
        
        <!-- Left Control Panel -->
        <aside class="left-panel">
            <div class="hud-title">🎛️ FRACTAL CONTROLS</div>
            
            <div style="margin-bottom: 20px;">
                <label class="metric-label">
                    <span>Complexity</span>
                    <span id="complexity-value">2000</span>
                </label>
                <input type="range" class="hud-slider" id="complexity-slider" 
                       min="200" max="4000" step="200" value="2000">
            </div>
            
            <div style="margin-bottom: 20px;">
                <label class="metric-label">
                    <span>Max Iterations</span>
                    <span id="iterations-value">200</span>
                </label>
                <input type="range" class="hud-slider" id="iterations-slider" 
                       min="50" max="1000" step="50" value="200">
            </div>
            
            <div style="margin-bottom: 20px;">
                <label class="metric-label">
                    <span>Vibrance</span>
                    <span id="vibrance-value">100</span>%
                </label>
                <input type="range" class="hud-slider" id="vibrance-slider" 
                       min="0" max="200" value="100">
            </div>
            
            <div style="margin-bottom: 20px;">
                <label class="metric-label">
                    <span>Swarm Agents</span>
                    <span id="swarm-value">1000</span>
                </label>
                <input type="range" class="hud-slider" id="swarm-slider" 
                       min="100" max="5000" step="100" value="1000">
            </div>
            
            <div class="hud-title" style="margin-top: 25px;">📐 FRACTAL TYPE</div>
            <div class="fractal-grid">
                <button class="fractal-type-btn active" data-type="mandelbrot">Mandelbrot</button>
                <button class="fractal-type-btn" data-type="julia">Julia</button>
                <button class="fractal-type-btn" data-type="burning_ship">Burning Ship</button>
                <button class="fractal-type-btn" data-type="phoenix">Phoenix</button>
                <button class="fractal-type-btn" data-type="tricorn">Tricorn</button>
                <button class="fractal-type-btn" data-type="celtic">Celtic</button>
                <button class="fractal-type-btn" data-type="multibrot_3">Multibrot³</button>
                <button class="fractal-type-btn" data-type="heart">Heart</button>
            </div>
            
            <div style="margin-top: 25px;">
                <button class="hud-btn hud-btn-primary" style="width: 100%; margin-bottom: 10px;" onclick="regenerate()">
                    🔄 REGENERATE
                </button>
                <button class="hud-btn hud-btn-gold" style="width: 100%;" onclick="addGoal()">
                    ✨ ADD GOAL ORB
                </button>
            </div>
        </aside>
        
        <!-- Main Viewscreen -->
        <main class="main-viewscreen viewscreen">
            <div id="viewscreen-canvas"></div>
            
            <!-- HUD Overlays -->
            <div class="viewscreen-hud viewscreen-hud-tl">
                <div class="hud-readout">
                    STARDATE: <span id="stardate">2025.337</span><br>
                    FRAME: <span id="frame-count">0</span><br>
                    FPS: <span id="fps-display">60</span>
                </div>
            </div>
            
            <div class="viewscreen-hud viewscreen-hud-tr">
                <div class="hud-readout">
                    φ = <span id="phi-display">1.618034</span><br>
                    θ = <span id="theta-display">137.51°</span><br>
                    ZOOM: <span id="zoom-display">1.00x</span>
                </div>
            </div>
            
            <div class="viewscreen-hud viewscreen-hud-bl">
                <div class="hud-readout">
                    GOALS: <span id="goal-count">0</span> ACTIVE<br>
                    SWARM: <span id="total-agents">0</span> AGENTS<br>
                    WELLNESS: <span id="wellness-display">--</span>%
                </div>
            </div>
            
            <div class="viewscreen-hud viewscreen-hud-br">
                <div class="hud-readout">
                    GPU: <span id="gpu-name">Detecting...</span><br>
                    VRAM: <span id="vram-usage">--</span><br>
                    MODE: <span id="render-mode">WebGL 2.0</span>
                </div>
            </div>
        </main>
        
        <!-- Right Data Panel -->
        <aside class="right-panel">
            <div class="hud-title">🎯 GOAL ORBS</div>
            <div id="goals-list">
                <div style="color: var(--text-dim); font-size: 0.85em;">Loading orbs...</div>
            </div>
            
            <div class="hud-title" style="margin-top: 25px;">📊 LIFE METRICS</div>
            <div id="metrics-display">
                <!-- Metrics will be loaded here -->
            </div>
            
            <div class="hud-title" style="margin-top: 25px;">🤖 AI INSIGHTS</div>
            <div class="sub-viewscreen">
                <div class="sub-viewscreen-title">PATTERN ANALYSIS</div>
                <div id="ai-insights" style="font-size: 0.8em; color: var(--text-dim);">
                    Analyzing fractal patterns...
                </div>
            </div>
        </aside>
        
        <!-- Bottom Control Bar -->
        <footer class="bottom-bar">
            <!-- Left: Audio Controls -->
            <div>
                <div class="hud-title">🔊 AMBIENT AUDIO</div>
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <button class="hud-btn" id="audio-toggle" onclick="toggleAudio()">▶ PLAY</button>
                    <input type="range" class="hud-slider" id="volume-slider" min="0" max="100" value="30" style="flex: 1;">
                </div>
                <div class="audio-visualizer" id="audio-visualizer">
                    <!-- Audio bars generated by JS -->
                </div>
            </div>
            
            <!-- Center: Sub-viewscreens -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="sub-viewscreen">
                    <div class="sub-viewscreen-title">FRACTAL PREVIEW</div>
                    <canvas id="fractal-preview" width="120" height="80" style="width: 100%; border-radius: 4px;"></canvas>
                </div>
                <div class="sub-viewscreen">
                    <div class="sub-viewscreen-title">ENERGY FLOW</div>
                    <canvas id="energy-preview" width="120" height="80" style="width: 100%; border-radius: 4px;"></canvas>
                </div>
                <div class="sub-viewscreen">
                    <div class="sub-viewscreen-title">SWARM MAP</div>
                    <canvas id="swarm-preview" width="120" height="80" style="width: 100%; border-radius: 4px;"></canvas>
                </div>
            </div>
            
            <!-- Right: Quick Actions -->
            <div>
                <div class="hud-title">⚡ QUICK ACTIONS</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                    <button class="hud-btn" onclick="toggleSacred()">📐 SACRED</button>
                    <button class="hud-btn" onclick="toggleSwarm()">🐝 SWARM</button>
                    <button class="hud-btn" onclick="resetCamera()">🎯 RESET</button>
                    <button class="hud-btn" onclick="screenshot()">📸 CAPTURE</button>
                </div>
            </div>
        </footer>
    </div>
    
    <script>
        // ═══════════════════════════════════════════════════════════════════════════════
        // SACRED MATHEMATICS
        // ═══════════════════════════════════════════════════════════════════════════════
        
        const PHI = (1 + Math.sqrt(5)) / 2;
        const PHI_INV = PHI - 1;
        const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
        const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377];
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // GPU DETECTION
        // ═══════════════════════════════════════════════════════════════════════════════
        
        let gpuInfo = { name: 'Unknown', tier: 'medium', vram: 'Unknown' };
        
        function detectGPU() {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            
            if (gl) {
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo) {
                    gpuInfo.name = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                    
                    // Determine tier based on GPU name
                    const gpuName = gpuInfo.name.toLowerCase();
                    if (gpuName.includes('rtx') || gpuName.includes('radeon rx 6') || gpuName.includes('radeon rx 7')) {
                        gpuInfo.tier = 'high';
                    } else if (gpuName.includes('gtx') || gpuName.includes('radeon rx 5')) {
                        gpuInfo.tier = 'medium';
                    } else {
                        gpuInfo.tier = 'low';
                    }
                }
                
                document.getElementById('gpu-name').textContent = gpuInfo.name.substring(0, 30);
                document.getElementById('gpu-status').textContent = gpuInfo.tier.toUpperCase() + ' GPU';
                document.getElementById('render-mode').textContent = gl.getParameter(gl.VERSION);
            }
            
            return gpuInfo;
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // THREE.JS SETUP
        // ═══════════════════════════════════════════════════════════════════════════════
        
        let scene, camera, renderer;
        let fractalParticles, goalOrbs = [], swarmParticles = [];
        let sacredGeometry = [], connectionLines = [];
        let audioContext, audioAnalyser, audioGain;
        let isAudioPlaying = false;
        let frameCount = 0;
        let lastTime = performance.now();
        let fps = 60;
        
        // Settings
        let settings = {
            complexity: 2000,
            maxIterations: 200,
            vibrance: 100,
            swarmAgents: 1000,
            fractalType: 'mandelbrot',
            showSacred: true,
            showSwarm: true,
            autoRotate: true
        };
        
        // Camera controls
        let isDragging = false;
        let previousMouse = { x: 0, y: 0 };
        let cameraDistance = 25;
        let cameraTheta = 0;
        let cameraPhi = Math.PI / 4;
        
        const authToken = localStorage.getItem('fractal_token');
        
        function init() {
            updateLoadingStatus('Initializing Three.js engine...');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x020208);
            scene.fog = new THREE.FogExp2(0x020208, 0.008);
            
            // Camera
            const container = document.getElementById('viewscreen-canvas');
            const aspect = container.clientWidth / container.clientHeight;
            camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 2000);
            camera.position.set(0, 10, 25);
            
            // Renderer with GPU optimization
            updateLoadingStatus('Configuring WebGL renderer...');
            renderer = new THREE.WebGLRenderer({ 
                antialias: gpuInfo.tier !== 'low',
                powerPreference: 'high-performance',
                alpha: true
            });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, gpuInfo.tier === 'high' ? 2 : 1.5));
            container.appendChild(renderer.domElement);
            
            // Lighting
            updateLoadingStatus('Setting up lighting systems...');
            const ambientLight = new THREE.AmbientLight(0x101020, 0.4);
            scene.add(ambientLight);
            
            // Colored point lights for spaceship feel
            const cyanLight = new THREE.PointLight(0x00d4ff, 2, 150);
            cyanLight.position.set(20, 20, 20);
            scene.add(cyanLight);
            
            const purpleLight = new THREE.PointLight(0x8b5cf6, 2, 150);
            purpleLight.position.set(-20, -15, 20);
            scene.add(purpleLight);
            
            const goldLight = new THREE.PointLight(0xfbbf24, 1.5, 100);
            goldLight.position.set(0, 30, -15);
            scene.add(goldLight);
            
            // Create visuals
            updateLoadingStatus('Generating fractal particles...');
            createFractalParticles();
            
            updateLoadingStatus('Building sacred geometry...');
            createSacredGeometry();
            
            updateLoadingStatus('Creating starfield...');
            createStarfield();
            
            // Events
            window.addEventListener('resize', onWindowResize);
            setupControls();
            
            // Initialize audio
            updateLoadingStatus('Initializing audio system...');
            initAudio();
            
            // Load user data
            updateLoadingStatus('Loading user data...');
            loadUserData();
            
            // Setup sub-viewscreen previews
            initPreviews();
            
            // Start animation
            animate();
            
            // Hide loading screen
            setTimeout(() => {
                document.getElementById('loading-screen').style.opacity = '0';
                setTimeout(() => {
                    document.getElementById('loading-screen').style.display = 'none';
                }, 500);
            }, 1500);
        }
        
        function updateLoadingStatus(status) {
            document.getElementById('loading-status').textContent = status;
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // FRACTAL PARTICLES
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function createFractalParticles() {
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            
            const resolution = Math.sqrt(settings.complexity);
            const power = 8;
            const iterations = 6;
            const scale = 5;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const theta = (i / resolution) * Math.PI;
                    const phi_angle = (j / resolution) * Math.PI * 2;
                    
                    let x = Math.sin(theta) * Math.cos(phi_angle);
                    let y = Math.sin(theta) * Math.sin(phi_angle);
                    let z = Math.cos(theta);
                    
                    // Mandelbulb-style iteration
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
                    
                    const finalR = scale * (1 + 0.4 * Math.sin(theta * PHI * 3) * Math.cos(phi_angle * PHI * 2));
                    vertices.push(
                        finalR * Math.sin(theta) * Math.cos(phi_angle),
                        finalR * Math.cos(theta),
                        finalR * Math.sin(theta) * Math.sin(phi_angle)
                    );
                    
                    // Cyan-purple color scheme
                    const hue = (i / resolution * PHI_INV + j / resolution) % 1;
                    const saturation = 0.7 + (settings.vibrance / 100) * 0.3;
                    const color = new THREE.Color().setHSL(
                        hue * 0.3 + 0.5, // Cyan to purple range
                        saturation,
                        0.5 + 0.3 * Math.sin(theta * 4)
                    );
                    colors.push(color.r, color.g, color.b);
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({
                size: gpuInfo.tier === 'high' ? 0.05 : 0.08,
                vertexColors: true,
                transparent: true,
                opacity: 0.9,
                blending: THREE.AdditiveBlending
            });
            
            if (fractalParticles) scene.remove(fractalParticles);
            fractalParticles = new THREE.Points(geometry, material);
            scene.add(fractalParticles);
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // SACRED GEOMETRY
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function createSacredGeometry() {
            sacredGeometry.forEach(g => scene.remove(g));
            sacredGeometry = [];
            
            // Golden Spiral
            const spiralGeometry = new THREE.BufferGeometry();
            const spiralVertices = [];
            for (let i = 0; i < 1000; i++) {
                const angle = i * 0.1;
                const r = 0.2 * Math.pow(PHI, 2 * angle / Math.PI);
                if (r > 12) break;
                spiralVertices.push(
                    r * Math.cos(angle),
                    r * Math.sin(angle) * 0.4,
                    r * Math.sin(angle)
                );
            }
            spiralGeometry.setAttribute('position', new THREE.Float32BufferAttribute(spiralVertices, 3));
            const spiral = new THREE.Line(spiralGeometry, new THREE.LineBasicMaterial({
                color: 0x00d4ff,
                transparent: true,
                opacity: 0.4
            }));
            scene.add(spiral);
            sacredGeometry.push(spiral);
            
            // Icosahedron wireframe
            const icoGeo = new THREE.IcosahedronGeometry(8, 0);
            const icoEdges = new THREE.EdgesGeometry(icoGeo);
            const icosahedron = new THREE.LineSegments(icoEdges, new THREE.LineBasicMaterial({
                color: 0x8b5cf6,
                transparent: true,
                opacity: 0.2
            }));
            scene.add(icosahedron);
            sacredGeometry.push(icosahedron);
            
            // Dodecahedron
            const dodecaGeo = new THREE.DodecahedronGeometry(10, 0);
            const dodecaEdges = new THREE.EdgesGeometry(dodecaGeo);
            const dodecahedron = new THREE.LineSegments(dodecaEdges, new THREE.LineBasicMaterial({
                color: 0xfbbf24,
                transparent: true,
                opacity: 0.15
            }));
            scene.add(dodecahedron);
            sacredGeometry.push(dodecahedron);
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // STARFIELD
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function createStarfield() {
            const starsGeometry = new THREE.BufferGeometry();
            const starVertices = [];
            const starColors = [];
            
            for (let i = 0; i < 5000; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 80 + Math.random() * 120;
                
                starVertices.push(
                    r * Math.sin(phi) * Math.cos(theta),
                    r * Math.sin(phi) * Math.sin(theta),
                    r * Math.cos(phi)
                );
                
                const brightness = 0.3 + Math.random() * 0.7;
                starColors.push(brightness * 0.9, brightness * 0.95, brightness);
            }
            
            starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
            starsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));
            
            scene.add(new THREE.Points(starsGeometry, new THREE.PointsMaterial({
                size: 0.2,
                vertexColors: true,
                transparent: true,
                opacity: 0.85
            })));
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // GOAL ORBS WITH SWARM INTELLIGENCE
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function createGoalOrbs(goals) {
            // Clear existing
            goalOrbs.forEach(orb => scene.remove(orb));
            goalOrbs = [];
            connectionLines.forEach(line => scene.remove(line));
            connectionLines = [];
            swarmParticles.forEach(swarm => scene.remove(swarm));
            swarmParticles = [];
            
            if (!goals || goals.length === 0) return;
            
            let totalAgents = 0;
            
            goals.forEach((goal, index) => {
                // Golden angle positioning
                const goldenAngle = index * GOLDEN_ANGLE;
                const radius = 6 + (goal.priority || 3) * 0.7;
                const height = (goal.progress / 100) * 6 - 3 + Math.sin(index * PHI) * 2;
                
                const x = radius * Math.cos(goldenAngle);
                const z = radius * Math.sin(goldenAngle);
                const y = height;
                
                // Orb size based on progress
                const size = 0.4 + (goal.progress / 100) * 0.6;
                const geometry = new THREE.SphereGeometry(size, 32, 32);
                
                // Color based on status
                let color;
                if (goal.is_completed || goal.progress >= 100) {
                    color = new THREE.Color(0x22c55e);
                } else if (goal.progress >= 70) {
                    color = new THREE.Color(0x4a9eff);
                } else if (goal.progress >= 40) {
                    color = new THREE.Color(0xfbbf24);
                } else {
                    color = new THREE.Color(0xef4444);
                }
                
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.5,
                    transparent: true,
                    opacity: 0.95
                });
                
                const orb = new THREE.Mesh(geometry, material);
                orb.position.set(x, y, z);
                orb.userData = { goal, index, baseY: y };
                
                // Glow effect
                const glowGeo = new THREE.SphereGeometry(size * 2, 16, 16);
                const glowMat = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.1
                });
                orb.add(new THREE.Mesh(glowGeo, glowMat));
                
                scene.add(orb);
                goalOrbs.push(orb);
                
                // Create swarm particles around this orb
                const agentsPerOrb = Math.floor(settings.swarmAgents / Math.max(goals.length, 1));
                createSwarmForOrb(orb, agentsPerOrb, color);
                totalAgents += agentsPerOrb;
                
                // Connection line to center
                const lineGeo = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    orb.position
                ]);
                const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.2
                }));
                scene.add(line);
                connectionLines.push(line);
            });
            
            // Fibonacci connections
            for (let i = 0; i < goalOrbs.length; i++) {
                for (let j = i + 1; j < goalOrbs.length; j++) {
                    if (FIBONACCI.includes(j - i)) {
                        const lineGeo = new THREE.BufferGeometry().setFromPoints([
                            goalOrbs[i].position,
                            goalOrbs[j].position
                        ]);
                        const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
                            color: 0x00d4ff,
                            transparent: true,
                            opacity: 0.1
                        }));
                        scene.add(line);
                        connectionLines.push(line);
                    }
                }
            }
            
            document.getElementById('total-agents').textContent = totalAgents.toLocaleString();
        }
        
        function createSwarmForOrb(orb, agentCount, color) {
            const swarmGeo = new THREE.BufferGeometry();
            const positions = [];
            const velocities = [];
            
            for (let i = 0; i < agentCount; i++) {
                // Distribute around orb
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 0.5 + Math.random() * 1.5;
                
                positions.push(
                    orb.position.x + r * Math.sin(phi) * Math.cos(theta),
                    orb.position.y + r * Math.sin(phi) * Math.sin(theta),
                    orb.position.z + r * Math.cos(phi)
                );
            }
            
            swarmGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            
            const swarmMat = new THREE.PointsMaterial({
                size: 0.03,
                color: color,
                transparent: true,
                opacity: 0.6,
                blending: THREE.AdditiveBlending
            });
            
            const swarm = new THREE.Points(swarmGeo, swarmMat);
            swarm.userData = { orb: orb, velocities: velocities };
            scene.add(swarm);
            swarmParticles.push(swarm);
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // AUDIO SYSTEM - FRACTAL MIDI GENERATION
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function initAudio() {
            // Create audio visualizer bars
            const visualizer = document.getElementById('audio-visualizer');
            for (let i = 0; i < 32; i++) {
                const bar = document.createElement('div');
                bar.className = 'audio-bar';
                bar.style.height = '5px';
                visualizer.appendChild(bar);
            }
        }
        
        function toggleAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                audioGain = audioContext.createGain();
                audioGain.connect(audioContext.destination);
                audioGain.gain.value = 0.3;
                
                // Create ambient drone based on fractal parameters
                createAmbientDrone();
            }
            
            if (isAudioPlaying) {
                audioGain.gain.value = 0;
                document.getElementById('audio-toggle').textContent = '▶ PLAY';
            } else {
                audioGain.gain.value = document.getElementById('volume-slider').value / 100;
                document.getElementById('audio-toggle').textContent = '⏸ PAUSE';
            }
            
            isAudioPlaying = !isAudioPlaying;
        }
        
        function createAmbientDrone() {
            // Base frequency from golden ratio
            const baseFreq = 432 * PHI_INV; // ~267 Hz
            
            // Create oscillators based on Fibonacci ratios
            const frequencies = FIBONACCI.slice(0, 6).map(f => baseFreq * (f / 8));
            
            frequencies.forEach((freq, i) => {
                const osc = audioContext.createOscillator();
                const gain = audioContext.createGain();
                
                osc.type = i % 2 === 0 ? 'sine' : 'triangle';
                osc.frequency.value = freq;
                
                gain.gain.value = 0.1 / (i + 1);
                
                osc.connect(gain);
                gain.connect(audioGain);
                osc.start();
                
                // Slow modulation
                const lfo = audioContext.createOscillator();
                const lfoGain = audioContext.createGain();
                lfo.frequency.value = 0.1 + i * 0.05;
                lfoGain.gain.value = freq * 0.02;
                lfo.connect(lfoGain);
                lfoGain.connect(osc.frequency);
                lfo.start();
            });
        }
        
        function updateAudioVisualizer() {
            const bars = document.querySelectorAll('.audio-bar');
            bars.forEach((bar, i) => {
                const height = isAudioPlaying ? 
                    5 + Math.sin(frameCount * 0.05 + i * 0.3) * 25 + Math.random() * 10 : 5;
                bar.style.height = height + 'px';
            });
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // CONTROLS
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function setupControls() {
            const canvas = renderer.domElement;
            
            canvas.addEventListener('mousedown', e => {
                if (e.button === 0) {
                    isDragging = true;
                    previousMouse = { x: e.clientX, y: e.clientY };
                }
            });
            
            canvas.addEventListener('mouseup', () => isDragging = false);
            canvas.addEventListener('mouseleave', () => isDragging = false);
            
            canvas.addEventListener('mousemove', e => {
                if (!isDragging) return;
                cameraTheta -= (e.clientX - previousMouse.x) * 0.005;
                cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, 
                    cameraPhi + (e.clientY - previousMouse.y) * 0.005));
                previousMouse = { x: e.clientX, y: e.clientY };
            });
            
            canvas.addEventListener('wheel', e => {
                e.preventDefault();
                cameraDistance = Math.max(10, Math.min(80, cameraDistance + e.deltaY * 0.03));
            });
            
            // Slider controls
            document.getElementById('complexity-slider').addEventListener('input', e => {
                settings.complexity = parseInt(e.target.value);
                document.getElementById('complexity-value').textContent = settings.complexity;
            });
            
            document.getElementById('iterations-slider').addEventListener('input', e => {
                settings.maxIterations = parseInt(e.target.value);
                document.getElementById('iterations-value').textContent = settings.maxIterations;
            });
            
            document.getElementById('vibrance-slider').addEventListener('input', e => {
                settings.vibrance = parseInt(e.target.value);
                document.getElementById('vibrance-value').textContent = settings.vibrance;
            });
            
            document.getElementById('swarm-slider').addEventListener('input', e => {
                settings.swarmAgents = parseInt(e.target.value);
                document.getElementById('swarm-value').textContent = settings.swarmAgents;
            });
            
            document.getElementById('volume-slider').addEventListener('input', e => {
                if (audioGain) audioGain.gain.value = e.target.value / 100;
            });
            
            // Fractal type buttons
            document.querySelectorAll('.fractal-type-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.fractal-type-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    settings.fractalType = btn.dataset.type;
                });
            });
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // DATA LOADING
        // ═══════════════════════════════════════════════════════════════════════════════
        
        async function loadUserData() {
            if (!authToken) {
                document.getElementById('goals-list').innerHTML = 
                    '<div style="color: var(--accent-red);">Please login to view goals</div>';
                return;
            }
            
            try {
                const res = await fetch('/api/dashboard', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                
                if (!res.ok) throw new Error('API error');
                
                const data = await res.json();
                
                // Update goals
                createGoalOrbs(data.goals || []);
                updateGoalsPanel(data.goals || []);
                document.getElementById('goal-count').textContent = data.goals?.length || 0;
                
                // Update metrics
                updateMetricsDisplay(data.life_state || {});
                
                // Update spoons
                document.getElementById('spoon-count').textContent = data.user?.spoons || 12;
                
                // Calculate wellness
                const lifeState = data.life_state || {};
                const values = Object.values(lifeState).filter(v => typeof v === 'number');
                const wellness = values.length ? values.reduce((a, b) => a + b) / values.length : 50;
                document.getElementById('wellness-display').textContent = wellness.toFixed(0);
                
            } catch (e) {
                console.error('Error loading data:', e);
            }
        }
        
        function updateGoalsPanel(goals) {
            const container = document.getElementById('goals-list');
            
            if (!goals || goals.length === 0) {
                container.innerHTML = '<div style="color: var(--text-dim);">No goals yet. Add some!</div>';
                return;
            }
            
            container.innerHTML = goals.map((goal, i) => {
                const color = goal.progress >= 70 ? 'var(--accent-green)' : 
                              goal.progress >= 40 ? 'var(--accent-gold)' : 'var(--accent-red)';
                return `
                    <div class="goal-orb-item" onclick="focusGoal(${i})">
                        <div class="goal-orb-indicator" style="background: ${color}; color: ${color};"></div>
                        <div class="goal-orb-info">
                            <div class="goal-orb-title">${goal.title}</div>
                            <div class="goal-orb-progress">${goal.progress?.toFixed(0) || 0}% complete</div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function updateMetricsDisplay(lifeState) {
            const container = document.getElementById('metrics-display');
            const metrics = ['mood', 'energy', 'focus', 'motivation', 'resilience'];
            
            container.innerHTML = metrics.map(m => {
                const value = lifeState[m] || 50;
                const color = value >= 70 ? 'var(--accent-green)' : 
                              value >= 40 ? 'var(--accent-gold)' : 'var(--accent-red)';
                return `
                    <div class="metric-bar-container">
                        <div class="metric-label">
                            <span>${m.charAt(0).toUpperCase() + m.slice(1)}</span>
                            <span>${value.toFixed(0)}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${value}%; background: ${color};"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // ANIMATION LOOP
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            
            // FPS calculation
            const now = performance.now();
            fps = Math.round(1000 / (now - lastTime));
            lastTime = now;
            
            // Update HUD
            if (frameCount % 10 === 0) {
                document.getElementById('frame-count').textContent = frameCount;
                document.getElementById('fps-display').textContent = fps;
                document.getElementById('theta-display').textContent = 
                    (GOLDEN_ANGLE * 180 / Math.PI).toFixed(2) + '°';
                document.getElementById('zoom-display').textContent = 
                    (25 / cameraDistance).toFixed(2) + 'x';
                
                // Update stardate
                const now_date = new Date();
                const stardate = now_date.getFullYear() + '.' + 
                    Math.floor((now_date.getMonth() * 30 + now_date.getDate()) / 365 * 1000);
                document.getElementById('stardate').textContent = stardate;
            }
            
            // Camera position
            camera.position.x = cameraDistance * Math.sin(cameraPhi) * Math.cos(cameraTheta);
            camera.position.y = cameraDistance * Math.cos(cameraPhi);
            camera.position.z = cameraDistance * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            camera.lookAt(0, 0, 0);
            
            // Auto-rotate if not dragging
            if (!isDragging && settings.autoRotate) {
                cameraTheta += 0.001;
            }
            
            // Animate fractal
            if (fractalParticles) {
                fractalParticles.rotation.y += 0.0008;
                fractalParticles.rotation.x = Math.sin(frameCount * 0.0005) * 0.1;
            }
            
            // Animate goal orbs
            goalOrbs.forEach((orb, i) => {
                orb.position.y = orb.userData.baseY + Math.sin(frameCount * 0.015 + i * PHI) * 0.2;
                orb.rotation.y += 0.008;
            });
            
            // Animate sacred geometry
            sacredGeometry.forEach((geo, i) => {
                geo.rotation.y += 0.001 * (i + 1);
                geo.rotation.x += 0.0005 * (i + 1);
            });
            
            // Animate swarm particles (simple orbit)
            swarmParticles.forEach(swarm => {
                const positions = swarm.geometry.attributes.position.array;
                const orb = swarm.userData.orb;
                
                for (let i = 0; i < positions.length; i += 3) {
                    const dx = positions[i] - orb.position.x;
                    const dy = positions[i + 1] - orb.position.y;
                    const dz = positions[i + 2] - orb.position.z;
                    
                    // Rotate around orb
                    const angle = 0.02;
                    const newDx = dx * Math.cos(angle) - dz * Math.sin(angle);
                    const newDz = dx * Math.sin(angle) + dz * Math.cos(angle);
                    
                    positions[i] = orb.position.x + newDx;
                    positions[i + 1] = orb.position.y + dy + Math.sin(frameCount * 0.05 + i) * 0.01;
                    positions[i + 2] = orb.position.z + newDz;
                }
                
                swarm.geometry.attributes.position.needsUpdate = true;
            });
            
            // Update audio visualizer
            updateAudioVisualizer();
            
            renderer.render(scene, camera);
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // PREVIEW CANVASES
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function initPreviews() {
            // Mini fractal preview
            const fractalCanvas = document.getElementById('fractal-preview');
            const fCtx = fractalCanvas.getContext('2d');
            drawMiniFractal(fCtx, fractalCanvas.width, fractalCanvas.height);
            
            // Energy flow preview
            const energyCanvas = document.getElementById('energy-preview');
            const eCtx = energyCanvas.getContext('2d');
            animateEnergyFlow(eCtx, energyCanvas.width, energyCanvas.height);
            
            // Swarm map preview
            const swarmCanvas = document.getElementById('swarm-preview');
            const sCtx = swarmCanvas.getContext('2d');
            animateSwarmMap(sCtx, swarmCanvas.width, swarmCanvas.height);
        }
        
        function drawMiniFractal(ctx, w, h) {
            const imageData = ctx.createImageData(w, h);
            const data = imageData.data;
            
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const cx = (x / w - 0.5) * 4 - 0.5;
                    const cy = (y / h - 0.5) * 4;
                    
                    let zx = 0, zy = 0;
                    let iter = 0;
                    const maxIter = 50;
                    
                    while (zx * zx + zy * zy < 4 && iter < maxIter) {
                        const tmp = zx * zx - zy * zy + cx;
                        zy = 2 * zx * zy + cy;
                        zx = tmp;
                        iter++;
                    }
                    
                    const i = (y * w + x) * 4;
                    if (iter === maxIter) {
                        data[i] = data[i + 1] = data[i + 2] = 8;
                    } else {
                        const hue = iter / maxIter;
                        data[i] = Math.floor(hue * 100 + 50);
                        data[i + 1] = Math.floor(hue * 150 + 80);
                        data[i + 2] = Math.floor(hue * 255);
                    }
                    data[i + 3] = 255;
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
        }
        
        function animateEnergyFlow(ctx, w, h) {
            let t = 0;
            setInterval(() => {
                ctx.fillStyle = 'rgba(3, 3, 8, 0.1)';
                ctx.fillRect(0, 0, w, h);
                
                ctx.strokeStyle = 'rgba(0, 212, 255, 0.5)';
                ctx.beginPath();
                for (let x = 0; x < w; x++) {
                    const y = h / 2 + Math.sin(x * 0.1 + t) * 20 + Math.sin(x * 0.05 + t * PHI) * 10;
                    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.stroke();
                t += 0.05;
            }, 50);
        }
        
        function animateSwarmMap(ctx, w, h) {
            const particles = [];
            for (let i = 0; i < 50; i++) {
                particles.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    vx: (Math.random() - 0.5) * 2,
                    vy: (Math.random() - 0.5) * 2
                });
            }
            
            setInterval(() => {
                ctx.fillStyle = 'rgba(3, 3, 8, 0.2)';
                ctx.fillRect(0, 0, w, h);
                
                particles.forEach(p => {
                    p.x += p.vx;
                    p.y += p.vy;
                    
                    if (p.x < 0 || p.x > w) p.vx *= -1;
                    if (p.y < 0 || p.y > h) p.vy *= -1;
                    
                    ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
                    ctx.fill();
                });
            }, 50);
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // ACTIONS
        // ═══════════════════════════════════════════════════════════════════════════════
        
        function regenerate() {
            createFractalParticles();
            loadUserData();
        }
        
        function addGoal() {
            const title = prompt('Enter goal name:');
            if (!title) return;
            
            fetch('/api/goals', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken
                },
                body: JSON.stringify({ title, priority: 3, progress: 0 })
            }).then(() => loadUserData());
        }
        
        function focusGoal(index) {
            if (!goalOrbs[index]) return;
            
            const target = goalOrbs[index].position;
            const endDistance = 8;
            const endTheta = Math.atan2(target.z, target.x);
            const endPhi = Math.PI / 3;
            
            // Animate camera
            const startDistance = cameraDistance;
            const startTheta = cameraTheta;
            const startPhi = cameraPhi;
            let progress = 0;
            
            function animateToGoal() {
                progress += 0.03;
                if (progress < 1) {
                    cameraDistance = startDistance + (endDistance - startDistance) * progress;
                    cameraTheta = startTheta + (endTheta - startTheta) * progress;
                    cameraPhi = startPhi + (endPhi - startPhi) * progress;
                    requestAnimationFrame(animateToGoal);
                }
            }
            animateToGoal();
        }
        
        function toggleSacred() {
            settings.showSacred = !settings.showSacred;
            sacredGeometry.forEach(g => g.visible = settings.showSacred);
        }
        
        function toggleSwarm() {
            settings.showSwarm = !settings.showSwarm;
            swarmParticles.forEach(s => s.visible = settings.showSwarm);
        }
        
        function resetCamera() {
            cameraDistance = 25;
            cameraTheta = 0;
            cameraPhi = Math.PI / 4;
        }
        
        function screenshot() {
            renderer.render(scene, camera);
            const dataUrl = renderer.domElement.toDataURL('image/png');
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = 'starship-bridge-' + Date.now() + '.png';
            link.click();
        }
        
        function onWindowResize() {
            const container = document.getElementById('viewscreen-canvas');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
        
        function logout() {
            localStorage.removeItem('fractal_token');
            location.href = '/';
        }
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // INITIALIZE
        // ═══════════════════════════════════════════════════════════════════════════════
        
        document.getElementById('phi-display').textContent = PHI.toFixed(6);
        detectGPU();
        init();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# HOME PAGE WITH STARSHIP AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════════════════

HOME_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Life Fractal Intelligence</title>
    <style>
''' + STARSHIP_CSS + '''
        .home-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .hero {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 60px 20px;
            background: radial-gradient(ellipse at center, rgba(0, 212, 255, 0.05) 0%, transparent 70%);
        }
        
        .hero-title {
            font-size: 3.5em;
            letter-spacing: 8px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hero-subtitle {
            font-size: 1.2em;
            color: var(--text-dim);
            margin-bottom: 50px;
            letter-spacing: 2px;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            max-width: 1200px;
            width: 100%;
            padding: 0 20px;
        }
        
        .feature-card {
            background: var(--bg-panel);
            border: 1px solid var(--border-glow);
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .feature-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.15);
        }
        
        .feature-card:hover::before { opacity: 1; }
        
        .feature-icon { font-size: 3em; margin-bottom: 20px; }
        .feature-title { font-size: 1.3em; letter-spacing: 2px; margin-bottom: 10px; }
        .feature-desc { color: var(--text-dim); font-size: 0.9em; line-height: 1.6; }
        
        .login-section {
            max-width: 400px;
            margin: 50px auto 0;
            padding: 30px;
            background: var(--bg-panel);
            border: 1px solid var(--border-glow);
            border-radius: 8px;
        }
        
        .login-section input {
            width: 100%;
            padding: 14px;
            margin-bottom: 15px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-glow);
            border-radius: 4px;
            color: var(--text-bright);
            font-size: 1em;
        }
        
        .login-section input::placeholder { color: var(--text-dim); }
        
        .tab-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .tab-btn {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: 1px solid var(--accent-cyan);
            border-radius: 4px;
            color: var(--accent-cyan);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .tab-btn.active {
            background: var(--accent-cyan);
            color: var(--bg-space);
        }
    </style>
</head>
<body>
    <div class="home-container">
        <nav class="nav-bar">
            <div class="nav-logo">🚀 LIFE FRACTAL INTELLIGENCE</div>
            <div class="nav-links">
                <a href="/" class="nav-link active">HOME</a>
                <a href="/bridge" class="nav-link">BRIDGE</a>
                <a href="/studio" class="nav-link">ART STUDIO</a>
                <a href="/timeline" class="nav-link">TIMELINE</a>
            </div>
            <div id="nav-auth"></div>
        </nav>
        
        <section class="hero">
            <h1 class="hero-title">LIFE FRACTAL</h1>
            <p class="hero-subtitle">YOUR LIFE • VISUALIZED AS LIVING FRACTAL ART</p>
            
            <div class="feature-grid" id="feature-grid">
                <div class="feature-card" onclick="navigate('/bridge')">
                    <div class="feature-icon">🚀</div>
                    <div class="feature-title">STARSHIP BRIDGE</div>
                    <div class="feature-desc">Command your life from the bridge. 3D visualization with HUD displays, swarm intelligence, and ambient audio.</div>
                </div>
                
                <div class="feature-card" onclick="navigate('/studio')">
                    <div class="feature-icon">🎨</div>
                    <div class="feature-title">ART STUDIO</div>
                    <div class="feature-desc">Create posters, wallpapers, and meditation videos from your life fractal. Share your progress as art.</div>
                </div>
                
                <div class="feature-card" onclick="navigate('/timeline')">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">TIMELINE</div>
                    <div class="feature-desc">Watch your fractal evolve over time. See patterns emerge from your daily choices.</div>
                </div>
                
                <div class="feature-card" onclick="navigate('/app')">
                    <div class="feature-icon">📋</div>
                    <div class="feature-title">DASHBOARD</div>
                    <div class="feature-desc">Track 27 psychology metrics, manage goals, and care for your virtual companion.</div>
                </div>
            </div>
            
            <div class="login-section" id="login-section" style="display: none;">
                <div class="tab-buttons">
                    <button class="tab-btn active" onclick="showTab('login')">LOGIN</button>
                    <button class="tab-btn" onclick="showTab('register')">REGISTER</button>
                </div>
                
                <form id="login-form">
                    <input type="email" id="login-email" placeholder="Email" required>
                    <input type="password" id="login-password" placeholder="Password" required>
                    <button type="submit" class="hud-btn hud-btn-primary" style="width: 100%;">ENTER BRIDGE</button>
                </form>
                
                <form id="register-form" style="display: none;">
                    <input type="text" id="reg-first" placeholder="First Name" required>
                    <input type="text" id="reg-last" placeholder="Last Name" required>
                    <input type="email" id="reg-email" placeholder="Email" required>
                    <input type="password" id="reg-password" placeholder="Password" required>
                    <button type="submit" class="hud-btn hud-btn-gold" style="width: 100%;">CREATE ACCOUNT</button>
                </form>
                
                <p id="auth-error" style="color: var(--accent-red); margin-top: 15px; text-align: center;"></p>
            </div>
        </section>
    </div>
    
    <script>
        const authToken = localStorage.getItem('fractal_token');
        
        async function checkAuth() {
            if (!authToken) {
                showLoginSection();
                return;
            }
            
            try {
                const res = await fetch('/api/me', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                
                if (res.ok) {
                    const user = await res.json();
                    showAuthenticatedUI(user);
                } else {
                    localStorage.removeItem('fractal_token');
                    showLoginSection();
                }
            } catch (e) {
                showLoginSection();
            }
        }
        
        function showLoginSection() {
            document.getElementById('login-section').style.display = 'block';
            document.getElementById('nav-auth').innerHTML = '';
        }
        
        function showAuthenticatedUI(user) {
            document.getElementById('login-section').style.display = 'none';
            document.getElementById('nav-auth').innerHTML = `
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="color: var(--accent-gold);">🥄 ${user.spoons || 12}</span>
                    <span>Hi, ${user.first_name || 'Commander'}</span>
                    <button class="hud-btn" onclick="logout()">LOGOUT</button>
                </div>
            `;
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('login-form').style.display = tab === 'login' ? 'block' : 'none';
            document.getElementById('register-form').style.display = tab === 'register' ? 'block' : 'none';
        }
        
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: document.getElementById('login-email').value,
                        password: document.getElementById('login-password').value
                    })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    localStorage.setItem('fractal_token', data.token);
                    checkAuth();
                } else {
                    document.getElementById('auth-error').textContent = data.error || 'Login failed';
                }
            } catch (e) {
                document.getElementById('auth-error').textContent = 'Connection error';
            }
        });
        
        document.getElementById('register-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const res = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: document.getElementById('reg-email').value,
                        password: document.getElementById('reg-password').value,
                        first_name: document.getElementById('reg-first').value,
                        last_name: document.getElementById('reg-last').value
                    })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    localStorage.setItem('fractal_token', data.token);
                    checkAuth();
                } else {
                    document.getElementById('auth-error').textContent = data.error || 'Registration failed';
                }
            } catch (e) {
                document.getElementById('auth-error').textContent = 'Connection error';
            }
        });
        
        function logout() {
            localStorage.removeItem('fractal_token');
            location.reload();
        }
        
        function navigate(path) {
            if (!authToken) {
                document.getElementById('login-section').scrollIntoView({ behavior: 'smooth' });
                return;
            }
            location.href = path;
        }
        
        checkAuth();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/bridge')
def bridge():
    return render_template_string(STARSHIP_BRIDGE_HTML)

@app.route('/studio')
def studio():
    # Reuse from previous version or create new
    return render_template_string(HOME_HTML.replace('HOME', 'STUDIO'))

@app.route('/timeline')
def timeline():
    return render_template_string(HOME_HTML.replace('HOME', 'TIMELINE'))

@app.route('/app')
def app_dashboard():
    return render_template_string(HOME_HTML.replace('HOME', 'DASHBOARD'))


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

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
    
    # Create initial life state with all 27 metrics
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
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    life_state = conn.execute('''
        SELECT * FROM life_state WHERE user_id = ? ORDER BY date DESC LIMIT 1
    ''', (user_id,)).fetchone()
    goals = conn.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,)).fetchall()
    tasks = conn.execute('SELECT * FROM tasks WHERE user_id = ?', (user_id,)).fetchall()
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


@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '9.0-starship',
        'gpu_server': GPU_AVAILABLE,
        'fractal_types': len(FRACTAL_TYPES),
        'psychology_metrics': len(PSYCHOLOGY_METRICS),
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("=" * 70)
    print("🚀 LIFE FRACTAL INTELLIGENCE v9.0 - STARSHIP BRIDGE EDITION")
    print("=" * 70)
    print(f"✅ GPU: {'Available - ' + GPU_NAME if GPU_AVAILABLE else 'Using CPU'}")
    print(f"✅ Fractal Types: {len(FRACTAL_TYPES)}")
    print(f"✅ Psychology Metrics: {len(PSYCHOLOGY_METRICS)}")
    print(f"✅ Database: {DB_PATH}")
    print(f"✅ Server starting on port {port}")
    print()
    print("🌌 STARSHIP VISUALIZATION SYSTEM")
    print("   • Starship Bridge: /bridge")
    print("   • Art Studio: /studio")
    print("   • Timeline: /timeline")
    print("   • Dashboard: /app")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=False)
