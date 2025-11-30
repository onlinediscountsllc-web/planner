#!/usr/bin/env python3
"""
LIFE FRACTAL INTELLIGENCE - Neurodivergent-Optimized Edition
Swedish minimalism + Autism-safe + Aphantasia-first + ADHD-friendly
"""

import os
import json
import math
import secrets
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from io import BytesIO
import base64

from flask import Flask, request, jsonify, render_template_string, g, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
from PIL import Image, ImageDraw

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATABASE = os.environ.get('DATABASE_PATH', '/tmp/life_fractal.db')
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Sacred Mathematics
PHI = (1 + math.sqrt(5)) / 2
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Autism-Safe Color Palette (Swedish minimalism + high contrast)
COLORS = {
    'primary': '#4A6FA5',      # Calm blue
    'secondary': '#7C9CB8',    # Soft blue-gray
    'background': '#F5F7FA',   # Very light gray
    'surface': '#FFFFFF',      # Pure white
    'text': '#2C3E50',         # Dark blue-gray
    'text_secondary': '#6C757D', # Medium gray
    'success': '#52A675',      # Muted green
    'warning': '#D4A574',      # Warm tan
    'error': '#C45C5C',        # Muted red
    'border': '#DEE2E6',       # Light gray
    'focus': '#5A8DC7'         # Bright blue for focus
}

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app)

# Database Functions
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            created_at TEXT NOT NULL,
            break_interval INTEGER DEFAULT 25,
            sound_enabled BOOLEAN DEFAULT 0,
            high_contrast BOOLEAN DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            progress REAL DEFAULT 0.0,
            target_date TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS habits (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            frequency TEXT DEFAULT 'daily',
            current_streak INTEGER DEFAULT 0,
            best_streak INTEGER DEFAULT 0,
            last_completed TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_entries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            mood_level INTEGER DEFAULT 50,
            energy_level INTEGER DEFAULT 50,
            stress_level INTEGER DEFAULT 50,
            notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pet_state (
            user_id TEXT PRIMARY KEY,
            species TEXT DEFAULT 'companion',
            name TEXT DEFAULT 'Friend',
            level INTEGER DEFAULT 1,
            experience INTEGER DEFAULT 0,
            last_interaction TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS break_reminders (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            reminder_time TEXT NOT NULL,
            completed BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    db.commit()

@app.before_request
def before_request():
    if not hasattr(g, 'db_initialized'):
        init_db()
        g.db_initialized = True

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Fractal Generation (GPU-free, optimized for accessibility)
def generate_fractal(mood=50, energy=50, stress=50):
    """Generate calm, predictable fractal based on wellness metrics"""
    width, height = 800, 800
    max_iter = 50
    
    # Calculate zoom and center based on metrics (smooth, predictable)
    zoom = 1.0 + (energy / 200)
    center_x = -0.5 + (stress - 50) / 300
    center_y = (mood - 50) / 300
    
    x = np.linspace(-2/zoom + center_x, 2/zoom + center_x, width)
    y = np.linspace(-1.5/zoom + center_y, 1.5/zoom + center_y, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    z = np.zeros_like(c)
    iterations = np.zeros((height, width))
    
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        iterations[mask] = i
    
    # Autism-safe color mapping (no flashing, predictable gradients)
    normalized = iterations / max_iter
    
    # Use calming blues and greens
    r = (normalized * 100 + 100).astype(np.uint8)
    g = (normalized * 150 + 100).astype(np.uint8)
    b = (normalized * 200 + 50).astype(np.uint8)
    
    rgb = np.dstack([r, g, b])
    image = Image.fromarray(rgb, 'RGB')
    
    # Add sacred geometry overlays (visible but not overwhelming)
    draw = ImageDraw.Draw(image, 'RGBA')
    cx, cy = width // 2, height // 2
    
    for i, fib in enumerate(FIBONACCI[:8]):
        radius = int(fib * 3)
        alpha = 30 + i * 10  # Gradual transparency
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(255, 255, 255, alpha),
            width=2
        )
    
    return image

# HTML Templates
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence - Neurodivergent-Friendly</title>
    <style>
        /* Swedish Minimalism + Autism-Safe Design */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: {{ COLORS.primary }};
            --secondary: {{ COLORS.secondary }};
            --background: {{ COLORS.background }};
            --surface: {{ COLORS.surface }};
            --text: {{ COLORS.text }};
            --text-secondary: {{ COLORS.text_secondary }};
            --success: {{ COLORS.success }};
            --warning: {{ COLORS.warning }};
            --error: {{ COLORS.error }};
            --border: {{ COLORS.border }};
            --focus: {{ COLORS.focus }};
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
            font-size: 16px;
        }
        
        /* High contrast mode support */
        body.high-contrast {
            --text: #000000;
            --background: #FFFFFF;
            --border: #000000;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header - Clean and simple */
        header {
            background: var(--surface);
            border-bottom: 2px solid var(--border);
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 28px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 5px;
        }
        
        .subtitle {
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 400;
        }
        
        /* Cards - Clear boundaries, no shadows */
        .card {
            background: var(--surface);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text);
        }
        
        /* Forms - Large, clear inputs */
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            font-size: 14px;
            color: var(--text);
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--text);
            transition: border-color 0.2s ease;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--focus);
            box-shadow: 0 0 0 3px rgba(90, 141, 199, 0.1);
        }
        
        /* Buttons - Clear, high contrast */
        button, .btn {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 500;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-block;
            text-decoration: none;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--focus);
        }
        
        .btn-secondary {
            background: var(--secondary);
            color: white;
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-warning {
            background: var(--warning);
            color: white;
        }
        
        /* Break Reminder - Non-intrusive but clear */
        .break-reminder {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--warning);
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            border: 2px solid var(--text);
            display: none;
            z-index: 1000;
        }
        
        .break-reminder.show {
            display: block;
        }
        
        /* Progress bars - Clear visual feedback */
        .progress-bar {
            width: 100%;
            height: 32px;
            background: var(--background);
            border: 2px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            margin: 12px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--success);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }
        
        /* Stats - Clear, grid layout */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: var(--surface);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        /* Fractal display - Optional, can be hidden */
        .fractal-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .fractal-container img {
            max-width: 100%;
            height: auto;
            border: 2px solid var(--border);
            border-radius: 8px;
        }
        
        .hidden {
            display: none;
        }
        
        /* Accessibility helpers */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--primary);
            color: white;
            padding: 8px;
            text-decoration: none;
        }
        
        .skip-link:focus {
            top: 0;
        }
        
        /* Timer display */
        .timer-display {
            font-size: 48px;
            font-weight: 700;
            text-align: center;
            color: var(--primary);
            margin: 20px 0;
            font-variant-numeric: tabular-nums;
        }
        
        /* Settings toggle */
        .toggle {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 12px 0;
        }
        
        .toggle input[type="checkbox"] {
            width: 48px;
            height: 24px;
        }
    </style>
</head>
<body>
    <a href="#main" class="skip-link">Skip to main content</a>
    
    <header>
        <div class="container">
            <h1>Life Fractal Intelligence</h1>
            <p class="subtitle">Neurodivergent-optimized life planning</p>
        </div>
    </header>
    
    <main id="main" class="container">
        {% if not logged_in %}
        <div class="card">
            <h2 class="card-title">Welcome</h2>
            <p style="margin-bottom: 20px;">A calm, predictable space for planning your life.</p>
            
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" placeholder="your@email.com" required>
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" placeholder="Enter secure password" required>
            </div>
            
            <div class="form-group">
                <label for="name">Name (optional)</label>
                <input type="text" id="name" placeholder="What should we call you?">
            </div>
            
            <button onclick="register()" class="btn btn-primary">Create Account</button>
            <button onclick="login()" class="btn btn-secondary" style="margin-left: 12px;">Login</button>
            
            <div id="message" style="margin-top: 20px;"></div>
        </div>
        {% else %}
        <div class="break-reminder" id="breakReminder">
            <p><strong>Time for a break</strong></p>
            <p>You've been working for {{ break_interval }} minutes.</p>
            <button onclick="dismissBreak()" class="btn btn-primary" style="margin-top: 12px;">Take Break (5 min)</button>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="goalCount">0</div>
                <div class="stat-label">Active Goals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="habitStreak">0</div>
                <div class="stat-label">Longest Streak</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="petLevel">1</div>
                <div class="stat-label">Companion Level</div>
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Today's Check-In</h2>
            
            <div class="form-group">
                <label for="mood">How is your mood? (1-100)</label>
                <input type="range" id="mood" min="1" max="100" value="50" oninput="updateSlider('mood', this.value)">
                <div id="moodValue" style="text-align: center; font-weight: 600;">50</div>
            </div>
            
            <div class="form-group">
                <label for="energy">Energy level? (1-100)</label>
                <input type="range" id="energy" min="1" max="100" value="50" oninput="updateSlider('energy', this.value)">
                <div id="energyValue" style="text-align: center; font-weight: 600;">50</div>
            </div>
            
            <div class="form-group">
                <label for="stress">Stress level? (1-100)</label>
                <input type="range" id="stress" min="1" max="100" value="50" oninput="updateSlider('stress', this.value)">
                <div id="stressValue" style="text-align: center; font-weight: 600;">50</div>
            </div>
            
            <div class="form-group">
                <label for="notes">Notes (optional - voice input supported)</label>
                <textarea id="notes" rows="4" placeholder="How are you feeling today?"></textarea>
            </div>
            
            <button onclick="saveDaily()" class="btn btn-success">Save Check-In</button>
        </div>
        
        <div class="card">
            <h2 class="card-title">Your Goals</h2>
            <div id="goalsList"></div>
            <button onclick="showAddGoal()" class="btn btn-primary">Add Goal</button>
        </div>
        
        <div class="card">
            <h2 class="card-title">Your Habits</h2>
            <div id="habitsList"></div>
            <button onclick="showAddHabit()" class="btn btn-primary">Add Habit</button>
        </div>
        
        <div class="card">
            <h2 class="card-title">Fractal Visualization</h2>
            <p style="margin-bottom: 16px; color: var(--text-secondary);">Optional: See your wellness as mathematical art</p>
            <button onclick="toggleFractal()" class="btn btn-secondary" id="fractalToggle">Show Visualization</button>
            <div id="fractalContainer" class="fractal-container hidden">
                <img id="fractalImage" src="" alt="Your personalized fractal">
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Settings</h2>
            
            <div class="toggle">
                <input type="checkbox" id="breakReminders" onchange="toggleBreaks()">
                <label for="breakReminders">Break reminders (every 25 minutes)</label>
            </div>
            
            <div class="toggle">
                <input type="checkbox" id="soundEnabled" onchange="toggleSound()">
                <label for="soundEnabled">Adaptive soundscape</label>
            </div>
            
            <div class="toggle">
                <input type="checkbox" id="highContrast" onchange="toggleContrast()">
                <label for="highContrast">High contrast mode</label>
            </div>
            
            <button onclick="logout()" class="btn btn-warning" style="margin-top: 20px;">Logout</button>
        </div>
        {% endif %}
    </main>
    
    <script>
        let userId = {{ user_id | tojson }};
        let breakTimer = null;
        let fractalVisible = false;
        
        function updateSlider(name, value) {
            document.getElementById(name + 'Value').textContent = value;
        }
        
        async function register() {
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const name = document.getElementById('name').value;
            
            try {
                const res = await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password, name})
                });
                const data = await res.json();
                
                if (data.success) {
                    location.reload();
                } else {
                    document.getElementById('message').innerHTML = 
                        '<p style="color: var(--error);">' + data.error + '</p>';
                }
            } catch (e) {
                document.getElementById('message').innerHTML = 
                    '<p style="color: var(--error);">Error: ' + e.message + '</p>';
            }
        }
        
        async function login() {
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if (data.success) {
                    location.reload();
                } else {
                    document.getElementById('message').innerHTML = 
                        '<p style="color: var(--error);">' + data.error + '</p>';
                }
            } catch (e) {
                document.getElementById('message').innerHTML = 
                    '<p style="color: var(--error);">Error: ' + e.message + '</p>';
            }
        }
        
        async function saveDaily() {
            const mood = document.getElementById('mood').value;
            const energy = document.getElementById('energy').value;
            const stress = document.getElementById('stress').value;
            const notes = document.getElementById('notes').value;
            
            try {
                const res = await fetch('/api/daily', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        mood_level: mood,
                        energy_level: energy,
                        stress_level: stress,
                        notes: notes
                    })
                });
                const data = await res.json();
                
                if (data.success) {
                    alert('Check-in saved successfully!');
                    loadStats();
                }
            } catch (e) {
                alert('Error saving: ' + e.message);
            }
        }
        
        async function toggleFractal() {
            if (fractalVisible) {
                document.getElementById('fractalContainer').classList.add('hidden');
                document.getElementById('fractalToggle').textContent = 'Show Visualization';
                fractalVisible = false;
            } else {
                const mood = document.getElementById('mood').value;
                const energy = document.getElementById('energy').value;
                const stress = document.getElementById('stress').value;
                
                try {
                    const res = await fetch(`/api/fractal?user_id=${userId}&mood=${mood}&energy=${energy}&stress=${stress}`);
                    const data = await res.json();
                    
                    document.getElementById('fractalImage').src = data.image;
                    document.getElementById('fractalContainer').classList.remove('hidden');
                    document.getElementById('fractalToggle').textContent = 'Hide Visualization';
                    fractalVisible = true;
                } catch (e) {
                    alert('Error loading fractal: ' + e.message);
                }
            }
        }
        
        function toggleBreaks() {
            const enabled = document.getElementById('breakReminders').checked;
            if (enabled) {
                startBreakTimer();
            } else {
                if (breakTimer) clearInterval(breakTimer);
            }
        }
        
        function startBreakTimer() {
            breakTimer = setInterval(() => {
                document.getElementById('breakReminder').classList.add('show');
            }, 25 * 60 * 1000); // 25 minutes
        }
        
        function dismissBreak() {
            document.getElementById('breakReminder').classList.remove('show');
        }
        
        function toggleSound() {
            // Implement adaptive soundscape
            const enabled = document.getElementById('soundEnabled').checked;
            console.log('Sound enabled:', enabled);
        }
        
        function toggleContrast() {
            const enabled = document.getElementById('highContrast').checked;
            if (enabled) {
                document.body.classList.add('high-contrast');
            } else {
                document.body.classList.remove('high-contrast');
            }
        }
        
        async function loadStats() {
            // Load user stats
            try {
                const res = await fetch(`/api/stats?user_id=${userId}`);
                const data = await res.json();
                
                document.getElementById('goalCount').textContent = data.goals || 0;
                document.getElementById('habitStreak').textContent = data.best_streak || 0;
                document.getElementById('petLevel').textContent = data.pet_level || 1;
            } catch (e) {
                console.error('Error loading stats:', e);
            }
        }
        
        function logout() {
            location.href = '/logout';
        }
        
        // Load data on page load
        if (userId) {
            loadStats();
        }
    </script>
</body>
</html>
'''

# Routes
@app.route('/')
def home():
    user_id = session.get('user_id')
    logged_in = user_id is not None
    break_interval = session.get('break_interval', 25)
    
    return render_template_string(
        MAIN_TEMPLATE,
        COLORS=COLORS,
        logged_in=logged_in,
        user_id=user_id,
        break_interval=break_interval
    )

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email already registered'}), 400
        
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute('''
            INSERT INTO users (id, email, password_hash, name, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, email, generate_password_hash(password), name or email.split('@')[0], now))
        
        cursor.execute('''
            INSERT INTO pet_state (user_id, name)
            VALUES (?, ?)
        ''', (user_id, 'Friend'))
        
        db.commit()
        
        session['user_id'] = user_id
        session.permanent = True
        
        return jsonify({'success': True, 'user_id': user_id}), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['user_id'] = user['id']
        session.permanent = True
        
        return jsonify({'success': True, 'user_id': user['id']}), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/api/daily', methods=['POST'])
def daily():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        db = get_db()
        cursor = db.cursor()
        today = datetime.now(timezone.utc).date().isoformat()
        
        entry_id = f"entry_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_entries 
            (id, user_id, date, mood_level, energy_level, stress_level, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_id, user_id, today, 
              data.get('mood_level', 50),
              data.get('energy_level', 50),
              data.get('stress_level', 50),
              data.get('notes', ''),
              now))
        
        db.commit()
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f"Daily entry error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fractal')
def fractal():
    user_id = request.args.get('user_id')
    mood = int(request.args.get('mood', 50))
    energy = int(request.args.get('energy', 50))
    stress = int(request.args.get('stress', 50))
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    image = generate_fractal(mood, energy, stress)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_str}',
        'mood': mood,
        'energy': energy,
        'stress': stress
    }), 200

@app.route('/api/stats')
def stats():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM goals WHERE user_id = ?", (user_id,))
    goals = cursor.fetchone()['count']
    
    cursor.execute("SELECT MAX(best_streak) as streak FROM habits WHERE user_id = ?", (user_id,))
    streak_row = cursor.fetchone()
    best_streak = streak_row['streak'] if streak_row and streak_row['streak'] else 0
    
    cursor.execute("SELECT level FROM pet_state WHERE user_id = ?", (user_id,))
    pet_row = cursor.fetchone()
    pet_level = pet_row['level'] if pet_row else 1
    
    return jsonify({
        'goals': goals,
        'best_streak': best_streak,
        'pet_level': pet_level
    }), 200

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'neurodivergent-optimized'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("Life Fractal Intelligence - Neurodivergent Edition starting...")
    app.run(host='0.0.0.0', port=port, debug=False)
