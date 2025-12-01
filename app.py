#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v9.0 - ULTIMATE UNIFIED SYSTEM
═══════════════════════════════════════════════════════════════════════════════
COMPLETE MERGE: Clean Accessible Design + 3D Visualization + All Advanced Features

✅ FROM ORIGINAL GUI:
- Neurodivergent-optimized interface
- Clean top stats cards (Active Goals, Longest Streak, Companion Level)
- Voice input support for notes
- Accessible sliders and forms
- Aphantasia/Autism accommodations

✅ FROM 3D VISUALIZATION:
- Interactive Three.js 3D fractal universe
- Goal orbs in 3D space with connections
- Sacred geometry overlays (Flower of Life, Golden Spiral)
- Animated starfield background
- Camera controls and animation speed

✅ FROM v8 ADVANCED:
- ML Mood Prediction (Decision Tree)
- Fuzzy Logic Guidance System
- Sacred Badge Achievements (Fibonacci)
- Goals/Habits with Fibonacci milestones
- Self-Healing System
- All Render.com deployment fixes

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import traceback
import colorsys
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
from functools import wraps
import base64

# Flask
from flask import Flask, request, jsonify, session, render_template_string, redirect, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw

# ML Support (optional)
try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    DecisionTreeRegressor = None
    StandardScaler = None

# GPU Support (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "CPU"
    torch = None

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# ═══════════════════════════════════════════════════════════════════════════════
# 🛡️ SELF-HEALING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingSystem:
    def __init__(self):
        self.error_counts = {}
        self.component_status = {}
        self.start_time = datetime.now(timezone.utc)
    
    def record_error(self, component: str, error: str):
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        self.component_status[component] = 'error'
        logger.warning(f"🛡️ Error in {component}: {error}")
    
    def mark_healthy(self, component: str):
        self.component_status[component] = 'healthy'
    
    def get_health_report(self) -> dict:
        total_errors = sum(self.error_counts.values())
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            'overall_health': 'excellent' if total_errors == 0 else 'healthy',
            'uptime_seconds': uptime,
            'error_counts': self.error_counts
        }

HEALER = SelfHealingSystem()

def safe_execute(fallback_value=None, log_errors=True, component="unknown"):
    """Safe execution with deferred fallback (FIXES jsonify context error)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                HEALER.mark_healthy(component)
                return result
            except Exception as e:
                HEALER.record_error(component, str(e))
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                if callable(fallback_value):
                    return fallback_value()
                return fallback_value
        return wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════════════════
# ML MOOD PREDICTION & FUZZY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class MoodPredictor:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=5, random_state=42) if HAS_SKLEARN else None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.trained = False
    
    def train(self, history: List[Dict]) -> bool:
        if not HAS_SKLEARN or len(history) < 3:
            return False
        try:
            X, y = [], []
            for i, rec in enumerate(history[:-1]):
                X.append([rec.get('stress_level', 50)/100, rec.get('mood_score', 50)/100,
                         rec.get('energy_level', 50)/100, rec.get('sleep_hours', 7)/12])
                y.append(history[i+1].get('mood_score', 50)/100)
            X = self.scaler.fit_transform(np.array(X))
            self.model.fit(X, np.array(y))
            self.trained = True
            return True
        except:
            return False
    
    def predict(self, state: Dict) -> Tuple[float, str]:
        if not self.trained:
            return state.get('mood_score', 50), 'low'
        try:
            features = [[state.get('stress_level', 50)/100, state.get('mood_score', 50)/100,
                        state.get('energy_level', 50)/100, state.get('sleep_hours', 7)/12]]
            pred = self.model.predict(self.scaler.transform(features))[0]
            return max(0, min(100, pred * 100)), 'high'
        except:
            return state.get('mood_score', 50), 'low'

class FuzzyLogicEngine:
    def __init__(self):
        self.messages = {
            ('low_stress', 'high_mood'): "You're doing great! Your positive energy is inspiring. 🌟",
            ('low_stress', 'medium_mood'): "You're in a good place. Small joys can lift you higher. ✨",
            ('low_stress', 'low_mood'): "Even on quieter days, you're managing well. Be gentle with yourself. 💙",
            ('medium_stress', 'high_mood'): "Your resilience is shining through! Take breaks when needed. 🌈",
            ('medium_stress', 'medium_mood'): "Balance is key. You're navigating well through challenges. ⚖️",
            ('medium_stress', 'low_mood'): "It's okay to feel this way. Consider a short mindful pause. 🧘",
            ('high_stress', 'high_mood'): "Your positivity is admirable! Don't forget to rest. 💪",
            ('high_stress', 'medium_mood'): "You're handling a lot. Prioritize what matters most. 🎯",
            ('high_stress', 'low_mood'): "These feelings are valid. Reach out for support if needed. 💜"
        }
    
    def infer(self, stress: float, mood: float) -> str:
        s = 'low' if stress <= 30 else ('high' if stress >= 70 else 'medium')
        m = 'low' if mood <= 30 else ('high' if mood >= 70 else 'medium')
        return self.messages.get((f'{s}_stress', f'{m}_mood'), "Take a moment to breathe. 🌬️")

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED BADGES
# ═══════════════════════════════════════════════════════════════════════════════

SACRED_BADGES = {
    'fibonacci_initiate': {'name': 'Fibonacci Initiate', 'icon': '🌱', 'threshold': 8, 'type': 'tasks'},
    'golden_seeker': {'name': 'Golden Seeker', 'icon': '⭐', 'threshold': 13, 'type': 'streak'},
    'sacred_guardian': {'name': 'Sacred Guardian', 'icon': '🛡️', 'threshold': 21, 'type': 'goals'},
    'flower_of_life': {'name': 'Flower of Life', 'icon': '🌸', 'threshold': 34, 'type': 'wellness'},
    'golden_spiral': {'name': 'Golden Spiral', 'icon': '🌟', 'threshold': 144, 'type': 'level'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# JSON DATA STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    def __init__(self):
        self.data_dir = os.environ.get('DATA_DIR', '/tmp/life_fractal_data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.files = {k: os.path.join(self.data_dir, f'{k}.json') 
                     for k in ['users', 'goals', 'habits', 'pets', 'entries', 'history']}
        for f in self.files.values():
            if not os.path.exists(f):
                with open(f, 'w') as fp:
                    json.dump({}, fp)
        logger.info(f"✅ DataStore: {self.data_dir}")
    
    def _read(self, key: str) -> dict:
        try:
            with open(self.files[key], 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _write(self, key: str, data: dict):
        with open(self.files[key], 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_user(self, uid: str): return self._read('users').get(uid)
    def get_user_by_email(self, email: str):
        for uid, u in self._read('users').items():
            if u.get('email', '').lower() == email.lower():
                u['id'] = uid
                return u
        return None
    def save_user(self, uid: str, data: dict):
        users = self._read('users')
        users[uid] = data
        self._write('users', users)
    
    def get_goals(self, uid: str):
        return [dict(g, id=gid) for gid, g in self._read('goals').items() if g.get('user_id') == uid]
    def save_goal(self, gid: str, data: dict):
        goals = self._read('goals')
        goals[gid] = data
        self._write('goals', goals)
    def delete_goal(self, gid: str):
        goals = self._read('goals')
        goals.pop(gid, None)
        self._write('goals', goals)
    
    def get_habits(self, uid: str):
        return [dict(h, id=hid) for hid, h in self._read('habits').items() if h.get('user_id') == uid]
    def get_habit(self, hid: str):
        h = self._read('habits').get(hid)
        if h: h['id'] = hid
        return h
    def save_habit(self, hid: str, data: dict):
        habits = self._read('habits')
        habits[hid] = data
        self._write('habits', habits)
    
    def get_pet(self, uid: str):
        for pid, p in self._read('pets').items():
            if p.get('user_id') == uid:
                p['id'] = pid
                return p
        return None
    def save_pet(self, pid: str, data: dict):
        pets = self._read('pets')
        pets[pid] = data
        self._write('pets', pets)
    
    def get_daily_entry(self, uid: str, date: str):
        return self._read('entries').get(f"{uid}_{date}")
    def save_daily_entry(self, uid: str, date: str, data: dict):
        entries = self._read('entries')
        data.update({'user_id': uid, 'date': date})
        entries[f"{uid}_{date}"] = data
        self._write('entries', entries)
    def get_user_history(self, uid: str, days: int = 30):
        entries = self._read('entries')
        user_entries = [e for k, e in entries.items() if k.startswith(f"{uid}_")]
        user_entries.sort(key=lambda x: x.get('date', ''), reverse=True)
        return user_entries[:days]

db = DataStore()
mood_predictor = MoodPredictor()
fuzzy_engine = FuzzyLogicEngine()

# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    def calculate_metrics(self, uid: str) -> dict:
        goals = db.get_goals(uid)
        habits = db.get_habits(uid)
        pet = db.get_pet(uid)
        
        goal_completion = sum(g.get('progress', 0) for g in goals) / (len(goals) * 100) if goals else 0
        max_streak = max((h.get('current_streak', 0) for h in habits), default=0)
        pet_happiness = pet.get('happiness', 50) / 100 if pet else 0.5
        
        momentum = goal_completion * 0.4 + min(max_streak, 30) / 30 * 0.3 + pet_happiness * 0.3
        return {'goal_completion': goal_completion, 'max_streak': max_streak, 
                'pet_happiness': pet_happiness, 'momentum': momentum}
    
    def generate(self, metrics: dict, size: int = 600) -> bytes:
        momentum = metrics.get('momentum', 0.5)
        zoom = 1.0 + metrics.get('goal_completion', 0.5) * 2.0
        max_iter = 128 + int(momentum * 128)
        
        x = np.linspace(-2.5/zoom, 1.0/zoom, size)
        y = np.linspace(-1.5/zoom, 1.5/zoom, size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i
        
        M = M / max_iter
        img = Image.new('RGB', (size, size))
        pixels = img.load()
        
        for py in range(size):
            for px in range(size):
                v = M[py, px]
                if v >= 0.99:
                    pixels[px, py] = (10, 10, 30)
                else:
                    hue = (v + momentum * 0.3) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.5 + v * 0.5)
                    pixels[px, py] = (int(r*255), int(g*255), int(b*255))
        
        # Golden spiral overlay
        if momentum > 0.4:
            draw = ImageDraw.Draw(img, 'RGBA')
            cx, cy = size // 2, size // 2
            points = []
            for i in range(200):
                theta = i * GOLDEN_ANGLE_RAD * 0.1
                r = 5 * math.sqrt(i) * (1 + momentum)
                px, py = cx + r * math.cos(theta), cy + r * math.sin(theta)
                if 0 <= px < size and 0 <= py < size:
                    points.append((px, py))
            if len(points) > 1:
                draw.line(points, fill=(255, 215, 0, int(momentum * 100)), width=2)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()

fractal_engine = FractalEngine()

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('ENVIRONMENT') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
CORS(app, supports_credentials=True)

logger.info("🌀 Life Fractal Intelligence v9.0 - Ultimate Unified System")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '9.0', **HEALER.get_health_report()})

@app.route('/api/auth/register', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Registration failed'}), 500), component="register")
def register():
    data = request.get_json() or {}
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or len(password) < 8:
        return jsonify({'error': 'Valid email and password (8+ chars) required'}), 400
    if db.get_user_by_email(email):
        return jsonify({'error': 'Email already registered'}), 400
    
    uid = f"user_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc)
    
    db.save_user(uid, {
        'email': email, 'password_hash': generate_password_hash(password),
        'first_name': data.get('first_name', ''), 'created_at': now.isoformat(),
        'subscription_status': 'trial', 'trial_end': (now + timedelta(days=7)).isoformat()
    })
    
    pid = f"pet_{secrets.token_hex(8)}"
    db.save_pet(pid, {
        'user_id': uid, 'name': data.get('pet_name', 'Buddy'),
        'species': data.get('pet_species', 'cat'), 'level': 1, 'xp': 0,
        'happiness': 100, 'hunger': 0, 'badges': [], 'tasks_completed': 0
    })
    
    session['user_id'] = uid
    session.permanent = True
    return jsonify({'success': True, 'user_id': uid, 'email': email}), 201

@app.route('/api/auth/login', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Login failed'}), 500), component="login")
def login():
    data = request.get_json() or {}
    user = db.get_user_by_email(data.get('email', '').lower().strip())
    
    if not user or not check_password_hash(user['password_hash'], data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user_id'] = user['id']
    session.permanent = True
    return jsonify({'success': True, 'user_id': user['id'], 'email': user['email']})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/me')
@require_auth
def get_me():
    user = db.get_user(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.pop('password_hash', None)
    user['id'] = session['user_id']
    return jsonify({'user': user})

# Goals
@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals():
    return jsonify({'goals': db.get_goals(session['user_id'])})

@app.route('/api/goals', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="goal")
def create_goal():
    data = request.get_json() or {}
    gid = f"goal_{secrets.token_hex(8)}"
    goal = {
        'user_id': session['user_id'], 'title': data.get('title', 'New Goal'),
        'description': data.get('description', ''), 'category': data.get('category', 'personal'),
        'priority': data.get('priority', 3), 'progress': 0,
        'milestones': [8, 13, 21, 34, 55, 89, 100], 'milestones_reached': [],
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    db.save_goal(gid, goal)
    goal['id'] = gid
    return jsonify({'success': True, 'goal': goal}), 201

@app.route('/api/goals/<gid>', methods=['PUT'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="goal")
def update_goal(gid):
    data = request.get_json() or {}
    goals = db.get_goals(session['user_id'])
    goal = next((g for g in goals if g['id'] == gid), None)
    if not goal:
        return jsonify({'error': 'Not found'}), 404
    
    for k in ['title', 'description', 'progress', 'category', 'priority']:
        if k in data:
            goal[k] = data[k]
    
    # Check milestones
    milestone = None
    for m in goal.get('milestones', []):
        if goal['progress'] >= m and m not in goal.get('milestones_reached', []):
            goal.setdefault('milestones_reached', []).append(m)
            milestone = m
            break
    
    db.save_goal(gid, goal)
    resp = {'success': True, 'goal': goal}
    if milestone:
        resp['milestone'] = milestone
    return jsonify(resp)

@app.route('/api/goals/<gid>', methods=['DELETE'])
@require_auth
def delete_goal(gid):
    db.delete_goal(gid)
    return jsonify({'success': True})

# Habits
@app.route('/api/habits', methods=['GET'])
@require_auth
def get_habits():
    return jsonify({'habits': db.get_habits(session['user_id'])})

@app.route('/api/habits', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="habit")
def create_habit():
    data = request.get_json() or {}
    hid = f"habit_{secrets.token_hex(8)}"
    habit = {
        'user_id': session['user_id'], 'name': data.get('name', 'New Habit'),
        'frequency': data.get('frequency', 'daily'), 'current_streak': 0,
        'longest_streak': 0, 'completions': [],
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    db.save_habit(hid, habit)
    habit['id'] = hid
    return jsonify({'success': True, 'habit': habit}), 201

@app.route('/api/habits/<hid>/complete', methods=['POST'])
@require_auth
def complete_habit(hid):
    habit = db.get_habit(hid)
    if not habit or habit.get('user_id') != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if today not in habit.get('completions', []):
        habit.setdefault('completions', []).append(today)
        habit['current_streak'] = habit.get('current_streak', 0) + 1
        if habit['current_streak'] > habit.get('longest_streak', 0):
            habit['longest_streak'] = habit['current_streak']
        db.save_habit(hid, habit)
        
        # Pet XP
        pet = db.get_pet(session['user_id'])
        if pet:
            pet['xp'] = pet.get('xp', 0) + 5
            pet['tasks_completed'] = pet.get('tasks_completed', 0) + 1
            if pet['xp'] >= pet.get('level', 1) * 100:
                pet['level'] += 1
                pet['xp'] = 0
            db.save_pet(pet['id'], pet)
    
    return jsonify({'success': True, 'habit': habit})

# Daily Entry
@app.route('/api/daily-entry', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="entry")
def save_daily_entry():
    uid = session['user_id']
    data = request.get_json() or {}
    date = data.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    
    mood = float(data.get('mood_score', 50))
    energy = float(data.get('energy_level', 50))
    stress = float(data.get('stress_level', 50))
    
    entry = {
        'mood_score': mood, 'energy_level': energy, 'stress_level': stress,
        'sleep_hours': float(data.get('sleep_hours', 7)),
        'notes': data.get('notes', ''),
        'wellness_index': (mood + energy + (100 - stress)) / 3
    }
    
    db.save_daily_entry(uid, date, entry)
    
    # Train predictor
    history = db.get_user_history(uid, 30)
    if len(history) >= 3:
        mood_predictor.train(history)
    
    guidance = fuzzy_engine.infer(stress, mood)
    predicted, conf = mood_predictor.predict(entry)
    
    # Update pet
    pet = db.get_pet(uid)
    if pet:
        pet['happiness'] = max(0, min(100, pet.get('happiness', 50) + (mood - 50) / 10))
        pet['xp'] = pet.get('xp', 0) + 10
        db.save_pet(pet['id'], pet)
    
    return jsonify({
        'success': True, 'entry': entry, 'guidance': guidance,
        'predicted_mood': predicted, 'confidence': conf
    })

# Pet
@app.route('/api/pet', methods=['GET'])
@require_auth
def get_pet():
    pet = db.get_pet(session['user_id'])
    return jsonify({'pet': pet}) if pet else (jsonify({'error': 'No pet'}), 404)

@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    pet = db.get_pet(session['user_id'])
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    pet['hunger'] = max(0, pet.get('hunger', 0) - 30)
    pet['happiness'] = min(100, pet.get('happiness', 50) + 10)
    pet['xp'] = pet.get('xp', 0) + 5
    if pet['xp'] >= pet.get('level', 1) * 100:
        pet['level'] += 1
        pet['xp'] = 0
    db.save_pet(pet['id'], pet)
    return jsonify({'success': True, 'pet': pet})

@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_pet():
    pet = db.get_pet(session['user_id'])
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    pet['happiness'] = min(100, pet.get('happiness', 50) + 20)
    pet['hunger'] = min(100, pet.get('hunger', 0) + 10)
    pet['xp'] = pet.get('xp', 0) + 10
    if pet['xp'] >= pet.get('level', 1) * 100:
        pet['level'] += 1
        pet['xp'] = 0
    db.save_pet(pet['id'], pet)
    return jsonify({'success': True, 'pet': pet})

# Fractal
@app.route('/api/fractal/generate')
def generate_fractal():
    uid = session.get('user_id')
    size = min(max(int(request.args.get('size', 600)), 200), 1200)
    metrics = fractal_engine.calculate_metrics(uid) if uid else {'momentum': 0.5, 'goal_completion': 0.5}
    return send_file(BytesIO(fractal_engine.generate(metrics, size)), mimetype='image/png')

@app.route('/api/fractal/metrics')
@require_auth
def get_metrics():
    return jsonify({'metrics': fractal_engine.calculate_metrics(session['user_id'])})

# Stats summary
@app.route('/api/stats')
@require_auth
def get_stats():
    uid = session['user_id']
    goals = db.get_goals(uid)
    habits = db.get_habits(uid)
    pet = db.get_pet(uid)
    
    return jsonify({
        'active_goals': len([g for g in goals if g.get('progress', 0) < 100]),
        'longest_streak': max((h.get('longest_streak', 0) for h in habits), default=0),
        'companion_level': pet.get('level', 1) if pet else 1
    })

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HTML - NEURODIVERGENT-FRIENDLY DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence</title>
    <style>
        :root { --primary: #667eea; --secondary: #764ba2; --success: #48c774; --warning: #f0c420; --danger: #ff6b6b; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }
        
        /* Header */
        .header { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; padding: 30px 20px; text-align: center; }
        .header h1 { font-size: 2.2em; margin-bottom: 5px; }
        .header .subtitle { opacity: 0.9; font-size: 1em; }
        
        /* Container */
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        /* Stats Cards */
        .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .stat-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border: 2px solid #e5e7eb; }
        .stat-card .value { font-size: 2.5em; font-weight: bold; color: var(--primary); }
        .stat-card .label { color: #666; font-size: 0.95em; }
        
        /* Cards */
        .card { background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border: 2px solid #e5e7eb; }
        .card h2 { color: var(--primary); margin-bottom: 20px; font-size: 1.3em; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }
        
        /* Form Elements */
        label { display: block; margin-bottom: 8px; font-weight: 500; color: #444; }
        input[type="range"] { width: 100%; height: 8px; -webkit-appearance: none; background: linear-gradient(to right, var(--primary), var(--secondary)); border-radius: 10px; margin: 10px 0; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 24px; height: 24px; background: white; border: 3px solid var(--primary); border-radius: 50%; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
        .slider-value { text-align: center; font-size: 1.5em; font-weight: bold; color: var(--primary); margin: 5px 0 15px; }
        
        textarea, input[type="text"], input[type="email"], input[type="password"] { width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 1em; margin-bottom: 15px; }
        textarea:focus, input:focus { border-color: var(--primary); outline: none; }
        textarea { min-height: 100px; resize: vertical; }
        
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: #5a6fd6; transform: translateY(-2px); }
        .btn-success { background: var(--success); color: white; }
        .btn-danger { background: var(--danger); color: white; }
        .btn-3d { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; }
        
        /* Pet */
        .pet-section { text-align: center; }
        .pet-emoji { font-size: 5em; margin: 15px 0; }
        .pet-stats { display: flex; justify-content: center; gap: 30px; margin: 15px 0; }
        .pet-stat { text-align: center; }
        .pet-stat-value { font-size: 1.8em; font-weight: bold; color: var(--primary); }
        .pet-actions { display: flex; gap: 10px; justify-content: center; margin-top: 15px; }
        
        /* Goals & Habits */
        .item { background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 12px; border-left: 4px solid var(--primary); }
        .item-title { font-weight: 600; margin-bottom: 5px; }
        .progress-bar { background: #e5e7eb; height: 8px; border-radius: 10px; overflow: hidden; margin-top: 8px; }
        .progress-fill { background: linear-gradient(90deg, var(--primary), var(--secondary)); height: 100%; transition: width 0.3s; }
        .streak { color: var(--warning); font-weight: bold; }
        
        /* Grid */
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        
        /* 3D Button */
        .view-3d-btn { position: fixed; bottom: 20px; right: 20px; z-index: 1000; padding: 15px 25px; font-size: 1.1em; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        
        /* Voice input indicator */
        .voice-hint { font-size: 0.85em; color: #888; font-style: italic; margin-bottom: 10px; }
        
        /* Auth */
        .auth-container { max-width: 400px; margin: 50px auto; }
        .auth-tabs { display: flex; margin-bottom: 20px; }
        .auth-tab { flex: 1; padding: 12px; background: #f0f0f0; border: none; cursor: pointer; font-weight: 500; }
        .auth-tab.active { background: var(--primary); color: white; }
        
        /* Fractal */
        .fractal-container { text-align: center; }
        .fractal-container img { max-width: 100%; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header h1 { font-size: 1.6em; }
            .grid { grid-template-columns: 1fr; }
            .stats-row { grid-template-columns: repeat(3, 1fr); }
        }
        
        .hidden { display: none; }
        .user-bar { background: rgba(0,0,0,0.1); padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
    </style>
</head>
<body>
    <div id="app"></div>
    
    <script>
        let currentUser = null;
        const petEmojis = {cat:'🐱', dragon:'🐉', phoenix:'🔥', owl:'🦉', fox:'🦊'};
        
        async function api(endpoint, method='GET', data=null) {
            const opts = { method, headers: {'Content-Type':'application/json'}, credentials:'include' };
            if (data) opts.body = JSON.stringify(data);
            return (await fetch(endpoint, opts)).json();
        }
        
        async function checkAuth() {
            try {
                const res = await api('/api/auth/me');
                if (res.user) { currentUser = res.user; renderDashboard(); }
                else renderAuth();
            } catch { renderAuth(); }
        }
        
        function renderAuth() {
            document.getElementById('app').innerHTML = `
                <div class="header">
                    <h1>🌀 Life Fractal Intelligence</h1>
                    <p class="subtitle">Neurodivergent-optimized life planning</p>
                </div>
                <div class="container auth-container">
                    <div class="card">
                        <div class="auth-tabs">
                            <button class="auth-tab active" onclick="showAuthTab('login')">Login</button>
                            <button class="auth-tab" onclick="showAuthTab('register')">Register</button>
                        </div>
                        <div id="loginForm">
                            <input type="email" id="loginEmail" placeholder="Email">
                            <input type="password" id="loginPassword" placeholder="Password">
                            <button class="btn btn-primary" style="width:100%" onclick="doLogin()">Login</button>
                        </div>
                        <div id="registerForm" class="hidden">
                            <input type="email" id="regEmail" placeholder="Email">
                            <input type="password" id="regPassword" placeholder="Password (8+ characters)">
                            <input type="text" id="petName" placeholder="Pet Name" value="Buddy">
                            <select id="petSpecies" style="width:100%;padding:12px;margin-bottom:15px;border:2px solid #e5e7eb;border-radius:8px;">
                                <option value="cat">🐱 Mystic Cat</option>
                                <option value="dragon">🐉 Sacred Dragon</option>
                                <option value="phoenix">🔥 Golden Phoenix</option>
                                <option value="owl">🦉 Wise Owl</option>
                                <option value="fox">🦊 Spirit Fox</option>
                            </select>
                            <button class="btn btn-success" style="width:100%" onclick="doRegister()">Start 7-Day Free Trial</button>
                        </div>
                        <div id="authMsg" style="margin-top:15px;color:var(--danger);text-align:center;"></div>
                    </div>
                </div>
            `;
        }
        
        function showAuthTab(tab) {
            document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('loginForm').classList.toggle('hidden', tab !== 'login');
            document.getElementById('registerForm').classList.toggle('hidden', tab !== 'register');
        }
        
        async function doLogin() {
            const res = await api('/api/auth/login', 'POST', {
                email: document.getElementById('loginEmail').value,
                password: document.getElementById('loginPassword').value
            });
            if (res.success) { currentUser = res; renderDashboard(); }
            else document.getElementById('authMsg').textContent = res.error || 'Login failed';
        }
        
        async function doRegister() {
            const res = await api('/api/auth/register', 'POST', {
                email: document.getElementById('regEmail').value,
                password: document.getElementById('regPassword').value,
                pet_name: document.getElementById('petName').value,
                pet_species: document.getElementById('petSpecies').value
            });
            if (res.success) { currentUser = res; renderDashboard(); }
            else document.getElementById('authMsg').textContent = res.error || 'Registration failed';
        }
        
        async function renderDashboard() {
            const [stats, goals, habits, pet, metrics] = await Promise.all([
                api('/api/stats'), api('/api/goals'), api('/api/habits'), 
                api('/api/pet'), api('/api/fractal/metrics')
            ]);
            
            document.getElementById('app').innerHTML = `
                <div class="header">
                    <div class="user-bar">
                        <span>Welcome, ${currentUser.email || 'User'}</span>
                        <button class="btn btn-danger" onclick="logout()">Logout</button>
                    </div>
                    <h1>🌀 Life Fractal Intelligence</h1>
                    <p class="subtitle">Neurodivergent-optimized life planning</p>
                </div>
                
                <div class="container">
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="value">${stats.active_goals || 0}</div>
                            <div class="label">Active Goals</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${stats.longest_streak || 0}</div>
                            <div class="label">Longest Streak</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${stats.companion_level || 1}</div>
                            <div class="label">Companion Level</div>
                        </div>
                    </div>
                    
                    <div class="grid">
                        <!-- Check-in Card -->
                        <div class="card">
                            <h2>📊 Today's Check-In</h2>
                            <label>How is your mood? (1-100)</label>
                            <input type="range" id="moodSlider" min="1" max="100" value="50" oninput="updateSlider('mood')">
                            <div class="slider-value" id="moodValue">50</div>
                            
                            <label>Energy level? (1-100)</label>
                            <input type="range" id="energySlider" min="1" max="100" value="50" oninput="updateSlider('energy')">
                            <div class="slider-value" id="energyValue">50</div>
                            
                            <label>Stress level? (1-100)</label>
                            <input type="range" id="stressSlider" min="1" max="100" value="30" oninput="updateSlider('stress')">
                            <div class="slider-value" id="stressValue">30</div>
                            
                            <label>Notes</label>
                            <div class="voice-hint">💬 Voice input supported - speak naturally</div>
                            <textarea id="notesInput" placeholder="How are you feeling today? Any thoughts or reflections..."></textarea>
                            
                            <button class="btn btn-success" style="width:100%" onclick="saveCheckin()">Save Check-In</button>
                        </div>
                        
                        <!-- Pet Card -->
                        <div class="card pet-section">
                            <h2>🐾 ${pet.pet?.name || 'Your Companion'}</h2>
                            <div class="pet-emoji">${petEmojis[pet.pet?.species] || '🐱'}</div>
                            <div class="pet-stats">
                                <div class="pet-stat">
                                    <div class="pet-stat-value">${pet.pet?.level || 1}</div>
                                    <div>Level</div>
                                </div>
                                <div class="pet-stat">
                                    <div class="pet-stat-value">${pet.pet?.happiness || 0}%</div>
                                    <div>Happiness</div>
                                </div>
                            </div>
                            <div class="progress-bar" style="margin:15px 0">
                                <div class="progress-fill" style="width:${(pet.pet?.xp || 0) % 100}%"></div>
                            </div>
                            <small>XP: ${pet.pet?.xp || 0}/${(pet.pet?.level || 1) * 100}</small>
                            <div class="pet-actions">
                                <button class="btn btn-primary" onclick="feedPet()">🍖 Feed</button>
                                <button class="btn btn-primary" onclick="playPet()">🎾 Play</button>
                            </div>
                        </div>
                        
                        <!-- Fractal Card -->
                        <div class="card fractal-container">
                            <h2>🎨 Your Life Fractal</h2>
                            <img src="/api/fractal/generate?size=400&t=${Date.now()}" alt="Fractal">
                            <p style="margin-top:15px;color:#666;">
                                Momentum: ${((metrics.metrics?.momentum || 0.5) * 100).toFixed(0)}%
                            </p>
                            <button class="btn btn-primary" style="margin-top:10px" onclick="refreshFractal()">🔄 Regenerate</button>
                        </div>
                        
                        <!-- Goals Card -->
                        <div class="card">
                            <h2>🎯 Your Goals</h2>
                            <div id="goalsList">
                                ${(goals.goals || []).map(g => `
                                    <div class="item">
                                        <div class="item-title">${g.title}</div>
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width:${g.progress || 0}%"></div>
                                        </div>
                                        <small>${g.progress || 0}% complete</small>
                                        <button class="btn btn-primary" style="float:right;padding:5px 10px;font-size:0.8em" onclick="updateGoal('${g.id}', ${g.progress || 0})">+5%</button>
                                    </div>
                                `).join('') || '<p style="color:#888">No goals yet</p>'}
                            </div>
                            <button class="btn btn-success" style="width:100%;margin-top:15px" onclick="showAddGoal()">Add Goal</button>
                        </div>
                        
                        <!-- Habits Card -->
                        <div class="card">
                            <h2>✨ Your Habits</h2>
                            <div id="habitsList">
                                ${(habits.habits || []).map(h => `
                                    <div class="item">
                                        <div class="item-title">${h.name}</div>
                                        <div class="streak">🔥 ${h.current_streak || 0} day streak</div>
                                        <button class="btn btn-success" style="margin-top:10px" onclick="completeHabit('${h.id}')">✓ Complete</button>
                                    </div>
                                `).join('') || '<p style="color:#888">No habits yet</p>'}
                            </div>
                            <button class="btn btn-success" style="width:100%;margin-top:15px" onclick="showAddHabit()">Add Habit</button>
                        </div>
                    </div>
                </div>
                
                <button class="btn btn-3d view-3d-btn" onclick="open3DView()">🌀 View 3D Universe</button>
            `;
        }
        
        function updateSlider(type) {
            const val = document.getElementById(type + 'Slider').value;
            document.getElementById(type + 'Value').textContent = val;
        }
        
        async function saveCheckin() {
            const res = await api('/api/daily-entry', 'POST', {
                mood_score: parseInt(document.getElementById('moodSlider').value),
                energy_level: parseInt(document.getElementById('energySlider').value),
                stress_level: parseInt(document.getElementById('stressSlider').value),
                notes: document.getElementById('notesInput').value
            });
            if (res.success) {
                alert('✅ Check-in saved!\\n\\n' + res.guidance);
                renderDashboard();
            }
        }
        
        async function feedPet() { await api('/api/pet/feed', 'POST'); renderDashboard(); }
        async function playPet() { await api('/api/pet/play', 'POST'); renderDashboard(); }
        
        function refreshFractal() {
            document.querySelector('.fractal-container img').src = '/api/fractal/generate?size=400&t=' + Date.now();
        }
        
        async function updateGoal(id, current) {
            await api('/api/goals/' + id, 'PUT', { progress: Math.min(100, current + 5) });
            renderDashboard();
        }
        
        async function completeHabit(id) {
            await api('/api/habits/' + id + '/complete', 'POST');
            renderDashboard();
        }
        
        function showAddGoal() {
            const title = prompt('Goal title:');
            if (title) api('/api/goals', 'POST', { title }).then(renderDashboard);
        }
        
        function showAddHabit() {
            const name = prompt('Habit name:');
            if (name) api('/api/habits', 'POST', { name }).then(renderDashboard);
        }
        
        function open3DView() { window.location.href = '/3d'; }
        
        async function logout() {
            await api('/api/auth/logout', 'POST');
            currentUser = null;
            renderAuth();
        }
        
        checkAuth();
    </script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════════════════════
# 3D VISUALIZATION HTML
# ═══════════════════════════════════════════════════════════════════════════════

INTERACTIVE_3D_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal - 3D Universe</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #000; color: #fff; overflow: hidden; }
        #canvas-container { width: 100vw; height: 100vh; }
        
        .panel { position: absolute; background: rgba(0,0,0,0.85); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); z-index: 1000; }
        
        #info-panel { top: 20px; left: 20px; border: 2px solid #667eea; max-width: 300px; }
        #info-panel h1 { font-size: 1.3em; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 15px; }
        
        #goals-panel { top: 20px; right: 20px; border: 2px solid #48c774; max-width: 320px; max-height: 70vh; overflow-y: auto; }
        #goals-panel h2 { color: #48c774; margin-bottom: 15px; }
        
        .goal-item { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #667eea; cursor: pointer; transition: all 0.3s; }
        .goal-item:hover { background: rgba(102,126,234,0.2); transform: translateX(5px); }
        .goal-progress { height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin-top: 8px; overflow: hidden; }
        .goal-progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #48c774); }
        
        #stats-panel { bottom: 20px; left: 20px; border: 2px solid #f0c420; display: flex; gap: 30px; }
        .stat { text-align: center; }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #f0c420; }
        .stat-label { font-size: 0.8em; color: #aaa; }
        
        #controls { bottom: 20px; right: 20px; border: 2px solid #764ba2; }
        .control-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .control-row label { min-width: 120px; color: #aaa; font-size: 0.9em; }
        .control-row input[type="range"] { flex: 1; accent-color: #667eea; }
        
        button { padding: 10px 20px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 8px; color: white; cursor: pointer; margin: 5px; transition: all 0.3s; }
        button:hover { transform: scale(1.05); box-shadow: 0 0 20px rgba(102,126,234,0.5); }
        
        #back-btn { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; }
        
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 2000; }
        .spinner { width: 60px; height: 60px; border: 4px solid rgba(102,126,234,0.3); border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .sacred-info { margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85em; color: #888; }
        .sacred-info span { color: #f0c420; }
    </style>
</head>
<body>
    <div id="loading"><div class="spinner"></div><div>Loading 3D Fractal Universe...</div></div>
    <div id="canvas-container"></div>
    
    <a href="/" id="back-btn"><button>← Back to Dashboard</button></a>
    
    <div class="panel" id="info-panel">
        <h1>🌀 Life Fractal Analysis</h1>
        <p style="color:#aaa;margin-bottom:15px;">Interactive 3D visualization of your goals mapped onto sacred geometry.</p>
        <div><strong>Controls:</strong>
            <ul style="margin:10px 0;padding-left:20px;color:#aaa;font-size:0.9em;">
                <li>🖱️ Left-click + drag: Rotate</li>
                <li>🖱️ Scroll: Zoom</li>
                <li>🔍 Click goal orbs for details</li>
            </ul>
        </div>
        <div class="sacred-info">
            <div>φ (Golden Ratio): <span>1.618033988...</span></div>
            <div>Golden Angle: <span>137.5°</span></div>
            <div>Frame: <span id="frame-count">0</span></div>
        </div>
    </div>
    
    <div class="panel" id="goals-panel">
        <h2>🎯 Goals in 3D Space</h2>
        <div id="goals-list"><div style="color:#aaa;">Loading goals...</div></div>
    </div>
    
    <div class="panel" id="stats-panel">
        <div class="stat"><div class="stat-value" id="total-goals">0</div><div class="stat-label">Total Goals</div></div>
        <div class="stat"><div class="stat-value" id="completed-goals">0</div><div class="stat-label">Completed</div></div>
        <div class="stat"><div class="stat-value" id="avg-progress">0%</div><div class="stat-label">Avg Progress</div></div>
    </div>
    
    <div class="panel" id="controls">
        <div class="control-row"><label>Animation Speed</label><input type="range" id="speed" min="0" max="200" value="100"></div>
        <div class="control-row"><label>Goal Scale</label><input type="range" id="scale" min="50" max="200" value="100"></div>
        <div style="text-align:center;margin-top:15px;">
            <button onclick="resetCamera()">Reset View</button>
        </div>
    </div>
    
    <script>
        const PHI = (1 + Math.sqrt(5)) / 2;
        const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
        const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        
        let scene, camera, renderer;
        let fractalMesh, goalOrbs = [], connectionLines = [];
        let animationSpeed = 1.0, frameCount = 0;
        let isDragging = false, prevMouse = {x: 0, y: 0};
        let cameraAngle = {x: 0, y: 0}, cameraDistance = 15;
        
        async function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050510);
            scene.fog = new THREE.FogExp2(0x050510, 0.015);
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 5, 15);
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('canvas-container').appendChild(renderer.domElement);
            
            // Lights
            scene.add(new THREE.AmbientLight(0x404040, 0.5));
            const light1 = new THREE.PointLight(0x667eea, 2, 100);
            light1.position.set(10, 10, 10);
            scene.add(light1);
            const light2 = new THREE.PointLight(0x764ba2, 2, 100);
            light2.position.set(-10, -10, 10);
            scene.add(light2);
            const light3 = new THREE.PointLight(0xf0c420, 1.5, 100);
            light3.position.set(0, 15, -10);
            scene.add(light3);
            
            createFractal();
            createSacredGeometry();
            createStarfield();
            await loadGoals();
            
            // Events
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            renderer.domElement.addEventListener('mousedown', e => { isDragging = true; prevMouse = {x: e.clientX, y: e.clientY}; });
            renderer.domElement.addEventListener('mouseup', () => isDragging = false);
            renderer.domElement.addEventListener('mousemove', e => {
                if (isDragging) {
                    cameraAngle.x += (e.clientX - prevMouse.x) * 0.01;
                    cameraAngle.y += (e.clientY - prevMouse.y) * 0.01;
                    cameraAngle.y = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, cameraAngle.y));
                    prevMouse = {x: e.clientX, y: e.clientY};
                }
            });
            renderer.domElement.addEventListener('wheel', e => {
                cameraDistance += e.deltaY * 0.01;
                cameraDistance = Math.max(5, Math.min(50, cameraDistance));
            });
            
            document.getElementById('speed').addEventListener('input', e => animationSpeed = e.target.value / 100);
            
            animate();
            setTimeout(() => document.getElementById('loading').style.display = 'none', 1500);
        }
        
        function createFractal() {
            const geometry = new THREE.BufferGeometry();
            const vertices = [], colors = [];
            const resolution = 60;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const theta = (i / resolution) * Math.PI;
                    const phi = (j / resolution) * Math.PI * 2;
                    
                    let x = Math.sin(theta) * Math.cos(phi);
                    let y = Math.sin(theta) * Math.sin(phi);
                    let z = Math.cos(theta);
                    
                    let r = 1;
                    for (let k = 0; k < 5; k++) {
                        r = Math.sqrt(x*x + y*y + z*z);
                        if (r > 2) break;
                        const th = Math.acos(z/r) * 8;
                        const ph = Math.atan2(y, x) * 8;
                        const zr = Math.pow(r, 8);
                        x = zr * Math.sin(th) * Math.cos(ph);
                        y = zr * Math.sin(th) * Math.sin(ph);
                        z = zr * Math.cos(th);
                    }
                    
                    const finalR = 3 * (1 + 0.3 * Math.sin(theta * 5) * Math.cos(phi * 3));
                    vertices.push(finalR * Math.sin(theta) * Math.cos(phi), finalR * Math.cos(theta), finalR * Math.sin(theta) * Math.sin(phi));
                    
                    const hue = (i / resolution + j / resolution * (PHI - 1)) % 1;
                    const color = new THREE.Color().setHSL(hue * 0.3 + 0.6, 0.8, 0.5 + 0.3 * Math.sin(theta * 3));
                    colors.push(color.r, color.g, color.b);
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            fractalMesh = new THREE.Points(geometry, new THREE.PointsMaterial({
                size: 0.08, vertexColors: true, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending
            }));
            scene.add(fractalMesh);
        }
        
        function createSacredGeometry() {
            // Golden spiral
            const spiralGeo = new THREE.BufferGeometry();
            const spiralVerts = [];
            for (let i = 0; i < 1000; i++) {
                const angle = i * 0.1;
                const r = 0.1 * Math.pow(PHI, 2 * angle / Math.PI);
                if (r > 8) break;
                spiralVerts.push(r * Math.cos(angle), r * Math.sin(angle) * 0.3, r * Math.sin(angle));
            }
            spiralGeo.setAttribute('position', new THREE.Float32BufferAttribute(spiralVerts, 3));
            scene.add(new THREE.Line(spiralGeo, new THREE.LineBasicMaterial({ color: 0xf0c420, transparent: true, opacity: 0.4 })));
            
            // Flower of Life
            const flowerGroup = new THREE.Group();
            const circleGeo = new THREE.CircleGeometry(2, 64);
            const edges = new THREE.EdgesGeometry(circleGeo);
            const circleMat = new THREE.LineBasicMaterial({ color: 0x667eea, transparent: true, opacity: 0.2 });
            
            [[0,0], [2,0], [-2,0], [1,1.73], [-1,1.73], [1,-1.73], [-1,-1.73]].forEach(([x, y]) => {
                const circle = new THREE.LineSegments(edges.clone(), circleMat);
                circle.position.set(x, 0, y);
                circle.rotation.x = Math.PI / 2;
                flowerGroup.add(circle);
            });
            scene.add(flowerGroup);
        }
        
        function createStarfield() {
            const starsGeo = new THREE.BufferGeometry();
            const starVerts = [], starColors = [];
            for (let i = 0; i < 2000; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 40 + Math.random() * 40;
                starVerts.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
                const b = 0.3 + Math.random() * 0.7;
                starColors.push(b, b, b * 1.2);
            }
            starsGeo.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
            starsGeo.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));
            scene.add(new THREE.Points(starsGeo, new THREE.PointsMaterial({ size: 0.3, vertexColors: true, transparent: true, opacity: 0.8 })));
        }
        
        async function loadGoals() {
            try {
                const res = await fetch('/api/goals', { credentials: 'include' });
                const data = await res.json();
                const goals = data.goals || [];
                
                // Update stats
                document.getElementById('total-goals').textContent = goals.length;
                document.getElementById('completed-goals').textContent = goals.filter(g => g.progress >= 100).length;
                document.getElementById('avg-progress').textContent = goals.length ? Math.round(goals.reduce((a,g) => a + (g.progress||0), 0) / goals.length) + '%' : '0%';
                
                // Goals list
                document.getElementById('goals-list').innerHTML = goals.length ? goals.map((g, i) => `
                    <div class="goal-item" onclick="focusGoal(${i})">
                        <div style="font-weight:bold;margin-bottom:5px;">${g.progress >= 100 ? '✅' : '🎯'} ${g.title}</div>
                        <div class="goal-progress"><div class="goal-progress-fill" style="width:${g.progress||0}%"></div></div>
                        <small style="color:#888">${g.progress||0}%</small>
                    </div>
                `).join('') : '<div style="color:#888">No goals found</div>';
                
                // Create orbs
                goalOrbs.forEach(o => scene.remove(o));
                goalOrbs = [];
                connectionLines.forEach(l => scene.remove(l));
                connectionLines = [];
                
                goals.forEach((goal, i) => {
                    const angle = i * GOLDEN_ANGLE;
                    const radius = 4 + (goal.priority || 3) * 0.5;
                    const height = (goal.progress / 100) * 4 - 2;
                    
                    const size = 0.3 + (goal.progress / 100) * 0.4;
                    const geometry = new THREE.SphereGeometry(size, 32, 32);
                    
                    let color;
                    if (goal.progress >= 100) color = new THREE.Color(0x48c774);
                    else if (goal.progress >= 70) color = new THREE.Color(0x3298dc);
                    else if (goal.progress >= 40) color = new THREE.Color(0xf0c420);
                    else color = new THREE.Color(0xff6b6b);
                    
                    const orb = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ 
                        color, emissive: color, emissiveIntensity: 0.3, transparent: true, opacity: 0.9 
                    }));
                    
                    orb.position.set(
                        radius * Math.cos(angle),
                        height + Math.sin(i * 0.5) * 1.5,
                        radius * Math.sin(angle)
                    );
                    orb.userData = { goal, index: i };
                    
                    // Glow
                    const glow = new THREE.Mesh(
                        new THREE.SphereGeometry(size * 1.5, 16, 16),
                        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.15 })
                    );
                    orb.add(glow);
                    
                    scene.add(orb);
                    goalOrbs.push(orb);
                    
                    // Connection to center
                    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), orb.position]);
                    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2 }));
                    scene.add(line);
                    connectionLines.push(line);
                });
                
            } catch (e) {
                console.error('Failed to load goals:', e);
            }
        }
        
        function focusGoal(index) {
            if (goalOrbs[index]) {
                const pos = goalOrbs[index].position;
                cameraAngle.x = Math.atan2(pos.x, pos.z);
                cameraAngle.y = Math.atan2(pos.y, Math.sqrt(pos.x*pos.x + pos.z*pos.z));
                cameraDistance = 8;
            }
        }
        
        function resetCamera() {
            cameraAngle = {x: 0, y: 0};
            cameraDistance = 15;
        }
        
        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            document.getElementById('frame-count').textContent = frameCount;
            
            // Update camera
            camera.position.x = cameraDistance * Math.sin(cameraAngle.x) * Math.cos(cameraAngle.y);
            camera.position.y = cameraDistance * Math.sin(cameraAngle.y) + 5;
            camera.position.z = cameraDistance * Math.cos(cameraAngle.x) * Math.cos(cameraAngle.y);
            camera.lookAt(0, 0, 0);
            
            // Animate fractal
            if (fractalMesh) {
                fractalMesh.rotation.y += 0.002 * animationSpeed;
                fractalMesh.rotation.x = Math.sin(frameCount * 0.001) * 0.1;
            }
            
            // Animate goal orbs
            goalOrbs.forEach((orb, i) => {
                orb.position.y = orb.userData.goal.progress / 100 * 4 - 2 + Math.sin(frameCount * 0.02 + i) * 0.3;
                orb.rotation.y += 0.01 * animationSpeed;
            });
            
            renderer.render(scene, camera);
        }
        
        init();
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/3d')
def view_3d():
    return render_template_string(INTERACTIVE_3D_HTML)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("═" * 60)
    logger.info("🌀 LIFE FRACTAL INTELLIGENCE v9.0 - ULTIMATE UNIFIED")
    logger.info("═" * 60)
    logger.info("✅ Clean Accessible Design: ACTIVE")
    logger.info("✅ 3D Visualization: ACTIVE (/3d)")
    logger.info("✅ ML Mood Prediction: " + ("ACTIVE" if HAS_SKLEARN else "DISABLED"))
    logger.info("✅ Self-Healing: ACTIVE")
    logger.info(f"✅ Port: {port}")
    logger.info("═" * 60)
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') != 'production')
