#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v8.0 - COMPLETE PRODUCTION BUILD
════════════════════════════════════════════════════════════════════════════════

For brains like mine - built with love for the neurodivergent community.

FIXES IN THIS VERSION:
✅ Neurodiversity symbol (infinity ∞) instead of wheelchair icon
✅ Full accessibility customization that saves properly
✅ Complete user management (register, login, password reset, secure storage)
✅ Working 3D fractal visualization with Three.js
✅ All data persists across sessions
✅ Pet companion system with proper state management
✅ Spoon theory energy tracking
✅ Sacred geometry and golden ratio mathematics

════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import hashlib
import threading
import sqlite3
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64
from contextlib import contextmanager

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, url_for, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]

# Mayan Calendar constants
MAYAN_DAY_SIGNS = [
    ("Imix", "primordial waters, new beginnings"),
    ("Ik", "breath of life, wind spirit"),
    ("Akbal", "darkness, inner reflection"),
    ("Kan", "seed, growth potential"),
    ("Chicchan", "serpent energy, kundalini"),
    ("Cimi", "transformation, death/rebirth"),
    ("Manik", "healing hand, accomplishment"),
    ("Lamat", "star harmony, abundance"),
    ("Muluc", "cosmic water, emotions"),
    ("Oc", "loyalty, heart guidance"),
    ("Chuen", "creative play, artistry"),
    ("Eb", "road of life, human journey"),
    ("Ben", "sky walker, pillars of light"),
    ("Ix", "jaguar wisdom, earth magic"),
    ("Men", "eagle vision, higher perspective"),
    ("Cib", "ancestral wisdom, forgiveness"),
    ("Caban", "earth force, synchronicity"),
    ("Etznab", "mirror truth, clarity"),
    ("Cauac", "thunder being, purification"),
    ("Ahau", "sun lord, enlightenment")
]

# ════════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════════════════════════

DATABASE_PATH = os.environ.get('DATABASE_PATH', 'life_fractal.db')

def get_db():
    """Get database connection for current request."""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with all tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table with all fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT DEFAULT '',
            display_name TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            is_admin INTEGER DEFAULT 0,
            email_verified INTEGER DEFAULT 0,
            subscription_status TEXT DEFAULT 'trial',
            trial_start_date TEXT,
            trial_end_date TEXT,
            stripe_customer_id TEXT,
            spoons INTEGER DEFAULT 12,
            max_spoons INTEGER DEFAULT 12,
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            created_at TEXT,
            last_login TEXT,
            password_reset_token TEXT,
            password_reset_expires TEXT
        )
    ''')
    
    # Accessibility preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accessibility_prefs (
            user_id TEXT PRIMARY KEY,
            reduced_motion INTEGER DEFAULT 0,
            high_contrast INTEGER DEFAULT 0,
            larger_text INTEGER DEFAULT 0,
            dyslexia_font INTEGER DEFAULT 0,
            screen_reader_mode INTEGER DEFAULT 0,
            focus_indicators INTEGER DEFAULT 1,
            simplified_layout INTEGER DEFAULT 0,
            calm_colors INTEGER DEFAULT 0,
            text_to_speech INTEGER DEFAULT 0,
            keyboard_navigation INTEGER DEFAULT 1,
            auto_save INTEGER DEFAULT 1,
            gentle_reminders INTEGER DEFAULT 1,
            celebration_intensity TEXT DEFAULT 'medium',
            time_blindness_helpers INTEGER DEFAULT 1,
            executive_function_support INTEGER DEFAULT 1,
            sensory_break_reminders INTEGER DEFAULT 0,
            task_chunking INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Pets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            user_id TEXT PRIMARY KEY,
            species TEXT DEFAULT 'cat',
            name TEXT DEFAULT 'Buddy',
            hunger REAL DEFAULT 50.0,
            energy REAL DEFAULT 50.0,
            mood REAL DEFAULT 50.0,
            stress REAL DEFAULT 30.0,
            bond REAL DEFAULT 20.0,
            level INTEGER DEFAULT 1,
            experience INTEGER DEFAULT 0,
            evolution_stage INTEGER DEFAULT 0,
            last_fed TEXT,
            last_played TEXT,
            last_rested TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Daily entries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            mood_score REAL DEFAULT 50.0,
            energy_level REAL DEFAULT 50.0,
            sleep_quality REAL DEFAULT 70.0,
            sleep_hours REAL DEFAULT 7.0,
            anxiety_level REAL DEFAULT 30.0,
            stress_level REAL DEFAULT 30.0,
            focus_clarity REAL DEFAULT 50.0,
            mindfulness_score REAL DEFAULT 50.0,
            gratitude_level REAL DEFAULT 50.0,
            spoons_used INTEGER DEFAULT 0,
            journal_entry TEXT DEFAULT '',
            wellness_index REAL DEFAULT 50.0,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            priority INTEGER DEFAULT 3,
            progress REAL DEFAULT 0.0,
            target_date TEXT,
            created_at TEXT,
            completed_at TEXT,
            color TEXT DEFAULT '#6B8E9F',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            goal_id TEXT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            spoon_cost INTEGER DEFAULT 1,
            priority INTEGER DEFAULT 3,
            due_date TEXT,
            completed INTEGER DEFAULT 0,
            completed_at TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (goal_id) REFERENCES goals(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")

# ════════════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════

def get_mayan_day(date: datetime = None) -> Dict[str, Any]:
    """Calculate Mayan Tzolkin day for given date."""
    if date is None:
        date = datetime.now()
    
    # Correlation constant (GMT correlation)
    correlation = 584283
    
    # Calculate Julian day number
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    jdn = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    # Calculate Tzolkin
    kin = (jdn - correlation) % 260
    day_number = (kin % 13) + 1
    day_sign_index = kin % 20
    
    day_sign, meaning = MAYAN_DAY_SIGNS[day_sign_index]
    
    return {
        'day_number': day_number,
        'day_sign': day_sign,
        'meaning': meaning,
        'full_name': f"{day_number} {day_sign}",
        'kin': kin + 1
    }

# ════════════════════════════════════════════════════════════════════════════════
# PET EMOJI MAPPING
# ════════════════════════════════════════════════════════════════════════════════

PET_EMOJIS = {
    'cat': {'happy': '😺', 'sad': '😿', 'tired': '😸', 'hungry': '🐱', 'playful': '😻', 'idle': '🐱'},
    'dragon': {'happy': '🐉', 'sad': '🐲', 'tired': '🐉', 'hungry': '🐲', 'playful': '🐉', 'idle': '🐲'},
    'phoenix': {'happy': '🦅', 'sad': '🪶', 'tired': '🦅', 'hungry': '🦅', 'playful': '🔥', 'idle': '🦅'},
    'owl': {'happy': '🦉', 'sad': '🦉', 'tired': '🦉', 'hungry': '🦉', 'playful': '🦉', 'idle': '🦉'},
    'fox': {'happy': '🦊', 'sad': '🦊', 'tired': '🦊', 'hungry': '🦊', 'playful': '🦊', 'idle': '🦊'},
    'bunny': {'happy': '🐰', 'sad': '🐇', 'tired': '🐰', 'hungry': '🐰', 'playful': '🐰', 'idle': '🐰'},
    'turtle': {'happy': '🐢', 'sad': '🐢', 'tired': '🐢', 'hungry': '🐢', 'playful': '🐢', 'idle': '🐢'},
    'butterfly': {'happy': '🦋', 'sad': '🦋', 'tired': '🦋', 'hungry': '🦋', 'playful': '🦋', 'idle': '🦋'}
}

def get_pet_emoji(species: str, mood: float) -> str:
    """Get appropriate emoji for pet based on species and mood."""
    emojis = PET_EMOJIS.get(species, PET_EMOJIS['cat'])
    if mood >= 70:
        return emojis['playful']
    elif mood >= 50:
        return emojis['happy']
    elif mood >= 30:
        return emojis['idle']
    else:
        return emojis['sad']

def get_pet_status(mood: float) -> str:
    """Get status text for pet mood."""
    if mood >= 80:
        return "Ecstatic 🌟"
    elif mood >= 60:
        return "Happy 😊"
    elif mood >= 40:
        return "Okay 😐"
    elif mood >= 20:
        return "Sad 😢"
    else:
        return "Needs Love 💔"

# ════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('PRODUCTION', 'false').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
CORS(app, supports_credentials=True)

app.teardown_appcontext(close_db)

# Configuration
SUBSCRIPTION_PRICE = 20.00
TRIAL_DAYS = 7
GOFUNDME_URL = 'https://gofund.me/8d9303d27'
STRIPE_PAYMENT_LINK = 'https://buy.stripe.com/4gw4jF3HU6Fx4jC5kk'

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    return True, "Password is valid"

def calculate_wellness(entry: dict) -> float:
    """Calculate wellness index using Fibonacci weighting."""
    weights = [FIBONACCI[i+3] for i in range(6)]  # [2, 3, 5, 8, 13, 21]
    total_weight = sum(weights)
    
    positive = (
        entry.get('mood_score', 50) * weights[0] +
        entry.get('energy_level', 50) * weights[1] +
        entry.get('focus_clarity', 50) * weights[2] +
        entry.get('mindfulness_score', 50) * weights[3] +
        entry.get('sleep_quality', 70) * weights[4] +
        entry.get('gratitude_level', 50) * weights[5]
    )
    
    negative = (entry.get('anxiety_level', 30) + entry.get('stress_level', 30)) * sum(weights[:2])
    
    wellness = max(0, min(100, (positive - negative / 2) / total_weight))
    return round(wellness, 1)

def generate_reset_token() -> str:
    """Generate a secure password reset token."""
    return secrets.token_urlsafe(32)

# ════════════════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_user(email: str, password: str, first_name: str = "") -> Optional[Dict]:
    """Create a new user with all defaults."""
    db = get_db()
    cursor = db.cursor()
    
    try:
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        trial_end = (datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)).isoformat()
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Create user
        cursor.execute('''
            INSERT INTO users (id, email, password_hash, first_name, display_name,
                             subscription_status, trial_start_date, trial_end_date,
                             spoons, max_spoons, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, 'trial', ?, ?, 12, 12, ?, ?)
        ''', (user_id, email.lower(), password_hash, first_name, 
              first_name or email.split('@')[0], now, trial_end, now, now))
        
        # Create default accessibility preferences
        cursor.execute('''
            INSERT INTO accessibility_prefs (user_id) VALUES (?)
        ''', (user_id,))
        
        # Create default pet
        cursor.execute('''
            INSERT INTO pets (user_id, species, name, hunger, energy, mood, bond)
            VALUES (?, 'cat', 'Buddy', 50, 50, 50, 20)
        ''', (user_id,))
        
        db.commit()
        
        return {
            'id': user_id,
            'email': email.lower(),
            'first_name': first_name,
            'display_name': first_name or email.split('@')[0],
            'spoons': 12,
            'max_spoons': 12,
            'trial_days_remaining': TRIAL_DAYS
        }
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email.lower(),))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def verify_password(user: Dict, password: str) -> bool:
    """Verify user password."""
    return check_password_hash(user['password_hash'], password)

def update_user_login(user_id: str):
    """Update last login timestamp."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        UPDATE users SET last_login = ? WHERE id = ?
    ''', (datetime.now(timezone.utc).isoformat(), user_id))
    db.commit()

def get_accessibility_prefs(user_id: str) -> Dict:
    """Get user's accessibility preferences."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM accessibility_prefs WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    if row:
        return dict(row)
    return {}

def save_accessibility_prefs(user_id: str, prefs: Dict) -> bool:
    """Save user's accessibility preferences."""
    db = get_db()
    cursor = db.cursor()
    
    fields = [
        'reduced_motion', 'high_contrast', 'larger_text', 'dyslexia_font',
        'screen_reader_mode', 'focus_indicators', 'simplified_layout', 'calm_colors',
        'text_to_speech', 'keyboard_navigation', 'auto_save', 'gentle_reminders',
        'celebration_intensity', 'time_blindness_helpers', 'executive_function_support',
        'sensory_break_reminders', 'task_chunking'
    ]
    
    updates = []
    values = []
    for field in fields:
        if field in prefs:
            updates.append(f"{field} = ?")
            values.append(prefs[field])
    
    if updates:
        values.append(user_id)
        cursor.execute(f'''
            UPDATE accessibility_prefs SET {', '.join(updates)} WHERE user_id = ?
        ''', values)
        db.commit()
        return True
    return False

def get_pet(user_id: str) -> Optional[Dict]:
    """Get user's pet."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    if row:
        pet = dict(row)
        pet['emoji'] = get_pet_emoji(pet['species'], pet['mood'])
        pet['status'] = get_pet_status(pet['mood'])
        pet['xp_for_next'] = FIBONACCI[min(pet['level'] + 5, len(FIBONACCI)-1)] * 10
        return pet
    return None

def update_pet(user_id: str, updates: Dict) -> bool:
    """Update pet state."""
    db = get_db()
    cursor = db.cursor()
    
    fields = ['hunger', 'energy', 'mood', 'stress', 'bond', 'level', 'experience',
              'last_fed', 'last_played', 'last_rested', 'name', 'species']
    
    update_parts = []
    values = []
    for field in fields:
        if field in updates:
            update_parts.append(f"{field} = ?")
            values.append(updates[field])
    
    if update_parts:
        values.append(user_id)
        cursor.execute(f'''
            UPDATE pets SET {', '.join(update_parts)} WHERE user_id = ?
        ''', values)
        db.commit()
        return True
    return False

def get_today_entry(user_id: str) -> Dict:
    """Get or create today's entry."""
    db = get_db()
    cursor = db.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute('''
        SELECT * FROM daily_entries WHERE user_id = ? AND date = ?
    ''', (user_id, today))
    row = cursor.fetchone()
    
    if row:
        return dict(row)
    
    # Create new entry
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute('''
        INSERT INTO daily_entries (user_id, date, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (user_id, today, now, now))
    db.commit()
    
    cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? AND date = ?', (user_id, today))
    return dict(cursor.fetchone())

def save_daily_entry(user_id: str, data: Dict) -> Dict:
    """Save daily entry data."""
    db = get_db()
    cursor = db.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now(timezone.utc).isoformat()
    
    # Calculate wellness
    data['wellness_index'] = calculate_wellness(data)
    
    fields = ['mood_score', 'energy_level', 'sleep_quality', 'sleep_hours',
              'anxiety_level', 'stress_level', 'focus_clarity', 'mindfulness_score',
              'gratitude_level', 'spoons_used', 'journal_entry', 'wellness_index']
    
    # Check if entry exists
    cursor.execute('SELECT id FROM daily_entries WHERE user_id = ? AND date = ?', (user_id, today))
    exists = cursor.fetchone()
    
    if exists:
        update_parts = [f"{f} = ?" for f in fields if f in data]
        values = [data[f] for f in fields if f in data]
        values.extend([now, user_id, today])
        
        cursor.execute(f'''
            UPDATE daily_entries SET {', '.join(update_parts)}, updated_at = ?
            WHERE user_id = ? AND date = ?
        ''', values)
    else:
        field_names = ['user_id', 'date', 'created_at', 'updated_at'] + [f for f in fields if f in data]
        placeholders = ', '.join(['?' for _ in field_names])
        values = [user_id, today, now, now] + [data[f] for f in fields if f in data]
        
        cursor.execute(f'''
            INSERT INTO daily_entries ({', '.join(field_names)}) VALUES ({placeholders})
        ''', values)
    
    db.commit()
    return get_today_entry(user_id)

def get_goals(user_id: str) -> List[Dict]:
    """Get all goals for user."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT * FROM goals WHERE user_id = ? ORDER BY priority, created_at DESC
    ''', (user_id,))
    return [dict(row) for row in cursor.fetchall()]

def create_goal(user_id: str, data: Dict) -> Dict:
    """Create a new goal."""
    db = get_db()
    cursor = db.cursor()
    
    goal_id = f"goal_{secrets.token_hex(6)}"
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute('''
        INSERT INTO goals (id, user_id, title, description, category, priority, target_date, color, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (goal_id, user_id, data.get('title', 'New Goal'), data.get('description', ''),
          data.get('category', 'general'), data.get('priority', 3),
          data.get('target_date'), data.get('color', '#6B8E9F'), now))
    
    db.commit()
    
    cursor.execute('SELECT * FROM goals WHERE id = ?', (goal_id,))
    return dict(cursor.fetchone())

def get_tasks(user_id: str, include_completed: bool = False) -> List[Dict]:
    """Get tasks for user."""
    db = get_db()
    cursor = db.cursor()
    
    if include_completed:
        cursor.execute('''
            SELECT * FROM tasks WHERE user_id = ? ORDER BY priority, due_date
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT * FROM tasks WHERE user_id = ? AND completed = 0 ORDER BY priority, due_date
        ''', (user_id,))
    
    return [dict(row) for row in cursor.fetchall()]

def create_task(user_id: str, data: Dict) -> Dict:
    """Create a new task."""
    db = get_db()
    cursor = db.cursor()
    
    task_id = f"task_{secrets.token_hex(6)}"
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute('''
        INSERT INTO tasks (id, user_id, goal_id, title, description, spoon_cost, priority, due_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (task_id, user_id, data.get('goal_id'), data.get('title', 'New Task'),
          data.get('description', ''), data.get('spoon_cost', 1),
          data.get('priority', 3), data.get('due_date'), now))
    
    db.commit()
    
    cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
    return dict(cursor.fetchone())

def complete_task(task_id: str, user_id: str) -> Optional[Dict]:
    """Mark task as complete."""
    db = get_db()
    cursor = db.cursor()
    
    # Get task and verify ownership
    cursor.execute('SELECT * FROM tasks WHERE id = ? AND user_id = ?', (task_id, user_id))
    task = cursor.fetchone()
    
    if not task:
        return None
    
    task = dict(task)
    now = datetime.now(timezone.utc).isoformat()
    
    # Mark complete
    cursor.execute('''
        UPDATE tasks SET completed = 1, completed_at = ? WHERE id = ?
    ''', (now, task_id))
    
    # Deduct spoons
    cursor.execute('''
        UPDATE users SET spoons = MAX(0, spoons - ?) WHERE id = ?
    ''', (task['spoon_cost'], user_id))
    
    # Update goal progress if linked
    if task['goal_id']:
        cursor.execute('''
            UPDATE goals SET progress = MIN(100, progress + 10) WHERE id = ?
        ''', (task['goal_id'],))
    
    # Award pet XP
    cursor.execute('''
        UPDATE pets SET experience = experience + ?, mood = MIN(100, mood + 5)
        WHERE user_id = ?
    ''', (task['spoon_cost'] * 5, user_id))
    
    db.commit()
    
    cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
    return dict(cursor.fetchone())

def set_password_reset_token(email: str) -> Optional[str]:
    """Set password reset token for user."""
    db = get_db()
    cursor = db.cursor()
    
    token = generate_reset_token()
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    
    cursor.execute('''
        UPDATE users SET password_reset_token = ?, password_reset_expires = ?
        WHERE email = ?
    ''', (token, expires, email.lower()))
    
    if cursor.rowcount > 0:
        db.commit()
        return token
    return None

def reset_password(token: str, new_password: str) -> bool:
    """Reset password using token."""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT id, password_reset_expires FROM users WHERE password_reset_token = ?
    ''', (token,))
    row = cursor.fetchone()
    
    if not row:
        return False
    
    # Check expiration
    expires = datetime.fromisoformat(row['password_reset_expires'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) > expires:
        return False
    
    # Update password
    password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
    cursor.execute('''
        UPDATE users SET password_hash = ?, password_reset_token = NULL, password_reset_expires = NULL
        WHERE id = ?
    ''', (password_hash, row['id']))
    
    db.commit()
    return True

# ════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        
        # Validate email
        if not email or not validate_email(email):
            return jsonify({'error': 'Please enter a valid email address'}), 400
        
        # Validate password
        valid, message = validate_password(password)
        if not valid:
            return jsonify({'error': message}), 400
        
        # Create user
        user = create_user(email, password, first_name)
        if not user:
            return jsonify({'error': 'An account with this email already exists'}), 400
        
        # Set session
        session['user_id'] = user['id']
        session.permanent = True
        
        return jsonify({
            'success': True,
            'message': 'Welcome to Life Fractal Intelligence! 🌀',
            'user': user,
            'trial_days_remaining': TRIAL_DAYS
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        user = get_user_by_email(email)
        if not user or not verify_password(user, password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user['is_active']:
            return jsonify({'error': 'This account has been deactivated'}), 403
        
        # Update login time
        update_user_login(user['id'])
        
        # Set session
        session['user_id'] = user['id']
        session.permanent = True
        
        # Calculate trial days
        trial_days = 0
        if user['trial_end_date']:
            try:
                trial_end = datetime.fromisoformat(user['trial_end_date'].replace('Z', '+00:00'))
                delta = trial_end - datetime.now(timezone.utc)
                trial_days = max(0, delta.days)
            except:
                pass
        
        return jsonify({
            'success': True,
            'message': f"Welcome back, {user['first_name'] or user['display_name']}! 👋",
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'display_name': user['display_name'],
                'spoons': user['spoons'],
                'max_spoons': user['max_spoons']
            },
            'trial_days_remaining': trial_days
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user."""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        token = set_password_reset_token(email)
        
        # Always return success to prevent email enumeration
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive reset instructions.',
            # In production, send email with reset link
            # For demo, include token in response
            'reset_token': token if token else None
        })
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500

@app.route('/api/auth/reset-password', methods=['POST'])
def do_reset_password():
    """Reset password with token."""
    try:
        data = request.get_json() or {}
        token = data.get('token', '')
        new_password = data.get('password', '')
        
        if not token or not new_password:
            return jsonify({'error': 'Token and new password are required'}), 400
        
        valid, message = validate_password(new_password)
        if not valid:
            return jsonify({'error': message}), 400
        
        if reset_password(token, new_password):
            return jsonify({
                'success': True,
                'message': 'Password reset successfully. You can now log in.'
            })
        else:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
            
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@app.route('/api/auth/session')
def check_session():
    """Check if user is logged in."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'authenticated': False}), 401
    
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({'authenticated': False}), 401
    
    return jsonify({
        'authenticated': True,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'first_name': user['first_name'],
            'display_name': user['display_name'],
            'spoons': user['spoons'],
            'max_spoons': user['max_spoons']
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# USER DATA ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard data for logged in user."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    pet = get_pet(user_id)
    today = get_today_entry(user_id)
    goals = get_goals(user_id)
    tasks = get_tasks(user_id)
    mayan = get_mayan_day()
    accessibility = get_accessibility_prefs(user_id)
    
    return jsonify({
        'user': {
            'id': user['id'],
            'email': user['email'],
            'first_name': user['first_name'],
            'display_name': user['display_name'],
            'spoons': user['spoons'],
            'max_spoons': user['max_spoons'],
            'current_streak': user['current_streak']
        },
        'pet': pet,
        'today': today,
        'goals': goals,
        'tasks': tasks,
        'mayan_day': mayan,
        'accessibility': accessibility,
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13]
        }
    })

@app.route('/api/accessibility', methods=['GET', 'POST'])
def handle_accessibility():
    """Get or save accessibility preferences."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        prefs = get_accessibility_prefs(user_id)
        return jsonify({'preferences': prefs})
    
    # POST - save preferences
    data = request.get_json() or {}
    save_accessibility_prefs(user_id, data)
    
    return jsonify({
        'success': True,
        'message': 'Accessibility preferences saved!',
        'preferences': get_accessibility_prefs(user_id)
    })

@app.route('/api/checkin', methods=['POST'])
def save_checkin():
    """Save daily check-in."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json() or {}
    entry = save_daily_entry(user_id, data)
    
    # Update pet based on mood
    if 'mood_score' in data:
        pet = get_pet(user_id)
        if pet:
            mood_delta = (data['mood_score'] - 50) * 0.3
            new_mood = max(0, min(100, pet['mood'] + mood_delta))
            update_pet(user_id, {'mood': new_mood})
    
    return jsonify({
        'success': True,
        'message': 'Check-in saved! 🌟',
        'entry': entry
    })

@app.route('/api/pet', methods=['GET'])
def get_pet_data():
    """Get pet data."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    pet = get_pet(user_id)
    return jsonify({'pet': pet})

@app.route('/api/pet/feed', methods=['POST'])
def feed_pet():
    """Feed the pet."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    pet = get_pet(user_id)
    if not pet:
        return jsonify({'error': 'No pet found'}), 404
    
    new_hunger = max(0, pet['hunger'] - 30)
    new_mood = min(100, pet['mood'] + 5)
    
    update_pet(user_id, {
        'hunger': new_hunger,
        'mood': new_mood,
        'last_fed': datetime.now(timezone.utc).isoformat()
    })
    
    return jsonify({
        'success': True,
        'message': f"{pet['name']} enjoyed the treat! 🍖",
        'pet': get_pet(user_id)
    })

@app.route('/api/pet/play', methods=['POST'])
def play_with_pet():
    """Play with the pet."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    pet = get_pet(user_id)
    if not pet:
        return jsonify({'error': 'No pet found'}), 404
    
    if pet['energy'] < 20:
        return jsonify({'error': f"{pet['name']} is too tired to play right now 😴"}), 400
    
    new_energy = max(0, pet['energy'] - 15)
    new_mood = min(100, pet['mood'] + 15)
    new_bond = min(100, pet['bond'] + 3)
    new_xp = pet['experience'] + 10
    
    # Check for level up
    xp_needed = FIBONACCI[min(pet['level'] + 5, len(FIBONACCI)-1)] * 10
    new_level = pet['level']
    if new_xp >= xp_needed:
        new_level += 1
        new_xp -= xp_needed
    
    update_pet(user_id, {
        'energy': new_energy,
        'mood': new_mood,
        'bond': new_bond,
        'experience': new_xp,
        'level': new_level,
        'last_played': datetime.now(timezone.utc).isoformat()
    })
    
    response = {
        'success': True,
        'message': f"You played with {pet['name']}! 🎮",
        'pet': get_pet(user_id)
    }
    
    if new_level > pet['level']:
        response['level_up'] = True
        response['message'] = f"🎉 {pet['name']} leveled up to level {new_level}!"
    
    return jsonify(response)

@app.route('/api/pet/rest', methods=['POST'])
def rest_pet():
    """Let pet rest."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    pet = get_pet(user_id)
    if not pet:
        return jsonify({'error': 'No pet found'}), 404
    
    new_energy = min(100, pet['energy'] + 25)
    new_stress = max(0, pet['stress'] - 10)
    
    update_pet(user_id, {
        'energy': new_energy,
        'stress': new_stress,
        'last_rested': datetime.now(timezone.utc).isoformat()
    })
    
    return jsonify({
        'success': True,
        'message': f"{pet['name']} is resting peacefully 😴",
        'pet': get_pet(user_id)
    })

@app.route('/api/goals', methods=['GET', 'POST'])
def handle_goals():
    """Get or create goals."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        goals = get_goals(user_id)
        return jsonify({'goals': goals})
    
    # POST - create goal
    data = request.get_json() or {}
    goal = create_goal(user_id, data)
    
    return jsonify({
        'success': True,
        'message': 'Goal created! 🎯',
        'goal': goal
    })

@app.route('/api/tasks', methods=['GET', 'POST'])
def handle_tasks():
    """Get or create tasks."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        include_completed = request.args.get('include_completed', 'false').lower() == 'true'
        tasks = get_tasks(user_id, include_completed)
        return jsonify({'tasks': tasks})
    
    # POST - create task
    data = request.get_json() or {}
    task = create_task(user_id, data)
    
    return jsonify({
        'success': True,
        'message': 'Task added! ✅',
        'task': task
    })

@app.route('/api/tasks/<task_id>/complete', methods=['POST'])
def mark_task_complete(task_id):
    """Mark task as complete."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    task = complete_task(task_id, user_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    user = get_user_by_id(user_id)
    pet = get_pet(user_id)
    
    return jsonify({
        'success': True,
        'message': f"Task complete! -{task['spoon_cost']} spoons 🥄",
        'task': task,
        'spoons_remaining': user['spoons'],
        'pet': pet
    })

# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/visualization/data')
def get_visualization_data():
    """Get data for 3D visualization."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    goals = get_goals(user_id)
    today = get_today_entry(user_id)
    pet = get_pet(user_id)
    
    # Create goal orbs positioned using golden angle
    orbs = []
    for i, goal in enumerate(goals):
        theta = i * GOLDEN_ANGLE_RAD
        r = 3 + (i * 0.5)
        
        orbs.append({
            'id': goal['id'],
            'title': goal['title'],
            'progress': goal['progress'],
            'color': goal['color'],
            'position': {
                'x': r * math.cos(theta),
                'y': (goal['progress'] / 100 * 2) - 1,
                'z': r * math.sin(theta)
            },
            'size': 0.3 + (goal['priority'] / 10)
        })
    
    return jsonify({
        'orbs': orbs,
        'wellness': today.get('wellness_index', 50),
        'mood': today.get('mood_score', 50),
        'energy': today.get('energy_level', 50),
        'pet_mood': pet['mood'] if pet else 50,
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# MAIN HTML TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=OpenDyslexic&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --primary: #6B8E9F;
            --primary-light: #8FB3C4;
            --primary-dark: #4A6B7C;
            --secondary: #9F8E6B;
            --accent: #8E6B9F;
            --success: #6B9F8E;
            --warning: #9F9F6B;
            --danger: #9F6B6B;
            --background: #F8F9FA;
            --surface: #FFFFFF;
            --text: #2D3748;
            --text-secondary: #718096;
            --border: #E2E8F0;
            --shadow: rgba(0, 0, 0, 0.1);
            --radius: 12px;
            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Calm colors mode */
        .calm-colors {
            --primary: #7A9E7A;
            --primary-light: #9EBE9E;
            --accent: #7A7A9E;
            --background: #F5F7F5;
        }
        
        /* High contrast mode */
        .high-contrast {
            --primary: #0066CC;
            --text: #000000;
            --background: #FFFFFF;
            --border: #000000;
        }
        
        /* Dyslexia font mode */
        .dyslexia-font {
            --font-main: 'OpenDyslexic', sans-serif;
        }
        
        /* Larger text mode */
        .larger-text {
            font-size: 18px;
        }
        .larger-text h1 { font-size: 2.5rem; }
        .larger-text h2 { font-size: 2rem; }
        .larger-text h3 { font-size: 1.5rem; }
        
        /* Reduced motion */
        .reduced-motion * {
            animation: none !important;
            transition: none !important;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--font-main);
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Neurodiversity Symbol */
        .neurodiversity-symbol {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            font-size: 20px;
            background: linear-gradient(135deg, #FF6B6B, #FFE66D, #4ECDC4, #45B7D1, #96E6A1, #DDA0DD);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: bold;
        }
        
        /* Header */
        .header {
            background: var(--surface);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary);
        }
        
        .logo-icon {
            font-size: 1.5rem;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .spoons-display {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--primary-light);
            color: white;
            border-radius: 20px;
            font-weight: 500;
        }
        
        .user-greeting {
            color: var(--text-secondary);
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: var(--radius);
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
        }
        
        .btn-secondary {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--background);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        /* Accessibility Banner */
        .accessibility-banner {
            background: linear-gradient(135deg, #E8F4F8, #F8E8F4);
            padding: 0.75rem 2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border);
        }
        
        .accessibility-banner a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }
        
        .accessibility-banner a:hover {
            text-decoration: underline;
        }
        
        /* Main Layout */
        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        
        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: 0 2px 8px var(--shadow);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
            color: var(--text);
        }
        
        .card-icon {
            font-size: 1.25rem;
        }
        
        /* Pet Card */
        .pet-display {
            text-align: center;
            padding: 1rem 0;
        }
        
        .pet-emoji {
            font-size: 4rem;
            margin-bottom: 0.5rem;
            animation: gentle-bounce 2s ease-in-out infinite;
        }
        
        @keyframes gentle-bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .pet-name {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .pet-status {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .pet-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin: 1rem 0;
        }
        
        .stat-box {
            background: var(--background);
            padding: 0.75rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }
        
        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
        }
        
        .pet-level {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }
        
        .pet-actions {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
        }
        
        /* Check-in Form */
        .checkin-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
        }
        
        .slider-value {
            font-weight: 600;
            color: var(--primary);
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: var(--border);
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
        }
        
        textarea {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
            min-height: 80px;
        }
        
        /* Tasks Section */
        .tasks-empty {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .task-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .task-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--background);
            border-radius: 8px;
        }
        
        .task-checkbox {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .task-title {
            flex: 1;
        }
        
        .task-spoons {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        /* Calendar Card */
        .mayan-day {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, var(--primary-light), var(--accent));
            border-radius: 8px;
            color: white;
        }
        
        .mayan-number {
            font-size: 3rem;
            font-weight: 700;
        }
        
        .mayan-sign {
            font-size: 1.25rem;
            margin-top: 0.5rem;
        }
        
        .mayan-meaning {
            font-size: 0.85rem;
            opacity: 0.9;
            margin-top: 0.25rem;
        }
        
        /* Fractal Buttons */
        .fractal-buttons {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
        }
        
        /* Goals Section */
        .goals-empty {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        /* Auth Pages */
        .auth-container {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background: linear-gradient(135deg, var(--background), #E8F4F8);
        }
        
        .auth-card {
            background: var(--surface);
            padding: 2.5rem;
            border-radius: var(--radius);
            box-shadow: 0 4px 20px var(--shadow);
            width: 100%;
            max-width: 400px;
        }
        
        .auth-logo {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .auth-logo h1 {
            color: var(--primary);
            font-size: 1.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .auth-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .form-group label {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .form-group input {
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-light);
        }
        
        .auth-footer {
            text-align: center;
            margin-top: 1.5rem;
            color: var(--text-secondary);
        }
        
        .auth-footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }
        
        .error-message {
            background: #FEE2E2;
            color: #DC2626;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .success-message {
            background: #D1FAE5;
            color: #059669;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        /* Accessibility Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
        }
        
        .modal-content {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 2rem;
            max-width: 600px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .modal-title {
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-secondary);
        }
        
        .accessibility-section {
            margin-bottom: 1.5rem;
        }
        
        .accessibility-section h3 {
            font-size: 1rem;
            color: var(--text);
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .toggle-group {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .toggle-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }
        
        .toggle-label {
            display: flex;
            flex-direction: column;
        }
        
        .toggle-label span:first-child {
            font-weight: 500;
        }
        
        .toggle-label span:last-child {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        .toggle-switch {
            position: relative;
            width: 48px;
            height: 26px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--border);
            border-radius: 13px;
            transition: 0.3s;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
        }
        
        input:checked + .toggle-slider {
            background: var(--primary);
        }
        
        input:checked + .toggle-slider:before {
            transform: translateX(22px);
        }
        
        /* 3D View */
        .fractal-view {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: #0A0E17;
            z-index: 200;
        }
        
        .fractal-controls {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(20, 30, 45, 0.9);
            padding: 1rem;
            border-radius: var(--radius);
            color: white;
            min-width: 250px;
        }
        
        .fractal-controls h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .control-group {
            margin-bottom: 0.75rem;
        }
        
        .control-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .fractal-controls input[type="range"] {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .fractal-info {
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            background: rgba(20, 30, 45, 0.9);
            padding: 1rem;
            border-radius: var(--radius);
            color: white;
            font-size: 0.85rem;
        }
        
        .hidden {
            display: none !important;
        }
        
        /* Loading State */
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        /* Focus Indicators */
        .focus-indicators *:focus {
            outline: 3px solid var(--primary);
            outline-offset: 2px;
        }
    </style>
</head>
<body>
    <!-- Auth Pages -->
    <div id="login-page" class="auth-container">
        <div class="auth-card">
            <div class="auth-logo">
                <h1>🌀 Life Fractal</h1>
            </div>
            <form id="login-form" class="auth-form">
                <div id="login-error" class="error-message hidden"></div>
                <div class="form-group">
                    <label for="login-email">Email</label>
                    <input type="email" id="login-email" required autocomplete="email">
                </div>
                <div class="form-group">
                    <label for="login-password">Password</label>
                    <input type="password" id="login-password" required autocomplete="current-password">
                </div>
                <button type="submit" class="btn btn-primary" style="width: 100%; justify-content: center;">
                    Log In
                </button>
            </form>
            <div class="auth-footer">
                <p>Don't have an account? <a href="#" onclick="showPage('register')">Sign up</a></p>
                <p style="margin-top: 0.5rem;"><a href="#" onclick="showPage('forgot-password')">Forgot password?</a></p>
            </div>
        </div>
    </div>
    
    <div id="register-page" class="auth-container hidden">
        <div class="auth-card">
            <div class="auth-logo">
                <h1>🌀 Life Fractal</h1>
            </div>
            <form id="register-form" class="auth-form">
                <div id="register-error" class="error-message hidden"></div>
                <div class="form-group">
                    <label for="register-name">Your Name (optional)</label>
                    <input type="text" id="register-name" autocomplete="given-name">
                </div>
                <div class="form-group">
                    <label for="register-email">Email</label>
                    <input type="email" id="register-email" required autocomplete="email">
                </div>
                <div class="form-group">
                    <label for="register-password">Password</label>
                    <input type="password" id="register-password" required autocomplete="new-password" 
                           minlength="8" placeholder="At least 8 characters">
                </div>
                <button type="submit" class="btn btn-primary" style="width: 100%; justify-content: center;">
                    Create Account
                </button>
            </form>
            <div class="auth-footer">
                <p>Already have an account? <a href="#" onclick="showPage('login')">Log in</a></p>
            </div>
        </div>
    </div>
    
    <div id="forgot-password-page" class="auth-container hidden">
        <div class="auth-card">
            <div class="auth-logo">
                <h1>🌀 Reset Password</h1>
            </div>
            <form id="forgot-form" class="auth-form">
                <div id="forgot-message" class="hidden"></div>
                <div class="form-group">
                    <label for="forgot-email">Email</label>
                    <input type="email" id="forgot-email" required>
                </div>
                <button type="submit" class="btn btn-primary" style="width: 100%; justify-content: center;">
                    Send Reset Link
                </button>
            </form>
            <div class="auth-footer">
                <p><a href="#" onclick="showPage('login')">Back to login</a></p>
            </div>
        </div>
    </div>
    
    <!-- Main Dashboard -->
    <div id="dashboard-page" class="hidden">
        <header class="header">
            <div class="logo">
                <span class="logo-icon">🌀</span>
                <span>Life Fractal Intelligence</span>
            </div>
            <div class="header-right">
                <div class="spoons-display">
                    <span>🥄</span>
                    <span id="spoons-count">12</span>
                    <span>Spoons</span>
                </div>
                <span class="user-greeting">Hi, <span id="user-name">there</span>!</span>
                <button class="btn btn-secondary" onclick="logout()">Logout</button>
            </div>
        </header>
        
        <div class="accessibility-banner">
            <span class="neurodiversity-symbol">∞</span>
            <span>Accessibility features enabled.</span>
            <a href="#" onclick="openAccessibilityModal()">Customize</a>
        </div>
        
        <main class="main-content">
            <div class="dashboard-grid">
                <!-- Pet Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">🐾</span>
                        <span>Your Companion</span>
                    </div>
                    <div class="pet-display">
                        <div class="pet-emoji" id="pet-emoji">🐱</div>
                        <div class="pet-name" id="pet-name">Buddy</div>
                        <div class="pet-status" id="pet-status">Okay 😐</div>
                    </div>
                    <div class="pet-stats">
                        <div class="stat-box">
                            <div class="stat-label">Hunger</div>
                            <div class="stat-value" id="pet-hunger">50</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Energy</div>
                            <div class="stat-value" id="pet-energy">50</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Mood</div>
                            <div class="stat-value" id="pet-mood">50</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Bond</div>
                            <div class="stat-value" id="pet-bond">20</div>
                        </div>
                    </div>
                    <div class="pet-level" id="pet-level">Level 1 • 0/80 XP</div>
                    <div class="pet-actions">
                        <button class="btn btn-secondary" onclick="feedPet()">🍖 Feed</button>
                        <button class="btn btn-primary" onclick="playWithPet()">✨ Play</button>
                        <button class="btn btn-secondary" onclick="restPet()">⭐ Rest</button>
                    </div>
                </div>
                
                <!-- Daily Check-in -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📝</span>
                        <span>Daily Check-in</span>
                    </div>
                    <form id="checkin-form" class="checkin-form">
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Mood</span>
                                <span class="slider-value" id="mood-value">50</span>
                            </div>
                            <input type="range" id="mood-slider" min="0" max="100" value="50" 
                                   oninput="updateSliderValue('mood')">
                        </div>
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Energy</span>
                                <span class="slider-value" id="energy-value">50</span>
                            </div>
                            <input type="range" id="energy-slider" min="0" max="100" value="50"
                                   oninput="updateSliderValue('energy')">
                        </div>
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Sleep Quality</span>
                                <span class="slider-value" id="sleep-value">70</span>
                            </div>
                            <input type="range" id="sleep-slider" min="0" max="100" value="70"
                                   oninput="updateSliderValue('sleep')">
                        </div>
                        <div class="form-group">
                            <label>Journal (optional)</label>
                            <textarea id="journal-entry" placeholder="How are you feeling today?"></textarea>
                        </div>
                        <button type="submit" class="btn btn-success" style="width: 100%; justify-content: center;">
                            Save Check-in
                        </button>
                    </form>
                </div>
                
                <!-- Today's Tasks -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">✅</span>
                        <span>Today's Tasks</span>
                    </div>
                    <div id="tasks-container">
                        <div class="tasks-empty">
                            No tasks yet. Add some goals first!
                        </div>
                    </div>
                    <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                        <button class="btn btn-primary" onclick="openAddTaskModal()">+ Add Task</button>
                        <button class="btn btn-secondary" onclick="suggestNextTask()">🎯 What's Next?</button>
                    </div>
                </div>
                
                <!-- Sacred Calendar -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📅</span>
                        <span>Sacred Calendar</span>
                    </div>
                    <div class="mayan-day">
                        <div>Today's Mayan Day</div>
                        <div class="mayan-number" id="mayan-number">11</div>
                        <div class="mayan-sign" id="mayan-sign">Oc</div>
                        <div class="mayan-meaning" id="mayan-meaning">loyalty, heart guidance</div>
                    </div>
                    <button class="btn btn-secondary" style="width: 100%; margin-top: 1rem; justify-content: center;">
                        View Full Calendar
                    </button>
                </div>
                
                <!-- Life Fractal -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">🌀</span>
                        <span>Your Life Fractal</span>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        Your metrics transformed into sacred geometry
                    </p>
                    <div class="fractal-buttons">
                        <button class="btn btn-primary" onclick="generate2DFractal()">Generate 2D</button>
                        <button class="btn btn-primary" onclick="open3DView()">Generate 3D</button>
                    </div>
                </div>
                
                <!-- Goals -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">🎯</span>
                        <span>Goals</span>
                    </div>
                    <div id="goals-container">
                        <div class="goals-empty">
                            No goals yet. Start by setting some!
                        </div>
                    </div>
                    <button class="btn btn-primary" style="width: 100%; margin-top: 1rem; justify-content: center;" 
                            onclick="openAddGoalModal()">
                        + Add Goal
                    </button>
                </div>
            </div>
        </main>
    </div>
    
    <!-- Accessibility Modal -->
    <div id="accessibility-modal" class="modal-overlay hidden">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">
                    <span class="neurodiversity-symbol">∞</span>
                    Accessibility Settings
                </h2>
                <button class="modal-close" onclick="closeAccessibilityModal()">×</button>
            </div>
            
            <div class="accessibility-section">
                <h3>Visual Preferences</h3>
                <div class="toggle-group">
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Reduced Motion</span>
                            <span>Minimize animations and transitions</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-reduced-motion" onchange="saveAccessibilityPref('reduced_motion', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>High Contrast</span>
                            <span>Increase color contrast for better visibility</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-high-contrast" onchange="saveAccessibilityPref('high_contrast', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Larger Text</span>
                            <span>Increase text size throughout the app</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-larger-text" onchange="saveAccessibilityPref('larger_text', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Dyslexia-Friendly Font</span>
                            <span>Use OpenDyslexic font for easier reading</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-dyslexia-font" onchange="saveAccessibilityPref('dyslexia_font', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Calm Colors</span>
                            <span>Use softer, less stimulating colors</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-calm-colors" onchange="saveAccessibilityPref('calm_colors', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
            
            <div class="accessibility-section">
                <h3>Navigation & Interaction</h3>
                <div class="toggle-group">
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Keyboard Navigation</span>
                            <span>Enhanced keyboard shortcuts and focus indicators</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-keyboard-nav" checked onchange="saveAccessibilityPref('keyboard_navigation', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Focus Indicators</span>
                            <span>Show clear outlines when navigating with keyboard</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-focus-indicators" checked onchange="saveAccessibilityPref('focus_indicators', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Simplified Layout</span>
                            <span>Reduce visual clutter for better focus</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-simplified-layout" onchange="saveAccessibilityPref('simplified_layout', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
            
            <div class="accessibility-section">
                <h3>Executive Function Support</h3>
                <div class="toggle-group">
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Time Blindness Helpers</span>
                            <span>Visual time indicators and gentle reminders</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-time-helpers" checked onchange="saveAccessibilityPref('time_blindness_helpers', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Task Chunking</span>
                            <span>Break tasks into smaller, manageable steps</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-task-chunking" checked onchange="saveAccessibilityPref('task_chunking', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Gentle Reminders</span>
                            <span>Soft, non-intrusive notifications</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-gentle-reminders" checked onchange="saveAccessibilityPref('gentle_reminders', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Sensory Break Reminders</span>
                            <span>Reminders to take sensory breaks</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-sensory-breaks" onchange="saveAccessibilityPref('sensory_break_reminders', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-label">
                            <span>Auto-Save</span>
                            <span>Automatically save your progress</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="pref-auto-save" checked onchange="saveAccessibilityPref('auto_save', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; text-align: center;">
                <button class="btn btn-primary" onclick="closeAccessibilityModal()">
                    Save & Close
                </button>
            </div>
        </div>
    </div>
    
    <!-- 3D Fractal View -->
    <div id="fractal-view" class="fractal-view hidden">
        <div class="fractal-controls">
            <h2>🌀 Fractal Universe</h2>
            <button class="btn btn-primary" style="width: 100%; margin-bottom: 0.75rem;" onclick="regenerateFractal()">
                🔄 Regenerate
            </button>
            <button class="btn btn-secondary" style="width: 100%; margin-bottom: 0.75rem;" onclick="addGoalOrb()">
                + Add Goal Orb
            </button>
            <button class="btn btn-secondary" style="width: 100%; margin-bottom: 0.75rem;" onclick="toggleSacredGeometry()">
                🔷 Toggle Sacred Geometry
            </button>
            <label class="toggle-switch" style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <input type="checkbox" id="auto-rotate" checked onchange="toggleAutoRotate()">
                <span class="toggle-slider"></span>
                <span style="color: white;">Auto-Rotate</span>
            </label>
            
            <div class="control-group">
                <div class="control-label">
                    <span>Mood Influence</span>
                    <span id="mood-influence-val">50</span>
                </div>
                <input type="range" id="mood-influence" min="0" max="100" value="50" oninput="updateFractalParam('mood')">
            </div>
            <div class="control-group">
                <div class="control-label">
                    <span>Energy Flow</span>
                    <span id="energy-flow-val">50</span>
                </div>
                <input type="range" id="energy-flow" min="0" max="100" value="50" oninput="updateFractalParam('energy')">
            </div>
            <div class="control-group">
                <div class="control-label">
                    <span>Complexity</span>
                    <span id="complexity-val">2000</span>
                </div>
                <input type="range" id="complexity" min="500" max="5000" value="2000" oninput="updateFractalParam('complexity')">
            </div>
            <div class="control-group">
                <div class="control-label">
                    <span>Goal Orbs</span>
                    <span id="orbs-val">0</span>
                </div>
            </div>
            
            <button class="btn btn-secondary" style="width: 100%; margin-top: 0.5rem;" onclick="resetCamera()">
                Reset Camera
            </button>
            <button class="btn btn-secondary" style="width: 100%; margin-top: 0.5rem;" onclick="close3DView()">
                ← Back to Dashboard
            </button>
        </div>
        
        <div class="fractal-info">
            <div><strong>Controls</strong></div>
            <div>Mouse: Drag to rotate</div>
            <div>Scroll: Zoom in/out</div>
            <div>Right-click: Pan camera</div>
            <div>Space: Pause/Resume rotation</div>
            <div style="margin-top: 0.5rem; color: var(--primary-light);">
                Your fractal evolves with your wellness data.<br>
                Each orb represents a goal, colored by progress and importance.
            </div>
        </div>
        
        <div id="fractal-container" style="width: 100%; height: 100%;"></div>
    </div>

    <script>
        // ============================================================
        // STATE MANAGEMENT
        // ============================================================
        let currentUser = null;
        let accessibilityPrefs = {};
        let threeScene = null;
        let threeCamera = null;
        let threeRenderer = null;
        let fractalPoints = null;
        let sacredGeometry = null;
        let goalOrbs = [];
        let autoRotate = true;
        let animationId = null;
        
        const PHI = 1.618033988749895;
        const GOLDEN_ANGLE = 137.5077640500378;
        const GOLDEN_ANGLE_RAD = GOLDEN_ANGLE * Math.PI / 180;
        
        // ============================================================
        // PAGE NAVIGATION
        // ============================================================
        function showPage(page) {
            document.getElementById('login-page').classList.add('hidden');
            document.getElementById('register-page').classList.add('hidden');
            document.getElementById('forgot-password-page').classList.add('hidden');
            document.getElementById('dashboard-page').classList.add('hidden');
            
            document.getElementById(page + '-page').classList.remove('hidden');
        }
        
        // ============================================================
        // AUTHENTICATION
        // ============================================================
        async function checkSession() {
            try {
                const res = await fetch('/api/auth/session');
                const data = await res.json();
                
                if (data.authenticated) {
                    currentUser = data.user;
                    showPage('dashboard');
                    loadDashboard();
                } else {
                    showPage('login');
                }
            } catch (e) {
                showPage('login');
            }
        }
        
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const errorEl = document.getElementById('login-error');
            errorEl.classList.add('hidden');
            
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    currentUser = data.user;
                    showPage('dashboard');
                    loadDashboard();
                } else {
                    errorEl.textContent = data.error;
                    errorEl.classList.remove('hidden');
                }
            } catch (e) {
                errorEl.textContent = 'Connection error. Please try again.';
                errorEl.classList.remove('hidden');
            }
        });
        
        document.getElementById('register-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const errorEl = document.getElementById('register-error');
            errorEl.classList.add('hidden');
            
            const first_name = document.getElementById('register-name').value;
            const email = document.getElementById('register-email').value;
            const password = document.getElementById('register-password').value;
            
            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password, first_name })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    currentUser = data.user;
                    showPage('dashboard');
                    loadDashboard();
                } else {
                    errorEl.textContent = data.error;
                    errorEl.classList.remove('hidden');
                }
            } catch (e) {
                errorEl.textContent = 'Connection error. Please try again.';
                errorEl.classList.remove('hidden');
            }
        });
        
        document.getElementById('forgot-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const msgEl = document.getElementById('forgot-message');
            
            const email = document.getElementById('forgot-email').value;
            
            try {
                const res = await fetch('/api/auth/forgot-password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email })
                });
                
                const data = await res.json();
                
                msgEl.textContent = data.message;
                msgEl.className = 'success-message';
                msgEl.classList.remove('hidden');
            } catch (e) {
                msgEl.textContent = 'Connection error. Please try again.';
                msgEl.className = 'error-message';
                msgEl.classList.remove('hidden');
            }
        });
        
        async function logout() {
            await fetch('/api/auth/logout', { method: 'POST' });
            currentUser = null;
            showPage('login');
        }
        
        // ============================================================
        // DASHBOARD
        // ============================================================
        async function loadDashboard() {
            try {
                const res = await fetch('/api/dashboard');
                const data = await res.json();
                
                if (!res.ok) {
                    showPage('login');
                    return;
                }
                
                currentUser = data.user;
                accessibilityPrefs = data.accessibility || {};
                
                // Update UI
                document.getElementById('user-name').textContent = data.user.display_name || data.user.first_name || 'there';
                document.getElementById('spoons-count').textContent = data.user.spoons;
                
                // Update pet
                if (data.pet) {
                    document.getElementById('pet-emoji').textContent = data.pet.emoji;
                    document.getElementById('pet-name').textContent = data.pet.name;
                    document.getElementById('pet-status').textContent = data.pet.status;
                    document.getElementById('pet-hunger').textContent = Math.round(data.pet.hunger);
                    document.getElementById('pet-energy').textContent = Math.round(data.pet.energy);
                    document.getElementById('pet-mood').textContent = Math.round(data.pet.mood);
                    document.getElementById('pet-bond').textContent = Math.round(data.pet.bond);
                    document.getElementById('pet-level').textContent = 
                        `Level ${data.pet.level} • ${data.pet.experience}/${data.pet.xp_for_next} XP`;
                }
                
                // Update mayan day
                if (data.mayan_day) {
                    document.getElementById('mayan-number').textContent = data.mayan_day.day_number;
                    document.getElementById('mayan-sign').textContent = data.mayan_day.day_sign;
                    document.getElementById('mayan-meaning').textContent = data.mayan_day.meaning;
                }
                
                // Update today's entry
                if (data.today) {
                    document.getElementById('mood-slider').value = data.today.mood_score || 50;
                    document.getElementById('energy-slider').value = data.today.energy_level || 50;
                    document.getElementById('sleep-slider').value = data.today.sleep_quality || 70;
                    document.getElementById('journal-entry').value = data.today.journal_entry || '';
                    updateSliderValue('mood');
                    updateSliderValue('energy');
                    updateSliderValue('sleep');
                }
                
                // Update goals
                updateGoalsDisplay(data.goals || []);
                
                // Update tasks
                updateTasksDisplay(data.tasks || []);
                
                // Apply accessibility preferences
                applyAccessibilityPrefs();
                
            } catch (e) {
                console.error('Error loading dashboard:', e);
            }
        }
        
        function updateSliderValue(type) {
            const slider = document.getElementById(type + '-slider');
            const valueEl = document.getElementById(type + '-value');
            valueEl.textContent = slider.value;
        }
        
        document.getElementById('checkin-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const data = {
                mood_score: parseInt(document.getElementById('mood-slider').value),
                energy_level: parseInt(document.getElementById('energy-slider').value),
                sleep_quality: parseInt(document.getElementById('sleep-slider').value),
                journal_entry: document.getElementById('journal-entry').value
            };
            
            try {
                const res = await fetch('/api/checkin', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (res.ok) {
                    const result = await res.json();
                    alert(result.message);
                    loadDashboard();
                }
            } catch (e) {
                console.error('Error saving check-in:', e);
            }
        });
        
        // ============================================================
        // PET INTERACTIONS
        // ============================================================
        async function feedPet() {
            try {
                const res = await fetch('/api/pet/feed', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    loadDashboard();
                    alert(data.message);
                } else {
                    alert(data.error);
                }
            } catch (e) {
                console.error('Error feeding pet:', e);
            }
        }
        
        async function playWithPet() {
            try {
                const res = await fetch('/api/pet/play', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    loadDashboard();
                    if (data.level_up) {
                        alert(data.message);
                    }
                } else {
                    alert(data.error);
                }
            } catch (e) {
                console.error('Error playing with pet:', e);
            }
        }
        
        async function restPet() {
            try {
                const res = await fetch('/api/pet/rest', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    loadDashboard();
                }
            } catch (e) {
                console.error('Error resting pet:', e);
            }
        }
        
        // ============================================================
        // GOALS & TASKS
        // ============================================================
        function updateGoalsDisplay(goals) {
            const container = document.getElementById('goals-container');
            
            if (goals.length === 0) {
                container.innerHTML = '<div class="goals-empty">No goals yet. Start by setting some!</div>';
                return;
            }
            
            container.innerHTML = goals.map(goal => `
                <div class="task-item" style="border-left: 3px solid ${goal.color};">
                    <div class="task-title">
                        <strong>${goal.title}</strong>
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">
                            ${Math.round(goal.progress)}% complete
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function updateTasksDisplay(tasks) {
            const container = document.getElementById('tasks-container');
            
            if (tasks.length === 0) {
                container.innerHTML = '<div class="tasks-empty">No tasks yet. Add some goals first!</div>';
                return;
            }
            
            container.innerHTML = `<div class="task-list">
                ${tasks.map(task => `
                    <div class="task-item">
                        <input type="checkbox" class="task-checkbox" 
                               onchange="completeTask('${task.id}')" ${task.completed ? 'checked disabled' : ''}>
                        <span class="task-title" ${task.completed ? 'style="text-decoration: line-through;"' : ''}>
                            ${task.title}
                        </span>
                        <span class="task-spoons">🥄 ${task.spoon_cost}</span>
                    </div>
                `).join('')}
            </div>`;
        }
        
        async function completeTask(taskId) {
            try {
                const res = await fetch(`/api/tasks/${taskId}/complete`, { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    document.getElementById('spoons-count').textContent = data.spoons_remaining;
                    loadDashboard();
                }
            } catch (e) {
                console.error('Error completing task:', e);
            }
        }
        
        async function openAddGoalModal() {
            const title = prompt('What goal would you like to set?');
            if (!title) return;
            
            try {
                const res = await fetch('/api/goals', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title })
                });
                
                if (res.ok) {
                    loadDashboard();
                }
            } catch (e) {
                console.error('Error creating goal:', e);
            }
        }
        
        async function openAddTaskModal() {
            const title = prompt('What task would you like to add?');
            if (!title) return;
            
            const spoons = parseInt(prompt('How many spoons will this cost? (1-5)', '2')) || 2;
            
            try {
                const res = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, spoon_cost: Math.min(5, Math.max(1, spoons)) })
                });
                
                if (res.ok) {
                    loadDashboard();
                }
            } catch (e) {
                console.error('Error creating task:', e);
            }
        }
        
        function suggestNextTask() {
            alert('🎯 Based on your energy levels and goals, consider starting with your smallest, lowest-spoon task first. Build momentum with quick wins!');
        }
        
        // ============================================================
        // ACCESSIBILITY
        // ============================================================
        function openAccessibilityModal() {
            document.getElementById('accessibility-modal').classList.remove('hidden');
            loadAccessibilityPrefs();
        }
        
        function closeAccessibilityModal() {
            document.getElementById('accessibility-modal').classList.add('hidden');
        }
        
        function loadAccessibilityPrefs() {
            document.getElementById('pref-reduced-motion').checked = accessibilityPrefs.reduced_motion || false;
            document.getElementById('pref-high-contrast').checked = accessibilityPrefs.high_contrast || false;
            document.getElementById('pref-larger-text').checked = accessibilityPrefs.larger_text || false;
            document.getElementById('pref-dyslexia-font').checked = accessibilityPrefs.dyslexia_font || false;
            document.getElementById('pref-calm-colors').checked = accessibilityPrefs.calm_colors || false;
            document.getElementById('pref-keyboard-nav').checked = accessibilityPrefs.keyboard_navigation !== false;
            document.getElementById('pref-focus-indicators').checked = accessibilityPrefs.focus_indicators !== false;
            document.getElementById('pref-simplified-layout').checked = accessibilityPrefs.simplified_layout || false;
            document.getElementById('pref-time-helpers').checked = accessibilityPrefs.time_blindness_helpers !== false;
            document.getElementById('pref-task-chunking').checked = accessibilityPrefs.task_chunking !== false;
            document.getElementById('pref-gentle-reminders').checked = accessibilityPrefs.gentle_reminders !== false;
            document.getElementById('pref-sensory-breaks').checked = accessibilityPrefs.sensory_break_reminders || false;
            document.getElementById('pref-auto-save').checked = accessibilityPrefs.auto_save !== false;
        }
        
        async function saveAccessibilityPref(key, value) {
            accessibilityPrefs[key] = value ? 1 : 0;
            
            try {
                await fetch('/api/accessibility', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ [key]: value ? 1 : 0 })
                });
                
                applyAccessibilityPrefs();
            } catch (e) {
                console.error('Error saving accessibility pref:', e);
            }
        }
        
        function applyAccessibilityPrefs() {
            document.body.classList.toggle('reduced-motion', accessibilityPrefs.reduced_motion);
            document.body.classList.toggle('high-contrast', accessibilityPrefs.high_contrast);
            document.body.classList.toggle('larger-text', accessibilityPrefs.larger_text);
            document.body.classList.toggle('dyslexia-font', accessibilityPrefs.dyslexia_font);
            document.body.classList.toggle('calm-colors', accessibilityPrefs.calm_colors);
            document.body.classList.toggle('focus-indicators', accessibilityPrefs.focus_indicators !== false);
        }
        
        // ============================================================
        // 3D VISUALIZATION
        // ============================================================
        function open3DView() {
            document.getElementById('fractal-view').classList.remove('hidden');
            initThreeJS();
        }
        
        function close3DView() {
            document.getElementById('fractal-view').classList.add('hidden');
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }
        
        function initThreeJS() {
            const container = document.getElementById('fractal-container');
            container.innerHTML = '';
            
            // Scene
            threeScene = new THREE.Scene();
            threeScene.background = new THREE.Color(0x0A0E17);
            
            // Camera
            threeCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            threeCamera.position.z = 10;
            
            // Renderer
            threeRenderer = new THREE.WebGLRenderer({ antialias: true });
            threeRenderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(threeRenderer.domElement);
            
            // Create fractal
            createFractal();
            
            // Create sacred geometry
            createSacredGeometry();
            
            // Mouse controls
            let isDragging = false;
            let previousMousePosition = { x: 0, y: 0 };
            
            threeRenderer.domElement.addEventListener('mousedown', (e) => {
                isDragging = true;
                previousMousePosition = { x: e.clientX, y: e.clientY };
            });
            
            threeRenderer.domElement.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const deltaMove = {
                    x: e.clientX - previousMousePosition.x,
                    y: e.clientY - previousMousePosition.y
                };
                
                threeScene.rotation.y += deltaMove.x * 0.005;
                threeScene.rotation.x += deltaMove.y * 0.005;
                
                previousMousePosition = { x: e.clientX, y: e.clientY };
            });
            
            threeRenderer.domElement.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
            threeRenderer.domElement.addEventListener('wheel', (e) => {
                threeCamera.position.z += e.deltaY * 0.01;
                threeCamera.position.z = Math.max(3, Math.min(30, threeCamera.position.z));
            });
            
            // Window resize
            window.addEventListener('resize', () => {
                threeCamera.aspect = container.clientWidth / container.clientHeight;
                threeCamera.updateProjectionMatrix();
                threeRenderer.setSize(container.clientWidth, container.clientHeight);
            });
            
            // Start animation
            animate();
            
            // Load visualization data
            loadVisualizationData();
        }
        
        function createFractal() {
            const complexity = parseInt(document.getElementById('complexity').value) || 2000;
            const mood = parseInt(document.getElementById('mood-influence').value) || 50;
            const energy = parseInt(document.getElementById('energy-flow').value) || 50;
            
            // Remove old points
            if (fractalPoints) {
                threeScene.remove(fractalPoints);
            }
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            
            for (let i = 0; i < complexity; i++) {
                // Golden spiral positioning
                const theta = i * GOLDEN_ANGLE_RAD;
                const r = Math.sqrt(i) * 0.1;
                const phi = (i / complexity) * Math.PI * 2 * (mood / 50);
                
                const x = r * Math.cos(theta) * Math.sin(phi);
                const y = r * Math.sin(theta) * Math.sin(phi);
                const z = r * Math.cos(phi) * (energy / 50);
                
                positions.push(x, y, z);
                
                // Color based on position and parameters
                const hue = (i / complexity + mood / 200) % 1;
                const color = new THREE.Color().setHSL(hue, 0.7, 0.5 + energy / 200);
                colors.push(color.r, color.g, color.b);
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            // Material
            const material = new THREE.PointsMaterial({
                size: 0.05,
                vertexColors: true,
                transparent: true,
                opacity: 0.8
            });
            
            fractalPoints = new THREE.Points(geometry, material);
            threeScene.add(fractalPoints);
        }
        
        function createSacredGeometry() {
            if (sacredGeometry) {
                sacredGeometry.forEach(obj => threeScene.remove(obj));
            }
            sacredGeometry = [];
            
            // Icosahedron (Platonic solid)
            const icoGeometry = new THREE.IcosahedronGeometry(3, 0);
            const icoMaterial = new THREE.MeshBasicMaterial({
                color: 0x6B8E9F,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            });
            const icosahedron = new THREE.Mesh(icoGeometry, icoMaterial);
            threeScene.add(icosahedron);
            sacredGeometry.push(icosahedron);
            
            // Dodecahedron
            const dodecaGeometry = new THREE.DodecahedronGeometry(2.5, 0);
            const dodecaMaterial = new THREE.MeshBasicMaterial({
                color: 0x9F8E6B,
                wireframe: true,
                transparent: true,
                opacity: 0.2
            });
            const dodecahedron = new THREE.Mesh(dodecaGeometry, dodecaMaterial);
            threeScene.add(dodecahedron);
            sacredGeometry.push(dodecahedron);
        }
        
        function animate() {
            animationId = requestAnimationFrame(animate);
            
            if (autoRotate) {
                threeScene.rotation.y += 0.002;
            }
            
            // Animate sacred geometry
            if (sacredGeometry) {
                sacredGeometry.forEach((obj, i) => {
                    obj.rotation.x += 0.001 * (i + 1);
                    obj.rotation.z += 0.0005 * (i + 1);
                });
            }
            
            // Animate goal orbs
            goalOrbs.forEach((orb, i) => {
                const time = Date.now() * 0.001;
                orb.position.y = orb.userData.baseY + Math.sin(time + i) * 0.2;
            });
            
            threeRenderer.render(threeScene, threeCamera);
        }
        
        async function loadVisualizationData() {
            try {
                const res = await fetch('/api/visualization/data');
                const data = await res.json();
                
                if (res.ok && data.orbs) {
                    // Clear existing orbs
                    goalOrbs.forEach(orb => threeScene.remove(orb));
                    goalOrbs = [];
                    
                    // Create new orbs
                    data.orbs.forEach(orbData => {
                        const geometry = new THREE.SphereGeometry(orbData.size, 32, 32);
                        const color = new THREE.Color(orbData.color);
                        const material = new THREE.MeshBasicMaterial({
                            color: color,
                            transparent: true,
                            opacity: 0.8
                        });
                        
                        const orb = new THREE.Mesh(geometry, material);
                        orb.position.set(orbData.position.x, orbData.position.y, orbData.position.z);
                        orb.userData = { ...orbData, baseY: orbData.position.y };
                        
                        threeScene.add(orb);
                        goalOrbs.push(orb);
                    });
                    
                    document.getElementById('orbs-val').textContent = goalOrbs.length;
                    
                    // Update sliders with user data
                    if (data.mood) {
                        document.getElementById('mood-influence').value = data.mood;
                        document.getElementById('mood-influence-val').textContent = Math.round(data.mood);
                    }
                    if (data.energy) {
                        document.getElementById('energy-flow').value = data.energy;
                        document.getElementById('energy-flow-val').textContent = Math.round(data.energy);
                    }
                }
            } catch (e) {
                console.error('Error loading visualization data:', e);
            }
        }
        
        function regenerateFractal() {
            createFractal();
        }
        
        function addGoalOrb() {
            const geometry = new THREE.SphereGeometry(0.3, 32, 32);
            const material = new THREE.MeshBasicMaterial({
                color: Math.random() * 0xffffff,
                transparent: true,
                opacity: 0.8
            });
            
            const orb = new THREE.Mesh(geometry, material);
            
            // Position using golden angle
            const i = goalOrbs.length;
            const theta = i * GOLDEN_ANGLE_RAD;
            const r = 3 + i * 0.5;
            
            orb.position.set(
                r * Math.cos(theta),
                Math.random() * 2 - 1,
                r * Math.sin(theta)
            );
            orb.userData = { baseY: orb.position.y };
            
            threeScene.add(orb);
            goalOrbs.push(orb);
            
            document.getElementById('orbs-val').textContent = goalOrbs.length;
        }
        
        function toggleSacredGeometry() {
            sacredGeometry.forEach(obj => {
                obj.visible = !obj.visible;
            });
        }
        
        function toggleAutoRotate() {
            autoRotate = document.getElementById('auto-rotate').checked;
        }
        
        function resetCamera() {
            threeCamera.position.set(0, 0, 10);
            threeScene.rotation.set(0, 0, 0);
        }
        
        function updateFractalParam(type) {
            const value = document.getElementById(type === 'mood' ? 'mood-influence' : 
                         type === 'energy' ? 'energy-flow' : 'complexity').value;
            document.getElementById(type === 'mood' ? 'mood-influence-val' : 
                         type === 'energy' ? 'energy-flow-val' : 'complexity-val').textContent = value;
            
            createFractal();
        }
        
        function generate2DFractal() {
            alert('2D fractal generation coming soon! For now, try the immersive 3D view.');
        }
        
        // ============================================================
        // INITIALIZATION
        // ============================================================
        document.addEventListener('DOMContentLoaded', checkSession);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !document.getElementById('fractal-view').classList.contains('hidden')) {
                e.preventDefault();
                document.getElementById('auto-rotate').checked = !autoRotate;
                autoRotate = !autoRotate;
            }
            if (e.code === 'Escape') {
                if (!document.getElementById('accessibility-modal').classList.contains('hidden')) {
                    closeAccessibilityModal();
                }
                if (!document.getElementById('fractal-view').classList.contains('hidden')) {
                    close3DView();
                }
            }
        });
    </script>
</body>
</html>
'''

# ════════════════════════════════════════════════════════════════════════════════
# MAIN PAGE ROUTE
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve main application."""
    return render_template_string(MAIN_HTML)

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '8.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'features': {
            'accessibility': True,
            'user_management': True,
            '3d_visualization': True,
            'pet_system': True,
            'mayan_calendar': True,
            'spoon_theory': True
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 70)
    print("🌀 LIFE FRACTAL INTELLIGENCE v8.0 - COMPLETE PRODUCTION BUILD")
    print("=" * 70)
    print("   For brains like mine - built with love for the neurodivergent community")
    print("=" * 70)
    print(f"✨ Golden Ratio (φ):     {PHI:.15f}")
    print(f"🌻 Golden Angle:         {GOLDEN_ANGLE:.10f}°")
    print(f"🔢 Fibonacci:            {FIBONACCI[:10]}...")
    print("=" * 70)
    print("\n🌟 Features:")
    print("   ∞ Neurodiversity-focused accessibility")
    print("   🔐 Complete user management system")
    print("   🎨 Interactive 3D fractal visualization")
    print("   🐱 Virtual pet companion")
    print("   🥄 Spoon theory energy tracking")
    print("   📅 Mayan sacred calendar")
    print("=" * 70)
    print(f"\n💰 Subscription: ${SUBSCRIPTION_PRICE}/month, {TRIAL_DAYS}-day free trial")
    print(f"🎁 GoFundMe: {GOFUNDME_URL}")
    print("=" * 70)


if __name__ == '__main__':
    print_banner()
    
    # Initialize database
    with app.app_context():
        init_db()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
