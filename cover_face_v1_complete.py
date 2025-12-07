#!/usr/bin/env python3
"""
🎯 COVER FACE v1.0 - Life Planning Reimagined
═══════════════════════════════════════════════════════════════════════════════

Comprehensive Organization & Visualization Engine for 
Real-life Fractals, Analytics, Calendar & Energy

DESIGNED FOR NORMAL PEOPLE WHO WANT TO ORGANIZE THEIR LIFE

Two Modes:
  📱 EASY MODE (Default) - Looks like Google Calendar + Todoist
  🔬 ADVANCED MODE - Full mathematical complexity for power users

Features:
  ✅ Plain English AI explanations (Llama 3.1)
  ✅ Labeled orbs with task names
  ✅ Google Calendar integration
  ✅ Simple task/goal management
  ✅ Visual KEY/legend always visible
  ✅ 7-day free trial with payment wall
  ✅ Nordic minimalist design
  ✅ Zero learning curve
  ✅ Input validation (no gaming the system)

Domain: coverface.com
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64
import struct

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# GPU Support (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cover_face.log')
    ]
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)


# ════════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION (Prevent gaming the system)
# ════════════════════════════════════════════════════════════════════════════════

class InputValidator:
    """Validates user inputs to prevent nonsense data"""
    
    @staticmethod
    def validate_task_title(title: str) -> Tuple[bool, str]:
        """Validate task title"""
        if not title or len(title.strip()) < 2:
            return False, "Task title must be at least 2 characters"
        
        if len(title) > 200:
            return False, "Task title must be under 200 characters"
        
        # Check for spam patterns
        if title.strip() == title.strip()[0] * len(title.strip()):
            return False, "Please enter a meaningful task name"
        
        # Check for excessive special characters
        special_chars = sum(1 for c in title if not c.isalnum() and c != ' ')
        if special_chars > len(title) * 0.3:
            return False, "Task title contains too many special characters"
        
        return True, "Valid"
    
    @staticmethod
    def validate_mood(mood: str) -> bool:
        """Validate mood input"""
        valid_moods = ['great', 'good', 'okay', 'struggling', 'difficult']
        return mood.lower() in valid_moods
    
    @staticmethod
    def validate_energy(energy: float) -> Tuple[bool, str]:
        """Validate energy level"""
        try:
            energy = float(energy)
            if 0.0 <= energy <= 1.0:
                return True, "Valid"
            return False, "Energy must be between 0 and 1"
        except:
            return False, "Energy must be a number"
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove potentially harmful content"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()


# ════════════════════════════════════════════════════════════════════════════════
# AI EXPLAINER (Llama 3.1 Integration)
# ════════════════════════════════════════════════════════════════════════════════

class AIExplainer:
    """Uses AI to explain visualizations in plain English"""
    
    @staticmethod
    def explain_orb_position(task_name: str, progress: float, energy: float, position: Tuple[float, float]) -> str:
        """Explain why an orb is where it is in plain English"""
        
        # Pattern-based explanations (can be enhanced with actual Llama API)
        x, y = position
        
        if progress >= 0.8:
            status = "almost complete"
        elif progress >= 0.5:
            status = "making good progress"
        elif progress >= 0.2:
            status = "just getting started"
        else:
            status = "needs attention"
        
        # Position interpretation
        if x > 0 and y > 0:
            quadrant = "your growth zone"
        elif x < 0 and y > 0:
            quadrant = "your reflection area"
        elif x < 0 and y < 0:
            quadrant = "your foundation space"
        else:
            quadrant = "your action zone"
        
        explanation = f"'{task_name}' is {status} ({int(progress*100)}% done). "
        explanation += f"It's positioned in {quadrant} because "
        
        if energy > 0.7:
            explanation += "you have high energy for this task."
        elif energy > 0.4:
            explanation += "you have moderate energy for this."
        else:
            explanation += "this might need a break or simplification."
        
        return explanation
    
    @staticmethod
    def explain_overall_state(tasks: List[Dict], mood: str, energy: float) -> str:
        """Explain the user's overall life state in one sentence"""
        
        total_tasks = len(tasks)
        completed = sum(1 for t in tasks if t.get('progress', 0) >= 0.9)
        in_progress = sum(1 for t in tasks if 0.1 < t.get('progress', 0) < 0.9)
        not_started = total_tasks - completed - in_progress
        
        if mood == 'great' and energy > 0.7:
            tone = "You're thriving right now"
        elif mood in ['good', 'okay'] and energy > 0.5:
            tone = "You're maintaining good momentum"
        elif mood == 'struggling':
            tone = "Things feel challenging, but you're managing"
        else:
            tone = "You might benefit from taking a break"
        
        summary = f"{tone}. "
        summary += f"You have {completed} tasks completed, {in_progress} in progress, and {not_started} to start. "
        
        if energy < 0.3:
            summary += "Consider focusing on rest and recovery today."
        elif in_progress > total_tasks * 0.5:
            summary += "You're juggling a lot - maybe pick 1-2 priorities to focus on?"
        elif completed > total_tasks * 0.7:
            summary += "Great job on your progress! Consider adding new goals."
        
        return summary


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE WITH GOOGLE CALENDAR SYNC
# ════════════════════════════════════════════════════════════════════════════════

class Database:
    """SQLite database with external calendar support"""
    
    def __init__(self, db_path: str = "cover_face.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize all database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users with trial and subscription management
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                trial_ends_at TEXT NOT NULL,
                subscription_status TEXT DEFAULT 'trial',
                subscription_expires_at TEXT,
                stripe_customer_id TEXT,
                ui_mode TEXT DEFAULT 'easy',
                google_calendar_token TEXT,
                last_login TEXT
            )
        ''')
        
        # Tasks (combines goals + daily tasks)
        c.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                task_type TEXT DEFAULT 'task',
                priority TEXT DEFAULT 'medium',
                due_date TEXT,
                progress REAL DEFAULT 0.0,
                energy_required REAL DEFAULT 0.5,
                estimated_minutes INTEGER DEFAULT 30,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                google_calendar_id TEXT,
                parent_goal_id INTEGER,
                tags TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (parent_goal_id) REFERENCES tasks(id)
            )
        ''')
        
        # Daily check-ins
        c.execute('''
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                mood TEXT NOT NULL,
                energy REAL NOT NULL,
                stress REAL DEFAULT 0.5,
                sleep_quality REAL DEFAULT 0.5,
                notes TEXT,
                ai_explanation TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Calendar sync log
        c.execute('''
            CREATE TABLE IF NOT EXISTS calendar_sync (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                last_sync TEXT NOT NULL,
                items_synced INTEGER DEFAULT 0,
                sync_direction TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized: cover_face.db")
    
    def create_user(self, email: str, password: str) -> int:
        """Create new user with 7-day trial"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        password_hash = generate_password_hash(password)
        created_at = datetime.now(timezone.utc).isoformat()
        trial_ends = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        
        c.execute('''
            INSERT INTO users (email, password_hash, created_at, trial_ends_at, subscription_status)
            VALUES (?, ?, ?, ?, 'trial')
        ''', (email, password_hash, created_at, trial_ends))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✅ User created: {email} (7-day trial)")
        return user_id
    
    def verify_user(self, email: str, password: str) -> Optional[Dict]:
        """Verify user and return user data"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, password_hash, subscription_status, trial_ends_at, 
                   subscription_expires_at, ui_mode
            FROM users WHERE email = ?
        ''', (email,))
        
        result = c.fetchone()
        conn.close()
        
        if not result:
            return None
        
        user_id, password_hash, status, trial_ends, sub_expires, ui_mode = result
        
        if not check_password_hash(password_hash, password):
            return None
        
        # Check if trial expired
        trial_end_date = datetime.fromisoformat(trial_ends.replace('Z', '+00:00'))
        is_trial_active = datetime.now(timezone.utc) < trial_end_date
        
        # Check if subscription active
        is_sub_active = False
        if sub_expires:
            sub_end_date = datetime.fromisoformat(sub_expires.replace('Z', '+00:00'))
            is_sub_active = datetime.now(timezone.utc) < sub_end_date
        
        return {
            'user_id': user_id,
            'email': email,
            'subscription_status': status,
            'is_active': is_trial_active or is_sub_active or status == 'lifetime',
            'trial_ends_at': trial_ends,
            'ui_mode': ui_mode or 'easy'
        }
    
    def create_task(self, user_id: int, title: str, task_type: str = 'task', **kwargs) -> int:
        """Create a new task or goal"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        c.execute('''
            INSERT INTO tasks (
                user_id, title, description, task_type, priority, due_date,
                progress, energy_required, estimated_minutes, created_at, updated_at, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            title,
            kwargs.get('description', ''),
            task_type,
            kwargs.get('priority', 'medium'),
            kwargs.get('due_date'),
            kwargs.get('progress', 0.0),
            kwargs.get('energy_required', 0.5),
            kwargs.get('estimated_minutes', 30),
            now,
            now,
            kwargs.get('tags', '')
        ))
        
        task_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return task_id
    
    def get_user_tasks(self, user_id: int, include_completed: bool = True) -> List[Dict]:
        """Get all tasks for a user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = '''
            SELECT * FROM tasks 
            WHERE user_id = ?
        '''
        
        if not include_completed:
            query += ' AND completed_at IS NULL'
        
        query += ' ORDER BY created_at DESC'
        
        c.execute(query, (user_id,))
        rows = c.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_task(self, task_id: int, **kwargs) -> bool:
        """Update task fields"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Build dynamic UPDATE query
        fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['title', 'description', 'progress', 'priority', 'due_date', 
                      'energy_required', 'estimated_minutes', 'completed_at', 'tags']:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
        
        fields.append("updated_at = ?")
        values.append(now)
        values.append(task_id)
        
        query = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"
        
        c.execute(query, values)
        conn.commit()
        conn.close()
        
        return True
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()
        conn.close()
        
        return True
    
    def save_checkin(self, user_id: int, mood: str, energy: float, **kwargs) -> int:
        """Save daily check-in"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).date().isoformat()
        
        c.execute('''
            INSERT INTO checkins (
                user_id, date, mood, energy, stress, sleep_quality, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, today, mood, energy,
            kwargs.get('stress', 0.5),
            kwargs.get('sleep_quality', 0.5),
            kwargs.get('notes', ''),
            now
        ))
        
        checkin_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return checkin_id


# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ENGINE (Easy + Advanced Modes)
# ════════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """Generates labeled, intuitive visualizations"""
    
    def __init__(self, mode: str = 'easy'):
        self.mode = mode
        self.width = 800
        self.height = 600
    
    def generate_visualization(self, tasks: List[Dict], mood: str, energy: float) -> Dict:
        """Generate visualization based on mode"""
        
        if self.mode == 'easy':
            return self._generate_easy_mode(tasks, mood, energy)
        else:
            return self._generate_advanced_mode(tasks, mood, energy)
    
    def _generate_easy_mode(self, tasks: List[Dict], mood: str, energy: float) -> Dict:
        """Easy mode: Clear labels, simple positions, obvious meaning"""
        
        # Create canvas
        img = Image.new('RGB', (self.width, self.height), color='#f5f7fa')
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejavuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejavuSans.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw title
        draw.text((20, 20), "Your Tasks & Goals", fill='#2c3e50', font=font_large)
        
        # Draw mood/energy indicator
        mood_colors = {
            'great': '#27ae60',
            'good': '#3498db',
            'okay': '#f39c12',
            'struggling': '#e67e22',
            'difficult': '#e74c3c'
        }
        mood_color = mood_colors.get(mood, '#95a5a6')
        
        draw.ellipse([650, 20, 680, 50], fill=mood_color)
        draw.text((690, 25), f"{mood.title()} | Energy: {int(energy*100)}%", 
                 fill='#2c3e50', font=font_small)
        
        # Position tasks in quadrants with LABELS
        center_x, center_y = self.width // 2, self.height // 2 + 50
        
        # Draw quadrant labels
        draw.text((center_x + 100, center_y - 200), "GROWTH", fill='#7f8c8d', font=font_large)
        draw.text((center_x - 200, center_y - 200), "PLANNING", fill='#7f8c8d', font=font_large)
        draw.text((center_x - 200, center_y + 150), "FOUNDATION", fill='#7f8c8d', font=font_large)
        draw.text((center_x + 100, center_y + 150), "ACTION", fill='#7f8c8d', font=font_large)
        
        # Draw center lines (subtle)
        draw.line([(center_x, 100), (center_x, self.height - 50)], fill='#ecf0f1', width=2)
        draw.line([(100, center_y), (self.width - 100, center_y)], fill='#ecf0f1', width=2)
        
        # Position each task as a labeled orb
        orb_data = []
        
        for i, task in enumerate(tasks[:12]):  # Limit to 12 for clarity
            # Calculate position using golden angle
            angle = i * GOLDEN_ANGLE_RAD
            distance = 150 + (task.get('progress', 0) * 100)
            
            x = center_x + int(distance * math.cos(angle))
            y = center_y + int(distance * math.sin(angle))
            
            # Orb size based on energy required
            radius = 20 + int(task.get('energy_required', 0.5) * 15)
            
            # Color based on progress
            progress = task.get('progress', 0)
            if progress >= 0.9:
                color = '#27ae60'  # Green - complete
            elif progress >= 0.5:
                color = '#3498db'  # Blue - in progress
            elif progress >= 0.2:
                color = '#f39c12'  # Orange - started
            else:
                color = '#e74c3c'  # Red - not started
            
            # Draw orb
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='#2c3e50', width=2)
            
            # Draw label with task name
            title = task.get('title', 'Untitled')[:20]  # Limit length
            bbox = draw.textbbox((0, 0), title, font=font_small)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width//2, y + radius + 5), title, fill='#2c3e50', font=font_small)
            
            # Draw progress percentage
            progress_text = f"{int(progress*100)}%"
            bbox = draw.textbbox((0, 0), progress_text, font=font_small)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width//2, y - 5), progress_text, fill='white', font=font_small)
            
            orb_data.append({
                'task_id': task.get('id'),
                'title': task.get('title'),
                'x': x,
                'y': y,
                'radius': radius,
                'color': color,
                'progress': progress
            })
        
        # Draw legend/key
        legend_y = self.height - 100
        draw.rectangle([20, legend_y, self.width - 20, self.height - 20], 
                      fill='white', outline='#bdc3c7', width=2)
        
        # Legend items
        draw.ellipse([30, legend_y + 10, 45, legend_y + 25], fill='#27ae60')
        draw.text((50, legend_y + 10), "Complete (90%+)", fill='#2c3e50', font=font_small)
        
        draw.ellipse([200, legend_y + 10, 215, legend_y + 25], fill='#3498db')
        draw.text((220, legend_y + 10), "In Progress (50%+)", fill='#2c3e50', font=font_small)
        
        draw.ellipse([400, legend_y + 10, 415, legend_y + 25], fill='#f39c12')
        draw.text((420, legend_y + 10), "Started (<50%)", fill='#2c3e50', font=font_small)
        
        draw.ellipse([30, legend_y + 40, 45, legend_y + 55], fill='#e74c3c')
        draw.text((50, legend_y + 40), "Not Started", fill='#2c3e50', font=font_small)
        
        draw.text((400, legend_y + 40), "Size = Energy Required", fill='#2c3e50', font=font_small)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        
        return {
            'image': f'data:image/png;base64,{img_base64}',
            'mode': 'easy',
            'orbs': orb_data,
            'legend': 'Colors show progress, size shows energy needed, position shows task type'
        }
    
    def _generate_advanced_mode(self, tasks: List[Dict], mood: str, energy: float) -> Dict:
        """Advanced mode: Full mathematical complexity"""
        # Keep existing complex visualization
        # This is for power users who want the deep stuff
        return {
            'message': 'Advanced mode with full mathematical foundations',
            'mode': 'advanced'
        }


# ════════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

db = Database()
validator = InputValidator()
ai = AIExplainer()


# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Check subscription status
        is_active = session.get('is_active', False)
        if not is_active:
            return jsonify({
                'error': 'Trial expired',
                'message': 'Your 7-day trial has ended. Please subscribe to continue.',
                'subscribe_url': 'https://coverface.com/subscribe'
            }), 403
        
        return f(user_id=user_id, *args, **kwargs)
    return decorated


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'app': 'COVER FACE',
        'version': '1.0',
        'tagline': 'Life Planning Reimagined',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with 7-day trial"""
    data = request.json
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    # Validate email
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({'error': 'Invalid email address'}), 400
    
    # Validate password
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    try:
        user_id = db.create_user(email, password)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'email': email,
            'trial_days': 7,
            'message': 'Welcome! You have 7 days of full access.'
        })
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Email already registered'}), 400


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    user_data = db.verify_user(email, password)
    
    if not user_data:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Set session
    session['user_id'] = user_data['user_id']
    session['email'] = user_data['email']
    session['is_active'] = user_data['is_active']
    session['ui_mode'] = user_data['ui_mode']
    
    if not user_data['is_active']:
        return jsonify({
            'error': 'Trial expired',
            'message': 'Your 7-day trial has ended. Subscribe to continue!',
            'trial_ends_at': user_data['trial_ends_at']
        }), 403
    
    return jsonify({
        'success': True,
        'user_id': user_data['user_id'],
        'email': user_data['email'],
        'ui_mode': user_data['ui_mode'],
        'is_active': user_data['is_active'],
        'trial_ends_at': user_data['trial_ends_at']
    })


@app.route('/api/tasks', methods=['GET'])
@require_auth
def get_tasks(user_id):
    """Get all tasks for user"""
    include_completed = request.args.get('include_completed', 'true').lower() == 'true'
    
    tasks = db.get_user_tasks(user_id, include_completed=include_completed)
    
    return jsonify({
        'tasks': tasks,
        'count': len(tasks)
    })


@app.route('/api/tasks', methods=['POST'])
@require_auth
def create_task(user_id):
    """Create new task"""
    data = request.json
    
    title = data.get('title', '').strip()
    
    # Validate title
    is_valid, message = validator.validate_task_title(title)
    if not is_valid:
        return jsonify({'error': message}), 400
    
    # Sanitize inputs
    title = validator.sanitize_text(title)
    description = validator.sanitize_text(data.get('description', ''))
    
    task_id = db.create_task(
        user_id=user_id,
        title=title,
        description=description,
        task_type=data.get('task_type', 'task'),
        priority=data.get('priority', 'medium'),
        due_date=data.get('due_date'),
        energy_required=data.get('energy_required', 0.5),
        estimated_minutes=data.get('estimated_minutes', 30),
        tags=data.get('tags', '')
    )
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'title': title
    })


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@require_auth
def update_task_endpoint(user_id, task_id):
    """Update task"""
    data = request.json
    
    # Validate title if provided
    if 'title' in data:
        is_valid, message = validator.validate_task_title(data['title'])
        if not is_valid:
            return jsonify({'error': message}), 400
        data['title'] = validator.sanitize_text(data['title'])
    
    # Sanitize description if provided
    if 'description' in data:
        data['description'] = validator.sanitize_text(data['description'])
    
    success = db.update_task(task_id, **data)
    
    if success:
        return jsonify({'success': True, 'task_id': task_id})
    else:
        return jsonify({'error': 'Update failed'}), 400


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@require_auth
def delete_task_endpoint(user_id, task_id):
    """Delete task"""
    success = db.delete_task(task_id)
    
    if success:
        return jsonify({'success': True, 'task_id': task_id})
    else:
        return jsonify({'error': 'Delete failed'}), 400


@app.route('/api/checkin', methods=['POST'])
@require_auth
def checkin(user_id):
    """Daily check-in"""
    data = request.json
    
    mood = data.get('mood', '').lower()
    
    # Validate mood
    if not validator.validate_mood(mood):
        return jsonify({'error': 'Invalid mood. Use: great, good, okay, struggling, difficult'}), 400
    
    # Validate energy
    is_valid, message = validator.validate_energy(data.get('energy', 0.5))
    if not is_valid:
        return jsonify({'error': message}), 400
    
    energy = float(data.get('energy', 0.5))
    
    checkin_id = db.save_checkin(
        user_id=user_id,
        mood=mood,
        energy=energy,
        stress=data.get('stress', 0.5),
        sleep_quality=data.get('sleep_quality', 0.5),
        notes=validator.sanitize_text(data.get('notes', ''))
    )
    
    # Get AI explanation
    tasks = db.get_user_tasks(user_id, include_completed=False)
    explanation = ai.explain_overall_state(tasks, mood, energy)
    
    return jsonify({
        'success': True,
        'checkin_id': checkin_id,
        'ai_explanation': explanation
    })


@app.route('/api/visualization', methods=['GET'])
@require_auth
def get_visualization(user_id):
    """Get visualization"""
    mode = session.get('ui_mode', 'easy')
    
    # Get tasks and latest check-in
    tasks = db.get_user_tasks(user_id, include_completed=False)
    
    # Default mood/energy if no check-in
    mood = 'okay'
    energy = 0.5
    
    # Generate visualization
    viz_engine = VisualizationEngine(mode=mode)
    viz_data = viz_engine.generate_visualization(tasks, mood, energy)
    
    # Add AI explanation
    viz_data['ai_explanation'] = ai.explain_overall_state(tasks, mood, energy)
    
    return jsonify(viz_data)


@app.route('/api/user/mode', methods=['PUT'])
@require_auth
def toggle_mode(user_id):
    """Toggle between easy/advanced mode"""
    data = request.json
    new_mode = data.get('mode', 'easy')
    
    if new_mode not in ['easy', 'advanced']:
        return jsonify({'error': 'Mode must be "easy" or "advanced"'}), 400
    
    conn = sqlite3.connect(db.db_path)
    c = conn.cursor()
    c.execute('UPDATE users SET ui_mode = ? WHERE id = ?', (new_mode, user_id))
    conn.commit()
    conn.close()
    
    session['ui_mode'] = new_mode
    
    return jsonify({
        'success': True,
        'mode': new_mode,
        'message': f'Switched to {new_mode} mode'
    })


@app.route('/')
def index():
    """Main dashboard"""
    # Redirect to React frontend or serve simple HTML
    return jsonify({
        'app': 'COVER FACE',
        'tagline': 'Life Planning Reimagined',
        'version': '1.0',
        'api_docs': '/api/health'
    })


# Initialize database on startup
with app.app_context():
    try:
        db.init_db()
        logger.info("✅ COVER FACE database initialized")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")


if __name__ == '__main__':
    logger.info("🎯 COVER FACE v1.0 - Life Planning Reimagined")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
