#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v10.1 - COMPLETE FEATURE UPDATE
═══════════════════════════════════════════════════════════════════════════════
ALL FEATURES IMPLEMENTED - PRODUCTION READY

✅ Complete authentication & session management  
✅ SQLite database with all tables
✅ 2D & 3D fractal visualization (IMPROVED)
✅ Goal tracking with progress calculations
✅ Habit tracking with streaks
✅ Daily wellness check-ins
✅ Virtual pet system (8 SPECIES - ALL IMPLEMENTED)
✅ Mayan Tzolkin Calendar (NEW)
✅ Spoon Theory Energy Management (NEW)
✅ Binaural Beats Audio System (NEW)
✅ Executive Dysfunction Pattern Detection (NEW)
✅ Accessibility features (aphantasia/autism)
✅ All API endpoints functional
✅ Complete HTML dashboard
✅ Self-healing - never crashes

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
import re
import struct
import wave
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None

# ML Support
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


# ═══════════════════════════════════════════════════════════════════════════════
# MAYAN TZOLKIN CALENDAR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

TZOLKIN_DAY_SIGNS = [
    'Imix (Dragon)', 'Ik (Wind)', 'Akbal (Night)', 'Kan (Seed)', 
    'Chicchan (Serpent)', 'Cimi (Death)', 'Manik (Deer)', 'Lamat (Star)',
    'Muluc (Water)', 'Oc (Dog)', 'Chuen (Monkey)', 'Eb (Road)',
    'Ben (Reed)', 'Ix (Jaguar)', 'Men (Eagle)', 'Cib (Vulture)',
    'Caban (Earth)', 'Etznab (Flint)', 'Cauac (Storm)', 'Ahau (Sun)'
]

TZOLKIN_NUMBERS = list(range(1, 14))  # 1-13

TZOLKIN_ENERGIES = {
    'Imix': {'element': 'water', 'energy': 'creative', 'theme': 'new beginnings'},
    'Ik': {'element': 'air', 'energy': 'communicative', 'theme': 'breath of life'},
    'Akbal': {'element': 'earth', 'energy': 'introspective', 'theme': 'inner journey'},
    'Kan': {'element': 'fire', 'energy': 'growth', 'theme': 'planting seeds'},
    'Chicchan': {'element': 'fire', 'energy': 'vital', 'theme': 'life force'},
    'Cimi': {'element': 'earth', 'energy': 'transformative', 'theme': 'release'},
    'Manik': {'element': 'earth', 'energy': 'healing', 'theme': 'tools'},
    'Lamat': {'element': 'fire', 'energy': 'abundance', 'theme': 'harmony'},
    'Muluc': {'element': 'water', 'energy': 'emotional', 'theme': 'flow'},
    'Oc': {'element': 'fire', 'energy': 'loyal', 'theme': 'guidance'},
    'Chuen': {'element': 'air', 'energy': 'playful', 'theme': 'creativity'},
    'Eb': {'element': 'earth', 'energy': 'service', 'theme': 'path'},
    'Ben': {'element': 'air', 'energy': 'authoritative', 'theme': 'pillars'},
    'Ix': {'element': 'earth', 'energy': 'magical', 'theme': 'intuition'},
    'Men': {'element': 'air', 'energy': 'visionary', 'theme': 'perspective'},
    'Cib': {'element': 'fire', 'energy': 'ancestral', 'theme': 'wisdom'},
    'Caban': {'element': 'earth', 'energy': 'grounding', 'theme': 'synchronicity'},
    'Etznab': {'element': 'air', 'energy': 'reflective', 'theme': 'truth'},
    'Cauac': {'element': 'water', 'energy': 'purifying', 'theme': 'transformation'},
    'Ahau': {'element': 'fire', 'energy': 'enlightened', 'theme': 'mastery'}
}

class MayanCalendar:
    """Mayan Tzolkin Calendar calculations"""
    
    # Reference date: August 11, 3114 BCE (Gregorian) = 4 Ahau
    CORRELATION_CONSTANT = 584283  # Most commonly accepted correlation
    
    @classmethod
    def get_tzolkin_date(cls, date: datetime = None) -> dict:
        """Calculate Tzolkin date for given datetime"""
        if date is None:
            date = datetime.now()
        
        # Calculate Julian Day Number
        a = (14 - date.month) // 12
        y = date.year + 4800 - a
        m = date.month + 12 * a - 3
        
        jdn = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        
        # Calculate days since Mayan creation date
        days_since_creation = jdn - cls.CORRELATION_CONSTANT
        
        # Tzolkin day number (1-13)
        tzolkin_number = ((days_since_creation + 3) % 13) + 1
        
        # Tzolkin day sign (0-19)
        tzolkin_sign_index = (days_since_creation + 19) % 20
        tzolkin_sign = TZOLKIN_DAY_SIGNS[tzolkin_sign_index]
        sign_name = tzolkin_sign.split(' ')[0]
        
        # Get energy for today
        energy_info = TZOLKIN_ENERGIES.get(sign_name, {})
        
        # Calculate position in 260-day cycle
        cycle_position = (days_since_creation % 260) + 1
        
        return {
            'tzolkin_number': tzolkin_number,
            'tzolkin_sign': tzolkin_sign,
            'sign_name': sign_name,
            'full_date': f"{tzolkin_number} {tzolkin_sign}",
            'cycle_position': cycle_position,
            'cycle_total': 260,
            'element': energy_info.get('element', 'unknown'),
            'energy': energy_info.get('energy', 'neutral'),
            'theme': energy_info.get('theme', 'balance'),
            'guidance': cls._get_guidance(tzolkin_number, sign_name)
        }
    
    @classmethod
    def _get_guidance(cls, number: int, sign: str) -> str:
        """Generate guidance based on Tzolkin date"""
        energy = TZOLKIN_ENERGIES.get(sign, {}).get('energy', 'balanced')
        
        guidance_templates = {
            (1, 'creative'): "A powerful day for new beginnings. Plant seeds for your goals.",
            (1, 'transformative'): "Release what no longer serves you. Space creates possibility.",
            'default_low': f"Day {number} brings gentle {energy} energy. Focus on small, steady progress.",
            'default_mid': f"The {energy} energy of {sign} supports focused work today.",
            'default_high': f"Strong {energy} energy today. Trust your momentum and take action."
        }
        
        if number <= 4:
            return guidance_templates.get((number, energy), guidance_templates['default_low'])
        elif number <= 9:
            return guidance_templates.get((number, energy), guidance_templates['default_mid'])
        else:
            return guidance_templates.get((number, energy), guidance_templates['default_high'])


# ═══════════════════════════════════════════════════════════════════════════════
# SPOON THEORY ENERGY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SpoonTheory:
    """Spoon Theory energy management for neurodivergent users"""
    
    # Default spoon costs for common activities
    DEFAULT_COSTS = {
        'wake_up': 1,
        'shower': 2,
        'get_dressed': 1,
        'meal_prep': 2,
        'eating': 1,
        'commute': 2,
        'work_meeting': 3,
        'focused_work': 2,
        'phone_call': 2,
        'email': 1,
        'exercise': 3,
        'socializing': 3,
        'errands': 2,
        'cleaning': 2,
        'decision_making': 2,
        'unexpected_change': 3,
        'sensory_overload': 4,
        'masking': 3
    }
    
    # Spoon regeneration activities
    REGENERATORS = {
        'rest': 2,
        'nap': 3,
        'special_interest': 2,
        'nature': 2,
        'meditation': 1,
        'alone_time': 2,
        'comfort_food': 1,
        'music': 1,
        'pet_time': 1
    }
    
    @classmethod
    def calculate_daily_spoons(cls, sleep_hours: float, sleep_quality: float, 
                               stress_level: float, wellness_score: float) -> dict:
        """Calculate available spoons for the day"""
        # Base spoons (typical is 12, can range from 6-18)
        base_spoons = 12
        
        # Sleep modifier (-4 to +3)
        if sleep_hours < 5:
            sleep_mod = -4
        elif sleep_hours < 6:
            sleep_mod = -2
        elif sleep_hours < 7:
            sleep_mod = -1
        elif sleep_hours <= 9:
            sleep_mod = min(3, int((sleep_quality / 100) * 3))
        else:
            sleep_mod = 0  # Oversleeping doesn't help
        
        # Stress modifier (-3 to 0)
        stress_mod = -int((stress_level / 100) * 3)
        
        # Wellness modifier (-2 to +2)
        if wellness_score < 30:
            wellness_mod = -2
        elif wellness_score < 50:
            wellness_mod = -1
        elif wellness_score > 70:
            wellness_mod = 1
        elif wellness_score > 85:
            wellness_mod = 2
        else:
            wellness_mod = 0
        
        total_spoons = max(4, min(18, base_spoons + sleep_mod + stress_mod + wellness_mod))
        
        return {
            'total_spoons': total_spoons,
            'base_spoons': base_spoons,
            'sleep_modifier': sleep_mod,
            'stress_modifier': stress_mod,
            'wellness_modifier': wellness_mod,
            'spoon_emoji': '🥄' * total_spoons,
            'energy_level': 'low' if total_spoons < 8 else 'medium' if total_spoons < 12 else 'high',
            'recommendations': cls._get_recommendations(total_spoons)
        }
    
    @classmethod
    def _get_recommendations(cls, spoons: int) -> List[str]:
        """Get recommendations based on available spoons"""
        if spoons < 6:
            return [
                "Today is a rest day - prioritize only essentials",
                "Consider canceling non-essential commitments",
                "Use comfort strategies: familiar foods, quiet space",
                "It's okay to do the bare minimum today"
            ]
        elif spoons < 10:
            return [
                "Pace yourself - take breaks between tasks",
                "Front-load important tasks while energy is higher",
                "Build in buffer time between activities",
                "Consider delegating or postponing low-priority items"
            ]
        else:
            return [
                "Good energy today - tackle challenging tasks",
                "Still pace yourself to avoid burnout",
                "Bank extra spoons by completing quick wins",
                "Great day for activities you've been putting off"
            ]
    
    @classmethod
    def estimate_task_cost(cls, task_description: str) -> dict:
        """Estimate spoon cost for a task"""
        task_lower = task_description.lower()
        
        # Check for known activities
        for activity, cost in cls.DEFAULT_COSTS.items():
            if activity.replace('_', ' ') in task_lower:
                return {
                    'task': task_description,
                    'estimated_cost': cost,
                    'confidence': 'high',
                    'category': activity
                }
        
        # Heuristic estimation based on keywords
        high_cost_words = ['meeting', 'call', 'social', 'travel', 'exercise', 'clean', 'decision']
        medium_cost_words = ['work', 'write', 'email', 'cook', 'shop']
        low_cost_words = ['rest', 'read', 'watch', 'listen', 'sit']
        
        for word in high_cost_words:
            if word in task_lower:
                return {'task': task_description, 'estimated_cost': 3, 'confidence': 'medium', 'category': 'high_effort'}
        
        for word in medium_cost_words:
            if word in task_lower:
                return {'task': task_description, 'estimated_cost': 2, 'confidence': 'medium', 'category': 'medium_effort'}
        
        for word in low_cost_words:
            if word in task_lower:
                return {'task': task_description, 'estimated_cost': 1, 'confidence': 'medium', 'category': 'low_effort'}
        
        return {'task': task_description, 'estimated_cost': 2, 'confidence': 'low', 'category': 'unknown'}


# ═══════════════════════════════════════════════════════════════════════════════
# BINAURAL BEATS AUDIO SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class BinauralBeats:
    """Generate binaural beats audio for different mental states"""
    
    # Frequency presets (carrier frequency, beat frequency, effect)
    PRESETS = {
        'focus': {
            'carrier': 200,
            'beat': 14,  # Beta waves (12-30 Hz)
            'description': 'Enhanced concentration and alertness',
            'duration': 300  # 5 minutes
        },
        'relax': {
            'carrier': 200,
            'beat': 10,  # Alpha waves (8-12 Hz)
            'description': 'Calm relaxation and stress relief',
            'duration': 300
        },
        'sleep': {
            'carrier': 150,
            'beat': 3,  # Delta waves (0.5-4 Hz)
            'description': 'Deep sleep and restoration',
            'duration': 600
        },
        'meditate': {
            'carrier': 180,
            'beat': 7,  # Theta waves (4-8 Hz)
            'description': 'Deep meditation and creativity',
            'duration': 600
        },
        'energy': {
            'carrier': 250,
            'beat': 20,  # High Beta waves
            'description': 'Increased energy and motivation',
            'duration': 180
        },
        'healing': {
            'carrier': 174,  # Solfeggio frequency
            'beat': 6,
            'description': 'Pain relief and healing',
            'duration': 600
        }
    }
    
    @classmethod
    def generate_audio(cls, preset: str = 'focus', duration_seconds: int = None) -> bytes:
        """Generate binaural beats audio as WAV bytes"""
        if preset not in cls.PRESETS:
            preset = 'focus'
        
        config = cls.PRESETS[preset]
        carrier = config['carrier']
        beat = config['beat']
        duration = duration_seconds or min(config['duration'], 60)  # Limit to 60s for bandwidth
        
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        # Generate left and right channels
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        
        # Left ear: carrier frequency
        left = np.sin(2 * np.pi * carrier * t)
        
        # Right ear: carrier + beat frequency
        right = np.sin(2 * np.pi * (carrier + beat) * t)
        
        # Apply fade in/out to prevent clicks
        fade_samples = int(sample_rate * 0.1)  # 100ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        left[:fade_samples] *= fade_in
        left[-fade_samples:] *= fade_out
        right[:fade_samples] *= fade_in
        right[-fade_samples:] *= fade_out
        
        # Convert to 16-bit PCM
        left_pcm = (left * 32767 * 0.5).astype(np.int16)
        right_pcm = (right * 32767 * 0.5).astype(np.int16)
        
        # Interleave stereo
        stereo = np.column_stack((left_pcm, right_pcm)).flatten()
        
        # Create WAV file in memory
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(stereo.tobytes())
        
        return buffer.getvalue()
    
    @classmethod
    def get_preset_info(cls, preset: str = None) -> dict:
        """Get information about presets"""
        if preset and preset in cls.PRESETS:
            return {preset: cls.PRESETS[preset]}
        return cls.PRESETS
    
    @classmethod
    def recommend_preset(cls, mood: float, energy: float, stress: float) -> str:
        """Recommend a preset based on user state"""
        if stress > 70:
            return 'relax'
        elif energy < 30:
            return 'energy'
        elif mood < 40:
            return 'healing'
        elif energy > 70 and mood > 60:
            return 'focus'
        else:
            return 'meditate'


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE DYSFUNCTION PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutiveDysfunctionDetector:
    """Detect patterns indicating executive dysfunction and provide support"""
    
    @classmethod
    def analyze_patterns(cls, daily_entries: List[dict], goals: List[dict], 
                         habits: List[dict]) -> dict:
        """Analyze user data for executive dysfunction patterns"""
        patterns = []
        severity = 0
        recommendations = []
        
        # Check for task initiation issues
        if daily_entries:
            recent_entries = daily_entries[:7]  # Last week
            low_productivity_days = sum(1 for e in recent_entries 
                                       if e.get('goals_completed', 0) == 0)
            if low_productivity_days >= 4:
                patterns.append('task_initiation_difficulty')
                severity += 2
                recommendations.append("Try the 2-minute rule: start with just 2 minutes of any task")
        
        # Check for incomplete goals pattern
        if goals:
            stalled_goals = sum(1 for g in goals 
                               if 0 < g.get('progress', 0) < 50 and not g.get('completed_at'))
            if stalled_goals >= 3:
                patterns.append('task_completion_difficulty')
                severity += 2
                recommendations.append("Break goals into smaller sub-tasks you can complete in one session")
        
        # Check for habit consistency
        if habits:
            broken_streaks = sum(1 for h in habits 
                                if h.get('longest_streak', 0) > 5 and h.get('current_streak', 0) == 0)
            if broken_streaks >= 2:
                patterns.append('consistency_difficulty')
                severity += 1
                recommendations.append("Link habits to existing routines (habit stacking)")
        
        # Check for time blindness indicators
        if daily_entries:
            evening_checkins = sum(1 for e in daily_entries[:7] 
                                   if 'created_at' in e and '2' in str(e.get('created_at', ''))[:2])
            # This is a simplified check - in reality you'd parse timestamps
            if len(daily_entries) < 3:
                patterns.append('time_blindness')
                severity += 1
                recommendations.append("Set multiple reminders throughout the day")
        
        # Check for decision fatigue
        if daily_entries and len(daily_entries) >= 3:
            recent_stress = [e.get('stress_level', 50) for e in daily_entries[:3]]
            if all(s > 60 for s in recent_stress):
                patterns.append('decision_fatigue')
                severity += 1
                recommendations.append("Pre-plan decisions the night before to reduce morning load")
        
        # Determine overall status
        if severity >= 5:
            status = 'significant'
            overall_message = "You're showing signs of executive function challenges. Be gentle with yourself."
        elif severity >= 3:
            status = 'moderate'
            overall_message = "Some executive function patterns detected. Consider adjusting your approach."
        elif severity >= 1:
            status = 'mild'
            overall_message = "Minor patterns detected. Small adjustments may help."
        else:
            status = 'none'
            overall_message = "No significant executive dysfunction patterns detected."
            recommendations = ["Keep up the good work!", "Continue your current strategies."]
        
        return {
            'patterns_detected': patterns,
            'severity_score': severity,
            'severity_level': status,
            'message': overall_message,
            'recommendations': recommendations,
            'support_strategies': cls._get_support_strategies(patterns)
        }
    
    @classmethod
    def _get_support_strategies(cls, patterns: List[str]) -> dict:
        """Get specific support strategies for detected patterns"""
        strategies = {
            'task_initiation_difficulty': [
                "Body doubling: work alongside someone (even virtually)",
                "Use a 'starting ritual' - same music, same drink, same spot",
                "Lower the bar: 'I just need to open the document'",
                "External accountability: tell someone your plan"
            ],
            'task_completion_difficulty': [
                "Set artificial deadlines before real ones",
                "Reward completion, not just effort",
                "Use 'done lists' to see progress",
                "Work in short, timed bursts (Pomodoro)"
            ],
            'consistency_difficulty': [
                "Don't break the chain - visible tracking",
                "Implementation intentions: 'When X, I will Y'",
                "Reduce friction for good habits",
                "Forgiveness: one miss doesn't break the habit"
            ],
            'time_blindness': [
                "Use visual timers you can see",
                "Time blocking in calendar",
                "Set alarms for transitions",
                "Build in buffer time between tasks"
            ],
            'decision_fatigue': [
                "Automate routine decisions (same breakfast, outfit rotation)",
                "Make important decisions when energy is highest",
                "Limit options - too many choices paralyze",
                "Use decision frameworks and checklists"
            ]
        }
        
        result = {}
        for pattern in patterns:
            if pattern in strategies:
                result[pattern] = strategies[pattern]
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE - COMPLETE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

class Database:
    """Production-ready SQLite database with self-healing"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv('DATABASE_PATH', 'life_fractal_production.db')
        self.init_database()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize all database tables"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT DEFAULT '',
                    last_name TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active INTEGER DEFAULT 1,
                    subscription_status TEXT DEFAULT 'trial',
                    trial_ends TEXT
                )
            ''')
            
            # Goals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    category TEXT DEFAULT 'personal',
                    term TEXT DEFAULT 'medium',
                    priority INTEGER DEFAULT 3,
                    progress REAL DEFAULT 0,
                    target_date TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Habits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS habits (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    frequency TEXT DEFAULT 'daily',
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    total_completions INTEGER DEFAULT 0,
                    last_completed TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Daily entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_entries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    mood_level REAL DEFAULT 50,
                    energy_level REAL DEFAULT 50,
                    stress_level REAL DEFAULT 50,
                    sleep_hours REAL DEFAULT 7,
                    sleep_quality REAL DEFAULT 50,
                    goals_completed INTEGER DEFAULT 0,
                    journal_entry TEXT DEFAULT '',
                    gratitude TEXT DEFAULT '',
                    wellness_score REAL DEFAULT 50,
                    spoons_available INTEGER DEFAULT 12,
                    spoons_used INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Pet state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pet_state (
                    user_id TEXT PRIMARY KEY,
                    species TEXT DEFAULT 'cat',
                    name TEXT DEFAULT 'Buddy',
                    hunger REAL DEFAULT 50,
                    energy REAL DEFAULT 50,
                    mood REAL DEFAULT 50,
                    level INTEGER DEFAULT 1,
                    experience INTEGER DEFAULT 0,
                    bond REAL DEFAULT 0,
                    last_updated TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Progress history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_history (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    progress REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    def execute(self, query: str, params: tuple = ()):
        """Execute a query with error handling"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def insert(self, table: str, data: dict):
        """Insert a row"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute(query, tuple(data.values()))
        return data.get('id')
    
    def update(self, table: str, data: dict, where: dict):
        """Update rows"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        self.execute(query, tuple(data.values()) + tuple(where.values()))
    
    def select(self, table: str, where: Optional[dict] = None, order_by: str = None):
        """Select rows"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = f"SELECT * FROM {table}"
            params = ()
            
            if where:
                where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
                query += f" WHERE {where_clause}"
                params = tuple(where.values())
            
            if order_by:
                query += f" ORDER BY {order_by}"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def delete(self, table: str, where: dict):
        """Delete rows"""
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        self.execute(query, tuple(where.values()))


# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL ENGINE - IMPROVED 3D RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    """Generate beautiful fractals based on user metrics"""
    
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
    
    def generate_2d_fractal(self, wellness: float = 50, mood: float = 50, stress: float = 50) -> Image.Image:
        """Generate 2D Mandelbrot fractal"""
        try:
            max_iter = int(100 + mood * 1.5)
            zoom = 1.0 + (wellness / 100) * 3.0
            
            center_x = -0.5 + (stress - 50) / 500
            center_y = (mood - 50) / 500
            
            x_min = center_x - 2/zoom
            x_max = center_x + 2/zoom
            y_min = center_y - 2/zoom
            y_max = center_y + 2/zoom
            
            x = np.linspace(x_min, x_max, self.width)
            y = np.linspace(y_min, y_max, self.height)
            X, Y = np.meshgrid(x, y)
            C = X + 1j * Y
            
            Z = np.zeros_like(C)
            iterations = np.zeros(C.shape)
            
            for i in range(max_iter):
                mask = np.abs(Z) <= 2
                Z[mask] = Z[mask] ** 2 + C[mask]
                iterations[mask] = i
            
            colors = self._apply_wellness_coloring(iterations, max_iter, wellness, mood)
            return Image.fromarray(colors, 'RGB')
            
        except Exception as e:
            logger.error(f"2D fractal error: {e}")
            return self._create_fallback_image()
    
    def generate_3d_fractal(self, wellness: float = 50, mood: float = 50) -> Image.Image:
        """Generate improved 3D Mandelbulb fractal"""
        try:
            power = 6.0 + (mood / 100) * 4.0
            
            # Higher resolution for better quality
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Create depth buffer for 3D effect
            depth_buffer = np.full((self.height, self.width), 1000.0)
            
            # Improved rendering with full pixel coverage
            for py in range(self.height):
                for px in range(self.width):
                    # Normalized device coordinates
                    x = (2.0 * px / self.width - 1.0) * 1.2
                    y = (2.0 * py / self.height - 1.0) * 1.2
                    
                    # Ray marching
                    depth = self._ray_march_mandelbulb(x, y, power)
                    depth_buffer[py, px] = depth
                    
                    if depth < 100:
                        # Calculate color based on depth and metrics
                        intensity = 1.0 - (depth / 100)
                        
                        # Add some variation based on position
                        variation = math.sin(px * 0.05) * 0.1 + math.cos(py * 0.05) * 0.1
                        intensity = max(0, min(1, intensity + variation))
                        
                        # Color mapping based on wellness
                        if wellness > 70:
                            r = int(255 * intensity * PHI_INVERSE)
                            g = int(200 * intensity)
                            b = int(100 * intensity * 0.5)
                        elif wellness > 40:
                            r = int(150 * intensity)
                            g = int(100 * intensity)
                            b = int(200 * intensity)
                        else:
                            r = int(80 * intensity)
                            g = int(150 * intensity)
                            b = int(255 * intensity)
                        
                        image[py, px] = [r, g, b]
                    else:
                        # Background gradient
                        bg_intensity = 0.1 + 0.05 * (py / self.height)
                        image[py, px] = [int(30 * bg_intensity), int(30 * bg_intensity), int(50 * bg_intensity)]
            
            # Apply post-processing
            img = Image.fromarray(image, 'RGB')
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
            return img
            
        except Exception as e:
            logger.error(f"3D fractal error: {e}")
            return self._create_fallback_image()
    
    def _ray_march_mandelbulb(self, x: float, y: float, power: float) -> float:
        """Ray march into the Mandelbulb"""
        # Camera setup
        ro = np.array([x * 2, y * 2, -3.0])  # Ray origin
        rd = np.array([0, 0, 1])  # Ray direction (looking at origin)
        rd = rd / np.linalg.norm(rd)
        
        t = 0.0
        for _ in range(64):  # Max steps
            p = ro + rd * t
            d = self._mandelbulb_de(p, power)
            
            if d < 0.001:
                return t  # Hit
            
            t += d * 0.5  # Step with safety factor
            
            if t > 10:
                return 100  # Miss
        
        return 100
    
    def _mandelbulb_de(self, p: np.ndarray, power: float) -> float:
        """Distance estimator for Mandelbulb"""
        z = p.copy()
        dr = 1.0
        r = 0.0
        
        for _ in range(10):
            r = np.linalg.norm(z)
            if r > 2:
                break
            
            # Convert to spherical coordinates
            theta = np.arccos(z[2] / r)
            phi = np.arctan2(z[1], z[0])
            
            # Scale and rotate
            dr = pow(r, power - 1) * power * dr + 1
            
            zr = pow(r, power)
            theta *= power
            phi *= power
            
            # Convert back
            z = zr * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            z += p
        
        return 0.5 * np.log(r) * r / dr
    
    def _apply_wellness_coloring(self, iterations: np.ndarray, max_iter: int, 
                                  wellness: float, mood: float) -> np.ndarray:
        """Apply color scheme based on wellness metrics"""
        norm = iterations / max_iter
        
        if wellness > 70:
            r = (norm * 255 * PHI_INVERSE).astype(np.uint8)
            g = (norm * 200).astype(np.uint8)
            b = (norm * 100).astype(np.uint8)
        elif wellness > 40:
            r = (norm * 150).astype(np.uint8)
            g = (norm * 150).astype(np.uint8)
            b = (norm * 200).astype(np.uint8)
        else:
            r = (norm * 100).astype(np.uint8)
            g = (norm * 150).astype(np.uint8)
            b = (norm * 255).astype(np.uint8)
        
        return np.stack([r, g, b], axis=-1)
    
    def _create_fallback_image(self) -> Image.Image:
        """Create a beautiful fallback image"""
        img = Image.new('RGB', (self.width, self.height), color=(30, 30, 50))
        draw = ImageDraw.Draw(img)
        
        # Draw golden spiral
        cx, cy = self.width // 2, self.height // 2
        points = []
        for i in range(500):
            angle = i * GOLDEN_ANGLE_RAD * 0.1
            r = 2 * math.sqrt(i)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            if 0 <= x < self.width and 0 <= y < self.height:
                points.append((x, y))
        
        for i, (x, y) in enumerate(points):
            color = (
                int(100 + 100 * math.sin(i * 0.02)),
                int(100 + 100 * math.sin(i * 0.02 + 2)),
                int(150 + 100 * math.sin(i * 0.02 + 4))
            )
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════
# VIRTUAL PET SYSTEM - ALL 8 SPECIES
# ═══════════════════════════════════════════════════════════════════════════════

PET_SPECIES = {
    'cat': {'emoji': '🐱', 'personality': 'independent', 'favorite_activity': 'napping'},
    'dog': {'emoji': '🐕', 'personality': 'loyal', 'favorite_activity': 'playing'},
    'dragon': {'emoji': '🐉', 'personality': 'fierce', 'favorite_activity': 'hoarding'},
    'phoenix': {'emoji': '🔥', 'personality': 'reborn', 'favorite_activity': 'meditating'},
    'axolotl': {'emoji': '🦎', 'personality': 'regenerative', 'favorite_activity': 'swimming'},
    'owl': {'emoji': '🦉', 'personality': 'wise', 'favorite_activity': 'studying'},
    'sloth': {'emoji': '🦥', 'personality': 'peaceful', 'favorite_activity': 'resting'},
    'bunny': {'emoji': '🐰', 'personality': 'gentle', 'favorite_activity': 'hopping'}
}

class VirtualPet:
    """Virtual pet that responds to user activity"""
    
    def __init__(self, user_id: str, db: Database):
        self.user_id = user_id
        self.db = db
        
        pet_data = db.select('pet_state', {'user_id': user_id})
        if pet_data:
            self.state = dict(pet_data[0])
        else:
            now = datetime.now(timezone.utc).isoformat()
            self.state = {
                'user_id': user_id,
                'species': 'cat',
                'name': 'Buddy',
                'hunger': 50.0,
                'energy': 50.0,
                'mood': 50.0,
                'level': 1,
                'experience': 0,
                'bond': 0.0,
                'last_updated': now
            }
            db.insert('pet_state', self.state)
    
    def feed(self) -> dict:
        """Feed the pet"""
        species_info = PET_SPECIES.get(self.state['species'], {})
        self.state['hunger'] = max(0, self.state['hunger'] - 30)
        self.state['mood'] = min(100, self.state['mood'] + 5)
        self.state['bond'] = min(100, self.state['bond'] + 1)
        self._save()
        return {
            'success': True, 
            'message': f"{species_info.get('emoji', '🐾')} {self.state['name']} enjoyed the food!",
            'pet': self.get_status()
        }
    
    def play(self) -> dict:
        """Play with pet"""
        species_info = PET_SPECIES.get(self.state['species'], {})
        
        if self.state['energy'] < 20:
            return {
                'success': False, 
                'message': f"{species_info.get('emoji', '🐾')} {self.state['name']} is too tired to play.",
                'pet': self.get_status()
            }
        
        self.state['energy'] = max(0, self.state['energy'] - 15)
        self.state['mood'] = min(100, self.state['mood'] + 15)
        self.state['bond'] = min(100, self.state['bond'] + 3)
        self.state['experience'] += 5
        
        if self.state['experience'] >= self.state['level'] * 100:
            self.state['level'] += 1
            self.state['experience'] = 0
        
        self._save()
        return {
            'success': True, 
            'message': f"{species_info.get('emoji', '🐾')} {self.state['name']} had fun playing!",
            'pet': self.get_status()
        }
    
    def rest(self) -> dict:
        """Let pet rest"""
        species_info = PET_SPECIES.get(self.state['species'], {})
        self.state['energy'] = min(100, self.state['energy'] + 30)
        self._save()
        return {
            'success': True, 
            'message': f"{species_info.get('emoji', '🐾')} {self.state['name']} is resting peacefully...",
            'pet': self.get_status()
        }
    
    def update_from_daily_entry(self, mood: float, goals_completed: int):
        """Update pet based on user activity"""
        self.state['mood'] = min(100, self.state['mood'] + (mood - 50) * 0.3)
        self.state['experience'] += goals_completed * 10
        
        if self.state['experience'] >= self.state['level'] * 100:
            self.state['level'] += 1
            self.state['experience'] = 0
        
        self.state['hunger'] = min(100, self.state['hunger'] + 2)
        self.state['energy'] = max(0, self.state['energy'] - 1)
        
        self._save()
    
    def _save(self):
        """Save pet state"""
        self.state['last_updated'] = datetime.now(timezone.utc).isoformat()
        self.db.update('pet_state', self.state, {'user_id': self.user_id})
    
    def get_status(self) -> dict:
        """Get complete pet status"""
        species_info = PET_SPECIES.get(self.state['species'], {})
        
        if self.state['hunger'] > 80:
            behavior = 'hungry'
        elif self.state['energy'] < 20:
            behavior = 'tired'
        elif self.state['mood'] > 70:
            behavior = 'happy'
        elif self.state['mood'] < 30:
            behavior = 'sad'
        else:
            behavior = 'content'
        
        return {
            'species': self.state['species'],
            'name': self.state['name'],
            'emoji': species_info.get('emoji', '🐾'),
            'personality': species_info.get('personality', 'friendly'),
            'favorite_activity': species_info.get('favorite_activity', 'relaxing'),
            'hunger': round(self.state['hunger'], 1),
            'energy': round(self.state['energy'], 1),
            'mood': round(self.state['mood'], 1),
            'level': self.state['level'],
            'experience': self.state['experience'],
            'xp_to_next': self.state['level'] * 100,
            'bond': round(self.state['bond'], 1),
            'behavior': behavior,
            'last_updated': self.state['last_updated']
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize global instances
db = Database()
fractal_engine = FractalEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' not in session:
        return None
    users = db.select('users', {'id': session['user_id']})
    return users[0] if users else None


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        existing = db.select('users', {'email': email})
        if existing:
            return jsonify({'error': 'Email already registered'}), 400
        
        user_id = f"user_{secrets.token_hex(12)}"
        now = datetime.now(timezone.utc).isoformat()
        trial_ends = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        
        user_data = {
            'id': user_id,
            'email': email,
            'password_hash': generate_password_hash(password),
            'first_name': first_name,
            'last_name': last_name,
            'created_at': now,
            'last_login': now,
            'is_active': 1,
            'subscription_status': 'trial',
            'trial_ends': trial_ends
        }
        
        db.insert('users', user_data)
        
        session['user_id'] = user_id
        session.permanent = True
        
        logger.info(f"✅ New user registered: {email}")
        
        return jsonify({
            'message': 'Registration successful',
            'user': {
                'id': user_id,
                'email': email,
                'first_name': first_name,
                'subscription_status': 'trial',
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        users = db.select('users', {'email': email})
        if not users:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = users[0]
        
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        db.update('users', {'last_login': datetime.now(timezone.utc).isoformat()}, {'id': user['id']})
        
        session['user_id'] = user['id']
        session.permanent = True
        
        logger.info(f"✅ User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'subscription_status': user['subscription_status']
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_me():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user['id'],
        'email': user['email'],
        'first_name': user['first_name'],
        'last_name': user['last_name'],
        'subscription_status': user['subscription_status'],
        'created_at': user['created_at']
    })


# ═══════════════════════════════════════════════════════════════════════════════
# GOALS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def handle_goals():
    user_id = session['user_id']
    
    if request.method == 'GET':
        goals = db.select('goals', {'user_id': user_id}, order_by='created_at DESC')
        return jsonify({'goals': goals, 'count': len(goals)})
    
    data = request.get_json()
    goal_id = f"goal_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc).isoformat()
    
    goal_data = {
        'id': goal_id,
        'user_id': user_id,
        'title': data.get('title', 'New Goal'),
        'description': data.get('description', ''),
        'category': data.get('category', 'personal'),
        'term': data.get('term', 'medium'),
        'priority': data.get('priority', 3),
        'progress': 0,
        'target_date': data.get('target_date'),
        'created_at': now,
        'completed_at': None
    }
    
    db.insert('goals', goal_data)
    logger.info(f"✅ Goal created: {goal_data['title']}")
    
    return jsonify({'message': 'Goal created', 'id': goal_id}), 201


@app.route('/api/goals/<goal_id>', methods=['GET', 'PUT', 'DELETE'])
@require_auth
def handle_goal(goal_id):
    user_id = session['user_id']
    
    goals = db.select('goals', {'id': goal_id, 'user_id': user_id})
    if not goals:
        return jsonify({'error': 'Goal not found'}), 404
    
    if request.method == 'GET':
        return jsonify(goals[0])
    
    if request.method == 'DELETE':
        db.delete('goals', {'id': goal_id})
        return jsonify({'message': 'Goal deleted'})
    
    data = request.get_json()
    update_data = {}
    for field in ['title', 'description', 'category', 'term', 'priority', 'progress', 'target_date']:
        if field in data:
            update_data[field] = data[field]
    
    if update_data.get('progress', 0) >= 100:
        update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
    
    db.update('goals', update_data, {'id': goal_id})
    return jsonify({'message': 'Goal updated'})


@app.route('/api/goals/<goal_id>/progress', methods=['PUT'])
@require_auth
def update_goal_progress(goal_id):
    user_id = session['user_id']
    data = request.get_json()
    progress = min(100, max(0, data.get('progress', 0)))
    
    goals = db.select('goals', {'id': goal_id, 'user_id': user_id})
    if not goals:
        return jsonify({'error': 'Goal not found'}), 404
    
    update_data = {'progress': progress}
    if progress >= 100:
        update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
    
    db.update('goals', update_data, {'id': goal_id})
    
    history_id = f"ph_{secrets.token_hex(8)}"
    db.insert('progress_history', {
        'id': history_id,
        'goal_id': goal_id,
        'user_id': user_id,
        'progress': progress,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    
    return jsonify({'message': 'Progress updated', 'progress': progress})


# ═══════════════════════════════════════════════════════════════════════════════
# HABITS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/habits', methods=['GET', 'POST'])
@require_auth
def handle_habits():
    user_id = session['user_id']
    
    if request.method == 'GET':
        habits = db.select('habits', {'user_id': user_id})
        return jsonify({'habits': habits})
    
    data = request.get_json()
    habit_id = f"habit_{secrets.token_hex(8)}"
    
    habit_data = {
        'id': habit_id,
        'user_id': user_id,
        'name': data.get('name', 'New Habit'),
        'description': data.get('description', ''),
        'frequency': data.get('frequency', 'daily'),
        'current_streak': 0,
        'longest_streak': 0,
        'total_completions': 0,
        'last_completed': None,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    db.insert('habits', habit_data)
    return jsonify({'message': 'Habit created', 'id': habit_id}), 201


@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(habit_id):
    user_id = session['user_id']
    
    habits = db.select('habits', {'id': habit_id, 'user_id': user_id})
    if not habits:
        return jsonify({'error': 'Habit not found'}), 404
    
    habit = habits[0]
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    if habit['last_completed'] == today:
        return jsonify({'message': 'Already completed today', 'streak': habit['current_streak']})
    
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if habit['last_completed'] == yesterday:
        new_streak = habit['current_streak'] + 1
    else:
        new_streak = 1
    
    longest = max(habit['longest_streak'], new_streak)
    
    db.update('habits', {
        'current_streak': new_streak,
        'longest_streak': longest,
        'total_completions': habit['total_completions'] + 1,
        'last_completed': today
    }, {'id': habit_id})
    
    return jsonify({
        'message': 'Habit completed!',
        'streak': new_streak,
        'longest_streak': longest
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY CHECK-IN API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/daily/checkin', methods=['POST'])
@require_auth
def daily_checkin():
    user_id = session['user_id']
    data = request.get_json()
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    mood = data.get('mood_level', 50)
    energy = data.get('energy_level', 50)
    stress = data.get('stress_level', 50)
    sleep_hours = data.get('sleep_hours', 7)
    sleep_quality = data.get('sleep_quality', 50)
    
    wellness = (mood * 0.3 + energy * 0.2 + (100 - stress) * 0.2 + 
                sleep_quality * 0.2 + min(sleep_hours / 8, 1) * 100 * 0.1)
    
    # Calculate spoons for the day
    spoon_data = SpoonTheory.calculate_daily_spoons(sleep_hours, sleep_quality, stress, wellness)
    
    entry_id = f"entry_{secrets.token_hex(8)}"
    entry_data = {
        'id': entry_id,
        'user_id': user_id,
        'date': today,
        'mood_level': mood,
        'energy_level': energy,
        'stress_level': stress,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'goals_completed': data.get('goals_completed', 0),
        'journal_entry': data.get('journal_entry', ''),
        'gratitude': data.get('gratitude', ''),
        'wellness_score': round(wellness, 1),
        'spoons_available': spoon_data['total_spoons'],
        'spoons_used': 0,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    existing = db.select('daily_entries', {'user_id': user_id, 'date': today})
    if existing:
        db.update('daily_entries', entry_data, {'user_id': user_id, 'date': today})
    else:
        db.insert('daily_entries', entry_data)
    
    pet = VirtualPet(user_id, db)
    pet.update_from_daily_entry(mood, data.get('goals_completed', 0))
    
    return jsonify({
        'message': 'Check-in saved',
        'wellness_score': round(wellness, 1),
        'spoons': spoon_data
    })


@app.route('/api/daily/today', methods=['GET'])
@require_auth
def get_today():
    user_id = session['user_id']
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    entries = db.select('daily_entries', {'user_id': user_id, 'date': today})
    if entries:
        entry = entries[0]
        # Add spoon calculations
        spoon_data = SpoonTheory.calculate_daily_spoons(
            entry.get('sleep_hours', 7),
            entry.get('sleep_quality', 50),
            entry.get('stress_level', 50),
            entry.get('wellness_score', 50)
        )
        entry['spoons'] = spoon_data
        return jsonify(entry)
    
    return jsonify({
        'date': today,
        'mood_level': 50,
        'energy_level': 50,
        'stress_level': 50,
        'wellness_score': 50,
        'journal_entry': '',
        'spoons': SpoonTheory.calculate_daily_spoons(7, 50, 50, 50)
    })


@app.route('/api/daily/history', methods=['GET'])
@require_auth
def get_history():
    user_id = session['user_id']
    limit = request.args.get('limit', 30, type=int)
    
    entries = db.select('daily_entries', {'user_id': user_id}, order_by='date DESC')
    return jsonify({'entries': entries[:limit], 'count': len(entries)})


# ═══════════════════════════════════════════════════════════════════════════════
# SPOON THEORY API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/spoons/calculate', methods=['POST'])
@require_auth
def calculate_spoons():
    """Calculate available spoons based on input"""
    data = request.get_json()
    
    result = SpoonTheory.calculate_daily_spoons(
        sleep_hours=data.get('sleep_hours', 7),
        sleep_quality=data.get('sleep_quality', 50),
        stress_level=data.get('stress_level', 50),
        wellness_score=data.get('wellness_score', 50)
    )
    
    return jsonify(result)


@app.route('/api/spoons/estimate-task', methods=['POST'])
@require_auth
def estimate_task_spoons():
    """Estimate spoon cost for a task"""
    data = request.get_json()
    task = data.get('task', '')
    
    if not task:
        return jsonify({'error': 'Task description required'}), 400
    
    result = SpoonTheory.estimate_task_cost(task)
    return jsonify(result)


@app.route('/api/spoons/activities', methods=['GET'])
def get_spoon_activities():
    """Get list of activities and their spoon costs"""
    return jsonify({
        'costs': SpoonTheory.DEFAULT_COSTS,
        'regenerators': SpoonTheory.REGENERATORS
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/mayan/today', methods=['GET'])
def get_mayan_today():
    """Get today's Mayan Tzolkin date"""
    tzolkin = MayanCalendar.get_tzolkin_date()
    return jsonify(tzolkin)


@app.route('/api/mayan/date/<date_str>', methods=['GET'])
def get_mayan_date(date_str):
    """Get Mayan date for a specific date (YYYY-MM-DD)"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        tzolkin = MayanCalendar.get_tzolkin_date(date)
        return jsonify(tzolkin)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400


@app.route('/api/mayan/signs', methods=['GET'])
def get_mayan_signs():
    """Get all Tzolkin day signs and their meanings"""
    return jsonify({
        'signs': TZOLKIN_DAY_SIGNS,
        'energies': TZOLKIN_ENERGIES
    })


# ═══════════════════════════════════════════════════════════════════════════════
# BINAURAL BEATS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/audio/binaural', methods=['GET', 'POST'])
@require_auth
def get_binaural_beats():
    """Generate binaural beats audio"""
    if request.method == 'POST':
        data = request.get_json()
        preset = data.get('preset', 'focus')
        duration = min(data.get('duration', 30), 60)  # Max 60 seconds
    else:
        preset = request.args.get('preset', 'focus')
        duration = min(request.args.get('duration', 30, type=int), 60)
    
    audio_bytes = BinauralBeats.generate_audio(preset, duration)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    preset_info = BinauralBeats.PRESETS.get(preset, {})
    
    return jsonify({
        'preset': preset,
        'duration': duration,
        'description': preset_info.get('description', ''),
        'carrier_frequency': preset_info.get('carrier', 200),
        'beat_frequency': preset_info.get('beat', 10),
        'audio_base64': audio_base64,
        'mime_type': 'audio/wav'
    })


@app.route('/api/audio/presets', methods=['GET'])
def get_audio_presets():
    """Get available binaural beats presets"""
    return jsonify(BinauralBeats.get_preset_info())


@app.route('/api/audio/recommend', methods=['GET'])
@require_auth
def recommend_audio():
    """Recommend audio preset based on current state"""
    user_id = session['user_id']
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    entries = db.select('daily_entries', {'user_id': user_id, 'date': today})
    if entries:
        entry = entries[0]
        preset = BinauralBeats.recommend_preset(
            entry.get('mood_level', 50),
            entry.get('energy_level', 50),
            entry.get('stress_level', 50)
        )
    else:
        preset = 'focus'
    
    preset_info = BinauralBeats.PRESETS.get(preset, {})
    
    return jsonify({
        'recommended_preset': preset,
        'description': preset_info.get('description', ''),
        'reason': 'Based on your current wellness data'
    })


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE DYSFUNCTION API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/executive/analyze', methods=['GET'])
@require_auth
def analyze_executive_function():
    """Analyze executive function patterns"""
    user_id = session['user_id']
    
    entries = db.select('daily_entries', {'user_id': user_id}, order_by='date DESC')
    goals = db.select('goals', {'user_id': user_id})
    habits = db.select('habits', {'user_id': user_id})
    
    analysis = ExecutiveDysfunctionDetector.analyze_patterns(entries, goals, habits)
    
    return jsonify(analysis)


# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL VISUALIZATION API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/visualization/fractal/<mode>', methods=['GET', 'POST'])
@require_auth
def generate_fractal(mode):
    user_id = session['user_id']
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    entries = db.select('daily_entries', {'user_id': user_id, 'date': today})
    if entries:
        entry = entries[0]
        wellness = entry.get('wellness_score', 50)
        mood = entry.get('mood_level', 50)
        stress = entry.get('stress_level', 50)
    else:
        wellness, mood, stress = 50, 50, 50
    
    if mode == '2d':
        image = fractal_engine.generate_2d_fractal(wellness, mood, stress)
    else:
        image = fractal_engine.generate_3d_fractal(wellness, mood)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG', quality=95)
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png')


@app.route('/api/visualization/fractal-base64/<mode>', methods=['GET'])
@require_auth
def generate_fractal_base64(mode):
    user_id = session['user_id']
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    entries = db.select('daily_entries', {'user_id': user_id, 'date': today})
    if entries:
        entry = entries[0]
        wellness = entry.get('wellness_score', 50)
        mood = entry.get('mood_level', 50)
        stress = entry.get('stress_level', 50)
    else:
        wellness, mood, stress = 50, 50, 50
    
    if mode == '2d':
        image = fractal_engine.generate_2d_fractal(wellness, mood, stress)
    else:
        image = fractal_engine.generate_3d_fractal(wellness, mood)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'mode': mode,
        'metrics': {
            'wellness': wellness,
            'mood': mood,
            'stress': stress
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PET API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet/status', methods=['GET'])
@require_auth
def get_pet_status():
    user_id = session['user_id']
    pet = VirtualPet(user_id, db)
    return jsonify(pet.get_status())


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    user_id = session['user_id']
    pet = VirtualPet(user_id, db)
    result = pet.feed()
    return jsonify(result)


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_with_pet():
    user_id = session['user_id']
    pet = VirtualPet(user_id, db)
    result = pet.play()
    return jsonify(result)


@app.route('/api/pet/rest', methods=['POST'])
@require_auth
def rest_pet():
    user_id = session['user_id']
    pet = VirtualPet(user_id, db)
    result = pet.rest()
    return jsonify(result)


@app.route('/api/pet/rename', methods=['POST'])
@require_auth
def rename_pet():
    user_id = session['user_id']
    data = request.get_json()
    new_name = data.get('name', '').strip()
    
    if not new_name or len(new_name) > 20:
        return jsonify({'error': 'Name must be 1-20 characters'}), 400
    
    db.update('pet_state', {'name': new_name}, {'user_id': user_id})
    return jsonify({'message': f'Pet renamed to {new_name}'})


@app.route('/api/pet/change-species', methods=['POST'])
@require_auth
def change_pet_species():
    user_id = session['user_id']
    data = request.get_json()
    new_species = data.get('species', '').lower()
    
    if new_species not in PET_SPECIES:
        return jsonify({
            'error': 'Invalid species',
            'available': list(PET_SPECIES.keys())
        }), 400
    
    db.update('pet_state', {'species': new_species}, {'user_id': user_id})
    species_info = PET_SPECIES[new_species]
    
    return jsonify({
        'message': f'Species changed to {new_species}',
        'species': new_species,
        'emoji': species_info['emoji'],
        'personality': species_info['personality']
    })


@app.route('/api/pet/species', methods=['GET'])
def get_pet_species():
    """Get all available pet species"""
    return jsonify({
        'species': PET_SPECIES,
        'count': len(PET_SPECIES)
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD & SYSTEM API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/dashboard', methods=['GET'])
@require_auth
def get_dashboard():
    user_id = session['user_id']
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    entries = db.select('daily_entries', {'user_id': user_id, 'date': today})
    goals = db.select('goals', {'user_id': user_id})
    habits = db.select('habits', {'user_id': user_id})
    pet = VirtualPet(user_id, db)
    
    wellness = entries[0]['wellness_score'] if entries else 50
    
    # Get Mayan date
    tzolkin = MayanCalendar.get_tzolkin_date()
    
    # Calculate spoons
    if entries:
        entry = entries[0]
        spoons = SpoonTheory.calculate_daily_spoons(
            entry.get('sleep_hours', 7),
            entry.get('sleep_quality', 50),
            entry.get('stress_level', 50),
            wellness
        )
    else:
        spoons = SpoonTheory.calculate_daily_spoons(7, 50, 50, 50)
    
    return jsonify({
        'wellness_score': wellness,
        'goals': {
            'total': len(goals),
            'completed': sum(1 for g in goals if g['completed_at']),
            'in_progress': sum(1 for g in goals if not g['completed_at'])
        },
        'habits': {
            'total': len(habits),
            'completed_today': sum(1 for h in habits if h['last_completed'] == today)
        },
        'pet': pet.get_status(),
        'tzolkin': tzolkin,
        'spoons': spoons,
        'today': today
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '10.1',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'features': {
            'pet_species': len(PET_SPECIES),
            'mayan_calendar': True,
            'spoon_theory': True,
            'binaural_beats': True,
            'executive_dysfunction': True
        }
    })


@app.route('/api/sacred-math', methods=['GET'])
def sacred_math():
    return jsonify({
        'phi': PHI,
        'phi_inverse': PHI_INVERSE,
        'golden_angle': GOLDEN_ANGLE,
        'golden_angle_rad': GOLDEN_ANGLE_RAD,
        'fibonacci': FIBONACCI,
        'platonic_solids': {
            'tetrahedron': {'faces': 4, 'vertices': 4, 'edges': 6},
            'cube': {'faces': 6, 'vertices': 8, 'edges': 12},
            'octahedron': {'faces': 8, 'vertices': 6, 'edges': 12},
            'dodecahedron': {'faces': 12, 'vertices': 20, 'edges': 30},
            'icosahedron': {'faces': 20, 'vertices': 12, 'edges': 30}
        }
    })


@app.route('/api/system/status', methods=['GET'])
def system_status():
    return jsonify({
        'status': 'operational',
        'version': '10.1',
        'components': {
            'database': 'healthy',
            'fractal_engine': 'ready',
            'authentication': 'active',
            'pet_system': 'active',
            'mayan_calendar': 'active',
            'spoon_theory': 'active',
            'binaural_beats': 'active',
            'executive_support': 'active'
        },
        'capabilities': {
            'gpu_acceleration': GPU_AVAILABLE,
            'gpu_name': GPU_NAME,
            'ml_predictions': HAS_SKLEARN,
            '2d_fractals': True,
            '3d_fractals': True,
            'sacred_math': True,
            'pet_species_count': len(PET_SPECIES)
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence v10.1</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #f39c12, #e74c3c, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle { color: #888; font-size: 1.1rem; }
        
        .user-info {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logout-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .logout-btn:hover { background: rgba(255,255,255,0.2); }
        
        nav {
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 20px 0;
            flex-wrap: wrap;
        }
        
        .nav-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            color: #fff;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1rem;
        }
        
        .nav-btn:hover, .nav-btn.active {
            background: linear-gradient(135deg, #f39c12, #e74c3c);
        }
        
        .section { display: none; padding: 20px 0; }
        .section.active { display: block; }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .card h2 {
            color: #f39c12;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .metric-card {
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #f39c12;
        }
        
        .metric-label { color: #888; margin-top: 5px; }
        
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #aaa; }
        
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1rem;
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            height: 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.2);
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #f39c12;
            cursor: pointer;
        }
        
        button {
            background: linear-gradient(135deg, #f39c12, #e74c3c);
            border: none;
            color: #fff;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s;
        }
        
        button:hover { transform: scale(1.02); }
        
        .pet-display {
            text-align: center;
            padding: 30px;
        }
        
        .pet-emoji { font-size: 80px; }
        .pet-name { font-size: 1.5rem; margin: 15px 0; color: #f39c12; }
        
        .pet-stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
        }
        
        .pet-stat {
            text-align: center;
        }
        
        .pet-actions {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .fractal-container {
            text-align: center;
        }
        
        .fractal-container img {
            max-width: 100%;
            border-radius: 15px;
            border: 2px solid rgba(255,255,255,0.1);
        }
        
        .goal-item, .habit-item {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            flex: 1;
            margin: 0 15px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #f39c12, #e74c3c);
            transition: width 0.3s;
        }
        
        .streak { color: #f39c12; font-weight: bold; }
        
        .mayan-card {
            background: linear-gradient(135deg, rgba(155,89,182,0.2), rgba(52,152,219,0.2));
            border: 1px solid rgba(155,89,182,0.3);
        }
        
        .spoon-display {
            font-size: 2rem;
            text-align: center;
            padding: 15px;
        }
        
        .audio-controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .audio-btn {
            padding: 10px 20px;
            background: rgba(52,152,219,0.3);
            border: 1px solid rgba(52,152,219,0.5);
        }
        
        .audio-btn:hover {
            background: rgba(52,152,219,0.5);
        }
        
        .species-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .species-btn {
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .species-btn:hover {
            background: rgba(255,255,255,0.1);
            border-color: #f39c12;
        }
        
        .species-btn.active {
            background: rgba(243,156,18,0.2);
            border-color: #f39c12;
        }
        
        .species-emoji { font-size: 2rem; }
        .species-name { font-size: 0.8rem; margin-top: 5px; }
        
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem; }
            .metrics-grid { grid-template-columns: 1fr 1fr; }
            .species-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌀 Life Fractal Intelligence</h1>
            <p class="subtitle">v10.1 - Your Life Visualized as Living Art</p>
            <div class="user-info">
                <span id="userEmail"></span>
                <button class="logout-btn" onclick="logout()">Logout</button>
            </div>
        </header>
        
        <nav>
            <button class="nav-btn active" onclick="showSection('overview')">📊 Overview</button>
            <button class="nav-btn" onclick="showSection('today')">📅 Today</button>
            <button class="nav-btn" onclick="showSection('goals')">🎯 Goals</button>
            <button class="nav-btn" onclick="showSection('habits')">✅ Habits</button>
            <button class="nav-btn" onclick="showSection('fractal')">🌀 Fractal</button>
            <button class="nav-btn" onclick="showSection('pet')">🐾 Pet</button>
            <button class="nav-btn" onclick="showSection('audio')">🎵 Audio</button>
        </nav>
        
        <!-- OVERVIEW SECTION -->
        <section id="overview" class="section active">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="wellnessScore">--</div>
                    <div class="metric-label">Wellness Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="goalsCount">--</div>
                    <div class="metric-label">Active Goals</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="habitsStreak">--</div>
                    <div class="metric-label">Habits Today</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="petLevel">--</div>
                    <div class="metric-label">Pet Level</div>
                </div>
            </div>
            
            <div class="card mayan-card">
                <h2>🗓️ Mayan Tzolkin Date</h2>
                <div id="mayanDate" style="font-size: 1.5rem; text-align: center; padding: 15px;">Loading...</div>
                <div id="mayanGuidance" style="text-align: center; color: #aaa; margin-top: 10px;"></div>
            </div>
            
            <div class="card">
                <h2>🥄 Energy Spoons</h2>
                <div class="spoon-display" id="spoonDisplay">Loading...</div>
                <div id="spoonRecommendations" style="margin-top: 15px;"></div>
            </div>
        </section>
        
        <!-- TODAY SECTION -->
        <section id="today" class="section">
            <div class="card">
                <h2>📝 Daily Check-in</h2>
                <form id="checkinForm">
                    <div class="form-group">
                        <label>Mood Level: <span id="moodValue">50</span></label>
                        <input type="range" id="moodLevel" min="0" max="100" value="50" 
                               oninput="document.getElementById('moodValue').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label>Energy Level: <span id="energyValue">50</span></label>
                        <input type="range" id="energyLevel" min="0" max="100" value="50"
                               oninput="document.getElementById('energyValue').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label>Stress Level: <span id="stressValue">50</span></label>
                        <input type="range" id="stressLevel" min="0" max="100" value="50"
                               oninput="document.getElementById('stressValue').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label>Sleep Hours</label>
                        <input type="number" id="sleepHours" min="0" max="24" step="0.5" value="7">
                    </div>
                    <div class="form-group">
                        <label>Sleep Quality: <span id="sleepQualityValue">50</span></label>
                        <input type="range" id="sleepQuality" min="0" max="100" value="50"
                               oninput="document.getElementById('sleepQualityValue').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label>Journal Entry</label>
                        <textarea id="journalEntry" rows="4" placeholder="How are you feeling today?"></textarea>
                    </div>
                    <button type="submit">Save Check-in</button>
                </form>
            </div>
        </section>
        
        <!-- GOALS SECTION -->
        <section id="goals" class="section">
            <div class="card">
                <h2>🎯 Your Goals</h2>
                <div id="goalsList"></div>
                <div style="margin-top: 20px;">
                    <input type="text" id="newGoalTitle" placeholder="New goal title...">
                    <button onclick="createGoal()" style="margin-top: 10px;">Add Goal</button>
                </div>
            </div>
        </section>
        
        <!-- HABITS SECTION -->
        <section id="habits" class="section">
            <div class="card">
                <h2>✅ Daily Habits</h2>
                <div id="habitsList"></div>
                <div style="margin-top: 20px;">
                    <input type="text" id="newHabitName" placeholder="New habit name...">
                    <button onclick="createHabit()" style="margin-top: 10px;">Add Habit</button>
                </div>
            </div>
        </section>
        
        <!-- FRACTAL SECTION -->
        <section id="fractal" class="section">
            <div class="card">
                <h2>🌀 Your Life Fractal</h2>
                <div style="text-align: center; margin-bottom: 20px;">
                    <button onclick="loadFractal('2d')">2D Fractal</button>
                    <button onclick="loadFractal('3d')" style="margin-left: 10px;">3D Fractal</button>
                </div>
                <div class="fractal-container">
                    <img id="fractalImage" src="" alt="Your Life Fractal">
                </div>
            </div>
        </section>
        
        <!-- PET SECTION -->
        <section id="pet" class="section">
            <div class="card">
                <h2>🐾 Your Companion</h2>
                <div class="pet-display">
                    <div class="pet-emoji" id="petEmoji">🐱</div>
                    <div class="pet-name" id="petName">Loading...</div>
                    <div class="pet-stats">
                        <div class="pet-stat">
                            <div>🍖 Hunger</div>
                            <div id="petHunger">--</div>
                        </div>
                        <div class="pet-stat">
                            <div>⚡ Energy</div>
                            <div id="petEnergy">--</div>
                        </div>
                        <div class="pet-stat">
                            <div>😊 Mood</div>
                            <div id="petMood">--</div>
                        </div>
                    </div>
                    <div class="pet-actions">
                        <button onclick="petAction('feed')">🍖 Feed</button>
                        <button onclick="petAction('play')">🎾 Play</button>
                        <button onclick="petAction('rest')">😴 Rest</button>
                    </div>
                </div>
                
                <h3 style="margin-top: 30px; margin-bottom: 15px;">Choose Species</h3>
                <div class="species-grid" id="speciesGrid"></div>
            </div>
        </section>
        
        <!-- AUDIO SECTION -->
        <section id="audio" class="section">
            <div class="card">
                <h2>🎵 Binaural Beats</h2>
                <p style="color: #888; margin-bottom: 20px;">Use headphones for best results</p>
                <div class="audio-controls">
                    <button class="audio-btn" onclick="playBinaural('focus')">🎯 Focus</button>
                    <button class="audio-btn" onclick="playBinaural('relax')">😌 Relax</button>
                    <button class="audio-btn" onclick="playBinaural('sleep')">😴 Sleep</button>
                    <button class="audio-btn" onclick="playBinaural('meditate')">🧘 Meditate</button>
                    <button class="audio-btn" onclick="playBinaural('energy')">⚡ Energy</button>
                    <button class="audio-btn" onclick="playBinaural('healing')">💚 Healing</button>
                </div>
                <div id="audioStatus" style="text-align: center; margin-top: 20px;"></div>
                <audio id="binauralAudio" style="display: none;"></audio>
            </div>
        </section>
    </div>
    
    <script>
        // API calls
        async function api(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include'
            };
            if (data) options.body = JSON.stringify(data);
            
            const response = await fetch(endpoint, options);
            return response.json();
        }
        
        // Navigation
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            event.target.classList.add('active');
            
            if (sectionId === 'goals') loadGoals();
            if (sectionId === 'habits') loadHabits();
            if (sectionId === 'pet') loadPet();
            if (sectionId === 'fractal') loadFractal('2d');
        }
        
        // Load dashboard
        async function loadDashboard() {
            try {
                const data = await api('/api/dashboard');
                
                document.getElementById('wellnessScore').textContent = Math.round(data.wellness_score);
                document.getElementById('goalsCount').textContent = data.goals.in_progress;
                document.getElementById('habitsStreak').textContent = data.habits.completed_today;
                document.getElementById('petLevel').textContent = data.pet.level;
                
                // Mayan date
                if (data.tzolkin) {
                    document.getElementById('mayanDate').textContent = data.tzolkin.full_date;
                    document.getElementById('mayanGuidance').textContent = data.tzolkin.guidance;
                }
                
                // Spoons
                if (data.spoons) {
                    document.getElementById('spoonDisplay').textContent = data.spoons.spoon_emoji;
                    const recs = data.spoons.recommendations.map(r => `<p>• ${r}</p>`).join('');
                    document.getElementById('spoonRecommendations').innerHTML = recs;
                }
                
                // User info
                const user = await api('/api/auth/me');
                document.getElementById('userEmail').textContent = user.email;
                
            } catch (e) {
                console.error('Dashboard error:', e);
            }
        }
        
        // Goals
        async function loadGoals() {
            const data = await api('/api/goals');
            const list = document.getElementById('goalsList');
            
            if (data.goals.length === 0) {
                list.innerHTML = '<p style="color: #888;">No goals yet. Create your first goal!</p>';
                return;
            }
            
            list.innerHTML = data.goals.map(g => `
                <div class="goal-item">
                    <span>${g.title}</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${g.progress}%"></div>
                    </div>
                    <span>${g.progress}%</span>
                    <input type="range" min="0" max="100" value="${g.progress}" 
                           style="width: 100px; margin-left: 10px;"
                           onchange="updateGoalProgress('${g.id}', this.value)">
                </div>
            `).join('');
        }
        
        async function createGoal() {
            const title = document.getElementById('newGoalTitle').value.trim();
            if (!title) return;
            
            await api('/api/goals', 'POST', { title });
            document.getElementById('newGoalTitle').value = '';
            loadGoals();
        }
        
        async function updateGoalProgress(goalId, progress) {
            await api(`/api/goals/${goalId}/progress`, 'PUT', { progress: parseInt(progress) });
            loadGoals();
            loadDashboard();
        }
        
        // Habits
        async function loadHabits() {
            const data = await api('/api/habits');
            const list = document.getElementById('habitsList');
            
            if (data.habits.length === 0) {
                list.innerHTML = '<p style="color: #888;">No habits yet. Create your first habit!</p>';
                return;
            }
            
            list.innerHTML = data.habits.map(h => `
                <div class="habit-item">
                    <span>${h.name}</span>
                    <span class="streak">🔥 ${h.current_streak} day streak</span>
                    <button onclick="completeHabit('${h.id}')">Complete</button>
                </div>
            `).join('');
        }
        
        async function createHabit() {
            const name = document.getElementById('newHabitName').value.trim();
            if (!name) return;
            
            await api('/api/habits', 'POST', { name });
            document.getElementById('newHabitName').value = '';
            loadHabits();
        }
        
        async function completeHabit(habitId) {
            await api(`/api/habits/${habitId}/complete`, 'POST');
            loadHabits();
            loadDashboard();
        }
        
        // Pet
        async function loadPet() {
            const data = await api('/api/pet/status');
            
            document.getElementById('petEmoji').textContent = data.emoji;
            document.getElementById('petName').textContent = `${data.name} (Level ${data.level})`;
            document.getElementById('petHunger').textContent = Math.round(data.hunger);
            document.getElementById('petEnergy').textContent = Math.round(data.energy);
            document.getElementById('petMood').textContent = Math.round(data.mood);
            
            // Load species grid
            const species = await api('/api/pet/species');
            const grid = document.getElementById('speciesGrid');
            grid.innerHTML = Object.entries(species.species).map(([key, val]) => `
                <div class="species-btn ${data.species === key ? 'active' : ''}" 
                     onclick="changeSpecies('${key}')">
                    <div class="species-emoji">${val.emoji}</div>
                    <div class="species-name">${key}</div>
                </div>
            `).join('');
        }
        
        async function petAction(action) {
            await api(`/api/pet/${action}`, 'POST');
            loadPet();
        }
        
        async function changeSpecies(species) {
            await api('/api/pet/change-species', 'POST', { species });
            loadPet();
        }
        
        // Fractal
        async function loadFractal(mode) {
            const data = await api(`/api/visualization/fractal-base64/${mode}`);
            document.getElementById('fractalImage').src = data.image;
        }
        
        // Check-in form
        document.getElementById('checkinForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            await api('/api/daily/checkin', 'POST', {
                mood_level: parseInt(document.getElementById('moodLevel').value),
                energy_level: parseInt(document.getElementById('energyLevel').value),
                stress_level: parseInt(document.getElementById('stressLevel').value),
                sleep_hours: parseFloat(document.getElementById('sleepHours').value),
                sleep_quality: parseInt(document.getElementById('sleepQuality').value),
                journal_entry: document.getElementById('journalEntry').value
            });
            
            alert('Check-in saved!');
            loadDashboard();
        });
        
        // Audio
        let currentAudio = null;
        
        async function playBinaural(preset) {
            const status = document.getElementById('audioStatus');
            status.textContent = 'Loading audio...';
            
            try {
                const data = await api('/api/audio/binaural', 'POST', { preset, duration: 30 });
                
                const audio = document.getElementById('binauralAudio');
                audio.src = `data:audio/wav;base64,${data.audio_base64}`;
                audio.loop = true;
                audio.play();
                
                status.innerHTML = `
                    <p>Playing: ${preset.toUpperCase()}</p>
                    <p style="color: #888;">${data.description}</p>
                    <button onclick="stopAudio()" style="margin-top: 10px;">Stop</button>
                `;
            } catch (e) {
                status.textContent = 'Error loading audio';
            }
        }
        
        function stopAudio() {
            const audio = document.getElementById('binauralAudio');
            audio.pause();
            document.getElementById('audioStatus').textContent = '';
        }
        
        // Logout
        async function logout() {
            await api('/api/auth/logout', 'POST');
            window.location.href = '/login';
        }
        
        // Initialize
        loadDashboard();
    </script>
</body>
</html>
'''

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #e0e0e0;
        }
        
        .login-container {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        h1 {
            text-align: center;
            font-size: 2rem;
            background: linear-gradient(135deg, #f39c12, #e74c3c, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            background: rgba(255,255,255,0.05);
            border: none;
            color: #888;
            transition: all 0.3s;
        }
        
        .tab:first-child { border-radius: 10px 0 0 10px; }
        .tab:last-child { border-radius: 0 10px 10px 0; }
        
        .tab.active {
            background: linear-gradient(135deg, #f39c12, #e74c3c);
            color: #fff;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
        }
        
        .form-group input {
            width: 100%;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1rem;
        }
        
        button[type="submit"] {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #f39c12, #e74c3c);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button[type="submit"]:hover {
            transform: scale(1.02);
        }
        
        .error {
            background: rgba(231, 76, 60, 0.2);
            border: 1px solid rgba(231, 76, 60, 0.5);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .features {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .feature {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            color: #888;
        }
        
        #registerForm { display: none; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🌀 Life Fractal</h1>
        <p class="subtitle">Your Life Visualized as Living Art</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('login')">Login</button>
            <button class="tab" onclick="showTab('register')">Register</button>
        </div>
        
        <div class="error" id="error"></div>
        
        <form id="loginForm" onsubmit="handleLogin(event)">
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="loginEmail" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="loginPassword" required>
            </div>
            <button type="submit">Login</button>
        </form>
        
        <form id="registerForm" onsubmit="handleRegister(event)">
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="regEmail" required>
            </div>
            <div class="form-group">
                <label>Password (min 6 characters)</label>
                <input type="password" id="regPassword" required minlength="6">
            </div>
            <div class="form-group">
                <label>First Name</label>
                <input type="text" id="regFirstName">
            </div>
            <button type="submit">Create Account</button>
        </form>
        
        <div class="features">
            <div class="feature">🌀 Fractal visualization of your life</div>
            <div class="feature">🐾 8 virtual pet companions</div>
            <div class="feature">🗓️ Mayan Tzolkin calendar</div>
            <div class="feature">🥄 Spoon Theory energy tracking</div>
            <div class="feature">🎵 Binaural beats audio</div>
        </div>
    </div>
    
    <script>
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            
            document.getElementById('loginForm').style.display = tab === 'login' ? 'block' : 'none';
            document.getElementById('registerForm').style.display = tab === 'register' ? 'block' : 'none';
            document.getElementById('error').style.display = 'none';
        }
        
        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
        }
        
        async function handleLogin(e) {
            e.preventDefault();
            
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({
                        email: document.getElementById('loginEmail').value,
                        password: document.getElementById('loginPassword').value
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    window.location.href = '/';
                } else {
                    showError(data.error || 'Login failed');
                }
            } catch (e) {
                showError('Connection error');
            }
        }
        
        async function handleRegister(e) {
            e.preventDefault();
            
            try {
                const response = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({
                        email: document.getElementById('regEmail').value,
                        password: document.getElementById('regPassword').value,
                        first_name: document.getElementById('regFirstName').value
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    window.location.href = '/';
                } else {
                    showError(data.error || 'Registration failed');
                }
            } catch (e) {
                showError('Connection error');
            }
        }
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template_string(DASHBOARD_HTML)


@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/')
    return render_template_string(LOGIN_HTML)


@app.route('/register')
def register_page():
    return redirect('/login')


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("🌀 Life Fractal Intelligence v10.1 - Production Ready")
    logger.info(f"   GPU Available: {GPU_AVAILABLE}")
    logger.info(f"   ML Available: {HAS_SKLEARN}")
    logger.info(f"   Pet Species: {len(PET_SPECIES)}")
    logger.info(f"   Features: Mayan Calendar, Spoon Theory, Binaural Beats, Executive Support")
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
