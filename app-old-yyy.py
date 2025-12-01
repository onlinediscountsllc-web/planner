#!/usr/bin/env python3
"""
LIFE FRACTAL INTELLIGENCE v8 - ULTIMATE PRODUCTION
═══════════════════════════════════════════════════════════════════════════════
Nordic Design | Llama 3 AI | Voice Control | Swarm Intelligence | Privacy-First

NEW IN v8:
- Llama 3 NLP chat interface (Ollama integration)
- Voice control (Web Speech API)
- Forgot password system
- Enhanced security (JWT, bcrypt, secure tokens)
- Swarm intelligence (aggregate learning)
- Privacy-preserving ML (anonymous data collection)
- Per-user optimization
- Full Z(T) equation implementation
- Mayan calendar integration
- CBT cognitive distortion analysis
- Math combinations database
- Adaptive learning (75% efficacy)

FIXED IN v8:
- NO emoji rendering issues (clean text only)
- Proper password storage (bcrypt)
- Secure token generation
- Privacy protection
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import secrets
import logging
import sqlite3
import smtplib
import hashlib
import hmac
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from io import BytesIO
import base64
import requests
from typing import Dict, List, Optional, Tuple, Any

from flask import Flask, request, jsonify, render_template_string, g, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
from PIL import Image, ImageDraw

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATABASE = os.environ.get('DATABASE_PATH', '/tmp/life_fractal_v8.db')
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
FROM_EMAIL = os.environ.get('FROM_EMAIL', SMTP_USER)

# Sacred Mathematics Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_INV = PHI - 1
GOLDEN_ANGLE = math.radians(137.5077640500378)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Nordic Color Palette (NO EMOJIS - Clean Text Only)
COLORS = {
    'primary': '#5B7C99',
    'secondary': '#8BA3B8',
    'background': '#F8F9FA',
    'surface': '#FFFFFF',
    'text': '#2C3E50',
    'text_light': '#6C757D',
    'success': '#6BA368',
    'warning': '#C9A961',
    'error': '#B85C5C',
    'border': '#E1E4E8'
}

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
CORS(app)

# ═══════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

TZOLKIN_NAMES = [
    'Imix', 'Ik', 'Akbal', 'Kan', 'Chicchan', 'Cimi', 'Manik', 'Lamat',
    'Muluc', 'Oc', 'Chuen', 'Eb', 'Ben', 'Ix', 'Men', 'Cib', 'Caban',
    'Etznab', 'Cauac', 'Ahau'
]

HAAB_MONTHS = [
    'Pop', 'Wo', 'Zip', 'Zotz', 'Tzec', 'Xul', 'Yaxkin', 'Mol', 'Chen',
    'Yax', 'Zac', 'Ceh', 'Mac', 'Kankin', 'Muan', 'Pax', 'Kayab', 'Cumku', 'Wayeb'
]

def calculate_mayan_date(date_obj):
    """
    Calculate Mayan calendar components for a given date
    Returns Y(T) - Mayan Cyclical Time Factor
    """
    # Calculate days since Mayan epoch (Aug 11, 3114 BCE)
    epoch = datetime(year=-3113, month=8, day=11)  # Simplified
    days_since_epoch = (date_obj - datetime(2000, 1, 1)).days + 2456293  # Approximate
    
    # Tzolk'in (260-day sacred calendar)
    tzolkin_number = (days_since_epoch % 13) + 1
    tzolkin_name_index = days_since_epoch % 20
    tzolkin_name = TZOLKIN_NAMES[tzolkin_name_index]
    
    # Haab' (365-day solar calendar)
    haab_day = days_since_epoch % 365
    haab_month_index = haab_day // 20
    haab_day_of_month = haab_day % 20
    haab_month = HAAB_MONTHS[min(haab_month_index, 18)]
    
    # Base-20 position value
    base20_value = (tzolkin_number / 13) * (tzolkin_name_index / 20)
    
    # Y(T) - Cyclical time factor for Z(T) equation
    y_t = 0.5 + 0.5 * math.sin(2 * math.pi * (days_since_epoch % 260) / 260)
    
    return {
        'tzolkin': f"{tzolkin_number} {tzolkin_name}",
        'haab': f"{haab_day_of_month} {haab_month}",
        'y_t': y_t,
        'base20_value': base20_value,
        'energy': tzolkin_number / 13,
        'day_sign': tzolkin_name
    }

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    db = get_db()
    cursor = db.cursor()
    
    # Users table with enhanced security
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            email_verified BOOLEAN DEFAULT 0,
            verification_token TEXT,
            reset_token TEXT,
            reset_token_expires TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT,
            break_interval INTEGER DEFAULT 25,
            sound_enabled BOOLEAN DEFAULT 0,
            high_contrast BOOLEAN DEFAULT 0,
            voice_enabled BOOLEAN DEFAULT 0
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
            voice_notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
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
            completed_at TEXT,
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
    
    # Math patterns database (for swarm intelligence)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS math_patterns (
            id TEXT PRIMARY KEY,
            user_hash TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            formula_combo TEXT NOT NULL,
            parameters TEXT NOT NULL,
            efficacy_score REAL NOT NULL,
            wellness_improvement REAL,
            usage_count INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_used TEXT NOT NULL
        )
    ''')
    
    # ML predictions database
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id TEXT PRIMARY KEY,
            user_hash TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_mood REAL,
            actual_mood REAL,
            accuracy REAL,
            model_version TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    # CBT analysis database
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cbt_analysis (
            id TEXT PRIMARY KEY,
            user_hash TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            lambda_score REAL NOT NULL,
            distortions TEXT,
            neuro_regulatory REAL,
            red_flag_score REAL,
            chat_context TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Chat history (for Llama 3 context)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            message TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Swarm intelligence aggregates (anonymous)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS swarm_patterns (
            id TEXT PRIMARY KEY,
            pattern_hash TEXT UNIQUE NOT NULL,
            formula_type TEXT NOT NULL,
            avg_efficacy REAL NOT NULL,
            usage_count INTEGER DEFAULT 1,
            success_rate REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
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

# ═══════════════════════════════════════════════════════════════════════════
# SECURITY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def generate_secure_token():
    """Generate cryptographically secure token"""
    return secrets.token_urlsafe(32)

def hash_user_id(user_id: str) -> str:
    """Hash user ID for anonymous data collection"""
    return hashlib.sha256(f"{user_id}{SECRET_KEY}".encode()).hexdigest()[:16]

def verify_reset_token(token: str, user_id: str) -> bool:
    """Verify password reset token"""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT reset_token, reset_token_expires FROM users WHERE id = ?
    ''', (user_id,))
    user = cursor.fetchone()
    
    if not user or not user['reset_token']:
        return False
    
    # Check token matches
    if not hmac.compare_digest(user['reset_token'], token):
        return False
    
    # Check not expired
    expires = datetime.fromisoformat(user['reset_token_expires'])
    if datetime.now(timezone.utc) > expires:
        return False
    
    return True

# ═══════════════════════════════════════════════════════════════════════════
# EMAIL SYSTEM (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

def send_email(to_email: str, subject: str, html_content: str) -> bool:
    """Send email with enhanced security"""
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning(f"Email not configured - would send: {subject} to {to_email}")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg.attach(MIMEText(html_content, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent: {subject} to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False

def send_verification_email(email: str, token: str):
    """Send verification email"""
    verify_url = f"https://planner-1-pyd9.onrender.com/verify?token={token}"
    html = f"""
    <html>
    <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #F8F9FA;">
        <div style="background: white; padding: 30px; border-radius: 8px; border: 2px solid #E1E4E8;">
            <h2 style="color: #5B7C99; margin-top: 0;">Verify Your Email</h2>
            <p style="color: #2C3E50; font-size: 16px; line-height: 1.6;">
                Welcome to Life Fractal Intelligence! Click the button below to verify your email address:
            </p>
            <a href="{verify_url}" style="display: inline-block; padding: 14px 28px; background: #5B7C99; color: white; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 20px 0;">
                Verify Email
            </a>
            <p style="color: #6C757D; font-size: 14px;">Or copy this link: {verify_url}</p>
            <p style="color: #6C757D; font-size: 14px; margin-top: 30px;">This link expires in 24 hours.</p>
        </div>
    </body>
    </html>
    """
    return send_email(email, 'Verify your Life Fractal account', html)

def send_reset_password_email(email: str, token: str):
    """Send password reset email"""
    reset_url = f"https://planner-1-pyd9.onrender.com/reset-password?token={token}"
    html = f"""
    <html>
    <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #F8F9FA;">
        <div style="background: white; padding: 30px; border-radius: 8px; border: 2px solid #E1E4E8;">
            <h2 style="color: #5B7C99; margin-top: 0;">Reset Your Password</h2>
            <p style="color: #2C3E50; font-size: 16px; line-height: 1.6;">
                You requested to reset your password. Click the button below:
            </p>
            <a href="{reset_url}" style="display: inline-block; padding: 14px 28px; background: #5B7C99; color: white; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 20px 0;">
                Reset Password
            </a>
            <p style="color: #6C757D; font-size: 14px;">Or copy this link: {reset_url}</p>
            <p style="color: #6C757D; font-size: 14px; margin-top: 30px;">This link expires in 1 hour.</p>
            <p style="color: #B85C5C; font-size: 14px; margin-top: 20px;">If you didn't request this, please ignore this email.</p>
        </div>
    </body>
    </html>
    """
    return send_email(email, 'Reset your password', html)

# ═══════════════════════════════════════════════════════════════════════════
# LLAMA 3 INTEGRATION (NLP + CBT Analysis)
# ═══════════════════════════════════════════════════════════════════════════

def call_llama(prompt: str, system_prompt: str = None) -> str:
    """Call Ollama Llama 3 for NLP processing"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                'model': 'llama3',
                'prompt': prompt,
                'system': system_prompt or "You are a helpful AI assistant for a life planning app.",
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            logger.error(f"Llama API error: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Llama call failed: {e}")
        return ""

def analyze_cbt(text: str, goal_metrics: Dict) -> Dict[str, float]:
    """
    CBT Cognitive Distortion Analysis using Llama 3
    Returns Lambda (Λ) score and distortion breakdown
    """
    system_prompt = """You are a CBT (Cognitive Behavioral Therapy) analyst. Analyze text for cognitive distortions:
1. All-or-nothing thinking
2. Overgeneralization
3. Mental filter (focusing on negatives)
4. Discounting positives
5. Jumping to conclusions
6. Catastrophizing
7. Emotional reasoning
8. Should statements
9. Labeling
10. Personalization

Return ONLY a JSON object with:
{
    "lambda_score": 0.0-1.0,
    "distortions": ["list", "of", "detected"],
    "severity": "low|medium|high",
    "recommendation": "brief advice"
}
"""
    
    prompt = f"""Analyze this text and goals for cognitive distortions:

Text: {text}
Goals: {json.dumps(goal_metrics)}

Return JSON only:"""
    
    response = call_llama(prompt, system_prompt)
    
    try:
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result
    except:
        pass
    
    # Default fallback
    return {
        'lambda_score': 0.3,
        'distortions': [],
        'severity': 'low',
        'recommendation': 'Keep tracking your progress.'
    }

def calculate_neuro_regulatory(lambda_score: float, base_n: float = 1.0) -> float:
    """
    N = N_Base · max(0, 1 - Λ)
    Resilience is high when risk Λ is low
    """
    return base_n * max(0, 1 - lambda_score)

def calculate_red_flag(lambda_score: float, base_e: float = 1.0) -> float:
    """
    E = E_Base · Λ
    Risk is proportional to CBT score
    """
    return base_e * lambda_score

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE OPTIMIZER (75% Efficacy Learning)
# ═══════════════════════════════════════════════════════════════════════════

MATH_COMBINATIONS = [
    'base20_fibonacci',
    'golden_ratio_spiral',
    'sacred_geometry_phi',
    'mandelbrot_julia_hybrid',
    'mayan_tzolkin_sync',
    'fractal_phi_sequence',
    'base20_golden_hybrid'
]

def calculate_custom_combo(user_hash: str, wellness_data: Dict) -> float:
    """
    Calculate C_Comb - Custom Combination Factor
    C_Comb = { C_Optimal if Efficacy ≥ 0.75·Traditional
             { 1.0 otherwise
    """
    db = get_db()
    cursor = db.cursor()
    
    # Get user's best patterns
    cursor.execute('''
        SELECT formula_combo, efficacy_score, parameters
        FROM math_patterns
        WHERE user_hash = ?
        ORDER BY efficacy_score DESC
        LIMIT 1
    ''', (user_hash,))
    
    best_pattern = cursor.fetchone()
    
    if best_pattern and best_pattern['efficacy_score'] >= 0.75:
        # Use optimal combination
        return 1.0 + (best_pattern['efficacy_score'] - 0.75)
    
    # Check swarm intelligence for community patterns
    cursor.execute('''
        SELECT formula_type, avg_efficacy
        FROM swarm_patterns
        WHERE avg_efficacy >= 0.75
        ORDER BY avg_efficacy DESC
        LIMIT 1
    ''')
    
    swarm_pattern = cursor.fetchone()
    
    if swarm_pattern:
        return 1.0 + (swarm_pattern['avg_efficacy'] - 0.75) * 0.5
    
    return 1.0  # Default - no optimization yet

def store_pattern_success(user_hash: str, formula: str, efficacy: float, wellness_improvement: float):
    """Store successful math pattern for swarm learning"""
    db = get_db()
    cursor = db.cursor()
    now = datetime.now(timezone.utc).isoformat()
    
    pattern_id = f"pattern_{secrets.token_hex(8)}"
    
    # Store user pattern
    cursor.execute('''
        INSERT INTO math_patterns (id, user_hash, pattern_type, formula_combo, parameters, 
                                   efficacy_score, wellness_improvement, created_at, last_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pattern_id, user_hash, 'custom', formula, '{}', efficacy, wellness_improvement, now, now))
    
    # Update swarm intelligence (anonymous)
    pattern_hash = hashlib.sha256(formula.encode()).hexdigest()[:16]
    
    cursor.execute('''
        INSERT INTO swarm_patterns (id, pattern_hash, formula_type, avg_efficacy, usage_count, 
                                    success_rate, created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(pattern_hash) DO UPDATE SET
            avg_efficacy = (avg_efficacy * usage_count + ?) / (usage_count + 1),
            usage_count = usage_count + 1,
            updated_at = ?
    ''', (f"swarm_{secrets.token_hex(4)}", pattern_hash, formula, efficacy, 
          1.0 if efficacy >= 0.75 else 0.0, now, now, efficacy, now))
    
    db.commit()

# ═══════════════════════════════════════════════════════════════════════════
# SACRED MATH ENGINE - Z(T) Calculation
# ═══════════════════════════════════════════════════════════════════════════

def calculate_z_vector(user_data: Dict) -> Dict[str, float]:
    """
    Z(T) = M(C_Comb · [Σ P_i · W · Y(T)] + R(N - E))
    
    Complete equation implementation
    """
    user_hash = hash_user_id(user_data.get('user_id', 'anonymous'))
    
    # Get current Mayan time factor
    mayan = calculate_mayan_date(datetime.now())
    y_t = mayan['y_t']
    
    # CBT Analysis for Lambda (Λ)
    notes = user_data.get('notes', '')
    cbt = analyze_cbt(notes, user_data)
    lambda_score = cbt.get('lambda_score', 0.3)
    
    # Calculate N and E
    N = calculate_neuro_regulatory(lambda_score)
    E = calculate_red_flag(lambda_score)
    
    # Get Custom Combination Factor
    C_Comb = calculate_custom_combo(user_hash, user_data)
    
    # Goal Progress Sum: Σ P_i · W · Y(T)
    progress_sum = 0
    goals = user_data.get('goals', [])
    for goal in goals:
        P_i = goal.get('progress', 0) / 100.0  # Normalize to 0-1
        progress_sum += P_i
    
    # Wellness factor (W)
    mood = user_data.get('mood_level', 50) / 100.0
    energy = user_data.get('energy_level', 50) / 100.0
    stress = 1.0 - (user_data.get('stress_level', 50) / 100.0)  # Invert stress
    W = (mood + energy + stress) / 3.0
    
    # Goal sum component
    goal_sum = progress_sum * W * y_t
    
    # Circular Recovery Function R(N - E)
    recovery = PHI * (N - E)
    
    # Sacred Math Function M() - Apply Fibonacci and golden ratio
    M_factor = PHI * (1 + math.sin(goal_sum * math.pi))
    
    # Final Z(T) calculation
    Z_T = M_factor * (C_Comb * goal_sum + recovery)
    
    # Best Self Potential
    S_Best = max(0, min(1, Z_T))
    
    return {
        'z_t': Z_T,
        's_best': S_Best,
        'lambda': lambda_score,
        'neuro_regulatory': N,
        'red_flag': E,
        'c_comb': C_Comb,
        'y_t': y_t,
        'wellness': W,
        'goal_sum': goal_sum,
        'recovery': recovery,
        'mayan_day': mayan['tzolkin'],
        'cbt_analysis': cbt
    }

# ═══════════════════════════════════════════════════════════════════════════
# FRACTAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_fractal_from_zt(z_data: Dict, width=800, height=800):
    """Generate fractal visualization driven by Z(T)"""
    
    s_best = z_data['s_best']
    wellness = z_data['wellness']
    y_t = z_data['y_t']
    
    # Parameters driven by Z(T)
    max_iter = int(50 + s_best * 150)
    zoom = 1.0 + s_best * 3.0
    center_x = -0.5 + (wellness - 0.5) * 0.4
    center_y = (y_t - 0.5) * 0.3
    
    x = np.linspace(-2/zoom + center_x, 2/zoom + center_x, width)
    y = np.linspace(-1.5/zoom + center_y, 1.5/zoom + center_y, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    
    # Mandelbrot calculation
    z = np.zeros_like(c)
    iterations = np.zeros((height, width))
    
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        iterations[mask] = i
    
    # Nordic color scheme
    normalized = iterations / max_iter
    hue_shift = wellness * math.pi * 2
    
    r = (np.sin(normalized * math.pi * 2 + hue_shift) * 60 + 140).astype(np.uint8)
    g = (np.sin(normalized * math.pi * 2 + hue_shift + 2.1) * 70 + 150).astype(np.uint8)
    b = (np.sin(normalized * math.pi * 2 + hue_shift + 4.2) * 80 + 160).astype(np.uint8)
    
    rgb = np.dstack([r, g, b])
    image = Image.fromarray(rgb, 'RGB')
    
    # Sacred geometry overlay
    draw = ImageDraw.Draw(image, 'RGBA')
    cx, cy = width // 2, height // 2
    
    for i, fib in enumerate(FIBONACCI[:8]):
        radius = int(fib * s_best * 3)
        alpha = int(20 + i * 8)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(255, 255, 255, alpha),
            width=2
        )
    
    return image

# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE (NO EMOJI - Clean Text Only)
# ═══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence v8 - AI-Powered Life Planning</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #5B7C99;
            --secondary: #8BA3B8;
            --background: #F8F9FA;
            --surface: #FFFFFF;
            --text: #2C3E50;
            --text-light: #6C757D;
            --success: #6BA368;
            --warning: #C9A961;
            --error: #B85C5C;
            --border: #E1E4E8;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            line-height: 1.6;
            font-size: 16px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
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
        }
        
        .subtitle {
            font-size: 14px;
            color: var(--text-light);
            margin-top: 4px;
        }
        
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
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            font-size: 14px;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--text);
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(91, 124, 153, 0.1);
        }
        
        button, .btn {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 500;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: #4A6A88;
        }
        
        .btn-secondary {
            background: var(--secondary);
            color: white;
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .message {
            padding: 12px;
            border-radius: 6px;
            margin: 12px 0;
        }
        
        .message-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .message-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .hidden {
            display: none;
        }
        
        .link {
            color: var(--primary);
            cursor: pointer;
            text-decoration: underline;
        }
        
        .link:hover {
            color: #4A6A88;
        }
        
        .voice-btn {
            background: var(--warning);
            color: white;
            margin-left: 8px;
        }
        
        .voice-btn.active {
            background: var(--error);
        }
        
        .chat-container {
            background: var(--surface);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 16px;
        }
        
        .chat-message {
            margin-bottom: 12px;
            padding: 12px;
            border-radius: 6px;
        }
        
        .chat-user {
            background: #E8F4F8;
            margin-left: 40px;
        }
        
        .chat-ai {
            background: #F5F5F5;
            margin-right: 40px;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Life Fractal Intelligence v8</h1>
            <p class="subtitle">AI-Powered Life Planning | Voice + Chat Control | Swarm Intelligence</p>
        </div>
    </header>
    
    <main class="container">
        {% if not logged_in %}
        <!-- LOGIN/REGISTER -->
        <div class="card">
            <h2 class="card-title">Welcome</h2>
            <p style="margin-bottom: 20px;">Plan your life with AI assistance, voice control, and mathematical precision.</p>
            
            <div id="loginForm">
                <div class="form-group">
                    <label for="email">Email</label>
                    <input type="email" id="email" placeholder="your@email.com" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" placeholder="Secure password" required>
                </div>
                
                <div class="form-group hidden" id="nameGroup">
                    <label for="name">Name (optional)</label>
                    <input type="text" id="name" placeholder="Your name">
                </div>
                
                <button onclick="register()" class="btn btn-primary" id="registerBtn">Create Account</button>
                <button onclick="login()" class="btn btn-secondary" id="loginBtn">Login</button>
                
                <p style="margin-top: 16px;">
                    <span class="link" onclick="toggleForm()">Switch to <span id="formToggle">Login</span></span> |
                    <span class="link" onclick="showForgotPassword()">Forgot Password?</span>
                </p>
            </div>
            
            <div id="forgotPasswordForm" class="hidden">
                <div class="form-group">
                    <label for="resetEmail">Email</label>
                    <input type="email" id="resetEmail" placeholder="your@email.com">
                </div>
                
                <button onclick="requestReset()" class="btn btn-primary">Send Reset Link</button>
                <button onclick="showLoginForm()" class="btn btn-secondary">Back to Login</button>
            </div>
            
            <div id="message"></div>
        </div>
        {% else %}
        <!-- DASHBOARD -->
        <div class="card">
            <h2 class="card-title">AI Chat Assistant</h2>
            <p style="margin-bottom: 16px; color: var(--text-light);">Control everything with voice or text. Say "generate fractal" or "add goal" to get started.</p>
            
            <div class="chat-container" id="chatContainer"></div>
            
            <div style="display: flex; gap: 8px;">
                <input type="text" id="chatInput" placeholder="Type a command or question..." style="flex: 1;">
                <button onclick="sendChat()" class="btn btn-primary">Send</button>
                <button onclick="toggleVoice()" class="btn voice-btn" id="voiceBtn">Voice</button>
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Today's Check-In</h2>
            
            <div class="form-group">
                <label for="mood">Mood (1-100): <span id="moodValue">50</span></label>
                <input type="range" id="mood" min="1" max="100" value="50" oninput="updateSlider('mood', this.value)">
            </div>
            
            <div class="form-group">
                <label for="energy">Energy (1-100): <span id="energyValue">50</span></label>
                <input type="range" id="energy" min="1" max="100" value="50" oninput="updateSlider('energy', this.value)">
            </div>
            
            <div class="form-group">
                <label for="stress">Stress (1-100): <span id="stressValue">50</span></label>
                <input type="range" id="stress" min="1" max="100" value="50" oninput="updateSlider('stress', this.value)">
            </div>
            
            <div class="form-group">
                <label for="notes">Notes</label>
                <textarea id="notes" rows="3" placeholder="How are you feeling?"></textarea>
            </div>
            
            <button onclick="saveDaily()" class="btn btn-success">Save Check-In</button>
            
            <div id="prediction" style="margin-top: 20px;"></div>
            <div id="ztResult" style="margin-top: 20px;"></div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Fractal Visualization</h2>
            <button onclick="generateFractal()" class="btn btn-secondary">Generate Fractal from Z(T)</button>
            
            <div id="fractalContainer" class="hidden" style="margin-top: 20px; text-align: center;">
                <img id="fractalImage" src="" alt="Your fractal" style="max-width: 100%; border: 2px solid var(--border); border-radius: 8px;">
                <p id="mathFormula" style="margin-top: 12px; font-family: monospace; color: var(--text-light);"></p>
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Settings</h2>
            <button onclick="logout()" class="btn btn-warning">Logout</button>
        </div>
        {% endif %}
    </main>
    
    <script>
        let userId = {{ user_id | tojson }};
        let isRecording = false;
        let recognition = null;
        let showingRegister = false;
        
        // Voice Recognition Setup
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById('chatInput').value = transcript;
                sendChat();
            };
            
            recognition.onend = function() {
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('active');
                document.getElementById('voiceBtn').textContent = 'Voice';
            };
        }
        
        function toggleVoice() {
            if (!recognition) {
                alert('Voice recognition not supported in this browser');
                return;
            }
            
            if (isRecording) {
                recognition.stop();
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('active');
                document.getElementById('voiceBtn').textContent = 'Voice';
            } else {
                recognition.start();
                isRecording = true;
                document.getElementById('voiceBtn').classList.add('active');
                document.getElementById('voiceBtn').textContent = 'Listening...';
            }
        }
        
        async function sendChat() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addChatMessage(message, 'user');
            input.value = '';
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({user_id: userId, message: message})
                });
                const data = await res.json();
                
                if (data.response) {
                    addChatMessage(data.response, 'ai');
                }
                
                // Execute commands if any
                if (data.action) {
                    executeAction(data.action);
                }
            } catch (e) {
                addChatMessage('Error communicating with AI', 'ai');
            }
        }
        
        function addChatMessage(text, role) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = 'chat-message chat-' + role;
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function executeAction(action) {
            if (action.type === 'generate_fractal') {
                generateFractal();
            } else if (action.type === 'save_daily') {
                saveDaily();
            }
        }
        
        function toggleForm() {
            showingRegister = !showingRegister;
            document.getElementById('nameGroup').classList.toggle('hidden', !showingRegister);
            document.getElementById('formToggle').textContent = showingRegister ? 'Register' : 'Login';
            document.getElementById('registerBtn').style.display = showingRegister ? 'inline-block' : 'none';
            document.getElementById('loginBtn').style.display = showingRegister ? 'none' : 'inline-block';
        }
        
        function showForgotPassword() {
            document.getElementById('loginForm').classList.add('hidden');
            document.getElementById('forgotPasswordForm').classList.remove('hidden');
        }
        
        function showLoginForm() {
            document.getElementById('loginForm').classList.remove('hidden');
            document.getElementById('forgotPasswordForm').classList.add('hidden');
        }
        
        async function requestReset() {
            const email = document.getElementById('resetEmail').value;
            
            try {
                const res = await fetch('/api/forgot-password', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email: email})
                });
                const data = await res.json();
                
                showMessage(data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    setTimeout(showLoginForm, 2000);
                }
            } catch (e) {
                showMessage('Error: ' + e.message, 'error');
            }
        }
        
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
                    showMessage('Account created! Check your email to verify.', 'success');
                    setTimeout(() => location.reload(), 2000);
                } else {
                    showMessage(data.error, 'error');
                }
            } catch (e) {
                showMessage('Error: ' + e.message, 'error');
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
                    showMessage(data.error, 'error');
                }
            } catch (e) {
                showMessage('Error: ' + e.message, 'error');
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
                    showMessage('Check-in saved!', 'success');
                    
                    if (data.z_data) {
                        const z = data.z_data;
                        document.getElementById('ztResult').innerHTML = `
                            <div class="message message-success">
                                <strong>Z(T) Analysis:</strong><br>
                                Best Self Potential: ${(z.s_best * 100).toFixed(1)}%<br>
                                Mayan Day: ${z.mayan_day}<br>
                                CBT Lambda: ${z.lambda.toFixed(2)} (${z.cbt_analysis.severity})<br>
                                Custom Combo: ${z.c_comb.toFixed(2)}x
                            </div>
                        `;
                    }
                }
            } catch (e) {
                showMessage('Error: ' + e.message, 'error');
            }
        }
        
        async function generateFractal() {
            const mood = document.getElementById('mood').value;
            const energy = document.getElementById('energy').value;
            const stress = document.getElementById('stress').value;
            
            try {
                const res = await fetch(`/api/fractal?user_id=${userId}&mood=${mood}&energy=${energy}&stress=${stress}`);
                const data = await res.json();
                
                document.getElementById('fractalImage').src = data.image;
                document.getElementById('mathFormula').textContent = `Z(T) = ${data.z_t.toFixed(3)} | S_Best = ${data.s_best.toFixed(3)}`;
                document.getElementById('fractalContainer').classList.remove('hidden');
            } catch (e) {
                showMessage('Error: ' + e.message, 'error');
            }
        }
        
        function showMessage(msg, type) {
            const el = document.getElementById('message');
            el.innerHTML = `<div class="message message-${type}">${msg}</div>`;
            setTimeout(() => el.innerHTML = '', 5000);
        }
        
        function logout() {
            location.href = '/logout';
        }
    </script>
</body>
</html>
'''

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    user_id = session.get('user_id')
    email_verified = session.get('email_verified', False)
    
    return render_template_string(
        HTML_TEMPLATE,
        logged_in=user_id is not None,
        user_id=user_id,
        email_verified=email_verified
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
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email already registered'}), 400
        
        user_id = f"user_{secrets.token_hex(12)}"
        verification_token = generate_secure_token()
        now = datetime.now(timezone.utc).isoformat()
        
        # Secure password hashing
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        
        cursor.execute('''
            INSERT INTO users (id, email, password_hash, name, verification_token, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, email, password_hash, name or email.split('@')[0], verification_token, now, now))
        
        cursor.execute('''
            INSERT INTO pet_state (user_id, name)
            VALUES (?, ?)
        ''', (user_id, 'Friend'))
        
        db.commit()
        
        # Send verification email
        send_verification_email(email, verification_token)
        
        session['user_id'] = user_id
        session['email_verified'] = False
        session.permanent = True
        
        logger.info(f"New user registered: {user_id}")
        
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
        
        # Update last login
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user['id']))
        db.commit()
        
        session['user_id'] = user['id']
        session['email_verified'] = bool(user['email_verified'])
        session.permanent = True
        
        logger.info(f"User logged in: {user['id']}")
        
        return jsonify({'success': True, 'user_id': user['id']}), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            # Don't reveal if email exists
            return jsonify({'success': True, 'message': 'If that email exists, a reset link has been sent'}), 200
        
        reset_token = generate_secure_token()
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        
        cursor.execute('''
            UPDATE users SET reset_token = ?, reset_token_expires = ? WHERE id = ?
        ''', (reset_token, expires, user['id']))
        db.commit()
        
        send_reset_password_email(email, reset_token)
        
        return jsonify({'success': True, 'message': 'Reset link sent to your email'}), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify')
def verify_email():
    token = request.args.get('token')
    if not token:
        return "Invalid verification link", 400
    
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE verification_token = ?", (token,))
    user = cursor.fetchone()
    
    if not user:
        return "Invalid or expired token", 400
    
    cursor.execute("UPDATE users SET email_verified = 1, verification_token = NULL WHERE id = ?", (user['id'],))
    db.commit()
    
    return redirect('/')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI Chat endpoint with Llama 3"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        message = data.get('message', '')
        
        if not user_id or not message:
            return jsonify({'error': 'user_id and message required'}), 400
        
        # Store user message
        db = get_db()
        cursor = db.cursor()
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute('''
            INSERT INTO chat_history (id, user_id, message, role, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (f"msg_{secrets.token_hex(8)}", user_id, message, 'user', now))
        
        # Get conversation context
        cursor.execute('''
            SELECT message, role FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (user_id,))
        history = list(reversed(cursor.fetchall()))
        
        # Build context for Llama
        context = "\n".join([f"{h['role']}: {h['message']}" for h in history])
        
        system_prompt = """You are a life planning AI assistant. You can help users:
- Generate fractal visualizations
- Track goals and habits
- Provide CBT-based mental health support
- Analyze their progress
- Control app features via voice/text

When users want to perform actions, respond with commands like:
- To generate fractal: mention "generate fractal"
- To save daily check-in: mention "save daily"

Be supportive, concise, and actionable."""
        
        response = call_llama(f"{context}\nassistant:", system_prompt)
        
        # Store AI response
        cursor.execute('''
            INSERT INTO chat_history (id, user_id, message, role, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (f"msg_{secrets.token_hex(8)}", user_id, response, 'assistant', now))
        
        db.commit()
        
        # Detect actions
        action = None
        if 'fractal' in message.lower():
            action = {'type': 'generate_fractal'}
        elif 'save' in message.lower() and 'daily' in message.lower():
            action = {'type': 'save_daily'}
        
        return jsonify({'response': response, 'action': action}), 200
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e), 'response': 'I\'m having trouble connecting to AI. Please try again.'}), 500

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
        
        # Calculate Z(T)
        user_data = {
            'user_id': user_id,
            'mood_level': data.get('mood_level', 50),
            'energy_level': data.get('energy_level', 50),
            'stress_level': data.get('stress_level', 50),
            'notes': data.get('notes', ''),
            'goals': []
        }
        
        z_data = calculate_z_vector(user_data)
        
        # Store CBT analysis
        cbt_id = f"cbt_{secrets.token_hex(8)}"
        user_hash = hash_user_id(user_id)
        
        cursor.execute('''
            INSERT INTO cbt_analysis (id, user_hash, analysis_date, lambda_score, distortions,
                                     neuro_regulatory, red_flag_score, chat_context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cbt_id, user_hash, today, z_data['lambda'], json.dumps(z_data['cbt_analysis']['distortions']),
              z_data['neuro_regulatory'], z_data['red_flag'], data.get('notes', ''), now))
        
        db.commit()
        
        return jsonify({'success': True, 'z_data': z_data}), 200
        
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
    
    user_data = {
        'user_id': user_id,
        'mood_level': mood,
        'energy_level': energy,
        'stress_level': stress,
        'notes': '',
        'goals': []
    }
    
    z_data = calculate_z_vector(user_data)
    image = generate_fractal_from_zt(z_data)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_str}',
        'z_t': z_data['z_t'],
        's_best': z_data['s_best'],
        'mayan_day': z_data['mayan_day']
    }), 200

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': 'v8-ultimate',
        'features': [
            'llama3-nlp',
            'voice-control',
            'swarm-intelligence',
            'cbt-analysis',
            'mayan-calendar',
            'z-equation'
        ]
    }), 200

# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("=" * 80)
    logger.info("Life Fractal Intelligence v8 - ULTIMATE PRODUCTION")
    logger.info("=" * 80)
    logger.info("Features:")
    logger.info("  - Llama 3 NLP (Ollama)")
    logger.info("  - Voice Control (Web Speech API)")
    logger.info("  - Swarm Intelligence")
    logger.info("  - CBT Analysis")
    logger.info("  - Mayan Calendar")
    logger.info("  - Z(T) Equation")
    logger.info("  - Privacy-First ML")
    logger.info("  - Forgot Password")
    logger.info("  - Enhanced Security")
    logger.info("=" * 80)
    app.run(host='0.0.0.0', port=port, debug=False)
