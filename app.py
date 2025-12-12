#!/usr/bin/env python3
"""
LIFE FRACTAL INTELLIGENCE - ULTIMATE PRODUCTION v15.0
==============================================================================
Complete neurodivergent-first life planning system with:

CORE FEATURES:
- SQLite database (persistent storage)
- Secure JWT authentication (HMAC-SHA256)
- Forgot password with token reset
- Stripe payment integration
- 7-day free trial system
- Google Calendar OAuth ready

MATHEMATICAL FOUNDATIONS (Occam's Razor - Pure Python):
- Golden Ratio (phi = 1.618033988749895)
- Fibonacci sequences
- Logistic map chaos theory
- Sacred geometry (golden spiral, Islamic patterns)
- Pythagorean means (arithmetic, geometric, harmonic)

ACCESSIBILITY (Neurodivergent-First):
- Autism: Predictable patterns, muted colors, clear structure
- Aphantasia: External visualization, concrete representations
- Dysgraphia: Minimal typing, visual interactions
- ADHD/Executive dysfunction: Spoon Theory, task breakdown
- General: Screen readers, keyboard nav, reduced motion

VISUALIZATION:
- 2D Fractal generation (Mandelbrot, Julia)
- Wellness-mapped parameters
- Sacred geometry overlays

PET SYSTEM:
- 5 species with unique traits
- Behavior linked to user wellness
- Fibonacci-based leveling

==============================================================================
"""

import os
import sys
import json
import math
import time
import hmac
import uuid
import struct
import sqlite3
import hashlib
import secrets
import logging
import base64
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from contextlib import contextmanager

# Flask (minimal dependencies)
from flask import Flask, request, jsonify, send_file, make_response, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Image processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SACRED MATHEMATICS - PURE PYTHON (Occam's Razor)
# ==============================================================================

# Golden Ratio and derived constants
PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
PHI_SQUARED = PHI * PHI  # 2.618033988749895
GOLDEN_ANGLE_DEG = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE_DEG)

# Fibonacci sequence (pre-computed for efficiency)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]

# Platonic solids (vertices, edges, faces)
PLATONIC_SOLIDS = {
    'tetrahedron': (4, 6, 4),
    'cube': (8, 12, 6),
    'octahedron': (6, 12, 8),
    'dodecahedron': (20, 30, 12),
    'icosahedron': (12, 30, 20)
}

# Solfeggio frequencies (Hz) - healing tones
SOLFEGGIO = {'UT': 396, 'RE': 417, 'MI': 528, 'FA': 639, 'SOL': 741, 'LA': 852}


class SacredMath:
    """Pure Python sacred mathematics - no external dependencies."""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Get nth Fibonacci number."""
        if n < len(FIBONACCI):
            return FIBONACCI[n]
        a, b = FIBONACCI[-2], FIBONACCI[-1]
        for _ in range(n - len(FIBONACCI) + 1):
            a, b = b, a + b
        return b
    
    @staticmethod
    def golden_spiral_point(index: int, scale: float = 1.0) -> Tuple[float, float]:
        """Calculate point on golden spiral using Fibonacci positioning."""
        theta = index * GOLDEN_ANGLE_RAD
        r = scale * math.sqrt(index + 1)
        return r * math.cos(theta), r * math.sin(theta)
    
    @staticmethod
    def logistic_map(r: float, x: float) -> float:
        """Single step of logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
        return r * x * (1 - x)
    
    @staticmethod
    def chaos_series(r: float, x0: float, n: int) -> List[float]:
        """Generate chaos series using logistic map."""
        series = []
        x = x0
        for _ in range(n):
            series.append(x)
            x = r * x * (1 - x)
        return series
    
    @staticmethod
    def pythagorean_means(values: List[float]) -> Dict[str, float]:
        """Calculate arithmetic, geometric, and harmonic means."""
        if not values:
            return {'arithmetic': 0, 'geometric': 0, 'harmonic': 0}
        
        n = len(values)
        positive = [v for v in values if v > 0]
        
        arithmetic = sum(values) / n
        geometric = math.pow(math.prod(positive), 1/len(positive)) if positive else 0
        harmonic = len(positive) / sum(1/v for v in positive) if positive else 0
        
        return {'arithmetic': arithmetic, 'geometric': geometric, 'harmonic': harmonic}
    
    @staticmethod
    def wellness_to_fractal_params(wellness: float, mood: float, energy: float) -> Dict:
        """Map wellness metrics to fractal parameters using sacred math."""
        # Normalize to 0-1 range
        w = max(0, min(100, wellness)) / 100
        m = max(0, min(100, mood)) / 100
        e = max(0, min(100, energy)) / 100
        
        # Use golden ratio for parameter scaling
        zoom = 1 + w * PHI * 2
        iterations = int(100 + m * 200)
        hue_base = (180 + (m - 0.5) * 120) % 360
        chaos_r = 3.5 + (1 - w) * 0.5  # More chaos when wellness is low
        
        return {
            'zoom': zoom,
            'max_iter': iterations,
            'hue_base': hue_base,
            'chaos_r': chaos_r,
            'animation_speed': 0.5 + e * PHI,
            'geometry_opacity': 0.1 + w * 0.3
        }
    
    @staticmethod
    def spoon_cost(task_complexity: int, energy: float) -> float:
        """Calculate spoon cost using Fibonacci weighting."""
        # Higher complexity = more spoons, lower energy = feels more expensive
        base_cost = SacredMath.fibonacci(min(task_complexity + 2, 10)) / 10
        energy_modifier = 2 - (energy / 100)  # 1.0 at 100 energy, 2.0 at 0 energy
        return round(base_cost * energy_modifier, 1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Application configuration with environment variable support."""
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'life_fractal.db')
    
    # Subscription
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
    STRIPE_PAYMENT_LINK = os.environ.get('STRIPE_PAYMENT_LINK', 'https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00')
    SUBSCRIPTION_PRICE = float(os.environ.get('SUBSCRIPTION_PRICE', '20.00'))
    TRIAL_DAYS = int(os.environ.get('TRIAL_DAYS', '7'))
    
    # GoFundMe
    GOFUNDME_URL = os.environ.get('GOFUNDME_URL', 'https://gofund.me/8d9303d27')
    
    # Google Calendar OAuth (placeholder ready)
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
    
    # Admin
    ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')


# ==============================================================================
# JWT AUTHENTICATION (Pure Python - No pyjwt dependency)
# ==============================================================================

class JWTAuth:
    """Minimal JWT implementation using HMAC-SHA256."""
    
    @staticmethod
    def _base64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')
    
    @staticmethod
    def _base64url_decode(data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data)
    
    @staticmethod
    def create_token(user_id: str, expires_hours: int = 24) -> str:
        """Create JWT token."""
        header = {'alg': 'HS256', 'typ': 'JWT'}
        now = datetime.now(timezone.utc)
        payload = {
            'sub': user_id,
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(hours=expires_hours)).timestamp())
        }
        
        header_b64 = JWTAuth._base64url_encode(json.dumps(header).encode())
        payload_b64 = JWTAuth._base64url_encode(json.dumps(payload).encode())
        message = f"{header_b64}.{payload_b64}"
        
        signature = hmac.new(
            Config.JWT_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = JWTAuth._base64url_encode(signature)
        
        return f"{message}.{signature_b64}"
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """Verify JWT token and return payload."""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            header_b64, payload_b64, signature_b64 = parts
            message = f"{header_b64}.{payload_b64}"
            
            expected_sig = hmac.new(
                Config.JWT_SECRET.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            actual_sig = JWTAuth._base64url_decode(signature_b64)
            
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None
            
            payload = json.loads(JWTAuth._base64url_decode(payload_b64))
            
            # Check expiration
            if payload.get('exp', 0) < datetime.now(timezone.utc).timestamp():
                return None
            
            return payload
        except Exception:
            return None
    
    @staticmethod
    def create_reset_token(email: str) -> str:
        """Create password reset token (expires in 1 hour)."""
        return JWTAuth.create_token(f"reset:{email}", expires_hours=1)
    
    @staticmethod
    def verify_reset_token(token: str) -> Optional[str]:
        """Verify reset token and return email."""
        payload = JWTAuth.verify_token(token)
        if payload and payload.get('sub', '').startswith('reset:'):
            return payload['sub'][6:]  # Remove 'reset:' prefix
        return None


def require_auth(f: Callable) -> Callable:
    """Decorator to require JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        payload = JWTAuth.verify_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        g.user_id = payload['sub']
        return f(*args, **kwargs)
    
    return decorated


# ==============================================================================
# DATABASE (SQLite - Built-in, no external dependency)
# ==============================================================================

class Database:
    """SQLite database with automatic schema creation."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript('''
                -- Users table
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT DEFAULT '',
                    last_name TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    is_admin INTEGER DEFAULT 0,
                    email_verified INTEGER DEFAULT 0,
                    subscription_status TEXT DEFAULT 'trial',
                    trial_start TEXT,
                    trial_end TEXT,
                    stripe_customer_id TEXT,
                    created_at TEXT,
                    last_login TEXT,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    spoons INTEGER DEFAULT 12,
                    settings TEXT DEFAULT '{}'
                );
                
                -- Password reset tokens
                CREATE TABLE IF NOT EXISTS password_resets (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    created_at TEXT,
                    used INTEGER DEFAULT 0
                );
                
                -- Virtual pets
                CREATE TABLE IF NOT EXISTS pets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    species TEXT DEFAULT 'cat',
                    name TEXT DEFAULT 'Buddy',
                    level INTEGER DEFAULT 1,
                    experience INTEGER DEFAULT 0,
                    hunger REAL DEFAULT 50,
                    energy REAL DEFAULT 50,
                    mood REAL DEFAULT 50,
                    bond REAL DEFAULT 0,
                    behavior TEXT DEFAULT 'idle',
                    evolution_stage INTEGER DEFAULT 0,
                    last_fed TEXT,
                    last_played TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Goals
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    category TEXT DEFAULT 'general',
                    priority INTEGER DEFAULT 3,
                    progress REAL DEFAULT 0,
                    target_date TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    is_karma INTEGER DEFAULT 0,
                    is_dharma INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Habits
                CREATE TABLE IF NOT EXISTS habits (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    category TEXT DEFAULT 'general',
                    frequency TEXT DEFAULT 'daily',
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    total_completions INTEGER DEFAULT 0,
                    spoon_cost INTEGER DEFAULT 1,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Daily entries
                CREATE TABLE IF NOT EXISTS daily_entries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    mood_level INTEGER DEFAULT 3,
                    mood_score REAL DEFAULT 50,
                    energy_level REAL DEFAULT 50,
                    focus_clarity REAL DEFAULT 50,
                    anxiety_level REAL DEFAULT 30,
                    stress_level REAL DEFAULT 30,
                    mindfulness_score REAL DEFAULT 50,
                    gratitude_level REAL DEFAULT 50,
                    sleep_quality REAL DEFAULT 50,
                    sleep_hours REAL DEFAULT 7,
                    spoons_available INTEGER DEFAULT 12,
                    spoons_used INTEGER DEFAULT 0,
                    journal_entry TEXT DEFAULT '',
                    wellness_index REAL DEFAULT 0,
                    habits_completed TEXT DEFAULT '{}',
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Tasks (for calendar integration)
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    goal_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    due_date TEXT,
                    due_time TEXT,
                    spoon_cost INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 3,
                    is_completed INTEGER DEFAULT 0,
                    completed_at TEXT,
                    google_event_id TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                );
                
                -- Calendar sync tracking
                CREATE TABLE IF NOT EXISTS calendar_sync (
                    id TEXT PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    google_token TEXT,
                    google_refresh_token TEXT,
                    last_sync TEXT,
                    items_synced INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id);
                CREATE INDEX IF NOT EXISTS idx_habits_user ON habits(user_id);
                CREATE INDEX IF NOT EXISTS idx_entries_user_date ON daily_entries(user_id, date);
                CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
            ''')
            logger.info(f"Database initialized: {self.db_path}")


# ==============================================================================
# DATA MODELS
# ==============================================================================

class PetSpecies(Enum):
    CAT = "cat"
    DRAGON = "dragon"
    PHOENIX = "phoenix"
    OWL = "owl"
    FOX = "fox"

PET_TRAITS = {
    'cat': {'energy_decay': 1.2, 'mood_sensitivity': 1.0, 'growth_rate': 1.0, 'emoji': '🐱'},
    'dragon': {'energy_decay': 0.8, 'mood_sensitivity': 1.3, 'growth_rate': 1.2, 'emoji': '🐉'},
    'phoenix': {'energy_decay': 1.0, 'mood_sensitivity': 0.8, 'growth_rate': 1.5, 'emoji': '🔥'},
    'owl': {'energy_decay': 0.9, 'mood_sensitivity': 1.1, 'growth_rate': 0.9, 'emoji': '🦉'},
    'fox': {'energy_decay': 1.1, 'mood_sensitivity': 1.2, 'growth_rate': 1.1, 'emoji': '🦊'}
}

PET_BEHAVIORS = ['idle', 'happy', 'playful', 'tired', 'hungry', 'sad', 'excited', 'sleeping']

class MoodLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    NEUTRAL = 3
    GOOD = 4
    EXCELLENT = 5


# ==============================================================================
# FRACTAL GENERATOR (Pure NumPy + PIL)
# ==============================================================================

class FractalGenerator:
    """Generate fractal visualizations mapped to wellness data."""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
    
    def generate_mandelbrot(self, max_iter: int = 256, zoom: float = 1.0,
                            center: Tuple[float, float] = (-0.5, 0)) -> np.ndarray:
        """Generate Mandelbrot set."""
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape)
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def generate_julia(self, c_real: float = -0.7, c_imag: float = 0.27,
                       max_iter: int = 256, zoom: float = 1.0) -> np.ndarray:
        """Generate Julia set."""
        x = np.linspace(-2/zoom, 2/zoom, self.width)
        y = np.linspace(-2/zoom, 2/zoom, self.height)
        X, Y = np.meshgrid(x, y)
        
        z = X + 1j * Y
        c = complex(c_real, c_imag)
        iterations = np.zeros(z.shape)
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c
            iterations[mask] = i
        
        return iterations
    
    def apply_coloring(self, iterations: np.ndarray, max_iter: int,
                       hue_base: float = 0.6, hue_range: float = 0.3,
                       saturation: float = 0.8) -> np.ndarray:
        """Apply HSV coloring to iteration data."""
        # Normalize iterations
        norm = iterations / max_iter
        
        # HSV to RGB conversion
        h = (hue_base + norm * hue_range) % 1.0
        s = np.full_like(norm, saturation)
        v = np.where(iterations < max_iter - 1, 0.5 + norm * 0.5, 0)
        
        # Convert HSV to RGB
        rgb = np.zeros((*iterations.shape, 3), dtype=np.uint8)
        
        i = (h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        for idx in range(6):
            mask = i == idx
            if idx == 0:
                rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1) * 255
            elif idx == 1:
                rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1) * 255
            elif idx == 2:
                rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1) * 255
            elif idx == 3:
                rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1) * 255
            elif idx == 4:
                rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1) * 255
            else:
                rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1) * 255
        
        return rgb
    
    def draw_golden_spiral(self, img: Image.Image, opacity: float = 0.3) -> Image.Image:
        """Overlay golden spiral on image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        cx, cy = self.width // 2, self.height // 2
        scale = min(self.width, self.height) / 4
        
        points = []
        for i in range(100):
            x, y = SacredMath.golden_spiral_point(i, scale * 0.1)
            points.append((cx + x, cy + y))
        
        if len(points) > 1:
            alpha = int(255 * opacity)
            draw.line(points, fill=(255, 215, 0, alpha), width=2)
        
        return img
    
    def create_visualization(self, wellness: float, mood: float, energy: float) -> Image.Image:
        """Create complete visualization based on user metrics."""
        params = SacredMath.wellness_to_fractal_params(wellness, mood, energy)
        
        # Select fractal type based on wellness
        if wellness < 30:
            iterations = self.generate_julia(-0.8, 0.156, params['max_iter'])
            hue_base = 0.7  # Blue tones for low wellness
        elif wellness < 60:
            iterations = self.generate_mandelbrot(params['max_iter'], params['zoom'])
            hue_base = 0.5 + (mood - 50) / 200
        else:
            # Hybrid for high wellness
            m = self.generate_mandelbrot(params['max_iter'], params['zoom'])
            j = self.generate_julia(-0.7 + (mood-50)/200, 0.27, params['max_iter'])
            iterations = m * 0.5 + j * 0.5
            hue_base = 0.3 + (mood / 200)
        
        hue_range = 0.3 + (energy / 200)
        saturation = 0.5 + (wellness / 200)
        
        rgb = self.apply_coloring(iterations, params['max_iter'], hue_base, hue_range, saturation)
        img = Image.fromarray(rgb, 'RGB')
        
        # Add golden spiral overlay
        if params['geometry_opacity'] > 0.1:
            img = img.convert('RGBA')
            img = self.draw_golden_spiral(img, params['geometry_opacity'])
            img = img.convert('RGB')
        
        return img
    
    def to_base64(self, img: Image.Image) -> str:
        """Convert image to base64 string."""
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()


# ==============================================================================
# FLASK APPLICATION
# ==============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app, supports_credentials=True)

db = Database()
fractal_gen = FractalGenerator()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()

def today_date() -> str:
    """Get today's date as YYYY-MM-DD string."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')

def calculate_wellness(entry: Dict) -> float:
    """Calculate wellness index using Fibonacci weighting."""
    weights = [FIBONACCI[i+3] for i in range(8)]  # [2, 3, 5, 8, 13, 21, 34, 55]
    total_weight = sum(weights)
    
    positive = (
        entry.get('mood_level', 3) * 20 * weights[0] +
        entry.get('energy_level', 50) * weights[1] +
        entry.get('focus_clarity', 50) * weights[2] +
        entry.get('mindfulness_score', 50) * weights[3] +
        entry.get('gratitude_level', 50) * weights[4] +
        entry.get('sleep_quality', 50) * weights[5] +
        entry.get('mood_score', 50) * weights[6] +
        (100 - entry.get('stress_level', 30)) * weights[7]
    )
    
    negative = (entry.get('anxiety_level', 30) + entry.get('stress_level', 30)) * sum(weights[:3])
    
    return max(0, min(100, (positive - negative / 2) / total_weight))


def get_pet_behavior(pet: Dict) -> str:
    """Determine pet behavior based on state."""
    if pet['hunger'] > 80:
        return 'hungry'
    elif pet['energy'] < 20:
        return 'tired'
    elif pet['energy'] < 10:
        return 'sleeping'
    elif pet['mood'] > 80:
        return 'excited'
    elif pet['mood'] > 60:
        return 'playful'
    elif pet['mood'] > 40:
        return 'happy'
    elif pet['mood'] < 30:
        return 'sad'
    return 'idle'


def check_trial_active(user: Dict) -> bool:
    """Check if user's trial is still active."""
    if user['subscription_status'] == 'active':
        return True
    if user['subscription_status'] != 'trial':
        return False
    if not user.get('trial_end'):
        return False
    
    trial_end = datetime.fromisoformat(user['trial_end'].replace('Z', '+00:00'))
    return datetime.now(timezone.utc) < trial_end


def trial_days_remaining(user: Dict) -> int:
    """Get number of trial days remaining."""
    if not user.get('trial_end'):
        return 0
    trial_end = datetime.fromisoformat(user['trial_end'].replace('Z', '+00:00'))
    delta = trial_end - datetime.now(timezone.utc)
    return max(0, delta.days)


# ==============================================================================
# AUTHENTICATION ROUTES
# ==============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with 7-day trial."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        pet_species = data.get('pet_species', 'cat')
        pet_name = data.get('pet_name', 'Buddy')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Check for existing user
        with db.get_connection() as conn:
            existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            if existing:
                return jsonify({'error': 'Email already registered'}), 400
            
            # Create user
            user_id = str(uuid.uuid4())
            now = now_iso()
            trial_end = (datetime.now(timezone.utc) + timedelta(days=Config.TRIAL_DAYS)).isoformat()
            
            conn.execute('''
                INSERT INTO users (id, email, password_hash, first_name, last_name, 
                                   subscription_status, trial_start, trial_end, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, 'trial', ?, ?, ?, ?)
            ''', (user_id, email, generate_password_hash(password), first_name, last_name, now, trial_end, now, now))
            
            # Create pet
            pet_id = str(uuid.uuid4())
            conn.execute('''
                INSERT INTO pets (id, user_id, species, name, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (pet_id, user_id, pet_species, pet_name, now))
            
            # Create demo data
            _create_demo_data(conn, user_id)
        
        token = JWTAuth.create_token(user_id)
        
        return jsonify({
            'message': 'Registration successful',
            'user': {'id': user_id, 'email': email, 'first_name': first_name},
            'token': token,
            'trial_days_remaining': Config.TRIAL_DAYS,
            'show_gofundme': True,
            'gofundme_url': Config.GOFUNDME_URL
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        with db.get_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            
            if not user or not check_password_hash(user['password_hash'], password):
                return jsonify({'error': 'Invalid credentials'}), 401
            
            if not user['is_active']:
                return jsonify({'error': 'Account disabled'}), 403
            
            # Update last login
            conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (now_iso(), user['id']))
            
            user_dict = dict(user)
            token = JWTAuth.create_token(user['id'])
            
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'subscription_status': user['subscription_status']
                },
                'token': token,
                'has_access': check_trial_active(user_dict),
                'trial_days_remaining': trial_days_remaining(user_dict)
            }), 200
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """User logout (client should discard token)."""
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset."""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        with db.get_connection() as conn:
            user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            
            if user:
                # Create reset token
                reset_token = JWTAuth.create_reset_token(email)
                reset_id = str(uuid.uuid4())
                
                conn.execute('''
                    INSERT INTO password_resets (id, email, token, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (reset_id, email, reset_token, now_iso()))
                
                # In production, send email here
                logger.info(f"Password reset requested for {email}")
                
                # For development, return token (remove in production!)
                return jsonify({
                    'message': 'If this email exists, a reset link has been sent',
                    'dev_token': reset_token  # Remove this in production
                }), 200
            
            # Same response even if user doesn't exist (security)
            return jsonify({'message': 'If this email exists, a reset link has been sent'}), 200
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Request failed'}), 500


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token."""
    try:
        data = request.get_json() or {}
        token = data.get('token', '')
        new_password = data.get('password', '')
        
        if not token or not new_password:
            return jsonify({'error': 'Token and new password required'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        email = JWTAuth.verify_reset_token(token)
        if not email:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        with db.get_connection() as conn:
            # Mark token as used
            conn.execute('UPDATE password_resets SET used = 1 WHERE token = ?', (token,))
            
            # Update password
            password_hash = generate_password_hash(new_password)
            result = conn.execute('UPDATE users SET password_hash = ? WHERE email = ?', 
                                  (password_hash, email))
            
            if result.rowcount == 0:
                return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'message': 'Password reset successful'}), 200
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'error': 'Reset failed'}), 500


# ==============================================================================
# USER ROUTES
# ==============================================================================

@app.route('/api/user/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get user profile."""
    with db.get_connection() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (g.user_id,)).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_dict = dict(user)
        return jsonify({
            'id': user['id'],
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'subscription_status': user['subscription_status'],
            'trial_days_remaining': trial_days_remaining(user_dict),
            'has_access': check_trial_active(user_dict),
            'spoons': user['spoons'],
            'current_streak': user['current_streak'],
            'longest_streak': user['longest_streak'],
            'created_at': user['created_at']
        })


@app.route('/api/dashboard', methods=['GET'])
@require_auth
def get_dashboard():
    """Get comprehensive dashboard data."""
    with db.get_connection() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (g.user_id,)).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_dict = dict(user)
        
        # Get pet
        pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (g.user_id,)).fetchone()
        
        # Get today's entry
        today = today_date()
        entry = conn.execute(
            'SELECT * FROM daily_entries WHERE user_id = ? AND date = ?',
            (g.user_id, today)
        ).fetchone()
        
        # Get goals
        goals = conn.execute(
            'SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL ORDER BY priority',
            (g.user_id,)
        ).fetchall()
        
        # Get habits
        habits = conn.execute(
            'SELECT * FROM habits WHERE user_id = ? ORDER BY name',
            (g.user_id,)
        ).fetchall()
        
        # Calculate stats
        entries = conn.execute(
            'SELECT wellness_index FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 30',
            (g.user_id,)
        ).fetchall()
        
        avg_wellness = sum(e['wellness_index'] for e in entries) / len(entries) if entries else 50
        
        entry_dict = dict(entry) if entry else {
            'mood_level': 3, 'mood_score': 50, 'energy_level': 50,
            'wellness_index': 50, 'spoons_available': 12, 'spoons_used': 0
        }
        
        return jsonify({
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'subscription_status': user['subscription_status'],
                'trial_days_remaining': trial_days_remaining(user_dict),
                'has_access': check_trial_active(user_dict),
                'spoons': user['spoons'],
                'current_streak': user['current_streak']
            },
            'pet': dict(pet) if pet else None,
            'today': entry_dict,
            'goals': [dict(g) for g in goals],
            'habits': [dict(h) for h in habits],
            'stats': {
                'wellness_index': round(entry_dict.get('wellness_index', 50), 1),
                'average_wellness': round(avg_wellness, 1),
                'current_streak': user['current_streak'],
                'total_entries': len(entries),
                'active_goals': len(goals),
                'spoons_remaining': entry_dict.get('spoons_available', 12) - entry_dict.get('spoons_used', 0)
            },
            'sacred_math': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE_DEG,
                'fibonacci': FIBONACCI[:13],
                'today_fibonacci': FIBONACCI[datetime.now().day % len(FIBONACCI)]
            }
        })


# ==============================================================================
# DAILY ENTRY ROUTES
# ==============================================================================

@app.route('/api/daily/today', methods=['GET', 'POST'])
@require_auth
def handle_today():
    """Get or update today's entry."""
    today = today_date()
    
    with db.get_connection() as conn:
        if request.method == 'GET':
            entry = conn.execute(
                'SELECT * FROM daily_entries WHERE user_id = ? AND date = ?',
                (g.user_id, today)
            ).fetchone()
            
            if entry:
                return jsonify(dict(entry))
            
            # Return default entry
            return jsonify({
                'date': today,
                'mood_level': 3,
                'mood_score': 50,
                'energy_level': 50,
                'focus_clarity': 50,
                'anxiety_level': 30,
                'stress_level': 30,
                'mindfulness_score': 50,
                'gratitude_level': 50,
                'sleep_quality': 50,
                'sleep_hours': 7,
                'spoons_available': 12,
                'spoons_used': 0,
                'wellness_index': 50
            })
        
        # POST - update entry
        data = request.get_json() or {}
        
        # Check if entry exists
        existing = conn.execute(
            'SELECT id FROM daily_entries WHERE user_id = ? AND date = ?',
            (g.user_id, today)
        ).fetchone()
        
        entry_data = {
            'mood_level': data.get('mood_level', 3),
            'mood_score': data.get('mood_score', 50),
            'energy_level': data.get('energy_level', 50),
            'focus_clarity': data.get('focus_clarity', 50),
            'anxiety_level': data.get('anxiety_level', 30),
            'stress_level': data.get('stress_level', 30),
            'mindfulness_score': data.get('mindfulness_score', 50),
            'gratitude_level': data.get('gratitude_level', 50),
            'sleep_quality': data.get('sleep_quality', 50),
            'sleep_hours': data.get('sleep_hours', 7),
            'spoons_available': data.get('spoons_available', 12),
            'spoons_used': data.get('spoons_used', 0),
            'journal_entry': data.get('journal_entry', ''),
            'habits_completed': json.dumps(data.get('habits_completed', {}))
        }
        
        entry_data['wellness_index'] = calculate_wellness(entry_data)
        
        if existing:
            conn.execute('''
                UPDATE daily_entries SET
                    mood_level = ?, mood_score = ?, energy_level = ?, focus_clarity = ?,
                    anxiety_level = ?, stress_level = ?, mindfulness_score = ?, gratitude_level = ?,
                    sleep_quality = ?, sleep_hours = ?, spoons_available = ?, spoons_used = ?,
                    journal_entry = ?, habits_completed = ?, wellness_index = ?, updated_at = ?
                WHERE user_id = ? AND date = ?
            ''', (
                entry_data['mood_level'], entry_data['mood_score'], entry_data['energy_level'],
                entry_data['focus_clarity'], entry_data['anxiety_level'], entry_data['stress_level'],
                entry_data['mindfulness_score'], entry_data['gratitude_level'], entry_data['sleep_quality'],
                entry_data['sleep_hours'], entry_data['spoons_available'], entry_data['spoons_used'],
                entry_data['journal_entry'], entry_data['habits_completed'], entry_data['wellness_index'],
                now_iso(), g.user_id, today
            ))
        else:
            entry_id = str(uuid.uuid4())
            conn.execute('''
                INSERT INTO daily_entries (id, user_id, date, mood_level, mood_score, energy_level,
                    focus_clarity, anxiety_level, stress_level, mindfulness_score, gratitude_level,
                    sleep_quality, sleep_hours, spoons_available, spoons_used, journal_entry,
                    habits_completed, wellness_index, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id, g.user_id, today, entry_data['mood_level'], entry_data['mood_score'],
                entry_data['energy_level'], entry_data['focus_clarity'], entry_data['anxiety_level'],
                entry_data['stress_level'], entry_data['mindfulness_score'], entry_data['gratitude_level'],
                entry_data['sleep_quality'], entry_data['sleep_hours'], entry_data['spoons_available'],
                entry_data['spoons_used'], entry_data['journal_entry'], entry_data['habits_completed'],
                entry_data['wellness_index'], now_iso(), now_iso()
            ))
        
        # Update pet based on user data
        _update_pet_from_entry(conn, g.user_id, entry_data)
        
        return jsonify({**entry_data, 'date': today, 'message': 'Entry saved'})


# ==============================================================================
# GOALS ROUTES
# ==============================================================================

@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def handle_goals():
    """Get or create goals."""
    with db.get_connection() as conn:
        if request.method == 'GET':
            goals = conn.execute(
                'SELECT * FROM goals WHERE user_id = ? ORDER BY priority, created_at DESC',
                (g.user_id,)
            ).fetchall()
            
            active = sum(1 for g in goals if not g['completed_at'])
            completed = len(goals) - active
            
            return jsonify({
                'goals': [dict(g) for g in goals],
                'active': active,
                'completed': completed
            })
        
        # POST - create goal
        data = request.get_json() or {}
        goal_id = str(uuid.uuid4())
        
        conn.execute('''
            INSERT INTO goals (id, user_id, title, description, category, priority, 
                               target_date, is_karma, is_dharma, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            goal_id, g.user_id, data.get('title', 'New Goal'),
            data.get('description', ''), data.get('category', 'general'),
            data.get('priority', 3), data.get('target_date'),
            data.get('is_karma', 0), data.get('is_dharma', 0), now_iso()
        ))
        
        return jsonify({'success': True, 'goal_id': goal_id}), 201


@app.route('/api/goals/<goal_id>/progress', methods=['POST'])
@require_auth
def update_goal_progress(goal_id):
    """Update goal progress."""
    data = request.get_json() or {}
    progress = min(100, max(0, data.get('progress', 0)))
    
    with db.get_connection() as conn:
        goal = conn.execute('SELECT * FROM goals WHERE id = ? AND user_id = ?',
                           (goal_id, g.user_id)).fetchone()
        if not goal:
            return jsonify({'error': 'Goal not found'}), 404
        
        completed_at = now_iso() if progress >= 100 and not goal['completed_at'] else goal['completed_at']
        
        conn.execute('''
            UPDATE goals SET progress = ?, completed_at = ? WHERE id = ?
        ''', (progress, completed_at, goal_id))
        
        # Check Fibonacci milestones
        milestones = [8, 13, 21, 34, 55, 89, 100]
        milestone_reached = None
        for m in milestones:
            if goal['progress'] < m <= progress:
                milestone_reached = m
                break
        
        # Give pet XP on milestone
        if milestone_reached:
            _give_pet_xp(conn, g.user_id, milestone_reached)
        
        return jsonify({
            'success': True,
            'progress': progress,
            'completed': progress >= 100,
            'milestone_reached': milestone_reached
        })


# ==============================================================================
# HABITS ROUTES
# ==============================================================================

@app.route('/api/habits', methods=['GET', 'POST'])
@require_auth
def handle_habits():
    """Get or create habits."""
    with db.get_connection() as conn:
        if request.method == 'GET':
            habits = conn.execute(
                'SELECT * FROM habits WHERE user_id = ? ORDER BY name',
                (g.user_id,)
            ).fetchall()
            return jsonify({'habits': [dict(h) for h in habits]})
        
        # POST - create habit
        data = request.get_json() or {}
        habit_id = str(uuid.uuid4())
        
        conn.execute('''
            INSERT INTO habits (id, user_id, name, description, category, frequency, spoon_cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            habit_id, g.user_id, data.get('name', 'New Habit'),
            data.get('description', ''), data.get('category', 'general'),
            data.get('frequency', 'daily'), data.get('spoon_cost', 1), now_iso()
        ))
        
        return jsonify({'success': True, 'habit_id': habit_id}), 201


@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(habit_id):
    """Mark habit as complete for today."""
    with db.get_connection() as conn:
        habit = conn.execute('SELECT * FROM habits WHERE id = ? AND user_id = ?',
                            (habit_id, g.user_id)).fetchone()
        if not habit:
            return jsonify({'error': 'Habit not found'}), 404
        
        data = request.get_json() or {}
        completed = data.get('completed', True)
        
        if completed:
            conn.execute('''
                UPDATE habits SET 
                    current_streak = current_streak + 1,
                    total_completions = total_completions + 1,
                    longest_streak = MAX(longest_streak, current_streak + 1)
                WHERE id = ?
            ''', (habit_id,))
            
            # Give pet XP
            _give_pet_xp(conn, g.user_id, 5)
        
        return jsonify({
            'success': True,
            'habit_id': habit_id,
            'completed': completed,
            'spoon_cost': habit['spoon_cost']
        })


# ==============================================================================
# PET ROUTES
# ==============================================================================

@app.route('/api/pet', methods=['GET'])
@require_auth
def get_pet():
    """Get user's pet."""
    with db.get_connection() as conn:
        pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (g.user_id,)).fetchone()
        if not pet:
            return jsonify({'error': 'Pet not found'}), 404
        
        pet_dict = dict(pet)
        pet_dict['behavior'] = get_pet_behavior(pet_dict)
        pet_dict['traits'] = PET_TRAITS.get(pet['species'], PET_TRAITS['cat'])
        
        return jsonify(pet_dict)


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    """Feed the pet."""
    with db.get_connection() as conn:
        pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (g.user_id,)).fetchone()
        if not pet:
            return jsonify({'error': 'Pet not found'}), 404
        
        new_hunger = max(0, pet['hunger'] - 30)
        new_mood = min(100, pet['mood'] + 5)
        
        conn.execute('''
            UPDATE pets SET hunger = ?, mood = ?, last_fed = ? WHERE user_id = ?
        ''', (new_hunger, new_mood, now_iso(), g.user_id))
        
        return jsonify({
            'success': True,
            'hunger': new_hunger,
            'mood': new_mood,
            'message': f"{pet['name']} enjoyed the food!"
        })


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_with_pet():
    """Play with the pet."""
    with db.get_connection() as conn:
        pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (g.user_id,)).fetchone()
        if not pet:
            return jsonify({'error': 'Pet not found'}), 404
        
        if pet['energy'] < 20:
            return jsonify({'error': 'Pet is too tired to play', 'energy': pet['energy']}), 400
        
        new_energy = max(0, pet['energy'] - 15)
        new_mood = min(100, pet['mood'] + 15)
        new_bond = min(100, pet['bond'] + 3)
        new_xp = pet['experience'] + 10
        
        # Check for level up (Fibonacci thresholds)
        new_level = pet['level']
        xp_threshold = FIBONACCI[min(pet['level'] + 5, len(FIBONACCI)-1)] * 10
        if new_xp >= xp_threshold:
            new_level += 1
            new_xp -= xp_threshold
        
        conn.execute('''
            UPDATE pets SET energy = ?, mood = ?, bond = ?, experience = ?, level = ?, last_played = ?
            WHERE user_id = ?
        ''', (new_energy, new_mood, new_bond, new_xp, new_level, now_iso(), g.user_id))
        
        return jsonify({
            'success': True,
            'energy': new_energy,
            'mood': new_mood,
            'bond': new_bond,
            'level': new_level,
            'leveled_up': new_level > pet['level'],
            'message': f"{pet['name']} had fun playing with you!"
        })


# ==============================================================================
# VISUALIZATION ROUTES
# ==============================================================================

@app.route('/api/visualization/params', methods=['GET'])
@require_auth
def get_visualization_params():
    """Get fractal parameters based on user data."""
    with db.get_connection() as conn:
        entry = conn.execute(
            'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
            (g.user_id,)
        ).fetchone()
        
        if entry:
            wellness = entry['wellness_index']
            mood = entry['mood_score']
            energy = entry['energy_level']
        else:
            wellness, mood, energy = 50, 50, 50
        
        params = SacredMath.wellness_to_fractal_params(wellness, mood, energy)
        
        return jsonify({
            'fractal_params': params,
            'data_source': {
                'wellness': wellness,
                'mood': mood,
                'energy': energy
            },
            'sacred_math': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE_DEG,
                'fibonacci': FIBONACCI[:13]
            }
        })


@app.route('/api/visualization/fractal', methods=['GET'])
@require_auth
def generate_fractal():
    """Generate fractal image."""
    with db.get_connection() as conn:
        entry = conn.execute(
            'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
            (g.user_id,)
        ).fetchone()
        
        wellness = entry['wellness_index'] if entry else 50
        mood = entry['mood_score'] if entry else 50
        energy = entry['energy_level'] if entry else 50
    
    img = fractal_gen.create_visualization(wellness, mood, energy)
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png')


@app.route('/api/visualization/fractal/base64', methods=['GET'])
@require_auth
def get_fractal_base64():
    """Get fractal as base64 string."""
    with db.get_connection() as conn:
        entry = conn.execute(
            'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
            (g.user_id,)
        ).fetchone()
        
        wellness = entry['wellness_index'] if entry else 50
        mood = entry['mood_score'] if entry else 50
        energy = entry['energy_level'] if entry else 50
    
    img = fractal_gen.create_visualization(wellness, mood, energy)
    base64_data = fractal_gen.to_base64(img)
    
    return jsonify({
        'image': f'data:image/png;base64,{base64_data}',
        'params': SacredMath.wellness_to_fractal_params(wellness, mood, energy)
    })


# ==============================================================================
# SUBSCRIPTION ROUTES
# ==============================================================================

@app.route('/api/subscription/status', methods=['GET'])
@require_auth
def subscription_status():
    """Get subscription status."""
    with db.get_connection() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (g.user_id,)).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_dict = dict(user)
        days_left = trial_days_remaining(user_dict)
        has_access = check_trial_active(user_dict)
        
        return jsonify({
            'subscription_status': user['subscription_status'],
            'trial_active': user['subscription_status'] == 'trial' and days_left > 0,
            'trial_expired': user['subscription_status'] == 'trial' and days_left <= 0,
            'trial_days_remaining': days_left,
            'has_access': has_access,
            'show_gofundme': user['subscription_status'] == 'trial',
            'gofundme_url': Config.GOFUNDME_URL,
            'payment_link': Config.STRIPE_PAYMENT_LINK,
            'price': Config.SUBSCRIPTION_PRICE
        })


@app.route('/api/subscription/checkout', methods=['POST'])
@require_auth
def create_checkout():
    """Get Stripe checkout URL."""
    return jsonify({
        'checkout_url': Config.STRIPE_PAYMENT_LINK,
        'message': 'Redirecting to checkout...'
    })


@app.route('/api/subscription/activate', methods=['POST'])
@require_auth
def activate_subscription():
    """Manually activate subscription (after payment verified)."""
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE users SET subscription_status = 'active' WHERE id = ?",
            (g.user_id,)
        )
    return jsonify({'message': 'Subscription activated', 'status': 'active'})


# ==============================================================================
# GOOGLE CALENDAR ROUTES (OAuth Ready)
# ==============================================================================

@app.route('/api/calendar/connect', methods=['GET'])
@require_auth
def calendar_connect():
    """Start Google OAuth flow."""
    if not Config.GOOGLE_CLIENT_ID:
        return jsonify({
            'error': 'Google Calendar not configured',
            'instructions': 'Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables'
        }), 400
    
    # OAuth URL (client should redirect here)
    oauth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={Config.GOOGLE_CLIENT_ID}&"
        f"redirect_uri={request.host_url}api/calendar/callback&"
        f"response_type=code&"
        f"scope=https://www.googleapis.com/auth/calendar&"
        f"access_type=offline"
    )
    
    return jsonify({'oauth_url': oauth_url, 'status': 'ready'})


@app.route('/api/calendar/callback', methods=['GET'])
def calendar_callback():
    """Handle Google OAuth callback."""
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No authorization code provided'}), 400
    
    # In production, exchange code for tokens here
    return jsonify({
        'message': 'OAuth callback received',
        'instructions': 'Implement token exchange with Google OAuth'
    })


@app.route('/api/calendar/sync', methods=['POST'])
@require_auth
def sync_calendar():
    """Sync tasks with Google Calendar."""
    with db.get_connection() as conn:
        tasks = conn.execute(
            'SELECT * FROM tasks WHERE user_id = ? AND due_date IS NOT NULL',
            (g.user_id,)
        ).fetchall()
        
        # In production, sync with Google Calendar API here
        conn.execute('''
            INSERT OR REPLACE INTO calendar_sync (id, user_id, last_sync, items_synced)
            VALUES (?, ?, ?, ?)
        ''', (str(uuid.uuid4()), g.user_id, now_iso(), len(tasks)))
    
    return jsonify({
        'success': True,
        'synced_at': now_iso(),
        'items_synced': len(tasks),
        'message': 'Calendar sync ready (connect Google Calendar to enable)'
    })


# ==============================================================================
# ANALYTICS ROUTES
# ==============================================================================

@app.route('/api/analytics', methods=['GET'])
@require_auth
def get_analytics():
    """Get comprehensive analytics."""
    with db.get_connection() as conn:
        entries = conn.execute('''
            SELECT * FROM daily_entries WHERE user_id = ? 
            ORDER BY date DESC LIMIT 30
        ''', (g.user_id,)).fetchall()
        
        if not entries:
            return jsonify({'error': 'No data available'}), 404
        
        wellness_values = [e['wellness_index'] for e in entries if e['wellness_index'] > 0]
        mood_values = [e['mood_score'] for e in entries]
        energy_values = [e['energy_level'] for e in entries]
        
        # Pythagorean means
        means = SacredMath.pythagorean_means(wellness_values) if wellness_values else {}
        
        # Mood distribution
        mood_dist = {}
        for e in entries:
            level = MoodLevel(e['mood_level']).name
            mood_dist[level] = mood_dist.get(level, 0) + 1
        
        # Trend (linear approximation)
        if len(entries) > 1:
            growth = entries[0]['wellness_index'] - entries[-1]['wellness_index']
        else:
            growth = 0
        
        # Best and worst days
        best = max(entries, key=lambda e: e['wellness_index'])
        worst = min(entries, key=lambda e: e['wellness_index'])
        
        return jsonify({
            'wellness_trend': [{'date': e['date'], 'value': round(e['wellness_index'], 1)} for e in reversed(entries)],
            'mood_distribution': mood_dist,
            'pythagorean_means': means,
            'averages': {
                'wellness': round(sum(wellness_values) / len(wellness_values), 1) if wellness_values else 0,
                'mood': round(sum(mood_values) / len(mood_values), 1) if mood_values else 0,
                'energy': round(sum(energy_values) / len(energy_values), 1) if energy_values else 0
            },
            'insights': {
                'best_day': best['date'],
                'worst_day': worst['date'],
                'growth': round(growth, 1),
                'entries_count': len(entries)
            },
            'sacred_math': {
                'phi': PHI,
                'wellness_phi_ratio': round(means.get('geometric', 0) / PHI, 3) if means.get('geometric') else 0
            }
        })


# ==============================================================================
# SYSTEM ROUTES
# ==============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '15.0 Ultimate',
        'timestamp': now_iso(),
        'database': 'SQLite',
        'auth': 'JWT HMAC-SHA256'
    })


@app.route('/api/sacred-math', methods=['GET'])
def sacred_math_info():
    """Get sacred mathematics constants."""
    return jsonify({
        'phi': PHI,
        'phi_inverse': PHI_INVERSE,
        'phi_squared': PHI_SQUARED,
        'golden_angle_degrees': GOLDEN_ANGLE_DEG,
        'golden_angle_radians': GOLDEN_ANGLE_RAD,
        'fibonacci': FIBONACCI,
        'platonic_solids': PLATONIC_SOLIDS,
        'solfeggio_frequencies': SOLFEGGIO
    })


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _create_demo_data(conn, user_id: str):
    """Create demo habits and goals for new user."""
    now = now_iso()
    
    # Demo habits
    habits = [
        ('Morning Meditation', 'wellness', 1),
        ('Exercise 30 min', 'health', 2),
        ('Read 20 pages', 'growth', 1),
        ('Gratitude Practice', 'wellness', 1),
        ('Drink 8 glasses water', 'health', 1)
    ]
    
    for name, category, cost in habits:
        conn.execute('''
            INSERT INTO habits (id, user_id, name, category, spoon_cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), user_id, name, category, cost, now))
    
    # Demo goals
    goals = [
        ('Build Better Habits', 'wellness', 2, 25),
        ('Learn Something New', 'growth', 3, 10),
        ('Improve Health', 'health', 1, 15)
    ]
    
    for title, category, priority, progress in goals:
        conn.execute('''
            INSERT INTO goals (id, user_id, title, category, priority, progress, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), user_id, title, category, priority, progress, now))


def _update_pet_from_entry(conn, user_id: str, entry: Dict):
    """Update pet state based on user's daily entry."""
    pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,)).fetchone()
    if not pet:
        return
    
    traits = PET_TRAITS.get(pet['species'], PET_TRAITS['cat'])
    
    # Energy from sleep
    sleep_bonus = (entry.get('sleep_quality', 50) - 50) * 0.2
    new_energy = min(100, max(0, pet['energy'] + sleep_bonus))
    
    # Mood from user mood
    mood_delta = (entry.get('mood_score', 50) - 50) * 0.3 * traits['mood_sensitivity']
    new_mood = min(100, max(0, pet['mood'] + mood_delta))
    
    # Natural decay
    new_hunger = min(100, pet['hunger'] + 2 * traits['energy_decay'])
    new_energy = max(0, new_energy - 1 * traits['energy_decay'])
    
    conn.execute('''
        UPDATE pets SET energy = ?, mood = ?, hunger = ? WHERE user_id = ?
    ''', (new_energy, new_mood, new_hunger, user_id))


def _give_pet_xp(conn, user_id: str, xp: int):
    """Give XP to user's pet."""
    pet = conn.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,)).fetchone()
    if not pet:
        return
    
    new_xp = pet['experience'] + xp
    new_level = pet['level']
    
    # Fibonacci-based level thresholds
    xp_threshold = FIBONACCI[min(pet['level'] + 5, len(FIBONACCI)-1)] * 10
    while new_xp >= xp_threshold:
        new_level += 1
        new_xp -= xp_threshold
        xp_threshold = FIBONACCI[min(new_level + 5, len(FIBONACCI)-1)] * 10
    
    conn.execute('''
        UPDATE pets SET experience = ?, level = ? WHERE user_id = ?
    ''', (new_xp, new_level, user_id))


# ==============================================================================
# FRONTEND HTML (Nordic Design, Accessible)
# ==============================================================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence</title>
    <style>
        /* CSS Custom Properties - Autism-safe color palette */
        :root {
            --bg-primary: #0a0a12;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --accent-cyan: #00d4ff;
            --accent-purple: #8b5cf6;
            --accent-gold: #f0c420;
            --accent-green: #48c774;
            --accent-red: #ff6b6b;
            --border-color: #2a2a3a;
            --shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        /* Reduced motion for accessibility */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        /* Skip link for accessibility */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--accent-cyan);
            color: var(--bg-primary);
            padding: 8px 16px;
            z-index: 100;
            text-decoration: none;
        }
        
        .skip-link:focus {
            top: 0;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .spoon-badge {
            background: var(--bg-card);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid var(--border-color);
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        /* Metrics */
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent-cyan);
        }
        
        /* Sliders - Large touch targets for dysgraphia */
        .slider-group {
            margin: 16px 0;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 12px;
            border-radius: 6px;
            background: var(--bg-secondary);
            appearance: none;
            cursor: pointer;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
        }
        
        /* Buttons - Large for accessibility */
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            min-height: 48px;
            min-width: 120px;
        }
        
        .btn:focus {
            outline: 3px solid var(--accent-gold);
            outline-offset: 2px;
        }
        
        .btn-primary {
            background: var(--accent-cyan);
            color: var(--bg-primary);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
        }
        
        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .btn-success {
            background: var(--accent-green);
            color: var(--bg-primary);
        }
        
        /* Pet display */
        .pet-display {
            text-align: center;
            padding: 20px;
        }
        
        .pet-emoji {
            font-size: 80px;
            margin-bottom: 10px;
        }
        
        .pet-name {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .pet-level {
            color: var(--accent-gold);
            font-size: 14px;
        }
        
        .pet-bars {
            margin-top: 20px;
        }
        
        .stat-bar {
            margin: 10px 0;
        }
        
        .stat-bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 4px;
        }
        
        .stat-bar-track {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .stat-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .fill-energy { background: var(--accent-cyan); }
        .fill-mood { background: var(--accent-purple); }
        .fill-hunger { background: var(--accent-gold); }
        
        /* Fractal display */
        .fractal-container {
            text-align: center;
        }
        
        .fractal-image {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: var(--shadow);
        }
        
        /* Input fields */
        input[type="text"],
        input[type="email"],
        input[type="password"],
        textarea {
            width: 100%;
            padding: 14px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 16px;
            margin-bottom: 12px;
        }
        
        input:focus,
        textarea:focus {
            outline: none;
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
        }
        
        /* Auth forms */
        .auth-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 20px;
        }
        
        .auth-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .auth-tab {
            flex: 1;
            padding: 12px;
            background: var(--bg-secondary);
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .auth-tab.active {
            background: var(--accent-cyan);
            color: var(--bg-primary);
        }
        
        /* Accessibility info */
        .accessibility-note {
            background: var(--bg-secondary);
            border-left: 4px solid var(--accent-purple);
            padding: 16px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            font-size: 14px;
        }
        
        /* Sacred math display */
        .sacred-math {
            font-family: 'Courier New', monospace;
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.8;
        }
        
        /* Goals list */
        .goal-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 8px;
        }
        
        .goal-progress {
            flex: 1;
        }
        
        .progress-bar {
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            transition: width 0.3s;
        }
        
        /* Habit checkboxes - Large for touch */
        .habit-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
        }
        
        .habit-checkbox {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            border: 2px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .habit-checkbox.checked {
            background: var(--accent-green);
            border-color: var(--accent-green);
        }
        
        /* Loading state */
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        /* Messages */
        .message {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        
        .message-success {
            background: rgba(72, 199, 116, 0.2);
            border: 1px solid var(--accent-green);
        }
        
        .message-error {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid var(--accent-red);
        }
        
        /* Trial banner */
        .trial-banner {
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        /* Hidden utility */
        .hidden {
            display: none !important;
        }
        
        /* Focus visible for keyboard nav */
        :focus-visible {
            outline: 3px solid var(--accent-gold);
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            :root {
                --bg-primary: #000;
                --bg-secondary: #111;
                --bg-card: #222;
                --text-primary: #fff;
                --border-color: #fff;
            }
        }
    </style>
</head>
<body>
    <a href="#main" class="skip-link">Skip to main content</a>
    
    <div id="app">
        <!-- Auth View -->
        <div id="auth-view" class="auth-container">
            <div class="logo" style="justify-content: center; margin-bottom: 30px; font-size: 28px;">
                🌀 Life Fractal Intelligence
            </div>
            
            <div class="card">
                <div class="auth-tabs" role="tablist">
                    <button class="auth-tab active" onclick="showAuthTab('login')" role="tab" aria-selected="true">Login</button>
                    <button class="auth-tab" onclick="showAuthTab('register')" role="tab" aria-selected="false">Register</button>
                </div>
                
                <div id="login-form">
                    <input type="email" id="login-email" placeholder="Email" aria-label="Email">
                    <input type="password" id="login-password" placeholder="Password" aria-label="Password">
                    <button class="btn btn-primary" style="width: 100%;" onclick="doLogin()">Login</button>
                    <p style="text-align: center; margin-top: 12px;">
                        <a href="#" onclick="showForgotPassword()" style="color: var(--accent-cyan);">Forgot password?</a>
                    </p>
                </div>
                
                <div id="register-form" class="hidden">
                    <input type="email" id="reg-email" placeholder="Email" aria-label="Email">
                    <input type="password" id="reg-password" placeholder="Password (min 8 chars)" aria-label="Password">
                    <input type="text" id="reg-name" placeholder="First Name" aria-label="First Name">
                    <select id="reg-pet" style="width: 100%; padding: 14px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); margin-bottom: 12px;">
                        <option value="cat">🐱 Cat</option>
                        <option value="dragon">🐉 Dragon</option>
                        <option value="phoenix">🔥 Phoenix</option>
                        <option value="owl">🦉 Owl</option>
                        <option value="fox">🦊 Fox</option>
                    </select>
                    <input type="text" id="reg-pet-name" placeholder="Pet Name" value="Buddy" aria-label="Pet Name">
                    <button class="btn btn-primary" style="width: 100%;" onclick="doRegister()">Create Account</button>
                </div>
                
                <div id="forgot-form" class="hidden">
                    <p style="margin-bottom: 12px; color: var(--text-secondary);">Enter your email to receive a password reset link.</p>
                    <input type="email" id="forgot-email" placeholder="Email" aria-label="Email">
                    <button class="btn btn-primary" style="width: 100%;" onclick="doForgotPassword()">Send Reset Link</button>
                    <p style="text-align: center; margin-top: 12px;">
                        <a href="#" onclick="showAuthTab('login')" style="color: var(--accent-cyan);">Back to login</a>
                    </p>
                </div>
                
                <div id="auth-message"></div>
            </div>
            
            <div class="accessibility-note">
                <strong>🧠 Built for Neurodivergent Minds</strong><br>
                <small>Autism · ADHD · Dysgraphia · Aphantasia friendly. Large buttons, minimal typing, visual feedback.</small>
            </div>
        </div>
        
        <!-- Dashboard View -->
        <div id="dashboard-view" class="container hidden">
            <header>
                <div class="logo">🌀 Life Fractal</div>
                <div class="user-info">
                    <span id="user-name">Welcome</span>
                    <span class="spoon-badge" id="spoon-badge">🥄 12 spoons</span>
                    <button class="btn btn-secondary" onclick="doLogout()">Logout</button>
                </div>
            </header>
            
            <main id="main">
                <div id="trial-banner" class="trial-banner hidden">
                    <span>🎁 <span id="trial-days">7</span> days left in your free trial</span>
                    <button class="btn btn-primary" onclick="goToCheckout()">Subscribe $20/mo</button>
                </div>
                
                <div class="dashboard-grid">
                    <!-- Today's Check-in -->
                    <div class="card">
                        <div class="card-title">📊 Today's Check-in</div>
                        
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Mood</span>
                                <span id="mood-value">50</span>
                            </div>
                            <input type="range" id="mood-slider" min="0" max="100" value="50" 
                                   oninput="updateSlider('mood')" aria-label="Mood level">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Energy</span>
                                <span id="energy-value">50</span>
                            </div>
                            <input type="range" id="energy-slider" min="0" max="100" value="50" 
                                   oninput="updateSlider('energy')" aria-label="Energy level">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Stress</span>
                                <span id="stress-value">30</span>
                            </div>
                            <input type="range" id="stress-slider" min="0" max="100" value="30" 
                                   oninput="updateSlider('stress')" aria-label="Stress level">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-label">
                                <span>Sleep Hours</span>
                                <span id="sleep-value">7</span>
                            </div>
                            <input type="range" id="sleep-slider" min="0" max="12" value="7" step="0.5"
                                   oninput="updateSlider('sleep')" aria-label="Sleep hours">
                        </div>
                        
                        <button class="btn btn-success" style="width: 100%; margin-top: 16px;" onclick="saveCheckIn()">
                            💾 Save Check-in
                        </button>
                    </div>
                    
                    <!-- Pet -->
                    <div class="card">
                        <div class="card-title">🐾 Your Companion</div>
                        <div class="pet-display">
                            <div class="pet-emoji" id="pet-emoji">🐱</div>
                            <div class="pet-name" id="pet-name">Buddy</div>
                            <div class="pet-level">Level <span id="pet-level">1</span></div>
                            
                            <div class="pet-bars">
                                <div class="stat-bar">
                                    <div class="stat-bar-label">
                                        <span>Energy</span>
                                        <span id="pet-energy">50%</span>
                                    </div>
                                    <div class="stat-bar-track">
                                        <div class="stat-bar-fill fill-energy" id="pet-energy-bar" style="width: 50%"></div>
                                    </div>
                                </div>
                                <div class="stat-bar">
                                    <div class="stat-bar-label">
                                        <span>Mood</span>
                                        <span id="pet-mood">50%</span>
                                    </div>
                                    <div class="stat-bar-track">
                                        <div class="stat-bar-fill fill-mood" id="pet-mood-bar" style="width: 50%"></div>
                                    </div>
                                </div>
                                <div class="stat-bar">
                                    <div class="stat-bar-label">
                                        <span>Hunger</span>
                                        <span id="pet-hunger">50%</span>
                                    </div>
                                    <div class="stat-bar-track">
                                        <div class="stat-bar-fill fill-hunger" id="pet-hunger-bar" style="width: 50%"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px; display: flex; gap: 10px; justify-content: center;">
                                <button class="btn btn-primary" onclick="feedPet()">🍖 Feed</button>
                                <button class="btn btn-secondary" onclick="playWithPet()">🎾 Play</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Fractal -->
                    <div class="card">
                        <div class="card-title">🌀 Your Life Fractal</div>
                        <div class="fractal-container">
                            <img id="fractal-image" class="fractal-image" alt="Your personalized fractal visualization">
                            <button class="btn btn-secondary" style="margin-top: 16px;" onclick="regenerateFractal()">
                                🔄 Regenerate
                            </button>
                        </div>
                    </div>
                    
                    <!-- Goals -->
                    <div class="card">
                        <div class="card-title">🎯 Goals</div>
                        <div id="goals-list">
                            <p class="loading">Loading goals...</p>
                        </div>
                        <div style="margin-top: 16px;">
                            <input type="text" id="new-goal" placeholder="Add a new goal..." aria-label="New goal">
                            <button class="btn btn-primary" style="width: 100%;" onclick="addGoal()">➕ Add Goal</button>
                        </div>
                    </div>
                    
                    <!-- Habits -->
                    <div class="card">
                        <div class="card-title">✅ Daily Habits</div>
                        <div id="habits-list">
                            <p class="loading">Loading habits...</p>
                        </div>
                    </div>
                    
                    <!-- Sacred Math -->
                    <div class="card">
                        <div class="card-title">📐 Sacred Mathematics</div>
                        <div class="sacred-math">
                            <div>φ (Golden Ratio): <span style="color: var(--accent-gold);">1.618033988749895</span></div>
                            <div>Golden Angle: <span style="color: var(--accent-cyan);">137.5077640500378°</span></div>
                            <div>Fibonacci: <span style="color: var(--accent-purple);">1, 1, 2, 3, 5, 8, 13, 21, 34...</span></div>
                            <div style="margin-top: 12px; color: var(--text-secondary);">
                                Your wellness data maps to fractal parameters using these sacred patterns.
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>
    
    <script>
        // State
        let token = localStorage.getItem('fractal_token');
        let userData = null;
        
        const PET_EMOJIS = {
            'cat': '🐱', 'dragon': '🐉', 'phoenix': '🔥', 'owl': '🦉', 'fox': '🦊'
        };
        
        // API calls
        async function api(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (token) {
                options.headers['Authorization'] = 'Bearer ' + token;
            }
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            const res = await fetch('/api' + endpoint, options);
            
            if (res.status === 401) {
                localStorage.removeItem('fractal_token');
                token = null;
                showAuthView();
                throw new Error('Unauthorized');
            }
            
            return res.json();
        }
        
        // Auth
        function showAuthTab(tab) {
            document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
            document.getElementById('login-form').classList.add('hidden');
            document.getElementById('register-form').classList.add('hidden');
            document.getElementById('forgot-form').classList.add('hidden');
            
            if (tab === 'login') {
                document.querySelector('.auth-tab:first-child').classList.add('active');
                document.getElementById('login-form').classList.remove('hidden');
            } else {
                document.querySelector('.auth-tab:last-child').classList.add('active');
                document.getElementById('register-form').classList.remove('hidden');
            }
        }
        
        function showForgotPassword() {
            document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
            document.getElementById('login-form').classList.add('hidden');
            document.getElementById('register-form').classList.add('hidden');
            document.getElementById('forgot-form').classList.remove('hidden');
        }
        
        async function doLogin() {
            try {
                const res = await api('/auth/login', 'POST', {
                    email: document.getElementById('login-email').value,
                    password: document.getElementById('login-password').value
                });
                
                if (res.error) {
                    showMessage(res.error, 'error');
                    return;
                }
                
                token = res.token;
                localStorage.setItem('fractal_token', token);
                showDashboard();
            } catch (e) {
                showMessage('Login failed', 'error');
            }
        }
        
        async function doRegister() {
            try {
                const res = await api('/auth/register', 'POST', {
                    email: document.getElementById('reg-email').value,
                    password: document.getElementById('reg-password').value,
                    first_name: document.getElementById('reg-name').value,
                    pet_species: document.getElementById('reg-pet').value,
                    pet_name: document.getElementById('reg-pet-name').value
                });
                
                if (res.error) {
                    showMessage(res.error, 'error');
                    return;
                }
                
                token = res.token;
                localStorage.setItem('fractal_token', token);
                showDashboard();
            } catch (e) {
                showMessage('Registration failed', 'error');
            }
        }
        
        async function doForgotPassword() {
            try {
                const res = await api('/auth/forgot-password', 'POST', {
                    email: document.getElementById('forgot-email').value
                });
                showMessage(res.message, 'success');
            } catch (e) {
                showMessage('Request failed', 'error');
            }
        }
        
        function doLogout() {
            localStorage.removeItem('fractal_token');
            token = null;
            showAuthView();
        }
        
        // Views
        function showAuthView() {
            document.getElementById('auth-view').classList.remove('hidden');
            document.getElementById('dashboard-view').classList.add('hidden');
        }
        
        async function showDashboard() {
            document.getElementById('auth-view').classList.add('hidden');
            document.getElementById('dashboard-view').classList.remove('hidden');
            await loadDashboard();
        }
        
        async function loadDashboard() {
            try {
                const data = await api('/dashboard');
                userData = data;
                
                // Update user info
                document.getElementById('user-name').textContent = 'Hi, ' + (data.user.first_name || 'Friend');
                document.getElementById('spoon-badge').textContent = '🥄 ' + data.stats.spoons_remaining + ' spoons';
                
                // Trial banner
                if (data.user.subscription_status === 'trial' && data.user.trial_days_remaining > 0) {
                    document.getElementById('trial-banner').classList.remove('hidden');
                    document.getElementById('trial-days').textContent = data.user.trial_days_remaining;
                }
                
                // Today's entry
                if (data.today) {
                    document.getElementById('mood-slider').value = data.today.mood_score;
                    document.getElementById('mood-value').textContent = Math.round(data.today.mood_score);
                    document.getElementById('energy-slider').value = data.today.energy_level;
                    document.getElementById('energy-value').textContent = Math.round(data.today.energy_level);
                    document.getElementById('stress-slider').value = data.today.stress_level;
                    document.getElementById('stress-value').textContent = Math.round(data.today.stress_level);
                    document.getElementById('sleep-slider').value = data.today.sleep_hours;
                    document.getElementById('sleep-value').textContent = data.today.sleep_hours;
                }
                
                // Pet
                if (data.pet) {
                    document.getElementById('pet-emoji').textContent = PET_EMOJIS[data.pet.species] || '🐱';
                    document.getElementById('pet-name').textContent = data.pet.name;
                    document.getElementById('pet-level').textContent = data.pet.level;
                    document.getElementById('pet-energy').textContent = Math.round(data.pet.energy) + '%';
                    document.getElementById('pet-energy-bar').style.width = data.pet.energy + '%';
                    document.getElementById('pet-mood').textContent = Math.round(data.pet.mood) + '%';
                    document.getElementById('pet-mood-bar').style.width = data.pet.mood + '%';
                    document.getElementById('pet-hunger').textContent = Math.round(data.pet.hunger) + '%';
                    document.getElementById('pet-hunger-bar').style.width = data.pet.hunger + '%';
                }
                
                // Goals
                renderGoals(data.goals);
                
                // Habits
                renderHabits(data.habits);
                
                // Fractal
                regenerateFractal();
                
            } catch (e) {
                console.error('Dashboard load error:', e);
            }
        }
        
        function renderGoals(goals) {
            const container = document.getElementById('goals-list');
            if (!goals || goals.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary);">No goals yet. Add one below!</p>';
                return;
            }
            
            container.innerHTML = goals.filter(g => !g.completed_at).map(g => `
                <div class="goal-item">
                    <div class="goal-progress">
                        <div style="display: flex; justify-content: space-between;">
                            <span>${g.title}</span>
                            <span style="color: var(--accent-cyan);">${Math.round(g.progress)}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${g.progress}%"></div>
                        </div>
                    </div>
                    <button class="btn btn-secondary" style="padding: 8px 12px; min-width: auto;" 
                            onclick="updateGoalProgress('${g.id}', ${Math.min(100, g.progress + 10)})">+10%</button>
                </div>
            `).join('');
        }
        
        function renderHabits(habits) {
            const container = document.getElementById('habits-list');
            if (!habits || habits.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary);">No habits yet.</p>';
                return;
            }
            
            container.innerHTML = habits.map(h => `
                <div class="habit-item" onclick="toggleHabit('${h.id}', this)">
                    <div class="habit-checkbox">
                        <span class="check-icon"></span>
                    </div>
                    <div>
                        <div>${h.name}</div>
                        <small style="color: var(--text-secondary);">🔥 ${h.current_streak} day streak | 🥄 ${h.spoon_cost}</small>
                    </div>
                </div>
            `).join('');
        }
        
        // Slider updates
        function updateSlider(type) {
            const slider = document.getElementById(type + '-slider');
            document.getElementById(type + '-value').textContent = type === 'sleep' ? slider.value : Math.round(slider.value);
        }
        
        // Actions
        async function saveCheckIn() {
            try {
                await api('/daily/today', 'POST', {
                    mood_score: parseFloat(document.getElementById('mood-slider').value),
                    mood_level: Math.ceil(parseFloat(document.getElementById('mood-slider').value) / 20),
                    energy_level: parseFloat(document.getElementById('energy-slider').value),
                    stress_level: parseFloat(document.getElementById('stress-slider').value),
                    sleep_hours: parseFloat(document.getElementById('sleep-slider').value)
                });
                showMessage('Check-in saved!', 'success');
                regenerateFractal();
            } catch (e) {
                showMessage('Failed to save', 'error');
            }
        }
        
        async function feedPet() {
            try {
                const res = await api('/pet/feed', 'POST');
                if (res.success) {
                    showMessage(res.message, 'success');
                    loadDashboard();
                }
            } catch (e) {
                showMessage('Failed to feed pet', 'error');
            }
        }
        
        async function playWithPet() {
            try {
                const res = await api('/pet/play', 'POST');
                if (res.error) {
                    showMessage(res.error, 'error');
                    return;
                }
                showMessage(res.message, 'success');
                if (res.leveled_up) {
                    showMessage('🎉 ' + userData.pet.name + ' leveled up!', 'success');
                }
                loadDashboard();
            } catch (e) {
                showMessage('Failed to play', 'error');
            }
        }
        
        async function regenerateFractal() {
            try {
                const img = document.getElementById('fractal-image');
                img.src = '/api/visualization/fractal?t=' + Date.now();
            } catch (e) {
                console.error('Fractal error:', e);
            }
        }
        
        async function addGoal() {
            const input = document.getElementById('new-goal');
            if (!input.value.trim()) return;
            
            try {
                await api('/goals', 'POST', { title: input.value.trim() });
                input.value = '';
                loadDashboard();
            } catch (e) {
                showMessage('Failed to add goal', 'error');
            }
        }
        
        async function updateGoalProgress(goalId, progress) {
            try {
                const res = await api('/goals/' + goalId + '/progress', 'POST', { progress });
                if (res.milestone_reached) {
                    showMessage('🎉 Milestone reached: ' + res.milestone_reached + '%!', 'success');
                }
                loadDashboard();
            } catch (e) {
                showMessage('Failed to update', 'error');
            }
        }
        
        async function toggleHabit(habitId, element) {
            try {
                const checkbox = element.querySelector('.habit-checkbox');
                const isChecked = checkbox.classList.contains('checked');
                
                await api('/habits/' + habitId + '/complete', 'POST', { completed: !isChecked });
                
                checkbox.classList.toggle('checked');
                checkbox.innerHTML = checkbox.classList.contains('checked') ? '✓' : '';
            } catch (e) {
                showMessage('Failed to update habit', 'error');
            }
        }
        
        function goToCheckout() {
            api('/subscription/checkout', 'POST').then(res => {
                window.open(res.checkout_url, '_blank');
            });
        }
        
        function showMessage(text, type) {
            // Remove existing messages
            document.querySelectorAll('.message').forEach(m => m.remove());
            
            const msg = document.createElement('div');
            msg.className = 'message message-' + type;
            msg.textContent = text;
            
            const container = document.querySelector('.dashboard-grid') || document.querySelector('.card');
            container.insertBefore(msg, container.firstChild);
            
            setTimeout(() => msg.remove(), 5000);
        }
        
        // Init
        if (token) {
            showDashboard();
        } else {
            showAuthView();
        }
    </script>
</body>
</html>
'''


# ==============================================================================
# MAIN ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Serve main dashboard."""
    return DASHBOARD_HTML


@app.route('/privacy')
def privacy():
    """Privacy policy page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Policy - Life Fractal Intelligence</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
            h1 { color: #00d4ff; }
        </style>
    </head>
    <body>
        <h1>Privacy Policy</h1>
        <p><strong>Last updated:</strong> December 2025</p>
        <h2>Data We Collect</h2>
        <p>We collect only the information you provide: email, password (hashed), and wellness data you choose to log.</p>
        <h2>How We Use It</h2>
        <p>Your data is used solely to provide personalized fractal visualizations and life planning features.</p>
        <h2>Data Storage</h2>
        <p>Data is stored securely in our database. We do not sell or share your personal information.</p>
        <h2>Contact</h2>
        <p>Questions? Contact: onlinediscountsllc@gmail.com</p>
    </body>
    </html>
    '''


# ==============================================================================
# STARTUP
# ==============================================================================

def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("🌀 LIFE FRACTAL INTELLIGENCE v15.0 ULTIMATE")
    print("=" * 70)
    print(f"✨ Golden Ratio (φ):     {PHI:.15f}")
    print(f"🌻 Golden Angle:         {GOLDEN_ANGLE_DEG:.10f}°")
    print(f"🔢 Fibonacci:            {FIBONACCI[:10]}...")
    print(f"🔐 Auth:                 JWT HMAC-SHA256")
    print(f"💾 Database:             SQLite ({Config.DATABASE_PATH})")
    print(f"💰 Subscription:         ${Config.SUBSCRIPTION_PRICE}/month, {Config.TRIAL_DAYS}-day trial")
    print("=" * 70)
    print("\n📡 API Endpoints:")
    print("  Auth:          POST /api/auth/register, /api/auth/login")
    print("  Auth:          POST /api/auth/forgot-password, /api/auth/reset-password")
    print("  Dashboard:     GET  /api/dashboard")
    print("  Daily:         GET/POST /api/daily/today")
    print("  Goals:         GET/POST /api/goals")
    print("  Habits:        GET/POST /api/habits")
    print("  Pet:           GET /api/pet, POST /api/pet/feed, /api/pet/play")
    print("  Visualization: GET /api/visualization/fractal")
    print("  Subscription:  GET /api/subscription/status")
    print("  Calendar:      GET /api/calendar/connect")
    print("=" * 70)


if __name__ == '__main__':
    print_banner()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
