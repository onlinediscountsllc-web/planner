"""
🌀 LIFE FRACTAL INTELLIGENCE - RENDER.COM PRODUCTION VERSION
═══════════════════════════════════════════════════════════════════════════════════════
Optimized for Render.com deployment with proper port binding and production settings
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
import re

# Flask
from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ML
try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# GPU
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None


# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - RENDER COMPATIBLE
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    """Production configuration for Render"""
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
    STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID', '')
    GOFUNDME_CAMPAIGN_URL = os.environ.get('GOFUNDME_CAMPAIGN_URL', 'https://gofundme.com/life-fractal')
    SUBSCRIPTION_PRICE = 20.00
    TRIAL_DAYS = 7
    JWT_EXPIRY_HOURS = 24
    MAX_GOALS = 50
    MAX_TASKS_PER_GOAL = 100
    # Use /tmp for ephemeral storage on Render
    DATA_DIR = os.environ.get('DATA_DIR', '/tmp/data')
    FRACTAL_CACHE_DIR = os.path.join(DATA_DIR, 'fractals')


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA MODELS (Same as before)
# ═══════════════════════════════════════════════════════════════════════════════════════

class PetSpecies(Enum):
    CAT = "cat"
    DRAGON = "dragon"
    PHOENIX = "phoenix"
    OWL = "owl"
    FOX = "fox"


class GoalPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FractalType(Enum):
    MANDELBROT = "mandelbrot"
    JULIA = "julia"
    BURNING_SHIP = "burning_ship"
    PHOENIX = "phoenix"
    NEWTON = "newton"
    HYBRID = "hybrid"


@dataclass
class VirtualPet:
    species: str
    name: str
    level: int = 1
    experience: int = 0
    happiness: int = 100
    health: int = 100
    hunger: int = 0
    last_fed: Optional[str] = None
    last_played: Optional[str] = None
    unlocked_abilities: List[str] = field(default_factory=list)
    favorite_fractal: str = "mandelbrot"
    
    def gain_experience(self, amount: int):
        self.experience += amount
        while self.experience >= self.level * 100:
            self.experience -= self.level * 100
            self.level += 1
    
    def feed(self):
        self.hunger = max(0, self.hunger - 50)
        self.happiness = min(100, self.happiness + 10)
        self.last_fed = datetime.now(timezone.utc).isoformat()
    
    def play(self):
        self.happiness = min(100, self.happiness + 20)
        self.last_played = datetime.now(timezone.utc).isoformat()
    
    def update_stats(self):
        now = datetime.now(timezone.utc)
        if self.last_fed:
            hours_since_fed = (now - datetime.fromisoformat(self.last_fed)).total_seconds() / 3600
            self.hunger = min(100, int(hours_since_fed * 2))
        if self.last_played:
            hours_since_played = (now - datetime.fromisoformat(self.last_played)).total_seconds() / 3600
            self.happiness = max(0, self.happiness - int(hours_since_played))


@dataclass
class Task:
    id: str
    title: str
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    priority: str = "medium"
    estimated_hours: float = 1.0
    actual_hours: float = 0.0


@dataclass
class Goal:
    id: str
    title: str
    description: str
    category: str
    priority: str
    target_date: str
    progress: float = 0.0
    tasks: List[Task] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed: bool = False
    
    def update_progress(self):
        if not self.tasks:
            return
        completed = sum(1 for t in self.tasks if t.completed)
        self.progress = (completed / len(self.tasks)) * 100


@dataclass
class Habit:
    id: str
    title: str
    description: str
    frequency: str
    streak: int = 0
    best_streak: int = 0
    completions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class JournalEntry:
    id: str
    content: str
    sentiment_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)


@dataclass
class User:
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subscription_status: str = "trial"
    trial_start: Optional[str] = None
    subscription_id: Optional[str] = None
    customer_id: Optional[str] = None
    pet: Optional[VirtualPet] = None
    goals: List[Goal] = field(default_factory=list)
    habits: List[Habit] = field(default_factory=list)
    journal: List[JournalEntry] = field(default_factory=list)
    total_tasks_completed: int = 0
    total_goals_completed: int = 0
    total_xp: int = 0
    theme: str = "cosmic"
    fractal_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        data = asdict(self)
        data.pop('password_hash', None)
        return data


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED FRACTAL ENGINE (CPU-only for Render)
# ═══════════════════════════════════════════════════════════════════════════════════════

class SimpleFractalEngine:
    """Lightweight CPU fractal engine for cloud deployment"""
    
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
    
    def generate_from_user_data(self, user: User, fractal_type: str = "auto") -> bytes:
        metrics = self._calculate_life_metrics(user)
        
        if fractal_type == "auto":
            fractal_type = self._select_fractal_type(user, metrics)
        
        fractal_array = self._generate_fractal(fractal_type, metrics)
        colored = self._apply_color_palette(fractal_array, metrics)
        img = Image.fromarray(colored)
        img = self._add_overlay(img, user, metrics)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _calculate_life_metrics(self, user: User) -> Dict[str, float]:
        total_goals = len(user.goals)
        completed_goals = sum(1 for g in user.goals if g.completed)
        goal_completion_rate = completed_goals / total_goals if total_goals > 0 else 0
        
        all_tasks = [t for g in user.goals for t in g.tasks]
        total_tasks = len(all_tasks)
        completed_tasks = sum(1 for t in all_tasks if t.completed)
        task_completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        avg_streak = np.mean([h.streak for h in user.habits]) if user.habits else 0
        max_streak = max([h.best_streak for h in user.habits]) if user.habits else 0
        
        recent_entries = user.journal[-30:]
        avg_sentiment = np.mean([e.sentiment_score for e in recent_entries]) if recent_entries else 0
        
        pet_happiness = user.pet.happiness if user.pet else 50
        pet_level = user.pet.level if user.pet else 1
        
        momentum = (goal_completion_rate * 0.3 + task_completion_rate * 0.2 +
                   (avg_streak / 100) * 0.2 + ((avg_sentiment + 1) / 2) * 0.15 +
                   (pet_happiness / 100) * 0.15)
        
        return {
            'goal_completion_rate': goal_completion_rate,
            'task_completion_rate': task_completion_rate,
            'avg_streak': avg_streak,
            'max_streak': max_streak,
            'avg_sentiment': avg_sentiment,
            'pet_happiness': pet_happiness,
            'pet_level': pet_level,
            'momentum': momentum,
            'total_xp': user.total_xp,
            'total_tasks': total_tasks
        }
    
    def _select_fractal_type(self, user: User, metrics: Dict) -> str:
        if user.pet and hasattr(user.pet, 'favorite_fractal'):
            return user.pet.favorite_fractal
        if metrics['momentum'] > 0.8:
            return "phoenix"
        elif metrics['avg_sentiment'] > 0.5:
            return "julia"
        else:
            return "mandelbrot"
    
    def _generate_fractal(self, ftype: str, metrics: Dict) -> np.ndarray:
        max_iter = int(128 + metrics['momentum'] * 128)
        zoom = 1.0 + metrics['goal_completion_rate'] * 20
        
        x = np.linspace(-2.5/zoom, 2.5/zoom, self.width)
        y = np.linspace(-2.5/zoom, 2.5/zoom, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 4
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i
        
        return M.astype(np.float32)
    
    def _apply_color_palette(self, fractal: np.ndarray, metrics: Dict) -> np.ndarray:
        fractal_norm = (fractal - fractal.min()) / (fractal.max() - fractal.min() + 1e-10)
        
        sentiment = metrics['avg_sentiment']
        
        if sentiment > 0.3:
            r = np.uint8(255 * np.power(fractal_norm, 0.8))
            g = np.uint8(200 * np.power(fractal_norm, 1.2))
            b = np.uint8(100 * fractal_norm)
        elif sentiment < -0.3:
            r = np.uint8(100 * fractal_norm)
            g = np.uint8(150 * fractal_norm)
            b = np.uint8(255 * np.power(fractal_norm, 0.9))
        else:
            r = np.uint8(180 * np.power(fractal_norm, 1.1))
            g = np.uint8(100 * fractal_norm)
            b = np.uint8(220 * np.power(fractal_norm, 0.95))
        
        intensity = 0.5 + metrics['momentum'] * 0.5
        r = np.uint8(r * intensity)
        g = np.uint8(g * intensity)
        b = np.uint8(b * intensity)
        
        return np.stack([r, g, b], axis=2)
    
    def _add_overlay(self, img: Image.Image, user: User, metrics: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        
        # Simple text overlay
        text = f"Momentum: {metrics['momentum']:.0%} | XP: {user.total_xp}"
        draw.text((20, 20), text, fill=(255, 255, 255))
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class DataManager:
    def __init__(self, data_dir: str = '/tmp/data'):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, 'users.json')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(Config.FRACTAL_CACHE_DIR, exist_ok=True)
    
    def load_users(self) -> Dict[str, User]:
        if not os.path.exists(self.users_file):
            return {}
        
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
        except:
            return {}
        
        users = {}
        for email, user_data in data.items():
            user = User(
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                created_at=user_data.get('created_at', datetime.now(timezone.utc).isoformat()),
                subscription_status=user_data.get('subscription_status', 'trial'),
                trial_start=user_data.get('trial_start'),
                total_tasks_completed=user_data.get('total_tasks_completed', 0),
                total_goals_completed=user_data.get('total_goals_completed', 0),
                total_xp=user_data.get('total_xp', 0)
            )
            
            if 'pet' in user_data and user_data['pet']:
                user.pet = VirtualPet(**user_data['pet'])
            
            for goal_data in user_data.get('goals', []):
                tasks = [Task(**t) for t in goal_data.get('tasks', [])]
                goal = Goal(**{k: v for k, v in goal_data.items() if k != 'tasks'})
                goal.tasks = tasks
                user.goals.append(goal)
            
            for habit_data in user_data.get('habits', []):
                user.habits.append(Habit(**habit_data))
            
            for entry_data in user_data.get('journal', []):
                user.journal.append(JournalEntry(**entry_data))
            
            users[email] = user
        
        return users
    
    def save_users(self, users: Dict[str, User]):
        data = {email: user.to_dict() for email, user in users.items()}
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class AuthManager:
    @staticmethod
    def create_token(email: str) -> str:
        payload = {
            'email': email,
            'exp': datetime.now(timezone.utc) + timedelta(hours=Config.JWT_EXPIRY_HOURS)
        }
        token_data = json.dumps(payload)
        return base64.b64encode(token_data.encode()).decode()
    
    @staticmethod
    def verify_token(token: str) -> Optional[str]:
        try:
            token_data = base64.b64decode(token).decode()
            payload = json.loads(token_data)
            exp = datetime.fromisoformat(payload['exp'])
            if exp < datetime.now(timezone.utc):
                return None
            return payload['email']
        except:
            return None


# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

data_manager = DataManager(Config.DATA_DIR)
users = data_manager.load_users()
fractal_engine = SimpleFractalEngine()

logger.info(f"🚀 Life Fractal Intelligence starting on Render...")
logger.info(f"📊 Loaded {len(users)} users")


def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        email = AuthManager.verify_token(token)
        if not email or email not in users:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(users[email], *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return jsonify({
        'app': 'Life Fractal Intelligence',
        'status': 'live',
        'version': '1.0.0',
        'users': len(users),
        'endpoints': {
            'register': 'POST /api/register',
            'login': 'POST /api/login',
            'fractal': 'GET /api/fractal/generate',
            'dashboard': 'GET /api/dashboard',
            'health': 'GET /health'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'users': len(users),
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    pet_species = data.get('pet_species', 'cat')
    pet_name = data.get('pet_name', 'Buddy')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    if email in users:
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        email=email,
        password_hash=generate_password_hash(password),
        trial_start=datetime.now(timezone.utc).isoformat()
    )
    
    user.pet = VirtualPet(species=pet_species, name=pet_name)
    
    users[email] = user
    data_manager.save_users(users)
    
    token = AuthManager.create_token(email)
    
    return jsonify({
        'message': 'Registration successful',
        'token': token,
        'user': user.to_dict()
    })


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if email not in users:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    user = users[email]
    
    if not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if user.pet:
        user.pet.update_stats()
        data_manager.save_users(users)
    
    token = AuthManager.create_token(email)
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    })


@app.route('/api/fractal/generate', methods=['GET'])
@require_auth
def generate_fractal(user: User):
    fractal_type = request.args.get('type', 'auto')
    
    try:
        img_bytes = fractal_engine.generate_from_user_data(user, fractal_type)
        return send_file(BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({'error': 'Fractal generation failed'}), 500


@app.route('/api/fractal/metrics', methods=['GET'])
@require_auth
def get_fractal_metrics(user: User):
    metrics = fractal_engine._calculate_life_metrics(user)
    return jsonify({
        'metrics': metrics,
        'fractal_type': fractal_engine._select_fractal_type(user, metrics)
    })


@app.route('/api/dashboard', methods=['GET'])
@require_auth
def dashboard(user: User):
    metrics = fractal_engine._calculate_life_metrics(user)
    return jsonify({
        'user': user.to_dict(),
        'metrics': metrics,
        'recent_goals': [asdict(g) for g in user.goals[-5:]],
        'recent_habits': [asdict(h) for h in user.habits[-5:]],
        'recent_journal': [asdict(e) for e in user.journal[-5:]]
    })


# Additional endpoints abbreviated for brevity - add goals, tasks, habits, etc.

if __name__ == '__main__':
    # For local testing only - Gunicorn handles this in production
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
