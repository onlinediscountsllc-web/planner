"""
🌀 LIFE FRACTAL INTELLIGENCE - COMPLETE WITH FRONTEND + ChatGPT PRIVACY
═══════════════════════════════════════════════════════════════════════════════
Production-ready app with integrated HTML dashboard - works on Render.com
Includes privacy policy and terms endpoints for ChatGPT Custom GPT integration
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
from functools import wraps

# Flask
from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw

# ML
try:
    from sklearn.tree import DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# GPU (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    torch = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Sacred Math
PHI = (1 + math.sqrt(5)) / 2
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    DATA_DIR = os.environ.get('DATA_DIR', '/tmp/data')
    TRIAL_DAYS = 7
    MAX_GOALS = 50


# Data Models
class PetSpecies(Enum):
    CAT = "cat"
    DRAGON = "dragon"
    PHOENIX = "phoenix"
    OWL = "owl"
    FOX = "fox"


@dataclass
class VirtualPet:
    species: str
    name: str
    level: int = 1
    experience: int = 0
    happiness: int = 100
    hunger: int = 0
    last_fed: Optional[str] = None
    last_played: Optional[str] = None
    
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


@dataclass
class Task:
    id: str
    title: str
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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


@dataclass
class JournalEntry:
    id: str
    content: str
    sentiment_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class User:
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subscription_status: str = "trial"
    trial_start: Optional[str] = None
    pet: Optional[VirtualPet] = None
    goals: List[Goal] = field(default_factory=list)
    habits: List[Habit] = field(default_factory=list)
    journal: List[JournalEntry] = field(default_factory=list)
    total_tasks_completed: int = 0
    total_xp: int = 0
    
    def to_dict(self):
        data = asdict(self)
        data.pop('password_hash', None)
        return data


# Fractal Engine
class SimpleFractalEngine:
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
    
    def generate_from_user_data(self, user: User) -> bytes:
        metrics = self._calculate_metrics(user)
        fractal = self._generate_fractal(metrics)
        colored = self._apply_colors(fractal, metrics)
        img = Image.fromarray(colored)
        
        # Add stats
        draw = ImageDraw.Draw(img)
        text = f"Momentum: {metrics['momentum']:.0%} | XP: {user.total_xp}"
        draw.text((20, 20), text, fill=(255, 255, 255))
        
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _calculate_metrics(self, user: User) -> Dict:
        total_goals = len(user.goals)
        completed_goals = sum(1 for g in user.goals if g.completed)
        goal_rate = completed_goals / total_goals if total_goals > 0 else 0
        
        all_tasks = [t for g in user.goals for t in g.tasks]
        completed_tasks = sum(1 for t in all_tasks if t.completed)
        task_rate = completed_tasks / len(all_tasks) if all_tasks else 0
        
        avg_streak = np.mean([h.streak for h in user.habits]) if user.habits else 0
        max_streak = max([h.best_streak for h in user.habits]) if user.habits else 0
        
        sentiment = np.mean([e.sentiment_score for e in user.journal[-30:]]) if user.journal else 0
        
        pet_happiness = user.pet.happiness if user.pet else 50
        
        momentum = goal_rate * 0.3 + task_rate * 0.2 + (avg_streak/100) * 0.2 + ((sentiment+1)/2) * 0.15 + (pet_happiness/100) * 0.15
        
        return {
            'goal_completion_rate': goal_rate,
            'task_completion_rate': task_rate,
            'avg_streak': avg_streak,
            'max_streak': max_streak,
            'avg_sentiment': sentiment,
            'pet_happiness': pet_happiness,
            'momentum': momentum,
            'total_xp': user.total_xp,
            'total_tasks': len(all_tasks)
        }
    
    def _generate_fractal(self, metrics: Dict) -> np.ndarray:
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
    
    def _apply_colors(self, fractal: np.ndarray, metrics: Dict) -> np.ndarray:
        norm = (fractal - fractal.min()) / (fractal.max() - fractal.min() + 1e-10)
        
        sentiment = metrics['avg_sentiment']
        
        if sentiment > 0.3:
            r = np.uint8(255 * norm**0.8)
            g = np.uint8(200 * norm**1.2)
            b = np.uint8(100 * norm)
        else:
            r = np.uint8(180 * norm**1.1)
            g = np.uint8(100 * norm)
            b = np.uint8(220 * norm**0.95)
        
        intensity = 0.5 + metrics['momentum'] * 0.5
        return np.stack([r*intensity, g*intensity, b*intensity], axis=2).astype(np.uint8)


# Data Manager
class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, 'users.json')
        os.makedirs(data_dir, exist_ok=True)
    
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
            json.dump(data, f)


class AuthManager:
    @staticmethod
    def create_token(email: str) -> str:
        payload = {'email': email, 'exp': (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()}
        return base64.b64encode(json.dumps(payload).encode()).decode()
    
    @staticmethod
    def verify_token(token: str) -> Optional[str]:
        try:
            payload = json.loads(base64.b64decode(token).decode())
            if datetime.fromisoformat(payload['exp']) < datetime.now(timezone.utc):
                return None
            return payload['email']
        except:
            return None


# Flask App
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

data_manager = DataManager(Config.DATA_DIR)
users = data_manager.load_users()
fractal_engine = SimpleFractalEngine()

logger.info(f"🚀 Life Fractal Intelligence - {len(users)} users loaded")


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        email = AuthManager.verify_token(token)
        if not email or email not in users:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(users[email], *args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVACY POLICY & TERMS OF SERVICE ENDPOINTS (FOR CHATGPT)
# ═══════════════════════════════════════════════════════════════════════════════

PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - Fractal Explorer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        h2 { color: #764ba2; margin-top: 30px; }
        .highlight { background: #f0f4ff; padding: 15px; border-left: 4px solid #667eea; margin: 20px 0; }
        ul { padding-left: 20px; }
        li { margin: 10px 0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Fractal Explorer Privacy Policy</h1>
        
        <div class="highlight">
            <strong>Last Updated:</strong> December 2024<br>
            <strong>Effective Date:</strong> December 2024
        </div>

        <h2>🌟 Our Commitment to Your Privacy</h2>
        <p>Fractal Explorer is designed with privacy at its core. We collect minimal data and never sell or share your personal information.</p>

        <h2>📊 What We Collect</h2>
        <ul>
            <li><strong>Session Data:</strong> Temporary conversation context (auto-deleted after 24 hours)</li>
            <li><strong>Goals/Tasks:</strong> Only what you explicitly share for fractal generation</li>
            <li><strong>Pet Preferences:</strong> Your chosen virtual companion species</li>
        </ul>
        
        <div class="highlight">
            <strong>We DO NOT collect:</strong> Names, emails, locations, payment info, or any personally identifiable information through ChatGPT.
        </div>

        <h2>🔒 Data Security</h2>
        <ul>
            <li>All data transmitted via HTTPS encryption</li>
            <li>No persistent storage of personal information</li>
            <li>Session data automatically purged every 24 hours</li>
            <li>No third-party data sharing</li>
        </ul>

        <h2>🎯 How We Use Data</h2>
        <ul>
            <li>Generate personalized fractal artwork</li>
            <li>Track virtual pet interactions during your session</li>
            <li>Improve the user experience</li>
        </ul>

        <h2>👤 Your Rights</h2>
        <ul>
            <li><strong>Access:</strong> Request what data we have (minimal)</li>
            <li><strong>Deletion:</strong> Request data removal anytime</li>
            <li><strong>Portability:</strong> Export your fractal images</li>
        </ul>

        <h2>📧 Contact</h2>
        <p>Questions? Email: <a href="mailto:onlinediscountsllc@gmail.com">onlinediscountsllc@gmail.com</a></p>

        <div class="footer">
            <p>🌀 Fractal Explorer - Where Goals Become Art</p>
            <p>© 2024 Life Fractal Intelligence. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""

TERMS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - Fractal Explorer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        h2 { color: #764ba2; margin-top: 30px; }
        .highlight { background: #f0f4ff; padding: 15px; border-left: 4px solid #667eea; margin: 20px 0; }
        ul { padding-left: 20px; }
        li { margin: 10px 0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📜 Fractal Explorer Terms of Service</h1>
        
        <div class="highlight">
            <strong>Last Updated:</strong> December 2024<br>
            <strong>Effective Date:</strong> December 2024
        </div>

        <h2>🎯 Service Description</h2>
        <p>Fractal Explorer is an AI-powered tool that transforms your goals and intentions into beautiful fractal artwork, featuring virtual pet companions and sacred geometry visualizations.</p>

        <h2>✅ Acceptable Use</h2>
        <ul>
            <li>Create fractal art from your personal goals</li>
            <li>Interact with virtual pet companions</li>
            <li>Share generated artwork (with attribution appreciated)</li>
            <li>Use for personal inspiration and motivation</li>
        </ul>

        <h2>❌ Prohibited Uses</h2>
        <ul>
            <li>Automated bulk requests or API abuse</li>
            <li>Attempting to extract or reverse-engineer the service</li>
            <li>Using the service for illegal purposes</li>
            <li>Impersonating others or providing false information</li>
        </ul>

        <h2>🎨 Intellectual Property</h2>
        <ul>
            <li><strong>Your Content:</strong> You retain rights to goals/text you provide</li>
            <li><strong>Generated Art:</strong> You may use generated fractals freely</li>
            <li><strong>Our Service:</strong> The underlying algorithms remain our property</li>
        </ul>

        <h2>⚠️ Disclaimers</h2>
        <ul>
            <li>Service provided "as is" without warranties</li>
            <li>We may modify or discontinue features</li>
            <li>Not responsible for decisions made based on generated content</li>
        </ul>

        <h2>📧 Contact</h2>
        <p>Questions? Email: <a href="mailto:onlinediscountsllc@gmail.com">onlinediscountsllc@gmail.com</a></p>

        <div class="footer">
            <p>🌀 Fractal Explorer - Where Goals Become Art</p>
            <p>© 2024 Life Fractal Intelligence. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""


@app.route('/privacy')
def privacy_policy():
    """Serve privacy policy as HTML page - Required for ChatGPT Custom GPT"""
    return render_template_string(PRIVACY_HTML)


@app.route('/terms')
def terms_of_service():
    """Serve terms of service as HTML page - Required for ChatGPT Custom GPT"""
    return render_template_string(TERMS_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .auth-box {
            max-width: 400px;
            margin: 100px auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .auth-box h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            border-bottom: 3px solid #eee;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        input, select {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 2px solid #eee;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #5a6fd6; }

        /* Dashboard */
        .dashboard { display: none; }
        .dashboard.active { display: block; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .logo { font-size: 24px; font-weight: bold; color: #667eea; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .stat-bar {
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
        }
        .stat-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s;
        }
        .pet-emoji {
            font-size: 80px;
            text-align: center;
        }
        .fractal-image {
            max-width: 100%;
            border-radius: 10px;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Auth Box -->
        <div class="auth-box" id="authBox">
            <h1>🌀 Life Fractal</h1>
            <div class="tabs">
                <div class="tab active" onclick="showTab('login')">Login</div>
                <div class="tab" onclick="showTab('register')">Register</div>
            </div>

            <div class="tab-content active" id="loginTab">
                <input type="email" id="loginEmail" placeholder="Email" />
                <input type="password" id="loginPassword" placeholder="Password" />
                <button onclick="login()">Login</button>
            </div>

            <div class="tab-content" id="registerTab">
                <input type="email" id="regEmail" placeholder="Email" />
                <input type="password" id="regPassword" placeholder="Password (8+ chars)" />
                <input type="text" id="petName" placeholder="Pet Name" value="Buddy" />
                <select id="petSpecies">
                    <option value="cat">🐱 Cat</option>
                    <option value="dragon">🐉 Dragon</option>
                    <option value="phoenix">🔥 Phoenix</option>
                    <option value="owl">🦉 Owl</option>
                    <option value="fox">🦊 Fox</option>
                </select>
                <button onclick="register()">Create Account</button>
            </div>
        </div>

        <!-- Dashboard -->
        <div class="dashboard" id="dashboard">
            <div class="header">
                <div class="logo">🌀 Life Fractal Intelligence</div>
                <div>
                    <span id="momentum">0%</span> Momentum | 
                    <span id="totalXP">0</span> XP
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>🎨 Your Fractal</h2>
                    <img id="fractalImage" class="fractal-image hidden" />
                    <button onclick="generateFractal()">Generate New Fractal</button>
                </div>

                <div class="card">
                    <h2>🐉 Virtual Pet</h2>
                    <div style="text-align: center;">
                        <div class="pet-emoji" id="petEmoji">🐱</div>
                        <h3 id="petName">Loading...</h3>
                        <p>Level <span id="petLevel">1</span></p>
                        <div style="margin-top: 20px;">
                            <div>Happiness</div>
                            <div class="stat-bar"><div class="stat-fill" id="happinessBar"></div></div>
                        </div>
                        <div style="margin-top: 20px;">
                            <button onclick="feedPet()" style="width: 48%; display: inline-block; margin: 1%;">🍖 Feed</button>
                            <button onclick="playWithPet()" style="width: 48%; display: inline-block; margin: 1%;">🎾 Play</button>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>⚡ Quick Actions</h2>
                    <input type="text" id="newGoalTitle" placeholder="New goal..." />
                    <button onclick="createGoal()">Add Goal</button>
                    <input type="text" id="newHabitTitle" placeholder="New habit..." />
                    <button onclick="createHabit()">Add Habit</button>
                    <button onclick="logout()" style="background: #f44336; margin-top: 15px;">🚪 Logout</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = window.location.origin + '/api';
        let authToken = localStorage.getItem('fractal_token');

        if (authToken) {
            document.getElementById('authBox').style.display = 'none';
            document.getElementById('dashboard').classList.add('active');
            loadDashboard();
        }

        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + 'Tab').classList.add('active');
        }

        async function register() {
            const email = document.getElementById('regEmail').value;
            const password = document.getElementById('regPassword').value;
            const petName = document.getElementById('petName').value;
            const petSpecies = document.getElementById('petSpecies').value;

            const response = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, pet_name: petName, pet_species: petSpecies })
            });

            const data = await response.json();
            if (response.ok) {
                authToken = data.token;
                localStorage.setItem('fractal_token', authToken);
                document.getElementById('authBox').style.display = 'none';
                document.getElementById('dashboard').classList.add('active');
                loadDashboard();
            } else {
                alert(data.error);
            }
        }

        async function login() {
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;

            const response = await fetch(`${API_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();
            if (response.ok) {
                authToken = data.token;
                localStorage.setItem('fractal_token', authToken);
                document.getElementById('authBox').style.display = 'none';
                document.getElementById('dashboard').classList.add('active');
                loadDashboard();
            } else {
                alert(data.error);
            }
        }

        function logout() {
            localStorage.removeItem('fractal_token');
            location.reload();
        }

        async function loadDashboard() {
            await Promise.all([loadMetrics(), loadPet(), generateFractal()]);
        }

        async function loadMetrics() {
            const response = await fetch(`${API_URL}/fractal/metrics`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const data = await response.json();
            document.getElementById('momentum').textContent = Math.round(data.metrics.momentum * 100) + '%';
            document.getElementById('totalXP').textContent = data.metrics.total_xp;
        }

        async function loadPet() {
            const response = await fetch(`${API_URL}/pet`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const data = await response.json();
            const emojiMap = {'cat': '🐱', 'dragon': '🐉', 'phoenix': '🔥', 'owl': '🦉', 'fox': '🦊'};
            document.getElementById('petEmoji').textContent = emojiMap[data.pet.species];
            document.getElementById('petName').textContent = data.pet.name;
            document.getElementById('petLevel').textContent = data.pet.level;
            document.getElementById('happinessBar').style.width = data.pet.happiness + '%';
        }

        async function generateFractal() {
            const response = await fetch(`${API_URL}/fractal/generate`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const img = document.getElementById('fractalImage');
            img.src = url;
            img.classList.remove('hidden');
        }

        async function feedPet() {
            await fetch(`${API_URL}/pet/feed`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            await loadPet();
        }

        async function playWithPet() {
            await fetch(`${API_URL}/pet/play`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            await loadPet();
        }

        async function createGoal() {
            const title = document.getElementById('newGoalTitle').value;
            if (!title) return;
            await fetch(`${API_URL}/goals`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title, description: '', category: 'personal',
                    priority: 'medium', target_date: new Date(Date.now() + 30*24*60*60*1000).toISOString()
                })
            });
            document.getElementById('newGoalTitle').value = '';
            alert('Goal created! 🎯');
            await loadMetrics();
        }

        async function createHabit() {
            const title = document.getElementById('newHabitTitle').value;
            if (!title) return;
            await fetch(`${API_URL}/habits`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title, description: '', frequency: 'daily' })
            });
            document.getElementById('newHabitTitle').value = '';
            alert('Habit created! 📅');
        }
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'users': len(users)})


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or len(password) < 8:
        return jsonify({'error': 'Invalid input'}), 400
    
    if email in users:
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        email=email,
        password_hash=generate_password_hash(password),
        trial_start=datetime.now(timezone.utc).isoformat()
    )
    user.pet = VirtualPet(
        species=data.get('pet_species', 'cat'),
        name=data.get('pet_name', 'Buddy')
    )
    
    users[email] = user
    data_manager.save_users(users)
    
    return jsonify({
        'message': 'Success',
        'token': AuthManager.create_token(email),
        'user': user.to_dict()
    })


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if email not in users or not check_password_hash(users[email].password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return jsonify({
        'message': 'Success',
        'token': AuthManager.create_token(email),
        'user': users[email].to_dict()
    })


@app.route('/api/fractal/generate')
@require_auth
def generate_fractal(user: User):
    img_bytes = fractal_engine.generate_from_user_data(user)
    return send_file(BytesIO(img_bytes), mimetype='image/png')


@app.route('/api/fractal/metrics')
@require_auth
def fractal_metrics(user: User):
    return jsonify({'metrics': fractal_engine._calculate_metrics(user)})


@app.route('/api/pet')
@require_auth
def get_pet(user: User):
    return jsonify({'pet': asdict(user.pet) if user.pet else {}})


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet(user: User):
    if user.pet:
        user.pet.feed()
        data_manager.save_users(users)
    return jsonify({'message': 'Fed!'})


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_pet(user: User):
    if user.pet:
        user.pet.play()
        data_manager.save_users(users)
    return jsonify({'message': 'Played!'})


@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def goals(user: User):
    if request.method == 'GET':
        return jsonify({'goals': [asdict(g) for g in user.goals]})
    
    data = request.json
    goal = Goal(
        id=secrets.token_hex(8),
        title=data['title'],
        description=data.get('description', ''),
        category=data.get('category', 'personal'),
        priority=data.get('priority', 'medium'),
        target_date=data.get('target_date', '')
    )
    user.goals.append(goal)
    data_manager.save_users(users)
    return jsonify({'goal': asdict(goal)})


@app.route('/api/habits', methods=['POST'])
@require_auth
def habits(user: User):
    data = request.json
    habit = Habit(
        id=secrets.token_hex(8),
        title=data['title'],
        description=data.get('description', ''),
        frequency=data.get('frequency', 'daily')
    )
    user.habits.append(habit)
    data_manager.save_users(users)
    return jsonify({'habit': asdict(habit)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
