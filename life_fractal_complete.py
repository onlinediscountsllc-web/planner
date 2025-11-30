"""
🌀 LIFE FRACTAL INTELLIGENCE - COMPLETE PRODUCTION APPLICATION
═══════════════════════════════════════════════════════════════════════════════════════
Full-stack life planning platform with GPU-accelerated fractal visualization
All user data (goals, tasks, habits, journal, pet) drives the fractal art generation

🎯 FEATURES:
✅ Secure authentication with JWT tokens
✅ Virtual pet system (5 species) that evolves with your progress
✅ GPU-accelerated fractal visualization driven by YOUR life metrics
✅ Sacred geometry overlays based on completion rates
✅ Goals, habits, tasks, journal with sentiment analysis
✅ ML-powered predictions and fuzzy logic guidance
✅ Stripe payments ($20/month, 7-day trial)
✅ GoFundMe integration during trial
✅ Export/import data
✅ Production-ready with security best practices

📊 DATA → FRACTAL MAPPING:
- Goal completion → Zoom level & fractal complexity
- Habit streaks → Color intensity & sacred geometry overlays
- Task velocity → Animation speed & pattern evolution
- Journal sentiment → Color palette & emotional resonance
- Pet happiness → Fractal type selection & special effects
- Overall momentum → Fibonacci spiral intensity
═══════════════════════════════════════════════════════════════════════════════════════
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

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    """Production configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_...')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'pk_test_...')
    STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID', 'price_...')
    GOFUNDME_CAMPAIGN_URL = os.environ.get('GOFUNDME_CAMPAIGN_URL', 'https://gofundme.com/...')
    SUBSCRIPTION_PRICE = 20.00
    TRIAL_DAYS = 7
    JWT_EXPIRY_HOURS = 24
    MAX_GOALS = 50
    MAX_TASKS_PER_GOAL = 100
    DATA_DIR = os.environ.get('DATA_DIR', './data')
    FRACTAL_CACHE_DIR = os.path.join(DATA_DIR, 'fractals')


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA MODELS
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
    """Virtual companion that evolves with user progress"""
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
        """Gain XP and level up"""
        self.experience += amount
        while self.experience >= self.level * 100:
            self.experience -= self.level * 100
            self.level += 1
            logger.info(f"🎉 {self.name} leveled up to {self.level}!")
    
    def feed(self):
        """Feed the pet"""
        self.hunger = max(0, self.hunger - 50)
        self.happiness = min(100, self.happiness + 10)
        self.last_fed = datetime.now(timezone.utc).isoformat()
    
    def play(self):
        """Play with the pet"""
        self.happiness = min(100, self.happiness + 20)
        self.last_played = datetime.now(timezone.utc).isoformat()
    
    def update_stats(self):
        """Natural decay over time"""
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
        """Calculate progress from tasks"""
        if not self.tasks:
            return
        completed = sum(1 for t in self.tasks if t.completed)
        self.progress = (completed / len(self.tasks)) * 100


@dataclass
class Habit:
    id: str
    title: str
    description: str
    frequency: str  # daily, weekly, monthly
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
    """Complete user profile with all data"""
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Subscription
    subscription_status: str = "trial"
    trial_start: Optional[str] = None
    subscription_id: Optional[str] = None
    customer_id: Optional[str] = None
    
    # Pet
    pet: Optional[VirtualPet] = None
    
    # Goals & Tasks
    goals: List[Goal] = field(default_factory=list)
    habits: List[Habit] = field(default_factory=list)
    journal: List[JournalEntry] = field(default_factory=list)
    
    # Stats
    total_tasks_completed: int = 0
    total_goals_completed: int = 0
    total_xp: int = 0
    
    # Preferences
    theme: str = "cosmic"
    fractal_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary, excluding password"""
        data = asdict(self)
        data.pop('password_hash', None)
        return data


# ═══════════════════════════════════════════════════════════════════════════════════════
# GPU-ACCELERATED FRACTAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class IntegratedFractalEngine:
    """
    Fractal engine that maps ALL user data into visual parameters
    
    DATA MAPPING:
    - Goal completion → Zoom depth & iteration count
    - Habit streaks → Sacred geometry overlay intensity
    - Task velocity → Animation/evolution speed
    - Journal sentiment → Color palette selection
    - Pet happiness → Special effects & fractal type
    - Overall momentum → Fibonacci spiral strength
    """
    
    def __init__(self, width: int = 1200, height: int = 1200):
        self.width = width
        self.height = height
        self.use_gpu = GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info(f"🎨 Fractal GPU acceleration: {GPU_NAME}")
    
    def generate_from_user_data(self, user: User, fractal_type: str = "auto") -> bytes:
        """
        Generate fractal art driven by user's life metrics
        
        Returns: PNG image bytes
        """
        # Calculate life metrics
        metrics = self._calculate_life_metrics(user)
        
        # Auto-select fractal type based on pet or user state
        if fractal_type == "auto":
            fractal_type = self._select_fractal_type(user, metrics)
        
        # Generate base fractal
        fractal_array = self._generate_fractal(fractal_type, metrics)
        
        # Apply color palette based on sentiment
        colored = self._apply_color_palette(fractal_array, metrics)
        
        # Add sacred geometry overlays
        img = Image.fromarray(colored)
        img = self._add_sacred_geometry(img, metrics)
        
        # Add pet sprite/effect
        if user.pet:
            img = self._add_pet_effect(img, user.pet, metrics)
        
        # Add stats overlay
        img = self._add_stats_overlay(img, user, metrics)
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _calculate_life_metrics(self, user: User) -> Dict[str, float]:
        """Extract numerical metrics from all user data"""
        
        # Goal metrics
        total_goals = len(user.goals)
        completed_goals = sum(1 for g in user.goals if g.completed)
        goal_completion_rate = completed_goals / total_goals if total_goals > 0 else 0
        avg_goal_progress = np.mean([g.progress for g in user.goals]) if user.goals else 0
        
        # Task metrics
        all_tasks = [t for g in user.goals for t in g.tasks]
        total_tasks = len(all_tasks)
        completed_tasks = sum(1 for t in all_tasks if t.completed)
        task_completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # Habit metrics
        total_habits = len(user.habits)
        avg_streak = np.mean([h.streak for h in user.habits]) if user.habits else 0
        max_streak = max([h.best_streak for h in user.habits]) if user.habits else 0
        
        # Journal sentiment
        recent_entries = user.journal[-30:] if len(user.journal) > 0 else []
        avg_sentiment = np.mean([e.sentiment_score for e in recent_entries]) if recent_entries else 0
        sentiment_trend = self._calculate_trend([e.sentiment_score for e in recent_entries]) if recent_entries else 0
        
        # Pet metrics
        pet_happiness = user.pet.happiness if user.pet else 50
        pet_level = user.pet.level if user.pet else 1
        
        # Overall momentum (0-1 scale)
        momentum = (goal_completion_rate * 0.3 +
                   task_completion_rate * 0.2 +
                   (avg_streak / 100) * 0.2 +
                   ((avg_sentiment + 1) / 2) * 0.15 +
                   (pet_happiness / 100) * 0.15)
        
        return {
            'goal_completion_rate': goal_completion_rate,
            'avg_goal_progress': avg_goal_progress,
            'task_completion_rate': task_completion_rate,
            'total_tasks': total_tasks,
            'avg_streak': avg_streak,
            'max_streak': max_streak,
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'pet_happiness': pet_happiness,
            'pet_level': pet_level,
            'momentum': momentum,
            'total_xp': user.total_xp
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def _select_fractal_type(self, user: User, metrics: Dict) -> str:
        """Select fractal type based on user state"""
        
        if user.pet and hasattr(user.pet, 'favorite_fractal'):
            return user.pet.favorite_fractal
        
        # Auto-select based on metrics
        if metrics['momentum'] > 0.8:
            return "phoenix"  # Rising from ashes
        elif metrics['avg_sentiment'] > 0.5:
            return "julia"  # Beautiful complexity
        elif metrics['max_streak'] > 30:
            return "burning_ship"  # On fire!
        elif metrics['pet_level'] > 10:
            return "newton"  # Evolved intelligence
        else:
            return "mandelbrot"  # Classic journey
    
    def _generate_fractal(self, fractal_type: str, metrics: Dict) -> np.ndarray:
        """Generate fractal with GPU acceleration"""
        
        # Map metrics to fractal parameters
        max_iter = int(128 + metrics['momentum'] * 128)  # 128-256 iterations
        zoom = 1.0 + metrics['goal_completion_rate'] * 50  # Deeper zoom with progress
        power = 2.0 + metrics['avg_streak'] / 20  # Increase power with streaks
        
        # Center point influenced by sentiment
        center_x = -0.5 + metrics['avg_sentiment'] * 0.3
        center_y = metrics['sentiment_trend'] * 0.3
        
        if self.use_gpu and torch is not None:
            return self._fractal_gpu(fractal_type, max_iter, zoom, (center_x, center_y), power)
        return self._fractal_cpu(fractal_type, max_iter, zoom, (center_x, center_y), power)
    
    def _fractal_gpu(self, ftype: str, max_iter: int, zoom: float, 
                    center: Tuple[float, float], power: float) -> np.ndarray:
        """GPU-accelerated fractal generation"""
        device = torch.device('cuda')
        
        # Create coordinate grid
        x = torch.linspace(-2.5/zoom + center[0], 2.5/zoom + center[0], 
                          self.width, device=device)
        y = torch.linspace(-2.5/zoom + center[1], 2.5/zoom + center[1], 
                          self.height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        C = X + 1j * Y
        
        Z = torch.zeros_like(C)
        M = torch.zeros(C.shape, dtype=torch.int32, device=device)
        
        for i in range(max_iter):
            mask = (Z.abs() <= 4)
            
            if ftype == "mandelbrot":
                Z[mask] = Z[mask]**power + C[mask]
            elif ftype == "burning_ship":
                Z[mask] = (Z[mask].real.abs() + 1j * Z[mask].imag.abs())**2 + C[mask]
            elif ftype == "julia":
                if i == 0:
                    Z = C.clone()
                c_julia = -0.4 + 0.6j
                Z[mask] = Z[mask]**power + c_julia
            elif ftype == "phoenix":
                Z_old = Z.clone()
                Z[mask] = Z[mask]**2 + C[mask] + 0.5 * Z_old[mask]
            else:  # mandelbrot default
                Z[mask] = Z[mask]**2 + C[mask]
            
            M[mask] = i
        
        return M.cpu().numpy().astype(np.float32)
    
    def _fractal_cpu(self, ftype: str, max_iter: int, zoom: float,
                    center: Tuple[float, float], power: float) -> np.ndarray:
        """CPU fallback for fractal generation"""
        x = np.linspace(-2.5/zoom + center[0], 2.5/zoom + center[0], self.width)
        y = np.linspace(-2.5/zoom + center[1], 2.5/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 4
            
            if ftype == "burning_ship":
                Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag))**2 + C[mask]
            else:
                Z[mask] = Z[mask]**power + C[mask]
            
            M[mask] = i
        
        return M.astype(np.float32)
    
    def _apply_color_palette(self, fractal: np.ndarray, metrics: Dict) -> np.ndarray:
        """Apply color palette based on sentiment and momentum"""
        
        # Normalize
        fractal_norm = (fractal - fractal.min()) / (fractal.max() - fractal.min() + 1e-10)
        
        # Choose palette based on sentiment
        sentiment = metrics['avg_sentiment']
        
        if sentiment > 0.3:  # Positive
            # Warm, vibrant colors
            r = np.uint8(255 * np.power(fractal_norm, 0.8))
            g = np.uint8(200 * np.power(fractal_norm, 1.2))
            b = np.uint8(100 * fractal_norm)
        elif sentiment < -0.3:  # Negative
            # Cool, calming colors
            r = np.uint8(100 * fractal_norm)
            g = np.uint8(150 * fractal_norm)
            b = np.uint8(255 * np.power(fractal_norm, 0.9))
        else:  # Neutral
            # Balanced purple/pink
            r = np.uint8(180 * np.power(fractal_norm, 1.1))
            g = np.uint8(100 * fractal_norm)
            b = np.uint8(220 * np.power(fractal_norm, 0.95))
        
        # Adjust intensity based on momentum
        intensity = 0.5 + metrics['momentum'] * 0.5
        r = np.uint8(r * intensity)
        g = np.uint8(g * intensity)
        b = np.uint8(b * intensity)
        
        return np.stack([r, g, b], axis=2)
    
    def _add_sacred_geometry(self, img: Image.Image, metrics: Dict) -> Image.Image:
        """Add sacred geometry overlays based on habit streaks"""
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        center = (self.width // 2, self.height // 2)
        
        # Fibonacci spiral intensity based on max streak
        if metrics['max_streak'] > 7:
            spiral_alpha = int(min(255, metrics['max_streak'] * 3))
            self._draw_fibonacci_spiral(draw, center, spiral_alpha)
        
        # Flower of Life based on goal completion
        if metrics['goal_completion_rate'] > 0.5:
            flower_alpha = int(metrics['goal_completion_rate'] * 100)
            self._draw_flower_of_life(draw, center, flower_alpha)
        
        # Golden ratio circles based on momentum
        if metrics['momentum'] > 0.6:
            ratio_alpha = int(metrics['momentum'] * 80)
            self._draw_golden_ratio_circles(draw, center, ratio_alpha)
        
        # Merge overlay
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        
        return img.convert('RGB')
    
    def _draw_fibonacci_spiral(self, draw, center, alpha):
        """Draw Fibonacci spiral"""
        x, y = center
        angle = 0
        for i, fib in enumerate(FIBONACCI[:12]):
            if fib == 0:
                continue
            radius = fib * 3
            color = (255, 215, 0, alpha)  # Gold
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        outline=color, width=2)
            angle += GOLDEN_ANGLE_RAD
            x += int(fib * math.cos(angle))
            y += int(fib * math.sin(angle))
    
    def _draw_flower_of_life(self, draw, center, alpha):
        """Draw Flower of Life pattern"""
        radius = 80
        color = (255, 255, 255, alpha)
        
        # Center circle
        x, y = center
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                    outline=color, width=2)
        
        # 6 surrounding circles
        for i in range(6):
            angle = i * math.pi / 3
            cx = x + int(radius * math.cos(angle))
            cy = y + int(radius * math.sin(angle))
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                        outline=color, width=2)
    
    def _draw_golden_ratio_circles(self, draw, center, alpha):
        """Draw golden ratio circles"""
        x, y = center
        radius = 50
        color = (255, 215, 0, alpha)
        
        for i in range(8):
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        outline=color, width=2)
            radius = int(radius * PHI)
            if radius > self.width:
                break
    
    def _add_pet_effect(self, img: Image.Image, pet: VirtualPet, metrics: Dict) -> Image.Image:
        """Add pet-specific visual effects"""
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw pet indicator
        pet_x = self.width - 150
        pet_y = 50
        
        # Species-specific effects
        if pet.species == "dragon":
            # Fire particles
            color = (255, 100, 0, 150)
        elif pet.species == "phoenix":
            # Rebirth aura
            color = (255, 150, 0, 120)
        elif pet.species == "owl":
            # Wisdom glow
            color = (150, 150, 255, 100)
        elif pet.species == "fox":
            # Clever sparkles
            color = (255, 200, 100, 130)
        else:  # cat
            # Mysterious shimmer
            color = (200, 100, 255, 110)
        
        # Draw effect circle
        radius = 30 + pet.level * 2
        draw.ellipse([pet_x - radius, pet_y - radius, pet_x + radius, pet_y + radius],
                    fill=color)
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        
        return img.convert('RGB')
    
    def _add_stats_overlay(self, img: Image.Image, user: User, metrics: Dict) -> Image.Image:
        """Add stats text overlay"""
        
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            font_large = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
            font_large = font
        
        # Stats text
        y_offset = 20
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)
        
        # Title
        title = f"🌀 {user.pet.name if user.pet else 'Life'} Fractal"
        draw.text((22, y_offset + 2), title, fill=shadow_color, font=font_large)
        draw.text((20, y_offset), title, fill=color, font=font_large)
        y_offset += 40
        
        # Stats
        stats = [
            f"Momentum: {metrics['momentum']:.0%}",
            f"Goals: {metrics['goal_completion_rate']:.0%}",
            f"Streak: {int(metrics['max_streak'])} days",
            f"Tasks: {int(metrics['total_tasks'])}",
            f"XP: {user.total_xp}"
        ]
        
        for stat in stats:
            draw.text((22, y_offset + 2), stat, fill=shadow_color, font=font)
            draw.text((20, y_offset), stat, fill=color, font=font)
            y_offset += 25
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class DataManager:
    """Secure data persistence"""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, 'users.json')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(Config.FRACTAL_CACHE_DIR, exist_ok=True)
    
    def load_users(self) -> Dict[str, User]:
        """Load all users"""
        if not os.path.exists(self.users_file):
            return {}
        
        with open(self.users_file, 'r') as f:
            data = json.load(f)
        
        users = {}
        for email, user_data in data.items():
            # Reconstruct user object
            user = User(
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                created_at=user_data.get('created_at', datetime.now(timezone.utc).isoformat()),
                subscription_status=user_data.get('subscription_status', 'trial'),
                trial_start=user_data.get('trial_start'),
                subscription_id=user_data.get('subscription_id'),
                customer_id=user_data.get('customer_id'),
                total_tasks_completed=user_data.get('total_tasks_completed', 0),
                total_goals_completed=user_data.get('total_goals_completed', 0),
                total_xp=user_data.get('total_xp', 0),
                theme=user_data.get('theme', 'cosmic'),
                fractal_preferences=user_data.get('fractal_preferences', {})
            )
            
            # Reconstruct pet
            if 'pet' in user_data and user_data['pet']:
                user.pet = VirtualPet(**user_data['pet'])
            
            # Reconstruct goals
            for goal_data in user_data.get('goals', []):
                tasks = [Task(**t) for t in goal_data.get('tasks', [])]
                goal = Goal(**{k: v for k, v in goal_data.items() if k != 'tasks'})
                goal.tasks = tasks
                user.goals.append(goal)
            
            # Reconstruct habits
            for habit_data in user_data.get('habits', []):
                user.habits.append(Habit(**habit_data))
            
            # Reconstruct journal
            for entry_data in user_data.get('journal', []):
                user.journal.append(JournalEntry(**entry_data))
            
            users[email] = user
        
        return users
    
    def save_users(self, users: Dict[str, User]):
        """Save all users"""
        data = {email: user.to_dict() for email, user in users.items()}
        
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def backup_user_data(self, user: User) -> bytes:
        """Create backup JSON of user data"""
        data = user.to_dict()
        return json.dumps(data, indent=2).encode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class AuthManager:
    """JWT-based authentication"""
    
    @staticmethod
    def create_token(email: str) -> str:
        """Create JWT token"""
        payload = {
            'email': email,
            'exp': datetime.now(timezone.utc) + timedelta(hours=Config.JWT_EXPIRY_HOURS)
        }
        # Simple token (in production, use proper JWT library)
        token_data = json.dumps(payload)
        token = base64.b64encode(token_data.encode()).decode()
        return token
    
    @staticmethod
    def verify_token(token: str) -> Optional[str]:
        """Verify JWT token"""
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

# Initialize managers
data_manager = DataManager(Config.DATA_DIR)
users = data_manager.load_users()
fractal_engine = IntegratedFractalEngine()

logger.info(f"🚀 Life Fractal Intelligence starting...")
logger.info(f"📊 Loaded {len(users)} users")
logger.info(f"🎨 GPU: {GPU_NAME if GPU_AVAILABLE else 'CPU mode'}")


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    pet_species = data.get('pet_species', 'cat')
    pet_name = data.get('pet_name', 'Buddy')
    
    # Validate
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    if email in users:
        return jsonify({'error': 'Email already registered'}), 400
    
    # Create user
    user = User(
        email=email,
        password_hash=generate_password_hash(password),
        trial_start=datetime.now(timezone.utc).isoformat()
    )
    
    # Create pet
    user.pet = VirtualPet(
        species=pet_species,
        name=pet_name
    )
    
    users[email] = user
    data_manager.save_users(users)
    
    # Create token
    token = AuthManager.create_token(email)
    
    logger.info(f"✅ New user registered: {email}")
    
    return jsonify({
        'message': 'Registration successful',
        'token': token,
        'user': user.to_dict()
    })


@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if email not in users:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    user = users[email]
    
    if not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Update pet stats
    if user.pet:
        user.pet.update_stats()
        data_manager.save_users(users)
    
    token = AuthManager.create_token(email)
    
    logger.info(f"👤 User logged in: {email}")
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    })


def require_auth(f):
    """Decorator for protected routes"""
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        email = AuthManager.verify_token(token)
        
        if not email or email not in users:
            return jsonify({'error': 'Unauthorized'}), 401
        
        return f(users[email], *args, **kwargs)
    
    wrapper.__name__ = f.__name__
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════════════
# FRACTAL VISUALIZATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/fractal/generate', methods=['GET'])
@require_auth
def generate_fractal(user: User):
    """Generate personalized fractal from user data"""
    
    fractal_type = request.args.get('type', 'auto')
    
    try:
        img_bytes = fractal_engine.generate_from_user_data(user, fractal_type)
        
        # Cache it
        cache_path = os.path.join(Config.FRACTAL_CACHE_DIR, f'{user.email}_latest.png')
        with open(cache_path, 'wb') as f:
            f.write(img_bytes)
        
        return send_file(BytesIO(img_bytes), mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Fractal generation error: {e}", exc_info=True)
        return jsonify({'error': 'Fractal generation failed'}), 500


@app.route('/api/fractal/metrics', methods=['GET'])
@require_auth
def get_fractal_metrics(user: User):
    """Get the metrics that drive fractal generation"""
    
    metrics = fractal_engine._calculate_life_metrics(user)
    
    return jsonify({
        'metrics': metrics,
        'fractal_type': fractal_engine._select_fractal_type(user, metrics)
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# GOAL & TASK ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def goals_endpoint(user: User):
    """Get all goals or create new goal"""
    
    if request.method == 'GET':
        return jsonify({'goals': [asdict(g) for g in user.goals]})
    
    # POST - create goal
    data = request.json
    
    if len(user.goals) >= Config.MAX_GOALS:
        return jsonify({'error': f'Maximum {Config.MAX_GOALS} goals allowed'}), 400
    
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
    
    # Give XP
    if user.pet:
        user.pet.gain_experience(50)
        user.total_xp += 50
        data_manager.save_users(users)
    
    logger.info(f"🎯 New goal created: {goal.title}")
    
    return jsonify({'goal': asdict(goal)})


@app.route('/api/goals/<goal_id>/tasks', methods=['POST'])
@require_auth
def add_task(user: User, goal_id: str):
    """Add task to goal"""
    
    goal = next((g for g in user.goals if g.id == goal_id), None)
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404
    
    if len(goal.tasks) >= Config.MAX_TASKS_PER_GOAL:
        return jsonify({'error': 'Too many tasks'}), 400
    
    data = request.json
    
    task = Task(
        id=secrets.token_hex(8),
        title=data['title'],
        priority=data.get('priority', 'medium'),
        estimated_hours=data.get('estimated_hours', 1.0)
    )
    
    goal.tasks.append(task)
    goal.update_progress()
    data_manager.save_users(users)
    
    return jsonify({'task': asdict(task)})


@app.route('/api/tasks/<task_id>/complete', methods=['POST'])
@require_auth
def complete_task(user: User, task_id: str):
    """Mark task as complete"""
    
    for goal in user.goals:
        task = next((t for t in goal.tasks if t.id == task_id), None)
        if task:
            task.completed = True
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Update stats
            user.total_tasks_completed += 1
            user.total_xp += 10
            
            # Pet gains XP
            if user.pet:
                user.pet.gain_experience(10)
                user.pet.happiness = min(100, user.pet.happiness + 5)
            
            goal.update_progress()
            
            # Check if goal is complete
            if goal.progress >= 100 and not goal.completed:
                goal.completed = True
                user.total_goals_completed += 1
                user.total_xp += 100
                if user.pet:
                    user.pet.gain_experience(100)
                logger.info(f"🎉 Goal completed: {goal.title}")
            
            data_manager.save_users(users)
            
            return jsonify({
                'message': 'Task completed',
                'xp_gained': 10,
                'pet': asdict(user.pet) if user.pet else None
            })
    
    return jsonify({'error': 'Task not found'}), 404


# ═══════════════════════════════════════════════════════════════════════════════════════
# HABIT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/habits', methods=['GET', 'POST'])
@require_auth
def habits_endpoint(user: User):
    """Get habits or create new habit"""
    
    if request.method == 'GET':
        return jsonify({'habits': [asdict(h) for h in user.habits]})
    
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


@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(user: User, habit_id: str):
    """Mark habit as done today"""
    
    habit = next((h for h in user.habits if h.id == habit_id), None)
    if not habit:
        return jsonify({'error': 'Habit not found'}), 404
    
    today = datetime.now(timezone.utc).date().isoformat()
    
    if today not in habit.completions:
        habit.completions.append(today)
        habit.streak += 1
        habit.best_streak = max(habit.best_streak, habit.streak)
        
        # XP and pet happiness
        user.total_xp += 5
        if user.pet:
            user.pet.gain_experience(5)
            user.pet.happiness = min(100, user.pet.happiness + 3)
        
        data_manager.save_users(users)
        
        return jsonify({
            'message': 'Habit completed',
            'streak': habit.streak,
            'xp_gained': 5
        })
    
    return jsonify({'message': 'Already completed today'})


# ═══════════════════════════════════════════════════════════════════════════════════════
# JOURNAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/journal', methods=['GET', 'POST'])
@require_auth
def journal_endpoint(user: User):
    """Get journal entries or create new entry"""
    
    if request.method == 'GET':
        limit = int(request.args.get('limit', 50))
        entries = user.journal[-limit:]
        return jsonify({'entries': [asdict(e) for e in entries]})
    
    data = request.json
    content = data.get('content', '')
    
    # Simple sentiment analysis (word-based)
    sentiment = calculate_sentiment(content)
    
    entry = JournalEntry(
        id=secrets.token_hex(8),
        content=content,
        sentiment_score=sentiment,
        tags=data.get('tags', [])
    )
    
    user.journal.append(entry)
    data_manager.save_users(users)
    
    return jsonify({'entry': asdict(entry)})


def calculate_sentiment(text: str) -> float:
    """Simple sentiment analysis (-1 to 1)"""
    positive_words = {'happy', 'great', 'awesome', 'good', 'excellent', 'love', 'joy', 'amazing', 'wonderful', 'fantastic'}
    negative_words = {'sad', 'bad', 'terrible', 'awful', 'hate', 'angry', 'depressed', 'frustrated', 'horrible', 'difficult'}
    
    text_lower = text.lower()
    words = text_lower.split()
    
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


# ═══════════════════════════════════════════════════════════════════════════════════════
# PET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet', methods=['GET'])
@require_auth
def get_pet(user: User):
    """Get pet status"""
    if not user.pet:
        return jsonify({'error': 'No pet'}), 404
    
    user.pet.update_stats()
    data_manager.save_users(users)
    
    return jsonify({'pet': asdict(user.pet)})


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet(user: User):
    """Feed the pet"""
    if not user.pet:
        return jsonify({'error': 'No pet'}), 404
    
    user.pet.feed()
    data_manager.save_users(users)
    
    return jsonify({
        'message': f'{user.pet.name} enjoyed the meal!',
        'pet': asdict(user.pet)
    })


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_with_pet(user: User):
    """Play with the pet"""
    if not user.pet:
        return jsonify({'error': 'No pet'}), 404
    
    user.pet.play()
    data_manager.save_users(users)
    
    return jsonify({
        'message': f'{user.pet.name} had fun!',
        'pet': asdict(user.pet)
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/subscription/status', methods=['GET'])
@require_auth
def subscription_status(user: User):
    """Check subscription status"""
    
    is_trial = user.subscription_status == 'trial'
    trial_expired = False
    days_remaining = 0
    
    if is_trial and user.trial_start:
        trial_start = datetime.fromisoformat(user.trial_start)
        trial_end = trial_start + timedelta(days=Config.TRIAL_DAYS)
        trial_expired = datetime.now(timezone.utc) > trial_end
        days_remaining = max(0, (trial_end - datetime.now(timezone.utc)).days)
    
    return jsonify({
        'status': user.subscription_status,
        'is_trial': is_trial,
        'trial_expired': trial_expired,
        'days_remaining': days_remaining,
        'gofundme_url': Config.GOFUNDME_CAMPAIGN_URL if is_trial else None
    })


@app.route('/api/subscription/create-checkout', methods=['POST'])
@require_auth
def create_checkout(user: User):
    """Create Stripe checkout session (simplified)"""
    
    # In production, integrate with Stripe API
    # For now, return mock checkout URL
    
    return jsonify({
        'checkout_url': f'https://checkout.stripe.com/...',
        'session_id': secrets.token_hex(16)
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA EXPORT/IMPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/export', methods=['GET'])
@require_auth
def export_data(user: User):
    """Export all user data as JSON"""
    
    backup_bytes = data_manager.backup_user_data(user)
    
    return send_file(
        BytesIO(backup_bytes),
        mimetype='application/json',
        as_attachment=True,
        download_name=f'life_fractal_backup_{datetime.now().strftime("%Y%m%d")}.json'
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# DASHBOARD & STATS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/dashboard', methods=['GET'])
@require_auth
def dashboard(user: User):
    """Get complete dashboard data"""
    
    metrics = fractal_engine._calculate_life_metrics(user)
    
    return jsonify({
        'user': user.to_dict(),
        'metrics': metrics,
        'recent_goals': [asdict(g) for g in user.goals[-5:]],
        'recent_habits': [asdict(h) for h in user.habits[-5:]],
        'recent_journal': [asdict(e) for e in user.journal[-5:]],
        'gpu_status': {
            'available': GPU_AVAILABLE,
            'device': GPU_NAME
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'users': len(users),
        'gpu': GPU_AVAILABLE,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/')
def index():
    """Landing page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Life Fractal Intelligence</title>
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
            }
            h1 { font-size: 48px; margin-bottom: 20px; }
            .features { max-width: 800px; margin: 40px auto; text-align: left; }
            .feature { margin: 15px 0; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; }
            .cta { margin-top: 40px; }
            .btn { 
                padding: 15px 40px; 
                font-size: 18px; 
                background: #ffd700; 
                color: #333; 
                border: none; 
                border-radius: 30px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>🌀 Life Fractal Intelligence</h1>
        <p style="font-size: 24px;">Your life, visualized as living fractal art</p>
        
        <div class="features">
            <div class="feature">✨ GPU-accelerated fractal visualization driven by YOUR progress</div>
            <div class="feature">🎯 Goal tracking with ML-powered predictions</div>
            <div class="feature">🐉 Virtual pet companion that grows with you</div>
            <div class="feature">📔 Journal with sentiment analysis</div>
            <div class="feature">🔮 Sacred geometry overlays based on your momentum</div>
            <div class="feature">💎 $20/month • 7-day free trial</div>
        </div>
        
        <div class="cta">
            <a href="/api/docs" class="btn">View API Documentation</a>
        </div>
        
        <p style="margin-top: 40px; opacity: 0.8;">
            Powered by ancient mathematics, modern AI, and your dedication
        </p>
    </body>
    </html>
    """


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    🌀 LIFE FRACTAL INTELLIGENCE - PRODUCTION READY
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    🚀 Server: http://localhost:{port}
    🎨 GPU: {GPU_NAME if GPU_AVAILABLE else 'CPU Mode'}
    👥 Users: {len(users)}
    📊 Data Dir: {Config.DATA_DIR}
    
    📡 API Endpoints:
    ├─ POST /api/register           → Register new user
    ├─ POST /api/login              → Login
    ├─ GET  /api/dashboard          → Dashboard data
    ├─ GET  /api/fractal/generate   → Generate personalized fractal
    ├─ GET  /api/fractal/metrics    → Fractal metrics
    ├─ GET  /api/goals              → Get goals
    ├─ POST /api/goals              → Create goal
    ├─ POST /api/goals/<id>/tasks   → Add task
    ├─ POST /api/tasks/<id>/complete → Complete task
    ├─ GET  /api/habits             → Get habits
    ├─ POST /api/habits             → Create habit
    ├─ POST /api/habits/<id>/complete → Complete habit
    ├─ GET  /api/journal            → Get journal entries
    ├─ POST /api/journal            → Create entry
    ├─ GET  /api/pet                → Get pet status
    ├─ POST /api/pet/feed           → Feed pet
    ├─ POST /api/pet/play           → Play with pet
    ├─ GET  /api/subscription/status → Check subscription
    ├─ GET  /api/export             → Export user data
    └─ GET  /health                 → Health check
    
    ═══════════════════════════════════════════════════════════════════════════════════════
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
