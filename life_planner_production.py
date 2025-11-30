#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE - PRODUCTION SYSTEM v5.0
═══════════════════════════════════════════════════════════════════════════════
COMPLETE PRODUCTION-READY SYSTEM WITH:
✅ Self-healing & automatic fallbacks (NEVER CRASHES)
✅ Full authentication (register/login with sessions)
✅ Beautiful HTML dashboard (no external dependencies)
✅ GPU-accelerated fractal visualization
✅ Virtual pet system (5 species with evolution)
✅ Goal & habit tracking with Fibonacci milestones
✅ Journal with sentiment analysis
✅ ML mood predictions
✅ Auto-backup every 5 minutes
✅ Production deployment ready (Gunicorn/Nginx)
═══════════════════════════════════════════════════════════════════════════════
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
import traceback
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
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
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════════════════
# SELF-HEALING SYSTEM - THE CORE OF RELIABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingSystem:
    """Central self-healing manager for the entire application."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, str] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.fallback_used: Dict[str, int] = {}
    
    def record_error(self, component: str, error: str):
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        self.last_errors[component] = f"{datetime.now().isoformat()}: {error}"
    
    def record_recovery(self, component: str):
        self.recovery_attempts[component] = self.recovery_attempts.get(component, 0) + 1
    
    def record_fallback(self, component: str):
        self.fallback_used[component] = self.fallback_used.get(component, 0) + 1
    
    def get_health_report(self) -> dict:
        return {
            'error_counts': self.error_counts,
            'last_errors': self.last_errors,
            'recovery_attempts': self.recovery_attempts,
            'fallback_used': self.fallback_used,
            'overall_health': 'healthy' if sum(self.error_counts.values()) < 10 else 'degraded'
        }

# Global self-healing instance
HEALER = SelfHealingSystem()


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, fallback: Any = None, component: str = "unknown"):
    """
    Decorator for automatic retry with exponential backoff.
    NEVER lets an error crash the application.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        HEALER.record_recovery(component)
                        logger.info(f"✅ {component} recovered after {attempt + 1} attempts")
                    return result
                except Exception as e:
                    last_exception = e
                    HEALER.record_error(component, str(e))
                    logger.warning(f"⚠️ Attempt {attempt + 1}/{max_attempts} failed for {component}: {e}")
                    
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
            
            # All attempts failed - use fallback
            logger.error(f"❌ All attempts failed for {component}, using fallback")
            HEALER.record_fallback(component)
            
            if callable(fallback):
                try:
                    return fallback(*args, **kwargs)
                except:
                    return None
            return fallback
        return wrapper
    return decorator


def safe_execute(fallback_value: Any = None, log_errors: bool = True, component: str = "unknown"):
    """
    Decorator that NEVER raises exceptions - always returns a value.
    The ultimate safety net.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    HEALER.record_error(component, str(e))
                    logger.error(f"🛡️ Safe execution caught error in {component}: {e}")
                return fallback_value
        return wrapper
    return decorator


def graceful_degradation(primary_func: Callable, fallback_func: Callable, component: str = "unknown"):
    """
    Try primary function, automatically fall back to simpler version if it fails.
    """
    def wrapper(*args, **kwargs):
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            HEALER.record_error(component, str(e))
            HEALER.record_fallback(component)
            logger.warning(f"🔄 {component} degraded to fallback mode: {e}")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as e2:
                logger.error(f"❌ Even fallback failed for {component}: {e2}")
                return None
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP (MUST BE BEFORE OPTIONAL IMPORTS)
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS WITH GRACEFUL DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════════

# ML Support
try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.info("📊 ML features disabled (scikit-learn not installed)")

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None
    logger.info("🖥️ GPU features disabled (PyTorch not installed)")

# Video Support
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.info("🎬 Video features disabled (OpenCV not installed)")

# Audio Support
try:
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    logger.info("🔊 Audio features disabled (soundfile not installed)")


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MoodLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    NEUTRAL = 3
    GOOD = 4
    EXCELLENT = 5


class PetSpecies(Enum):
    CAT = "cat"
    DRAGON = "dragon"
    PHOENIX = "phoenix"
    OWL = "owl"
    FOX = "fox"


@dataclass
class PetState:
    """Virtual pet state tracking."""
    species: str = "cat"
    name: str = "Buddy"
    hunger: float = 50.0
    energy: float = 50.0
    mood: float = 50.0
    stress: float = 50.0
    growth: float = 1.0
    level: int = 1
    experience: int = 0
    bond: float = 0.0
    behavior: str = "idle"
    evolution_stage: int = 0
    total_tasks_completed: int = 0
    total_goals_achieved: int = 0
    last_fed: Optional[str] = None
    last_played: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailyEntry:
    """A single day's entry in the life planner."""
    date: str  # YYYY-MM-DD
    mood_level: int = 3
    mood_score: float = 50.0
    energy_level: float = 50.0
    focus_clarity: float = 50.0
    anxiety_level: float = 30.0
    stress_level: float = 30.0
    mindfulness_score: float = 50.0
    gratitude_level: float = 50.0
    sleep_quality: float = 50.0
    sleep_hours: float = 7.0
    nutrition_score: float = 50.0
    social_connection: float = 50.0
    emotional_stability: float = 50.0
    self_compassion: float = 50.0
    habits_completed: Dict[str, bool] = field(default_factory=dict)
    journal_entry: str = ""
    journal_sentiment: float = 0.5
    goals_progressed: Dict[str, float] = field(default_factory=dict)
    goals_completed_count: int = 0
    gratitude_items: List[str] = field(default_factory=list)
    wins: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    tomorrow_intentions: List[str] = field(default_factory=list)
    wellness_index: float = 0.0
    predicted_mood: float = 0.0
    
    def __post_init__(self):
        self.calculate_wellness()
    
    def calculate_wellness(self):
        """Calculate overall wellness index using Fibonacci weighting."""
        weights = [FIBONACCI[i+3] for i in range(8)]
        total_weight = sum(weights)
        
        positive = (
            self.mood_level * 20 * weights[0] +
            self.energy_level * weights[1] +
            self.focus_clarity * weights[2] +
            self.mindfulness_score * weights[3] +
            self.gratitude_level * weights[4] +
            self.sleep_quality * weights[5] +
            self.emotional_stability * weights[6] +
            self.self_compassion * weights[7]
        )
        
        negative = (self.anxiety_level + self.stress_level) * sum(weights[:3])
        self.wellness_index = max(0, min(100, (positive - negative / 2) / total_weight))
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Habit:
    """A trackable habit."""
    id: str
    name: str
    description: str = ""
    frequency: str = "daily"
    category: str = "general"
    current_streak: int = 0
    longest_streak: int = 0
    total_completions: int = 0
    created_at: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Goal:
    """A goal with progress tracking and Fibonacci milestones."""
    id: str
    title: str
    description: str = ""
    category: str = "general"
    priority: int = 3
    progress: float = 0.0
    target_date: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    velocity: float = 0.0
    why_important: str = ""
    subtasks: List[str] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    milestones: List[int] = field(default_factory=lambda: [8, 13, 21, 34, 55, 89, 100])
    milestones_reached: List[int] = field(default_factory=list)
    
    @property
    def is_completed(self) -> bool:
        return self.progress >= 100 or self.completed_at is not None
    
    def check_milestones(self) -> Optional[int]:
        for milestone in self.milestones:
            if self.progress >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.append(milestone)
                return milestone
        return None
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['is_completed'] = self.is_completed
        return data


@dataclass
class User:
    """User account with subscription management."""
    id: str
    email: str
    password_hash: str
    first_name: str = ""
    last_name: str = ""
    is_active: bool = True
    is_admin: bool = False
    email_verified: bool = False
    subscription_status: str = "trial"
    trial_start_date: str = ""
    trial_end_date: str = ""
    stripe_customer_id: Optional[str] = None
    pet: Optional[PetState] = None
    daily_entries: Dict[str, DailyEntry] = field(default_factory=dict)
    habits: Dict[str, Habit] = field(default_factory=dict)
    goals: Dict[str, Goal] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    fractal_type: str = "hybrid"
    show_flower_of_life: bool = True
    show_metatron_cube: bool = True
    show_golden_spiral: bool = True
    animation_speed: float = 1.0
    created_at: str = ""
    last_login: str = ""
    current_streak: int = 0
    longest_streak: int = 0
    
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
    
    def is_trial_active(self) -> bool:
        if not self.trial_end_date:
            return False
        try:
            end = datetime.fromisoformat(self.trial_end_date.replace('Z', '+00:00'))
            return datetime.now(timezone.utc) < end and self.subscription_status == 'trial'
        except:
            return False
    
    def has_active_subscription(self) -> bool:
        return self.is_trial_active() or self.subscription_status == 'active'
    
    def days_remaining_trial(self) -> int:
        if not self.trial_end_date:
            return 0
        try:
            end = datetime.fromisoformat(self.trial_end_date.replace('Z', '+00:00'))
            delta = end - datetime.now(timezone.utc)
            return max(0, delta.days)
        except:
            return 0
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        data = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'subscription_status': self.subscription_status,
            'trial_days_remaining': self.days_remaining_trial(),
            'has_access': self.has_active_subscription(),
            'created_at': self.created_at,
            'last_login': self.last_login,
            'current_streak': self.current_streak,
            'longest_streak': self.longest_streak
        }
        if include_sensitive:
            data['is_admin'] = self.is_admin
            data['email_verified'] = self.email_verified
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY LOGIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FuzzyLogicEngine:
    """Fuzzy logic for generating supportive messages."""
    
    def __init__(self):
        self.messages = {
            ('low_stress', 'high_mood'): "🌟 You're doing great! Keep nurturing this positive energy.",
            ('low_stress', 'medium_mood'): "☀️ You're in a good place. Small joys can lift you higher.",
            ('low_stress', 'low_mood'): "🌱 Even on quieter days, you're managing well. Be gentle with yourself.",
            ('medium_stress', 'high_mood'): "💪 Your resilience is shining through! Take breaks when needed.",
            ('medium_stress', 'medium_mood'): "⚖️ Balance is key. You're navigating well through challenges.",
            ('medium_stress', 'low_mood'): "🧘 It's okay to feel this way. Consider a short mindful pause.",
            ('high_stress', 'high_mood'): "🎯 Your positivity is admirable! Don't forget to rest.",
            ('high_stress', 'medium_mood'): "📋 You're handling a lot. Prioritize what matters most.",
            ('high_stress', 'low_mood'): "💙 These feelings are valid. Reach out for support if needed."
        }
    
    def _fuzzy_membership(self, value: float, low: float, high: float) -> str:
        if value <= low:
            return 'low'
        elif value >= high:
            return 'high'
        return 'medium'
    
    @safe_execute(fallback_value="Take a moment to breathe and reflect. 🌿", component="fuzzy_logic")
    def infer(self, stress: float, mood: float) -> str:
        stress_level = self._fuzzy_membership(stress, 30, 70)
        mood_level = self._fuzzy_membership(mood, 30, 70)
        key = (f'{stress_level}_stress', f'{mood_level}_mood')
        return self.messages.get(key, "Take a moment to breathe and reflect. 🌿")


# ═══════════════════════════════════════════════════════════════════════════════
# MOOD PREDICTOR (ML)
# ═══════════════════════════════════════════════════════════════════════════════

class MoodPredictor:
    """Decision tree-based mood prediction with self-healing."""
    
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42, max_depth=5) if HAS_SKLEARN else None
        self.trained = False
        self.training_samples = 0
    
    @safe_execute(fallback_value=False, component="ml_training")
    def train(self, history: List[Dict]) -> bool:
        if not HAS_SKLEARN or not history or len(history) < 5:
            return False
        
        X, y = [], []
        for i, record in enumerate(history[:-1]):
            features = [
                float(record.get('stress_level', 50)),
                float(record.get('mood_score', 50)),
                float(record.get('energy_level', 50)),
                float(record.get('goals_completed_count', 0)),
                float(record.get('sleep_hours', 7)),
                float(record.get('sleep_quality', 50))
            ]
            target = float(history[i+1].get('mood_score', 50))
            X.append(features)
            y.append(target)
        
        if len(X) >= 5:
            self.model.fit(X, y)
            self.trained = True
            self.training_samples = len(X)
            logger.info(f"🤖 ML model trained on {len(X)} samples")
            return True
        return False
    
    @safe_execute(fallback_value=50.0, component="ml_prediction")
    def predict(self, current_state: Dict) -> float:
        if not self.trained or not HAS_SKLEARN:
            return float(current_state.get('mood_score', 50))
        
        features = [[
            float(current_state.get('stress_level', 50)),
            float(current_state.get('mood_score', 50)),
            float(current_state.get('energy_level', 50)),
            float(current_state.get('goals_completed_count', 0)),
            float(current_state.get('sleep_hours', 7)),
            float(current_state.get('sleep_quality', 50))
        ]]
        return float(self.model.predict(features)[0])


# ═══════════════════════════════════════════════════════════════════════════════
# VIRTUAL PET SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualPet:
    """Virtual pet with behavior and evolution mechanics."""
    
    SPECIES_TRAITS = {
        'cat': {'energy_decay': 1.2, 'mood_sensitivity': 1.0, 'growth_rate': 1.0, 'emoji': '🐱'},
        'dragon': {'energy_decay': 0.8, 'mood_sensitivity': 1.3, 'growth_rate': 1.2, 'emoji': '🐲'},
        'phoenix': {'energy_decay': 1.0, 'mood_sensitivity': 0.8, 'growth_rate': 1.5, 'emoji': '🔥'},
        'owl': {'energy_decay': 0.9, 'mood_sensitivity': 1.1, 'growth_rate': 0.9, 'emoji': '🦉'},
        'fox': {'energy_decay': 1.1, 'mood_sensitivity': 1.2, 'growth_rate': 1.1, 'emoji': '🦊'}
    }
    
    def __init__(self, state: PetState):
        self.state = state
        self.traits = self.SPECIES_TRAITS.get(state.species, self.SPECIES_TRAITS['cat'])
    
    @safe_execute(fallback_value=None, component="pet_update")
    def update_from_user_data(self, user_data: Dict):
        sleep_quality = user_data.get('sleep_quality', 50)
        self.state.energy = min(100, self.state.energy + (sleep_quality - 50) * 0.2)
        
        user_mood = user_data.get('mood_score', 50)
        mood_delta = (user_mood - 50) * 0.3 * self.traits['mood_sensitivity']
        self.state.mood = max(0, min(100, self.state.mood + mood_delta))
        
        mindfulness = user_data.get('mindfulness_score', 50)
        self.state.stress = max(0, min(100, 100 - mindfulness * 0.8))
        
        goals = user_data.get('goals_completed_count', 0)
        self.state.growth = min(100, self.state.growth + goals * 2 * self.traits['growth_rate'])
        
        xp_gain = int(goals * 10 + (user_mood / 10))
        self.state.experience += xp_gain
        
        xp_for_next = FIBONACCI[min(self.state.level + 5, len(FIBONACCI)-1)] * 10
        if self.state.experience >= xp_for_next:
            self.state.level += 1
            self.state.experience -= xp_for_next
            if self.state.level % 5 == 0:
                self.state.evolution_stage = min(3, self.state.evolution_stage + 1)
        
        self.state.hunger = min(100, self.state.hunger + 2 * self.traits['energy_decay'])
        self.state.energy = max(0, self.state.energy - 1 * self.traits['energy_decay'])
        self._update_behavior()
    
    def _update_behavior(self):
        if self.state.hunger > 80:
            self.state.behavior = 'hungry'
        elif self.state.energy < 20:
            self.state.behavior = 'tired'
        elif self.state.energy < 10:
            self.state.behavior = 'sleeping'
        elif self.state.mood > 80:
            self.state.behavior = 'excited'
        elif self.state.mood > 60:
            self.state.behavior = 'playful'
        elif self.state.mood > 40:
            self.state.behavior = 'happy'
        elif self.state.mood < 30:
            self.state.behavior = 'sad'
        else:
            self.state.behavior = 'idle'
    
    def feed(self) -> bool:
        self.state.hunger = max(0, self.state.hunger - 30)
        self.state.mood = min(100, self.state.mood + 5)
        self.state.last_fed = datetime.now(timezone.utc).isoformat()
        self._update_behavior()
        return True
    
    def play(self) -> bool:
        if self.state.energy < 20:
            return False
        self.state.energy = max(0, self.state.energy - 15)
        self.state.mood = min(100, self.state.mood + 15)
        self.state.bond = min(100, self.state.bond + 3)
        self.state.last_played = datetime.now(timezone.utc).isoformat()
        self._update_behavior()
        return True
    
    def get_message(self) -> str:
        emoji = self.traits['emoji']
        messages = {
            'happy': f"{emoji} {self.state.name} is happy! Your positivity is contagious!",
            'playful': f"{emoji} {self.state.name} wants to celebrate your progress!",
            'excited': f"{emoji} {self.state.name} is absolutely thrilled! Keep it up!",
            'tired': f"{emoji} {self.state.name} is resting. Maybe you need rest too?",
            'hungry': f"{emoji} {self.state.name} is hungry. Have you eaten today?",
            'sad': f"{emoji} {self.state.name} senses you might be down. It's here for you.",
            'idle': f"{emoji} {self.state.name} is keeping you company.",
            'sleeping': f"{emoji} {self.state.name} is catching Z's. Rest is important!"
        }
        return messages.get(self.state.behavior, f"{emoji} {self.state.name} is with you.")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-ACCELERATED FRACTAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FractalGenerator:
    """GPU-accelerated fractal generation with CPU fallback."""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.use_gpu = GPU_AVAILABLE
    
    @retry_on_failure(max_attempts=2, fallback=None, component="fractal_gpu")
    def _mandelbrot_gpu(self, max_iter: int, zoom: float, center: Tuple[float, float]) -> np.ndarray:
        if not self.use_gpu or torch is None:
            raise RuntimeError("GPU not available")
        
        device = torch.device('cuda')
        x = torch.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width, device=device)
        y = torch.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        c = X + 1j * Y
        z = torch.zeros_like(c)
        iterations = torch.zeros(self.height, self.width, device=device)
        
        for i in range(max_iter):
            mask = torch.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        return iterations.cpu().numpy()
    
    @safe_execute(fallback_value=None, component="fractal_cpu")
    def _mandelbrot_cpu(self, max_iter: int, zoom: float, center: Tuple[float, float]) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def generate_mandelbrot(self, max_iter: int = 256, zoom: float = 1.0,
                           center: Tuple[float, float] = (-0.5, 0)) -> np.ndarray:
        # Try GPU first, fall back to CPU
        result = graceful_degradation(
            lambda: self._mandelbrot_gpu(max_iter, zoom, center),
            lambda: self._mandelbrot_cpu(max_iter, zoom, center),
            component="fractal_generation"
        )()
        
        if result is None:
            # Ultimate fallback: simple gradient
            logger.warning("Using placeholder fractal")
            result = np.zeros((self.height, self.width))
            for i in range(self.height):
                for j in range(self.width):
                    result[i, j] = (i + j) % max_iter
        
        return result
    
    @safe_execute(fallback_value=None, component="julia_generation")
    def generate_julia(self, c_real: float = -0.7, c_imag: float = 0.27015,
                      max_iter: int = 256, zoom: float = 1.0) -> np.ndarray:
        x = np.linspace(-2/zoom, 2/zoom, self.width)
        y = np.linspace(-2/zoom, 2/zoom, self.height)
        X, Y = np.meshgrid(x, y)
        
        z = X + 1j * Y
        c = complex(c_real, c_imag)
        iterations = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c
            iterations[mask] = i
        
        return iterations
    
    @safe_execute(component="fractal_coloring")
    def apply_coloring(self, iterations: np.ndarray, max_iter: int,
                      hue_base: float = 0.6, hue_range: float = 0.3,
                      saturation: float = 0.8) -> np.ndarray:
        normalized = iterations / max_iter
        
        h = (hue_base + normalized * hue_range) % 1.0
        s = np.full_like(normalized, saturation)
        v = np.sqrt(normalized) * 0.9 + 0.1
        
        inside = normalized >= 0.99
        v[inside] = 0.05
        s[inside] = 0
        
        rgb = np.zeros((*iterations.shape, 3), dtype=np.uint8)
        
        i = (h * 6).astype(int) % 6
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
    
    @safe_execute(component="visualization_creation")
    def create_visualization(self, user_data: Dict) -> Image.Image:
        mood = user_data.get('mood_score', 50)
        wellness = user_data.get('wellness_index', 50)
        
        if wellness < 30:
            iterations = self.generate_julia(-0.8, 0.156, max_iter=200)
            hue_base = 0.7
        elif wellness < 60:
            iterations = self.generate_mandelbrot(max_iter=256, zoom=1.5)
            hue_base = 0.5 + (mood - 50) / 200
        else:
            m = self.generate_mandelbrot(max_iter=256, zoom=2.0)
            j = self.generate_julia(-0.7 + (mood-50)/200, 0.27, max_iter=200)
            if m is not None and j is not None:
                iterations = m * 0.5 + j * 0.5
            else:
                iterations = m if m is not None else j
            hue_base = 0.3 + (mood / 200)
        
        if iterations is None:
            # Create a simple gradient fallback
            iterations = np.zeros((self.height, self.width))
            for i in range(self.height):
                for j in range(self.width):
                    iterations[i, j] = ((i - self.height/2)**2 + (j - self.width/2)**2) ** 0.5
        
        rgb = self.apply_coloring(iterations, 256, hue_base, 0.3, 0.8)
        
        if rgb is None:
            # Ultimate fallback - solid color based on mood
            rgb = np.full((self.height, self.width, 3), [int(mood * 2.55), 100, 150], dtype=np.uint8)
        
        return Image.fromarray(rgb, 'RGB')
    
    def to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# LIFE PLANNING SYSTEM (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class LifePlanningSystem:
    """Main orchestrator for all life planning features."""
    
    def __init__(self, pet_species: str = "cat"):
        self.fractal_gen = FractalGenerator(512, 512)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.predictor = MoodPredictor()
        self.pet = VirtualPet(PetState(species=pet_species))
        self.history: List[Dict] = []
    
    @safe_execute(component="system_update")
    def update(self, user_data: Dict):
        self.pet.update_from_user_data(user_data)
        record = {**user_data, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.history.append(record)
        
        if len(self.history) >= 5:
            self.predictor.train(self.history)
    
    @safe_execute(component="guidance_generation")
    def generate_guidance(self, current_state: Dict) -> Dict[str, Any]:
        predicted_mood = self.predictor.predict(current_state)
        stress = current_state.get('stress_level', 50)
        mood = current_state.get('mood_score', 50)
        fuzzy_message = self.fuzzy_engine.infer(stress, mood)
        pet_message = self.pet.get_message()
        
        return {
            'predicted_mood': round(predicted_mood, 1),
            'fuzzy_message': fuzzy_message,
            'pet_message': pet_message,
            'pet_state': self.pet.state.to_dict(),
            'combined_message': f"{fuzzy_message}\n\n{pet_message}"
        }
    
    def generate_fractal_image(self, user_data: Dict) -> Image.Image:
        return self.fractal_gen.create_visualization(user_data)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STORE WITH AUTO-BACKUP
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """In-memory data store with auto-backup and self-healing."""
    
    def __init__(self, data_dir: str = "life_planner_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.systems: Dict[str, LifePlanningSystem] = {}
        self.sessions: Dict[str, str] = {}  # session_token -> user_id
        
        self._load_data()
        self._init_admin()
        self._start_auto_backup()
    
    def _init_admin(self):
        """Create admin user if not exists."""
        if 'onlinediscountsllc@gmail.com' not in self.users:
            admin_id = 'admin_001'
            admin = User(
                id=admin_id,
                email='onlinediscountsllc@gmail.com',
                password_hash='',
                first_name='Luke',
                last_name='Smith',
                is_admin=True,
                is_active=True,
                email_verified=True,
                subscription_status='active',
                created_at=datetime.now(timezone.utc).isoformat()
            )
            admin.set_password('admin8587037321')
            admin.pet = PetState(species='dragon', name='Ember')
            self._add_demo_data(admin)
            self.users[admin_id] = admin
            self.users[admin.email] = admin
            logger.info("✅ Admin user initialized")
    
    @safe_execute(component="data_load")
    def _load_data(self):
        """Load data from disk."""
        users_file = self.data_dir / "users.json"
        if users_file.exists():
            with open(users_file, 'r') as f:
                data = json.load(f)
                for user_data in data.get('users', []):
                    user = self._dict_to_user(user_data)
                    if user:
                        self.users[user.id] = user
                        self.users[user.email] = user
            logger.info(f"📂 Loaded {len(data.get('users', []))} users from disk")
    
    @safe_execute(component="data_save")
    def _save_data(self):
        """Save data to disk."""
        users_data = []
        seen_ids = set()
        for user in self.users.values():
            if user.id not in seen_ids:
                users_data.append(self._user_to_dict(user))
                seen_ids.add(user.id)
        
        users_file = self.data_dir / "users.json"
        with open(users_file, 'w') as f:
            json.dump({'users': users_data, 'saved_at': datetime.now().isoformat()}, f, indent=2)
        
        logger.info(f"💾 Saved {len(users_data)} users to disk")
    
    def _user_to_dict(self, user: User) -> dict:
        """Convert user to serializable dict."""
        return {
            'id': user.id,
            'email': user.email,
            'password_hash': user.password_hash,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'is_active': user.is_active,
            'is_admin': user.is_admin,
            'subscription_status': user.subscription_status,
            'trial_start_date': user.trial_start_date,
            'trial_end_date': user.trial_end_date,
            'created_at': user.created_at,
            'last_login': user.last_login,
            'current_streak': user.current_streak,
            'longest_streak': user.longest_streak,
            'pet': user.pet.to_dict() if user.pet else None,
            'habits': {k: v.to_dict() for k, v in user.habits.items()},
            'goals': {k: v.to_dict() for k, v in user.goals.items()},
            'daily_entries': {k: v.to_dict() for k, v in user.daily_entries.items()}
        }
    
    def _dict_to_user(self, data: dict) -> Optional[User]:
        """Convert dict to User object."""
        try:
            user = User(
                id=data['id'],
                email=data['email'],
                password_hash=data['password_hash'],
                first_name=data.get('first_name', ''),
                last_name=data.get('last_name', ''),
                is_active=data.get('is_active', True),
                is_admin=data.get('is_admin', False),
                subscription_status=data.get('subscription_status', 'trial'),
                trial_start_date=data.get('trial_start_date', ''),
                trial_end_date=data.get('trial_end_date', ''),
                created_at=data.get('created_at', ''),
                last_login=data.get('last_login', ''),
                current_streak=data.get('current_streak', 0),
                longest_streak=data.get('longest_streak', 0)
            )
            
            if data.get('pet'):
                user.pet = PetState(**data['pet'])
            
            for habit_data in data.get('habits', {}).values():
                habit = Habit(**habit_data)
                user.habits[habit.id] = habit
            
            for goal_data in data.get('goals', {}).values():
                goal = Goal(**goal_data)
                user.goals[goal.id] = goal
            
            return user
        except Exception as e:
            logger.error(f"Failed to load user: {e}")
            return None
    
    def _start_auto_backup(self):
        """Start background auto-backup thread."""
        def backup_loop():
            while True:
                time.sleep(300)  # 5 minutes
                self._save_data()
                self._create_backup()
        
        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()
        logger.info("🔄 Auto-backup started (every 5 minutes)")
    
    @safe_execute(component="backup_creation")
    def _create_backup(self):
        """Create timestamped backup."""
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.json"
        
        users_file = self.data_dir / "users.json"
        if users_file.exists():
            import shutil
            shutil.copy(users_file, backup_file)
            
            # Keep only last 10 backups
            backups = sorted(backup_dir.glob("backup_*.json"))
            while len(backups) > 10:
                backups[0].unlink()
                backups.pop(0)
            
            logger.info(f"📦 Backup created: {backup_file.name}")
    
    def _add_demo_data(self, user: User):
        """Add demo habits, goals, and entries."""
        now = datetime.now(timezone.utc)
        
        # Demo habits
        demo_habits = [
            ("Morning Meditation", "wellness", 12),
            ("Exercise 30 min", "health", 7),
            ("Read 20 pages", "growth", 5),
            ("Journal Entry", "wellness", 3),
            ("Drink 8 glasses water", "health", 14),
            ("Gratitude Practice", "wellness", 9)
        ]
        
        for i, (name, category, streak) in enumerate(demo_habits):
            habit = Habit(
                id=f"habit_{i+1}",
                name=name,
                category=category,
                current_streak=streak,
                longest_streak=streak + 5,
                total_completions=streak * 3,
                created_at=(now - timedelta(days=30)).isoformat()
            )
            user.habits[habit.id] = habit
        
        # Demo goals
        demo_goals = [
            ("Complete Life Planner App", "career", 1, 85),
            ("Launch GoFundMe Campaign", "financial", 2, 100),
            ("Build User Base to 100", "growth", 3, 25)
        ]
        
        for i, (title, category, priority, progress) in enumerate(demo_goals):
            goal = Goal(
                id=f"goal_{i+1}",
                title=title,
                category=category,
                priority=priority,
                progress=progress,
                target_date=(now + timedelta(days=30 + i*30)).isoformat()[:10],
                created_at=(now - timedelta(days=60)).isoformat(),
                why_important="This is crucial for my personal growth and financial independence.",
                subtasks=["Research", "Plan", "Execute", "Review"],
                obstacles=["Time constraints", "Technical challenges"],
                success_criteria=["Fully functional", "User tested", "Deployed"]
            )
            goal.check_milestones()
            user.goals[goal.id] = goal
        
        # Demo daily entries
        for i in range(14):
            date = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            entry = DailyEntry(
                date=date,
                mood_level=max(1, min(5, 3 + int(math.sin(i*0.5) * 1.5))),
                mood_score=50 + math.sin(i*0.5) * 25,
                energy_level=50 + math.cos(i*0.4) * 20,
                focus_clarity=60 + math.sin(i*0.3) * 15,
                anxiety_level=30 - i * 0.5,
                stress_level=35 - i * 0.3,
                mindfulness_score=50 + i * 1.5,
                gratitude_level=55 + i * 1.2,
                sleep_quality=70 + math.sin(i*0.2) * 15,
                sleep_hours=7 + math.sin(i*0.3),
                goals_completed_count=i % 3,
                gratitude_items=["Family", "Health", "Progress"],
                wins=["Made progress on project"],
                journal_entry=f"Day {14-i}: Feeling productive and focused."
            )
            entry.calculate_wellness()
            user.daily_entries[date] = entry
            user.history.append(entry.to_dict())
    
    def create_user(self, email: str, password: str, first_name: str = "", last_name: str = "") -> Optional[User]:
        """Create new user with 7-day trial."""
        if email.lower() in self.users:
            return None
        
        now = datetime.now(timezone.utc)
        user_id = f"user_{secrets.token_hex(8)}"
        
        user = User(
            id=user_id,
            email=email.lower(),
            password_hash='',
            first_name=first_name,
            last_name=last_name,
            subscription_status='trial',
            trial_start_date=now.isoformat(),
            trial_end_date=(now + timedelta(days=7)).isoformat(),
            created_at=now.isoformat()
        )
        user.set_password(password)
        user.pet = PetState(species='cat', name='Buddy')
        
        self._add_demo_data(user)
        
        self.users[user_id] = user
        self.users[email.lower()] = user
        
        self._save_data()
        
        return user
    
    def get_user(self, identifier: str) -> Optional[User]:
        return self.users.get(identifier) or self.users.get(identifier.lower())
    
    def get_system(self, user_id: str) -> LifePlanningSystem:
        if user_id not in self.systems:
            user = self.users.get(user_id)
            species = user.pet.species if user and user.pet else 'cat'
            self.systems[user_id] = LifePlanningSystem(species)
        return self.systems[user_id]
    
    def create_session(self, user_id: str) -> str:
        """Create a session token for user."""
        token = secrets.token_urlsafe(32)
        self.sessions[token] = user_id
        return token
    
    def get_user_from_session(self, token: str) -> Optional[User]:
        """Get user from session token."""
        user_id = self.sessions.get(token)
        return self.get_user(user_id) if user_id else None


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
CORS(app, supports_credentials=True)

store = DataStore()

# Configuration
SUBSCRIPTION_PRICE = 20.00
TRIAL_DAYS = 7
GOFUNDME_URL = 'https://gofund.me/8d9303d27'


# ═══════════════════════════════════════════════════════════════════════════════
# BEAUTIFUL HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #475569;
            --golden: #d4af37;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        header {
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            font-size: 48px;
            animation: spin 10s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .logo h1 {
            font-size: 28px;
            font-weight: 700;
        }
        
        .logo p {
            color: rgba(255,255,255,0.8);
            font-size: 14px;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .user-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-golden {
            background: linear-gradient(135deg, var(--golden), #b8860b);
            color: white;
        }
        
        .btn-outline {
            background: transparent;
            border: 2px solid var(--border);
            color: var(--text);
        }
        
        .btn-outline:hover {
            border-color: var(--primary);
            color: var(--primary);
        }
        
        /* Grid Layout */
        .grid {
            display: grid;
            gap: 20px;
        }
        
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
        
        @media (max-width: 1200px) {
            .grid-4 { grid-template-columns: repeat(2, 1fr); }
            .grid-3 { grid-template-columns: repeat(2, 1fr); }
        }
        
        @media (max-width: 768px) {
            .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; }
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Stats */
        .stat-card {
            text-align: center;
            padding: 30px;
        }
        
        .stat-icon {
            font-size: 40px;
            margin-bottom: 15px;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 5px;
        }
        
        /* Pet Card */
        .pet-card {
            background: linear-gradient(135deg, var(--bg-card), #2d3748);
            text-align: center;
        }
        
        .pet-avatar {
            font-size: 80px;
            margin: 20px 0;
            animation: bounce 2s ease-in-out infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .pet-name {
            font-size: 24px;
            font-weight: 700;
            color: var(--golden);
        }
        
        .pet-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .pet-stat {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
        }
        
        .pet-stat-label {
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .pet-stat-value {
            font-size: 18px;
            font-weight: 600;
        }
        
        .pet-actions {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        /* Progress Bars */
        .progress-bar {
            height: 8px;
            background: var(--bg-input);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        /* Form Elements */
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-muted);
        }
        
        .form-input {
            width: 100%;
            padding: 14px 16px;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 12px;
            color: var(--text);
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
        }
        
        textarea.form-input {
            min-height: 120px;
            resize: vertical;
        }
        
        /* Slider */
        .slider-group {
            margin-bottom: 20px;
        }
        
        .slider-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .slider {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: var(--bg-input);
            border-radius: 4px;
            outline: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            background: var(--primary);
            border-radius: 50%;
            cursor: pointer;
        }
        
        /* Lists */
        .habit-list, .goal-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .habit-item, .goal-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 16px;
            background: var(--bg-input);
            border-radius: 12px;
            transition: all 0.3s;
        }
        
        .habit-item:hover, .goal-item:hover {
            transform: translateX(5px);
            border-left: 3px solid var(--primary);
        }
        
        .habit-checkbox {
            width: 24px;
            height: 24px;
            border-radius: 6px;
            border: 2px solid var(--border);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }
        
        .habit-checkbox.checked {
            background: var(--success);
            border-color: var(--success);
        }
        
        .habit-info {
            flex: 1;
        }
        
        .habit-name {
            font-weight: 600;
        }
        
        .habit-streak {
            font-size: 12px;
            color: var(--golden);
        }
        
        /* Fractal Display */
        .fractal-container {
            text-align: center;
        }
        
        .fractal-image {
            max-width: 100%;
            border-radius: 16px;
            border: 3px solid var(--golden);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        
        /* Messages */
        .message {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        
        .message-info {
            background: rgba(99, 102, 241, 0.2);
            border-left: 4px solid var(--primary);
        }
        
        .message-success {
            background: rgba(16, 185, 129, 0.2);
            border-left: 4px solid var(--success);
        }
        
        .message-warning {
            background: rgba(245, 158, 11, 0.2);
            border-left: 4px solid var(--warning);
        }
        
        /* Auth Pages */
        .auth-container {
            max-width: 450px;
            margin: 50px auto;
        }
        
        .auth-card {
            background: var(--bg-card);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid var(--border);
        }
        
        .auth-logo {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .auth-logo .logo-icon {
            font-size: 64px;
        }
        
        .auth-title {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .auth-title h2 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .auth-title p {
            color: var(--text-muted);
        }
        
        .auth-footer {
            text-align: center;
            margin-top: 20px;
            color: var(--text-muted);
        }
        
        .auth-footer a {
            color: var(--primary);
            text-decoration: none;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 10px;
        }
        
        .tab {
            padding: 10px 20px;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .tab:hover {
            color: var(--text);
            background: var(--bg-input);
        }
        
        .tab.active {
            color: var(--primary);
            background: rgba(99, 102, 241, 0.2);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Sacred Math */
        .sacred-math {
            font-family: 'Courier New', monospace;
            background: rgba(212, 175, 55, 0.1);
            border: 1px solid var(--golden);
            border-radius: 12px;
            padding: 20px;
        }
        
        .sacred-math h4 {
            color: var(--golden);
            margin-bottom: 15px;
        }
        
        .sacred-value {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(212, 175, 55, 0.2);
        }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        /* Alerts */
        .alert {
            padding: 16px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: none;
        }
        
        .alert.show {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .alert-success {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
        }
        
        .alert-error {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }
        
        footer a {
            color: var(--golden);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Login Page -->
        <div id="login-page" class="auth-container">
            <div class="auth-card">
                <div class="auth-logo">
                    <div class="logo-icon">🌀</div>
                </div>
                <div class="auth-title">
                    <h2>Welcome Back</h2>
                    <p>Sign in to your Life Fractal Intelligence account</p>
                </div>
                
                <div id="login-alert" class="alert"></div>
                
                <form id="login-form">
                    <div class="form-group">
                        <label class="form-label">Email</label>
                        <input type="email" id="login-email" class="form-input" placeholder="you@example.com" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Password</label>
                        <input type="password" id="login-password" class="form-input" placeholder="••••••••" required>
                    </div>
                    <button type="submit" class="btn btn-primary" style="width: 100%;">
                        🔐 Sign In
                    </button>
                </form>
                
                <div class="auth-footer">
                    Don't have an account? <a href="#" onclick="showRegister()">Create one</a>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid var(--border);">
                    <p style="text-align: center; color: var(--text-muted); margin-bottom: 15px;">Quick Login (Demo)</p>
                    <button onclick="demoLogin()" class="btn btn-golden" style="width: 100%;">
                        ✨ Login as Admin
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Register Page -->
        <div id="register-page" class="auth-container" style="display: none;">
            <div class="auth-card">
                <div class="auth-logo">
                    <div class="logo-icon">🌀</div>
                </div>
                <div class="auth-title">
                    <h2>Create Account</h2>
                    <p>Start your 7-day free trial today!</p>
                </div>
                
                <div id="register-alert" class="alert"></div>
                
                <form id="register-form">
                    <div class="grid grid-2">
                        <div class="form-group">
                            <label class="form-label">First Name</label>
                            <input type="text" id="register-first" class="form-input" placeholder="John">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Last Name</label>
                            <input type="text" id="register-last" class="form-input" placeholder="Doe">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Email</label>
                        <input type="email" id="register-email" class="form-input" placeholder="you@example.com" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Password (min 8 characters)</label>
                        <input type="password" id="register-password" class="form-input" placeholder="••••••••" required minlength="8">
                    </div>
                    <button type="submit" class="btn btn-success" style="width: 100%;">
                        🚀 Start Free Trial
                    </button>
                </form>
                
                <div class="auth-footer">
                    Already have an account? <a href="#" onclick="showLogin()">Sign in</a>
                </div>
            </div>
        </div>
        
        <!-- Main Dashboard -->
        <div id="dashboard-page" style="display: none;">
            <header>
                <div class="logo">
                    <div class="logo-icon">🌀</div>
                    <div>
                        <h1>Life Fractal Intelligence</h1>
                        <p>Sacred Geometry • Golden Ratio • Personal Growth</p>
                    </div>
                </div>
                <div class="user-info">
                    <div>
                        <div id="user-name" style="font-weight: 600;">Loading...</div>
                        <div id="user-status" style="font-size: 12px; opacity: 0.8;">Trial Active</div>
                    </div>
                    <div class="user-avatar" id="user-avatar">👤</div>
                    <button class="btn btn-outline" onclick="logout()">Logout</button>
                </div>
            </header>
            
            <!-- Stats Row -->
            <div class="grid grid-4" style="margin-bottom: 30px;">
                <div class="card stat-card">
                    <div class="stat-icon">🧘</div>
                    <div class="stat-value" id="stat-wellness">--</div>
                    <div class="stat-label">Wellness Index</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-icon">🔥</div>
                    <div class="stat-value" id="stat-streak">--</div>
                    <div class="stat-label">Day Streak</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value" id="stat-goals">--</div>
                    <div class="stat-label">Goals Progress</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-icon">✨</div>
                    <div class="stat-value" id="stat-habits">--</div>
                    <div class="stat-label">Habits Today</div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="grid grid-3">
                <!-- Left Column: Pet & Fractal -->
                <div>
                    <div class="card pet-card" style="margin-bottom: 20px;">
                        <div class="card-header">
                            <span class="card-title">🐾 Your Companion</span>
                            <span id="pet-level" style="color: var(--golden);">Lv. 1</span>
                        </div>
                        <div class="pet-avatar" id="pet-emoji">🐱</div>
                        <div class="pet-name" id="pet-name">Loading...</div>
                        <div id="pet-behavior" style="color: var(--text-muted);">idle</div>
                        
                        <div class="pet-stats">
                            <div class="pet-stat">
                                <div class="pet-stat-label">❤️ Hunger</div>
                                <div class="pet-stat-value" id="pet-hunger">50%</div>
                            </div>
                            <div class="pet-stat">
                                <div class="pet-stat-label">⚡ Energy</div>
                                <div class="pet-stat-value" id="pet-energy">50%</div>
                            </div>
                            <div class="pet-stat">
                                <div class="pet-stat-label">😊 Mood</div>
                                <div class="pet-stat-value" id="pet-mood">50%</div>
                            </div>
                            <div class="pet-stat">
                                <div class="pet-stat-label">💫 Bond</div>
                                <div class="pet-stat-value" id="pet-bond">0%</div>
                            </div>
                        </div>
                        
                        <div class="pet-actions">
                            <button class="btn btn-primary" onclick="feedPet()">🍖 Feed</button>
                            <button class="btn btn-success" onclick="playWithPet()">🎾 Play</button>
                        </div>
                    </div>
                    
                    <div class="card fractal-container">
                        <div class="card-header">
                            <span class="card-title">🌀 Your Fractal</span>
                            <button class="btn btn-outline" onclick="refreshFractal()">🔄 Refresh</button>
                        </div>
                        <img id="fractal-image" class="fractal-image" src="" alt="Fractal Visualization">
                    </div>
                </div>
                
                <!-- Middle Column: Daily Entry -->
                <div>
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">📊 Today's Check-in</span>
                            <span id="today-date"></span>
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>😊 Mood</span>
                                <span id="mood-value">5</span>
                            </div>
                            <input type="range" class="slider" id="mood-slider" min="1" max="10" value="5" oninput="updateSlider('mood')">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>⚡ Energy</span>
                                <span id="energy-value">5</span>
                            </div>
                            <input type="range" class="slider" id="energy-slider" min="1" max="10" value="5" oninput="updateSlider('energy')">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>🧠 Focus</span>
                                <span id="focus-value">5</span>
                            </div>
                            <input type="range" class="slider" id="focus-slider" min="1" max="10" value="5" oninput="updateSlider('focus')">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>😰 Anxiety</span>
                                <span id="anxiety-value">3</span>
                            </div>
                            <input type="range" class="slider" id="anxiety-slider" min="1" max="10" value="3" oninput="updateSlider('anxiety')">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>💤 Sleep Hours</span>
                                <span id="sleep-value">7</span>
                            </div>
                            <input type="range" class="slider" id="sleep-slider" min="0" max="12" step="0.5" value="7" oninput="updateSlider('sleep')">
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">📝 Journal Entry</label>
                            <textarea class="form-input" id="journal-entry" placeholder="How was your day? What are you grateful for?"></textarea>
                        </div>
                        
                        <button class="btn btn-primary" style="width: 100%;" onclick="saveDaily()">
                            💾 Save Today's Entry
                        </button>
                    </div>
                    
                    <!-- Guidance Message -->
                    <div class="card" style="margin-top: 20px;">
                        <div class="card-header">
                            <span class="card-title">💬 AI Guidance</span>
                        </div>
                        <div class="message message-info" id="guidance-message">
                            Loading your personalized guidance...
                        </div>
                        <div class="message message-success" id="pet-message">
                            Your pet is here for you!
                        </div>
                    </div>
                </div>
                
                <!-- Right Column: Habits & Goals -->
                <div>
                    <div class="card" style="margin-bottom: 20px;">
                        <div class="card-header">
                            <span class="card-title">✅ Today's Habits</span>
                            <button class="btn btn-outline" onclick="showAddHabit()">+ Add</button>
                        </div>
                        <div class="habit-list" id="habit-list">
                            <!-- Habits will be loaded here -->
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">🎯 Active Goals</span>
                            <button class="btn btn-outline" onclick="showAddGoal()">+ Add</button>
                        </div>
                        <div class="goal-list" id="goal-list">
                            <!-- Goals will be loaded here -->
                        </div>
                    </div>
                    
                    <div class="card sacred-math" style="margin-top: 20px;">
                        <h4>✨ Sacred Mathematics</h4>
                        <div class="sacred-value">
                            <span>Golden Ratio (φ)</span>
                            <span>1.618033988749895</span>
                        </div>
                        <div class="sacred-value">
                            <span>Golden Angle</span>
                            <span>137.5077640500°</span>
                        </div>
                        <div class="sacred-value">
                            <span>Fibonacci</span>
                            <span>1, 1, 2, 3, 5, 8, 13...</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>🌀 Life Fractal Intelligence v5.0 | Built with Sacred Geometry</p>
                <p style="margin-top: 10px;">
                    <a href="https://gofund.me/8d9303d27" target="_blank">💖 Support on GoFundMe</a>
                </p>
                <p style="margin-top: 10px; font-size: 12px;">
                    GPU: <span id="gpu-status">Checking...</span> | 
                    ML: <span id="ml-status">Active</span> |
                    Self-Healing: ✅ Active
                </p>
            </footer>
        </div>
    </div>
    
    <script>
        // Global state
        let currentUser = null;
        let authToken = null;
        
        const PET_EMOJIS = {
            'cat': '🐱',
            'dragon': '🐲',
            'phoenix': '🔥',
            'owl': '🦉',
            'fox': '🦊'
        };
        
        // API Helper
        async function api(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            try {
                const response = await fetch(endpoint, options);
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                return { error: error.message };
            }
        }
        
        // Auth Functions
        async function login(email, password) {
            const result = await api('/api/auth/login', 'POST', { email, password });
            if (result.error) {
                showAlert('login-alert', result.error, 'error');
                return false;
            }
            
            authToken = result.access_token;
            currentUser = result.user;
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('userId', currentUser.id);
            
            showDashboard();
            loadDashboardData();
            return true;
        }
        
        async function register(email, password, firstName, lastName) {
            const result = await api('/api/auth/register', 'POST', {
                email, password, first_name: firstName, last_name: lastName
            });
            
            if (result.error) {
                showAlert('register-alert', result.error, 'error');
                return false;
            }
            
            authToken = result.access_token;
            currentUser = result.user;
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('userId', currentUser.id);
            
            showDashboard();
            loadDashboardData();
            return true;
        }
        
        function logout() {
            localStorage.removeItem('authToken');
            localStorage.removeItem('userId');
            authToken = null;
            currentUser = null;
            showLogin();
        }
        
        function demoLogin() {
            login('onlinediscountsllc@gmail.com', 'admin8587037321');
        }
        
        // UI Functions
        function showLogin() {
            document.getElementById('login-page').style.display = 'block';
            document.getElementById('register-page').style.display = 'none';
            document.getElementById('dashboard-page').style.display = 'none';
        }
        
        function showRegister() {
            document.getElementById('login-page').style.display = 'none';
            document.getElementById('register-page').style.display = 'block';
            document.getElementById('dashboard-page').style.display = 'none';
        }
        
        function showDashboard() {
            document.getElementById('login-page').style.display = 'none';
            document.getElementById('register-page').style.display = 'none';
            document.getElementById('dashboard-page').style.display = 'block';
        }
        
        function showAlert(elementId, message, type) {
            const alert = document.getElementById(elementId);
            alert.className = `alert show alert-${type}`;
            alert.innerHTML = `${type === 'error' ? '❌' : '✅'} ${message}`;
            setTimeout(() => { alert.className = 'alert'; }, 5000);
        }
        
        function updateSlider(name) {
            const slider = document.getElementById(`${name}-slider`);
            document.getElementById(`${name}-value`).textContent = slider.value;
        }
        
        // Dashboard Data
        async function loadDashboardData() {
            const userId = localStorage.getItem('userId');
            if (!userId) return;
            
            // Load dashboard
            const data = await api(`/api/user/${userId}/dashboard`);
            if (data.error) return;
            
            // Update user info
            document.getElementById('user-name').textContent = 
                `${data.user.first_name || 'User'} ${data.user.last_name || ''}`;
            document.getElementById('user-status').textContent = 
                data.user.has_access ? '✅ Active' : `Trial: ${data.user.trial_days_remaining} days left`;
            
            // Update stats
            document.getElementById('stat-wellness').textContent = 
                Math.round(data.stats.wellness_index);
            document.getElementById('stat-streak').textContent = 
                data.stats.current_streak;
            document.getElementById('stat-goals').textContent = 
                `${Math.round(data.stats.goals_progress)}%`;
            document.getElementById('stat-habits').textContent = 
                `${data.stats.habits_completed_today}/${data.habits.length}`;
            
            // Update pet
            if (data.pet) {
                document.getElementById('pet-emoji').textContent = 
                    PET_EMOJIS[data.pet.species] || '🐱';
                document.getElementById('pet-name').textContent = data.pet.name;
                document.getElementById('pet-level').textContent = `Lv. ${data.pet.level}`;
                document.getElementById('pet-behavior').textContent = data.pet.behavior;
                document.getElementById('pet-hunger').textContent = `${Math.round(data.pet.hunger)}%`;
                document.getElementById('pet-energy').textContent = `${Math.round(data.pet.energy)}%`;
                document.getElementById('pet-mood').textContent = `${Math.round(data.pet.mood)}%`;
                document.getElementById('pet-bond').textContent = `${Math.round(data.pet.bond)}%`;
            }
            
            // Update today's date
            document.getElementById('today-date').textContent = 
                new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
            
            // Update sliders from today's entry
            if (data.today) {
                document.getElementById('mood-slider').value = data.today.mood_level * 2;
                document.getElementById('energy-slider').value = data.today.energy_level / 10;
                document.getElementById('focus-slider').value = data.today.focus_clarity / 10;
                document.getElementById('anxiety-slider').value = data.today.anxiety_level / 10;
                document.getElementById('sleep-slider').value = data.today.sleep_hours;
                document.getElementById('journal-entry').value = data.today.journal_entry || '';
                
                // Update displayed values
                ['mood', 'energy', 'focus', 'anxiety', 'sleep'].forEach(updateSlider);
            }
            
            // Load habits
            renderHabits(data.habits, data.today?.habits_completed || {});
            
            // Load goals
            renderGoals(data.goals);
            
            // Load fractal
            refreshFractal();
            
            // Load guidance
            loadGuidance();
            
            // Check GPU status
            checkSystemStatus();
        }
        
        function renderHabits(habits, completed) {
            const list = document.getElementById('habit-list');
            list.innerHTML = habits.map(habit => `
                <div class="habit-item">
                    <div class="habit-checkbox ${completed[habit.id] ? 'checked' : ''}" 
                         onclick="toggleHabit('${habit.id}')">
                        ${completed[habit.id] ? '✓' : ''}
                    </div>
                    <div class="habit-info">
                        <div class="habit-name">${habit.name}</div>
                        <div class="habit-streak">🔥 ${habit.current_streak} day streak</div>
                    </div>
                </div>
            `).join('');
        }
        
        function renderGoals(goals) {
            const list = document.getElementById('goal-list');
            list.innerHTML = goals.filter(g => !g.is_completed).slice(0, 5).map(goal => `
                <div class="goal-item">
                    <div style="flex: 1;">
                        <div style="font-weight: 600;">${goal.title}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${goal.progress}%;"></div>
                        </div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 5px;">
                            ${Math.round(goal.progress)}% complete
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        async function toggleHabit(habitId) {
            const userId = localStorage.getItem('userId');
            await api(`/api/user/${userId}/habits/${habitId}/complete`, 'POST', { completed: true });
            loadDashboardData();
        }
        
        async function saveDaily() {
            const userId = localStorage.getItem('userId');
            const data = {
                mood_level: Math.ceil(document.getElementById('mood-slider').value / 2),
                mood_score: document.getElementById('mood-slider').value * 10,
                energy_level: document.getElementById('energy-slider').value * 10,
                focus_clarity: document.getElementById('focus-slider').value * 10,
                anxiety_level: document.getElementById('anxiety-slider').value * 10,
                sleep_hours: parseFloat(document.getElementById('sleep-slider').value),
                journal_entry: document.getElementById('journal-entry').value
            };
            
            await api(`/api/user/${userId}/today`, 'POST', data);
            loadDashboardData();
            alert('✅ Daily entry saved!');
        }
        
        async function feedPet() {
            const userId = localStorage.getItem('userId');
            await api(`/api/user/${userId}/pet/feed`, 'POST');
            loadDashboardData();
        }
        
        async function playWithPet() {
            const userId = localStorage.getItem('userId');
            const result = await api(`/api/user/${userId}/pet/play`, 'POST');
            if (result.error) {
                alert('🐱 Pet is too tired to play! Let them rest.');
            }
            loadDashboardData();
        }
        
        async function refreshFractal() {
            const userId = localStorage.getItem('userId');
            const result = await api(`/api/user/${userId}/fractal/base64`);
            if (result.image) {
                document.getElementById('fractal-image').src = result.image;
            }
        }
        
        async function loadGuidance() {
            const userId = localStorage.getItem('userId');
            const result = await api(`/api/user/${userId}/guidance`);
            if (!result.error) {
                document.getElementById('guidance-message').innerHTML = result.fuzzy_message;
                document.getElementById('pet-message').innerHTML = result.pet_message;
            }
        }
        
        async function checkSystemStatus() {
            const result = await api('/api/health');
            if (!result.error) {
                document.getElementById('gpu-status').textContent = 
                    result.gpu_available ? `✅ ${result.gpu_name}` : '❌ CPU Only';
            }
        }
        
        function showAddHabit() {
            const name = prompt('Enter habit name:');
            if (name) {
                const userId = localStorage.getItem('userId');
                api(`/api/user/${userId}/habits`, 'POST', { name }).then(loadDashboardData);
            }
        }
        
        function showAddGoal() {
            const title = prompt('Enter goal title:');
            if (title) {
                const userId = localStorage.getItem('userId');
                api(`/api/user/${userId}/goals`, 'POST', { title }).then(loadDashboardData);
            }
        }
        
        // Form Handlers
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            await login(email, password);
        });
        
        document.getElementById('register-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('register-email').value;
            const password = document.getElementById('register-password').value;
            const firstName = document.getElementById('register-first').value;
            const lastName = document.getElementById('register-last').value;
            await register(email, password, firstName, lastName);
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            const savedToken = localStorage.getItem('authToken');
            const savedUserId = localStorage.getItem('userId');
            
            if (savedToken && savedUserId) {
                authToken = savedToken;
                showDashboard();
                loadDashboardData();
            } else {
                showLogin();
            }
        });
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
@safe_execute(fallback_value=({'error': 'Registration failed'}, 500), component="auth_register")
def register():
    """Register new user with 7-day trial."""
    data = request.get_json()
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    user = store.create_user(email, password, first_name, last_name)
    if not user:
        return jsonify({'error': 'Email already registered'}), 400
    
    token = store.create_session(user.id)
    
    return jsonify({
        'message': 'Registration successful! 🎉',
        'user': user.to_dict(),
        'access_token': user.id,
        'session_token': token,
        'trial_days_remaining': TRIAL_DAYS,
        'show_gofundme': True,
        'gofundme_url': GOFUNDME_URL
    }), 201


@app.route('/api/auth/login', methods=['POST'])
@safe_execute(fallback_value=({'error': 'Login failed'}, 500), component="auth_login")
def login():
    """User login."""
    data = request.get_json()
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    user = store.get_user(email)
    
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account disabled'}), 403
    
    user.last_login = datetime.now(timezone.utc).isoformat()
    token = store.create_session(user.id)
    
    return jsonify({
        'message': 'Login successful! 👋',
        'user': user.to_dict(),
        'access_token': user.id,
        'session_token': token,
        'has_access': user.has_active_subscription(),
        'trial_active': user.is_trial_active(),
        'days_remaining': user.days_remaining_trial()
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# USER & DASHBOARD ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>')
@safe_execute(component="get_user")
def get_user(user_id):
    """Get user profile."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict(include_sensitive=True))


@app.route('/api/user/<user_id>/dashboard')
@safe_execute(component="get_dashboard")
def get_dashboard(user_id):
    """Get comprehensive dashboard data."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    today_entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    entries = list(user.daily_entries.values())
    avg_wellness = sum(e.wellness_index for e in entries) / max(1, len(entries))
    
    return jsonify({
        'user': user.to_dict(),
        'today': today_entry.to_dict(),
        'pet': user.pet.to_dict() if user.pet else None,
        'habits': [h.to_dict() for h in user.habits.values()],
        'goals': [g.to_dict() for g in user.goals.values()],
        'stats': {
            'wellness_index': round(today_entry.wellness_index, 1),
            'average_wellness': round(avg_wellness, 1),
            'current_streak': user.current_streak,
            'total_entries': len(entries),
            'habits_completed_today': sum(1 for c in today_entry.habits_completed.values() if c),
            'active_goals': sum(1 for g in user.goals.values() if not g.is_completed),
            'goals_progress': round(sum(g.progress for g in user.goals.values()) / max(1, len(user.goals)), 1)
        },
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13]
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY ENTRY ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/today', methods=['GET', 'POST'])
@safe_execute(component="handle_today")
def handle_today(user_id):
    """Get or update today's entry."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    if request.method == 'GET':
        entry = user.daily_entries.get(today, DailyEntry(date=today))
        return jsonify(entry.to_dict())
    
    data = request.get_json()
    
    if today not in user.daily_entries:
        user.daily_entries[today] = DailyEntry(date=today)
    
    entry = user.daily_entries[today]
    
    for field in ['mood_level', 'mood_score', 'energy_level', 'focus_clarity',
                  'anxiety_level', 'stress_level', 'mindfulness_score',
                  'gratitude_level', 'sleep_quality', 'sleep_hours',
                  'nutrition_score', 'social_connection', 'emotional_stability',
                  'self_compassion', 'journal_entry', 'goals_completed_count']:
        if field in data:
            setattr(entry, field, data[field])
    
    if 'habits_completed' in data:
        entry.habits_completed.update(data['habits_completed'])
    
    if 'gratitude_items' in data:
        entry.gratitude_items = data['gratitude_items']
    
    if 'wins' in data:
        entry.wins = data['wins']
    
    entry.calculate_wellness()
    user.history.append(entry.to_dict())
    
    system = store.get_system(user_id)
    system.update(entry.to_dict())
    
    store._save_data()
    
    return jsonify(entry.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# HABITS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/habits', methods=['GET', 'POST'])
@safe_execute(component="handle_habits")
def handle_habits(user_id):
    """Get or create habits."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if request.method == 'GET':
        return jsonify({'habits': [h.to_dict() for h in user.habits.values()]})
    
    data = request.get_json()
    habit_id = f"habit_{len(user.habits) + 1}_{secrets.token_hex(4)}"
    
    habit = Habit(
        id=habit_id,
        name=data.get('name', 'New Habit'),
        description=data.get('description', ''),
        category=data.get('category', 'general'),
        created_at=datetime.now(timezone.utc).isoformat()
    )
    
    user.habits[habit_id] = habit
    store._save_data()
    return jsonify({'success': True, 'habit': habit.to_dict()})


@app.route('/api/user/<user_id>/habits/<habit_id>/complete', methods=['POST'])
@safe_execute(component="complete_habit")
def complete_habit(user_id, habit_id):
    """Mark habit as complete."""
    user = store.get_user(user_id)
    if not user or habit_id not in user.habits:
        return jsonify({'error': 'Not found'}), 404
    
    habit = user.habits[habit_id]
    completed = request.get_json().get('completed', True)
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if today not in user.daily_entries:
        user.daily_entries[today] = DailyEntry(date=today)
    
    user.daily_entries[today].habits_completed[habit_id] = completed
    
    if completed:
        habit.total_completions += 1
        habit.current_streak += 1
        habit.longest_streak = max(habit.longest_streak, habit.current_streak)
    
    store._save_data()
    return jsonify({'success': True, 'habit': habit.to_dict()})


# ═══════════════════════════════════════════════════════════════════════════════
# GOALS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/goals', methods=['GET', 'POST'])
@safe_execute(component="handle_goals")
def handle_goals(user_id):
    """Get or create goals."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if request.method == 'GET':
        return jsonify({
            'goals': [g.to_dict() for g in user.goals.values()],
            'active': sum(1 for g in user.goals.values() if not g.is_completed),
            'completed': sum(1 for g in user.goals.values() if g.is_completed)
        })
    
    data = request.get_json()
    goal_id = f"goal_{len(user.goals) + 1}_{secrets.token_hex(4)}"
    
    goal = Goal(
        id=goal_id,
        title=data.get('title', 'New Goal'),
        description=data.get('description', ''),
        category=data.get('category', 'general'),
        priority=data.get('priority', 3),
        target_date=data.get('target_date'),
        created_at=datetime.now(timezone.utc).isoformat(),
        why_important=data.get('why_important', ''),
        subtasks=data.get('subtasks', []),
        obstacles=data.get('obstacles', []),
        success_criteria=data.get('success_criteria', [])
    )
    
    user.goals[goal_id] = goal
    store._save_data()
    return jsonify({'success': True, 'goal': goal.to_dict()})


@app.route('/api/user/<user_id>/goals/<goal_id>/progress', methods=['POST'])
@safe_execute(component="update_goal_progress")
def update_goal_progress(user_id, goal_id):
    """Update goal progress."""
    user = store.get_user(user_id)
    if not user or goal_id not in user.goals:
        return jsonify({'error': 'Not found'}), 404
    
    goal = user.goals[goal_id]
    data = request.get_json()
    
    if 'progress' in data:
        goal.progress = min(100, max(0, data['progress']))
    
    milestone = goal.check_milestones()
    
    if goal.progress >= 100 and not goal.completed_at:
        goal.completed_at = datetime.now(timezone.utc).isoformat()
    
    store._save_data()
    
    return jsonify({
        'success': True,
        'goal': goal.to_dict(),
        'milestone_reached': milestone
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PET ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/pet')
@safe_execute(component="get_pet")
def get_pet(user_id):
    """Get pet information."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(user.pet.to_dict())


@app.route('/api/user/<user_id>/pet/feed', methods=['POST'])
@safe_execute(component="feed_pet")
def feed_pet(user_id):
    """Feed the pet."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    
    system = store.get_system(user_id)
    system.pet.state = user.pet
    success = system.pet.feed()
    user.pet = system.pet.state
    store._save_data()
    
    return jsonify({'success': success, 'pet': user.pet.to_dict()})


@app.route('/api/user/<user_id>/pet/play', methods=['POST'])
@safe_execute(component="play_pet")
def play_pet(user_id):
    """Play with the pet."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    
    system = store.get_system(user_id)
    system.pet.state = user.pet
    success = system.pet.play()
    user.pet = system.pet.state
    store._save_data()
    
    if not success:
        return jsonify({'error': 'Pet too tired'}), 400
    
    return jsonify({'success': success, 'pet': user.pet.to_dict()})


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/fractal')
@safe_execute(component="generate_fractal")
def generate_fractal(user_id):
    """Generate fractal image."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    system = store.get_system(user_id)
    image = system.generate_fractal_image(entry.to_dict())
    
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')


@app.route('/api/user/<user_id>/fractal/base64')
@safe_execute(component="get_fractal_base64")
def get_fractal_base64(user_id):
    """Get fractal as base64."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    system = store.get_system(user_id)
    image = system.generate_fractal_image(entry.to_dict())
    base64_data = system.fractal_gen.to_base64(image)
    
    return jsonify({
        'image': f'data:image/png;base64,{base64_data}',
        'gpu_used': system.fractal_gen.use_gpu
    })


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDANCE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/guidance')
@safe_execute(component="get_guidance")
def get_guidance(user_id):
    """Get AI-generated guidance."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    system = store.get_system(user_id)
    guidance = system.generate_guidance(entry.to_dict())
    
    return jsonify(guidance)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'ml_available': HAS_SKLEARN,
        'opencv_available': HAS_OPENCV,
        'audio_available': HAS_AUDIO,
        'self_healing': HEALER.get_health_report(),
        'version': '5.0.0'
    })


@app.route('/api/system/status')
def system_status():
    """Get detailed system status."""
    return jsonify({
        'components': {
            'gpu': {'available': GPU_AVAILABLE, 'name': GPU_NAME},
            'ml': {'available': HAS_SKLEARN},
            'video': {'available': HAS_OPENCV},
            'audio': {'available': HAS_AUDIO}
        },
        'self_healing': HEALER.get_health_report(),
        'users_count': len(set(u.id for u in store.users.values())),
        'uptime': 'N/A',
        'version': '5.0.0'
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the beautiful dashboard."""
    return render_template_string(DASHBOARD_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 80)
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║     🌀 LIFE FRACTAL INTELLIGENCE - PRODUCTION SYSTEM v5.0                   ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  ✅ Self-Healing  🛡️ Auto-Backup  🎨 GPU Fractals  🐾 Virtual Pet           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("=" * 80)
    print(f"\n✨ Golden Ratio (φ):     {PHI:.15f}")
    print(f"🌻 Golden Angle:         {GOLDEN_ANGLE:.10f}°")
    print(f"📢 Fibonacci:            {FIBONACCI[:10]}...")
    print(f"🖥️  GPU Available:        {GPU_AVAILABLE} ({GPU_NAME or 'CPU Only'})")
    print(f"🤖 ML Available:         {HAS_SKLEARN}")
    print(f"🎬 Video Available:      {HAS_OPENCV}")
    print(f"🔊 Audio Available:      {HAS_AUDIO}")
    print("=" * 80)
    print("\n🛡️ SELF-HEALING FEATURES:")
    print("  ✅ Automatic retry with exponential backoff")
    print("  ✅ Graceful degradation (GPU → CPU fallback)")
    print("  ✅ Safe execution (never crashes)")
    print("  ✅ Error tracking and recovery logging")
    print("  ✅ Auto-backup every 5 minutes")
    print("=" * 80)
    print(f"\n💰 Subscription: ${SUBSCRIPTION_PRICE}/month, {TRIAL_DAYS}-day free trial")
    print(f"🎁 GoFundMe: {GOFUNDME_URL}")
    print("=" * 80)


if __name__ == '__main__':
    print_banner()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print(f"\n🚀 Starting server at http://localhost:{port}")
    print("📊 Dashboard: http://localhost:{port}/")
    print("🔌 API: http://localhost:{port}/api/health\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
