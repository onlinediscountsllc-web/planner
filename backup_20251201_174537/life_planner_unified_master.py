"""
ðŸŒ€ LIFE FRACTAL INTELLIGENCE - UNIFIED MASTER APPLICATION
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
Complete life planning system with:
- Full authentication (login/register)
- Virtual pet system (5 species)
- GPU-accelerated fractal visualization
- Sacred geometry overlays
- Fibonacci music generation
- Daily/Weekly/Monthly/Yearly views
- Goal & habit tracking
- Journal with sentiment analysis
- Decision tree predictions
- Fuzzy logic guidance
- Stripe payment integration ($20/month, 7-day trial)

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Enhanced features import
try:
    from life_fractal_enhanced_implementation import (
        EmotionalPetAI,
        FractalTimeCalendar,
        FibonacciTaskScheduler,
        ExecutiveFunctionSupport,
        AutismSafeColors,
        AphantasiaSupport,
        PrivacyPreservingML
    )
    ENHANCED_FEATURES_AVAILABLE = True
    print("Enhanced features loaded successfully")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"Enhanced features not available: {e}")


# Data processing
import numpy as np
from PIL import Image

# ML
try:
    from sklearn.tree import DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOGGING SETUP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SACRED MATHEMATICS CONSTANTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
PLATONIC_SOLIDS = {
    'tetrahedron': {'faces': 4, 'vertices': 4, 'edges': 6},
    'cube': {'faces': 6, 'vertices': 8, 'edges': 12},
    'octahedron': {'faces': 8, 'vertices': 6, 'edges': 12},
    'dodecahedron': {'faces': 12, 'vertices': 20, 'edges': 30},
    'icosahedron': {'faces': 20, 'vertices': 12, 'edges': 30}
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ANCIENT MATHEMATICS UTILITIES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AncientMathUtil:
    """Classical mathematical sequences and sacred geometry utilities."""
    
    @staticmethod
    def golden_ratio() -> float:
        return PHI
    
    @staticmethod
    def fibonacci_sequence(n: int) -> List[int]:
        if n <= 0:
            return []
        return FIBONACCI[:n]
    
    @staticmethod
    def logistic_map(r: float, x: float) -> float:
        """Single step of logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
        return r * x * (1 - x)
    
    @staticmethod
    def logistic_map_series(r: float, x0: float, n: int) -> List[float]:
        """Generate series using logistic map for chaos theory applications."""
        series = []
        x = x0
        for _ in range(n):
            series.append(x)
            x = r * x * (1 - x)
        return series
    
    @staticmethod
    def archimedes_spiral(theta: float, a: float = 1.0, b: float = 0.5) -> Tuple[float, float]:
        """Archimedes spiral: r = a + b*theta (225 BC)"""
        r = a + b * theta
        return r * math.cos(theta), r * math.sin(theta)
    
    @staticmethod
    def golden_spiral_point(index: int) -> Tuple[float, float]:
        """Get point on golden spiral using Fibonacci positioning."""
        theta = index * GOLDEN_ANGLE_RAD
        r = math.sqrt(index)
        return r * math.cos(theta), r * math.sin(theta)
    
    @staticmethod
    def islamic_star_pattern(n: int, scale: float = 1.0) -> List[Tuple[float, float]]:
        """Generate Islamic geometric star pattern (8th-15th century)."""
        points = []
        for i in range(n):
            angle = i * 2 * math.pi / n
            r_outer = scale
            r_inner = scale * PHI_INVERSE
            angle_inner = angle + math.pi / n
            points.append((r_outer * math.cos(angle), r_outer * math.sin(angle)))
            points.append((r_inner * math.cos(angle_inner), r_inner * math.sin(angle_inner)))
        return points
    
    @staticmethod
    def pythagorean_means(values: List[float]) -> Dict[str, float]:
        """Calculate the three Pythagorean means (6th century BC)."""
        arr = np.array(values)
        arr_positive = arr[arr > 0]
        return {
            'arithmetic': float(np.mean(arr)),
            'geometric': float(np.power(np.prod(arr_positive), 1.0 / len(arr_positive))) if len(arr_positive) > 0 else 0,
            'harmonic': float(len(arr_positive) / np.sum(1.0 / arr_positive)) if len(arr_positive) > 0 else 0
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA MODELS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
    
    # Mood and mental health (0-100)
    mood_level: int = 3  # 1-5 scale
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
    
    # Habits
    habits_completed: Dict[str, bool] = field(default_factory=dict)
    
    # Journal
    journal_entry: str = ""
    journal_sentiment: float = 0.5
    
    # Goals
    goals_progressed: Dict[str, float] = field(default_factory=dict)
    goals_completed_count: int = 0
    
    # Period tracking
    period: str = "daily"
    
    # Computed
    wellness_index: float = 0.0
    predicted_mood: float = 0.0
    
    def __post_init__(self):
        self.calculate_wellness()
    
    def calculate_wellness(self):
        """Calculate overall wellness index using Fibonacci weighting."""
        weights = [FIBONACCI[i+3] for i in range(8)]  # [2, 3, 5, 8, 13, 21, 34, 55]
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
    velocity: float = 0.0  # Progress per day
    
    # Fibonacci milestones
    milestones: List[int] = field(default_factory=lambda: [8, 13, 21, 34, 55, 89, 100])
    milestones_reached: List[int] = field(default_factory=list)
    
    @property
    def is_completed(self) -> bool:
        return self.progress >= 100 or self.completed_at is not None
    
    def check_milestones(self) -> Optional[int]:
        """Check if a new milestone was reached."""
        for milestone in self.milestones:
            if self.progress >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.append(milestone)
                return milestone
        return None
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'priority': self.priority,
            'progress': self.progress,
            'target_date': self.target_date,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'is_completed': self.is_completed,
            'velocity': self.velocity,
            'milestones': self.milestones,
            'milestones_reached': self.milestones_reached
        }


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
    
    # Subscription
    subscription_status: str = "trial"  # trial, active, cancelled, expired
    trial_start_date: str = ""
    trial_end_date: str = ""
    stripe_customer_id: Optional[str] = None
    
    # Data
    pet: Optional[PetState] = None
    daily_entries: Dict[str, DailyEntry] = field(default_factory=dict)
    habits: Dict[str, Habit] = field(default_factory=dict)
    goals: Dict[str, Goal] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    
    # Settings
    fractal_type: str = "hybrid"
    show_flower_of_life: bool = True
    show_metatron_cube: bool = True
    show_golden_spiral: bool = True
    animation_speed: float = 1.0
    
    # Timestamps
    created_at: str = ""
    last_login: str = ""
    
    # Stats
    current_streak: int = 0
    longest_streak: int = 0
    
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
    
    def is_trial_active(self) -> bool:
        if not self.trial_end_date:
            return False
        end = datetime.fromisoformat(self.trial_end_date.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < end and self.subscription_status == 'trial'
    
    def has_active_subscription(self) -> bool:
        return self.is_trial_active() or self.subscription_status == 'active'
    
    def days_remaining_trial(self) -> int:
        if not self.trial_end_date:
            return 0
        end = datetime.fromisoformat(self.trial_end_date.replace('Z', '+00:00'))
        delta = end - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        data = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'subscription_status': self.subscription_status,
            'trial_days_remaining': self.days_remaining_trial(),
            'created_at': self.created_at,
            'last_login': self.last_login,
            'current_streak': self.current_streak,
            'longest_streak': self.longest_streak
        }
        if include_sensitive:
            data['is_admin'] = self.is_admin
            data['email_verified'] = self.email_verified
        return data


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FUZZY LOGIC ENGINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class FuzzyLogicEngine:
    """Fuzzy logic for generating supportive messages based on mood and stress."""
    
    def __init__(self):
        self.messages = {
            ('low_stress', 'high_mood'): "You're doing great! Keep nurturing this positive energy.",
            ('low_stress', 'medium_mood'): "You're in a good place. Small joys can lift you higher.",
            ('low_stress', 'low_mood'): "Even on quieter days, you're managing well. Be gentle with yourself.",
            ('medium_stress', 'high_mood'): "Your resilience is shining through! Take breaks when needed.",
            ('medium_stress', 'medium_mood'): "Balance is key. You're navigating well through challenges.",
            ('medium_stress', 'low_mood'): "It's okay to feel this way. Consider a short mindful pause.",
            ('high_stress', 'high_mood'): "Your positivity is admirable! Don't forget to rest.",
            ('high_stress', 'medium_mood'): "You're handling a lot. Prioritize what matters most right now.",
            ('high_stress', 'low_mood'): "These feelings are valid. Reach out for support if needed."
        }
    
    def _fuzzy_membership(self, value: float, low: float, high: float) -> str:
        """Determine fuzzy membership category."""
        if value <= low:
            return 'low'
        elif value >= high:
            return 'high'
        return 'medium'
    
    def infer(self, stress: float, mood: float) -> str:
        """Generate supportive message based on fuzzy inference."""
        stress_level = self._fuzzy_membership(stress, 30, 70)
        mood_level = self._fuzzy_membership(mood, 30, 70)
        
        key = (f'{stress_level}_stress', f'{mood_level}_mood')
        return self.messages.get(key, "Take a moment to breathe and reflect.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DECISION TREE PREDICTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class MoodPredictor:
    """Decision tree-based mood prediction."""
    
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42) if HAS_SKLEARN else None
        self.trained = False
    
    def train(self, history: List[Dict]) -> bool:
        """Train on user history."""
        if not HAS_SKLEARN or not history or len(history) < 3:
            return False
        
        try:
            X = []
            y = []
            for i, record in enumerate(history[:-1]):
                features = [
                    float(record.get('stress_level', 50)),
                    float(record.get('mood_score', 50)),
                    float(record.get('energy_level', 50)),
                    float(record.get('goals_completed_count', 0)),
                    float(record.get('sleep_hours', 7)),
                    float(record.get('sleep_quality', 50))
                ]
                # Target is next day's mood
                target = float(history[i+1].get('mood_score', 50))
                X.append(features)
                y.append(target)
            
            if len(X) >= 3:
                self.model.fit(X, y)
                self.trained = True
                return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
        return False
    
    def predict(self, current_state: Dict) -> float:
        """Predict next mood."""
        if not self.trained or not HAS_SKLEARN:
            return float(current_state.get('mood_score', 50))
        
        try:
            features = [[
                float(current_state.get('stress_level', 50)),
                float(current_state.get('mood_score', 50)),
                float(current_state.get('energy_level', 50)),
                float(current_state.get('goals_completed_count', 0)),
                float(current_state.get('sleep_hours', 7)),
                float(current_state.get('sleep_quality', 50))
            ]]
            return float(self.model.predict(features)[0])
        except:
            return float(current_state.get('mood_score', 50))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# VIRTUAL PET SYSTEM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class VirtualPet:
    """Virtual pet with behavior and evolution mechanics."""
    
    SPECIES_TRAITS = {
        'cat': {'energy_decay': 1.2, 'mood_sensitivity': 1.0, 'growth_rate': 1.0},
        'dragon': {'energy_decay': 0.8, 'mood_sensitivity': 1.3, 'growth_rate': 1.2},
        'phoenix': {'energy_decay': 1.0, 'mood_sensitivity': 0.8, 'growth_rate': 1.5},
        'owl': {'energy_decay': 0.9, 'mood_sensitivity': 1.1, 'growth_rate': 0.9},
        'fox': {'energy_decay': 1.1, 'mood_sensitivity': 1.2, 'growth_rate': 1.1}
    }
    
    BEHAVIORS = ['idle', 'happy', 'playful', 'tired', 'hungry', 'sad', 'excited', 'sleeping']
    
    def __init__(self, state: PetState):
        self.state = state
        self.traits = self.SPECIES_TRAITS.get(state.species, self.SPECIES_TRAITS['cat'])
    
    def update_from_user_data(self, user_data: Dict):
        """Update pet state based on user activity data."""
        # Energy affected by user's sleep
        sleep_quality = user_data.get('sleep_quality', 50)
        self.state.energy = min(100, self.state.energy + (sleep_quality - 50) * 0.2)
        
        # Mood affected by user's mood
        user_mood = user_data.get('mood_score', 50)
        mood_delta = (user_mood - 50) * 0.3 * self.traits['mood_sensitivity']
        self.state.mood = max(0, min(100, self.state.mood + mood_delta))
        
        # Stress inverse to user's mindfulness
        mindfulness = user_data.get('mindfulness_score', 50)
        self.state.stress = max(0, min(100, 100 - mindfulness * 0.8))
        
        # Growth from goals completed
        goals = user_data.get('goals_completed_count', 0)
        self.state.growth = min(100, self.state.growth + goals * 2 * self.traits['growth_rate'])
        
        # Experience and leveling
        xp_gain = int(goals * 10 + (user_mood / 10))
        self.state.experience += xp_gain
        
        # Level up check (Fibonacci-based XP thresholds)
        xp_for_next = FIBONACCI[min(self.state.level + 5, len(FIBONACCI)-1)] * 10
        if self.state.experience >= xp_for_next:
            self.state.level += 1
            self.state.experience -= xp_for_next
            if self.state.level % 5 == 0:
                self.state.evolution_stage = min(3, self.state.evolution_stage + 1)
        
        # Natural decay
        self.state.hunger = min(100, self.state.hunger + 2 * self.traits['energy_decay'])
        self.state.energy = max(0, self.state.energy - 1 * self.traits['energy_decay'])
        
        # Determine behavior
        self._update_behavior()
    
    def _update_behavior(self):
        """Determine current behavior based on state."""
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
        """Feed the pet."""
        self.state.hunger = max(0, self.state.hunger - 30)
        self.state.mood = min(100, self.state.mood + 5)
        self.state.last_fed = datetime.now(timezone.utc).isoformat()
        self._update_behavior()
        return True
    
    def play(self) -> bool:
        """Play with the pet."""
        if self.state.energy < 20:
            return False
        self.state.energy = max(0, self.state.energy - 15)
        self.state.mood = min(100, self.state.mood + 15)
        self.state.bond = min(100, self.state.bond + 3)
        self.state.last_played = datetime.now(timezone.utc).isoformat()
        self._update_behavior()
        return True
    
    def rest(self):
        """Let pet rest."""
        self.state.energy = min(100, self.state.energy + 25)
        self.state.stress = max(0, self.state.stress - 10)
        self._update_behavior()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GPU-ACCELERATED FRACTAL GENERATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class FractalGenerator:
    """GPU-accelerated fractal generation with CPU fallback."""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.use_gpu = GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {GPU_NAME}")
        else:
            logger.info("Using CPU for fractal generation")
    
    def generate_mandelbrot(self, max_iter: int = 256, zoom: float = 1.0,
                           center: Tuple[float, float] = (-0.5, 0)) -> np.ndarray:
        """Generate Mandelbrot set."""
        if self.use_gpu and torch is not None:
            return self._mandelbrot_gpu(max_iter, zoom, center)
        return self._mandelbrot_cpu(max_iter, zoom, center)
    
    def _mandelbrot_gpu(self, max_iter: int, zoom: float, center: Tuple[float, float]) -> np.ndarray:
        try:
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
        except Exception as e:
            logger.error(f"GPU generation failed: {e}")
            return self._mandelbrot_cpu(max_iter, zoom, center)
    
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
    
    def generate_julia(self, c_real: float = -0.7, c_imag: float = 0.27015,
                      max_iter: int = 256, zoom: float = 1.0) -> np.ndarray:
        """Generate Julia set."""
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
    
    def apply_coloring(self, iterations: np.ndarray, max_iter: int,
                      hue_base: float = 0.6, hue_range: float = 0.3,
                      saturation: float = 0.8) -> np.ndarray:
        """Apply color mapping to iteration data."""
        normalized = iterations / max_iter
        
        # HSV to RGB
        h = (hue_base + normalized * hue_range) % 1.0
        s = np.full_like(normalized, saturation)
        v = np.sqrt(normalized) * 0.9 + 0.1
        
        # Inside set is dark
        inside = normalized >= 0.99
        v[inside] = 0.05
        s[inside] = 0
        
        rgb = np.zeros((*iterations.shape, 3), dtype=np.uint8)
        
        # HSV to RGB conversion
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
    
    def create_visualization(self, user_data: Dict, pet_state: Optional[Dict] = None) -> Image.Image:
        """Create complete visualization based on user data."""
        # Map user data to fractal parameters
        mood = user_data.get('mood_score', 50)
        energy = user_data.get('energy_level', 50)
        anxiety = user_data.get('anxiety_level', 30)
        mindfulness = user_data.get('mindfulness_score', 50)
        wellness = user_data.get('wellness_index', 50)
        
        # Determine fractal type based on wellness
        if wellness < 30:
            iterations = self.generate_julia(-0.8, 0.156, max_iter=200)
            hue_base = 0.7  # Blue tones
        elif wellness < 60:
            iterations = self.generate_mandelbrot(max_iter=256, zoom=1.5)
            hue_base = 0.5 + (mood - 50) / 200  # Cyan to green
        else:
            # Hybrid
            m = self.generate_mandelbrot(max_iter=256, zoom=2.0)
            j = self.generate_julia(-0.7 + (mood-50)/200, 0.27, max_iter=200)
            iterations = m * 0.5 + j * 0.5
            hue_base = 0.3 + (mood / 200)  # Yellow to cyan
        
        # Color based on mood
        hue_range = 0.3 + (energy / 200)
        saturation = 0.5 + (mindfulness / 200)
        
        rgb = self.apply_coloring(iterations, 256, hue_base, hue_range, saturation)
        
        return Image.fromarray(rgb, 'RGB')
    
    def to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert image to base64."""
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LIFE PLANNING SYSTEM (Main Orchestrator)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LifePlanningSystem:
    """Main orchestrator for all life planning features."""
    
    def __init__(self, pet_species: str = "cat"):
        self.fractal_gen = FractalGenerator(512, 512)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.predictor = MoodPredictor()
        self.pet = VirtualPet(PetState(species=pet_species))
        self.history: List[Dict] = []
    
    def update(self, user_data: Dict):
        """Update system with new user data."""
        # Update pet
        self.pet.update_from_user_data(user_data)
        
        # Store in history
        record = {**user_data, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.history.append(record)
        
        # Train predictor when enough data
        if len(self.history) >= 5:
            self.predictor.train(self.history)
    
    def generate_guidance(self, current_state: Dict) -> Dict[str, Any]:
        """Generate guidance messages."""
        # Predict next mood
        predicted_mood = self.predictor.predict(current_state)
        
        # Fuzzy logic message
        stress = current_state.get('stress_level', 50)
        mood = current_state.get('mood_score', 50)
        fuzzy_message = self.fuzzy_engine.infer(stress, mood)
        
        # Pet message
        pet_behavior = self.pet.state.behavior
        pet_messages = {
            'happy': f"{self.pet.state.name} is wagging happily! Your positivity is contagious!",
            'playful': f"{self.pet.state.name} wants to celebrate your progress!",
            'excited': f"{self.pet.state.name} is absolutely thrilled! Keep up the great work!",
            'tired': f"{self.pet.state.name} is resting. Maybe you need rest too?",
            'hungry': f"{self.pet.state.name} is hungry. Have you eaten well today?",
            'sad': f"{self.pet.state.name} senses you might be down. It's here for you.",
            'idle': f"{self.pet.state.name} is keeping you company.",
            'sleeping': f"{self.pet.state.name} is catching some Z's. Rest is important!"
        }
        pet_message = pet_messages.get(pet_behavior, f"{self.pet.state.name} is with you.")
        
        return {
            'predicted_mood': round(predicted_mood, 1),
            'fuzzy_message': fuzzy_message,
            'pet_message': pet_message,
            'pet_state': self.pet.state.to_dict(),
            'combined_message': f"{fuzzy_message} {pet_message} (Predicted mood: {predicted_mood:.0f}/100)"
        }
    
    def generate_fractal_image(self, user_data: Dict) -> Image.Image:
        """Generate visualization based on user data."""
        return self.fractal_gen.create_visualization(user_data, self.pet.state.to_dict())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA STORE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DataStore:
    """In-memory data store (replace with database in production)."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.systems: Dict[str, LifePlanningSystem] = {}
        self._init_admin()
    
    def _init_admin(self):
        """Create admin user."""
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
        self.users[admin_id] = admin
        self.users[admin.email] = admin
    
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
        
        # Add demo data
        self._add_demo_data(user)
        
        self.users[user_id] = user
        self.users[email.lower()] = user
        
        return user
    
    def _add_demo_data(self, user: User):
        """Add demo habits and goals."""
        now = datetime.now(timezone.utc)
        
        # Demo habits
        habits = [
            ("Morning Meditation", "wellness", 12),
            ("Exercise 30 min", "health", 7),
            ("Read 20 pages", "growth", 5),
            ("Journal Entry", "wellness", 3),
            ("Drink 8 glasses water", "health", 14),
            ("Gratitude Practice", "wellness", 9)
        ]
        
        for i, (name, category, streak) in enumerate(habits):
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
        goals = [
            ("Complete Project Alpha", "work", 1, 75),
            ("Learn Meditation Course", "wellness", 2, 40),
            ("Read 12 Books This Year", "growth", 3, 83)
        ]
        
        for i, (title, category, priority, progress) in enumerate(goals):
            goal = Goal(
                id=f"goal_{i+1}",
                title=title,
                category=category,
                priority=priority,
                progress=progress,
                target_date=(now + timedelta(days=30 + i*30)).isoformat()[:10],
                created_at=(now - timedelta(days=60)).isoformat()
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
                goals_completed_count=i % 3
            )
            entry.calculate_wellness()
            user.daily_entries[date] = entry
            user.history.append(entry.to_dict())
    
    def get_user(self, identifier: str) -> Optional[User]:
        """Get user by ID or email."""
        return self.users.get(identifier) or self.users.get(identifier.lower())
    
    def get_system(self, user_id: str) -> LifePlanningSystem:
        """Get or create life planning system for user."""
        if user_id not in self.systems:
            user = self.users.get(user_id)
            species = user.pet.species if user and user.pet else 'cat'
            self.systems[user_id] = LifePlanningSystem(species)
        return self.systems[user_id]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FLASK APPLICATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'life-fractal-secret-key-2025')
CORS(app)

store = DataStore()

# Configuration
SUBSCRIPTION_PRICE = 20.00
TRIAL_DAYS = 7
GOFUNDME_URL = 'https://gofund.me/8d9303d27'


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AUTH ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with 7-day trial."""
    try:
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
        
        return jsonify({
            'message': 'Registration successful',
            'user': user.to_dict(),
            'access_token': user.id,
            'trial_days_remaining': TRIAL_DAYS,
            'show_gofundme': True,
            'gofundme_url': GOFUNDME_URL
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login."""
    try:
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
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': user.id,
            'has_access': user.has_active_subscription(),
            'trial_active': user.is_trial_active(),
            'days_remaining': user.days_remaining_trial()
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# USER ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>')
def get_user(user_id):
    """Get user profile."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict(include_sensitive=True))


@app.route('/api/user/<user_id>/dashboard')
def get_dashboard(user_id):
    """Get comprehensive dashboard data."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    today_entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    # Calculate stats
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DAILY ENTRY ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/today', methods=['GET', 'POST'])
def handle_today(user_id):
    """Get or update today's entry."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    if request.method == 'GET':
        entry = user.daily_entries.get(today, DailyEntry(date=today))
        return jsonify(entry.to_dict())
    
    # POST - update
    data = request.get_json()
    
    if today not in user.daily_entries:
        user.daily_entries[today] = DailyEntry(date=today)
    
    entry = user.daily_entries[today]
    
    # Update fields
    for field in ['mood_level', 'mood_score', 'energy_level', 'focus_clarity',
                  'anxiety_level', 'stress_level', 'mindfulness_score',
                  'gratitude_level', 'sleep_quality', 'sleep_hours',
                  'nutrition_score', 'social_connection', 'emotional_stability',
                  'self_compassion', 'journal_entry', 'goals_completed_count']:
        if field in data:
            setattr(entry, field, data[field])
    
    if 'habits_completed' in data:
        entry.habits_completed.update(data['habits_completed'])
    
    entry.calculate_wellness()
    
    # Update history
    user.history.append(entry.to_dict())
    
    # Update life planning system
    system = store.get_system(user_id)
    system.update(entry.to_dict())
    
    return jsonify(entry.to_dict())


@app.route('/api/user/<user_id>/entries')
def get_entries(user_id):
    """Get entries for date range."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    view = request.args.get('view', 'week')
    now = datetime.now(timezone.utc)
    
    if view == 'week':
        start = now - timedelta(days=7)
    elif view == 'month':
        start = now - timedelta(days=30)
    elif view == 'year':
        start = now - timedelta(days=365)
    else:
        start = now - timedelta(days=7)
    
    entries = [
        e.to_dict() for e in user.daily_entries.values()
        if datetime.strptime(e.date, '%Y-%m-%d') >= start.replace(tzinfo=None)
    ]
    
    return jsonify({
        'entries': sorted(entries, key=lambda x: x['date'], reverse=True),
        'count': len(entries)
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HABITS ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/habits', methods=['GET', 'POST'])
def handle_habits(user_id):
    """Get or create habits."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if request.method == 'GET':
        return jsonify({'habits': [h.to_dict() for h in user.habits.values()]})
    
    # POST - create
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
    return jsonify({'success': True, 'habit': habit.to_dict()})


@app.route('/api/user/<user_id>/habits/<habit_id>/complete', methods=['POST'])
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
    
    return jsonify({'success': True, 'habit': habit.to_dict()})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GOALS ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/goals', methods=['GET', 'POST'])
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
    
    # POST - create
    data = request.get_json()
    goal_id = f"goal_{len(user.goals) + 1}_{secrets.token_hex(4)}"
    
    goal = Goal(
        id=goal_id,
        title=data.get('title', 'New Goal'),
        description=data.get('description', ''),
        category=data.get('category', 'general'),
        priority=data.get('priority', 3),
        target_date=data.get('target_date'),
        created_at=datetime.now(timezone.utc).isoformat()
    )
    
    user.goals[goal_id] = goal
    return jsonify({'success': True, 'goal': goal.to_dict()})


@app.route('/api/user/<user_id>/goals/<goal_id>/progress', methods=['POST'])
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
    
    return jsonify({
        'success': True,
        'goal': goal.to_dict(),
        'milestone_reached': milestone
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PET ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/pet')
def get_pet(user_id):
    """Get pet information."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(user.pet.to_dict())


@app.route('/api/user/<user_id>/pet/feed', methods=['POST'])
def feed_pet(user_id):
    """Feed the pet."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    
    system = store.get_system(user_id)
    system.pet.state = user.pet
    success = system.pet.feed()
    user.pet = system.pet.state
    
    return jsonify({'success': success, 'pet': user.pet.to_dict()})


@app.route('/api/user/<user_id>/pet/play', methods=['POST'])
def play_pet(user_id):
    """Play with the pet."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    
    system = store.get_system(user_id)
    system.pet.state = user.pet
    success = system.pet.play()
    user.pet = system.pet.state
    
    if not success:
        return jsonify({'error': 'Pet too tired'}), 400
    
    return jsonify({'success': success, 'pet': user.pet.to_dict()})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# VISUALIZATION ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/visualization')
def get_visualization(user_id):
    """Get comprehensive 3D visualization data."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    # Map to fractal parameters
    wellness = entry.wellness_index
    
    if wellness < 30:
        fractal_type = 'julia'
    elif wellness < 50:
        fractal_type = 'mandelbrot'
    elif wellness < 70:
        fractal_type = 'hybrid'
    else:
        fractal_type = 'sacred'
    
    hue_base = 180 + (entry.mood_level - 3) * 30
    anim_speed = 0.5 + (entry.energy_level / 100) * 1.5
    chaos = 3.5 + (entry.anxiety_level / 100) * 0.5
    
    # Wellness metrics positioned on golden spiral
    wellness_points = []
    metrics = [
        ('Mood', entry.mood_score, 200, '😊', 'Your emotional state'),
        ('Energy', entry.energy_level, 60, '⚡', 'Physical vitality'),
        ('Focus', entry.focus_clarity, 180, '🎯', 'Mental clarity'),
        ('Calm', 100 - entry.anxiety_level, 120, '🧘', 'Inner peace'),
        ('Mindfulness', entry.mindfulness_score, 280, '🌸', 'Present awareness'),
        ('Sleep', entry.sleep_quality, 240, '😴', 'Rest quality'),
        ('Gratitude', entry.gratitude_level, 40, '🙏', 'Thankfulness'),
        ('Social', entry.social_connection, 320, '👥', 'Connections')
    ]
    
    for i, (label, value, hue, icon, desc) in enumerate(metrics):
        angle = i * GOLDEN_ANGLE_RAD
        radius = 0.15 + (value / 100) * 0.35
        z_pos = (value - 50) / 100  # Height based on value
        wellness_points.append({
            'type': 'wellness',
            'x': radius * math.cos(angle),
            'y': radius * math.sin(angle),
            'z': z_pos,
            'size': 8 + value / 10,
            'hue': hue,
            'color': f'hsl({hue}, 70%, 60%)',
            'label': label,
            'icon': icon,
            'value': round(value, 1),
            'description': desc,
            'pulse': value > 70  # Pulse animation for high values
        })
    
    # Goals positioned in outer spiral
    goal_points = []
    for i, goal in enumerate(list(user.goals.values())[:8]):
        angle = (i + len(metrics)) * GOLDEN_ANGLE_RAD
        progress_factor = goal.progress / 100
        radius = 0.45 + (FIBONACCI[(i+3) % len(FIBONACCI)] / 200)
        z_pos = progress_factor * 0.8  # Height increases with progress
        
        # Color based on progress
        if goal.is_completed:
            hue = 140
            icon = '✅'
        elif goal.progress > 70:
            hue = 120
            icon = '🎯'
        elif goal.progress > 40:
            hue = 45
            icon = '📊'
        else:
            hue = 0
            icon = '🎪'
        
        goal_points.append({
            'type': 'goal',
            'x': radius * math.cos(angle),
            'y': radius * math.sin(angle),
            'z': z_pos,
            'size': 12 + progress_factor * 8,
            'hue': hue,
            'color': f'hsl({hue}, 80%, 55%)',
            'label': goal.title,
            'icon': icon,
            'value': round(goal.progress, 1),
            'description': f'{goal.category} • Priority {goal.priority}',
            'completed': goal.is_completed,
            'milestones': goal.milestones_reached
        })
    
    # Habit connections - create paths between completed habits
    habit_nodes = []
    completed_today = sum(1 for h in user.habits.values() 
                         if entry.habits_completed.get(h.id, False))
    
    for i, habit in enumerate(list(user.habits.values())[:8]):
        angle = (i + len(metrics) + len(goal_points)) * GOLDEN_ANGLE_RAD
        completion_rate = habit.total_completions / max(1, habit.current_streak + habit.total_completions)
        radius = 0.3 + (habit.current_streak / 30) * 0.2
        z_pos = completion_rate * 0.5
        
        completed = entry.habits_completed.get(habit.id, False)
        hue = 120 if completed else 30
        
        habit_nodes.append({
            'type': 'habit',
            'x': radius * math.cos(angle),
            'y': radius * math.sin(angle),
            'z': z_pos,
            'size': 6 + habit.current_streak / 2,
            'hue': hue,
            'color': f'hsl({hue}, 70%, 60%)',
            'label': habit.name,
            'icon': '✓' if completed else '○',
            'value': habit.current_streak,
            'description': f'Streak: {habit.current_streak} days',
            'completed_today': completed,
            'streak': habit.current_streak
        })
    
    # Pet position - center, elevated based on wellness
    pet_icon = {'cat': '🐱', 'dragon': '🐉', 'phoenix': '🔥', 'owl': '🦉', 'fox': '🦊'}
    pet_data = {
        'type': 'pet',
        'x': 0,
        'y': 0,
        'z': wellness / 100,
        'size': 20 + user.pet.level * 2,
        'hue': 280,
        'color': f'hsl(280, 70%, 60%)',
        'label': user.pet.name,
        'icon': pet_icon.get(user.pet.species, '🐱'),
        'value': user.pet.level,
        'description': f'Level {user.pet.level} • {user.pet.behavior}',
        'behavior': user.pet.behavior,
        'stats': {
            'hunger': user.pet.hunger,
            'energy': user.pet.energy,
            'mood': user.pet.mood,
            'bond': user.pet.bond
        }
    }
    
    # Create connecting paths (for 3D visualization)
    connections = []
    # Connect pet to wellness metrics
    for point in wellness_points[:5]:
        connections.append({
            'from': [0, 0, pet_data['z']],
            'to': [point['x'], point['y'], point['z']],
            'strength': point['value'] / 100,
            'color': point['color']
        })
    
    # Legend data
    legend = {
        'wellness_metrics': {
            'icon': '😊',
            'description': 'Inner ring: Your daily wellness metrics',
            'color': 'Multi-hued spiral'
        },
        'goals': {
            'icon': '🎯',
            'description': 'Middle ring: Active goals (height = progress)',
            'color': 'Green (high) → Yellow (medium) → Red (low)'
        },
        'habits': {
            'icon': '✓',
            'description': 'Outer ring: Habit streaks',
            'color': 'Green (completed) → Orange (pending)'
        },
        'pet': {
            'icon': pet_icon.get(user.pet.species, '🐱'),
            'description': f'Center: {user.pet.name} (elevation = wellness)',
            'color': 'Purple glow'
        },
        'connections': {
            'icon': '〰️',
            'description': 'Energy flows between elements',
            'color': 'Gradient based on strength'
        }
    }
    
    return jsonify({
        'fractal_params': {
            'fractal_type': user.fractal_type or fractal_type,
            'hue_base': hue_base,
            'hue_range': 60,
            'animation_speed': anim_speed,
            'zoom': 1 + wellness / 100,
            'chaos_factor': chaos,
            'fibonacci_depth': min(13, 5 + int(entry.mindfulness_score / 20)),
            'show_flower_of_life': user.show_flower_of_life,
            'show_metatron_cube': user.show_metatron_cube,
            'show_golden_spiral': user.show_golden_spiral,
            'geometry_opacity': 0.1 + entry.gratitude_level / 200
        },
        'data_points': {
            'wellness': wellness_points,
            'goals': goal_points,
            'habits': habit_nodes,
            'pet': pet_data
        },
        'connections': connections,
        'legend': legend,
        'camera': {
            'position': [0, -1.5, 1.0],
            'target': [0, 0, wellness / 200],
            'fov': 60
        },
        'summary': {
            'wellness_index': round(wellness, 1),
            'mood_category': MoodLevel(entry.mood_level).name.lower(),
            'streak_days': user.current_streak,
            'goals_progress': round(sum(g.progress for g in user.goals.values()) / max(1, len(user.goals)), 1),
            'habits_completed_today': completed_today,
            'total_nodes': len(wellness_points) + len(goal_points) + len(habit_nodes) + 1
        },
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13]
        },
        'gpu_available': GPU_AVAILABLE
    })


@app.route('/api/user/<user_id>/fractal')
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GUIDANCE ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/guidance')
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ANALYTICS ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/user/<user_id>/analytics')
def get_analytics(user_id):
    """Get comprehensive analytics."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    entries = sorted(user.daily_entries.values(), key=lambda e: e.date)[-30:]
    
    if not entries:
        return jsonify({'error': 'No data'})
    
    wellness_trend = [{'date': e.date, 'value': round(e.wellness_index, 1)} for e in entries]
    
    mood_dist = {}
    for e in entries:
        mood = MoodLevel(e.mood_level).name
        mood_dist[mood] = mood_dist.get(mood, 0) + 1
    
    habit_stats = {}
    for habit in user.habits.values():
        completions = sum(1 for e in entries if e.habits_completed.get(habit.id, False))
        habit_stats[habit.name] = {
            'completion_rate': round(completions / max(1, len(entries)) * 100, 1),
            'streak': habit.current_streak
        }
    
    # Pythagorean means of wellness
    wellness_values = [e.wellness_index for e in entries if e.wellness_index > 0]
    means = AncientMathUtil.pythagorean_means(wellness_values) if wellness_values else {}
    
    return jsonify({
        'wellness_trend': wellness_trend,
        'mood_distribution': mood_dist,
        'habit_stats': habit_stats,
        'pythagorean_means': means,
        'averages': {
            'wellness': round(sum(e.wellness_index for e in entries) / len(entries), 1),
            'sleep': round(sum(e.sleep_hours for e in entries) / len(entries), 1),
            'mood': round(sum(e.mood_score for e in entries) / len(entries), 1)
        },
        'insights': {
            'best_day': max(entries, key=lambda e: e.wellness_index).date,
            'growth': round((entries[-1].wellness_index - entries[0].wellness_index), 1) if len(entries) > 1 else 0
        }
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SACRED MATH ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/sacred-math')
def sacred_math():
    """Get sacred mathematics constants."""
    return jsonify({
        'phi': PHI,
        'phi_inverse': PHI_INVERSE,
        'golden_angle_degrees': GOLDEN_ANGLE,
        'golden_angle_radians': GOLDEN_ANGLE_RAD,
        'fibonacci': FIBONACCI,
        'platonic_solids': PLATONIC_SOLIDS
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SYSTEM ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/api/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'version': '2.0.0'
    })


@app.route('/api/gpu-info')
def gpu_info():
    """Get GPU information."""
    return jsonify({
        'cuda_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'cupy_available': HAS_CUPY,
        'webgl_supported': True,
        'recommended_backend': 'cuda' if GPU_AVAILABLE else 'webgl'
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN PAGE ROUTE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route('/')
def index():
    """Serve main dashboard."""
    # Try to serve the HTML file
    try:
        return send_file('life_planner_dashboard.html')
    except:
        return jsonify({
            'message': 'Life Fractal Intelligence API',
            'version': '2.0.0',
            'endpoints': {
                'auth': '/api/auth/login, /api/auth/register',
                'user': '/api/user/<id>',
                'dashboard': '/api/user/<id>/dashboard',
                'visualization': '/api/user/<id>/visualization',
                'fractal': '/api/user/<id>/fractal',
                'guidance': '/api/user/<id>/guidance',
                'analytics': '/api/user/<id>/analytics'
            }
        })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def print_banner():
    print("\n" + "=" * 70)
    print("ðŸŒ€ LIFE FRACTAL INTELLIGENCE - UNIFIED MASTER APPLICATION v2.0")
    print("=" * 70)
    print(f"âœ¨ Golden Ratio (Ï†):     {PHI:.15f}")
    print(f"ðŸŒ» Golden Angle:         {GOLDEN_ANGLE:.10f}Â°")
    print(f"ðŸ”¢ Fibonacci:            {FIBONACCI[:10]}...")
    print(f"ðŸ–¥ï¸  GPU Available:        {GPU_AVAILABLE} ({GPU_NAME or 'CPU Only'})")
    print(f"ðŸ¤– ML Available:         {HAS_SKLEARN}")
    print("=" * 70)
    print("\nðŸ“¡ API Endpoints:")
    print("  Auth:          POST /api/auth/register, /api/auth/login")
    print("  Dashboard:     GET  /api/user/<id>/dashboard")
    print("  Today:         GET/POST /api/user/<id>/today")
    print("  Habits:        GET/POST /api/user/<id>/habits")
    print("  Goals:         GET/POST /api/user/<id>/goals")
    print("  Pet:           GET  /api/user/<id>/pet")
    print("  Visualization: GET  /api/user/<id>/visualization")
    print("  Fractal:       GET  /api/user/<id>/fractal")
    print("  Guidance:      GET  /api/user/<id>/guidance")
    print("  Analytics:     GET  /api/user/<id>/analytics")
    print("=" * 70)
    print(f"\nðŸ’° Subscription: ${SUBSCRIPTION_PRICE}/month, {TRIAL_DAYS}-day free trial")
    print(f"ðŸŽ GoFundMe: {GOFUNDME_URL}")
    print("=" * 70)


if __name__ == '__main__':
    print_banner()
    print(f"\nðŸš€ Starting server at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
