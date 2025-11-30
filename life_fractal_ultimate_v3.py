"""
🌀 LIFE FRACTAL INTELLIGENCE - ULTIMATE COMPREHENSIVE SYSTEM
═══════════════════════════════════════════════════════════════════════════════════════

Complete life planning system specifically designed for neurodivergent individuals:
- Visual representations for aphantasia
- Reduced cognitive load for autism/ADHD
- Stress tracking and reduction tools
- Dysgraphia-friendly interfaces
- Pattern recognition and predictive analytics

FEATURES:
- Full authentication with 7-day trial + Stripe integration
- Virtual pet system (5 species) with sacred badge achievements
- GPU-accelerated fractal visualization with CPU fallback
- Sacred geometry overlays (Flower of Life, Metatron's Cube, Golden Spiral)
- Fibonacci music generation based on life data
- Daily/Weekly/Monthly/Yearly timeline views
- Goal & habit tracking with Fibonacci milestones
- Journal with AI sentiment analysis
- Decision tree mood predictions
- Fuzzy logic guidance system
- Real-time 3D data point visualization
- Machine learning pattern recognition
- Recursive depth fractals with chaos theory
- Ancient mathematics integration throughout

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
import colorsys

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ML
try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory if GPU_AVAILABLE else 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    GPU_MEMORY = 0
    torch = None

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.is_available()
    CUPY_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if HAS_CUPY else None
except ImportError:
    HAS_CUPY = False
    CUPY_NAME = None
    cp = None

# MIDI Generation
try:
    import mido
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False
    mido = None


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
PLATONIC_SOLIDS = {
    'tetrahedron': {'faces': 4, 'vertices': 4, 'edges': 6, 'angle': 70.53},
    'cube': {'faces': 6, 'vertices': 8, 'edges': 12, 'angle': 90.0},
    'octahedron': {'faces': 8, 'vertices': 6, 'edges': 12, 'angle': 109.47},
    'dodecahedron': {'faces': 12, 'vertices': 20, 'edges': 30, 'angle': 116.57},
    'icosahedron': {'faces': 20, 'vertices': 12, 'edges': 30, 'angle': 138.19}
}

# MIDI Fibonacci scale
FIBONACCI_NOTES = [0, 1, 2, 3, 5, 8, 13, 21]
BASE_NOTE = 60  # Middle C


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
    
    @staticmethod
    def flower_of_life_points(rings: int = 3, radius: float = 50) -> List[Tuple[float, float]]:
        """Generate Flower of Life pattern points."""
        points = [(0, 0)]
        for ring in range(1, rings + 1):
            for i in range(6 * ring):
                angle = i * (2 * math.pi) / (6 * ring)
                x = ring * radius * math.cos(angle)
                y = ring * radius * math.sin(angle)
                points.append((x, y))
        return points
    
    @staticmethod
    def metatrons_cube_points(radius: float = 120) -> List[Tuple[float, float]]:
        """Generate Metatron's Cube vertices."""
        points = [(0, 0)]
        for i in range(6):
            angle = i * math.pi / 3
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
        return points


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED BADGE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

SACRED_BADGES = {
    'fibonacci_initiate': {
        'name': 'Fibonacci Initiate',
        'description': 'Complete 8 consecutive habits',
        'icon': '🌱',
        'requirement': lambda pet: pet.total_tasks_completed >= 8
    },
    'golden_seeker': {
        'name': 'Golden Seeker',
        'description': 'Reach 13 habit streak',
        'icon': '⭐',
        'requirement': lambda pet: any(h.current_streak >= 13 for h in pet.user.habits.values())
    },
    'sacred_guardian': {
        'name': 'Sacred Guardian',
        'description': 'Complete 21 goals',
        'icon': '🛡️',
        'requirement': lambda pet: sum(1 for g in pet.user.goals.values() if g.is_completed) >= 21
    },
    'flower_of_life': {
        'name': 'Flower of Life',
        'description': 'Maintain 34 day wellness streak',
        'icon': '🌸',
        'requirement': lambda pet: pet.user.current_streak >= 34
    },
    'metatrons_cube': {
        'name': "Metatron's Cube",
        'description': 'Achieve 55% average wellness',
        'icon': '🔷',
        'requirement': lambda pet: pet.wellness_average >= 55
    },
    'chaos_master': {
        'name': 'Chaos Master',
        'description': 'Navigate high stress 89 times',
        'icon': '🌀',
        'requirement': lambda pet: pet.high_stress_handled >= 89
    },
    'golden_spiral': {
        'name': 'Golden Spiral',
        'description': 'Level 144 pet evolution',
        'icon': '🌟',
        'requirement': lambda pet: pet.level >= 144
    },
    'fractal_sage': {
        'name': 'Fractal Sage',
        'description': 'Generate 233 fractals',
        'icon': '🧙',
        'requirement': lambda pet: pet.fractals_generated >= 233
    }
}


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
    """Virtual pet state with sacred badge achievements."""
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
    
    # Sacred achievements
    badges: List[str] = field(default_factory=list)
    fractals_generated: int = 0
    high_stress_handled: int = 0
    wellness_average: float = 50.0
    
    # Music generation stats
    midi_files_created: int = 0
    last_melody: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def check_badges(self, user) -> List[str]:
        """Check and award new badges."""
        self.user = user  # Temp reference for badge checking
        new_badges = []
        for badge_id, badge_info in SACRED_BADGES.items():
            if badge_id not in self.badges:
                if badge_info['requirement'](self):
                    self.badges.append(badge_id)
                    new_badges.append(badge_id)
                    logger.info(f"🏆 Badge earned: {badge_info['name']}")
        return new_badges


@dataclass
class DailyEntry:
    """A single day's entry with comprehensive metrics."""
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
    
    # Time period
    period: str = "daily"
    
    # Computed
    wellness_index: float = 0.0
    predicted_mood: float = 0.0
    chaos_score: float = 0.0
    fractal_complexity: int = 5
    
    def __post_init__(self):
        self.calculate_wellness()
        self.calculate_chaos()
    
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
    
    def calculate_chaos(self):
        """Calculate chaos score using logistic map."""
        # Use stress and anxiety as chaos parameters
        r = 3.5 + (self.stress_level / 100) * 0.5  # 3.5-4.0 (edge of chaos)
        x0 = (self.anxiety_level / 100)
        chaos_series = AncientMathUtil.logistic_map_series(r, x0, 10)
        self.chaos_score = np.std(chaos_series) * 100
        self.fractal_complexity = min(13, max(3, int(self.chaos_score / 10)))
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Habit:
    """A trackable habit with Fibonacci streak milestones."""
    id: str
    name: str
    description: str = ""
    frequency: str = "daily"
    category: str = "general"
    current_streak: int = 0
    longest_streak: int = 0
    total_completions: int = 0
    created_at: str = ""
    fibonacci_milestones_reached: List[int] = field(default_factory=list)
    
    def check_milestone(self) -> Optional[int]:
        """Check if a Fibonacci milestone was reached."""
        for fib in FIBONACCI[3:10]:  # [2, 3, 5, 8, 13, 21, 34]
            if self.current_streak >= fib and fib not in self.fibonacci_milestones_reached:
                self.fibonacci_milestones_reached.append(fib)
                return fib
        return None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Goal:
    """A goal with Fibonacci progress milestones."""
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
        """Check if a new Fibonacci milestone was reached."""
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
    """User account with comprehensive tracking."""
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
    recursion_depth: int = 5
    
    # Accessibility
    high_contrast: bool = False
    reduce_motion: bool = False
    font_size: str = "medium"
    enable_audio_feedback: bool = False
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY LOGIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FuzzyLogicEngine:
    """Fuzzy logic for generating supportive messages."""
    
    def __init__(self):
        self.messages = {
            ('low_stress', 'high_mood'): "You're doing great! Your positive energy is inspiring.",
            ('low_stress', 'medium_mood'): "You're in a good place. Small joys can lift you even higher.",
            ('low_stress', 'low_mood'): "Even on quieter days, you're managing well. Be gentle with yourself.",
            ('medium_stress', 'high_mood'): "Your resilience is shining through! Remember to take breaks.",
            ('medium_stress', 'medium_mood'): "Balance is key. You're navigating challenges well.",
            ('medium_stress', 'low_mood'): "It's okay to feel this way. Consider a short mindful pause.",
            ('high_stress', 'high_mood'): "Your positivity is admirable! Don't forget to rest.",
            ('high_stress', 'medium_mood'): "You're handling a lot. Prioritize what matters most right now.",
            ('high_stress', 'low_mood'): "These feelings are valid. Reach out for support if needed. You're not alone."
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


# ═══════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedMoodPredictor:
    """Enhanced decision tree with pattern recognition."""
    
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42, max_depth=5) if HAS_SKLEARN else None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.trained = False
    
    def train(self, history: List[Dict]) -> bool:
        """Train on user history with normalized features."""
        if not HAS_SKLEARN or not history or len(history) < 5:
            return False
        
        try:
            X = []
            y = []
            for i, record in enumerate(history[:-1]):
                features = [
                    float(record.get('stress_level', 50)) / 100,
                    float(record.get('mood_score', 50)) / 100,
                    float(record.get('energy_level', 50)) / 100,
                    float(record.get('goals_completed_count', 0)) / 10,
                    float(record.get('sleep_hours', 7)) / 12,
                    float(record.get('sleep_quality', 50)) / 100,
                    float(record.get('anxiety_level', 30)) / 100,
                    float(record.get('wellness_index', 50)) / 100
                ]
                target = float(history[i+1].get('mood_score', 50))
                X.append(features)
                y.append(target)
            
            if len(X) >= 5:
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.trained = True
                return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
        return False
    
    def predict(self, current_state: Dict) -> float:
        """Predict next mood with confidence score."""
        if not self.trained or not HAS_SKLEARN:
            return float(current_state.get('mood_score', 50))
        
        try:
            features = [[
                float(current_state.get('stress_level', 50)) / 100,
                float(current_state.get('mood_score', 50)) / 100,
                float(current_state.get('energy_level', 50)) / 100,
                float(current_state.get('goals_completed_count', 0)) / 10,
                float(current_state.get('sleep_hours', 7)) / 12,
                float(current_state.get('sleep_quality', 50)) / 100,
                float(current_state.get('anxiety_level', 30)) / 100,
                float(current_state.get('wellness_index', 50)) / 100
            ]]
            features_scaled = self.scaler.transform(features)
            return float(self.model.predict(features_scaled)[0])
        except:
            return float(current_state.get('mood_score', 50))


# ═══════════════════════════════════════════════════════════════════════════════
# MIDI FIBONACCI MUSIC GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FibonacciMusicGenerator:
    """Generate MIDI music based on Fibonacci sequences and life data."""
    
    def __init__(self):
        self.base_note = BASE_NOTE
        self.fibonacci_notes = FIBONACCI_NOTES
    
    def generate_sequence(self, length: int, mood: float, energy: float) -> List[int]:
        """Generate Fibonacci-based note sequence."""
        sequence = []
        note = self.base_note
        
        # Mood affects scale (higher mood = higher notes)
        mood_offset = int((mood - 50) / 10)
        
        # Energy affects rhythm variation
        rhythm_variety = max(1, int(energy / 20))
        
        for i in range(length):
            interval_idx = (i * rhythm_variety) % len(self.fibonacci_notes)
            interval = self.fibonacci_notes[interval_idx]
            sequence.append(note + interval + mood_offset)
            note += interval // 2  # Gradual progression
        
        return sequence
    
    def export_midi(self, notes: List[int], filename: str, velocity: int = 80, tempo: int = 120):
        """Export notes as MIDI file."""
        if not HAS_MIDI:
            logger.warning("MIDI library not available")
            return None
        
        try:
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
            
            time_per_note = 480  # 1 beat
            for note in notes:
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
                track.append(mido.Message('note_off', note=note, velocity=velocity, time=time_per_note))
            
            mid.save(filename)
            logger.info(f"MIDI file created: {filename}")
            return filename
        except Exception as e:
            logger.error(f"MIDI export failed: {e}")
            return None
    
    def generate_from_user_data(self, user: User, output_dir: str = "static/music") -> Optional[str]:
        """Generate personalized MIDI based on user's data."""
        os.makedirs(output_dir, exist_ok=True)
        
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        entry = user.daily_entries.get(today, DailyEntry(date=today))
        
        length = min(32, max(8, int(entry.wellness_index / 3)))
        notes = self.generate_sequence(length, entry.mood_score, entry.energy_level)
        
        velocity = int(40 + entry.energy_level * 0.6)
        tempo = int(60 + entry.mood_score * 0.8)
        
        filename = f"{output_dir}/{user.id}_{today}_fibonacci.mid"
        return self.export_midi(notes, filename, velocity, tempo)


# ═══════════════════════════════════════════════════════════════════════════════
# VIRTUAL PET SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualPet:
    """Enhanced virtual pet with behavior and evolution."""
    
    SPECIES_TRAITS = {
        'cat': {'energy_decay': 1.2, 'mood_sensitivity': 1.0, 'growth_rate': 1.0},
        'dragon': {'energy_decay': 0.8, 'mood_sensitivity': 1.3, 'growth_rate': 1.2},
        'phoenix': {'energy_decay': 1.0, 'mood_sensitivity': 0.8, 'growth_rate': 1.5},
        'owl': {'energy_decay': 0.9, 'mood_sensitivity': 1.1, 'growth_rate': 0.9},
        'fox': {'energy_decay': 1.1, 'mood_sensitivity': 1.2, 'growth_rate': 1.1}
    }
    
    BEHAVIORS = ['idle', 'happy', 'playful', 'tired', 'hungry', 'sad', 'excited', 'sleeping', 'meditating']
    
    def __init__(self, state: PetState):
        self.state = state
        self.traits = self.SPECIES_TRAITS.get(state.species, self.SPECIES_TRAITS['cat'])
    
    def update_from_user_data(self, user_data: Dict):
        """Update pet state based on user activity."""
        # Energy from sleep
        sleep_quality = user_data.get('sleep_quality', 50)
        self.state.energy = min(100, self.state.energy + (sleep_quality - 50) * 0.2)
        
        # Mood from user mood
        user_mood = user_data.get('mood_score', 50)
        mood_delta = (user_mood - 50) * 0.3 * self.traits['mood_sensitivity']
        self.state.mood = max(0, min(100, self.state.mood + mood_delta))
        
        # Stress inverse to mindfulness
        mindfulness = user_data.get('mindfulness_score', 50)
        self.state.stress = max(0, min(100, 100 - mindfulness * 0.8))
        
        # Growth from goals
        goals = user_data.get('goals_completed_count', 0)
        self.state.growth = min(100, self.state.growth + goals * 2 * self.traits['growth_rate'])
        
        # Experience and leveling
        xp_gain = int(goals * 10 + (user_mood / 10))
        self.state.experience += xp_gain
        
        # Level up (Fibonacci thresholds)
        xp_for_next = FIBONACCI[min(self.state.level + 5, len(FIBONACCI)-1)] * 10
        if self.state.experience >= xp_for_next:
            self.state.level += 1
            self.state.experience -= xp_for_next
            if self.state.level % 5 == 0:
                self.state.evolution_stage = min(3, self.state.evolution_stage + 1)
        
        # Natural decay
        self.state.hunger = min(100, self.state.hunger + 2 * self.traits['energy_decay'])
        self.state.energy = max(0, self.state.energy - 1 * self.traits['energy_decay'])
        
        # Update behavior
        self._update_behavior()
        
        # Update stats for badges
        if self.state.stress > 70:
            self.state.high_stress_handled += 1
        
        wellness = user_data.get('wellness_index', 50)
        if self.state.wellness_average == 0:
            self.state.wellness_average = wellness
        else:
            self.state.wellness_average = (self.state.wellness_average * 0.9 + wellness * 0.1)
    
    def _update_behavior(self):
        """Determine current behavior."""
        if self.state.hunger > 80:
            self.state.behavior = 'hungry'
        elif self.state.energy < 20:
            self.state.behavior = 'tired'
        elif self.state.energy < 10:
            self.state.behavior = 'sleeping'
        elif self.state.stress < 20 and self.state.mood > 70:
            self.state.behavior = 'meditating'
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


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-ACCELERATED FRACTAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedFractalGenerator:
    """GPU-accelerated fractal generation with sacred geometry overlays."""
    
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.use_gpu = GPU_AVAILABLE or HAS_CUPY
        self.music_gen = FibonacciMusicGenerator()
        
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {GPU_NAME or CUPY_NAME}")
        else:
            logger.info("Using CPU for fractal generation")
    
    def generate_mandelbrot(self, max_iter: int = 256, zoom: float = 1.0,
                           center: Tuple[float, float] = (-0.5, 0),
                           chaos_seed: float = 0.0) -> np.ndarray:
        """Generate Mandelbrot set with chaos influence."""
        if self.use_gpu and (torch is not None or HAS_CUPY):
            return self._mandelbrot_gpu(max_iter, zoom, center, chaos_seed)
        return self._mandelbrot_cpu(max_iter, zoom, center, chaos_seed)
    
    def _mandelbrot_gpu(self, max_iter: int, zoom: float, center: Tuple[float, float], chaos_seed: float) -> np.ndarray:
        try:
            if torch is not None and GPU_AVAILABLE:
                device = torch.device('cuda')
                x = torch.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width, device=device)
                y = torch.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height, device=device)
                X, Y = torch.meshgrid(x, y, indexing='xy')
                
                c = X + 1j * Y + chaos_seed * 0.1
                z = torch.zeros_like(c)
                iterations = torch.zeros(self.height, self.width, device=device)
                
                for i in range(max_iter):
                    mask = torch.abs(z) <= 2
                    z[mask] = z[mask] ** 2 + c[mask]
                    iterations[mask] = i
                
                return iterations.cpu().numpy()
            elif HAS_CUPY:
                x = cp.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
                y = cp.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
                X, Y = cp.meshgrid(x, y)
                
                c = X + 1j * Y + chaos_seed * 0.1
                z = cp.zeros_like(c)
                iterations = cp.zeros((self.height, self.width))
                
                for i in range(max_iter):
                    mask = cp.abs(z) <= 2
                    z[mask] = z[mask] ** 2 + c[mask]
                    iterations[mask] = i
                
                return cp.asnumpy(iterations)
        except Exception as e:
            logger.error(f"GPU generation failed: {e}")
            return self._mandelbrot_cpu(max_iter, zoom, center, chaos_seed)
    
    def _mandelbrot_cpu(self, max_iter: int, zoom: float, center: Tuple[float, float], chaos_seed: float) -> np.ndarray:
        x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
        y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y + chaos_seed * 0.1
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
    
    def apply_advanced_coloring(self, iterations: np.ndarray, max_iter: int,
                                hue_base: float = 0.6, hue_range: float = 0.3,
                                saturation: float = 0.8, wellness: float = 50.0) -> np.ndarray:
        """Apply advanced HSV coloring with wellness influence."""
        normalized = iterations / max_iter
        
        # Wellness affects color intensity
        wellness_factor = wellness / 100
        
        # HSV to RGB
        h = (hue_base + normalized * hue_range) % 1.0
        s = np.full_like(normalized, saturation * wellness_factor)
        v = np.sqrt(normalized) * (0.7 + wellness_factor * 0.3)
        
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
    
    def add_sacred_geometry_overlay(self, image: Image.Image, show_flower: bool = True,
                                   show_metatron: bool = True, show_spiral: bool = True) -> Image.Image:
        """Add sacred geometry overlays to fractal."""
        draw = ImageDraw.Draw(image, 'RGBA')
        center_x, center_y = self.width // 2, self.height // 2
        
        # Flower of Life
        if show_flower:
            points = AncientMathUtil.flower_of_life_points(rings=4, radius=self.width // 20)
            for x, y in points:
                px, py = center_x + x, center_y + y
                draw.ellipse([px-25, py-25, px+25, py+25], outline=(255, 215, 0, 100), width=2)
        
        # Metatron's Cube
        if show_metatron:
            points = AncientMathUtil.metatrons_cube_points(radius=self.width // 6)
            # Draw center circle
            draw.ellipse([center_x-20, center_y-20, center_x+20, center_y+20], 
                        outline=(0, 255, 255, 120), width=3)
            # Draw outer circles and connections
            for x, y in points[1:]:
                px, py = center_x + x, center_y + y
                draw.ellipse([px-15, py-15, px+15, py+15], outline=(0, 255, 255, 100), width=2)
                draw.line([center_x, center_y, px, py], fill=(0, 255, 255, 80), width=2)
        
        # Golden Spiral
        if show_spiral:
            spiral_points = []
            for i in range(100):
                x, y = AncientMathUtil.golden_spiral_point(i)
                px = center_x + x * (self.width / 40)
                py = center_y + y * (self.height / 40)
                spiral_points.append((px, py))
            if len(spiral_points) > 1:
                draw.line(spiral_points, fill=(255, 255, 255, 120), width=2)
        
        return image
    
    def create_comprehensive_visualization(self, user_data: Dict, pet_state: Optional[Dict] = None) -> Image.Image:
        """Create complete visualization with all enhancements."""
        # Map user data to fractal parameters
        mood = user_data.get('mood_score', 50)
        energy = user_data.get('energy_level', 50)
        anxiety = user_data.get('anxiety_level', 30)
        stress = user_data.get('stress_level', 30)
        mindfulness = user_data.get('mindfulness_score', 50)
        wellness = user_data.get('wellness_index', 50)
        chaos_score = user_data.get('chaos_score', 0)
        
        # Determine fractal type
        if wellness < 30:
            iterations = self.generate_julia(-0.8, 0.156, max_iter=256, zoom=1.5)
            hue_base = 0.7
        elif wellness < 60:
            iterations = self.generate_mandelbrot(max_iter=256, zoom=1.5 + wellness/100, chaos_seed=chaos_score/100)
            hue_base = 0.5 + (mood - 50) / 200
        else:
            # Hybrid
            m = self.generate_mandelbrot(max_iter=256, zoom=2.0, chaos_seed=chaos_score/100)
            j = self.generate_julia(-0.7 + (mood-50)/200, 0.27, max_iter=200)
            iterations = m * 0.5 + j * 0.5
            hue_base = 0.3 + (mood / 200)
        
        # Color based on mood and energy
        hue_range = 0.3 + (energy / 200)
        saturation = 0.5 + (mindfulness / 200)
        
        rgb = self.apply_advanced_coloring(iterations, 256, hue_base, hue_range, saturation, wellness)
        image = Image.fromarray(rgb, 'RGB')
        
        # Add sacred geometry overlays
        show_flower = user_data.get('show_flower_of_life', True)
        show_metatron = user_data.get('show_metatron_cube', True)
        show_spiral = user_data.get('show_golden_spiral', True)
        
        image = self.add_sacred_geometry_overlay(image, show_flower, show_metatron, show_spiral)
        
        return image
    
    def to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert image to base64."""
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# LIFE PLANNING SYSTEM (Main Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class ComprehensiveLifePlanningSystem:
    """Main orchestrator integrating all advanced features."""
    
    def __init__(self, pet_species: str = "cat"):
        self.fractal_gen = AdvancedFractalGenerator(1024, 1024)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.predictor = AdvancedMoodPredictor()
        self.music_gen = FibonacciMusicGenerator()
        self.pet = VirtualPet(PetState(species=pet_species))
        self.history: List[Dict] = []
    
    def update(self, user_data: Dict):
        """Update system with new user data."""
        # Update pet
        self.pet.update_from_user_data(user_data)
        
        # Store in history
        record = {**user_data, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.history.append(record)
        
        # Train predictor
        if len(self.history) >= 5:
            self.predictor.train(self.history)
        
        # Track fractal generation
        self.pet.state.fractals_generated += 1
    
    def generate_comprehensive_guidance(self, current_state: Dict, user: User) -> Dict[str, Any]:
        """Generate all guidance outputs."""
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
            'excited': f"{self.pet.state.name} is absolutely thrilled! Keep it up!",
            'tired': f"{self.pet.state.name} is resting. Maybe you need rest too?",
            'hungry': f"{self.pet.state.name} is hungry. Have you eaten well today?",
            'sad': f"{self.pet.state.name} senses you might be down. It's here for you.",
            'idle': f"{self.pet.state.name} is keeping you company.",
            'sleeping': f"{self.pet.state.name} is catching Z's. Rest is important!",
            'meditating': f"{self.pet.state.name} is in a zen state. Join in the calm."
        }
        pet_message = pet_messages.get(pet_behavior, f"{self.pet.state.name} is with you.")
        
        # Check for new badges
        new_badges = self.pet.state.check_badges(user)
        badge_messages = [f"🏆 {SACRED_BADGES[b]['name']}: {SACRED_BADGES[b]['description']}" for b in new_badges]
        
        # Generate music
        music_file = None
        if HAS_MIDI and current_state.get('wellness_index', 0) > 60:
            music_file = self.music_gen.generate_from_user_data(user)
            if music_file:
                self.pet.state.midi_files_created += 1
                self.pet.state.last_melody = music_file
        
        return {
            'predicted_mood': round(predicted_mood, 1),
            'fuzzy_message': fuzzy_message,
            'pet_message': pet_message,
            'pet_state': self.pet.state.to_dict(),
            'new_badges': badge_messages,
            'music_file': music_file,
            'combined_message': f"{fuzzy_message} {pet_message}",
            'prediction_confidence': 'high' if len(self.history) >= 10 else 'medium' if len(self.history) >= 5 else 'low'
        }
    
    def generate_fractal_visualization(self, user_data: Dict) -> Image.Image:
        """Generate comprehensive fractal visualization."""
        return self.fractal_gen.create_comprehensive_visualization(user_data, self.pet.state.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STORE
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """In-memory data store with demo data."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.systems: Dict[str, ComprehensiveLifePlanningSystem] = {}
        self._init_admin()
    
    def _init_admin(self):
        """Create admin user with comprehensive demo data."""
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
            created_at=datetime.now(timezone.utc).isoformat(),
            recursion_depth=8
        )
        admin.set_password('admin8587037321')
        
        # Create advanced pet
        admin.pet = PetState(
            species='dragon',
            name='Ember',
            level=25,
            evolution_stage=1,
            badges=['fibonacci_initiate', 'golden_seeker'],
            fractals_generated=50
        )
        
        self.users[admin_id] = admin
        self.users[admin.email] = admin
        
        # Add comprehensive demo data
        self._add_comprehensive_demo_data(admin)
    
    def _add_comprehensive_demo_data(self, user: User):
        """Add 30 days of varied demo data for testing all features."""
        now = datetime.now(timezone.utc)
        
        # Demo habits
        habits = [
            ("Morning Meditation", "wellness", 15),
            ("Exercise 30 min", "health", 12),
            ("Read 20 pages", "growth", 8),
            ("Journal Entry", "wellness", 5),
            ("Drink 8 glasses water", "health", 18),
            ("Gratitude Practice", "wellness", 11)
        ]
        
        for i, (name, category, streak) in enumerate(habits):
            habit = Habit(
                id=f"habit_{i+1}",
                name=name,
                category=category,
                current_streak=streak,
                longest_streak=streak + 7,
                total_completions=streak * 5,
                created_at=(now - timedelta(days=45)).isoformat()
            )
            habit.check_milestone()
            user.habits[habit.id] = habit
        
        # Demo goals
        goals = [
            ("Complete Project Alpha", "work", 1, 85),
            ("Learn Meditation Course", "wellness", 2, 55),
            ("Read 12 Books This Year", "growth", 3, 92)
        ]
        
        for i, (title, category, priority, progress) in enumerate(goals):
            goal = Goal(
                id=f"goal_{i+1}",
                title=title,
                category=category,
                priority=priority,
                progress=progress,
                target_date=(now + timedelta(days=30 + i*30)).isoformat()[:10],
                created_at=(now - timedelta(days=90)).isoformat()
            )
            goal.check_milestones()
            user.goals[goal.id] = goal
        
        # 30 days of detailed entries with patterns
        for i in range(30):
            date = (now - timedelta(days=29-i)).strftime('%Y-%m-%d')
            
            # Create realistic patterns
            week_factor = (i % 7) / 7
            trend_factor = i / 30
            
            # Weekend vs weekday
            is_weekend = (i % 7) in [5, 6]
            
            mood_base = 55 + trend_factor * 15
            if is_weekend:
                mood_base += 10
            
            entry = DailyEntry(
                date=date,
                mood_level=max(1, min(5, int(3 + math.sin(i*0.3) * 1.5))),
                mood_score=mood_base + math.sin(i*0.4) * 20,
                energy_level=50 + math.cos(i*0.3) * 25 + (10 if is_weekend else 0),
                focus_clarity=60 + math.sin(i*0.25) * 20,
                anxiety_level=max(10, 35 - trend_factor * 15 + week_factor * 10),
                stress_level=max(15, 40 - trend_factor * 10 + week_factor * 8),
                mindfulness_score=45 + trend_factor * 20 + math.sin(i*0.2) * 10,
                gratitude_level=50 + trend_factor * 15 + math.cos(i*0.3) * 12,
                sleep_quality=65 + math.sin(i*0.2) * 20 + (10 if is_weekend else 0),
                sleep_hours=6.5 + math.sin(i*0.25) * 1.5 + (1 if is_weekend else 0),
                nutrition_score=60 + math.cos(i*0.3) * 15,
                social_connection=50 + week_factor * 20 + (15 if is_weekend else 0),
                emotional_stability=55 + trend_factor * 10,
                self_compassion=50 + trend_factor * 12,
                goals_completed_count=(i % 3)
            )
            
            # Random habit completions
            for habit_id in user.habits.keys():
                entry.habits_completed[habit_id] = (i + hash(habit_id)) % 3 != 0
            
            entry.calculate_wellness()
            entry.calculate_chaos()
            user.daily_entries[date] = entry
            user.history.append(entry.to_dict())
        
        # Update streak
        user.current_streak = 15
        user.longest_streak = 21
    
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
        
        # Add basic demo data
        self._add_basic_demo_data(user)
        
        self.users[user_id] = user
        self.users[email.lower()] = user
        
        return user
    
    def _add_basic_demo_data(self, user: User):
        """Add minimal demo data for new users."""
        now = datetime.now(timezone.utc)
        
        # One habit
        habit = Habit(
            id="habit_1",
            name="Daily Check-in",
            category="wellness",
            created_at=now.isoformat()
        )
        user.habits[habit.id] = habit
        
        # One goal
        goal = Goal(
            id="goal_1",
            title="Complete First Week",
            category="general",
            priority=1,
            progress=14,
            created_at=now.isoformat()
        )
        user.goals[goal.id] = goal
        
        # Today's entry
        today = now.strftime('%Y-%m-%d')
        entry = DailyEntry(date=today)
        entry.calculate_wellness()
        user.daily_entries[today] = entry
    
    def get_user(self, identifier: str) -> Optional[User]:
        """Get user by ID or email."""
        return self.users.get(identifier) or self.users.get(identifier.lower())
    
    def get_system(self, user_id: str) -> ComprehensiveLifePlanningSystem:
        """Get or create life planning system for user."""
        if user_id not in self.systems:
            user = self.users.get(user_id)
            species = user.pet.species if user and user.pet else 'cat'
            self.systems[user_id] = ComprehensiveLifePlanningSystem(species)
        return self.systems[user_id]


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'life-fractal-ultimate-secret-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app)

store = DataStore()

# Configuration
SUBSCRIPTION_PRICE = 20.00
TRIAL_DAYS = 7
GOFUNDME_URL = 'https://gofund.me/8d9303d27'


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD & USER ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

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
    avg_chaos = sum(e.chaos_score for e in entries) / max(1, len(entries))
    
    return jsonify({
        'user': user.to_dict(),
        'today': today_entry.to_dict(),
        'pet': user.pet.to_dict() if user.pet else None,
        'habits': [h.to_dict() for h in user.habits.values()],
        'goals': [g.to_dict() for g in user.goals.values()],
        'stats': {
            'wellness_index': round(today_entry.wellness_index, 1),
            'average_wellness': round(avg_wellness, 1),
            'chaos_score': round(today_entry.chaos_score, 1),
            'average_chaos': round(avg_chaos, 1),
            'current_streak': user.current_streak,
            'total_entries': len(entries),
            'habits_completed_today': sum(1 for c in today_entry.habits_completed.values() if c),
            'active_goals': sum(1 for g in user.goals.values() if not g.is_completed),
            'goals_progress': round(sum(g.progress for g in user.goals.values()) / max(1, len(user.goals)), 1),
            'badges_earned': len(user.pet.badges) if user.pet else 0
        },
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13]
        },
        'gpu_info': {
            'available': GPU_AVAILABLE or HAS_CUPY,
            'name': GPU_NAME or CUPY_NAME,
            'memory_gb': GPU_MEMORY / (1024**3) if GPU_MEMORY else 0
        }
    })


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
    entry.calculate_chaos()
    
    # Update history
    user.history.append(entry.to_dict())
    
    # Update life planning system
    system = store.get_system(user_id)
    system.update(entry.to_dict())
    
    return jsonify(entry.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/visualization')
def get_visualization_data(user_id):
    """Get comprehensive 3D visualization data with all sacred elements."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
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
        z_pos = (value - 50) / 100
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
            'pulse': value > 70
        })
    
    # Goals
    goal_points = []
    for i, goal in enumerate(list(user.goals.values())[:8]):
        angle = (i + len(metrics)) * GOLDEN_ANGLE_RAD
        progress_factor = goal.progress / 100
        radius = 0.45 + (FIBONACCI[(i+3) % len(FIBONACCI)] / 200)
        z_pos = progress_factor * 0.8
        
        if goal.is_completed:
            hue, icon = 140, '✅'
        elif goal.progress > 70:
            hue, icon = 120, '🎯'
        elif goal.progress > 40:
            hue, icon = 45, '📊'
        else:
            hue, icon = 0, '🎪'
        
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
    
    # Habits
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
    
    # Pet at center
    pet_icon = {'cat': '🐱', 'dragon': '🐉', 'phoenix': '🔥', 'owl': '🦉', 'fox': '🦊'}
    pet_data = {
        'type': 'pet',
        'x': 0,
        'y': 0,
        'z': entry.wellness_index / 100,
        'size': 20 + user.pet.level * 2,
        'hue': 280,
        'color': 'hsl(280, 70%, 60%)',
        'label': user.pet.name,
        'icon': pet_icon.get(user.pet.species, '🐱'),
        'value': user.pet.level,
        'description': f'Level {user.pet.level} • {user.pet.behavior}',
        'behavior': user.pet.behavior,
        'badges': user.pet.badges,
        'stats': {
            'hunger': user.pet.hunger,
            'energy': user.pet.energy,
            'mood': user.pet.mood,
            'bond': user.pet.bond
        }
    }
    
    # Connections
    connections = []
    for point in wellness_points[:5]:
        connections.append({
            'from': [0, 0, pet_data['z']],
            'to': [point['x'], point['y'], point['z']],
            'strength': point['value'] / 100,
            'color': point['color']
        })
    
    # Sacred geometry points
    flower_points = AncientMathUtil.flower_of_life_points(rings=3, radius=0.8)
    metatron_points = AncientMathUtil.metatrons_cube_points(radius=1.0)
    
    return jsonify({
        'data_points': {
            'wellness': wellness_points,
            'goals': goal_points,
            'habits': habit_nodes,
            'pet': pet_data
        },
        'connections': connections,
        'sacred_geometry': {
            'flower_of_life': [{'x': p[0], 'y': p[1]} for p in flower_points],
            'metatrons_cube': [{'x': p[0], 'y': p[1]} for p in metatron_points],
            'show_flower': user.show_flower_of_life,
            'show_metatron': user.show_metatron_cube,
            'show_spiral': user.show_golden_spiral
        },
        'fractal_params': {
            'type': user.fractal_type,
            'recursion_depth': user.recursion_depth,
            'chaos_score': entry.chaos_score,
            'complexity': entry.fractal_complexity,
            'wellness': entry.wellness_index,
            'animation_speed': user.animation_speed
        },
        'legend': {
            'wellness_metrics': {
                'icon': '😊',
                'description': 'Inner ring: Daily wellness metrics on golden spiral',
                'color': 'Multi-hued gradient'
            },
            'goals': {
                'icon': '🎯',
                'description': 'Middle ring: Goals (height = progress)',
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
                'color': 'Purple with sacred badges'
            },
            'sacred_geometry': {
                'icon': '🌸',
                'description': 'Overlays: Flower of Life, Metatron\'s Cube, Golden Spiral',
                'color': 'Gold/Cyan transparency'
            }
        },
        'summary': {
            'wellness_index': round(entry.wellness_index, 1),
            'chaos_score': round(entry.chaos_score, 1),
            'mood_category': MoodLevel(entry.mood_level).name.lower(),
            'streak_days': user.current_streak,
            'goals_progress': round(sum(g.progress for g in user.goals.values()) / max(1, len(user.goals)), 1),
            'habits_completed_today': completed_today,
            'total_nodes': len(wellness_points) + len(goal_points) + len(habit_nodes) + 1,
            'badges_earned': len(user.pet.badges)
        },
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13],
            'platonic_solids': PLATONIC_SOLIDS
        },
        'gpu_available': GPU_AVAILABLE or HAS_CUPY
    })


@app.route('/api/user/<user_id>/fractal')
def generate_fractal_image(user_id):
    """Generate high-resolution fractal image."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    # Merge user settings with entry data
    viz_data = {
        **entry.to_dict(),
        'show_flower_of_life': user.show_flower_of_life,
        'show_metatron_cube': user.show_metatron_cube,
        'show_golden_spiral': user.show_golden_spiral,
        'recursion_depth': user.recursion_depth
    }
    
    system = store.get_system(user_id)
    image = system.generate_fractal_visualization(viz_data)
    
    img_io = BytesIO()
    image.save(img_io, 'PNG', optimize=True)
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')


@app.route('/api/user/<user_id>/fractal/base64')
def get_fractal_base64(user_id):
    """Get fractal as base64 for embedding."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    viz_data = {
        **entry.to_dict(),
        'show_flower_of_life': user.show_flower_of_life,
        'show_metatron_cube': user.show_metatron_cube,
        'show_golden_spiral': user.show_golden_spiral
    }
    
    system = store.get_system(user_id)
    image = system.generate_fractal_visualization(viz_data)
    base64_data = system.fractal_gen.to_base64(image)
    
    return jsonify({
        'image': f'data:image/png;base64,{base64_data}',
        'gpu_used': system.fractal_gen.use_gpu,
        'resolution': f'{image.width}x{image.height}'
    })


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDANCE & MUSIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/guidance')
def get_guidance(user_id):
    """Get comprehensive AI guidance."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    system = store.get_system(user_id)
    guidance = system.generate_comprehensive_guidance(entry.to_dict(), user)
    
    return jsonify(guidance)


@app.route('/api/user/<user_id>/music/generate', methods=['POST'])
def generate_music(user_id):
    """Generate Fibonacci music based on current state."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if not HAS_MIDI:
        return jsonify({'error': 'MIDI generation not available'}), 501
    
    system = store.get_system(user_id)
    music_file = system.music_gen.generate_from_user_data(user)
    
    if music_file:
        return jsonify({
            'success': True,
            'file': music_file,
            'total_generated': user.pet.midi_files_created
        })
    else:
        return jsonify({'error': 'Music generation failed'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# PET & BADGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/pet')
def get_pet(user_id):
    """Get pet information with badges."""
    user = store.get_user(user_id)
    if not user or not user.pet:
        return jsonify({'error': 'Not found'}), 404
    
    # Get badge details
    badges_detailed = []
    for badge_id in user.pet.badges:
        if badge_id in SACRED_BADGES:
            badge = SACRED_BADGES[badge_id]
            badges_detailed.append({
                'id': badge_id,
                'name': badge['name'],
                'description': badge['description'],
                'icon': badge['icon']
            })
    
    pet_data = user.pet.to_dict()
    pet_data['badges_detailed'] = badges_detailed
    pet_data['available_badges'] = len(SACRED_BADGES)
    
    return jsonify(pet_data)


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


# ═══════════════════════════════════════════════════════════════════════════════
# HABITS & GOALS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

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
        milestone = habit.check_milestone()
        
        return jsonify({
            'success': True,
            'habit': habit.to_dict(),
            'milestone_reached': milestone
        })
    
    return jsonify({'success': True, 'habit': habit.to_dict()})


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
        user.pet.total_goals_achieved += 1
    
    return jsonify({
        'success': True,
        'goal': goal.to_dict(),
        'milestone_reached': milestone
    })


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/analytics')
def get_analytics(user_id):
    """Get comprehensive analytics with ancient math."""
    user = store.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    entries = sorted(user.daily_entries.values(), key=lambda e: e.date)[-30:]
    
    if not entries:
        return jsonify({'error': 'No data'})
    
    wellness_trend = [{'date': e.date, 'value': round(e.wellness_index, 1)} for e in entries]
    chaos_trend = [{'date': e.date, 'value': round(e.chaos_score, 1)} for e in entries]
    
    mood_dist = {}
    for e in entries:
        mood = MoodLevel(e.mood_level).name
        mood_dist[mood] = mood_dist.get(mood, 0) + 1
    
    habit_stats = {}
    for habit in user.habits.values():
        completions = sum(1 for e in entries if e.habits_completed.get(habit.id, False))
        habit_stats[habit.name] = {
            'completion_rate': round(completions / max(1, len(entries)) * 100, 1),
            'streak': habit.current_streak,
            'fibonacci_milestones': habit.fibonacci_milestones_reached
        }
    
    # Pythagorean means
    wellness_values = [e.wellness_index for e in entries if e.wellness_index > 0]
    means = AncientMathUtil.pythagorean_means(wellness_values) if wellness_values else {}
    
    # Chaos analysis
    chaos_values = [e.chaos_score for e in entries]
    chaos_mean = np.mean(chaos_values) if chaos_values else 0
    chaos_std = np.std(chaos_values) if chaos_values else 0
    
    return jsonify({
        'wellness_trend': wellness_trend,
        'chaos_trend': chaos_trend,
        'mood_distribution': mood_dist,
        'habit_stats': habit_stats,
        'pythagorean_means': means,
        'chaos_analysis': {
            'mean': round(chaos_mean, 2),
            'std': round(chaos_std, 2),
            'edge_of_chaos': chaos_mean > 30 and chaos_mean < 70
        },
        'averages': {
            'wellness': round(sum(e.wellness_index for e in entries) / len(entries), 1),
            'chaos': round(chaos_mean, 1),
            'sleep': round(sum(e.sleep_hours for e in entries) / len(entries), 1),
            'mood': round(sum(e.mood_score for e in entries) / len(entries), 1)
        },
        'insights': {
            'best_day': max(entries, key=lambda e: e.wellness_index).date,
            'most_chaotic': max(entries, key=lambda e: e.chaos_score).date,
            'growth': round((entries[-1].wellness_index - entries[0].wellness_index), 1) if len(entries) > 1 else 0
        },
        'badges_progress': {
            'earned': len(user.pet.badges),
            'total': len(SACRED_BADGES),
            'next_badge': next((b for b_id, b in SACRED_BADGES.items() if b_id not in user.pet.badges), None)
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    """Health check with system info."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'gpu_available': GPU_AVAILABLE or HAS_CUPY,
        'gpu_name': GPU_NAME or CUPY_NAME,
        'gpu_memory_gb': GPU_MEMORY / (1024**3) if GPU_MEMORY else 0,
        'ml_available': HAS_SKLEARN,
        'midi_available': HAS_MIDI,
        'features': {
            'fractals': True,
            'sacred_geometry': True,
            'fibonacci_music': HAS_MIDI,
            'ai_predictions': HAS_SKLEARN,
            'gpu_acceleration': GPU_AVAILABLE or HAS_CUPY,
            'chaos_theory': True,
            'virtual_pet': True,
            'badge_system': True
        },
        'version': '3.0.0-ultimate'
    })


@app.route('/api/sacred-math')
def sacred_math():
    """Get all sacred mathematics constants."""
    return jsonify({
        'phi': PHI,
        'phi_inverse': PHI_INVERSE,
        'golden_angle_degrees': GOLDEN_ANGLE,
        'golden_angle_radians': GOLDEN_ANGLE_RAD,
        'fibonacci': FIBONACCI,
        'fibonacci_notes': FIBONACCI_NOTES,
        'platonic_solids': PLATONIC_SOLIDS,
        'sacred_badges': {k: {**v, 'requirement': 'function'} for k, v in SACRED_BADGES.items()}
    })


@app.route('/')
def index():
    """Serve main dashboard."""
    try:
        return send_file('life_planner_dashboard.html')
    except:
        return jsonify({
            'message': 'Life Fractal Intelligence - Ultimate System',
            'version': '3.0.0',
            'features': [
                'GPU-accelerated fractals',
                'Sacred geometry overlays',
                'Fibonacci music generation',
                'AI mood predictions',
                'Virtual pet with badges',
                'Chaos theory integration',
                '3D data visualization',
                'Ancient mathematics'
            ],
            'endpoints': {
                'auth': '/api/auth/login, /api/auth/register',
                'dashboard': '/api/user/<id>/dashboard',
                'visualization': '/api/user/<id>/visualization',
                'fractal': '/api/user/<id>/fractal',
                'guidance': '/api/user/<id>/guidance',
                'music': '/api/user/<id>/music/generate',
                'analytics': '/api/user/<id>/analytics'
            }
        })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print startup banner with system info."""
    print("\n" + "=" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE - ULTIMATE COMPREHENSIVE SYSTEM v3.0")
    print("=" * 80)
    print(f"✨ Golden Ratio (Φ):     {PHI:.15f}")
    print(f"🌻 Golden Angle:         {GOLDEN_ANGLE:.10f}°")
    print(f"🔢 Fibonacci:            {FIBONACCI[:10]}...")
    print(f"🖥️  GPU Available:        {GPU_AVAILABLE or HAS_CUPY}")
    if GPU_AVAILABLE or HAS_CUPY:
        print(f"   GPU Name:            {GPU_NAME or CUPY_NAME}")
        print(f"   GPU Memory:          {GPU_MEMORY / (1024**3):.2f} GB" if GPU_MEMORY else "   GPU Memory:          N/A")
    print(f"🤖 ML Available:         {HAS_SKLEARN}")
    print(f"🎵 MIDI Available:       {HAS_MIDI}")
    print("=" * 80)
    print("\n🎨 FEATURES:")
    print("  ✓ GPU-Accelerated Fractals (with CPU fallback)")
    print("  ✓ Sacred Geometry Overlays (Flower of Life, Metatron's Cube)")
    print("  ✓ Fibonacci Music Generation")
    print("  ✓ AI Mood Predictions")
    print("  ✓ Virtual Pet with Sacred Badges")
    print("  ✓ Chaos Theory Integration")
    print("  ✓ 3D Data Visualization")
    print("  ✓ Ancient Mathematics Throughout")
    print("  ✓ Designed for Neurodivergent Users")
    print("=" * 80)
    print("\n📡 API Endpoints:")
    print("  Auth:          POST /api/auth/register, /api/auth/login")
    print("  Dashboard:     GET  /api/user/<id>/dashboard")
    print("  Visualization: GET  /api/user/<id>/visualization")
    print("  Fractal:       GET  /api/user/<id>/fractal")
    print("  Guidance:      GET  /api/user/<id>/guidance")
    print("  Music:         POST /api/user/<id>/music/generate")
    print("  Pet:           GET  /api/user/<id>/pet")
    print("  Analytics:     GET  /api/user/<id>/analytics")
    print("=" * 80)
    print(f"\n💰 Subscription: ${SUBSCRIPTION_PRICE}/month, {TRIAL_DAYS}-day free trial")
    print(f"🎁 GoFundMe: {GOFUNDME_URL}")
    print("=" * 80)


if __name__ == '__main__':
    print_banner()
    print(f"\n🚀 Starting server at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
