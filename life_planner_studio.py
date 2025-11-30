#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ULTIMATE LIFE PLANNER STUDIO - VISION TO REALITY ENGINE            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Combines:                                                                    ║
║  • Life Planning & Goal Tracking                                             ║
║  • ComfyUI Integration (AI Image Generation)                                 ║
║  • Animated Video Export (MP4)                                               ║
║  • Machine Learning Predictions & Habit Analysis                             ║
║  • Decision Tree Planning                                                     ║
║  • Golden Ratio / Fibonacci Mathematics                                       ║
║  • Swarm Intelligence                                                         ║
║  • Therapeutic Audio (Brown/Pink/Green Noise)                                ║
║  • Entropy-Based Progress Analysis                                           ║
║  • Accessibility First (ADHD, Autism, Aphantasia)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import time
import random
import hashlib
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import traceback

# Core dependencies
import numpy as np

# GUI Framework
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QProgressBar, QTextEdit, QTabWidget,
        QListWidget, QFileDialog, QSlider, QSpinBox, QComboBox,
        QGroupBox, QSplitter, QListWidgetItem, QMessageBox, QCheckBox,
        QLineEdit, QScrollArea, QFrame, QGridLayout, QTreeWidget,
        QTreeWidgetItem, QDialog, QDialogButtonBox, QPlainTextEdit,
        QStackedWidget, QToolBar, QStatusBar, QDockWidget, QTableWidget,
        QTableWidgetItem, QHeaderView, QSizePolicy
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QSize, QUrl
    from PySide6.QtGui import (
        QColor, QPalette, QFont, QIcon, QPainter, QPen, QBrush,
        QLinearGradient, QPixmap, QImage
    )
    PYSIDE6_OK = True
except ImportError:
    PYSIDE6_OK = False
    print("[WARN] PySide6 not found. Install with: pip install PySide6")

# Optional dependencies
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[WARN] OpenCV not found. Install with: pip install opencv-python")

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] Pillow not found. Install with: pip install Pillow")

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("[WARN] Requests not found. Install with: pip install requests")

try:
    import sounddevice as sd
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("[WARN] sounddevice not found. Audio features disabled.")

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found. ML features limited.")

try:
    import torch
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_OK = False
    DEVICE = "cpu"
    print("[WARN] PyTorch not found. GPU features disabled.")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# MATHEMATICAL CONSTANTS (Ancient Wisdom)
# ============================================================
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ = 1.618033988...
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
GOLDEN_ANGLE = 137.5077640500378  # degrees
PI = math.pi
E = math.e
SACRED_FREQUENCIES = {
    'liberation': 396,
    'change': 417,
    'transformation': 528,
    'connection': 639,
    'expression': 741,
    'intuition': 852,
    'universal': 432
}
MAYAN_BASE = 20
BABYLONIAN_BASE = 60

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Goal:
    """Represents a single goal"""
    id: str
    category: str  # mental, financial, career, living
    title: str
    description: str = ""
    completed: bool = False
    created_date: str = ""
    completed_date: str = ""
    difficulty: int = 5  # 1-10
    importance: int = 5  # 1-10
    energy_required: int = 5  # 1-10
    notes: str = ""
    
@dataclass
class Milestone:
    """Major life milestone"""
    id: int
    title: str
    target_month: int
    completed: bool = False
    completed_date: str = ""
    fibonacci_number: int = 1
    energy_score: float = 0.0

@dataclass
class DailyLog:
    """Daily activity and mood log"""
    date: str
    mood: int = 5  # 1-10
    energy: int = 5  # 1-10
    focus_hours: float = 0.0
    tasks_completed: int = 0
    challenges: List[str] = field(default_factory=list)
    wins: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class HabitData:
    """Tracks habits over time"""
    name: str
    category: str
    is_good: bool = True
    occurrences: List[str] = field(default_factory=list)  # dates
    streak: int = 0
    best_streak: int = 0

@dataclass
class PredictionResult:
    """ML prediction output"""
    outcome: str
    probability: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    confidence: float = 0.0

# ============================================================
# ANCIENT MATHEMATICS ENGINE
# ============================================================
class AncientMathEngine:
    """
    Mathematical calculations using ancient wisdom:
    - Golden Ratio (φ)
    - Fibonacci Sequence
    - Pythagorean Harmonics
    - Mayan Base-20
    - Babylonian Mathematics
    """
    
    @staticmethod
    def fibonacci_progress(current: int, total: int) -> Dict:
        """Calculate progress using Fibonacci scaling"""
        if total == 0:
            return {'ratio': 0, 'fib_index': 0, 'momentum': 1}
        
        ratio = current / total
        fib_index = min(int(ratio * len(FIBONACCI)), len(FIBONACCI) - 1)
        
        prev_fib = FIBONACCI[max(0, fib_index - 1)]
        curr_fib = FIBONACCI[fib_index]
        
        return {
            'ratio': ratio,
            'fib_index': fib_index,
            'fib_number': curr_fib,
            'golden_progress': ratio ** (1 / GOLDEN_RATIO),
            'momentum': curr_fib / prev_fib if prev_fib > 0 else 1,
            'natural_alignment': abs(ratio - (1 / GOLDEN_RATIO))
        }
    
    @staticmethod
    def golden_spiral_position(index: int, scale: float = 1.0) -> Tuple[float, float]:
        """Calculate position on golden spiral"""
        theta = index * math.radians(GOLDEN_ANGLE)
        r = scale * (GOLDEN_RATIO ** (theta / (2 * PI)))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return (x, y)
    
    @staticmethod
    def pythagorean_harmony(values: List[float]) -> float:
        """Calculate harmonic balance using Pythagorean ratios"""
        if not values or len(values) < 2:
            return 1.0
        
        # Sacred ratios: 1:1, 2:1, 3:2, 4:3
        sacred_ratios = [1.0, 2.0, 1.5, 4/3]
        
        harmony_score = 0
        pairs = 0
        
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[j] != 0:
                    ratio = values[i] / values[j]
                    # Find closest sacred ratio
                    closest = min(sacred_ratios, key=lambda x: abs(x - ratio))
                    harmony_score += 1 - abs(ratio - closest) / max(ratio, closest)
                    pairs += 1
        
        return harmony_score / pairs if pairs > 0 else 1.0
    
    @staticmethod
    def entropy_score(data: List[float]) -> float:
        """Calculate Shannon entropy for uncertainty measurement"""
        if not data:
            return 0.0
        
        # Normalize to probabilities
        total = sum(abs(x) for x in data) or 1
        probs = [abs(x) / total for x in data]
        
        # Shannon entropy
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        
        # Normalize to 0-1
        max_entropy = math.log2(len(data)) if len(data) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    @staticmethod
    def mayan_base20(number: int) -> str:
        """Convert to Mayan base-20 representation"""
        symbols = ['☉', '☽', '★', '◇', '△', '□', '○', '◎', '✧', '❋',
                   '✦', '◈', '❖', '✶', '❂', '✴', '❀', '✿', '❁', '✾']
        
        if number == 0:
            return symbols[0]
        
        result = ''
        while number > 0:
            result = symbols[number % MAYAN_BASE] + result
            number //= MAYAN_BASE
        
        return result
    
    @staticmethod
    def life_energy_score(milestones: List[Milestone], current_month: int) -> Dict:
        """Calculate overall life energy using multiple mathematical systems"""
        completed = sum(1 for m in milestones if m.completed)
        total = len(milestones)
        
        fib_progress = AncientMathEngine.fibonacci_progress(completed, total)
        
        # Calculate phase
        if completed < 3:
            phase = 'awakening'
        elif completed < 6:
            phase = 'building'
        elif completed < 9:
            phase = 'manifesting'
        else:
            phase = 'thriving'
        
        # Energy calculation using golden ratio alignment
        golden_alignment = 1 - fib_progress['natural_alignment']
        energy = golden_alignment * 100
        
        return {
            'energy': energy,
            'phase': phase,
            'momentum': fib_progress['momentum'],
            'sacred_number': fib_progress['fib_number'],
            'mayan_id': AncientMathEngine.mayan_base20(completed + current_month),
            'golden_alignment': golden_alignment
        }

# ============================================================
# SWARM INTELLIGENCE ENGINE
# ============================================================
class SwarmIntelligence:
    """
    Multi-agent decision making using swarm principles:
    - Collective voting
    - Emergent behavior
    - Pattern recognition
    """
    
    def __init__(self, num_agents: int = 100):
        self.num_agents = num_agents
        self.agent_weights = np.random.dirichlet(np.ones(num_agents))
    
    def vote_on_priority(self, options: List[str], context: Dict) -> List[Tuple[str, float]]:
        """Swarm votes on priority of options"""
        phase = context.get('phase', 1)
        completed_categories = context.get('completed', {})
        
        # Phase-based weightings
        phase_weights = {
            1: {'mental': 0.35, 'financial': 0.25, 'career': 0.25, 'living': 0.15},
            2: {'mental': 0.25, 'financial': 0.25, 'career': 0.35, 'living': 0.15},
            3: {'mental': 0.20, 'financial': 0.30, 'career': 0.25, 'living': 0.25},
            4: {'mental': 0.20, 'financial': 0.25, 'career': 0.25, 'living': 0.30}
        }
        
        weights = phase_weights.get(phase, phase_weights[1])
        
        votes = {}
        for option in options:
            base_weight = weights.get(option, 0.25)
            urgency = 0.5 if completed_categories.get(option, False) else 1.0
            
            # Each agent votes with some randomness
            agent_votes = [
                base_weight * urgency * (1 + np.random.normal(0, 0.1)) * w
                for w in self.agent_weights
            ]
            votes[option] = sum(agent_votes)
        
        # Normalize and sort
        total = sum(votes.values()) or 1
        results = [(k, v / total) for k, v in votes.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def predict_burnout_risk(self, activity_data: Dict) -> Dict:
        """Analyze activity patterns for burnout risk"""
        goals_completed = activity_data.get('goals_completed', 0)
        days_active = activity_data.get('days_active', 1)
        challenges = activity_data.get('challenges', 0)
        
        intensity = goals_completed / max(days_active, 1)
        challenge_ratio = challenges / max(goals_completed, 1)
        
        if intensity > 2 and challenge_ratio > 0.5:
            risk = 'high'
            suggestion = "Consider taking a rest day. Progress compounds over time."
        elif challenge_ratio > 0.7:
            risk = 'medium'
            suggestion = "Many challenges noted. Would smaller steps help?"
        elif intensity < 0.2 and days_active > 7:
            risk = 'low-activity'
            suggestion = "Activity is low. One small win today could build momentum."
        else:
            risk = 'low'
            suggestion = "Good pace! Keep going."
        
        return {
            'risk': risk,
            'intensity': intensity,
            'challenge_ratio': challenge_ratio,
            'suggestion': suggestion
        }
    
    def emergent_pattern(self, history: List[DailyLog]) -> Dict:
        """Find emergent patterns in behavior"""
        if len(history) < 7:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        # Analyze mood trends
        moods = [log.mood for log in history[-14:]]
        energies = [log.energy for log in history[-14:]]
        
        mood_trend = np.polyfit(range(len(moods)), moods, 1)[0]
        energy_trend = np.polyfit(range(len(energies)), energies, 1)[0]
        
        if mood_trend > 0.1 and energy_trend > 0.1:
            pattern = 'ascending'
            insight = "Both mood and energy are improving. You're on an upward trajectory!"
        elif mood_trend < -0.1 and energy_trend < -0.1:
            pattern = 'descending'
            insight = "Mood and energy trending down. Consider what's draining you."
        elif mood_trend > 0.1 and energy_trend < -0.1:
            pattern = 'happy_tired'
            insight = "Happy but tired - you may be pushing hard. Rest could help."
        elif mood_trend < -0.1 and energy_trend > 0.1:
            pattern = 'energetic_low_mood'
            insight = "Energy is there but mood is low. What would bring you joy?"
        else:
            pattern = 'stable'
            insight = "Stable patterns. Good foundation for growth."
        
        return {
            'pattern': pattern,
            'insight': insight,
            'mood_trend': mood_trend,
            'energy_trend': energy_trend,
            'confidence': min(len(history) / 30, 1.0)
        }

# ============================================================
# MACHINE LEARNING PREDICTION ENGINE
# ============================================================
class MLPredictionEngine:
    """
    Machine learning for:
    - Habit prediction
    - Outcome forecasting
    - Decision tree generation
    """
    
    def __init__(self):
        self.habit_model = None
        self.outcome_model = None
        self.scaler = StandardScaler() if SKLEARN_OK else None
        self.training_data = []
        self.is_trained = False
    
    def add_training_data(self, features: Dict, outcome: str):
        """Add data point for training"""
        self.training_data.append({
            'features': features,
            'outcome': outcome
        })
        
        # Retrain after enough data
        if len(self.training_data) >= 20:
            self._train_models()
    
    def _train_models(self):
        """Train prediction models"""
        if not SKLEARN_OK or len(self.training_data) < 10:
            return
        
        try:
            # Prepare data
            feature_names = list(self.training_data[0]['features'].keys())
            X = np.array([[d['features'].get(f, 0) for f in feature_names] 
                         for d in self.training_data])
            y = [d['outcome'] for d in self.training_data]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train decision tree
            self.outcome_model = DecisionTreeClassifier(max_depth=5)
            self.outcome_model.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info("ML models trained successfully")
            
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    def predict_outcome(self, features: Dict) -> PredictionResult:
        """Predict outcome based on features"""
        if not self.is_trained or not SKLEARN_OK:
            # Fallback to rule-based prediction
            return self._rule_based_prediction(features)
        
        try:
            feature_names = list(self.training_data[0]['features'].keys())
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.outcome_model.predict(X_scaled)[0]
            probabilities = self.outcome_model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
            
            # Feature importance
            importances = dict(zip(feature_names, self.outcome_model.feature_importances_))
            
            return PredictionResult(
                outcome=prediction,
                probability=confidence,
                factors=importances,
                confidence=confidence,
                recommendation=self._generate_recommendation(prediction, importances)
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: Dict) -> PredictionResult:
        """Fallback rule-based prediction"""
        energy = features.get('energy', 5)
        mood = features.get('mood', 5)
        streak = features.get('streak', 0)
        difficulty = features.get('difficulty', 5)
        
        score = (energy * 0.3 + mood * 0.3 + min(streak, 10) * 0.2 + (10 - difficulty) * 0.2) / 10
        
        if score > 0.7:
            outcome = 'success_likely'
            recommendation = "Conditions are favorable. Good time to tackle this goal."
        elif score > 0.4:
            outcome = 'moderate_chance'
            recommendation = "Consider breaking this into smaller steps."
        else:
            outcome = 'needs_preparation'
            recommendation = "Build up energy and mood first. Self-care today, goals tomorrow."
        
        return PredictionResult(
            outcome=outcome,
            probability=score,
            factors={'energy': 0.3, 'mood': 0.3, 'streak': 0.2, 'difficulty': 0.2},
            confidence=0.6,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, outcome: str, factors: Dict) -> str:
        """Generate actionable recommendation"""
        top_factor = max(factors.items(), key=lambda x: x[1])[0]
        
        recommendations = {
            'success_likely': f"Great conditions! Your {top_factor} is a key strength.",
            'moderate_chance': f"Focus on improving {top_factor} to increase success odds.",
            'needs_preparation': f"Build up your {top_factor} before attempting this goal."
        }
        
        return recommendations.get(outcome, "Keep tracking to improve predictions.")
    
    def generate_decision_tree(self, goal: Goal, context: Dict) -> Dict:
        """Generate decision tree for goal planning"""
        tree = {
            'goal': goal.title,
            'root': {
                'question': 'Do I have enough energy today?',
                'yes': {
                    'question': 'Is this aligned with my phase priorities?',
                    'yes': {
                        'action': 'DO IT NOW',
                        'confidence': 0.9,
                        'reasoning': 'High energy + aligned priorities = optimal time'
                    },
                    'no': {
                        'question': 'Is it urgent?',
                        'yes': {
                            'action': 'DO IT, but schedule phase-aligned work for tomorrow',
                            'confidence': 0.7
                        },
                        'no': {
                            'action': 'DEFER to better time, focus on phase priorities',
                            'confidence': 0.8
                        }
                    }
                },
                'no': {
                    'question': 'Is this goal energy-restoring?',
                    'yes': {
                        'action': 'DO IT - it may boost your energy',
                        'confidence': 0.6
                    },
                    'no': {
                        'action': 'REST first, then reassess',
                        'confidence': 0.85,
                        'reasoning': 'Low energy + draining task = burnout risk'
                    }
                }
            }
        }
        
        return tree

# ============================================================
# THERAPEUTIC AUDIO ENGINE
# ============================================================
class TherapeuticAudioEngine:
    """
    Generate therapeutic sounds:
    - Brown noise (grounding)
    - Pink noise (focus)
    - Green noise (clarity)
    - Binaural beats
    - Sacred frequencies
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.current_stream = None
    
    def generate_noise(self, noise_type: str, duration: float = 1.0) -> np.ndarray:
        """Generate colored noise"""
        samples = int(duration * self.sample_rate)
        white = np.random.randn(samples)
        
        if noise_type == 'brown':
            # Brown noise - integrate white noise
            brown = np.cumsum(white)
            brown = brown / np.max(np.abs(brown))
            return brown * 0.3
        
        elif noise_type == 'pink':
            # Pink noise - 1/f spectrum
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(samples, 1/self.sample_rate)
            freqs[0] = 1  # Avoid division by zero
            fft = fft / np.sqrt(freqs)
            pink = np.fft.irfft(fft, samples)
            return pink / np.max(np.abs(pink)) * 0.3
        
        elif noise_type == 'green':
            # Green noise - mid-frequency emphasis
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(samples, 1/self.sample_rate)
            # Bandpass around 500Hz
            mask = np.exp(-((freqs - 500) ** 2) / (2 * 200 ** 2))
            fft = fft * mask
            green = np.fft.irfft(fft, samples)
            return green / np.max(np.abs(green)) * 0.3
        
        else:  # white
            return white * 0.2
    
    def generate_tone(self, frequency: float, duration: float = 2.0) -> np.ndarray:
        """Generate pure tone at sacred frequency"""
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        
        # Add subtle harmonics for richness
        tone = np.sin(2 * np.pi * frequency * t) * 0.5
        tone += np.sin(2 * np.pi * frequency * 2 * t) * 0.2
        tone += np.sin(2 * np.pi * frequency * 3 * t) * 0.1
        
        # Envelope
        envelope = np.ones_like(tone)
        fade_samples = int(0.1 * self.sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return tone * envelope * 0.3
    
    def generate_binaural(self, base_freq: float, beat_freq: float, duration: float = 10.0) -> np.ndarray:
        """Generate binaural beats (requires stereo headphones)"""
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        
        left = np.sin(2 * np.pi * base_freq * t)
        right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
        
        stereo = np.column_stack((left, right)) * 0.3
        return stereo
    
    def play_sound(self, audio_data: np.ndarray):
        """Play audio using sounddevice"""
        if not AUDIO_OK:
            logger.warning("Audio playback not available")
            return
        
        try:
            self.stop()
            sd.play(audio_data, self.sample_rate)
            self.is_playing = True
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def stop(self):
        """Stop current playback"""
        if AUDIO_OK:
            sd.stop()
        self.is_playing = False
    
    def play_celebration(self):
        """Play celebration sound"""
        frequencies = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        duration = 0.3
        
        full_audio = np.array([])
        for freq in frequencies:
            tone = self.generate_tone(freq, duration)
            full_audio = np.concatenate([full_audio, tone])
        
        self.play_sound(full_audio)

# ============================================================
# COMFYUI INTEGRATION
# ============================================================
class ComfyUIClient:
    """
    Connect to ComfyUI for AI image generation
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(hashlib.md5(str(time.time()).encode()).hexdigest())[:8]
        self.connected = False
    
    def check_connection(self) -> bool:
        """Check if ComfyUI is running"""
        if not REQUESTS_OK:
            return False
        
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=2)
            self.connected = response.status_code == 200
            return self.connected
        except:
            self.connected = False
            return False
    
    def generate_image(self, prompt: str, negative_prompt: str = "", 
                       width: int = 512, height: int = 512,
                       steps: int = 20, cfg: float = 7.0) -> Optional[bytes]:
        """Generate image using ComfyUI API"""
        if not self.connected:
            if not self.check_connection():
                logger.warning("ComfyUI not connected")
                return None
        
        # Basic txt2img workflow
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": random.randint(0, 2**32),
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt or "ugly, blurry, low quality",
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "life_planner",
                    "images": ["8", 0]
                }
            }
        }
        
        try:
            # Queue prompt
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id}
            )
            
            if response.status_code == 200:
                prompt_id = response.json().get('prompt_id')
                
                # Wait for completion (simplified - real implementation would use websocket)
                for _ in range(60):  # Max 60 seconds
                    time.sleep(1)
                    history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        if '9' in outputs:
                            images = outputs['9'].get('images', [])
                            if images:
                                filename = images[0]['filename']
                                subfolder = images[0].get('subfolder', '')
                                img_response = requests.get(
                                    f"{self.base_url}/view",
                                    params={"filename": filename, "subfolder": subfolder}
                                )
                                return img_response.content
                        break
            
            return None
            
        except Exception as e:
            logger.error(f"ComfyUI generation error: {e}")
            return None

# ============================================================
# VIDEO ANIMATION ENGINE
# ============================================================
class VideoAnimationEngine:
    """
    Create animated videos from life plan:
    - Vision board animations
    - Progress visualizations
    - Motivational videos
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
    
    def create_golden_spiral_animation(self, milestones: List[Milestone], 
                                       duration: float = 10.0,
                                       output_path: str = "spiral_progress.mp4"):
        """Animate progress along golden spiral"""
        if not CV2_OK or not PIL_OK:
            logger.error("OpenCV and Pillow required for video generation")
            return None
        
        total_frames = int(duration * self.fps)
        frames = []
        
        # Colors
        bg_color = (30, 26, 42)  # Dark purple
        spiral_color = (146, 154, 124)  # Sage green
        completed_color = (80, 175, 76)  # Green
        pending_color = (85, 85, 85)  # Gray
        
        center_x, center_y = self.width // 2, self.height // 2
        
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            
            # Create frame
            img = Image.new('RGB', (self.width, self.height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw spiral path
            spiral_points = []
            for i in range(200):
                t = i * 0.1 * progress * 2
                pos = AncientMathEngine.golden_spiral_position(i, scale=8)
                x = center_x + pos[0] * 20
                y = center_y + pos[1] * 20
                spiral_points.append((x, y))
            
            if len(spiral_points) > 1:
                for i in range(len(spiral_points) - 1):
                    alpha = int(255 * (i / len(spiral_points)) * progress)
                    draw.line([spiral_points[i], spiral_points[i+1]], 
                             fill=spiral_color, width=2)
            
            # Draw milestones
            for idx, milestone in enumerate(milestones):
                reveal_time = (idx + 1) / (len(milestones) + 1)
                if progress >= reveal_time:
                    pos = AncientMathEngine.golden_spiral_position(idx * 6, scale=8)
                    mx = int(center_x + pos[0] * 25)
                    my = int(center_y + pos[1] * 25)
                    
                    color = completed_color if milestone.completed else pending_color
                    radius = 15 if milestone.completed else 10
                    
                    # Pulsing effect for completed
                    if milestone.completed:
                        pulse = math.sin(frame_num * 0.1) * 3
                        radius += int(pulse)
                    
                    draw.ellipse([mx - radius, my - radius, mx + radius, my + radius],
                                fill=color, outline=(255, 255, 255))
            
            # Add text overlay
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            # Title
            title = "Your Journey on the Golden Spiral"
            draw.text((self.width // 2 - 200, 50), title, fill=(255, 255, 255), font=font)
            
            # Stats
            completed = sum(1 for m in milestones if m.completed)
            stats = f"Milestones: {completed}/{len(milestones)} | φ = {GOLDEN_RATIO:.3f}"
            draw.text((50, self.height - 60), stats, fill=(200, 200, 200), font=small_font)
            
            # Convert to numpy for OpenCV
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to {output_path}")
        return output_path
    
    def create_vision_board_video(self, images: List[str], affirmations: List[str],
                                  duration: float = 30.0,
                                  output_path: str = "vision_board.mp4"):
        """Create animated vision board video"""
        if not CV2_OK or not PIL_OK:
            logger.error("OpenCV and Pillow required")
            return None
        
        total_frames = int(duration * self.fps)
        frames_per_image = total_frames // max(len(images), 1)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        for img_idx, img_path in enumerate(images):
            try:
                img = Image.open(img_path).resize((self.width, self.height))
            except:
                # Create placeholder
                img = Image.new('RGB', (self.width, self.height), (50, 50, 70))
                draw = ImageDraw.Draw(img)
                draw.text((self.width//2 - 100, self.height//2), 
                         "Vision Image", fill=(200, 200, 200))
            
            # Add affirmation overlay
            aff_idx = img_idx % len(affirmations) if affirmations else 0
            affirmation = affirmations[aff_idx] if affirmations else ""
            
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            # Add semi-transparent overlay for text
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([0, self.height - 150, self.width, self.height],
                                  fill=(0, 0, 0, 180))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            
            draw = ImageDraw.Draw(img)
            draw.text((50, self.height - 120), f'"{affirmation}"', 
                     fill=(255, 255, 255), font=font)
            
            # Write frames with Ken Burns effect
            for frame_num in range(frames_per_image):
                progress = frame_num / frames_per_image
                
                # Subtle zoom
                zoom = 1.0 + progress * 0.05
                new_size = (int(self.width * zoom), int(self.height * zoom))
                zoomed = img.resize(new_size, Image.LANCZOS)
                
                # Crop to original size
                left = (new_size[0] - self.width) // 2
                top = (new_size[1] - self.height) // 2
                cropped = zoomed.crop((left, top, left + self.width, top + self.height))
                
                frame = np.array(cropped)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
        
        out.release()
        logger.info(f"Vision board video saved to {output_path}")
        return output_path
    
    def create_progress_timeline_video(self, monthly_data: List[Dict],
                                       duration: float = 20.0,
                                       output_path: str = "progress_timeline.mp4"):
        """Create animated timeline of progress"""
        if not CV2_OK or not PIL_OK:
            return None
        
        total_frames = int(duration * self.fps)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        phase_colors = {
            1: (146, 154, 124),  # Sage
            2: (181, 197, 168),  # Light sage
            3: (150, 184, 212),  # Blue
            4: (183, 213, 232)   # Light blue
        }
        
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            current_month = int(progress * 24) + 1
            
            # Create frame
            img = Image.new('RGB', (self.width, self.height), (245, 241, 235))
            draw = ImageDraw.Draw(img)
            
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                title_font = ImageFont.load_default()
                font = title_font
            
            # Title
            draw.text((self.width // 2 - 200, 30), "Your 24-Month Journey",
                     fill=(61, 74, 68), font=title_font)
            
            # Timeline bar
            bar_y = 150
            bar_height = 40
            month_width = (self.width - 200) / 24
            
            for month in range(24):
                x = 100 + month * month_width
                phase = 1 if month < 4 else 2 if month < 10 else 3 if month < 16 else 4
                color = phase_colors[phase]
                
                # Animate reveal
                if month < current_month:
                    alpha = 255
                else:
                    alpha = 50
                
                draw.rectangle([x, bar_y, x + month_width - 2, bar_y + bar_height],
                              fill=color if month < current_month else (200, 200, 200))
                
                # Month label
                if month % 3 == 0:
                    draw.text((x, bar_y + bar_height + 5), str(month + 1),
                             fill=(100, 100, 100), font=font)
            
            # Current position marker
            marker_x = 100 + (current_month - 1) * month_width
            draw.polygon([(marker_x, bar_y - 20), 
                         (marker_x - 10, bar_y - 35),
                         (marker_x + 10, bar_y - 35)],
                        fill=(255, 100, 100))
            
            # Phase labels
            phases = [("Foundation", 4), ("Building", 10), ("Establishing", 16), ("Independence", 24)]
            for phase_name, end_month in phases:
                start = phases[phases.index((phase_name, end_month)) - 1][1] if phases.index((phase_name, end_month)) > 0 else 0
                mid_x = 100 + ((start + end_month) / 2) * month_width
                draw.text((mid_x - 50, bar_y + bar_height + 40), phase_name,
                         fill=(61, 74, 68), font=font)
            
            # Stats section
            stats_y = 300
            fib_num = FIBONACCI[min(current_month - 1, len(FIBONACCI) - 1)]
            
            draw.text((100, stats_y), f"Current Month: {current_month}",
                     fill=(61, 74, 68), font=font)
            draw.text((100, stats_y + 40), f"Fibonacci Energy: {fib_num}",
                     fill=(61, 74, 68), font=font)
            draw.text((100, stats_y + 80), f"Golden Ratio Progress: {(current_month/24 * GOLDEN_RATIO):.3f}",
                     fill=(61, 74, 68), font=font)
            
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        logger.info(f"Timeline video saved to {output_path}")
        return output_path

# ============================================================
# DATA PERSISTENCE
# ============================================================
class DataManager:
    """Save and load life planner data"""
    
    def __init__(self, data_dir: str = "life_planner_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.goals_file = self.data_dir / "goals.json"
        self.milestones_file = self.data_dir / "milestones.json"
        self.logs_file = self.data_dir / "daily_logs.json"
        self.habits_file = self.data_dir / "habits.json"
        self.settings_file = self.data_dir / "settings.json"
    
    def save_goals(self, goals: List[Goal]):
        """Save goals to file"""
        data = [asdict(g) for g in goals]
        with open(self.goals_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_goals(self) -> List[Goal]:
        """Load goals from file"""
        if not self.goals_file.exists():
            return []
        
        with open(self.goals_file, 'r') as f:
            data = json.load(f)
        
        return [Goal(**g) for g in data]
    
    def save_milestones(self, milestones: List[Milestone]):
        """Save milestones to file"""
        data = [asdict(m) for m in milestones]
        with open(self.milestones_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_milestones(self) -> List[Milestone]:
        """Load milestones from file"""
        if not self.milestones_file.exists():
            return self._default_milestones()
        
        with open(self.milestones_file, 'r') as f:
            data = json.load(f)
        
        return [Milestone(**m) for m in data]
    
    def _default_milestones(self) -> List[Milestone]:
        """Create default milestones"""
        return [
            Milestone(1, "Secured credit card opened", 1, fibonacci_number=1),
            Milestone(2, "First portfolio piece created", 3, fibonacci_number=2),
            Milestone(3, "LinkedIn profile optimized", 2, fibonacci_number=3),
            Milestone(4, "First income earned", 6, fibonacci_number=5),
            Milestone(5, "Emergency fund: $500", 8, fibonacci_number=8),
            Milestone(6, "Credit score: 650+", 12, fibonacci_number=13),
            Milestone(7, "Stable monthly income", 10, fibonacci_number=21),
            Milestone(8, "Emergency fund: 1 month expenses", 14, fibonacci_number=34),
            Milestone(9, "Apartment application ready", 16, fibonacci_number=55),
            Milestone(10, "Moved into own place", 20, fibonacci_number=89),
        ]
    
    def save_daily_log(self, log: DailyLog):
        """Save or update daily log"""
        logs = self.load_daily_logs()
        
        # Update existing or add new
        found = False
        for i, existing in enumerate(logs):
            if existing.date == log.date:
                logs[i] = log
                found = True
                break
        
        if not found:
            logs.append(log)
        
        data = [asdict(l) for l in logs]
        with open(self.logs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_daily_logs(self) -> List[DailyLog]:
        """Load all daily logs"""
        if not self.logs_file.exists():
            return []
        
        with open(self.logs_file, 'r') as f:
            data = json.load(f)
        
        return [DailyLog(**l) for l in data]
    
    def save_settings(self, settings: Dict):
        """Save user settings"""
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
    
    def load_settings(self) -> Dict:
        """Load user settings"""
        if not self.settings_file.exists():
            return {
                'name': '',
                'big_why': '',
                'current_month': 1,
                'accessibility_mode': 'normal',
                'audio_enabled': True,
                'comfyui_host': '127.0.0.1',
                'comfyui_port': 8188
            }
        
        with open(self.settings_file, 'r') as f:
            return json.load(f)

# ============================================================
# MAIN GUI APPLICATION
# ============================================================
if PYSIDE6_OK:
    
    class WorkerSignals(QObject):
        """Signals for background workers"""
        finished = Signal()
        error = Signal(str)
        result = Signal(object)
        progress = Signal(int)
    
    class ImageGenerationWorker(QThread):
        """Background worker for image generation"""
        
        def __init__(self, comfy_client: ComfyUIClient, prompt: str):
            super().__init__()
            self.comfy_client = comfy_client
            self.prompt = prompt
            self.signals = WorkerSignals()
        
        def run(self):
            try:
                result = self.comfy_client.generate_image(self.prompt)
                self.signals.result.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))
            finally:
                self.signals.finished.emit()
    
    class VideoGenerationWorker(QThread):
        """Background worker for video generation"""
        
        def __init__(self, video_engine: VideoAnimationEngine, method: str, **kwargs):
            super().__init__()
            self.video_engine = video_engine
            self.method = method
            self.kwargs = kwargs
            self.signals = WorkerSignals()
        
        def run(self):
            try:
                if self.method == 'spiral':
                    result = self.video_engine.create_golden_spiral_animation(**self.kwargs)
                elif self.method == 'vision':
                    result = self.video_engine.create_vision_board_video(**self.kwargs)
                elif self.method == 'timeline':
                    result = self.video_engine.create_progress_timeline_video(**self.kwargs)
                else:
                    result = None
                
                self.signals.result.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))
            finally:
                self.signals.finished.emit()
    
    class LifePlannerStudio(QMainWindow):
        """Main application window"""
        
        def __init__(self):
            super().__init__()
            
            # Initialize engines
            self.math_engine = AncientMathEngine()
            self.swarm = SwarmIntelligence()
            self.ml_engine = MLPredictionEngine()
            self.audio_engine = TherapeuticAudioEngine()
            self.comfy_client = ComfyUIClient()
            self.video_engine = VideoAnimationEngine()
            self.data_manager = DataManager()
            
            # Load data
            self.settings = self.data_manager.load_settings()
            self.milestones = self.data_manager.load_milestones()
            self.goals = self.data_manager.load_goals()
            self.daily_logs = self.data_manager.load_daily_logs()
            
            # State
            self.current_month = self.settings.get('current_month', 1)
            self.accessibility_mode = self.settings.get('accessibility_mode', 'normal')
            
            # Setup UI
            self.setWindowTitle("🌱 Ultimate Life Planner Studio")
            self.setMinimumSize(1400, 900)
            self.setup_ui()
            self.apply_accessibility_mode()
            
            # Check ComfyUI connection
            self.check_comfyui_status()
        
        def setup_ui(self):
            """Setup the main UI"""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QHBoxLayout(central_widget)
            
            # Left sidebar
            sidebar = self.create_sidebar()
            main_layout.addWidget(sidebar, 1)
            
            # Main content area
            self.content_stack = QStackedWidget()
            main_layout.addWidget(self.content_stack, 4)
            
            # Add pages
            self.content_stack.addWidget(self.create_dashboard_page())      # 0
            self.content_stack.addWidget(self.create_vision_board_page())   # 1
            self.content_stack.addWidget(self.create_goals_page())          # 2
            self.content_stack.addWidget(self.create_progress_page())       # 3
            self.content_stack.addWidget(self.create_predictions_page())    # 4
            self.content_stack.addWidget(self.create_video_page())          # 5
            self.content_stack.addWidget(self.create_settings_page())       # 6
            
            # Status bar
            self.statusBar().showMessage("Ready | φ = 1.618033988...")
            
            # ComfyUI status indicator
            self.comfy_status = QLabel("ComfyUI: Checking...")
            self.statusBar().addPermanentWidget(self.comfy_status)
        
        def create_sidebar(self) -> QWidget:
            """Create navigation sidebar"""
            sidebar = QFrame()
            sidebar.setObjectName("sidebar")
            sidebar.setMaximumWidth(250)
            
            layout = QVBoxLayout(sidebar)
            layout.setSpacing(10)
            
            # Logo/Title
            title = QLabel("🌱 Life Planner")
            title.setObjectName("sidebarTitle")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Navigation buttons
            nav_items = [
                ("📊 Dashboard", 0),
                ("🎯 Vision Board", 1),
                ("✅ Goals", 2),
                ("📈 Progress", 3),
                ("🔮 Predictions", 4),
                ("🎬 Create Video", 5),
                ("⚙️ Settings", 6),
            ]
            
            for text, index in nav_items:
                btn = QPushButton(text)
                btn.setObjectName("navButton")
                btn.clicked.connect(lambda checked, i=index: self.content_stack.setCurrentIndex(i))
                layout.addWidget(btn)
            
            layout.addStretch()
            
            # Quick stats
            stats_group = QGroupBox("Quick Stats")
            stats_layout = QVBoxLayout(stats_group)
            
            self.month_label = QLabel(f"Month: {self.current_month}/24")
            self.energy_label = QLabel("Energy: --")
            self.phase_label = QLabel("Phase: --")
            
            stats_layout.addWidget(self.month_label)
            stats_layout.addWidget(self.energy_label)
            stats_layout.addWidget(self.phase_label)
            
            layout.addWidget(stats_group)
            
            # Audio controls
            audio_group = QGroupBox("Focus Sounds")
            audio_layout = QVBoxLayout(audio_group)
            
            for name, freq in [("🌍 Brown (Ground)", "brown"), 
                              ("🌸 Pink (Focus)", "pink"),
                              ("🌿 Green (Clarity)", "green")]:
                btn = QPushButton(name)
                btn.clicked.connect(lambda checked, t=freq: self.play_focus_sound(t))
                audio_layout.addWidget(btn)
            
            layout.addWidget(audio_group)
            
            self.update_stats_display()
            
            return sidebar
        
        def create_dashboard_page(self) -> QWidget:
            """Create main dashboard"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setSpacing(20)
            
            # Welcome section
            welcome = QGroupBox("Welcome to Your Life Planner")
            welcome_layout = QVBoxLayout(welcome)
            
            name = self.settings.get('name', 'Friend')
            self.welcome_label = QLabel(f"Hello, {name}! 🌟")
            self.welcome_label.setObjectName("welcomeLabel")
            welcome_layout.addWidget(self.welcome_label)
            
            big_why = self.settings.get('big_why', '')
            if big_why:
                why_label = QLabel(f'Your Why: "{big_why}"')
                why_label.setWordWrap(True)
                welcome_layout.addWidget(why_label)
            
            layout.addWidget(welcome)
            
            # Life Energy Display
            energy_group = QGroupBox("Life Energy (Ancient Mathematics)")
            energy_layout = QGridLayout(energy_group)
            
            life_energy = self.math_engine.life_energy_score(self.milestones, self.current_month)
            
            energy_layout.addWidget(QLabel("Energy Score:"), 0, 0)
            energy_bar = QProgressBar()
            energy_bar.setValue(int(life_energy['energy']))
            energy_bar.setStyleSheet("QProgressBar::chunk { background-color: #7C9A92; }")
            energy_layout.addWidget(energy_bar, 0, 1)
            
            energy_layout.addWidget(QLabel("Phase:"), 1, 0)
            energy_layout.addWidget(QLabel(life_energy['phase'].title()), 1, 1)
            
            energy_layout.addWidget(QLabel("Sacred Number:"), 2, 0)
            energy_layout.addWidget(QLabel(str(life_energy['sacred_number'])), 2, 1)
            
            energy_layout.addWidget(QLabel("Mayan ID:"), 3, 0)
            energy_layout.addWidget(QLabel(life_energy['mayan_id']), 3, 1)
            
            energy_layout.addWidget(QLabel("Golden Alignment:"), 4, 0)
            energy_layout.addWidget(QLabel(f"{life_energy['golden_alignment']:.3f}"), 4, 1)
            
            layout.addWidget(energy_group)
            
            # AI Insights
            insights_group = QGroupBox("🤖 AI Insights (Swarm Intelligence)")
            insights_layout = QVBoxLayout(insights_group)
            
            # Burnout risk
            activity = {
                'goals_completed': sum(1 for g in self.goals if g.completed),
                'days_active': max(len(self.daily_logs), 1),
                'challenges': sum(len(l.challenges) for l in self.daily_logs)
            }
            burnout = self.swarm.predict_burnout_risk(activity)
            
            risk_colors = {'low': '#81C784', 'medium': '#FFE66D', 'high': '#FF6B6B', 'low-activity': '#64B5F6'}
            burnout_label = QLabel(f"Wellbeing Status: {burnout['risk'].upper()}")
            burnout_label.setStyleSheet(f"color: {risk_colors.get(burnout['risk'], '#FFF')}; font-weight: bold;")
            insights_layout.addWidget(burnout_label)
            insights_layout.addWidget(QLabel(burnout['suggestion']))
            
            # Priority suggestions
            phase = 1 if self.current_month <= 4 else 2 if self.current_month <= 10 else 3 if self.current_month <= 16 else 4
            completed_cats = {g.category: g.completed for g in self.goals}
            priorities = self.swarm.vote_on_priority(['mental', 'financial', 'career', 'living'],
                                                    {'phase': phase, 'completed': completed_cats})
            
            insights_layout.addWidget(QLabel("\n🎯 Recommended Focus:"))
            for cat, score in priorities[:2]:
                insights_layout.addWidget(QLabel(f"  • {cat.title()} ({score*100:.1f}%)"))
            
            layout.addWidget(insights_group)
            
            # Milestones overview
            milestones_group = QGroupBox("🏆 Milestone Progress")
            milestones_layout = QVBoxLayout(milestones_group)
            
            completed = sum(1 for m in self.milestones if m.completed)
            total = len(self.milestones)
            
            progress_bar = QProgressBar()
            progress_bar.setValue(int(completed / total * 100))
            progress_bar.setFormat(f"{completed}/{total} milestones completed")
            milestones_layout.addWidget(progress_bar)
            
            # Next milestone
            next_milestone = next((m for m in self.milestones if not m.completed), None)
            if next_milestone:
                milestones_layout.addWidget(QLabel(f"\nNext: {next_milestone.title}"))
                milestones_layout.addWidget(QLabel(f"Target: Month {next_milestone.target_month}"))
            
            layout.addWidget(milestones_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_vision_board_page(self) -> QWidget:
            """Create vision board with AI generation"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            # Header
            header = QLabel("🎯 Vision Board - Visualize Your Future")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # AI Generation section
            ai_group = QGroupBox("🎨 AI Image Generation (ComfyUI)")
            ai_layout = QVBoxLayout(ai_group)
            
            self.vision_prompt = QLineEdit()
            self.vision_prompt.setPlaceholderText("Describe your vision... (e.g., 'cozy apartment with plants and sunlight')")
            ai_layout.addWidget(self.vision_prompt)
            
            generate_btn = QPushButton("✨ Generate Vision Image")
            generate_btn.clicked.connect(self.generate_vision_image)
            ai_layout.addWidget(generate_btn)
            
            self.generated_image_label = QLabel("Generated images will appear here")
            self.generated_image_label.setMinimumHeight(200)
            self.generated_image_label.setAlignment(Qt.AlignCenter)
            self.generated_image_label.setStyleSheet("border: 2px dashed #7C9A92; border-radius: 10px;")
            ai_layout.addWidget(self.generated_image_label)
            
            layout.addWidget(ai_group)
            
            # Vision tiles
            tiles_group = QGroupBox("Vision Tiles (Click to customize)")
            tiles_layout = QGridLayout(tiles_group)
            
            vision_items = [
                ('🏠', 'My Future Home'),
                ('💼', 'Dream Workspace'),
                ('🌿', 'Peaceful Space'),
                ('🏆', 'Achievement'),
                ('💝', 'Connections'),
                ('🎨', 'Creativity'),
                ('🚗', 'Freedom'),
                ('📚', 'Growth'),
                ('✨', 'Joy'),
            ]
            
            for i, (icon, label) in enumerate(vision_items):
                tile = QPushButton(f"{icon}\n{label}")
                tile.setMinimumSize(150, 150)
                tile.setStyleSheet("""
                    QPushButton {
                        background-color: #F5F1EB;
                        border: 2px solid #E8E4DC;
                        border-radius: 10px;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #E8E4DC;
                    }
                """)
                tiles_layout.addWidget(tile, i // 3, i % 3)
            
            layout.addWidget(tiles_group)
            
            # Affirmations
            affirmations_group = QGroupBox("✨ Affirmations")
            aff_layout = QVBoxLayout(affirmations_group)
            
            affirmations = [
                "I am capable of building the life I want",
                "My unique brain is an asset",
                "Progress over perfection",
                "I deserve independence and peace",
                "Each small step moves me forward"
            ]
            
            for aff in affirmations:
                label = QLabel(f'"{aff}"')
                label.setStyleSheet("font-style: italic; padding: 5px; background: #F8F6F3; border-radius: 5px;")
                aff_layout.addWidget(label)
            
            layout.addWidget(affirmations_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_goals_page(self) -> QWidget:
            """Create goals management page"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            header = QLabel(f"✅ Month {self.current_month} Goals")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # Month selector
            month_layout = QHBoxLayout()
            month_layout.addWidget(QLabel("Current Month:"))
            
            self.month_spin = QSpinBox()
            self.month_spin.setRange(1, 24)
            self.month_spin.setValue(self.current_month)
            self.month_spin.valueChanged.connect(self.on_month_changed)
            month_layout.addWidget(self.month_spin)
            
            month_layout.addStretch()
            layout.addLayout(month_layout)
            
            # Goals by category
            categories = ['mental', 'financial', 'career', 'living']
            category_icons = {'mental': '🧠', 'financial': '💰', 'career': '💼', 'living': '🏠'}
            
            goals_layout = QGridLayout()
            
            for i, cat in enumerate(categories):
                group = QGroupBox(f"{category_icons[cat]} {cat.title()}")
                group_layout = QVBoxLayout(group)
                
                # Goal input
                goal_input = QTextEdit()
                goal_input.setMaximumHeight(80)
                goal_input.setPlaceholderText(f"Your {cat} goal for this month...")
                group_layout.addWidget(goal_input)
                
                # Suggestions
                suggestions = self.get_goal_suggestions(cat)
                if suggestions:
                    group_layout.addWidget(QLabel("Suggestions:"))
                    for s in suggestions[:3]:
                        group_layout.addWidget(QLabel(f"  • {s}"))
                
                # Complete checkbox
                complete_check = QCheckBox("Mark as complete")
                complete_check.toggled.connect(lambda checked, c=cat: self.on_goal_completed(c, checked))
                group_layout.addWidget(complete_check)
                
                goals_layout.addWidget(group, i // 2, i % 2)
            
            layout.addLayout(goals_layout)
            
            # Daily log
            log_group = QGroupBox("📝 Daily Check-in")
            log_layout = QGridLayout(log_group)
            
            log_layout.addWidget(QLabel("Mood (1-10):"), 0, 0)
            self.mood_slider = QSlider(Qt.Horizontal)
            self.mood_slider.setRange(1, 10)
            self.mood_slider.setValue(5)
            log_layout.addWidget(self.mood_slider, 0, 1)
            
            log_layout.addWidget(QLabel("Energy (1-10):"), 1, 0)
            self.energy_slider = QSlider(Qt.Horizontal)
            self.energy_slider.setRange(1, 10)
            self.energy_slider.setValue(5)
            log_layout.addWidget(self.energy_slider, 1, 1)
            
            log_layout.addWidget(QLabel("Today's win:"), 2, 0)
            self.win_input = QLineEdit()
            log_layout.addWidget(self.win_input, 2, 1)
            
            save_log_btn = QPushButton("Save Daily Log")
            save_log_btn.clicked.connect(self.save_daily_log)
            log_layout.addWidget(save_log_btn, 3, 0, 1, 2)
            
            layout.addWidget(log_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_progress_page(self) -> QWidget:
            """Create progress tracking page"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            header = QLabel("📈 Progress & Milestones")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # Milestones list
            milestones_group = QGroupBox("🏆 Milestones")
            milestones_layout = QVBoxLayout(milestones_group)
            
            self.milestones_list = QListWidget()
            self.refresh_milestones_list()
            self.milestones_list.itemClicked.connect(self.on_milestone_clicked)
            milestones_layout.addWidget(self.milestones_list)
            
            layout.addWidget(milestones_group)
            
            # Progress bars
            progress_group = QGroupBox("📊 Overall Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            # Timeline progress
            timeline_progress = QProgressBar()
            timeline_progress.setValue(int(self.current_month / 24 * 100))
            timeline_progress.setFormat(f"Timeline: {self.current_month}/24 months")
            progress_layout.addWidget(timeline_progress)
            
            # Milestones progress
            completed = sum(1 for m in self.milestones if m.completed)
            milestones_progress = QProgressBar()
            milestones_progress.setValue(int(completed / len(self.milestones) * 100))
            milestones_progress.setFormat(f"Milestones: {completed}/{len(self.milestones)}")
            progress_layout.addWidget(milestones_progress)
            
            # Goals progress
            goals_completed = sum(1 for g in self.goals if g.completed)
            goals_total = max(len(self.goals), 1)
            goals_progress = QProgressBar()
            goals_progress.setValue(int(goals_completed / goals_total * 100))
            goals_progress.setFormat(f"Goals: {goals_completed}/{goals_total}")
            progress_layout.addWidget(goals_progress)
            
            layout.addWidget(progress_group)
            
            # Pattern analysis
            if self.daily_logs:
                patterns_group = QGroupBox("🔍 Pattern Analysis")
                patterns_layout = QVBoxLayout(patterns_group)
                
                pattern = self.swarm.emergent_pattern(self.daily_logs)
                patterns_layout.addWidget(QLabel(f"Pattern: {pattern['pattern']}"))
                patterns_layout.addWidget(QLabel(pattern['insight']))
                patterns_layout.addWidget(QLabel(f"Confidence: {pattern['confidence']*100:.0f}%"))
                
                layout.addWidget(patterns_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_predictions_page(self) -> QWidget:
            """Create ML predictions and decision tree page"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            header = QLabel("🔮 Predictions & Decision Trees")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # Current prediction
            predict_group = QGroupBox("Today's Prediction")
            predict_layout = QVBoxLayout(predict_group)
            
            # Get prediction
            features = {
                'energy': self.daily_logs[-1].energy if self.daily_logs else 5,
                'mood': self.daily_logs[-1].mood if self.daily_logs else 5,
                'streak': len([l for l in self.daily_logs[-7:] if l.tasks_completed > 0]),
                'difficulty': 5
            }
            prediction = self.ml_engine.predict_outcome(features)
            
            predict_layout.addWidget(QLabel(f"Outcome: {prediction.outcome.replace('_', ' ').title()}"))
            
            prob_bar = QProgressBar()
            prob_bar.setValue(int(prediction.probability * 100))
            prob_bar.setFormat(f"Probability: {prediction.probability*100:.0f}%")
            predict_layout.addWidget(prob_bar)
            
            predict_layout.addWidget(QLabel(f"\n💡 {prediction.recommendation}"))
            
            layout.addWidget(predict_group)
            
            # Decision tree visualization
            tree_group = QGroupBox("🌳 Decision Tree")
            tree_layout = QVBoxLayout(tree_group)
            
            self.decision_tree = QTreeWidget()
            self.decision_tree.setHeaderLabels(["Decision", "Action/Next"])
            self.decision_tree.setColumnCount(2)
            
            # Build sample decision tree
            root = QTreeWidgetItem(["Do I have enough energy?", ""])
            yes_energy = QTreeWidgetItem(["YES", "→ Check priority alignment"])
            no_energy = QTreeWidgetItem(["NO", "→ Is task energy-restoring?"])
            
            yes_aligned = QTreeWidgetItem(["Aligned with phase?", ""])
            yes_aligned_yes = QTreeWidgetItem(["YES", "✅ DO IT NOW"])
            yes_aligned_no = QTreeWidgetItem(["NO", "⏰ Defer if not urgent"])
            yes_aligned.addChildren([yes_aligned_yes, yes_aligned_no])
            
            no_restoring = QTreeWidgetItem(["Energy-restoring?", ""])
            no_restoring_yes = QTreeWidgetItem(["YES", "✅ Do it - may boost energy"])
            no_restoring_no = QTreeWidgetItem(["NO", "😴 REST first"])
            no_restoring.addChildren([no_restoring_yes, no_restoring_no])
            
            yes_energy.addChild(yes_aligned)
            no_energy.addChild(no_restoring)
            root.addChildren([yes_energy, no_energy])
            
            self.decision_tree.addTopLevelItem(root)
            self.decision_tree.expandAll()
            
            tree_layout.addWidget(self.decision_tree)
            layout.addWidget(tree_group)
            
            # Future outcomes
            outcomes_group = QGroupBox("🔮 Potential Outcomes (Next 30 Days)")
            outcomes_layout = QVBoxLayout(outcomes_group)
            
            scenarios = [
                ("If you maintain current pace", "On track for Month " + str(min(self.current_month + 1, 24)) + " goals"),
                ("If you increase focus on career", "Could accelerate income milestone by 2 weeks"),
                ("If energy drops below 4", "Risk of falling behind - prioritize self-care"),
            ]
            
            for scenario, outcome in scenarios:
                outcomes_layout.addWidget(QLabel(f"📌 {scenario}:"))
                outcomes_layout.addWidget(QLabel(f"   → {outcome}"))
                outcomes_layout.addWidget(QLabel(""))
            
            layout.addWidget(outcomes_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_video_page(self) -> QWidget:
            """Create video generation page"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            header = QLabel("🎬 Create Motivational Videos")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # Video options
            options_group = QGroupBox("Video Types")
            options_layout = QVBoxLayout(options_group)
            
            # Golden Spiral video
            spiral_btn = QPushButton("🌀 Generate Golden Spiral Progress Video")
            spiral_btn.clicked.connect(lambda: self.generate_video('spiral'))
            options_layout.addWidget(spiral_btn)
            options_layout.addWidget(QLabel("   Visualize your milestone journey on the golden spiral"))
            
            # Vision Board video
            vision_btn = QPushButton("🎯 Generate Vision Board Animation")
            vision_btn.clicked.connect(lambda: self.generate_video('vision'))
            options_layout.addWidget(vision_btn)
            options_layout.addWidget(QLabel("   Animated slideshow of your vision with affirmations"))
            
            # Timeline video
            timeline_btn = QPushButton("📅 Generate Timeline Video")
            timeline_btn.clicked.connect(lambda: self.generate_video('timeline'))
            options_layout.addWidget(timeline_btn)
            options_layout.addWidget(QLabel("   Watch your 24-month journey unfold"))
            
            layout.addWidget(options_group)
            
            # Settings
            settings_group = QGroupBox("Video Settings")
            settings_layout = QGridLayout(settings_group)
            
            settings_layout.addWidget(QLabel("Resolution:"), 0, 0)
            self.resolution_combo = QComboBox()
            self.resolution_combo.addItems(["1920x1080 (Full HD)", "1280x720 (HD)", "3840x2160 (4K)"])
            settings_layout.addWidget(self.resolution_combo, 0, 1)
            
            settings_layout.addWidget(QLabel("Duration (seconds):"), 1, 0)
            self.duration_spin = QSpinBox()
            self.duration_spin.setRange(5, 120)
            self.duration_spin.setValue(20)
            settings_layout.addWidget(self.duration_spin, 1, 1)
            
            settings_layout.addWidget(QLabel("FPS:"), 2, 0)
            self.fps_combo = QComboBox()
            self.fps_combo.addItems(["30", "24", "60"])
            settings_layout.addWidget(self.fps_combo, 2, 1)
            
            layout.addWidget(settings_group)
            
            # Progress
            self.video_progress = QProgressBar()
            self.video_progress.setValue(0)
            self.video_progress.setVisible(False)
            layout.addWidget(self.video_progress)
            
            self.video_status = QLabel("")
            layout.addWidget(self.video_status)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        def create_settings_page(self) -> QWidget:
            """Create settings page"""
            page = QScrollArea()
            page.setWidgetResizable(True)
            
            content = QWidget()
            layout = QVBoxLayout(content)
            
            header = QLabel("⚙️ Settings")
            header.setObjectName("pageHeader")
            layout.addWidget(header)
            
            # Personal info
            personal_group = QGroupBox("Personal Information")
            personal_layout = QGridLayout(personal_group)
            
            personal_layout.addWidget(QLabel("Your Name:"), 0, 0)
            self.name_input = QLineEdit(self.settings.get('name', ''))
            personal_layout.addWidget(self.name_input, 0, 1)
            
            personal_layout.addWidget(QLabel("Your Big Why:"), 1, 0)
            self.why_input = QTextEdit()
            self.why_input.setPlainText(self.settings.get('big_why', ''))
            self.why_input.setMaximumHeight(100)
            personal_layout.addWidget(self.why_input, 1, 1)
            
            layout.addWidget(personal_group)
            
            # Accessibility
            access_group = QGroupBox("🧠 Accessibility")
            access_layout = QVBoxLayout(access_group)
            
            access_layout.addWidget(QLabel("Interface Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Simple (Fewer options)", "Normal (Balanced)", "Advanced (All features)"])
            self.mode_combo.setCurrentText({"simple": "Simple (Fewer options)", 
                                           "normal": "Normal (Balanced)",
                                           "advanced": "Advanced (All features)"}.get(self.accessibility_mode, "Normal (Balanced)"))
            self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
            access_layout.addWidget(self.mode_combo)
            
            self.audio_check = QCheckBox("Enable Audio Feedback")
            self.audio_check.setChecked(self.settings.get('audio_enabled', True))
            access_layout.addWidget(self.audio_check)
            
            layout.addWidget(access_group)
            
            # ComfyUI
            comfy_group = QGroupBox("🎨 ComfyUI Connection")
            comfy_layout = QGridLayout(comfy_group)
            
            comfy_layout.addWidget(QLabel("Host:"), 0, 0)
            self.comfy_host = QLineEdit(self.settings.get('comfyui_host', '127.0.0.1'))
            comfy_layout.addWidget(self.comfy_host, 0, 1)
            
            comfy_layout.addWidget(QLabel("Port:"), 1, 0)
            self.comfy_port = QSpinBox()
            self.comfy_port.setRange(1, 65535)
            self.comfy_port.setValue(self.settings.get('comfyui_port', 8188))
            comfy_layout.addWidget(self.comfy_port, 1, 1)
            
            test_comfy_btn = QPushButton("Test Connection")
            test_comfy_btn.clicked.connect(self.check_comfyui_status)
            comfy_layout.addWidget(test_comfy_btn, 2, 0, 1, 2)
            
            layout.addWidget(comfy_group)
            
            # Save button
            save_btn = QPushButton("💾 Save Settings")
            save_btn.clicked.connect(self.save_settings)
            layout.addWidget(save_btn)
            
            # Export/Import
            data_group = QGroupBox("📁 Data Management")
            data_layout = QHBoxLayout(data_group)
            
            export_btn = QPushButton("Export All Data")
            export_btn.clicked.connect(self.export_data)
            data_layout.addWidget(export_btn)
            
            import_btn = QPushButton("Import Data")
            import_btn.clicked.connect(self.import_data)
            data_layout.addWidget(import_btn)
            
            layout.addWidget(data_group)
            
            layout.addStretch()
            page.setWidget(content)
            return page
        
        # ==================== HELPER METHODS ====================
        
        def update_stats_display(self):
            """Update sidebar stats"""
            self.month_label.setText(f"Month: {self.current_month}/24")
            
            energy = self.math_engine.life_energy_score(self.milestones, self.current_month)
            self.energy_label.setText(f"Energy: {energy['energy']:.0f}%")
            self.phase_label.setText(f"Phase: {energy['phase'].title()}")
        
        def get_goal_suggestions(self, category: str) -> List[str]:
            """Get AI-powered goal suggestions"""
            phase = 1 if self.current_month <= 4 else 2 if self.current_month <= 10 else 3 if self.current_month <= 16 else 4
            
            suggestions = {
                1: {
                    'mental': ['Start therapy/counseling', 'Establish daily routine', 'Begin mood tracking'],
                    'financial': ['Apply for secured credit card', 'Track spending', 'Create budget'],
                    'career': ['Update LinkedIn', 'Complete online course', 'Create portfolio piece'],
                    'living': ['Research target areas', 'Calculate costs', 'Document needs']
                },
                2: {
                    'mental': ['Continue therapy', 'Build coping strategies', 'Start wins journal'],
                    'financial': ['Check credit monthly', 'Build emergency fund', 'Research insurance'],
                    'career': ['Apply for positions', 'Network weekly', 'Build case studies'],
                    'living': ['Research apartments', 'Calculate true costs', 'Build references']
                },
                3: {
                    'mental': ['Expand support network', 'Develop resilience toolkit'],
                    'financial': ['Target credit 650+', 'Save for move-in costs'],
                    'career': ['Establish stable income', 'Document achievements'],
                    'living': ['Active apartment hunting', 'Prepare applications']
                },
                4: {
                    'mental': ['Plan moving transition', 'Build local support'],
                    'financial': ['Continue savings growth', 'Research homebuyer programs'],
                    'career': ['Seek advancement', 'Evaluate self-employment'],
                    'living': ['Secure apartment', 'Create routines']
                }
            }
            
            return suggestions.get(phase, {}).get(category, [])
        
        def refresh_milestones_list(self):
            """Refresh the milestones list widget"""
            self.milestones_list.clear()
            
            for m in self.milestones:
                status = "✅" if m.completed else "⬜"
                item = QListWidgetItem(f"{status} {m.title} (Month {m.target_month}) - Fib: {m.fibonacci_number}")
                if m.completed:
                    item.setBackground(QColor(232, 245, 233))
                self.milestones_list.addItem(item)
        
        def apply_accessibility_mode(self):
            """Apply accessibility settings"""
            base_style = """
                QMainWindow {
                    background-color: #F5F1EB;
                }
                #sidebar {
                    background-color: #2C3E50;
                    border-radius: 10px;
                }
                #sidebarTitle {
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 15px;
                }
                #navButton {
                    background-color: rgba(255,255,255,0.1);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    text-align: left;
                    font-size: 14px;
                }
                #navButton:hover {
                    background-color: rgba(255,255,255,0.2);
                }
                #pageHeader {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3D4A44;
                    padding: 10px;
                }
                #welcomeLabel {
                    font-size: 28px;
                    color: #3D4A44;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #E8E4DC;
                    border-radius: 10px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QPushButton {
                    background-color: #7C9A92;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #5D7B6F;
                }
                QProgressBar {
                    border: none;
                    border-radius: 5px;
                    background-color: #E8E4DC;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #7C9A92;
                    border-radius: 5px;
                }
            """
            
            if self.accessibility_mode == 'simple':
                base_style += """
                    * { font-size: 16px; }
                    QPushButton { padding: 15px 25px; }
                """
            elif self.accessibility_mode == 'advanced':
                base_style += """
                    * { font-size: 13px; }
                """
            
            self.setStyleSheet(base_style)
        
        # ==================== EVENT HANDLERS ====================
        
        def on_month_changed(self, value):
            """Handle month change"""
            self.current_month = value
            self.update_stats_display()
            self.settings['current_month'] = value
            self.data_manager.save_settings(self.settings)
        
        def on_mode_changed(self, text):
            """Handle accessibility mode change"""
            mode_map = {
                "Simple (Fewer options)": "simple",
                "Normal (Balanced)": "normal",
                "Advanced (All features)": "advanced"
            }
            self.accessibility_mode = mode_map.get(text, 'normal')
            self.apply_accessibility_mode()
        
        def on_goal_completed(self, category: str, completed: bool):
            """Handle goal completion"""
            if completed and self.settings.get('audio_enabled', True):
                self.audio_engine.play_celebration()
            
            # Add to ML training data
            features = {
                'energy': self.daily_logs[-1].energy if self.daily_logs else 5,
                'mood': self.daily_logs[-1].mood if self.daily_logs else 5,
                'month': self.current_month,
                'category_id': ['mental', 'financial', 'career', 'living'].index(category)
            }
            self.ml_engine.add_training_data(features, 'completed' if completed else 'incomplete')
        
        def on_milestone_clicked(self, item):
            """Handle milestone click - toggle completion"""
            index = self.milestones_list.row(item)
            if 0 <= index < len(self.milestones):
                self.milestones[index].completed = not self.milestones[index].completed
                if self.milestones[index].completed:
                    self.milestones[index].completed_date = datetime.now().strftime("%Y-%m-%d")
                    if self.settings.get('audio_enabled', True):
                        self.audio_engine.play_celebration()
                else:
                    self.milestones[index].completed_date = ""
                
                self.data_manager.save_milestones(self.milestones)
                self.refresh_milestones_list()
                self.update_stats_display()
        
        def play_focus_sound(self, sound_type: str):
            """Play therapeutic focus sound"""
            freq_map = {
                'brown': SACRED_FREQUENCIES['liberation'],
                'pink': SACRED_FREQUENCIES['transformation'],
                'green': SACRED_FREQUENCIES['expression']
            }
            
            freq = freq_map.get(sound_type, 432)
            audio = self.audio_engine.generate_tone(freq, 3.0)
            self.audio_engine.play_sound(audio)
            
            self.statusBar().showMessage(f"Playing {sound_type} noise at {freq}Hz...", 3000)
        
        def save_daily_log(self):
            """Save daily check-in log"""
            log = DailyLog(
                date=datetime.now().strftime("%Y-%m-%d"),
                mood=self.mood_slider.value(),
                energy=self.energy_slider.value(),
                wins=[self.win_input.text()] if self.win_input.text() else []
            )
            
            self.data_manager.save_daily_log(log)
            self.daily_logs = self.data_manager.load_daily_logs()
            
            QMessageBox.information(self, "Saved", "Daily log saved successfully!")
            self.win_input.clear()
        
        def save_settings(self):
            """Save all settings"""
            self.settings['name'] = self.name_input.text()
            self.settings['big_why'] = self.why_input.toPlainText()
            self.settings['accessibility_mode'] = self.accessibility_mode
            self.settings['audio_enabled'] = self.audio_check.isChecked()
            self.settings['comfyui_host'] = self.comfy_host.text()
            self.settings['comfyui_port'] = self.comfy_port.value()
            self.settings['current_month'] = self.current_month
            
            self.data_manager.save_settings(self.settings)
            
            # Update ComfyUI client
            self.comfy_client = ComfyUIClient(
                self.settings['comfyui_host'],
                self.settings['comfyui_port']
            )
            
            QMessageBox.information(self, "Saved", "Settings saved successfully!")
        
        def check_comfyui_status(self):
            """Check ComfyUI connection status"""
            if self.comfy_client.check_connection():
                self.comfy_status.setText("ComfyUI: ✅ Connected")
                self.comfy_status.setStyleSheet("color: green;")
            else:
                self.comfy_status.setText("ComfyUI: ❌ Not Connected")
                self.comfy_status.setStyleSheet("color: red;")
        
        def generate_vision_image(self):
            """Generate image using ComfyUI"""
            prompt = self.vision_prompt.text()
            if not prompt:
                QMessageBox.warning(self, "Error", "Please enter a vision description")
                return
            
            if not self.comfy_client.connected:
                QMessageBox.warning(self, "Error", "ComfyUI not connected. Check settings.")
                return
            
            self.statusBar().showMessage("Generating image...")
            
            # Run in background thread
            self.image_worker = ImageGenerationWorker(self.comfy_client, prompt)
            self.image_worker.signals.result.connect(self.on_image_generated)
            self.image_worker.signals.error.connect(lambda e: QMessageBox.warning(self, "Error", e))
            self.image_worker.signals.finished.connect(lambda: self.statusBar().showMessage("Ready"))
            self.image_worker.start()
        
        def on_image_generated(self, image_data):
            """Handle generated image"""
            if image_data:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.generated_image_label.setPixmap(scaled)
                
                # Save to file
                save_path = Path("life_planner_data/generated_images")
                save_path.mkdir(parents=True, exist_ok=True)
                filename = save_path / f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                pixmap.save(str(filename))
                
                QMessageBox.information(self, "Success", f"Image saved to {filename}")
            else:
                QMessageBox.warning(self, "Error", "Failed to generate image")
        
        def generate_video(self, video_type: str):
            """Generate video in background"""
            self.video_progress.setVisible(True)
            self.video_progress.setValue(0)
            self.video_status.setText("Generating video...")
            
            # Parse settings
            resolution = self.resolution_combo.currentText()
            if "1920" in resolution:
                width, height = 1920, 1080
            elif "1280" in resolution:
                width, height = 1280, 720
            else:
                width, height = 3840, 2160
            
            duration = self.duration_spin.value()
            fps = int(self.fps_combo.currentText())
            
            self.video_engine = VideoAnimationEngine(width, height, fps)
            
            output_path = f"life_planner_data/videos/{video_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            Path("life_planner_data/videos").mkdir(parents=True, exist_ok=True)
            
            kwargs = {'duration': duration, 'output_path': output_path}
            
            if video_type == 'spiral':
                kwargs['milestones'] = self.milestones
            elif video_type == 'vision':
                kwargs['images'] = []  # Would be populated with vision board images
                kwargs['affirmations'] = [
                    "I am capable of building the life I want",
                    "Progress over perfection",
                    "Each small step moves me forward"
                ]
            elif video_type == 'timeline':
                kwargs['monthly_data'] = []  # Would be populated with monthly data
            
            self.video_worker = VideoGenerationWorker(self.video_engine, video_type, **kwargs)
            self.video_worker.signals.result.connect(self.on_video_generated)
            self.video_worker.signals.error.connect(lambda e: self.video_status.setText(f"Error: {e}"))
            self.video_worker.signals.finished.connect(lambda: self.video_progress.setVisible(False))
            self.video_worker.start()
        
        def on_video_generated(self, output_path):
            """Handle generated video"""
            if output_path:
                self.video_status.setText(f"✅ Video saved to: {output_path}")
                QMessageBox.information(self, "Success", f"Video generated!\n\nSaved to: {output_path}")
            else:
                self.video_status.setText("❌ Video generation failed")
        
        def export_data(self):
            """Export all data to JSON"""
            filename, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "JSON Files (*.json)")
            if filename:
                data = {
                    'settings': self.settings,
                    'milestones': [asdict(m) for m in self.milestones],
                    'goals': [asdict(g) for g in self.goals],
                    'daily_logs': [asdict(l) for l in self.daily_logs],
                    'exported_at': datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, "Exported", f"Data exported to {filename}")
        
        def import_data(self):
            """Import data from JSON"""
            filename, _ = QFileDialog.getOpenFileName(self, "Import Data", "", "JSON Files (*.json)")
            if filename:
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    
                    if 'settings' in data:
                        self.settings = data['settings']
                        self.data_manager.save_settings(self.settings)
                    
                    if 'milestones' in data:
                        self.milestones = [Milestone(**m) for m in data['milestones']]
                        self.data_manager.save_milestones(self.milestones)
                    
                    if 'goals' in data:
                        self.goals = [Goal(**g) for g in data['goals']]
                        self.data_manager.save_goals(self.goals)
                    
                    QMessageBox.information(self, "Imported", "Data imported successfully!")
                    
                    # Refresh UI
                    self.refresh_milestones_list()
                    self.update_stats_display()
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Import failed: {e}")
        
        def closeEvent(self, event):
            """Save data on close"""
            self.settings['current_month'] = self.current_month
            self.data_manager.save_settings(self.settings)
            self.data_manager.save_milestones(self.milestones)
            self.data_manager.save_goals(self.goals)
            event.accept()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ULTIMATE LIFE PLANNER STUDIO - VISION TO REALITY ENGINE            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Golden Ratio: φ = 1.618033988749895                                         ║
║  Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...                    ║
║  Sacred Frequencies: 396, 417, 528, 639, 741, 852 Hz                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not PYSIDE6_OK:
        print("\n[ERROR] PySide6 is required for the GUI.")
        print("Install with: pip install PySide6")
        print("\nAlternatively, run in CLI mode with: python life_planner_studio.py --cli")
        return 1
    
    app = QApplication(sys.argv)
    app.setApplicationName("Life Planner Studio")
    app.setOrganizationName("AncientMathematics")
    
    # Set fusion style for consistent cross-platform look
    app.setStyle("Fusion")
    
    window = LifePlannerStudio()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
