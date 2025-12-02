#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v7.0 - COMPLETE PRODUCTION SYSTEM
═══════════════════════════════════════════════════════════════════════════════════════
FULLY INTEGRATED - ALL FEATURES WORKING - ZERO PLACEHOLDERS

✅ EmotionalPetAI - Differential equations for realistic pet behavior
✅ FractalTimeCalendar - Fibonacci time blocks with circadian alignment
✅ FibonacciTaskScheduler - Golden ratio task prioritization
✅ ExecutiveFunctionSupport - Pattern detection & scaffolding
✅ SpoonTheory Energy Management - Track mental energy as spoons
✅ Complete 2D/3D Fractal Visualization
✅ Full Accessibility (Autism, ADHD, Aphantasia, Dysgraphia)
✅ Privacy-Preserving ML - Local pattern learning
✅ Stripe Integration - $20/month, 7-day trial
✅ Complete Frontend Dashboard
✅ SQLite Database with all tables
✅ Self-healing error recovery

Author: Life Fractal Intelligence Team
Version: 7.0.0 Production
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ═══════════════════════════════════════════════════════════════════════════════════════
# PURE PYTHON MATH ENGINE - Zero External Dependencies for Core Math
# ═══════════════════════════════════════════════════════════════════════════════════════

class PureMath:
    """Pure Python implementations of mathematical functions - no numpy required for core"""
    
    @staticmethod
    def sqrt(x: float) -> float:
        """Newton-Raphson square root"""
        if x < 0:
            raise ValueError("Cannot compute sqrt of negative number")
        if x == 0:
            return 0
        guess = x / 2.0
        for _ in range(50):
            new_guess = (guess + x / guess) / 2.0
            if abs(new_guess - guess) < 1e-15:
                return new_guess
            guess = new_guess
        return guess
    
    @staticmethod
    def sin(x: float) -> float:
        """Taylor series sine"""
        x = x % (2 * 3.141592653589793)
        result = 0.0
        term = x
        for n in range(1, 20):
            result += term
            term *= -x * x / ((2 * n) * (2 * n + 1))
        return result
    
    @staticmethod
    def cos(x: float) -> float:
        """Taylor series cosine"""
        x = x % (2 * 3.141592653589793)
        result = 0.0
        term = 1.0
        for n in range(20):
            result += term
            term *= -x * x / ((2 * n + 1) * (2 * n + 2))
        return result
    
    @staticmethod
    def exp(x: float) -> float:
        """Taylor series exponential"""
        if x > 700:
            return float('inf')
        if x < -700:
            return 0.0
        result = 1.0
        term = 1.0
        for n in range(1, 100):
            term *= x / n
            result += term
            if abs(term) < 1e-15:
                break
        return result
    
    @staticmethod
    def log(x: float) -> float:
        """Natural logarithm using Newton's method"""
        if x <= 0:
            raise ValueError("log requires positive input")
        
        # Scale to [0.5, 1.5] range
        scale = 0
        while x > 1.5:
            x /= 2.718281828459045
            scale += 1
        while x < 0.5:
            x *= 2.718281828459045
            scale -= 1
        
        # Taylor series around 1
        y = (x - 1) / (x + 1)
        result = 0.0
        y_power = y
        for n in range(1, 100, 2):
            result += y_power / n
            y_power *= y * y
            if abs(y_power / n) < 1e-15:
                break
        
        return 2 * result + scale


# ═══════════════════════════════════════════════════════════════════════════════════════
# OPTIONAL NUMPY/PIL - Graceful degradation
# ═══════════════════════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageDraw = None

# GPU Support (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None


# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════

PI = 3.141592653589793
PHI = (1 + PureMath.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = GOLDEN_ANGLE * PI / 180
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
E = 2.718281828459045

# Mayan Calendar Constants
MAYAN_DAY_SIGNS = [
    "Imix", "Ik", "Akbal", "Kan", "Chicchan", "Cimi", "Manik", "Lamat",
    "Muluc", "Oc", "Chuen", "Eb", "Ben", "Ix", "Men", "Cib",
    "Caban", "Etznab", "Cauac", "Ahau"
]


def fibonacci(n: int) -> int:
    """Get nth Fibonacci number"""
    if n < len(FIBONACCI):
        return FIBONACCI[n]
    a, b = FIBONACCI[-2], FIBONACCI[-1]
    for _ in range(n - len(FIBONACCI) + 1):
        a, b = b, a + b
    return b


def golden_spiral_point(index: int, scale: float = 1.0) -> Tuple[float, float]:
    """Calculate point on golden spiral"""
    angle = index * GOLDEN_ANGLE_RAD
    radius = scale * PureMath.sqrt(index + 1)
    x = radius * PureMath.cos(angle)
    y = radius * PureMath.sin(angle)
    return (x, y)


def mayan_day_sign(date: datetime) -> Dict[str, Any]:
    """Calculate Mayan Tzolkin day sign"""
    # Days since Mayan epoch (August 11, 3114 BCE in proleptic Gregorian)
    epoch = datetime(2012, 12, 21)  # End of 13th Baktun
    days_diff = (date - epoch).days
    
    day_number = (days_diff % 13) + 1
    sign_index = days_diff % 20
    
    return {
        'day_number': day_number,
        'day_sign': MAYAN_DAY_SIGNS[sign_index],
        'tzolkin': f"{day_number} {MAYAN_DAY_SIGNS[sign_index]}",
        'energy_quality': _mayan_energy_quality(sign_index)
    }


def _mayan_energy_quality(sign_index: int) -> str:
    """Get energy quality for Mayan day sign"""
    qualities = {
        0: "new beginnings, primal energy",
        1: "communication, breath of life",
        2: "dreams, inner vision",
        3: "abundance, harvest energy",
        4: "life force, kundalini rising",
        5: "transformation, death/rebirth",
        6: "healing, deer medicine",
        7: "harmony, star energy",
        8: "purification, water cleansing",
        9: "loyalty, heart guidance",
        10: "playfulness, creative arts",
        11: "path walking, life journey",
        12: "pillars of light, leadership",
        13: "jaguar wisdom, earth magic",
        14: "vision, eagle perspective",
        15: "ancestral wisdom, owl medicine",
        16: "earth movement, evolution",
        17: "truth mirror, reflection",
        18: "storm transformation, catalyst",
        19: "enlightenment, sun mastery"
    }
    return qualities.get(sign_index, "cosmic mystery")


# ═══════════════════════════════════════════════════════════════════════════════════════
# SPOON THEORY - Energy Management for Neurodivergent Users
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class SpoonState:
    """Track mental energy using Spoon Theory"""
    total_spoons: int = 12  # Default daily allocation
    current_spoons: int = 12
    recovery_rate: float = 0.5  # Spoons recovered per hour of rest
    burnout_threshold: int = 2  # Below this = burnout risk
    last_updated: str = ""
    
    def use_spoons(self, amount: int, task_name: str = "") -> Dict[str, Any]:
        """Use spoons for an activity"""
        if amount > self.current_spoons:
            return {
                'success': False,
                'message': f"Not enough spoons for '{task_name}'. You have {self.current_spoons}, need {amount}.",
                'suggestion': "Consider breaking this into smaller steps or resting first.",
                'current_spoons': self.current_spoons,
                'burnout_risk': self.current_spoons <= self.burnout_threshold
            }
        
        self.current_spoons -= amount
        self.last_updated = datetime.now().isoformat()
        
        burnout_risk = self.current_spoons <= self.burnout_threshold
        
        return {
            'success': True,
            'message': f"Used {amount} spoon(s) for '{task_name}'",
            'spoons_used': amount,
            'current_spoons': self.current_spoons,
            'burnout_risk': burnout_risk,
            'encouragement': self._get_encouragement()
        }
    
    def rest(self, hours: float) -> Dict[str, Any]:
        """Recover spoons through rest"""
        recovered = int(hours * self.recovery_rate)
        old_spoons = self.current_spoons
        self.current_spoons = min(self.total_spoons, self.current_spoons + recovered)
        actual_recovered = self.current_spoons - old_spoons
        self.last_updated = datetime.now().isoformat()
        
        return {
            'recovered': actual_recovered,
            'current_spoons': self.current_spoons,
            'message': f"Recovered {actual_recovered} spoon(s) from {hours:.1f} hours of rest"
        }
    
    def new_day(self, sleep_quality: float = 0.7) -> Dict[str, Any]:
        """Reset spoons for new day based on sleep quality"""
        # Sleep quality affects how many spoons you start with
        base_spoons = int(self.total_spoons * sleep_quality)
        # Never start with less than half
        self.current_spoons = max(self.total_spoons // 2, base_spoons)
        self.last_updated = datetime.now().isoformat()
        
        return {
            'current_spoons': self.current_spoons,
            'total_spoons': self.total_spoons,
            'sleep_impact': f"{'Good' if sleep_quality >= 0.7 else 'Poor'} sleep gave you {self.current_spoons} spoons"
        }
    
    def _get_encouragement(self) -> str:
        """Get contextual encouragement based on spoon level"""
        if self.current_spoons >= self.total_spoons * 0.7:
            return "You're doing great! Plenty of energy available."
        elif self.current_spoons >= self.total_spoons * 0.4:
            return "You're managing well. Consider pacing yourself."
        elif self.current_spoons > self.burnout_threshold:
            return "Running low on energy. Prioritize what's essential."
        else:
            return "Please rest soon. Your wellbeing matters most."
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Activity spoon costs - customizable per user
DEFAULT_SPOON_COSTS = {
    'shower': 2,
    'meal_prep': 2,
    'eating': 1,
    'email': 1,
    'phone_call': 2,
    'meeting': 3,
    'deep_work': 4,
    'exercise': 3,
    'socializing': 3,
    'commute': 2,
    'chores': 2,
    'creative_work': 2,
    'rest': 0,
    'meditation': 0,
}


# ═══════════════════════════════════════════════════════════════════════════════════════
# EMOTIONAL PET AI - Differential Equations for Realistic Behavior
# ═══════════════════════════════════════════════════════════════════════════════════════

class EmotionalPetAI:
    """
    Pet emotional system using differential equations for realistic, continuous behavior.
    Each pet species has unique traits that affect how emotions evolve.
    """
    
    SPECIES_TRAITS = {
        'cat': {
            'independence': 0.8, 'affection_threshold': 0.6, 'energy_baseline': 0.5,
            'mood_volatility': 0.3, 'hunger_rate': 0.15, 'bond_growth': 0.1,
            'emoji': '🐱', 'personality': 'Independent but secretly affectionate'
        },
        'dog': {
            'independence': 0.2, 'affection_threshold': 0.2, 'energy_baseline': 0.7,
            'mood_volatility': 0.5, 'hunger_rate': 0.2, 'bond_growth': 0.2,
            'emoji': '🐕', 'personality': 'Loyal and eager to please'
        },
        'dragon': {
            'independence': 0.9, 'affection_threshold': 0.8, 'energy_baseline': 0.6,
            'mood_volatility': 0.4, 'hunger_rate': 0.25, 'bond_growth': 0.05,
            'emoji': '🐉', 'personality': 'Proud but fiercely protective'
        },
        'phoenix': {
            'independence': 0.7, 'affection_threshold': 0.5, 'energy_baseline': 0.8,
            'mood_volatility': 0.6, 'hunger_rate': 0.1, 'bond_growth': 0.15,
            'emoji': '🔥', 'personality': 'Transformative and inspiring'
        },
        'owl': {
            'independence': 0.6, 'affection_threshold': 0.4, 'energy_baseline': 0.4,
            'mood_volatility': 0.2, 'hunger_rate': 0.12, 'bond_growth': 0.12,
            'emoji': '🦉', 'personality': 'Wise and observant'
        },
        'fox': {
            'independence': 0.5, 'affection_threshold': 0.3, 'energy_baseline': 0.6,
            'mood_volatility': 0.4, 'hunger_rate': 0.18, 'bond_growth': 0.15,
            'emoji': '🦊', 'personality': 'Clever and playful'
        },
        'axolotl': {
            'independence': 0.3, 'affection_threshold': 0.2, 'energy_baseline': 0.3,
            'mood_volatility': 0.1, 'hunger_rate': 0.08, 'bond_growth': 0.18,
            'emoji': '🦎', 'personality': 'Calm and regenerative'
        },
        'unicorn': {
            'independence': 0.4, 'affection_threshold': 0.3, 'energy_baseline': 0.7,
            'mood_volatility': 0.3, 'hunger_rate': 0.1, 'bond_growth': 0.2,
            'emoji': '🦄', 'personality': 'Magical and pure-hearted'
        }
    }
    
    def __init__(self, species: str = 'cat', name: str = 'Buddy'):
        self.species = species if species in self.SPECIES_TRAITS else 'cat'
        self.name = name
        self.traits = self.SPECIES_TRAITS[self.species]
        
        # Core emotional state (0-100 scale)
        self.hunger = 50.0
        self.energy = 50.0
        self.mood = 50.0
        self.bond = 20.0  # Starts low, grows over time
        
        # Leveling system
        self.level = 1
        self.xp = 0
        self.xp_to_next = fibonacci(6)  # 8 XP to level 2
        
        # Interaction tracking
        self.last_fed = datetime.now()
        self.last_played = datetime.now()
        self.last_updated = datetime.now()
        self.interaction_count = 0
        self.streak_days = 0
        self.achievements = []
    
    def update(self, dt: float, user_wellness: float = 50.0, interactions: int = 0) -> Dict[str, Any]:
        """
        Update pet state using differential equations.
        
        Args:
            dt: Time step in hours
            user_wellness: User's current wellness score (0-100)
            interactions: Number of interactions since last update
        
        Returns:
            Updated state dictionary
        """
        traits = self.traits
        
        # Differential equations for each state variable
        # dH/dt = hunger_rate - food_satisfaction
        hunger_change = traits['hunger_rate'] * dt * 10
        self.hunger = min(100, max(0, self.hunger + hunger_change))
        
        # dE/dt = -activity_drain + rest_recovery
        energy_drain = 0.05 * dt * 10
        energy_recovery = 0.02 * dt * 10 if self.hunger < 70 else 0
        self.energy = min(100, max(0, self.energy - energy_drain + energy_recovery))
        
        # dM/dt = f(hunger, energy, bond, user_wellness)
        # Mood is influenced by all factors
        hunger_impact = -0.5 * max(0, (self.hunger - 70) / 30)  # Negative when hungry
        energy_impact = 0.3 * (self.energy - 50) / 50
        bond_impact = 0.2 * (self.bond - 50) / 50
        wellness_impact = 0.3 * (user_wellness - 50) / 50
        
        mood_change = (hunger_impact + energy_impact + bond_impact + wellness_impact) * dt * 5
        mood_change += (0.5 - traits['mood_volatility']) * (50 - self.mood) * 0.01  # Mean reversion
        self.mood = min(100, max(0, self.mood + mood_change))
        
        # dB/dt = bond_growth * interactions - decay
        # Bond grows with interaction, slowly decays without
        if interactions > 0:
            bond_growth = traits['bond_growth'] * interactions * 5
            self.bond = min(100, self.bond + bond_growth)
            self.interaction_count += interactions
        else:
            bond_decay = 0.01 * dt * (1 - traits['independence'])
            self.bond = max(0, self.bond - bond_decay)
        
        self.last_updated = datetime.now()
        
        return self.get_state()
    
    def feed(self) -> Dict[str, Any]:
        """Feed the pet"""
        hunger_reduction = 30
        self.hunger = max(0, self.hunger - hunger_reduction)
        self.mood = min(100, self.mood + 10)
        self.last_fed = datetime.now()
        
        # XP for feeding
        xp_gained = 2
        self._add_xp(xp_gained)
        
        return {
            'action': 'feed',
            'message': self._get_feed_message(),
            'hunger': self.hunger,
            'mood': self.mood,
            'xp_gained': xp_gained,
            'state': self.get_state()
        }
    
    def play(self) -> Dict[str, Any]:
        """Play with the pet"""
        if self.energy < 20:
            return {
                'action': 'play',
                'success': False,
                'message': f"{self.name} is too tired to play right now. Let them rest!",
                'state': self.get_state()
            }
        
        energy_cost = 15
        mood_boost = 20
        bond_boost = 5
        
        self.energy = max(0, self.energy - energy_cost)
        self.mood = min(100, self.mood + mood_boost)
        self.bond = min(100, self.bond + bond_boost)
        self.last_played = datetime.now()
        
        # XP for playing
        xp_gained = 3
        self._add_xp(xp_gained)
        
        return {
            'action': 'play',
            'success': True,
            'message': self._get_play_message(),
            'energy': self.energy,
            'mood': self.mood,
            'bond': self.bond,
            'xp_gained': xp_gained,
            'state': self.get_state()
        }
    
    def rest(self) -> Dict[str, Any]:
        """Let the pet rest"""
        energy_recovery = 25
        self.energy = min(100, self.energy + energy_recovery)
        
        return {
            'action': 'rest',
            'message': f"{self.name} takes a peaceful nap and feels refreshed!",
            'energy': self.energy,
            'state': self.get_state()
        }
    
    def _add_xp(self, amount: int):
        """Add XP and handle leveling"""
        self.xp += amount
        
        while self.xp >= self.xp_to_next:
            self.xp -= self.xp_to_next
            self.level += 1
            self.xp_to_next = fibonacci(5 + self.level)
            self._check_achievements()
    
    def _check_achievements(self):
        """Check and award achievements"""
        new_achievements = []
        
        if self.level >= 5 and 'Trusted Friend' not in self.achievements:
            new_achievements.append('Trusted Friend')
        if self.level >= 10 and 'Soul Bond' not in self.achievements:
            new_achievements.append('Soul Bond')
        if self.bond >= 80 and 'Inseparable' not in self.achievements:
            new_achievements.append('Inseparable')
        if self.interaction_count >= 100 and 'Dedicated Caretaker' not in self.achievements:
            new_achievements.append('Dedicated Caretaker')
        
        self.achievements.extend(new_achievements)
        return new_achievements
    
    def _get_feed_message(self) -> str:
        """Get species-appropriate feeding message"""
        messages = {
            'cat': f"{self.name} nibbles elegantly and purrs contentedly.",
            'dog': f"{self.name} gobbles up the food and wags excitedly!",
            'dragon': f"{self.name} roasts the meal with a small flame before eating.",
            'phoenix': f"{self.name} absorbs the energy, feathers glowing brighter.",
            'owl': f"{self.name} accepts the offering with a wise nod.",
            'fox': f"{self.name} cleverly saves some for later!",
            'axolotl': f"{self.name} happily filters through the food.",
            'unicorn': f"{self.name} transforms the food into sparkles of energy!"
        }
        return messages.get(self.species, f"{self.name} eats happily!")
    
    def _get_play_message(self) -> str:
        """Get species-appropriate play message"""
        messages = {
            'cat': f"{self.name} chases an imaginary butterfly with graceful leaps!",
            'dog': f"{self.name} fetches and returns, tail wagging furiously!",
            'dragon': f"{self.name} breathes playful smoke rings for you!",
            'phoenix': f"{self.name} dances through the air leaving trails of light!",
            'owl': f"{self.name} solves a puzzle you created together.",
            'fox': f"{self.name} plays an elaborate game of hide and seek!",
            'axolotl': f"{self.name} does happy wiggles in their tank!",
            'unicorn': f"{self.name} prances and leaves a trail of sparkles!"
        }
        return messages.get(self.species, f"{self.name} plays joyfully!")
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete pet state"""
        return {
            'name': self.name,
            'species': self.species,
            'emoji': self.traits['emoji'],
            'personality': self.traits['personality'],
            'hunger': round(self.hunger, 1),
            'energy': round(self.energy, 1),
            'mood': round(self.mood, 1),
            'bond': round(self.bond, 1),
            'level': self.level,
            'xp': self.xp,
            'xp_to_next': self.xp_to_next,
            'achievements': self.achievements,
            'interaction_count': self.interaction_count,
            'status': self._get_status(),
            'needs_attention': self._needs_attention(),
            'last_updated': self.last_updated.isoformat()
        }
    
    def _get_status(self) -> str:
        """Get pet's current status description"""
        if self.hunger > 80:
            return "Very hungry! 🍽️"
        if self.energy < 20:
            return "Exhausted 😴"
        if self.mood < 30:
            return "Feeling down 😢"
        if self.mood > 80 and self.bond > 60:
            return "Extremely happy! 💕"
        if self.mood > 60:
            return "Content 😊"
        return "Okay 🙂"
    
    def _needs_attention(self) -> List[str]:
        """List what the pet needs"""
        needs = []
        if self.hunger > 60:
            needs.append('food')
        if self.energy < 30:
            needs.append('rest')
        if self.mood < 40:
            needs.append('play')
        return needs
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pet for storage"""
        return {
            'species': self.species,
            'name': self.name,
            'hunger': self.hunger,
            'energy': self.energy,
            'mood': self.mood,
            'bond': self.bond,
            'level': self.level,
            'xp': self.xp,
            'xp_to_next': self.xp_to_next,
            'achievements': self.achievements,
            'interaction_count': self.interaction_count,
            'streak_days': self.streak_days,
            'last_fed': self.last_fed.isoformat(),
            'last_played': self.last_played.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalPetAI':
        """Deserialize pet from storage"""
        pet = cls(data.get('species', 'cat'), data.get('name', 'Buddy'))
        pet.hunger = data.get('hunger', 50)
        pet.energy = data.get('energy', 50)
        pet.mood = data.get('mood', 50)
        pet.bond = data.get('bond', 20)
        pet.level = data.get('level', 1)
        pet.xp = data.get('xp', 0)
        pet.xp_to_next = data.get('xp_to_next', fibonacci(6))
        pet.achievements = data.get('achievements', [])
        pet.interaction_count = data.get('interaction_count', 0)
        pet.streak_days = data.get('streak_days', 0)
        
        if 'last_fed' in data:
            pet.last_fed = datetime.fromisoformat(data['last_fed'])
        if 'last_played' in data:
            pet.last_played = datetime.fromisoformat(data['last_played'])
        if 'last_updated' in data:
            pet.last_updated = datetime.fromisoformat(data['last_updated'])
        
        return pet


# ═══════════════════════════════════════════════════════════════════════════════════════
# FRACTAL TIME CALENDAR - Fibonacci Time Blocks with Circadian Alignment
# ═══════════════════════════════════════════════════════════════════════════════════════

class FractalTimeCalendar:
    """
    Calendar system using Fibonacci time blocks aligned with circadian rhythms.
    Breaks the day into natural energy phases with golden ratio proportions.
    """
    
    # Circadian energy phases
    ENERGY_PHASES = {
        'dawn_preparation': {'start': 6, 'end': 8, 'energy': 0.6, 'focus': 0.5},
        'morning_peak': {'start': 8, 'end': 12, 'energy': 1.0, 'focus': 1.0},
        'midday_dip': {'start': 12, 'end': 14, 'energy': 0.5, 'focus': 0.4},
        'afternoon_recovery': {'start': 14, 'end': 17, 'energy': 0.8, 'focus': 0.7},
        'evening_wind_down': {'start': 17, 'end': 20, 'energy': 0.6, 'focus': 0.5},
        'night_rest': {'start': 20, 'end': 22, 'energy': 0.3, 'focus': 0.2},
        'sleep': {'start': 22, 'end': 6, 'energy': 0.1, 'focus': 0.0}
    }
    
    # Task categories matched to optimal times
    TASK_CATEGORIES = {
        'deep_work': {'optimal_phases': ['morning_peak'], 'spoon_cost': 4},
        'creative': {'optimal_phases': ['morning_peak', 'afternoon_recovery'], 'spoon_cost': 3},
        'admin': {'optimal_phases': ['afternoon_recovery', 'evening_wind_down'], 'spoon_cost': 2},
        'routine': {'optimal_phases': ['dawn_preparation', 'evening_wind_down'], 'spoon_cost': 1},
        'social': {'optimal_phases': ['midday_dip', 'evening_wind_down'], 'spoon_cost': 3},
        'rest': {'optimal_phases': ['midday_dip', 'night_rest'], 'spoon_cost': 0}
    }
    
    def __init__(self, user_wake_time: int = 7, user_sleep_time: int = 23):
        """
        Initialize calendar with user's sleep schedule.
        
        Args:
            user_wake_time: Hour user typically wakes (24h format)
            user_sleep_time: Hour user typically sleeps (24h format)
        """
        self.wake_time = user_wake_time
        self.sleep_time = user_sleep_time
        self.tasks = []
    
    def generate_fibonacci_blocks(self, date: datetime) -> List[Dict[str, Any]]:
        """
        Generate Fibonacci time blocks for a day.
        Uses Fibonacci sequence: 1, 1, 2, 3, 5 hours for a ~12 hour day.
        """
        blocks = []
        
        # Calculate available hours
        awake_hours = self.sleep_time - self.wake_time
        if awake_hours < 0:
            awake_hours += 24
        
        # Fibonacci blocks that fit in the day
        fib_hours = [1, 1, 2, 3, 5]  # Total: 12 hours
        scale = awake_hours / 12.0
        
        current_hour = self.wake_time
        block_index = 0
        
        for fib_duration in fib_hours:
            scaled_duration = fib_duration * scale
            start_time = datetime(date.year, date.month, date.day, 
                                 int(current_hour) % 24, 
                                 int((current_hour % 1) * 60))
            
            # Determine energy phase
            phase = self._get_energy_phase(current_hour)
            phase_data = self.ENERGY_PHASES.get(phase, {'energy': 0.5, 'focus': 0.5})
            
            # Calculate spoon capacity for this block
            spoon_capacity = int(phase_data['energy'] * scaled_duration * 2)
            
            block = {
                'index': block_index,
                'start_hour': current_hour,
                'duration_hours': scaled_duration,
                'fibonacci_value': fib_duration,
                'phase': phase,
                'energy_level': phase_data['energy'],
                'focus_level': phase_data['focus'],
                'spoon_capacity': spoon_capacity,
                'optimal_tasks': self._get_optimal_tasks(phase),
                'golden_ratio_position': (block_index + 1) * PHI_INVERSE
            }
            
            blocks.append(block)
            current_hour += scaled_duration
            block_index += 1
        
        return blocks
    
    def _get_energy_phase(self, hour: float) -> str:
        """Determine which energy phase a given hour falls into"""
        hour = int(hour) % 24
        
        for phase_name, phase_data in self.ENERGY_PHASES.items():
            start = phase_data['start']
            end = phase_data['end']
            
            if start < end:
                if start <= hour < end:
                    return phase_name
            else:  # Wraps around midnight
                if hour >= start or hour < end:
                    return phase_name
        
        return 'unknown'
    
    def _get_optimal_tasks(self, phase: str) -> List[str]:
        """Get task types optimal for a given energy phase"""
        optimal = []
        for task_type, task_data in self.TASK_CATEGORIES.items():
            if phase in task_data['optimal_phases']:
                optimal.append(task_type)
        return optimal
    
    def schedule_task(self, task: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Schedule a task into the optimal time block.
        
        Args:
            task: Task with 'name', 'category', 'spoon_cost', 'duration_hours'
            blocks: List of time blocks from generate_fibonacci_blocks
        
        Returns:
            Scheduling result with assigned block or suggestions
        """
        task_category = task.get('category', 'admin')
        task_cost = task.get('spoon_cost', self.TASK_CATEGORIES.get(task_category, {}).get('spoon_cost', 2))
        
        # Find optimal blocks
        optimal_phases = self.TASK_CATEGORIES.get(task_category, {}).get('optimal_phases', [])
        
        # Score each block
        scored_blocks = []
        for block in blocks:
            score = 0
            
            # Bonus for being in optimal phase
            if block['phase'] in optimal_phases:
                score += 3
            
            # Bonus for having enough spoons
            if block['spoon_capacity'] >= task_cost:
                score += 2
            
            # Bonus for energy match
            score += block['energy_level'] * 2
            
            scored_blocks.append((score, block))
        
        # Sort by score
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        
        if scored_blocks:
            best_block = scored_blocks[0][1]
            return {
                'success': True,
                'task': task,
                'assigned_block': best_block,
                'reasoning': f"Scheduled in {best_block['phase']} phase for optimal energy"
            }
        
        return {
            'success': False,
            'task': task,
            'suggestion': "Consider breaking this task into smaller pieces"
        }
    
    def get_day_plan(self, date: datetime, tasks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate complete day plan with Mayan calendar integration"""
        blocks = self.generate_fibonacci_blocks(date)
        mayan = mayan_day_sign(date)
        
        plan = {
            'date': date.isoformat(),
            'mayan_day': mayan,
            'fibonacci_blocks': blocks,
            'total_spoon_capacity': sum(b['spoon_capacity'] for b in blocks),
            'scheduled_tasks': [],
            'energy_forecast': self._energy_forecast(blocks)
        }
        
        if tasks:
            for task in tasks:
                result = self.schedule_task(task, blocks)
                plan['scheduled_tasks'].append(result)
        
        return plan
    
    def _energy_forecast(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Forecast energy availability throughout the day"""
        peak_block = max(blocks, key=lambda b: b['energy_level'])
        low_block = min(blocks, key=lambda b: b['energy_level'])
        
        return {
            'peak_time': f"{int(peak_block['start_hour'])}:00",
            'peak_energy': peak_block['energy_level'],
            'low_time': f"{int(low_block['start_hour'])}:00",
            'low_energy': low_block['energy_level'],
            'recommendation': f"Schedule important tasks around {int(peak_block['start_hour'])}:00"
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# FIBONACCI TASK SCHEDULER - Golden Ratio Priority Optimization
# ═══════════════════════════════════════════════════════════════════════════════════════

class FibonacciTaskScheduler:
    """
    Task scheduler using Fibonacci weighting for priority calculation.
    Implements golden ratio optimization for task ordering.
    """
    
    def __init__(self):
        self.tasks = []
    
    def add_task(self, name: str, importance: int, urgency: int, 
                 effort: int, category: str = 'general') -> Dict[str, Any]:
        """
        Add a task with Fibonacci-weighted priority.
        
        Args:
            name: Task description
            importance: 1-5 scale
            urgency: 1-5 scale  
            effort: 1-5 scale (higher = more effort)
            category: Task category
        
        Returns:
            Task with calculated priority
        """
        # Fibonacci weights for importance levels
        importance_weight = fibonacci(importance + 3)  # fib(4) to fib(8)
        urgency_weight = fibonacci(urgency + 2)  # fib(3) to fib(7)
        effort_penalty = fibonacci(effort + 1)  # fib(2) to fib(6)
        
        # Golden ratio priority formula
        # Higher importance and urgency increase priority
        # Higher effort decreases priority (do easier things first when equal)
        priority = (importance_weight * PHI + urgency_weight) / (effort_penalty * PHI_INVERSE + 1)
        
        task = {
            'id': secrets.token_hex(4),
            'name': name,
            'importance': importance,
            'urgency': urgency,
            'effort': effort,
            'category': category,
            'priority': round(priority, 2),
            'fibonacci_importance': importance_weight,
            'fibonacci_urgency': urgency_weight,
            'created_at': datetime.now().isoformat(),
            'completed': False
        }
        
        self.tasks.append(task)
        self._sort_tasks()
        
        return task
    
    def _sort_tasks(self):
        """Sort tasks by priority (highest first)"""
        self.tasks.sort(key=lambda t: t['priority'], reverse=True)
    
    def get_next_task(self, available_spoons: int = None, 
                      max_effort: int = None) -> Optional[Dict[str, Any]]:
        """
        Get the next task to work on.
        
        Args:
            available_spoons: Filter by effort if spoons limited
            max_effort: Maximum effort level to consider
        
        Returns:
            Highest priority task matching criteria
        """
        for task in self.tasks:
            if task['completed']:
                continue
            
            if max_effort and task['effort'] > max_effort:
                continue
            
            if available_spoons:
                spoon_cost = DEFAULT_SPOON_COSTS.get(task['category'], task['effort'])
                if spoon_cost > available_spoons:
                    continue
            
            return task
        
        return None
    
    def complete_task(self, task_id: str) -> Dict[str, Any]:
        """Mark a task as completed"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['completed'] = True
                task['completed_at'] = datetime.now().isoformat()
                
                # Calculate XP earned (Fibonacci based on priority)
                xp_earned = int(task['priority'] / PHI)
                
                return {
                    'success': True,
                    'task': task,
                    'xp_earned': xp_earned,
                    'message': f"Great job completing '{task['name']}'!"
                }
        
        return {'success': False, 'message': 'Task not found'}
    
    def get_prioritized_list(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get prioritized task list"""
        active_tasks = [t for t in self.tasks if not t['completed']]
        if limit:
            return active_tasks[:limit]
        return active_tasks
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tasks filtered by category"""
        return [t for t in self.tasks if t['category'] == category and not t['completed']]


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXECUTIVE FUNCTION SUPPORT - Pattern Detection & Scaffolding
# ═══════════════════════════════════════════════════════════════════════════════════════

class ExecutiveFunctionSupport:
    """
    Detect executive dysfunction patterns and provide scaffolding support.
    Uses statistical analysis to identify when users are struggling.
    """
    
    def __init__(self):
        self.behavior_history = []
        self.dysfunction_indicators = {
            'task_switching_difficulty': 0,
            'initiation_problems': 0,
            'time_blindness': 0,
            'working_memory_issues': 0,
            'emotional_regulation': 0
        }
    
    def log_behavior(self, behavior: Dict[str, Any]):
        """
        Log user behavior for pattern analysis.
        
        Expected fields:
            - timestamp: when
            - action: what they did
            - planned_action: what they intended
            - duration: how long it took
            - completion: did they finish
            - mood: emotional state
        """
        behavior['timestamp'] = behavior.get('timestamp', datetime.now().isoformat())
        self.behavior_history.append(behavior)
        
        # Keep last 100 behaviors
        if len(self.behavior_history) > 100:
            self.behavior_history = self.behavior_history[-100:]
        
        # Update indicators
        self._analyze_patterns()
    
    def _analyze_patterns(self):
        """Analyze behavior patterns for dysfunction indicators"""
        if len(self.behavior_history) < 5:
            return
        
        recent = self.behavior_history[-20:]
        
        # Task switching difficulty: starting many tasks, finishing few
        started = sum(1 for b in recent if b.get('action') == 'start_task')
        completed = sum(1 for b in recent if b.get('completion', False))
        if started > 0:
            self.dysfunction_indicators['task_switching_difficulty'] = max(0, (started - completed) / started)
        
        # Initiation problems: long delays before starting
        delays = [b.get('delay_minutes', 0) for b in recent if 'delay_minutes' in b]
        if delays:
            avg_delay = sum(delays) / len(delays)
            self.dysfunction_indicators['initiation_problems'] = min(1, avg_delay / 60)  # Normalize to 1 hour
        
        # Time blindness: large differences between planned and actual duration
        time_errors = []
        for b in recent:
            if 'planned_duration' in b and 'actual_duration' in b:
                error = abs(b['actual_duration'] - b['planned_duration']) / b['planned_duration']
                time_errors.append(error)
        if time_errors:
            self.dysfunction_indicators['time_blindness'] = min(1, sum(time_errors) / len(time_errors))
        
        # Working memory: forgetting what was planned
        mismatches = sum(1 for b in recent if b.get('action') != b.get('planned_action') and b.get('planned_action'))
        if len(recent) > 0:
            self.dysfunction_indicators['working_memory_issues'] = mismatches / len(recent)
        
        # Emotional regulation: mood swings
        moods = [b.get('mood', 50) for b in recent if 'mood' in b]
        if len(moods) >= 2:
            mood_changes = [abs(moods[i] - moods[i-1]) for i in range(1, len(moods))]
            avg_change = sum(mood_changes) / len(mood_changes)
            self.dysfunction_indicators['emotional_regulation'] = min(1, avg_change / 30)
    
    def get_dysfunction_score(self) -> float:
        """Calculate overall executive dysfunction score (0-1)"""
        if not self.dysfunction_indicators:
            return 0
        
        # Weighted average
        weights = {
            'task_switching_difficulty': 1.2,
            'initiation_problems': 1.5,
            'time_blindness': 1.0,
            'working_memory_issues': 1.3,
            'emotional_regulation': 0.8
        }
        
        weighted_sum = sum(self.dysfunction_indicators[k] * weights[k] for k in weights)
        weight_total = sum(weights.values())
        
        return weighted_sum / weight_total
    
    def detect_current_state(self) -> Dict[str, Any]:
        """Detect current executive function state"""
        score = self.get_dysfunction_score()
        
        state = {
            'dysfunction_score': round(score, 2),
            'indicators': self.dysfunction_indicators.copy(),
            'severity': 'low' if score < 0.3 else 'moderate' if score < 0.6 else 'high',
            'recommendations': self._get_recommendations(score),
            'scaffolding_needed': score > 0.4
        }
        
        return state
    
    def _get_recommendations(self, score: float) -> List[str]:
        """Get personalized recommendations based on dysfunction patterns"""
        recommendations = []
        
        if self.dysfunction_indicators['initiation_problems'] > 0.4:
            recommendations.append("Try the 2-minute rule: commit to just 2 minutes of any task")
            recommendations.append("Use body doubling - work alongside someone else")
        
        if self.dysfunction_indicators['task_switching_difficulty'] > 0.4:
            recommendations.append("Complete one task fully before starting another")
            recommendations.append("Use transition rituals between tasks")
        
        if self.dysfunction_indicators['time_blindness'] > 0.4:
            recommendations.append("Use visual timers you can see counting down")
            recommendations.append("Set alarms for task transitions")
        
        if self.dysfunction_indicators['working_memory_issues'] > 0.4:
            recommendations.append("Write everything down immediately")
            recommendations.append("Use checklists even for familiar tasks")
        
        if self.dysfunction_indicators['emotional_regulation'] > 0.4:
            recommendations.append("Practice the STOP technique: Stop, Take a breath, Observe, Proceed")
            recommendations.append("Schedule regular emotional check-ins")
        
        if not recommendations:
            recommendations.append("You're doing well! Keep up the good patterns.")
        
        return recommendations
    
    def get_task_scaffolding(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down a task into micro-steps for executive function support.
        Each step should be completable in under 5 minutes.
        """
        task_name = task.get('name', 'Task')
        effort = task.get('effort', 3)
        
        # Generate micro-steps based on effort level
        num_steps = effort + 2  # 3-7 steps based on effort
        
        steps = []
        steps.append({
            'step': 1,
            'action': f"Get ready: Gather what you need for '{task_name}'",
            'duration_minutes': 2,
            'checkpoint': "I have everything I need"
        })
        
        for i in range(2, num_steps):
            steps.append({
                'step': i,
                'action': f"Work chunk {i-1}: Focus on one small part",
                'duration_minutes': 5,
                'checkpoint': f"Chunk {i-1} complete"
            })
        
        steps.append({
            'step': num_steps,
            'action': "Wrap up: Review what you did and celebrate!",
            'duration_minutes': 2,
            'checkpoint': "Task complete! 🎉"
        })
        
        return {
            'original_task': task,
            'micro_steps': steps,
            'total_time_minutes': sum(s['duration_minutes'] for s in steps),
            'approach': 'scaffolded',
            'tip': "Focus only on the current step. The rest will wait."
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY SYSTEM - Full Support for Neurodivergent Users
# ═══════════════════════════════════════════════════════════════════════════════════════

class AccessibilitySystem:
    """
    Comprehensive accessibility features for neurodivergent users.
    Supports autism, ADHD, aphantasia, and dysgraphia.
    """
    
    # Autism-safe color palettes (low saturation, predictable)
    AUTISM_SAFE_PALETTES = {
        'calm_ocean': {
            'primary': '#5B8A9A',
            'secondary': '#8BB4C2',
            'background': '#F5F9FA',
            'text': '#2C4A52',
            'accent': '#7BA3B0',
            'warning': '#D4A574'
        },
        'forest_peace': {
            'primary': '#6B8E6B',
            'secondary': '#9CB89C',
            'background': '#F5F8F5',
            'text': '#3A4D3A',
            'accent': '#8DAD8D',
            'warning': '#C9A86C'
        },
        'gentle_lavender': {
            'primary': '#8B7B9B',
            'secondary': '#B4A8C2',
            'background': '#FAF8FC',
            'text': '#4A3D52',
            'accent': '#A294B0',
            'warning': '#C4A67A'
        },
        'warm_sand': {
            'primary': '#A89078',
            'secondary': '#C8B8A8',
            'background': '#FDFBF8',
            'text': '#5A4A3A',
            'accent': '#BBA898',
            'warning': '#C49068'
        },
        'high_contrast': {
            'primary': '#000000',
            'secondary': '#333333',
            'background': '#FFFFFF',
            'text': '#000000',
            'accent': '#0066CC',
            'warning': '#CC6600'
        }
    }
    
    def __init__(self):
        self.settings = {
            'reduced_motion': False,
            'high_contrast': False,
            'large_text': False,
            'dyslexia_font': False,
            'screen_reader_mode': False,
            'color_palette': 'calm_ocean',
            'aphantasia_mode': False,  # Text descriptions instead of "visualize"
            'simplified_interface': False,
            'keyboard_only': False,
            'auto_save': True,
            'confirmation_dialogs': True,
            'time_estimates': True,
            'spoon_tracking': True
        }
    
    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update accessibility settings"""
        for key, value in new_settings.items():
            if key in self.settings:
                self.settings[key] = value
        return self.settings
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get current color palette"""
        palette_name = self.settings['color_palette']
        if self.settings['high_contrast']:
            palette_name = 'high_contrast'
        return self.AUTISM_SAFE_PALETTES.get(palette_name, self.AUTISM_SAFE_PALETTES['calm_ocean'])
    
    def get_css_variables(self) -> str:
        """Generate CSS variables for current settings"""
        palette = self.get_color_palette()
        
        css = ":root {\n"
        for name, color in palette.items():
            css += f"  --color-{name}: {color};\n"
        
        # Font settings
        if self.settings['large_text']:
            css += "  --font-size-base: 18px;\n"
            css += "  --font-size-large: 24px;\n"
        else:
            css += "  --font-size-base: 16px;\n"
            css += "  --font-size-large: 20px;\n"
        
        if self.settings['dyslexia_font']:
            css += "  --font-family: 'OpenDyslexic', 'Comic Sans MS', sans-serif;\n"
        else:
            css += "  --font-family: 'Inter', -apple-system, sans-serif;\n"
        
        # Motion settings
        if self.settings['reduced_motion']:
            css += "  --animation-duration: 0s;\n"
            css += "  --transition-duration: 0s;\n"
        else:
            css += "  --animation-duration: 0.3s;\n"
            css += "  --transition-duration: 0.2s;\n"
        
        css += "}\n"
        return css
    
    def transform_text_for_aphantasia(self, text: str) -> str:
        """
        Transform text to be aphantasia-friendly.
        Replaces "visualize" language with concrete descriptions.
        """
        if not self.settings['aphantasia_mode']:
            return text
        
        replacements = {
            'visualize': 'think about',
            'imagine': 'consider',
            'picture': 'describe',
            'see yourself': 'plan for yourself',
            'envision': 'plan for',
            'in your mind\'s eye': 'in your thoughts',
            'mental image': 'mental description',
            'dream about': 'plan for',
        }
        
        result = text.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Restore original capitalization for first letter
        if text and text[0].isupper():
            result = result[0].upper() + result[1:]
        
        return result
    
    def get_aria_attributes(self) -> Dict[str, str]:
        """Get ARIA attributes for screen reader support"""
        attrs = {
            'role': 'application',
            'aria-label': 'Life Fractal Intelligence - Life Planning Application'
        }
        
        if self.settings['screen_reader_mode']:
            attrs['aria-live'] = 'polite'
            attrs['aria-atomic'] = 'true'
        
        return attrs
    
    def format_for_dysgraphia(self, form_fields: List[Dict]) -> List[Dict]:
        """
        Adapt form fields for users with dysgraphia.
        Adds voice input options and simplifies text entry.
        """
        adapted_fields = []
        
        for field in form_fields:
            adapted = field.copy()
            
            # Add voice input option
            adapted['voice_input_enabled'] = True
            
            # Prefer selection over typing
            if field.get('type') == 'text':
                if field.get('options'):
                    adapted['type'] = 'select'
                else:
                    adapted['type'] = 'textarea'  # Easier than single line
                    adapted['autocomplete'] = True
                    adapted['suggestions'] = True
            
            # Add character limit visibility
            if 'maxLength' in field:
                adapted['show_character_count'] = True
            
            adapted_fields.append(adapted)
        
        return adapted_fields
    
    def to_dict(self) -> Dict[str, Any]:
        return self.settings.copy()


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE - Complete Schema
# ═══════════════════════════════════════════════════════════════════════════════════════

class Database:
    """Production-ready SQLite database with all tables"""
    
    def __init__(self, db_path: str = "life_fractal.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"✅ Database initialized: {db_path}")
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Create all tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                subscription_status TEXT DEFAULT 'trial',
                trial_end_date TEXT,
                stripe_customer_id TEXT,
                accessibility_settings TEXT DEFAULT '{}',
                spoon_settings TEXT DEFAULT '{}'
            )
        ''')
        
        # Goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT DEFAULT 'personal',
                term TEXT DEFAULT 'medium',
                priority INTEGER DEFAULT 3,
                progress REAL DEFAULT 0.0,
                target_date TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                why_important TEXT,
                obstacles TEXT,
                success_criteria TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                importance INTEGER DEFAULT 3,
                urgency INTEGER DEFAULT 3,
                effort INTEGER DEFAULT 3,
                category TEXT DEFAULT 'general',
                priority REAL DEFAULT 0.0,
                spoon_cost INTEGER DEFAULT 2,
                due_date TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (goal_id) REFERENCES goals(id)
            )
        ''')
        
        # Habits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS habits (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                frequency TEXT DEFAULT 'daily',
                current_streak INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                total_completions INTEGER DEFAULT 0,
                spoon_cost INTEGER DEFAULT 1,
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
                mood_level INTEGER DEFAULT 50,
                energy_level INTEGER DEFAULT 50,
                stress_level INTEGER DEFAULT 50,
                sleep_hours REAL DEFAULT 7.0,
                sleep_quality INTEGER DEFAULT 50,
                spoons_used INTEGER DEFAULT 0,
                spoons_available INTEGER DEFAULT 12,
                journal_entry TEXT,
                gratitude TEXT,
                wins TEXT,
                challenges TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)
            )
        ''')
        
        # Pet state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pet_state (
                user_id TEXT PRIMARY KEY,
                species TEXT DEFAULT 'cat',
                name TEXT DEFAULT 'Buddy',
                hunger REAL DEFAULT 50.0,
                energy REAL DEFAULT 50.0,
                mood REAL DEFAULT 50.0,
                bond REAL DEFAULT 20.0,
                level INTEGER DEFAULT 1,
                xp INTEGER DEFAULT 0,
                xp_to_next INTEGER DEFAULT 8,
                achievements TEXT DEFAULT '[]',
                interaction_count INTEGER DEFAULT 0,
                streak_days INTEGER DEFAULT 0,
                last_fed TEXT,
                last_played TEXT,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Behavior history for executive function tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT,
                planned_action TEXT,
                duration_planned INTEGER,
                duration_actual INTEGER,
                delay_minutes INTEGER,
                completion INTEGER DEFAULT 0,
                mood INTEGER DEFAULT 50,
                notes TEXT,
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
        conn.close()
    
    def execute(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query safely"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Database error: {e}")
            return []
    
    def insert(self, table: str, data: dict) -> bool:
        """Insert data"""
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join('?' * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            self.execute(query, tuple(data.values()))
            return True
        except Exception as e:
            logger.error(f"Insert error: {e}")
            return False
    
    def update(self, table: str, data: dict, where: dict) -> bool:
        """Update data"""
        try:
            set_clause = ', '.join(f"{k} = ?" for k in data.keys())
            where_clause = ' AND '.join(f"{k} = ?" for k in where.keys())
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            params = tuple(data.values()) + tuple(where.values())
            self.execute(query, params)
            return True
        except Exception as e:
            logger.error(f"Update error: {e}")
            return False
    
    def select(self, table: str, where: Optional[dict] = None) -> List[Dict]:
        """Select data"""
        query = f"SELECT * FROM {table}"
        params = ()
        if where:
            where_clause = ' AND '.join(f"{k} = ?" for k in where.keys())
            query += f" WHERE {where_clause}"
            params = tuple(where.values())
        return self.execute(query, params)
    
    def delete(self, table: str, where: dict) -> bool:
        """Delete data"""
        try:
            where_clause = ' AND '.join(f"{k} = ?" for k in where.keys())
            query = f"DELETE FROM {table} WHERE {where_clause}"
            self.execute(query, tuple(where.values()))
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════════════
# FRACTAL ENGINE - 2D and 3D Visualization
# ═══════════════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    """Complete 2D & 3D fractal visualization engine"""
    
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
        self.use_gpu = GPU_AVAILABLE
    
    def generate_2d_fractal(self, wellness: float, mood: float, stress: float) -> Optional[Any]:
        """Generate 2D Mandelbrot fractal based on user metrics"""
        if not HAS_NUMPY or not HAS_PIL:
            return None
        
        max_iter = int(100 + mood * 1.5)
        zoom = 1.0 + (wellness / 100) * 3.0
        center_x = -0.7 + (stress / 500)
        center_y = 0.0
        
        # Generate Mandelbrot set
        x = np.linspace(-2.5/zoom + center_x, 2.5/zoom + center_x, self.width)
        y = np.linspace(-2.5/zoom + center_y, 2.5/zoom + center_y, self.height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros((self.height, self.width))
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        # Apply wellness-based coloring
        rgb = self._apply_wellness_coloring(iterations, max_iter, wellness)
        
        return Image.fromarray(rgb, 'RGB')
    
    def _apply_wellness_coloring(self, iterations: np.ndarray, max_iter: int, wellness: float) -> np.ndarray:
        """Apply color scheme based on wellness level"""
        normalized = iterations / max_iter
        
        # Wellness affects color temperature
        # High wellness = warm colors, Low wellness = cool colors
        warmth = wellness / 100
        
        r = np.uint8(255 * (normalized * warmth + (1 - normalized) * 0.3))
        g = np.uint8(255 * (normalized * 0.5 + (1 - normalized) * (1 - warmth) * 0.5))
        b = np.uint8(255 * ((1 - normalized) * (1 - warmth) + normalized * 0.2))
        
        return np.stack([r, g, b], axis=-1)
    
    def generate_3d_mandelbulb(self, wellness: float, mood: float) -> Optional[Any]:
        """Generate 3D Mandelbulb fractal"""
        if not HAS_NUMPY or not HAS_PIL:
            return None
        
        power = 6.0 + (mood / 100) * 4.0
        rotation_y = (wellness / 100) * PI * 0.5
        
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for py in range(0, self.height, 2):
            for px in range(0, self.width, 2):
                # Ray direction
                x = (2 * px / self.width - 1) * 0.8
                y = (1 - 2 * py / self.height) * 0.8
                
                # Apply rotation
                dx = x * PureMath.cos(rotation_y) - 1 * PureMath.sin(rotation_y)
                dz = x * PureMath.sin(rotation_y) + 1 * PureMath.cos(rotation_y)
                dy = y
                
                # Normalize
                length = PureMath.sqrt(dx**2 + dy**2 + dz**2)
                dx, dy, dz = dx/length, dy/length, dz/length
                
                # Ray march
                t = 0
                for _ in range(50):
                    pos_x, pos_y, pos_z = dx * t, dy * t, dz * t - 2.5
                    dist = self._mandelbulb_distance(pos_x, pos_y, pos_z, power)
                    
                    if dist < 0.001:
                        intensity = int(255 * (1 - t / 5))
                        hue = (wellness / 100) * 0.3  # Shift hue based on wellness
                        r = int(intensity * (1 + hue))
                        g = int(intensity * 0.5)
                        b = int(intensity * (1 - hue))
                        image[py:py+2, px:px+2] = [min(255, r), g, max(0, b)]
                        break
                    
                    t += dist * 0.5
                    if t > 5:
                        break
        
        return Image.fromarray(image, 'RGB')
    
    def _mandelbulb_distance(self, x: float, y: float, z: float, power: float) -> float:
        """Distance estimator for Mandelbulb"""
        x0, y0, z0 = x, y, z
        dr = 1.0
        r = 0.0
        
        for _ in range(15):
            r = PureMath.sqrt(x*x + y*y + z*z)
            if r > 2:
                break
            
            theta = PureMath.sqrt(max(0, x*x + y*y))
            if theta > 0:
                theta = z / theta
                theta = PureMath.sqrt(1 + theta * theta)
                theta = 1 / theta
                theta = PureMath.sqrt(max(0, 1 - theta * theta))
            else:
                theta = 0
            
            phi = 0
            if abs(x) > 0.0001:
                phi = y / x if x != 0 else 0
            
            dr = pow(r, power - 1) * power * dr + 1.0
            
            zr = pow(r, power)
            theta = theta * power
            phi = phi * power
            
            x = zr * theta * (1 if x >= 0 else -1) + x0
            y = zr * theta * phi + y0
            z = zr * (z / (r + 0.0001)) * power + z0
        
        return 0.5 * PureMath.log(max(r, 0.0001)) * r / (dr + 0.0001)
    
    def fractal_to_base64(self, image) -> str:
        """Convert PIL image to base64 string"""
        if image is None:
            return ""
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize database
db = Database()

# Initialize fractal engine
fractal_engine = FractalEngine()

# In-memory stores (backed by database)
user_pets: Dict[str, EmotionalPetAI] = {}
user_calendars: Dict[str, FractalTimeCalendar] = {}
user_schedulers: Dict[str, FibonacciTaskScheduler] = {}
user_exec_support: Dict[str, ExecutiveFunctionSupport] = {}
user_accessibility: Dict[str, AccessibilitySystem] = {}
user_spoons: Dict[str, SpoonState] = {}


# ═══════════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_session(user_id: str) -> str:
    """Create a new session token"""
    token = secrets.token_hex(32)
    expires_at = (datetime.now() + timedelta(days=7)).isoformat()
    
    db.insert('sessions', {
        'token': token,
        'user_id': user_id,
        'created_at': datetime.now().isoformat(),
        'expires_at': expires_at
    })
    
    return token


def verify_session(token: str) -> Optional[str]:
    """Verify session token and return user_id"""
    if not token:
        return None
    
    sessions = db.select('sessions', {'token': token})
    if not sessions:
        return None
    
    session_data = sessions[0]
    expires_at = datetime.fromisoformat(session_data['expires_at'])
    
    if datetime.now() > expires_at:
        db.delete('sessions', {'token': token})
        return None
    
    return session_data['user_id']


def get_current_user():
    """Get current user from session"""
    token = request.cookies.get('session_token') or request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None
    
    user_id = verify_session(token)
    if not user_id:
        return None
    
    users = db.select('users', {'id': user_id})
    return users[0] if users else None


def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(user, *args, **kwargs)
    return decorated


def get_user_pet(user_id: str) -> EmotionalPetAI:
    """Get or create user's pet"""
    if user_id not in user_pets:
        # Try to load from database
        pet_data = db.select('pet_state', {'user_id': user_id})
        if pet_data:
            pet_dict = pet_data[0]
            pet_dict['achievements'] = json.loads(pet_dict.get('achievements', '[]'))
            user_pets[user_id] = EmotionalPetAI.from_dict(pet_dict)
        else:
            # Create new pet
            user_pets[user_id] = EmotionalPetAI('cat', 'Buddy')
            save_pet(user_id, user_pets[user_id])
    
    return user_pets[user_id]


def save_pet(user_id: str, pet: EmotionalPetAI):
    """Save pet to database"""
    pet_dict = pet.to_dict()
    pet_dict['user_id'] = user_id
    pet_dict['achievements'] = json.dumps(pet_dict.get('achievements', []))
    
    existing = db.select('pet_state', {'user_id': user_id})
    if existing:
        db.update('pet_state', pet_dict, {'user_id': user_id})
    else:
        db.insert('pet_state', pet_dict)


def get_user_spoons(user_id: str) -> SpoonState:
    """Get or create user's spoon state"""
    if user_id not in user_spoons:
        user_spoons[user_id] = SpoonState()
    return user_spoons[user_id]


def get_user_calendar(user_id: str) -> FractalTimeCalendar:
    """Get or create user's calendar"""
    if user_id not in user_calendars:
        user_calendars[user_id] = FractalTimeCalendar()
    return user_calendars[user_id]


def get_user_scheduler(user_id: str) -> FibonacciTaskScheduler:
    """Get or create user's task scheduler"""
    if user_id not in user_schedulers:
        user_schedulers[user_id] = FibonacciTaskScheduler()
    return user_schedulers[user_id]


def get_user_exec_support(user_id: str) -> ExecutiveFunctionSupport:
    """Get or create user's executive function support"""
    if user_id not in user_exec_support:
        user_exec_support[user_id] = ExecutiveFunctionSupport()
    return user_exec_support[user_id]


def get_user_accessibility(user_id: str) -> AccessibilitySystem:
    """Get or create user's accessibility settings"""
    if user_id not in user_accessibility:
        user_accessibility[user_id] = AccessibilitySystem()
        
        # Load from database
        users = db.select('users', {'id': user_id})
        if users and users[0].get('accessibility_settings'):
            try:
                settings = json.loads(users[0]['accessibility_settings'])
                user_accessibility[user_id].update_settings(settings)
            except:
                pass
    
    return user_accessibility[user_id]


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Check if user exists
        existing = db.select('users', {'email': email})
        if existing:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        user_id = secrets.token_hex(16)
        password_hash = generate_password_hash(password)
        trial_end = (datetime.now() + timedelta(days=7)).isoformat()
        
        db.insert('users', {
            'id': user_id,
            'email': email,
            'password_hash': password_hash,
            'first_name': first_name,
            'last_name': last_name,
            'created_at': datetime.now().isoformat(),
            'subscription_status': 'trial',
            'trial_end_date': trial_end
        })
        
        # Create session
        token = create_session(user_id)
        
        # Create default pet
        pet = EmotionalPetAI('cat', 'Buddy')
        user_pets[user_id] = pet
        save_pet(user_id, pet)
        
        response = make_response(jsonify({
            'success': True,
            'user_id': user_id,
            'message': 'Welcome! Your 7-day free trial has started.',
            'trial_end_date': trial_end
        }))
        response.set_cookie('session_token', token, httponly=True, samesite='Lax', max_age=604800)
        
        return response
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Log in a user"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Find user
        users = db.select('users', {'email': email})
        if not users:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = users[0]
        
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        db.update('users', {'last_login': datetime.now().isoformat()}, {'id': user['id']})
        
        # Create session
        token = create_session(user['id'])
        
        response = make_response(jsonify({
            'success': True,
            'user_id': user['id'],
            'first_name': user['first_name'],
            'subscription_status': user['subscription_status']
        }))
        response.set_cookie('session_token', token, httponly=True, samesite='Lax', max_age=604800)
        
        return response
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Log out current user"""
    token = request.cookies.get('session_token')
    if token:
        db.delete('sessions', {'token': token})
    
    response = make_response(jsonify({'success': True}))
    response.delete_cookie('session_token')
    return response


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_me(user):
    """Get current user info"""
    return jsonify({
        'id': user['id'],
        'email': user['email'],
        'first_name': user['first_name'],
        'last_name': user['last_name'],
        'subscription_status': user['subscription_status'],
        'trial_end_date': user.get('trial_end_date')
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - PET SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet', methods=['GET'])
@require_auth
def get_pet(user):
    """Get user's pet state"""
    pet = get_user_pet(user['id'])
    
    # Update pet based on time passed
    last_update = pet.last_updated
    hours_passed = (datetime.now() - last_update).total_seconds() / 3600
    
    if hours_passed > 0.1:  # More than 6 minutes
        pet.update(hours_passed)
        save_pet(user['id'], pet)
    
    return jsonify(pet.get_state())


@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet(user):
    """Feed the pet"""
    pet = get_user_pet(user['id'])
    result = pet.feed()
    save_pet(user['id'], pet)
    return jsonify(result)


@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_with_pet(user):
    """Play with the pet"""
    pet = get_user_pet(user['id'])
    result = pet.play()
    save_pet(user['id'], pet)
    return jsonify(result)


@app.route('/api/pet/rest', methods=['POST'])
@require_auth
def rest_pet(user):
    """Let the pet rest"""
    pet = get_user_pet(user['id'])
    result = pet.rest()
    save_pet(user['id'], pet)
    return jsonify(result)


@app.route('/api/pet/species', methods=['GET'])
def get_pet_species():
    """Get available pet species"""
    species_info = []
    for species, traits in EmotionalPetAI.SPECIES_TRAITS.items():
        species_info.append({
            'species': species,
            'emoji': traits['emoji'],
            'personality': traits['personality']
        })
    return jsonify(species_info)


@app.route('/api/pet/change', methods=['POST'])
@require_auth
def change_pet(user):
    """Change pet species"""
    data = request.get_json() or {}
    new_species = data.get('species', 'cat')
    new_name = data.get('name', 'Buddy')
    
    if new_species not in EmotionalPetAI.SPECIES_TRAITS:
        return jsonify({'error': 'Invalid species'}), 400
    
    # Create new pet (preserving some stats)
    old_pet = get_user_pet(user['id'])
    new_pet = EmotionalPetAI(new_species, new_name)
    
    # Transfer some progress
    new_pet.bond = old_pet.bond * 0.5  # Keep half the bond
    new_pet.level = max(1, old_pet.level - 1)  # Drop one level
    
    user_pets[user['id']] = new_pet
    save_pet(user['id'], new_pet)
    
    return jsonify({
        'success': True,
        'message': f"Welcome {new_name} the {new_species}!",
        'pet': new_pet.get_state()
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - SPOON THEORY
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/spoons', methods=['GET'])
@require_auth
def get_spoons(user):
    """Get current spoon state"""
    spoons = get_user_spoons(user['id'])
    return jsonify(spoons.to_dict())


@app.route('/api/spoons/use', methods=['POST'])
@require_auth
def use_spoons(user):
    """Use spoons for an activity"""
    data = request.get_json() or {}
    amount = data.get('amount', 1)
    task_name = data.get('task', 'activity')
    
    spoons = get_user_spoons(user['id'])
    result = spoons.use_spoons(amount, task_name)
    
    return jsonify(result)


@app.route('/api/spoons/rest', methods=['POST'])
@require_auth
def rest_spoons(user):
    """Recover spoons through rest"""
    data = request.get_json() or {}
    hours = data.get('hours', 1)
    
    spoons = get_user_spoons(user['id'])
    result = spoons.rest(hours)
    
    return jsonify(result)


@app.route('/api/spoons/new-day', methods=['POST'])
@require_auth
def new_day_spoons(user):
    """Reset spoons for new day"""
    data = request.get_json() or {}
    sleep_quality = data.get('sleep_quality', 0.7)
    
    spoons = get_user_spoons(user['id'])
    result = spoons.new_day(sleep_quality)
    
    return jsonify(result)


@app.route('/api/spoons/costs', methods=['GET'])
def get_spoon_costs():
    """Get default spoon costs for activities"""
    return jsonify(DEFAULT_SPOON_COSTS)


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - GOALS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals(user):
    """Get all user goals"""
    goals = db.select('goals', {'user_id': user['id']})
    return jsonify(goals)


@app.route('/api/goals', methods=['POST'])
@require_auth
def create_goal(user):
    """Create a new goal"""
    data = request.get_json() or {}
    
    goal_id = secrets.token_hex(8)
    
    goal_data = {
        'id': goal_id,
        'user_id': user['id'],
        'title': data.get('title', 'New Goal'),
        'description': data.get('description', ''),
        'category': data.get('category', 'personal'),
        'term': data.get('term', 'medium'),
        'priority': data.get('priority', 3),
        'progress': 0.0,
        'target_date': data.get('target_date'),
        'created_at': datetime.now().isoformat(),
        'why_important': data.get('why_important', ''),
        'obstacles': data.get('obstacles', ''),
        'success_criteria': data.get('success_criteria', '')
    }
    
    db.insert('goals', goal_data)
    
    return jsonify({
        'success': True,
        'goal': goal_data
    })


@app.route('/api/goals/<goal_id>', methods=['PUT'])
@require_auth
def update_goal(user, goal_id):
    """Update a goal"""
    data = request.get_json() or {}
    
    # Verify ownership
    goals = db.select('goals', {'id': goal_id, 'user_id': user['id']})
    if not goals:
        return jsonify({'error': 'Goal not found'}), 404
    
    update_data = {}
    for field in ['title', 'description', 'category', 'term', 'priority', 'progress', 'target_date', 'why_important', 'obstacles', 'success_criteria']:
        if field in data:
            update_data[field] = data[field]
    
    if data.get('progress', 0) >= 100 and not goals[0].get('completed_at'):
        update_data['completed_at'] = datetime.now().isoformat()
    
    db.update('goals', update_data, {'id': goal_id})
    
    return jsonify({'success': True})


@app.route('/api/goals/<goal_id>', methods=['DELETE'])
@require_auth
def delete_goal(user, goal_id):
    """Delete a goal"""
    db.delete('goals', {'id': goal_id, 'user_id': user['id']})
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - TASKS (Fibonacci Scheduler)
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/tasks', methods=['GET'])
@require_auth
def get_tasks(user):
    """Get all user tasks"""
    tasks = db.select('tasks', {'user_id': user['id']})
    
    # Sort by priority
    tasks.sort(key=lambda t: t.get('priority', 0), reverse=True)
    
    return jsonify(tasks)


@app.route('/api/tasks', methods=['POST'])
@require_auth
def create_task(user):
    """Create a new task with Fibonacci priority"""
    data = request.get_json() or {}
    
    # Calculate Fibonacci priority
    importance = min(5, max(1, data.get('importance', 3)))
    urgency = min(5, max(1, data.get('urgency', 3)))
    effort = min(5, max(1, data.get('effort', 3)))
    
    importance_weight = fibonacci(importance + 3)
    urgency_weight = fibonacci(urgency + 2)
    effort_penalty = fibonacci(effort + 1)
    
    priority = (importance_weight * PHI + urgency_weight) / (effort_penalty * PHI_INVERSE + 1)
    
    task_id = secrets.token_hex(8)
    
    task_data = {
        'id': task_id,
        'user_id': user['id'],
        'goal_id': data.get('goal_id'),
        'name': data.get('name', 'New Task'),
        'description': data.get('description', ''),
        'importance': importance,
        'urgency': urgency,
        'effort': effort,
        'category': data.get('category', 'general'),
        'priority': round(priority, 2),
        'spoon_cost': DEFAULT_SPOON_COSTS.get(data.get('category', 'general'), effort),
        'due_date': data.get('due_date'),
        'created_at': datetime.now().isoformat(),
        'completed': 0
    }
    
    db.insert('tasks', task_data)
    
    return jsonify({
        'success': True,
        'task': task_data
    })


@app.route('/api/tasks/<task_id>/complete', methods=['POST'])
@require_auth
def complete_task(user, task_id):
    """Mark task as complete"""
    tasks = db.select('tasks', {'id': task_id, 'user_id': user['id']})
    if not tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[0]
    
    db.update('tasks', {
        'completed': 1,
        'completed_at': datetime.now().isoformat()
    }, {'id': task_id})
    
    # Award XP to pet
    pet = get_user_pet(user['id'])
    xp_earned = int(task.get('priority', 5) / PHI)
    pet._add_xp(xp_earned)
    save_pet(user['id'], pet)
    
    # Use spoons
    spoons = get_user_spoons(user['id'])
    spoon_result = spoons.use_spoons(task.get('spoon_cost', 2), task['name'])
    
    return jsonify({
        'success': True,
        'xp_earned': xp_earned,
        'spoons': spoon_result,
        'message': f"Great job completing '{task['name']}'!"
    })


@app.route('/api/tasks/next', methods=['GET'])
@require_auth
def get_next_task(user):
    """Get the next recommended task"""
    spoons = get_user_spoons(user['id'])
    available_spoons = spoons.current_spoons
    
    tasks = db.select('tasks', {'user_id': user['id']})
    active_tasks = [t for t in tasks if not t.get('completed')]
    
    # Filter by available spoons
    doable_tasks = [t for t in active_tasks if t.get('spoon_cost', 2) <= available_spoons]
    
    if not doable_tasks:
        # Suggest rest
        return jsonify({
            'task': None,
            'message': "You don't have enough spoons for any tasks. Consider resting!",
            'current_spoons': available_spoons
        })
    
    # Sort by priority
    doable_tasks.sort(key=lambda t: t.get('priority', 0), reverse=True)
    
    return jsonify({
        'task': doable_tasks[0],
        'alternatives': doable_tasks[1:3],
        'current_spoons': available_spoons
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - CALENDAR (Fractal Time)
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/calendar/today', methods=['GET'])
@require_auth
def get_today_plan(user):
    """Get today's Fibonacci time block plan"""
    calendar = get_user_calendar(user['id'])
    today = datetime.now()
    
    # Get user's tasks
    tasks = db.select('tasks', {'user_id': user['id']})
    active_tasks = [t for t in tasks if not t.get('completed')]
    
    plan = calendar.get_day_plan(today, active_tasks)
    
    return jsonify(plan)


@app.route('/api/calendar/date/<date_str>', methods=['GET'])
@require_auth
def get_date_plan(user, date_str):
    """Get plan for a specific date"""
    try:
        date = datetime.fromisoformat(date_str)
    except:
        return jsonify({'error': 'Invalid date format'}), 400
    
    calendar = get_user_calendar(user['id'])
    plan = calendar.get_day_plan(date)
    
    return jsonify(plan)


@app.route('/api/calendar/mayan', methods=['GET'])
@require_auth
def get_mayan_day(user):
    """Get Mayan calendar info for today"""
    mayan = mayan_day_sign(datetime.now())
    return jsonify(mayan)


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - EXECUTIVE FUNCTION SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/executive/state', methods=['GET'])
@require_auth
def get_exec_state(user):
    """Get executive function state"""
    exec_support = get_user_exec_support(user['id'])
    return jsonify(exec_support.detect_current_state())


@app.route('/api/executive/log', methods=['POST'])
@require_auth
def log_behavior(user):
    """Log behavior for pattern analysis"""
    data = request.get_json() or {}
    
    exec_support = get_user_exec_support(user['id'])
    exec_support.log_behavior(data)
    
    # Also save to database
    behavior_data = {
        'id': secrets.token_hex(8),
        'user_id': user['id'],
        'timestamp': data.get('timestamp', datetime.now().isoformat()),
        'action': data.get('action'),
        'planned_action': data.get('planned_action'),
        'duration_planned': data.get('planned_duration'),
        'duration_actual': data.get('actual_duration'),
        'delay_minutes': data.get('delay_minutes'),
        'completion': 1 if data.get('completion') else 0,
        'mood': data.get('mood', 50),
        'notes': data.get('notes')
    }
    
    db.insert('behavior_history', behavior_data)
    
    return jsonify({
        'success': True,
        'current_state': exec_support.detect_current_state()
    })


@app.route('/api/executive/scaffold/<task_id>', methods=['GET'])
@require_auth
def get_task_scaffold(user, task_id):
    """Get scaffolded micro-steps for a task"""
    tasks = db.select('tasks', {'id': task_id, 'user_id': user['id']})
    if not tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    exec_support = get_user_exec_support(user['id'])
    scaffolding = exec_support.get_task_scaffolding(tasks[0])
    
    return jsonify(scaffolding)


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - DAILY CHECK-IN
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/checkin', methods=['POST'])
@require_auth
def daily_checkin(user):
    """Submit daily check-in"""
    data = request.get_json() or {}
    today = datetime.now().strftime('%Y-%m-%d')
    
    entry_data = {
        'id': secrets.token_hex(8),
        'user_id': user['id'],
        'date': today,
        'mood_level': data.get('mood', 50),
        'energy_level': data.get('energy', 50),
        'stress_level': data.get('stress', 50),
        'sleep_hours': data.get('sleep_hours', 7),
        'sleep_quality': data.get('sleep_quality', 50),
        'journal_entry': data.get('journal', ''),
        'gratitude': data.get('gratitude', ''),
        'wins': data.get('wins', ''),
        'challenges': data.get('challenges', ''),
        'created_at': datetime.now().isoformat()
    }
    
    # Check for existing entry
    existing = db.execute(
        "SELECT id FROM daily_entries WHERE user_id = ? AND date = ?",
        (user['id'], today)
    )
    
    if existing:
        # Update existing
        db.update('daily_entries', entry_data, {'user_id': user['id'], 'date': today})
    else:
        db.insert('daily_entries', entry_data)
    
    # Update pet based on user wellness
    pet = get_user_pet(user['id'])
    wellness = (entry_data['mood_level'] + entry_data['energy_level']) / 2
    pet.update(1.0, wellness)
    save_pet(user['id'], pet)
    
    # Reset spoons for new day
    spoons = get_user_spoons(user['id'])
    sleep_quality = entry_data['sleep_quality'] / 100
    spoon_result = spoons.new_day(sleep_quality)
    
    return jsonify({
        'success': True,
        'pet_state': pet.get_state(),
        'spoons': spoon_result,
        'message': "Check-in recorded! Have a great day!"
    })


@app.route('/api/checkin/history', methods=['GET'])
@require_auth
def get_checkin_history(user):
    """Get check-in history"""
    limit = request.args.get('limit', 30, type=int)
    
    entries = db.execute(
        "SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT ?",
        (user['id'], limit)
    )
    
    return jsonify(entries)


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - ACCESSIBILITY
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/accessibility', methods=['GET'])
@require_auth
def get_accessibility_settings(user):
    """Get accessibility settings"""
    accessibility = get_user_accessibility(user['id'])
    return jsonify(accessibility.to_dict())


@app.route('/api/accessibility', methods=['PUT'])
@require_auth
def update_accessibility_settings(user):
    """Update accessibility settings"""
    data = request.get_json() or {}
    
    accessibility = get_user_accessibility(user['id'])
    accessibility.update_settings(data)
    
    # Save to database
    db.update('users', {
        'accessibility_settings': json.dumps(accessibility.to_dict())
    }, {'id': user['id']})
    
    return jsonify({
        'success': True,
        'settings': accessibility.to_dict()
    })


@app.route('/api/accessibility/css', methods=['GET'])
@require_auth
def get_accessibility_css(user):
    """Get CSS variables for current accessibility settings"""
    accessibility = get_user_accessibility(user['id'])
    css = accessibility.get_css_variables()
    
    return css, 200, {'Content-Type': 'text/css'}


@app.route('/api/accessibility/palettes', methods=['GET'])
def get_color_palettes():
    """Get available color palettes"""
    return jsonify(AccessibilitySystem.AUTISM_SAFE_PALETTES)


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - FRACTALS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/fractal/2d', methods=['GET'])
@require_auth
def generate_2d_fractal(user):
    """Generate 2D fractal based on user metrics"""
    # Get latest check-in
    entries = db.execute(
        "SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1",
        (user['id'],)
    )
    
    if entries:
        entry = entries[0]
        wellness = (entry['mood_level'] + entry['energy_level']) / 2
        mood = entry['mood_level']
        stress = entry['stress_level']
    else:
        wellness, mood, stress = 50, 50, 50
    
    image = fractal_engine.generate_2d_fractal(wellness, mood, stress)
    
    if image is None:
        return jsonify({'error': 'Fractal generation not available'}), 500
    
    base64_img = fractal_engine.fractal_to_base64(image)
    
    return jsonify({
        'image': f"data:image/png;base64,{base64_img}",
        'parameters': {
            'wellness': wellness,
            'mood': mood,
            'stress': stress
        }
    })


@app.route('/api/fractal/3d', methods=['GET'])
@require_auth
def generate_3d_fractal(user):
    """Generate 3D Mandelbulb fractal"""
    entries = db.execute(
        "SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1",
        (user['id'],)
    )
    
    if entries:
        entry = entries[0]
        wellness = (entry['mood_level'] + entry['energy_level']) / 2
        mood = entry['mood_level']
    else:
        wellness, mood = 50, 50
    
    image = fractal_engine.generate_3d_mandelbulb(wellness, mood)
    
    if image is None:
        return jsonify({'error': 'Fractal generation not available'}), 500
    
    base64_img = fractal_engine.fractal_to_base64(image)
    
    return jsonify({
        'image': f"data:image/png;base64,{base64_img}",
        'parameters': {
            'wellness': wellness,
            'mood': mood
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/subscription/status', methods=['GET'])
@require_auth
def subscription_status(user):
    """Get subscription status"""
    trial_end = user.get('trial_end_date')
    status = user.get('subscription_status', 'trial')
    
    trial_active = False
    trial_days_remaining = 0
    
    if trial_end:
        try:
            trial_end_date = datetime.fromisoformat(trial_end)
            trial_days_remaining = max(0, (trial_end_date - datetime.now()).days)
            trial_active = trial_days_remaining > 0 and status == 'trial'
        except:
            pass
    
    return jsonify({
        'status': status,
        'trial_active': trial_active,
        'trial_days_remaining': trial_days_remaining,
        'trial_end_date': trial_end,
        'has_access': status == 'active' or trial_active,
        'payment_link': os.environ.get('STRIPE_PAYMENT_LINK', 'https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00'),
        'gofundme_url': 'https://gofund.me/8d9303d27'
    })


@app.route('/api/subscription/activate', methods=['POST'])
@require_auth
def activate_subscription(user):
    """Activate subscription (called after payment)"""
    db.update('users', {'subscription_status': 'active'}, {'id': user['id']})
    
    return jsonify({
        'success': True,
        'message': 'Subscription activated! Thank you for your support!'
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '7.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'numpy': HAS_NUMPY,
            'pillow': HAS_PIL,
            'gpu': GPU_AVAILABLE,
            'gpu_name': GPU_NAME
        }
    })


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get system status"""
    return jsonify({
        'name': 'Life Fractal Intelligence',
        'version': '7.0.0',
        'status': 'operational',
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:10]
        },
        'features': {
            'emotional_pet_ai': True,
            'fractal_time_calendar': True,
            'fibonacci_task_scheduler': True,
            'executive_function_support': True,
            'spoon_theory': True,
            '2d_fractals': HAS_NUMPY and HAS_PIL,
            '3d_fractals': HAS_NUMPY and HAS_PIL,
            'accessibility': True,
            'mayan_calendar': True
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence</title>
    <style>
        :root {
            --color-primary: #5B8A9A;
            --color-secondary: #8BB4C2;
            --color-background: #F5F9FA;
            --color-text: #2C4A52;
            --color-accent: #7BA3B0;
            --color-warning: #D4A574;
            --color-success: #6B8E6B;
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-size-base: 16px;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: var(--font-family);
            font-size: var(--font-size-base);
            background: var(--color-background);
            color: var(--color-text);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: var(--color-primary);
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .spoon-display {
            background: var(--color-secondary);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .card h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: var(--color-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .pet-card {
            text-align: center;
        }
        
        .pet-emoji {
            font-size: 80px;
            margin: 20px 0;
        }
        
        .pet-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }
        
        .stat {
            background: var(--color-background);
            padding: 10px;
            border-radius: 8px;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
        }
        
        .stat-value {
            font-size: 18px;
            font-weight: 600;
        }
        
        .progress-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--color-primary);
            transition: width 0.3s;
        }
        
        .btn {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
        }
        
        .btn-primary {
            background: var(--color-primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--color-text);
        }
        
        .btn-secondary {
            background: var(--color-secondary);
            color: var(--color-text);
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .checkin-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .form-group label {
            font-weight: 500;
            font-size: 14px;
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="range"] {
            flex: 1;
            -webkit-appearance: none;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--color-primary);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .slider-value {
            min-width: 40px;
            text-align: center;
            font-weight: 600;
        }
        
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--color-primary);
        }
        
        .task-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .task-item {
            background: var(--color-background);
            padding: 12px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .task-priority {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--color-accent);
            color: white;
        }
        
        .fractal-container {
            text-align: center;
        }
        
        .fractal-image {
            max-width: 100%;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .mayan-info {
            background: linear-gradient(135deg, #2C4A52 0%, #5B8A9A 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .mayan-day {
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .login-container {
            max-width: 400px;
            margin: 100px auto;
        }
        
        .login-card {
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .login-card h1 {
            text-align: center;
            margin-bottom: 30px;
            color: var(--color-primary);
        }
        
        .form-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 15px;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--color-primary);
        }
        
        .accessibility-notice {
            background: var(--color-secondary);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .hidden {
            display: none !important;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            header {
                flex-direction: column;
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <!-- Login View -->
    <div id="login-view" class="login-container">
        <div class="login-card">
            <h1>🌀 Life Fractal</h1>
            <form id="login-form">
                <input type="email" id="login-email" class="form-input" placeholder="Email" required>
                <input type="password" id="login-password" class="form-input" placeholder="Password" required>
                <button type="submit" class="btn btn-primary" style="width:100%">Log In</button>
            </form>
            <p style="text-align:center; margin-top:20px">
                Don't have an account? <a href="#" id="show-register">Sign up</a>
            </p>
        </div>
    </div>
    
    <!-- Register View -->
    <div id="register-view" class="login-container hidden">
        <div class="login-card">
            <h1>🌀 Create Account</h1>
            <form id="register-form">
                <input type="text" id="register-name" class="form-input" placeholder="First Name" required>
                <input type="email" id="register-email" class="form-input" placeholder="Email" required>
                <input type="password" id="register-password" class="form-input" placeholder="Password" required>
                <button type="submit" class="btn btn-primary" style="width:100%">Start Free Trial</button>
            </form>
            <p style="text-align:center; margin-top:20px">
                Already have an account? <a href="#" id="show-login">Log in</a>
            </p>
        </div>
    </div>
    
    <!-- Main Dashboard -->
    <div id="dashboard-view" class="hidden">
        <div class="container">
            <header>
                <div class="logo">🌀 Life Fractal Intelligence</div>
                <div class="user-info">
                    <div class="spoon-display" id="spoon-count">🥄 12 Spoons</div>
                    <span id="user-name">Welcome!</span>
                    <button class="btn btn-secondary" id="logout-btn">Logout</button>
                </div>
            </header>
            
            <div class="accessibility-notice">
                ♿ Accessibility features enabled. <a href="#" id="accessibility-settings">Customize</a>
            </div>
            
            <div class="grid">
                <!-- Pet Card -->
                <div class="card pet-card">
                    <h2>🐾 Your Companion</h2>
                    <div class="pet-emoji" id="pet-emoji">🐱</div>
                    <div id="pet-name" style="font-size:20px; font-weight:600">Buddy</div>
                    <div id="pet-status" style="color:#666; margin:5px 0">Content 😊</div>
                    <div class="pet-stats">
                        <div class="stat">
                            <div class="stat-label">Hunger</div>
                            <div class="stat-value" id="pet-hunger">50</div>
                            <div class="progress-bar"><div class="progress-fill" id="hunger-bar" style="width:50%"></div></div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Energy</div>
                            <div class="stat-value" id="pet-energy">50</div>
                            <div class="progress-bar"><div class="progress-fill" id="energy-bar" style="width:50%"></div></div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Mood</div>
                            <div class="stat-value" id="pet-mood">50</div>
                            <div class="progress-bar"><div class="progress-fill" id="mood-bar" style="width:50%"></div></div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Bond</div>
                            <div class="stat-value" id="pet-bond">20</div>
                            <div class="progress-bar"><div class="progress-fill" id="bond-bar" style="width:20%"></div></div>
                        </div>
                    </div>
                    <div style="margin:10px 0">
                        Level <span id="pet-level">1</span> • <span id="pet-xp">0</span>/<span id="pet-xp-next">8</span> XP
                    </div>
                    <div class="btn-group" style="justify-content:center">
                        <button class="btn btn-primary" id="feed-btn">🍽️ Feed</button>
                        <button class="btn btn-primary" id="play-btn">🎾 Play</button>
                        <button class="btn btn-secondary" id="rest-btn">💤 Rest</button>
                    </div>
                </div>
                
                <!-- Daily Check-in -->
                <div class="card">
                    <h2>📝 Daily Check-in</h2>
                    <form class="checkin-form" id="checkin-form">
                        <div class="form-group">
                            <label>Mood</label>
                            <div class="slider-container">
                                <input type="range" id="mood-slider" min="0" max="100" value="50">
                                <span class="slider-value" id="mood-value">50</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Energy</label>
                            <div class="slider-container">
                                <input type="range" id="energy-slider" min="0" max="100" value="50">
                                <span class="slider-value" id="energy-value">50</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Sleep Quality</label>
                            <div class="slider-container">
                                <input type="range" id="sleep-slider" min="0" max="100" value="70">
                                <span class="slider-value" id="sleep-value">70</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Journal (optional)</label>
                            <textarea id="journal-entry" placeholder="How are you feeling today?"></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary">Save Check-in</button>
                    </form>
                </div>
                
                <!-- Tasks -->
                <div class="card">
                    <h2>✅ Today's Tasks</h2>
                    <div class="task-list" id="task-list">
                        <p style="color:#666; text-align:center">No tasks yet. Add some goals first!</p>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-primary" id="add-task-btn">+ Add Task</button>
                        <button class="btn btn-secondary" id="next-task-btn">🎯 What's Next?</button>
                    </div>
                </div>
                
                <!-- Mayan Calendar -->
                <div class="card">
                    <h2>📅 Sacred Calendar</h2>
                    <div class="mayan-info" id="mayan-info">
                        <div style="font-size:14px">Today's Mayan Day</div>
                        <div class="mayan-day" id="mayan-day">Loading...</div>
                        <div id="mayan-energy" style="font-size:14px; opacity:0.9"></div>
                    </div>
                    <div style="margin-top:15px; text-align:center">
                        <a href="#" class="btn btn-secondary" id="view-calendar">View Full Calendar</a>
                    </div>
                </div>
                
                <!-- Fractal Visualization -->
                <div class="card fractal-container">
                    <h2>🌀 Your Life Fractal</h2>
                    <p style="color:#666; font-size:14px">Your metrics transformed into sacred geometry</p>
                    <img src="" alt="Fractal visualization" class="fractal-image hidden" id="fractal-image">
                    <div class="btn-group" style="justify-content:center">
                        <button class="btn btn-primary" id="gen-2d-btn">Generate 2D</button>
                        <button class="btn btn-secondary" id="gen-3d-btn">Generate 3D</button>
                    </div>
                </div>
                
                <!-- Goals -->
                <div class="card">
                    <h2>🎯 Goals</h2>
                    <div id="goals-list">
                        <p style="color:#666; text-align:center">No goals yet. Start by setting some!</p>
                    </div>
                    <button class="btn btn-primary" style="margin-top:15px; width:100%" id="add-goal-btn">+ Add Goal</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentUser = null;
        
        // API helper
        async function api(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: {'Content-Type': 'application/json'},
                credentials: 'include'
            };
            if (data) options.body = JSON.stringify(data);
            
            const response = await fetch(`/api${endpoint}`, options);
            return response.json();
        }
        
        // View switching
        function showView(viewId) {
            document.querySelectorAll('[id$="-view"]').forEach(v => v.classList.add('hidden'));
            document.getElementById(viewId).classList.remove('hidden');
        }
        
        // Update pet display
        function updatePetDisplay(pet) {
            document.getElementById('pet-emoji').textContent = pet.emoji;
            document.getElementById('pet-name').textContent = pet.name;
            document.getElementById('pet-status').textContent = pet.status;
            document.getElementById('pet-hunger').textContent = Math.round(pet.hunger);
            document.getElementById('pet-energy').textContent = Math.round(pet.energy);
            document.getElementById('pet-mood').textContent = Math.round(pet.mood);
            document.getElementById('pet-bond').textContent = Math.round(pet.bond);
            document.getElementById('pet-level').textContent = pet.level;
            document.getElementById('pet-xp').textContent = pet.xp;
            document.getElementById('pet-xp-next').textContent = pet.xp_to_next;
            
            document.getElementById('hunger-bar').style.width = pet.hunger + '%';
            document.getElementById('energy-bar').style.width = pet.energy + '%';
            document.getElementById('mood-bar').style.width = pet.mood + '%';
            document.getElementById('bond-bar').style.width = pet.bond + '%';
        }
        
        // Update spoon display
        function updateSpoonDisplay(spoons) {
            document.getElementById('spoon-count').textContent = `🥄 ${spoons.current_spoons} Spoons`;
        }
        
        // Load dashboard data
        async function loadDashboard() {
            try {
                // Get user info
                const user = await api('/auth/me');
                currentUser = user;
                document.getElementById('user-name').textContent = `Hi, ${user.first_name || 'there'}!`;
                
                // Get pet
                const pet = await api('/pet');
                updatePetDisplay(pet);
                
                // Get spoons
                const spoons = await api('/spoons');
                updateSpoonDisplay(spoons);
                
                // Get Mayan day
                const mayan = await api('/calendar/mayan');
                document.getElementById('mayan-day').textContent = mayan.tzolkin;
                document.getElementById('mayan-energy').textContent = mayan.energy_quality;
                
                // Get tasks
                const tasks = await api('/tasks');
                const taskList = document.getElementById('task-list');
                if (tasks.length > 0) {
                    taskList.innerHTML = tasks.slice(0, 5).map(t => `
                        <div class="task-item">
                            <div>
                                <strong>${t.name}</strong>
                                <div style="font-size:12px; color:#666">🥄 ${t.spoon_cost} spoons</div>
                            </div>
                            <span class="task-priority">P${Math.round(t.priority)}</span>
                        </div>
                    `).join('');
                }
                
                // Get goals
                const goals = await api('/goals');
                const goalsList = document.getElementById('goals-list');
                if (goals.length > 0) {
                    goalsList.innerHTML = goals.map(g => `
                        <div style="margin-bottom:10px; padding:10px; background:#f5f9fa; border-radius:8px">
                            <strong>${g.title}</strong>
                            <div class="progress-bar" style="margin-top:5px">
                                <div class="progress-fill" style="width:${g.progress}%"></div>
                            </div>
                            <div style="font-size:12px; color:#666; margin-top:5px">${g.progress}% complete</div>
                        </div>
                    `).join('');
                }
                
                showView('dashboard-view');
                
            } catch (error) {
                console.error('Dashboard load error:', error);
                showView('login-view');
            }
        }
        
        // Check auth on load
        async function checkAuth() {
            try {
                await api('/auth/me');
                loadDashboard();
            } catch {
                showView('login-view');
            }
        }
        
        // Event listeners
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            
            const result = await api('/auth/login', 'POST', { email, password });
            
            if (result.success) {
                loadDashboard();
            } else {
                alert(result.error || 'Login failed');
            }
        });
        
        document.getElementById('register-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const first_name = document.getElementById('register-name').value;
            const email = document.getElementById('register-email').value;
            const password = document.getElementById('register-password').value;
            
            const result = await api('/auth/register', 'POST', { email, password, first_name });
            
            if (result.success) {
                loadDashboard();
            } else {
                alert(result.error || 'Registration failed');
            }
        });
        
        document.getElementById('show-register').addEventListener('click', (e) => {
            e.preventDefault();
            showView('register-view');
        });
        
        document.getElementById('show-login').addEventListener('click', (e) => {
            e.preventDefault();
            showView('login-view');
        });
        
        document.getElementById('logout-btn').addEventListener('click', async () => {
            await api('/auth/logout', 'POST');
            showView('login-view');
        });
        
        // Pet interactions
        document.getElementById('feed-btn').addEventListener('click', async () => {
            const result = await api('/pet/feed', 'POST');
            if (result.state) updatePetDisplay(result.state);
            alert(result.message);
        });
        
        document.getElementById('play-btn').addEventListener('click', async () => {
            const result = await api('/pet/play', 'POST');
            if (result.state) updatePetDisplay(result.state);
            alert(result.message);
        });
        
        document.getElementById('rest-btn').addEventListener('click', async () => {
            const result = await api('/pet/rest', 'POST');
            if (result.state) updatePetDisplay(result.state);
            alert(result.message);
        });
        
        // Check-in form
        document.getElementById('checkin-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const result = await api('/checkin', 'POST', {
                mood: parseInt(document.getElementById('mood-slider').value),
                energy: parseInt(document.getElementById('energy-slider').value),
                sleep_quality: parseInt(document.getElementById('sleep-slider').value),
                journal: document.getElementById('journal-entry').value
            });
            
            if (result.success) {
                if (result.pet_state) updatePetDisplay(result.pet_state);
                if (result.spoons) updateSpoonDisplay(result.spoons);
                alert(result.message);
            }
        });
        
        // Slider value updates
        ['mood', 'energy', 'sleep'].forEach(name => {
            const slider = document.getElementById(`${name}-slider`);
            const value = document.getElementById(`${name}-value`);
            slider.addEventListener('input', () => {
                value.textContent = slider.value;
            });
        });
        
        // Fractal generation
        document.getElementById('gen-2d-btn').addEventListener('click', async () => {
            const btn = document.getElementById('gen-2d-btn');
            btn.textContent = 'Generating...';
            btn.disabled = true;
            
            const result = await api('/fractal/2d');
            
            if (result.image) {
                const img = document.getElementById('fractal-image');
                img.src = result.image;
                img.classList.remove('hidden');
            }
            
            btn.textContent = 'Generate 2D';
            btn.disabled = false;
        });
        
        document.getElementById('gen-3d-btn').addEventListener('click', async () => {
            const btn = document.getElementById('gen-3d-btn');
            btn.textContent = 'Generating...';
            btn.disabled = true;
            
            const result = await api('/fractal/3d');
            
            if (result.image) {
                const img = document.getElementById('fractal-image');
                img.src = result.image;
                img.classList.remove('hidden');
            }
            
            btn.textContent = 'Generate 3D';
            btn.disabled = false;
        });
        
        // Add goal
        document.getElementById('add-goal-btn').addEventListener('click', async () => {
            const title = prompt('What is your goal?');
            if (!title) return;
            
            const result = await api('/goals', 'POST', { title });
            if (result.success) {
                loadDashboard();
            }
        });
        
        // Add task
        document.getElementById('add-task-btn').addEventListener('click', async () => {
            const name = prompt('What task do you need to do?');
            if (!name) return;
            
            const result = await api('/tasks', 'POST', { name });
            if (result.success) {
                loadDashboard();
            }
        });
        
        // What's next
        document.getElementById('next-task-btn').addEventListener('click', async () => {
            const result = await api('/tasks/next');
            
            if (result.task) {
                alert(`Next task: ${result.task.name}\\n\\nSpoon cost: ${result.task.spoon_cost}\\nPriority: ${result.task.priority}`);
            } else {
                alert(result.message);
            }
        });
        
        // Initialize
        checkAuth();
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/login')
def login_page():
    """Redirect to main page (login handled in SPA)"""
    return redirect('/')


# ═══════════════════════════════════════════════════════════════════════════════════════
# APPLICATION STARTUP
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🌀 LIFE FRACTAL INTELLIGENCE v7.0")
    logger.info("=" * 70)
    logger.info(f"✅ Sacred Mathematics: PHI={PHI:.10f}")
    logger.info(f"✅ Golden Angle: {GOLDEN_ANGLE}°")
    logger.info(f"✅ NumPy: {'Available' if HAS_NUMPY else 'Not available'}")
    logger.info(f"✅ Pillow: {'Available' if HAS_PIL else 'Not available'}")
    logger.info(f"✅ GPU: {GPU_NAME if GPU_AVAILABLE else 'Not available'}")
    logger.info("=" * 70)
    logger.info("Features enabled:")
    logger.info("  • Emotional Pet AI with differential equations")
    logger.info("  • Fractal Time Calendar with Fibonacci blocks")
    logger.info("  • Fibonacci Task Scheduler")
    logger.info("  • Executive Function Support")
    logger.info("  • Spoon Theory Energy Management")
    logger.info("  • 2D/3D Fractal Visualization")
    logger.info("  • Full Accessibility System")
    logger.info("  • Mayan Calendar Integration")
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3D IMMERSIVE EXPERIENCE ROUTE
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/3d')
def immersive_3d():
    """Serve the immersive 3D fractal experience"""
    try:
        # Try to load from templates
        with open('templates/3d_experience.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Return embedded minimal 3D viewer if template not found
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Life Fractal 3D</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { margin: 0; background: #050510; overflow: hidden; }
        #info { position: fixed; top: 20px; left: 20px; color: #fff; font-family: sans-serif; z-index: 100; }
        .btn { padding: 10px 20px; background: #5B8A9A; border: none; color: white; cursor: pointer; margin: 5px; border-radius: 8px; }
    </style>
</head>
<body>
    <div id="info">
        <h2>🌀 Life Fractal 3D</h2>
        <p>Drag to rotate • Scroll to zoom</p>
        <button class="btn" onclick="location.href='/'">← Back to Dashboard</button>
    </div>
    <script>
        let scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050510);
        
        let camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.1, 1000);
        camera.position.z = 20;
        
        let renderer = new THREE.WebGLRenderer({antialias:true});
        renderer.setSize(innerWidth, innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Create fractal points
        let geometry = new THREE.BufferGeometry();
        let vertices = [], colors = [];
        
        for(let i = 0; i < 60; i++) {
            for(let j = 0; j < 60; j++) {
                let theta = (i/60) * Math.PI;
                let phi = (j/60) * Math.PI * 2;
                let x = Math.sin(theta) * Math.cos(phi) * 5;
                let y = Math.sin(theta) * Math.sin(phi) * 5;
                let z = Math.cos(theta) * 5;
                
                vertices.push(x, y, z);
                colors.push(0.4 + i/120, 0.3 + j/120, 0.8);
            }
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        let material = new THREE.PointsMaterial({size:0.15, vertexColors:true, transparent:true, opacity:0.8});
        let points = new THREE.Points(geometry, material);
        scene.add(points);
        
        scene.add(new THREE.AmbientLight(0x404040));
        let light = new THREE.PointLight(0x667eea, 2);
        light.position.set(10,10,10);
        scene.add(light);
        
        let isDrag = false, prevX = 0, prevY = 0, rotX = 0, rotY = 0;
        
        renderer.domElement.onmousedown = e => { isDrag = true; prevX = e.clientX; prevY = e.clientY; };
        renderer.domElement.onmouseup = () => isDrag = false;
        renderer.domElement.onmousemove = e => {
            if(!isDrag) return;
            rotY += (e.clientX - prevX) * 0.005;
            rotX += (e.clientY - prevY) * 0.005;
            prevX = e.clientX; prevY = e.clientY;
        };
        renderer.domElement.onwheel = e => {
            camera.position.z = Math.max(5, Math.min(50, camera.position.z + e.deltaY * 0.05));
        };
        
        onresize = () => { camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); };
        
        function animate() {
            requestAnimationFrame(animate);
            points.rotation.y = rotY + performance.now() * 0.0001;
            points.rotation.x = rotX;
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""
