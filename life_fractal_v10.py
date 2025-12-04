#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LIFE FRACTAL INTELLIGENCE v12.0 - LIVING MATHEMATICAL ORGANISM                                  ║
║  ════════════════════════════════════════════════════════════════════════════════════════════════║
║                                                                                                  ║
║  🌀 FULL-SCREEN INTERACTIVE 3D FRACTAL UNIVERSE                                                  ║
║  🧠 OLLAMA AI INTEGRATION - Self-spawning intelligent orbs with generated meaning                ║
║  🔬 ZOOM-BASED LABELS - Text visible on zoom, hidden when pulled out                             ║
║  🧬 SELF-AWARE ORBS - Each cell knows its purpose, replicates organically                        ║
║  🎨 NORDIC MINIMAL GUI - Hamburger menu, app-like navigation                                     ║
║  📅 MAYAN CALENDAR - Sacred time science for task prioritization                                 ║
║  🌊 SWARM INTELLIGENCE - Pattern seeking, trend analysis                                         ║
║  ⚖️ KARMA-DHARMA MATHEMATICS - Spiritual scoring engine                                          ║
║  🤖 FEDERATED AI - Recursive self-improvement on server                                          ║
║                                                                                                  ║
║  For neurodivergent minds: External visualization, energy tracking, compassionate UX            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
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
import random
import asyncio
import threading
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
from collections import defaultdict
import base64
import urllib.request
import urllib.error

# Flask
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, Response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ML Support
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ═══════════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS - UNIVERSAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INVERSE = 1 / PHI
PHI_SQUARED = PHI * PHI
GOLDEN_ANGLE = 360 / (PHI ** 2)
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
PLANCK_KARMA = 1e-43
DHARMA_FREQUENCY = 432
SCHUMANN_RESONANCE = 7.83
SOLFEGGIO = [174, 285, 396, 417, 528, 639, 741, 852, 963]

# Mayan Calendar Constants
MAYAN_KIN = 20  # Day signs
MAYAN_TRECENA = 13  # Number cycle
MAYAN_TZOLKIN = 260  # Sacred calendar cycle (13 × 20)
MAYAN_HAAB = 365  # Solar calendar
MAYAN_KATUN = 7200  # 20 years
MAYAN_BAKTUN = 144000  # 400 years

# Mayan day signs (Nahual)
MAYAN_SIGNS = [
    "Imix (Dragon)", "Ik (Wind)", "Akbal (Night)", "Kan (Seed)", "Chicchan (Serpent)",
    "Cimi (Death)", "Manik (Deer)", "Lamat (Star)", "Muluc (Water)", "Oc (Dog)",
    "Chuen (Monkey)", "Eb (Road)", "Ben (Reed)", "Ix (Jaguar)", "Men (Eagle)",
    "Cib (Owl)", "Caban (Earth)", "Etznab (Mirror)", "Cauac (Storm)", "Ahau (Sun)"
]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# OLLAMA AI INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

class OllamaAI:
    """
    Integration with Ollama for AI-generated orb meanings and self-spawning intelligence.
    Falls back to pattern-based generation if Ollama unavailable.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.1"
        self.available = False
        self.cache: Dict[str, str] = {}
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is running"""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                self.available = response.status == 200
                logger.info(f"🤖 Ollama AI: {'Connected' if self.available else 'Not available'}")
        except:
            self.available = False
            logger.info("🤖 Ollama AI: Using pattern-based generation (offline mode)")
    
    def generate(self, prompt: str, context: Dict = None) -> str:
        """Generate text using Ollama or fallback"""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.available:
            try:
                return self._ollama_generate(prompt)
            except:
                pass
        
        # Fallback to pattern-based generation
        return self._pattern_generate(prompt, context)
    
    def _ollama_generate(self, prompt: str) -> str:
        """Call Ollama API"""
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 100}
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('response', '').strip()
            self.cache[hashlib.md5(prompt.encode()).hexdigest()[:16]] = text
            return text
    
    def _pattern_generate(self, prompt: str, context: Dict = None) -> str:
        """Generate meaningful text using sacred mathematics patterns"""
        context = context or {}
        
        # Meaning templates based on orb type
        meanings = {
            'stem': [
                "The seed of infinite possibility, containing all paths",
                "Undifferentiated potential awaiting purpose",
                "The origin point from which all growth emerges",
                "Pure creative energy before manifestation"
            ],
            'neuron': [
                "A bridge connecting thought to action",
                "Processing the signals of intention",
                "The spark of consciousness in motion",
                "Weaving patterns of understanding"
            ],
            'memory': [
                "Holding the echoes of experience",
                "A vessel for learned wisdom",
                "The accumulation of past moments",
                "Preserving what matters for tomorrow"
            ],
            'sensor': [
                "Perceiving the subtle currents of change",
                "Attuned to the vibrations of environment",
                "Receiving messages from the universe",
                "The gateway between inner and outer worlds"
            ],
            'effector': [
                "Translating intention into reality",
                "The hand that shapes the world",
                "Where thought becomes action",
                "Manifesting will into being"
            ],
            'structural': [
                "The foundation upon which growth builds",
                "Providing stability in chaos",
                "The skeleton of possibility",
                "Supporting the architecture of dreams"
            ],
            'transport': [
                "Carrying energy where it's needed",
                "The river of life flowing through",
                "Connecting distant parts into whole",
                "Movement is the essence of life"
            ],
            'goal': [
                "A star to navigate by",
                "The destination that gives journey meaning",
                "Crystallized intention waiting to manifest",
                "Where desire meets determination"
            ],
            'habit': [
                "The groove worn by repetition",
                "Small actions, great transformations",
                "Building the self through daily practice",
                "The compound interest of self-improvement"
            ],
            'karma': [
                f"Resonating at {context.get('karma', 0):.2f} karmic frequency",
                "The echo of actions past and future",
                "Cause and effect in eternal dance",
                "What you send returns transformed"
            ]
        }
        
        orb_type = context.get('type', 'stem').lower()
        templates = meanings.get(orb_type, meanings['stem'])
        
        # Use golden ratio to select template
        index = int(abs(context.get('karma', 0) * PHI * 100)) % len(templates)
        base_meaning = templates[index]
        
        # Add contextual details
        if context.get('generation', 0) > 0:
            base_meaning += f" [Generation {context.get('generation')}]"
        
        if context.get('energy', 0) > 0.8:
            base_meaning += " ✨ High energy"
        elif context.get('energy', 0) < 0.3:
            base_meaning += " 🌙 Resting state"
        
        return base_meaning
    
    def generate_orb_meaning(self, orb_data: Dict) -> str:
        """Generate meaning for a specific orb"""
        prompt = f"""Generate a brief, poetic meaning (1-2 sentences) for a living orb with these properties:
Type: {orb_data.get('type', 'stem')}
Karma: {orb_data.get('karma', 0):.2f}
Energy: {orb_data.get('energy', 1.0):.2f}
Generation: {orb_data.get('generation', 0)}
State: {orb_data.get('state', 'growing')}

The meaning should be mystical but relevant to personal growth and planning."""
        
        return self.generate(prompt, orb_data)
    
    def generate_insight(self, patterns: Dict) -> str:
        """Generate AI insight from detected patterns"""
        prompt = f"""Based on these user patterns, provide one encouraging insight (1-2 sentences):
Karma trend: {patterns.get('karma_trend', 0):.3f}
Harmony level: {patterns.get('harmony', 0.5):.2f}
Active goals: {patterns.get('goals', 0)}
Habit streaks: {patterns.get('streaks', [])}

Be warm, supportive, and neurodivergent-friendly."""
        
        return self.generate(prompt, patterns)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MayanCalendar:
    """
    Sacred Mayan time science for task prioritization and cosmic alignment.
    """
    
    # Mayan epoch: August 11, 3114 BCE (Julian) = September 6, 3114 BCE (Gregorian)
    MAYAN_EPOCH = datetime(3114, 8, 11)  # Simplified
    
    def __init__(self):
        self.today = datetime.now()
    
    def get_tzolkin(self, date: datetime = None) -> Dict:
        """Calculate Tzolkin (sacred 260-day calendar) position"""
        date = date or datetime.now()
        
        # Days since a reference point (simplified calculation)
        ref_date = datetime(2012, 12, 21)  # End of 13th Baktun
        days_diff = (date - ref_date).days
        
        # Tzolkin position
        day_number = ((days_diff % 13) + 1)  # 1-13
        day_sign_index = days_diff % 20
        day_sign = MAYAN_SIGNS[day_sign_index]
        
        return {
            'number': day_number,
            'sign': day_sign,
            'sign_index': day_sign_index,
            'kin': (days_diff % 260) + 1,
            'energy': self._calculate_day_energy(day_number, day_sign_index)
        }
    
    def _calculate_day_energy(self, number: int, sign_index: int) -> Dict:
        """Calculate the energetic quality of a Tzolkin day"""
        # Each number has meaning
        number_meanings = {
            1: "New beginnings, unity",
            2: "Duality, choices",
            3: "Action, movement",
            4: "Stability, foundation",
            5: "Center, empowerment",
            6: "Flow, organic growth",
            7: "Reflection, mysticism",
            8: "Harmony, justice",
            9: "Completion, patience",
            10: "Manifestation, intention",
            11: "Resolution, change",
            12: "Understanding, wisdom",
            13: "Transcendence, completion"
        }
        
        # Sign element associations
        sign_elements = ['water', 'air', 'earth', 'earth', 'fire',
                        'earth', 'air', 'fire', 'water', 'fire',
                        'air', 'earth', 'water', 'earth', 'air',
                        'earth', 'earth', 'air', 'water', 'fire']
        
        return {
            'number_meaning': number_meanings.get(number, "Mystery"),
            'element': sign_elements[sign_index],
            'power_level': (number + sign_index) % 10 / 10,  # 0-1
            'recommended_activities': self._get_recommended_activities(number, sign_index)
        }
    
    def _get_recommended_activities(self, number: int, sign_index: int) -> List[str]:
        """Get recommended activities for this day's energy"""
        activities = []
        
        # Number-based recommendations
        if number in [1, 3, 5]:
            activities.append("Start new projects")
        if number in [4, 8]:
            activities.append("Work on foundations")
        if number in [7, 9, 13]:
            activities.append("Reflect and integrate")
        if number in [2, 6, 10]:
            activities.append("Collaborate with others")
        
        # Sign-based recommendations
        sign_activities = {
            0: "Creative work", 1: "Communication", 2: "Inner work",
            3: "Planting seeds", 4: "Transformation", 5: "Release",
            6: "Healing", 7: "Celebration", 8: "Purification",
            9: "Loyalty tasks", 10: "Play and art", 11: "Service",
            12: "Leadership", 13: "Intuition work", 14: "Vision",
            15: "Wisdom sharing", 16: "Grounding", 17: "Self-reflection",
            18: "Cleansing", 19: "Enlightenment"
        }
        
        activities.append(sign_activities.get(sign_index, "General tasks"))
        
        return activities
    
    def get_task_alignment(self, task_type: str, date: datetime = None) -> float:
        """Calculate how well a task type aligns with today's energy (0-1)"""
        tzolkin = self.get_tzolkin(date)
        energy = tzolkin['energy']
        
        # Task type to element mapping
        task_elements = {
            'creative': 'fire',
            'analytical': 'air',
            'physical': 'earth',
            'emotional': 'water',
            'social': 'air',
            'planning': 'earth',
            'rest': 'water',
            'learning': 'fire'
        }
        
        task_element = task_elements.get(task_type.lower(), 'earth')
        
        # Element harmony (simplified)
        element_harmony = {
            ('fire', 'fire'): 1.0, ('fire', 'air'): 0.8, ('fire', 'earth'): 0.5, ('fire', 'water'): 0.3,
            ('air', 'air'): 1.0, ('air', 'fire'): 0.8, ('air', 'water'): 0.5, ('air', 'earth'): 0.4,
            ('earth', 'earth'): 1.0, ('earth', 'water'): 0.7, ('earth', 'fire'): 0.5, ('earth', 'air'): 0.4,
            ('water', 'water'): 1.0, ('water', 'earth'): 0.7, ('water', 'air'): 0.5, ('water', 'fire'): 0.3
        }
        
        harmony = element_harmony.get((task_element, energy['element']), 0.5)
        
        # Adjust by power level
        return harmony * (0.5 + energy['power_level'] * 0.5)
    
    def get_today_summary(self) -> Dict:
        """Get complete Mayan calendar summary for today"""
        tzolkin = self.get_tzolkin()
        
        return {
            'tzolkin': tzolkin,
            'greeting': f"Today is {tzolkin['number']} {tzolkin['sign']}",
            'kin_number': tzolkin['kin'],
            'energy': tzolkin['energy'],
            'best_for': tzolkin['energy']['recommended_activities'],
            'power_level': tzolkin['energy']['power_level'],
            'cosmic_tone': self._get_cosmic_tone(tzolkin['number'])
        }
    
    def _get_cosmic_tone(self, number: int) -> str:
        """Get the cosmic tone name"""
        tones = {
            1: "Magnetic", 2: "Lunar", 3: "Electric", 4: "Self-Existing",
            5: "Overtone", 6: "Rhythmic", 7: "Resonant", 8: "Galactic",
            9: "Solar", 10: "Planetary", 11: "Spectral", 12: "Crystal", 13: "Cosmic"
        }
        return tones.get(number, "Unknown")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# KARMA-DHARMA ENGINE (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KarmicValence(Enum):
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    TRANSFORMATIVE = 2


@dataclass
class KarmicVector:
    """Multidimensional karmic representation"""
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    magnitude: float = 0.0
    valence: KarmicValence = KarmicValence.NEUTRAL
    velocity: float = 0.0
    spin: float = 0.0
    intention: float = 1.0
    awareness: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    meaning: str = ""
    
    @property
    def weight(self) -> float:
        base = self.magnitude * self.intention * self.awareness
        harmonic = PHI if self.valence == KarmicValence.POSITIVE else PHI_INVERSE
        return base * harmonic * (1 + abs(self.spin) * 0.1)
    
    def evolve(self, dt: float) -> 'KarmicVector':
        decay = PHI_INVERSE if self.valence == KarmicValence.NEGATIVE else 1.0
        growth = PHI if self.valence == KarmicValence.POSITIVE else 1.0
        new_mag = self.magnitude * (growth ** (dt * self.awareness)) * (decay ** (dt * (1 - self.awareness)))
        return KarmicVector(
            id=self.id, magnitude=new_mag, valence=self.valence,
            velocity=self.velocity * 0.99, spin=self.spin * 0.999,
            intention=self.intention, awareness=self.awareness,
            source=self.source, meaning=self.meaning
        )


class KarmaDharmaEngine:
    """Core spiritual mathematics engine with federated learning"""
    
    def __init__(self, ai: OllamaAI = None):
        self.vectors: List[KarmicVector] = []
        self.field_potential: float = 0.0
        self.dharmic_angle: float = 0.0
        self.ai = ai
        self.history: List[Dict] = []
        self.learning_rate: float = 0.01
    
    def add_action(self, action_type: str, magnitude: float, 
                   intention: float = 0.8, awareness: float = 0.7) -> KarmicVector:
        positive = {'complete', 'achieve', 'help', 'create', 'meditate', 'learn', 'grow'}
        negative = {'skip', 'avoid', 'procrastinate', 'abandon'}
        
        if action_type.lower() in positive:
            valence = KarmicValence.POSITIVE
        elif action_type.lower() in negative:
            valence = KarmicValence.NEGATIVE
        else:
            valence = KarmicValence.NEUTRAL
        
        fib_idx = int(magnitude * 10) % len(FIBONACCI)
        spin = FIBONACCI[fib_idx] * PHI_INVERSE
        
        # Generate meaning
        meaning = ""
        if self.ai:
            meaning = self.ai.generate_orb_meaning({
                'type': 'karma', 'karma': magnitude * valence.value,
                'energy': awareness, 'generation': len(self.vectors)
            })
        
        vector = KarmicVector(
            magnitude=magnitude, valence=valence,
            velocity=intention * awareness, spin=spin,
            intention=intention, awareness=awareness,
            source=action_type, meaning=meaning
        )
        
        self.vectors.append(vector)
        self._recalculate()
        
        self.history.append({
            'id': vector.id, 'action': action_type,
            'weight': vector.weight, 'valence': valence.name,
            'timestamp': time.time(), 'meaning': meaning
        })
        
        return vector
    
    def _recalculate(self):
        pos = sum(v.weight for v in self.vectors if v.valence == KarmicValence.POSITIVE)
        neg = sum(v.weight for v in self.vectors if v.valence == KarmicValence.NEGATIVE)
        trans = sum(v.weight for v in self.vectors if v.valence == KarmicValence.TRANSFORMATIVE)
        self.field_potential = pos * PHI - neg * PHI_INVERSE + trans * math.sqrt(PHI)
    
    def evolve(self, dt: float = 0.1):
        self.vectors = [v.evolve(dt) for v in self.vectors if v.magnitude > PLANCK_KARMA]
        self._recalculate()
        if self.field_potential > 0:
            self.dharmic_angle *= (1 - self.learning_rate * dt)
    
    def get_dharmic_alignment(self) -> float:
        return math.cos(self.dharmic_angle)
    
    def get_state(self) -> Dict:
        return {
            'field_potential': self.field_potential,
            'dharmic_alignment': self.get_dharmic_alignment(),
            'vector_count': len(self.vectors),
            'recent_actions': self.history[-20:]
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SELF-AWARE ORB SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════

class CellType(Enum):
    STEM = "stem"
    NEURON = "neuron"
    MEMORY = "memory"
    SENSOR = "sensor"
    EFFECTOR = "effector"
    STRUCTURAL = "structural"
    TRANSPORT = "transport"
    GOAL = "goal"
    HABIT = "habit"
    DREAM = "dream"


class CellState(Enum):
    NASCENT = "nascent"
    GROWING = "growing"
    MATURE = "mature"
    DIVIDING = "dividing"
    ENLIGHTENED = "enlightened"


@dataclass
class SelfAwareOrb:
    """
    A self-aware, self-replicating orb with AI-generated meaning.
    Each orb knows its purpose and can spawn children.
    """
    id: str = field(default_factory=lambda: secrets.token_hex(6))
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = 1.0
    cell_type: CellType = CellType.STEM
    state: CellState = CellState.NASCENT
    energy: float = 1.0
    age: float = 0.0
    generation: int = 0
    karma: float = 0.0
    dharma: float = 1.0
    
    # Self-awareness properties
    meaning: str = ""
    purpose: str = ""
    tags: List[str] = field(default_factory=list)
    index: int = 0
    
    # Connections
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)
    
    # Visual properties
    color: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    glow: float = 0.5
    pulse_phase: float = 0.0
    
    # Linked data (goals, habits, etc)
    linked_data: Dict = field(default_factory=dict)
    
    def update(self, dt: float, environment: Dict, ai: OllamaAI = None) -> Optional['SelfAwareOrb']:
        """Update orb state, may spawn child"""
        self.age += dt
        self.pulse_phase = (self.pulse_phase + dt * PHI) % (2 * math.pi)
        
        # Energy dynamics
        nutrients = environment.get('harmony', 0.5)
        self.energy += nutrients * dt * PHI * 0.1 - dt * 0.02
        self.energy = max(0.1, min(2.0, self.energy))
        
        # State transitions
        if self.state == CellState.NASCENT and self.age > 1.0:
            self.state = CellState.GROWING
        elif self.state == CellState.GROWING and self.age > 5.0:
            self.state = CellState.MATURE
        elif self.state == CellState.MATURE and self.energy > 1.5:
            self.state = CellState.DIVIDING
        elif self.age > 50.0 and self.karma > 0.5:
            self.state = CellState.ENLIGHTENED
        
        # Division
        if self.state == CellState.DIVIDING:
            child = self._spawn_child(ai)
            self.state = CellState.MATURE
            self.energy *= 0.6
            return child
        
        # Update color based on state
        self._update_visuals()
        
        return None
    
    def _spawn_child(self, ai: OllamaAI = None) -> 'SelfAwareOrb':
        """Create child orb with inherited and mutated properties"""
        angle = random.random() * 2 * math.pi
        offset = self.radius * PHI
        
        child_pos = [
            self.position[0] + math.cos(angle) * offset,
            self.position[1] + math.sin(angle) * offset,
            self.position[2] + random.uniform(-1, 1) * offset * 0.3
        ]
        
        # Differentiate
        new_type = self._differentiate()
        
        child = SelfAwareOrb(
            position=child_pos,
            radius=self.radius * PHI_INVERSE,
            cell_type=new_type,
            energy=self.energy * 0.5,
            generation=self.generation + 1,
            karma=self.karma * 0.9 + random.uniform(-0.1, 0.1),
            dharma=self.dharma,
            parent_id=self.id,
            index=self.index + 1,
            tags=self.tags.copy()
        )
        
        # Generate meaning for child
        if ai:
            child.meaning = ai.generate_orb_meaning({
                'type': new_type.value, 'karma': child.karma,
                'energy': child.energy, 'generation': child.generation,
                'state': child.state.value
            })
            child.purpose = f"Born from {self.cell_type.value}, seeking {new_type.value} expression"
        
        self.children_ids.append(child.id)
        
        return child
    
    def _differentiate(self) -> CellType:
        """Determine child type based on karma and randomness"""
        if self.cell_type == CellType.STEM:
            weights = [0.1, 0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.025, 0.025]
            if self.karma > 0.5:
                weights[7] = 0.2  # More likely to be GOAL
            types = [CellType.STEM, CellType.NEURON, CellType.MEMORY, CellType.SENSOR,
                    CellType.EFFECTOR, CellType.STRUCTURAL, CellType.TRANSPORT,
                    CellType.GOAL, CellType.HABIT]
            return random.choices(types, weights=weights)[0]
        return self.cell_type
    
    def _update_visuals(self):
        """Update color and glow based on state"""
        type_colors = {
            CellType.STEM: [0.8, 0.8, 0.8],
            CellType.NEURON: [0.3, 0.7, 1.0],
            CellType.MEMORY: [0.7, 0.3, 0.9],
            CellType.SENSOR: [0.2, 0.9, 0.5],
            CellType.EFFECTOR: [1.0, 0.5, 0.2],
            CellType.STRUCTURAL: [0.6, 0.5, 0.4],
            CellType.TRANSPORT: [0.9, 0.9, 0.3],
            CellType.GOAL: [1.0, 0.84, 0.0],
            CellType.HABIT: [0.5, 1.0, 0.5],
            CellType.DREAM: [0.8, 0.5, 1.0]
        }
        
        base_color = type_colors.get(self.cell_type, [0.5, 0.5, 0.5])
        
        # Modulate by karma
        karma_mod = 0.5 + self.karma * 0.5
        self.color = [min(1.0, c * karma_mod) for c in base_color]
        
        # Glow based on energy and state
        self.glow = 0.3 + self.energy * 0.3
        if self.state == CellState.ENLIGHTENED:
            self.glow = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'position': self.position,
            'velocity': self.velocity,
            'radius': self.radius,
            'type': self.cell_type.value,
            'state': self.state.value,
            'energy': self.energy,
            'age': self.age,
            'generation': self.generation,
            'karma': self.karma,
            'dharma': self.dharma,
            'meaning': self.meaning,
            'purpose': self.purpose,
            'tags': self.tags,
            'index': self.index,
            'color': self.color,
            'glow': self.glow,
            'pulse_phase': self.pulse_phase,
            'children_count': len(self.children_ids),
            'bindings_count': len(self.bindings),
            'linked_data': self.linked_data
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SWARM INTELLIGENCE (Pattern Seeking)
# ═══════════════════════════════════════════════════════════════════════════════════════════

class SwarmCollective:
    """Swarm intelligence for pattern detection and trend analysis"""
    
    def __init__(self, ai: OllamaAI = None):
        self.orbs: Dict[str, SelfAwareOrb] = {}
        self.ai = ai
        self.collective_karma: float = 0.0
        self.collective_dharma: float = 1.0
        self.patterns_detected: List[Dict] = []
        self.trend_data: List[Dict] = []
        self.orb_index: int = 0
    
    def spawn_orb(self, position: List[float] = None, cell_type: CellType = CellType.STEM,
                  karma: float = 0.0, linked_data: Dict = None) -> SelfAwareOrb:
        """Spawn a new self-aware orb"""
        pos = position or [
            random.uniform(-30, 30),
            random.uniform(-30, 30),
            random.uniform(-15, 15)
        ]
        
        self.orb_index += 1
        
        orb = SelfAwareOrb(
            position=pos, cell_type=cell_type,
            karma=karma, index=self.orb_index,
            linked_data=linked_data or {}
        )
        
        # Generate meaning
        if self.ai:
            orb.meaning = self.ai.generate_orb_meaning({
                'type': cell_type.value, 'karma': karma,
                'energy': orb.energy, 'generation': 0,
                'state': orb.state.value
            })
            orb.purpose = f"Manifested as {cell_type.value} orb #{self.orb_index}"
        
        self.orbs[orb.id] = orb
        return orb
    
    def spawn_golden_spiral(self, count: int = 21, center: List[float] = None):
        """Spawn orbs in golden spiral pattern"""
        center = center or [0, 0, 0]
        
        for i in range(count):
            angle = i * GOLDEN_ANGLE_RAD
            radius = math.sqrt(i + 1) * 5
            
            pos = [
                center[0] + math.cos(angle) * radius,
                center[1] + math.sin(angle) * radius,
                center[2] + (i % 5 - 2) * 3
            ]
            
            cell_type = list(CellType)[i % len(CellType)]
            karma = math.sin(i * PHI) * 0.5
            
            self.spawn_orb(pos, cell_type, karma)
    
    def update(self, dt: float, environment: Dict):
        """Update all orbs and detect patterns"""
        new_orbs = []
        
        for orb in list(self.orbs.values()):
            child = orb.update(dt, environment, self.ai)
            if child:
                new_orbs.append(child)
        
        for child in new_orbs:
            self.orbs[child.id] = child
        
        # Update collective metrics
        if self.orbs:
            self.collective_karma = sum(o.karma for o in self.orbs.values()) / len(self.orbs)
            self.collective_dharma = sum(o.dharma for o in self.orbs.values()) / len(self.orbs)
        
        # Record trend data
        self.trend_data.append({
            'timestamp': time.time(),
            'orb_count': len(self.orbs),
            'karma': self.collective_karma,
            'dharma': self.collective_dharma
        })
        
        # Keep last 1000 data points
        if len(self.trend_data) > 1000:
            self.trend_data = self.trend_data[-1000:]
        
        # Detect patterns periodically
        if len(self.trend_data) % 50 == 0:
            self._detect_patterns()
    
    def _detect_patterns(self):
        """Use swarm intelligence to detect patterns"""
        if len(self.trend_data) < 20:
            return
        
        recent = self.trend_data[-20:]
        
        # Karma trend
        karma_values = [d['karma'] for d in recent]
        karma_trend = karma_values[-1] - karma_values[0]
        
        # Growth trend
        orb_counts = [d['orb_count'] for d in recent]
        growth_trend = orb_counts[-1] - orb_counts[0]
        
        patterns = []
        insights = []
        
        if karma_trend > 0.1:
            patterns.append('karma_rising')
            insights.append("Your karmic energy is ascending. The universe responds to your positive actions.")
        elif karma_trend < -0.1:
            patterns.append('karma_falling')
            insights.append("Consider mindful actions to restore karmic balance.")
        
        if growth_trend > 5:
            patterns.append('rapid_growth')
            insights.append("Your life fractal is expanding rapidly. Embrace the growth.")
        
        # Type clustering
        type_counts = defaultdict(int)
        for orb in self.orbs.values():
            type_counts[orb.cell_type.value] += 1
        
        dominant_type = max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else 'stem'
        patterns.append(f'dominant_{dominant_type}')
        
        self.patterns_detected = [{
            'patterns': patterns,
            'insights': insights,
            'karma_trend': karma_trend,
            'growth_trend': growth_trend,
            'dominant_type': dominant_type,
            'type_distribution': dict(type_counts),
            'timestamp': time.time()
        }]
    
    def get_visualization_data(self) -> Dict:
        """Get data for Three.js visualization with zoom-based labels"""
        return {
            'orbs': [o.to_dict() for o in self.orbs.values()],
            'total_orbs': len(self.orbs),
            'collective_karma': self.collective_karma,
            'collective_dharma': self.collective_dharma,
            'patterns': self.patterns_detected,
            'connections': self._get_connections()
        }
    
    def _get_connections(self) -> List[Dict]:
        """Get orb-to-orb connections for visualization"""
        connections = []
        orb_list = list(self.orbs.values())
        
        for i, orb in enumerate(orb_list):
            # Connect to nearby orbs
            for other in orb_list[i+1:]:
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(orb.position, other.position)))
                if dist < 15:  # Connection threshold
                    strength = 1 - dist / 15
                    connections.append({
                        'source': orb.id,
                        'target': other.id,
                        'strength': strength
                    })
        
        return connections[:200]  # Limit for performance


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FEDERATED AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class FederatedAI:
    """
    Federated AI that lives on the server, learns from all users,
    and provides recursive self-improvement.
    """
    
    def __init__(self):
        self.collective_knowledge: Dict[str, Any] = {
            'patterns': [],
            'insights': [],
            'optimal_actions': {},
            'learning_iterations': 0
        }
        self.user_contributions: List[Dict] = []
        self.model_version: str = "1.0.0"
        self.last_training: float = 0
    
    def contribute(self, user_data: Dict):
        """Receive anonymized user data for collective learning"""
        # Strip identifying info
        contribution = {
            'karma_trend': user_data.get('karma_trend', 0),
            'habit_completion_rate': user_data.get('habit_rate', 0),
            'goal_progress': user_data.get('goal_progress', 0),
            'activity_patterns': user_data.get('patterns', []),
            'timestamp': time.time()
        }
        
        self.user_contributions.append(contribution)
        
        # Keep last 10000 contributions
        if len(self.user_contributions) > 10000:
            self.user_contributions = self.user_contributions[-10000:]
        
        # Periodic learning
        if len(self.user_contributions) % 100 == 0:
            self._learn()
    
    def _learn(self):
        """Recursive self-improvement through pattern analysis"""
        if len(self.user_contributions) < 50:
            return
        
        self.collective_knowledge['learning_iterations'] += 1
        
        # Analyze successful patterns
        successful = [c for c in self.user_contributions if c.get('goal_progress', 0) > 0.5]
        
        if successful:
            # Extract common patterns
            pattern_freq = defaultdict(int)
            for contrib in successful:
                for pattern in contrib.get('activity_patterns', []):
                    pattern_freq[pattern] += 1
            
            # Store most common patterns
            sorted_patterns = sorted(pattern_freq.items(), key=lambda x: -x[1])
            self.collective_knowledge['patterns'] = [p[0] for p in sorted_patterns[:20]]
        
        # Generate insights
        avg_karma = sum(c.get('karma_trend', 0) for c in self.user_contributions) / len(self.user_contributions)
        
        if avg_karma > 0:
            self.collective_knowledge['insights'].append({
                'type': 'collective_karma_positive',
                'message': "The collective karma is positive. Community growth is strong.",
                'timestamp': time.time()
            })
        
        self.last_training = time.time()
        
        # Version bump
        major, minor, patch = map(int, self.model_version.split('.'))
        self.model_version = f"{major}.{minor}.{patch + 1}"
    
    def get_recommendations(self, user_state: Dict) -> List[str]:
        """Get personalized recommendations based on collective learning"""
        recommendations = []
        
        # Match user patterns to successful patterns
        user_patterns = set(user_state.get('patterns', []))
        successful_patterns = set(self.collective_knowledge.get('patterns', []))
        
        missing = successful_patterns - user_patterns
        for pattern in list(missing)[:3]:
            recommendations.append(f"Consider exploring: {pattern}")
        
        # Karma-based recommendations
        if user_state.get('karma', 0) < 0:
            recommendations.append("Small positive actions can shift your karmic balance")
        
        return recommendations
    
    def get_state(self) -> Dict:
        return {
            'model_version': self.model_version,
            'learning_iterations': self.collective_knowledge['learning_iterations'],
            'known_patterns': len(self.collective_knowledge['patterns']),
            'contributions': len(self.user_contributions),
            'last_training': self.last_training
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Database:
    """Production SQLite database"""
    
    def __init__(self, db_path: str = "life_fractal_v12.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"✅ Database initialized: {db_path}")
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, first_name TEXT, last_name TEXT,
            created_at TEXT NOT NULL, last_login TEXT, is_active INTEGER DEFAULT 1
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT, category TEXT DEFAULT 'personal', priority INTEGER DEFAULT 3,
            progress REAL DEFAULT 0.0, target_date TEXT, created_at TEXT NOT NULL,
            completed_at TEXT, karma_invested REAL DEFAULT 0.0, orb_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS habits (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
            description TEXT, frequency TEXT DEFAULT 'daily',
            current_streak INTEGER DEFAULT 0, longest_streak INTEGER DEFAULT 0,
            total_completions INTEGER DEFAULT 0, last_completed TEXT,
            created_at TEXT NOT NULL, orb_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS daily_entries (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, date TEXT NOT NULL,
            mood_level INTEGER DEFAULT 50, energy_level INTEGER DEFAULT 50,
            focus_level INTEGER DEFAULT 50, stress_level INTEGER DEFAULT 50,
            spoons_available INTEGER DEFAULT 12, spoons_used INTEGER DEFAULT 0,
            journal_entry TEXT, created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id), UNIQUE(user_id, date)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS pet_state (
            user_id TEXT PRIMARY KEY, species TEXT DEFAULT 'cat',
            name TEXT DEFAULT 'Karma', hunger REAL DEFAULT 50.0,
            energy REAL DEFAULT 50.0, happiness REAL DEFAULT 50.0,
            level INTEGER DEFAULT 1, experience INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS karma_history (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, action_type TEXT NOT NULL,
            karma_value REAL NOT NULL, meaning TEXT, timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS orb_state (
            user_id TEXT PRIMARY KEY, orb_data TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS federated_data (
            id TEXT PRIMARY KEY, data TEXT NOT NULL, timestamp TEXT NOT NULL
        )''')
        
        conn.commit()
        conn.close()
    
    def execute(self, query: str, params: tuple = ()) -> Optional[Any]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Database error: {e}")
            return None
    
    def execute_one(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Database error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════════════════
# LIVING ORGANISM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════════════

class LivingOrganism:
    """Central orchestrator for the living mathematical organism"""
    
    def __init__(self):
        self.ai = OllamaAI()
        self.karma_engine = KarmaDharmaEngine(self.ai)
        self.swarm = SwarmCollective(self.ai)
        self.mayan = MayanCalendar()
        self.federated = FederatedAI()
        
        self.creation_time = time.time()
        self.uptime = 0.0
        self.harmony = 1.0
        
        # Initialize with golden spiral
        self.swarm.spawn_golden_spiral(FIBONACCI[7])  # 13 orbs
        
        logger.info("🌀 Living organism awakened")
    
    def update(self, dt: float = 0.1):
        """Main update loop"""
        self.uptime = time.time() - self.creation_time
        
        environment = {
            'harmony': self.harmony,
            'karma': self.karma_engine.field_potential
        }
        
        self.karma_engine.evolve(dt)
        self.swarm.update(dt, environment)
        
        # Update harmony
        self.harmony = (
            self.karma_engine.get_dharmic_alignment() * PHI +
            self.swarm.collective_dharma * PHI_INVERSE +
            0.5
        ) / (PHI + PHI_INVERSE + 0.5)
    
    def process_action(self, action_type: str, magnitude: float = 1.0,
                      intention: float = 0.8, awareness: float = 0.7,
                      linked_data: Dict = None) -> Dict:
        """Process user action through organism"""
        vector = self.karma_engine.add_action(action_type, magnitude, intention, awareness)
        
        # Spawn celebration orbs for positive actions
        if vector.valence == KarmicValence.POSITIVE:
            orb = self.swarm.spawn_orb(
                cell_type=CellType.GOAL if 'goal' in action_type.lower() else CellType.HABIT,
                karma=vector.weight * 0.1,
                linked_data=linked_data
            )
            orb.meaning = vector.meaning
        
        # Contribute to federated learning
        self.federated.contribute({
            'karma_trend': self.karma_engine.field_potential,
            'patterns': [p['patterns'] for p in self.swarm.patterns_detected] if self.swarm.patterns_detected else []
        })
        
        return {
            'karma_earned': vector.weight,
            'meaning': vector.meaning,
            'harmony': self.harmony,
            'orb_count': len(self.swarm.orbs)
        }
    
    def get_state(self) -> Dict:
        mayan = self.mayan.get_today_summary()
        
        return {
            'uptime': self.uptime,
            'harmony': self.harmony,
            'karma': self.karma_engine.get_state(),
            'swarm': self.swarm.get_visualization_data(),
            'mayan': mayan,
            'ai': {
                'ollama_available': self.ai.available,
                'federated': self.federated.get_state()
            },
            'sacred_constants': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE,
                'dharma_frequency': DHARMA_FREQUENCY,
                'fibonacci': FIBONACCI[:13]
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

db = Database()
organism = LivingOrganism()

# Background update thread
def background_loop():
    while True:
        try:
            organism.update(0.1)
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Background error: {e}")
            time.sleep(1)

threading.Thread(target=background_loop, daemon=True).start()

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user_id = secrets.token_hex(16)
    try:
        db.execute('''INSERT INTO users (id, email, password_hash, first_name, last_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (user_id, data['email'], generate_password_hash(data['password']),
             data.get('first_name', ''), data.get('last_name', ''),
             datetime.now(timezone.utc).isoformat()))
        
        db.execute('INSERT INTO pet_state (user_id) VALUES (?)', (user_id,))
        session['user_id'] = user_id
        return jsonify({'success': True, 'user_id': user_id})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 409


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = db.execute_one('SELECT * FROM users WHERE email = ?', (data.get('email', ''),))
    
    if not user or not check_password_hash(user['password_hash'], data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user_id'] = user['id']
    db.execute('UPDATE users SET last_login = ? WHERE id = ?',
              (datetime.now(timezone.utc).isoformat(), user['id']))
    return jsonify({'success': True})


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})


@app.route('/api/organism/state', methods=['GET'])
@require_auth
def get_organism_state():
    return jsonify(organism.get_state())


@app.route('/api/organism/visualization', methods=['GET'])
@require_auth
def get_visualization():
    return jsonify(organism.swarm.get_visualization_data())


@app.route('/api/organism/action', methods=['POST'])
@require_auth
def process_action():
    data = request.get_json()
    result = organism.process_action(
        data.get('action_type', 'neutral'),
        data.get('magnitude', 1.0),
        data.get('intention', 0.8),
        data.get('awareness', 0.7),
        data.get('linked_data')
    )
    
    # Store karma history
    db.execute('''INSERT INTO karma_history (id, user_id, action_type, karma_value, meaning, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (secrets.token_hex(8), session['user_id'], data.get('action_type', 'neutral'),
         result['karma_earned'], result.get('meaning', ''),
         datetime.now(timezone.utc).isoformat()))
    
    return jsonify(result)


@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals():
    goals = db.execute('SELECT * FROM goals WHERE user_id = ? ORDER BY created_at DESC',
                      (session['user_id'],))
    return jsonify([dict(g) for g in goals or []])


@app.route('/api/goals', methods=['POST'])
@require_auth
def create_goal():
    data = request.get_json()
    goal_id = secrets.token_hex(16)
    
    # Create linked orb
    orb = organism.swarm.spawn_orb(
        cell_type=CellType.GOAL,
        karma=0.5,
        linked_data={'type': 'goal', 'id': goal_id, 'title': data.get('title', '')}
    )
    
    db.execute('''INSERT INTO goals (id, user_id, title, description, category, priority, target_date, created_at, orb_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (goal_id, session['user_id'], data.get('title', 'New Goal'),
         data.get('description', ''), data.get('category', 'personal'),
         data.get('priority', 3), data.get('target_date'),
         datetime.now(timezone.utc).isoformat(), orb.id))
    
    result = organism.process_action('create_goal', 0.5, 0.8, 0.7, {'goal_id': goal_id})
    
    return jsonify({
        'id': goal_id, 'orb_id': orb.id, 'karma_earned': result['karma_earned'],
        'orb_meaning': orb.meaning
    })


@app.route('/api/goals/<goal_id>/progress', methods=['POST'])
@require_auth
def update_progress(goal_id):
    data = request.get_json()
    progress = min(100, max(0, data.get('progress', 0)))
    
    db.execute('UPDATE goals SET progress = ? WHERE id = ? AND user_id = ?',
              (progress, goal_id, session['user_id']))
    
    action = 'complete_goal' if progress >= 100 else 'progress_goal'
    result = organism.process_action(action, progress / 100, 0.9, 0.8)
    
    return jsonify({'success': True, 'progress': progress, **result})


@app.route('/api/habits', methods=['GET'])
@require_auth
def get_habits():
    habits = db.execute('SELECT * FROM habits WHERE user_id = ? ORDER BY created_at DESC',
                       (session['user_id'],))
    return jsonify([dict(h) for h in habits or []])


@app.route('/api/habits', methods=['POST'])
@require_auth
def create_habit():
    data = request.get_json()
    habit_id = secrets.token_hex(16)
    
    orb = organism.swarm.spawn_orb(
        cell_type=CellType.HABIT,
        karma=0.3,
        linked_data={'type': 'habit', 'id': habit_id, 'name': data.get('name', '')}
    )
    
    db.execute('''INSERT INTO habits (id, user_id, name, description, frequency, created_at, orb_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (habit_id, session['user_id'], data.get('name', 'New Habit'),
         data.get('description', ''), data.get('frequency', 'daily'),
         datetime.now(timezone.utc).isoformat(), orb.id))
    
    return jsonify({'id': habit_id, 'orb_id': orb.id, 'orb_meaning': orb.meaning})


@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(habit_id):
    habit = db.execute_one('SELECT * FROM habits WHERE id = ? AND user_id = ?',
                          (habit_id, session['user_id']))
    if not habit:
        return jsonify({'error': 'Habit not found'}), 404
    
    new_streak = habit['current_streak'] + 1
    longest = max(habit['longest_streak'], new_streak)
    total = habit['total_completions'] + 1
    
    db.execute('''UPDATE habits SET current_streak = ?, longest_streak = ?,
        total_completions = ?, last_completed = ? WHERE id = ?''',
        (new_streak, longest, total, datetime.now(timezone.utc).isoformat(), habit_id))
    
    # Fibonacci bonus
    fib_bonus = PHI if new_streak in FIBONACCI else 1.0
    result = organism.process_action('complete_habit', 0.3 * fib_bonus, 0.9, 0.85)
    
    return jsonify({
        'success': True, 'streak': new_streak, 'total': total,
        'fibonacci_bonus': new_streak in FIBONACCI, **result
    })


@app.route('/api/wellness/checkin', methods=['POST'])
@require_auth
def wellness_checkin():
    data = request.get_json()
    entry_id = secrets.token_hex(16)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    db.execute('''INSERT OR REPLACE INTO daily_entries
        (id, user_id, date, mood_level, energy_level, focus_level, stress_level,
         spoons_available, spoons_used, journal_entry, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (entry_id, session['user_id'], today,
         data.get('mood', 50), data.get('energy', 50),
         data.get('focus', 50), data.get('stress', 50),
         data.get('spoons_available', 12), data.get('spoons_used', 0),
         data.get('journal', ''), datetime.now(timezone.utc).isoformat()))
    
    result = organism.process_action('wellness_checkin', 0.4, 0.95, 0.9)
    
    return jsonify({'success': True, **result})


@app.route('/api/wellness/today', methods=['GET'])
@require_auth
def get_today_wellness():
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = db.execute_one('SELECT * FROM daily_entries WHERE user_id = ? AND date = ?',
                          (session['user_id'], today))
    return jsonify(dict(entry) if entry else {
        'mood_level': 50, 'energy_level': 50, 'focus_level': 50,
        'stress_level': 50, 'spoons_available': 12, 'spoons_used': 0
    })


@app.route('/api/mayan/today', methods=['GET'])
@require_auth
def get_mayan_today():
    return jsonify(organism.mayan.get_today_summary())


@app.route('/api/pet/state', methods=['GET'])
@require_auth
def get_pet():
    pet = db.execute_one('SELECT * FROM pet_state WHERE user_id = ?', (session['user_id'],))
    return jsonify(dict(pet) if pet else {'species': 'cat', 'name': 'Karma', 'hunger': 50, 'energy': 50, 'happiness': 50})


@app.route('/api/pet/interact', methods=['POST'])
@require_auth
def interact_pet():
    data = request.get_json()
    action = data.get('action', 'pet')
    
    pet = db.execute_one('SELECT * FROM pet_state WHERE user_id = ?', (session['user_id'],))
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    
    hunger, energy, happiness = pet['hunger'], pet['energy'], pet['happiness']
    
    if action == 'feed':
        hunger = min(100, hunger + 30)
        happiness = min(100, happiness + 10)
    elif action == 'play':
        energy = max(0, energy - 20)
        happiness = min(100, happiness + 25)
    elif action == 'rest':
        energy = min(100, energy + 40)
    elif action == 'pet':
        happiness = min(100, happiness + 15)
    
    db.execute('UPDATE pet_state SET hunger = ?, energy = ?, happiness = ? WHERE user_id = ?',
              (hunger, energy, happiness, session['user_id']))
    
    result = organism.process_action(f'pet_{action}', 0.2, 0.8, 0.9)
    
    return jsonify({
        'hunger': hunger, 'energy': energy, 'happiness': happiness,
        'emotion': 'happy' if happiness > 70 else 'content' if happiness > 40 else 'lonely',
        **result
    })


@app.route('/api/analytics/patterns', methods=['GET'])
@require_auth
def get_patterns():
    patterns = organism.swarm.patterns_detected
    recommendations = organism.federated.get_recommendations({
        'karma': organism.karma_engine.field_potential,
        'patterns': [p.get('patterns', []) for p in patterns]
    })
    
    return jsonify({
        'patterns': patterns,
        'recommendations': recommendations,
        'federated_state': organism.federated.get_state()
    })


@app.route('/api/analytics/karma-history', methods=['GET'])
@require_auth
def get_karma_history():
    history = db.execute(
        'SELECT * FROM karma_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50',
        (session['user_id'],))
    return jsonify([dict(h) for h in history or []])


@app.route('/api/fractal/2d', methods=['GET'])
@require_auth
def get_2d_fractal():
    """Generate 2D Mandelbrot fractal"""
    wellness = db.execute_one(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
        (session['user_id'],))
    
    wellness_data = dict(wellness) if wellness else {}
    energy = wellness_data.get('energy_level', 50) / 100
    focus = wellness_data.get('focus_level', 50) / 100
    mood = wellness_data.get('mood_level', 50) / 100
    
    # Generate fractal
    width, height = 400, 400
    img = Image.new('RGB', (width, height), (20, 20, 30))
    pixels = img.load()
    
    x_center = -0.5 + (energy - 0.5) * 0.3
    y_center = (mood - 0.5) * 0.3
    zoom = 1.5 + focus
    max_iter = int(50 + focus * 100)
    
    x_min, x_max = x_center - 2/zoom, x_center + 2/zoom
    y_min, y_max = y_center - 2/zoom, y_center + 2/zoom
    
    for px in range(width):
        for py in range(height):
            x0 = x_min + (x_max - x_min) * px / width
            y0 = y_min + (y_max - y_min) * py / height
            x, y = 0.0, 0.0
            iteration = 0
            
            while x*x + y*y <= 4 and iteration < max_iter:
                x_new = x*x - y*y + x0
                y = 2*x*y + y0
                x = x_new
                iteration += 1
            
            if iteration < max_iter:
                smooth = iteration + 1 - math.log(math.log(max(1, x*x + y*y))) / math.log(2)
                hue = (smooth * PHI * 10) % 360
                sat = 0.7
                val = min(1, smooth / max_iter * 2)
                
                # HSV to RGB
                c = val * sat
                x_c = c * (1 - abs((hue / 60) % 2 - 1))
                m = val - c
                
                if hue < 60: r, g, b = c, x_c, 0
                elif hue < 120: r, g, b = x_c, c, 0
                elif hue < 180: r, g, b = 0, c, x_c
                elif hue < 240: r, g, b = 0, x_c, c
                elif hue < 300: r, g, b = x_c, 0, c
                else: r, g, b = c, 0, x_c
                
                pixels[px, py] = (int((r+m)*255), int((g+m)*255), int((b+m)*255))
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return jsonify({
        'image': base64.b64encode(buffer.getvalue()).decode('utf-8'),
        'format': 'png'
    })


@app.route('/api/fractal/3d-params', methods=['GET'])
@require_auth
def get_3d_params():
    """Get parameters for client-side WebGL 3D rendering"""
    wellness = db.execute_one(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
        (session['user_id'],))
    
    w = dict(wellness) if wellness else {}
    
    return jsonify({
        'type': 'mandelbulb',
        'power': 8.0 + w.get('energy_level', 50) / 100 * 4,
        'iterations': int(8 + w.get('focus_level', 50) / 100 * 8),
        'zoom': 1.5 + w.get('focus_level', 50) / 100 * 0.5,
        'rotation': [
            w.get('mood_level', 50) / 100 * math.pi * 2,
            w.get('energy_level', 50) / 100 * math.pi,
            organism.harmony * math.pi * 0.5
        ],
        'color_scheme': {
            'base': [0.27, 0.51, 0.71],
            'accent': [1.0, 0.72, 0.3],
            'glow': [0.5, 0.8, 1.0]
        },
        'karma': organism.karma_engine.field_potential,
        'harmony': organism.harmony
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '12.0',
        'harmony': organism.harmony,
        'orbs': len(organism.swarm.orbs),
        'ollama': organism.ai.available,
        'ml': HAS_SKLEARN,
        'mayan': organism.mayan.get_today_summary()['greeting'],
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN HTML - FULL IMMERSIVE GUI
# ═══════════════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🌀 Life Fractal Intelligence</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --phi: 1.618033988749895;
            --bg-dark: #0a0a12;
            --bg-panel: rgba(15, 15, 25, 0.95);
            --accent-gold: #d4af37;
            --accent-blue: #4a90a4;
            --accent-purple: #8b5cf6;
            --text-primary: #e8e8e8;
            --text-muted: #888;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }
        
        /* Full Screen 3D Canvas */
        #fractal-universe {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1;
        }
        
        /* Hamburger Menu */
        .hamburger {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            width: 50px;
            height: 50px;
            background: var(--bg-panel);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 5px;
            transition: all 0.3s;
        }
        
        .hamburger:hover { border-color: var(--accent-gold); }
        .hamburger span {
            width: 24px;
            height: 2px;
            background: var(--accent-gold);
            transition: all 0.3s;
        }
        .hamburger.open span:nth-child(1) { transform: rotate(45deg) translate(5px, 5px); }
        .hamburger.open span:nth-child(2) { opacity: 0; }
        .hamburger.open span:nth-child(3) { transform: rotate(-45deg) translate(5px, -5px); }
        
        /* Side Navigation */
        .side-nav {
            position: fixed;
            top: 0;
            left: -320px;
            width: 300px;
            height: 100vh;
            background: var(--bg-panel);
            border-right: 1px solid rgba(212, 175, 55, 0.2);
            z-index: 999;
            transition: left 0.3s ease;
            padding: 80px 20px 20px;
            overflow-y: auto;
        }
        
        .side-nav.open { left: 0; }
        
        .nav-section { margin-bottom: 25px; }
        .nav-section h3 {
            color: var(--accent-gold);
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
            padding-left: 10px;
        }
        
        .nav-btn {
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            padding: 14px 16px;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.95em;
            cursor: pointer;
            border-radius: 10px;
            transition: all 0.2s;
            margin-bottom: 4px;
        }
        
        .nav-btn:hover { background: rgba(74, 144, 164, 0.15); }
        .nav-btn.active { 
            background: rgba(212, 175, 55, 0.15);
            border-left: 3px solid var(--accent-gold);
        }
        
        /* Top Bar */
        .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(180deg, rgba(10,10,18,0.95) 0%, rgba(10,10,18,0) 100%);
            z-index: 100;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0 80px;
        }
        
        .logo {
            font-size: 1.5em;
            color: var(--accent-gold);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Stats Bar */
        .stats-bar {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100;
            display: flex;
            gap: 15px;
        }
        
        .stat-pill {
            background: var(--bg-panel);
            border: 1px solid rgba(74, 144, 164, 0.3);
            border-radius: 20px;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }
        
        .stat-pill .value {
            color: var(--accent-gold);
            font-weight: 600;
        }
        
        /* Content Panels */
        .panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 40px);
            max-width: 800px;
            max-height: 60vh;
            background: var(--bg-panel);
            border: 1px solid rgba(74, 144, 164, 0.2);
            border-radius: 20px;
            z-index: 100;
            overflow: hidden;
            display: none;
        }
        
        .panel.active { display: block; }
        
        .panel-header {
            padding: 20px;
            border-bottom: 1px solid rgba(74, 144, 164, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-header h2 {
            color: var(--accent-gold);
            font-size: 1.2em;
        }
        
        .panel-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 1.5em;
            cursor: pointer;
        }
        
        .panel-content {
            padding: 20px;
            max-height: calc(60vh - 70px);
            overflow-y: auto;
        }
        
        /* Enter Fractal Button */
        .enter-fractal-btn {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            padding: 16px 40px;
            background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-blue) 100%);
            border: none;
            border-radius: 30px;
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 30px rgba(139, 92, 246, 0.4);
            transition: all 0.3s;
        }
        
        .enter-fractal-btn:hover {
            transform: translateX(-50%) translateY(-3px);
            box-shadow: 0 8px 40px rgba(139, 92, 246, 0.6);
        }
        
        .enter-fractal-btn.hidden { display: none; }
        
        /* Orb Tooltip */
        .orb-tooltip {
            position: fixed;
            background: var(--bg-panel);
            border: 1px solid var(--accent-gold);
            border-radius: 12px;
            padding: 15px;
            max-width: 300px;
            z-index: 1001;
            display: none;
            pointer-events: none;
        }
        
        .orb-tooltip.visible { display: block; }
        
        .orb-tooltip h4 {
            color: var(--accent-gold);
            margin-bottom: 8px;
        }
        
        .orb-tooltip p {
            color: var(--text-muted);
            font-size: 0.9em;
            line-height: 1.5;
        }
        
        .orb-tooltip .tags {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        
        .orb-tooltip .tag {
            background: rgba(74, 144, 164, 0.2);
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75em;
        }
        
        /* Mayan Calendar Widget */
        .mayan-widget {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--bg-panel);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 15px;
            padding: 15px;
            z-index: 100;
            min-width: 200px;
        }
        
        .mayan-widget h4 {
            color: var(--accent-gold);
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        
        .mayan-widget .kin {
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .mayan-widget .energy {
            color: var(--text-muted);
            font-size: 0.85em;
        }
        
        /* Forms */
        .form-group { margin-bottom: 15px; }
        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: var(--text-muted);
            font-size: 0.9em;
        }
        
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 12px;
            background: rgba(74, 144, 164, 0.1);
            border: 1px solid rgba(74, 144, 164, 0.3);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 1em;
        }
        
        .btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, var(--accent-blue) 0%, #357a8a 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn:hover { transform: translateY(-2px); }
        .btn-gold { background: linear-gradient(135deg, var(--accent-gold) 0%, #c49b30 100%); }
        
        /* Item Cards */
        .item-card {
            background: rgba(74, 144, 164, 0.1);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .item-card h3 { margin-bottom: 5px; }
        .item-card .meta { color: var(--text-muted); font-size: 0.85em; }
        
        /* Slider */
        .slider-group { margin-bottom: 20px; }
        .slider-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(74, 144, 164, 0.3);
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-gold);
            cursor: pointer;
        }
        
        /* Toast */
        .toast {
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: var(--accent-gold);
            color: var(--bg-dark);
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 500;
            z-index: 2000;
            opacity: 0;
            transition: all 0.3s;
        }
        
        .toast.show {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .stats-bar { display: none; }
            .mayan-widget { bottom: 100px; right: 10px; }
        }
    </style>
</head>
<body>
    <!-- Full Screen 3D Universe -->
    <canvas id="fractal-universe"></canvas>
    
    <!-- Hamburger Menu -->
    <button class="hamburger" onclick="toggleNav()">
        <span></span><span></span><span></span>
    </button>
    
    <!-- Side Navigation -->
    <nav class="side-nav" id="sideNav">
        <div class="nav-section">
            <h3>Planning</h3>
            <button class="nav-btn active" onclick="showPanel('dashboard')">📊 Dashboard</button>
            <button class="nav-btn" onclick="showPanel('goals')">🎯 Goals</button>
            <button class="nav-btn" onclick="showPanel('habits')">✨ Habits</button>
        </div>
        <div class="nav-section">
            <h3>Wellness</h3>
            <button class="nav-btn" onclick="showPanel('checkin')">💫 Check-in</button>
            <button class="nav-btn" onclick="showPanel('spoons')">🥄 Spoons</button>
        </div>
        <div class="nav-section">
            <h3>Companions</h3>
            <button class="nav-btn" onclick="showPanel('pet')">🐱 Pet</button>
        </div>
        <div class="nav-section">
            <h3>Insights</h3>
            <button class="nav-btn" onclick="showPanel('patterns')">🧠 ML Patterns</button>
            <button class="nav-btn" onclick="showPanel('karma')">⚖️ Karma</button>
        </div>
    </nav>
    
    <!-- Top Bar -->
    <div class="top-bar">
        <div class="logo">🌀 Life Fractal Intelligence</div>
    </div>
    
    <!-- Stats Bar -->
    <div class="stats-bar">
        <div class="stat-pill">
            <span>⚖️</span>
            <span class="value" id="karma-display">0.00</span>
        </div>
        <div class="stat-pill">
            <span>🔮</span>
            <span class="value" id="harmony-display">1.00</span>
        </div>
        <div class="stat-pill">
            <span>🧬</span>
            <span class="value" id="orb-display">0</span>
        </div>
    </div>
    
    <!-- Enter Fractal Button -->
    <button class="enter-fractal-btn" id="enterFractalBtn" onclick="enterFractal()">
        🌀 Enter the Fractal
    </button>
    
    <!-- Orb Tooltip -->
    <div class="orb-tooltip" id="orbTooltip">
        <h4 id="tooltipTitle">Orb</h4>
        <p id="tooltipMeaning">Meaning...</p>
        <div class="tags" id="tooltipTags"></div>
    </div>
    
    <!-- Mayan Calendar Widget -->
    <div class="mayan-widget" id="mayanWidget">
        <h4>📅 Mayan Calendar</h4>
        <div class="kin" id="mayanKin">Loading...</div>
        <div class="energy" id="mayanEnergy"></div>
    </div>
    
    <!-- Dashboard Panel -->
    <div class="panel" id="dashboard-panel">
        <div class="panel-header">
            <h2>📊 Dashboard</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div id="dashboard-content"></div>
        </div>
    </div>
    
    <!-- Goals Panel -->
    <div class="panel" id="goals-panel">
        <div class="panel-header">
            <h2>🎯 Goals</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div class="form-group">
                <input type="text" id="goalTitle" placeholder="What's your goal?">
            </div>
            <button class="btn btn-gold" onclick="createGoal()">Create Goal</button>
            <div id="goals-list" style="margin-top:20px;"></div>
        </div>
    </div>
    
    <!-- Habits Panel -->
    <div class="panel" id="habits-panel">
        <div class="panel-header">
            <h2>✨ Habits</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div class="form-group">
                <input type="text" id="habitName" placeholder="New habit...">
            </div>
            <button class="btn btn-gold" onclick="createHabit()">Add Habit</button>
            <div id="habits-list" style="margin-top:20px;"></div>
        </div>
    </div>
    
    <!-- Check-in Panel -->
    <div class="panel" id="checkin-panel">
        <div class="panel-header">
            <h2>💫 Daily Check-in</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div class="slider-group">
                <div class="slider-header"><span>😴 Energy</span><span id="energyVal">50</span></div>
                <input type="range" id="energySlider" min="0" max="100" value="50" 
                    oninput="document.getElementById('energyVal').textContent=this.value">
            </div>
            <div class="slider-group">
                <div class="slider-header"><span>😊 Mood</span><span id="moodVal">50</span></div>
                <input type="range" id="moodSlider" min="0" max="100" value="50"
                    oninput="document.getElementById('moodVal').textContent=this.value">
            </div>
            <div class="slider-group">
                <div class="slider-header"><span>🎯 Focus</span><span id="focusVal">50</span></div>
                <input type="range" id="focusSlider" min="0" max="100" value="50"
                    oninput="document.getElementById('focusVal').textContent=this.value">
            </div>
            <button class="btn btn-gold" onclick="submitCheckin()">Submit</button>
        </div>
    </div>
    
    <!-- Pet Panel -->
    <div class="panel" id="pet-panel">
        <div class="panel-header">
            <h2>🐱 Virtual Pet</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content" style="text-align:center;">
            <div style="font-size:4em;margin:20px 0;" id="petEmoji">🐱</div>
            <h3 id="petName">Karma</h3>
            <div style="display:flex;justify-content:space-around;margin:20px 0;">
                <div><div class="value" id="petHunger">50</div><div class="meta">Hunger</div></div>
                <div><div class="value" id="petEnergy">50</div><div class="meta">Energy</div></div>
                <div><div class="value" id="petHappy">50</div><div class="meta">Happy</div></div>
            </div>
            <div style="display:flex;gap:10px;justify-content:center;">
                <button class="btn" onclick="petAction('feed')">🍖 Feed</button>
                <button class="btn" onclick="petAction('play')">🎾 Play</button>
                <button class="btn" onclick="petAction('pet')">🤗 Pet</button>
            </div>
        </div>
    </div>
    
    <!-- Patterns Panel -->
    <div class="panel" id="patterns-panel">
        <div class="panel-header">
            <h2>🧠 ML Insights</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div id="patterns-content"></div>
        </div>
    </div>
    
    <!-- Karma Panel -->
    <div class="panel" id="karma-panel">
        <div class="panel-header">
            <h2>⚖️ Karma History</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div id="karma-history"></div>
        </div>
    </div>
    
    <!-- Spoons Panel -->
    <div class="panel" id="spoons-panel">
        <div class="panel-header">
            <h2>🥄 Spoon Theory</h2>
            <button class="panel-close" onclick="closePanel()">×</button>
        </div>
        <div class="panel-content">
            <div id="spoons-content"></div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <script>
    // Sacred Constants
    const PHI = 1.618033988749895;
    const GOLDEN_ANGLE = 137.5077640500378;
    
    // State
    let scene, camera, renderer, controls;
    let orbs = [];
    let orbMeshes = {};
    let isInsideFractal = false;
    let raycaster, mouse;
    let hoveredOrb = null;
    let organismData = {};
    
    // Initialize Three.js
    function initThreeJS() {
        const canvas = document.getElementById('fractal-universe');
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a12);
        scene.fog = new THREE.FogExp2(0x0a0a12, 0.008);
        
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        camera.position.set(0, 0, 100);
        
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Lights
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambient);
        
        const point = new THREE.PointLight(0xd4af37, 1, 500);
        point.position.set(50, 50, 50);
        scene.add(point);
        
        const point2 = new THREE.PointLight(0x4a90a4, 0.8, 400);
        point2.position.set(-50, -30, 30);
        scene.add(point2);
        
        // Stars background
        const starsGeometry = new THREE.BufferGeometry();
        const starPositions = [];
        for (let i = 0; i < 5000; i++) {
            starPositions.push(
                (Math.random() - 0.5) * 2000,
                (Math.random() - 0.5) * 2000,
                (Math.random() - 0.5) * 2000
            );
        }
        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
        const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 });
        const stars = new THREE.Points(starsGeometry, starsMaterial);
        scene.add(stars);
        
        // Raycaster for orb interaction
        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();
        
        // Event listeners
        window.addEventListener('resize', onResize);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('wheel', onWheel);
        canvas.addEventListener('click', onCanvasClick);
        
        // Touch controls
        let touchStart = { x: 0, y: 0 };
        canvas.addEventListener('touchstart', (e) => {
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        });
        canvas.addEventListener('touchmove', (e) => {
            const dx = e.touches[0].clientX - touchStart.x;
            const dy = e.touches[0].clientY - touchStart.y;
            scene.rotation.y += dx * 0.005;
            scene.rotation.x += dy * 0.005;
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        });
        
        animate();
    }
    
    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    function onMouseMove(e) {
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
        
        // Rotate scene slightly with mouse
        if (!isInsideFractal) {
            scene.rotation.y = mouse.x * 0.2;
            scene.rotation.x = mouse.y * 0.1;
        }
        
        // Check orb hover
        checkOrbHover(e);
    }
    
    function onWheel(e) {
        const zoomSpeed = 0.1;
        camera.position.z += e.deltaY * zoomSpeed;
        camera.position.z = Math.max(20, Math.min(200, camera.position.z));
        
        // Update label visibility based on zoom
        updateLabelVisibility();
    }
    
    function onCanvasClick(e) {
        if (hoveredOrb) {
            // Focus on clicked orb
            const orb = hoveredOrb;
            camera.position.set(
                orb.position.x,
                orb.position.y,
                orb.position.z + 15
            );
        }
    }
    
    function checkOrbHover(e) {
        raycaster.setFromCamera(mouse, camera);
        const meshes = Object.values(orbMeshes);
        const intersects = raycaster.intersectObjects(meshes);
        
        const tooltip = document.getElementById('orbTooltip');
        
        if (intersects.length > 0) {
            const mesh = intersects[0].object;
            const orbData = mesh.userData;
            hoveredOrb = mesh;
            
            // Only show tooltip when zoomed in
            if (camera.position.z < 80) {
                tooltip.style.left = e.clientX + 15 + 'px';
                tooltip.style.top = e.clientY + 15 + 'px';
                
                document.getElementById('tooltipTitle').textContent = 
                    `${orbData.type?.toUpperCase() || 'ORB'} #${orbData.index || 0}`;
                document.getElementById('tooltipMeaning').textContent = 
                    orbData.meaning || 'A living cell in your fractal universe...';
                
                const tagsEl = document.getElementById('tooltipTags');
                tagsEl.innerHTML = '';
                if (orbData.tags) {
                    orbData.tags.forEach(tag => {
                        const span = document.createElement('span');
                        span.className = 'tag';
                        span.textContent = tag;
                        tagsEl.appendChild(span);
                    });
                }
                
                tooltip.classList.add('visible');
            }
        } else {
            hoveredOrb = null;
            tooltip.classList.remove('visible');
        }
    }
    
    function updateLabelVisibility() {
        // Labels visible when camera.position.z < 50
        const showLabels = camera.position.z < 50;
        // Could update 3D text sprites here if implemented
    }
    
    function createOrbMesh(orbData) {
        const geometry = new THREE.SphereGeometry(orbData.radius || 1, 32, 32);
        
        const color = new THREE.Color(
            orbData.color?.[0] || 0.3,
            orbData.color?.[1] || 0.6,
            orbData.color?.[2] || 0.9
        );
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: orbData.glow || 0.3,
            shininess: 100,
            transparent: true,
            opacity: 0.9
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(
            orbData.position?.[0] || 0,
            orbData.position?.[1] || 0,
            orbData.position?.[2] || 0
        );
        mesh.userData = orbData;
        
        return mesh;
    }
    
    function updateOrbs(orbsData) {
        // Remove old orbs
        Object.keys(orbMeshes).forEach(id => {
            if (!orbsData.find(o => o.id === id)) {
                scene.remove(orbMeshes[id]);
                delete orbMeshes[id];
            }
        });
        
        // Add/update orbs
        orbsData.forEach(orbData => {
            if (orbMeshes[orbData.id]) {
                // Update existing
                const mesh = orbMeshes[orbData.id];
                mesh.position.set(...orbData.position);
                mesh.userData = orbData;
                
                // Pulse animation
                const pulse = 1 + Math.sin(orbData.pulse_phase || 0) * 0.1;
                mesh.scale.setScalar(pulse);
            } else {
                // Create new
                const mesh = createOrbMesh(orbData);
                scene.add(mesh);
                orbMeshes[orbData.id] = mesh;
            }
        });
        
        // Add connections
        // (Could add line geometry between connected orbs)
    }
    
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate orbs slightly
        Object.values(orbMeshes).forEach((mesh, i) => {
            mesh.rotation.y += 0.002;
            mesh.rotation.x += 0.001;
        });
        
        // Slow scene rotation when not inside fractal
        if (!isInsideFractal) {
            scene.rotation.y += 0.0005;
        }
        
        renderer.render(scene, camera);
    }
    
    function enterFractal() {
        isInsideFractal = true;
        document.getElementById('enterFractalBtn').classList.add('hidden');
        
        // Zoom in
        const tween = setInterval(() => {
            camera.position.z -= 2;
            if (camera.position.z <= 30) {
                clearInterval(tween);
            }
        }, 16);
        
        showToast('🌀 Entering the fractal universe...');
    }
    
    // Navigation
    function toggleNav() {
        document.querySelector('.hamburger').classList.toggle('open');
        document.getElementById('sideNav').classList.toggle('open');
    }
    
    function showPanel(name) {
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        
        const panel = document.getElementById(name + '-panel');
        if (panel) {
            panel.classList.add('active');
            loadPanelContent(name);
        }
        
        event.target.classList.add('active');
        toggleNav();
    }
    
    function closePanel() {
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    }
    
    async function loadPanelContent(name) {
        if (name === 'goals') await loadGoals();
        else if (name === 'habits') await loadHabits();
        else if (name === 'pet') await loadPet();
        else if (name === 'patterns') await loadPatterns();
        else if (name === 'karma') await loadKarmaHistory();
        else if (name === 'dashboard') await loadDashboard();
    }
    
    // API helpers
    async function api(endpoint, options = {}) {
        try {
            const res = await fetch(endpoint, {
                ...options,
                headers: { 'Content-Type': 'application/json', ...options.headers }
            });
            return await res.json();
        } catch (e) {
            console.error('API error:', e);
            return null;
        }
    }
    
    // Load functions
    async function loadOrganism() {
        const data = await api('/api/organism/state');
        if (data) {
            organismData = data;
            
            document.getElementById('karma-display').textContent = 
                data.karma?.field_potential?.toFixed(2) || '0.00';
            document.getElementById('harmony-display').textContent = 
                data.harmony?.toFixed(2) || '1.00';
            document.getElementById('orb-display').textContent = 
                data.swarm?.total_orbs || 0;
            
            // Update orbs in 3D
            if (data.swarm?.orbs) {
                updateOrbs(data.swarm.orbs);
            }
            
            // Update Mayan widget
            if (data.mayan) {
                document.getElementById('mayanKin').textContent = data.mayan.greeting;
                document.getElementById('mayanEnergy').textContent = 
                    data.mayan.energy?.number_meaning || '';
            }
        }
    }
    
    async function loadDashboard() {
        const container = document.getElementById('dashboard-content');
        container.innerHTML = `
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:15px;">
                <div class="item-card">
                    <div class="meta">Karma</div>
                    <div style="font-size:1.5em;color:var(--accent-gold);">
                        ${organismData.karma?.field_potential?.toFixed(2) || 0}
                    </div>
                </div>
                <div class="item-card">
                    <div class="meta">Harmony</div>
                    <div style="font-size:1.5em;color:var(--accent-blue);">
                        ${organismData.harmony?.toFixed(2) || 1}
                    </div>
                </div>
                <div class="item-card">
                    <div class="meta">Living Orbs</div>
                    <div style="font-size:1.5em;">${organismData.swarm?.total_orbs || 0}</div>
                </div>
                <div class="item-card">
                    <div class="meta">AI Status</div>
                    <div style="font-size:1.5em;">${organismData.ai?.ollama_available ? '🟢' : '🟡'}</div>
                </div>
            </div>
            <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
                <button class="btn" onclick="quickAction('complete')">✅ Complete Task</button>
                <button class="btn" onclick="quickAction('meditate')">🧘 Meditate</button>
                <button class="btn btn-gold" onclick="quickAction('achieve')">🏆 Achievement</button>
            </div>
        `;
    }
    
    async function loadGoals() {
        const goals = await api('/api/goals');
        const container = document.getElementById('goals-list');
        
        if (goals && goals.length > 0) {
            container.innerHTML = goals.map(g => `
                <div class="item-card">
                    <h3>${g.title}</h3>
                    <div class="meta">Progress: ${g.progress?.toFixed(0) || 0}%</div>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<p style="color:var(--text-muted);">No goals yet</p>';
        }
    }
    
    async function createGoal() {
        const title = document.getElementById('goalTitle').value;
        if (!title) return showToast('Enter a goal title');
        
        const result = await api('/api/goals', {
            method: 'POST',
            body: JSON.stringify({ title })
        });
        
        if (result) {
            showToast(`🎯 Goal created! +${result.karma_earned?.toFixed(2)} karma`);
            document.getElementById('goalTitle').value = '';
            loadGoals();
            loadOrganism();
        }
    }
    
    async function loadHabits() {
        const habits = await api('/api/habits');
        const container = document.getElementById('habits-list');
        
        if (habits && habits.length > 0) {
            container.innerHTML = habits.map(h => `
                <div class="item-card" style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <h3>${h.name}</h3>
                        <div class="meta">🔥 ${h.current_streak || 0} streak</div>
                    </div>
                    <button class="btn" onclick="completeHabit('${h.id}')">✓</button>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<p style="color:var(--text-muted);">No habits yet</p>';
        }
    }
    
    async function createHabit() {
        const name = document.getElementById('habitName').value;
        if (!name) return showToast('Enter a habit name');
        
        const result = await api('/api/habits', {
            method: 'POST',
            body: JSON.stringify({ name })
        });
        
        if (result) {
            showToast('✨ Habit created!');
            document.getElementById('habitName').value = '';
            loadHabits();
        }
    }
    
    async function completeHabit(id) {
        const result = await api(`/api/habits/${id}/complete`, { method: 'POST' });
        if (result) {
            let msg = `✅ +${result.karma_earned?.toFixed(2)} karma`;
            if (result.fibonacci_bonus) msg += ' 🌟 Fibonacci bonus!';
            showToast(msg);
            loadHabits();
            loadOrganism();
        }
    }
    
    async function submitCheckin() {
        const data = {
            energy: parseInt(document.getElementById('energySlider').value),
            mood: parseInt(document.getElementById('moodSlider').value),
            focus: parseInt(document.getElementById('focusSlider').value)
        };
        
        const result = await api('/api/wellness/checkin', {
            method: 'POST',
            body: JSON.stringify(data)
        });
        
        if (result) {
            showToast(`💫 Check-in complete! +${result.karma_earned?.toFixed(2)} karma`);
            closePanel();
            loadOrganism();
        }
    }
    
    async function loadPet() {
        const pet = await api('/api/pet/state');
        if (pet) {
            document.getElementById('petHunger').textContent = Math.round(pet.hunger || 50);
            document.getElementById('petEnergy').textContent = Math.round(pet.energy || 50);
            document.getElementById('petHappy').textContent = Math.round(pet.happiness || 50);
            
            const emoji = pet.happiness > 70 ? '😺' : pet.happiness > 40 ? '🐱' : '😿';
            document.getElementById('petEmoji').textContent = emoji;
        }
    }
    
    async function petAction(action) {
        const result = await api('/api/pet/interact', {
            method: 'POST',
            body: JSON.stringify({ action })
        });
        
        if (result) {
            showToast(`🐱 ${result.emotion}! +${result.karma_earned?.toFixed(2)} karma`);
            loadPet();
            loadOrganism();
        }
    }
    
    async function loadPatterns() {
        const data = await api('/api/analytics/patterns');
        const container = document.getElementById('patterns-content');
        
        if (data) {
            const patterns = data.patterns?.[0] || {};
            const recommendations = data.recommendations || [];
            
            container.innerHTML = `
                <div class="item-card">
                    <h3>Detected Patterns</h3>
                    <p>${patterns.insights?.join('<br>') || 'Keep using the app to detect patterns...'}</p>
                </div>
                <div class="item-card">
                    <h3>AI Recommendations</h3>
                    <p>${recommendations.join('<br>') || 'No recommendations yet'}</p>
                </div>
                <div class="item-card">
                    <h3>Federated AI</h3>
                    <p>Model v${data.federated_state?.model_version || '1.0.0'}<br>
                    Learning iterations: ${data.federated_state?.learning_iterations || 0}</p>
                </div>
            `;
        }
    }
    
    async function loadKarmaHistory() {
        const history = await api('/api/analytics/karma-history');
        const container = document.getElementById('karma-history');
        
        if (history && history.length > 0) {
            container.innerHTML = history.slice(0, 10).map(h => `
                <div class="item-card">
                    <div style="display:flex;justify-content:space-between;">
                        <span>${h.action_type}</span>
                        <span style="color:var(--accent-gold);">+${h.karma_value?.toFixed(2)}</span>
                    </div>
                    ${h.meaning ? `<p class="meta">${h.meaning}</p>` : ''}
                </div>
            `).join('');
        } else {
            container.innerHTML = '<p style="color:var(--text-muted);">No karma history yet</p>';
        }
    }
    
    async function quickAction(type) {
        const result = await api('/api/organism/action', {
            method: 'POST',
            body: JSON.stringify({ action_type: type, magnitude: 1.0 })
        });
        
        if (result) {
            showToast(`✨ +${result.karma_earned?.toFixed(2)} karma`);
            loadOrganism();
        }
    }
    
    function showToast(msg) {
        const toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 3000);
    }
    
    // Initialize
    document.addEventListener('DOMContentLoaded', () => {
        initThreeJS();
        loadOrganism();
        
        // Refresh every 5 seconds
        setInterval(loadOrganism, 5000);
    });
    </script>
</body>
</html>
'''

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal - Login</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card {
            background: rgba(15, 15, 25, 0.95);
            padding: 40px;
            border-radius: 20px;
            width: 100%;
            max-width: 400px;
            border: 1px solid rgba(212, 175, 55, 0.2);
        }
        .logo { font-size: 3em; text-align: center; margin-bottom: 10px; }
        h1 { text-align: center; color: #d4af37; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #888; }
        input {
            width: 100%;
            padding: 14px;
            background: rgba(74, 144, 164, 0.1);
            border: 1px solid rgba(74, 144, 164, 0.3);
            border-radius: 10px;
            color: #e8e8e8;
            font-size: 1em;
        }
        .btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #8b5cf6 0%, #4a90a4 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1em;
            cursor: pointer;
        }
        .switch { text-align: center; margin-top: 20px; color: #888; }
        .switch a { color: #d4af37; text-decoration: none; }
        .error { background: rgba(244, 67, 54, 0.2); color: #f44336; padding: 10px; border-radius: 8px; margin-bottom: 20px; display: none; }
    </style>
</head>
<body>
    <div class="card">
        <div class="logo">🌀</div>
        <h1 id="title">Login</h1>
        <div class="error" id="error"></div>
        <form id="form">
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="email" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="password" required>
            </div>
            <div class="form-group" id="nameGroup" style="display:none;">
                <label>Name</label>
                <input type="text" id="name">
            </div>
            <button type="submit" class="btn" id="submitBtn">Login</button>
        </form>
        <div class="switch">
            <span id="switchText">New here?</span>
            <a href="#" onclick="toggle(event)">Register</a>
        </div>
    </div>
    <script>
        let isLogin = true;
        function toggle(e) {
            e.preventDefault();
            isLogin = !isLogin;
            document.getElementById('title').textContent = isLogin ? 'Login' : 'Register';
            document.getElementById('submitBtn').textContent = isLogin ? 'Login' : 'Register';
            document.getElementById('nameGroup').style.display = isLogin ? 'none' : 'block';
            document.getElementById('switchText').textContent = isLogin ? 'New here?' : 'Have an account?';
        }
        document.getElementById('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {
                email: document.getElementById('email').value,
                password: document.getElementById('password').value
            };
            if (!isLogin) data.first_name = document.getElementById('name').value;
            try {
                const res = await fetch(isLogin ? '/api/auth/login' : '/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (res.ok) {
                    window.location.href = '/';
                } else {
                    document.getElementById('error').textContent = result.error;
                    document.getElementById('error').style.display = 'block';
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Connection error';
                document.getElementById('error').style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template_string(MAIN_HTML)


@app.route('/login')
def login_page():
    return render_template_string(LOGIN_HTML)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE v12.0 - LIVING MATHEMATICAL ORGANISM")
    print("═" * 80)
    print("\n✨ Features:")
    print("   🌐 Full-screen interactive 3D fractal universe")
    print("   🤖 Ollama AI integration for self-spawning orbs")
    print("   🔍 Zoom-based label visibility")
    print("   🧬 Self-aware, self-replicating cells")
    print("   📅 Mayan calendar time science")
    print("   🌊 Swarm intelligence pattern detection")
    print("   ⚖️ Karma-dharma spiritual mathematics")
    print("   🧠 Federated AI with recursive learning")
    print(f"\n🤖 Ollama: {'Connected' if organism.ai.available else 'Offline (using patterns)'}")
    print(f"📊 ML: {'Enabled' if HAS_SKLEARN else 'Disabled'}")
    print(f"📅 Today: {organism.mayan.get_today_summary()['greeting']}")
    print("\n" + "═" * 80 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
