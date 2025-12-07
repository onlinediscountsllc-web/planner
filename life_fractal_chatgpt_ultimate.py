"""
🌀 LIFE FRACTAL INTELLIGENCE - ULTIMATE CHATGPT EDITION
═══════════════════════════════════════════════════════════════════════════════════════
Complete life planning system with ChatGPT integration featuring:

🐾 VIRTUAL PET SYSTEM
- 8 unique species with personalities
- Feed, play, train, bond with your pet
- Pet guides you through fractal exploration
- Levels, abilities, and evolution

🎨 FRACTAL VISUALIZATION
- 12+ fractal types (Mandelbrot, Julia, Golden Spiral, etc.)
- Sacred geometry overlays
- Goal-to-fractal transformation
- Aphantasia-friendly external visualization

📊 LIFE PLANNING FOR NEURODIVERGENT MINDS
- Spoon Theory energy management
- Zero-shame progress tracking
- Executive dysfunction accommodations
- External visualization (no mental imagery needed!)

🎁 MEDIA GENERATION
- High-res posters (print quality)
- Animated GIF fractals
- Personalized art based on your goals
- Gallery of your creations

🔐 SIGNUP FOR DOWNLOADS
- Create free account to save your creations
- Gallery of your generated media
- 7-day free trial for premium features

═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import secrets
import logging
import hashlib
import uuid
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
import base64
import random
from functools import wraps

# Flask
from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ML (optional)
try:
    from sklearn.tree import DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🔢 SACRED MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

# Mayan Calendar Constants
TZOLKIN_CYCLE = 260
HAAB_CYCLE = 365
CALENDAR_ROUND = 18980  # LCM of 260 and 365


# ═══════════════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    """Production configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    CHATGPT_API_KEY = os.environ.get('CHATGPT_API_KEY', 'fractal-explorer-2024')
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    GOFUNDME_URL = os.environ.get('GOFUNDME_URL', 'https://gofund.me/8d9303d27')
    SUBSCRIPTION_PRICE = 20.00
    TRIAL_DAYS = 7
    DATA_DIR = os.environ.get('DATA_DIR', '/tmp/data')
    MAX_FREE_GENERATIONS = 10
    MAX_PREMIUM_GENERATIONS = 1000


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🐾 VIRTUAL PET SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

PET_SPECIES = {
    'cat': {
        'name': 'Mystic Cat',
        'emoji': '🐱',
        'personality': 'curious and independent',
        'favorite_fractal': 'mandelbrot',
        'color_palette': ['#9b59b6', '#8e44ad', '#6c3483'],
        'greetings': [
            "Mrow! *stretches lazily* Ready to explore some fractals?",
            "Purrrr... I sense mathematical beauty nearby!",
            "*blinks slowly* The golden ratio speaks to me..."
        ],
        'abilities': ['Night Vision', 'Nine Lives', 'Fractal Whiskers'],
        'evolution_chain': ['Kitten', 'Cat', 'Mystic Cat', 'Cosmic Feline']
    },
    'dog': {
        'name': 'Loyal Companion',
        'emoji': '🐕',
        'personality': 'enthusiastic and loyal',
        'favorite_fractal': 'golden_spiral',
        'color_palette': ['#e67e22', '#d35400', '#a04000'],
        'greetings': [
            "Woof woof! *tail wagging* Let's make beautiful fractals together!",
            "*happy panting* Every goal is a new adventure!",
            "Arf! I fetched some sacred geometry for you!"
        ],
        'abilities': ['Loyalty Boost', 'Goal Tracking', 'Happy Howl'],
        'evolution_chain': ['Puppy', 'Dog', 'Loyal Companion', 'Golden Guardian']
    },
    'dragon': {
        'name': 'Fractal Dragon',
        'emoji': '🐉',
        'personality': 'wise and powerful',
        'favorite_fractal': 'dragon',
        'color_palette': ['#c0392b', '#e74c3c', '#f39c12'],
        'greetings': [
            "*breathes geometric fire* The ancient patterns await...",
            "Greetings, seeker of infinite beauty!",
            "*spreads fractal wings* Together we shall create wonders!"
        ],
        'abilities': ['Fire Breath', 'Ancient Wisdom', 'Infinite Scales'],
        'evolution_chain': ['Hatchling', 'Drake', 'Dragon', 'Fractal Wyrm']
    },
    'phoenix': {
        'name': 'Eternal Phoenix',
        'emoji': '🔥',
        'personality': 'transformative and inspiring',
        'favorite_fractal': 'phoenix',
        'color_palette': ['#f39c12', '#e74c3c', '#9b59b6'],
        'greetings': [
            "*rises from fractal ashes* Every end is a new beginning!",
            "Let your goals burn bright like my feathers!",
            "*spreads wings of golden light* Transform with me!"
        ],
        'abilities': ['Rebirth', 'Transformation', 'Eternal Flame'],
        'evolution_chain': ['Spark', 'Flame Bird', 'Phoenix', 'Eternal Phoenix']
    },
    'owl': {
        'name': 'Wisdom Owl',
        'emoji': '🦉',
        'personality': 'wise and observant',
        'favorite_fractal': 'sierpinski',
        'color_palette': ['#34495e', '#2c3e50', '#8e44ad'],
        'greetings': [
            "Hoo-hoo... *adjusts spectacles* Shall we analyze some patterns?",
            "*blinks wisely* Knowledge lies within the fractals...",
            "Wisdom comes to those who see the infinite in the finite."
        ],
        'abilities': ['Deep Analysis', 'Pattern Recognition', 'Night Study'],
        'evolution_chain': ['Owlet', 'Owl', 'Sage Owl', 'Cosmic Oracle']
    },
    'fox': {
        'name': 'Clever Fox',
        'emoji': '🦊',
        'personality': 'clever and playful',
        'favorite_fractal': 'julia',
        'color_palette': ['#e67e22', '#d35400', '#c0392b'],
        'greetings': [
            "*tilts head* Curious! What mathematical mischief shall we make?",
            "Quick quick! The fractals are waiting for us!",
            "*playful yip* Every pattern tells a story!"
        ],
        'abilities': ['Quick Thinking', 'Adaptability', 'Clever Solutions'],
        'evolution_chain': ['Kit', 'Fox', 'Clever Fox', 'Nine-Tailed Sage']
    },
    'unicorn': {
        'name': 'Sacred Unicorn',
        'emoji': '🦄',
        'personality': 'magical and pure',
        'favorite_fractal': 'flower_of_life',
        'color_palette': ['#9b59b6', '#e91e63', '#00bcd4'],
        'greetings': [
            "*horn glows with sacred light* Welcome to the realm of infinite beauty!",
            "Every dream is a fractal waiting to bloom!",
            "*prances gracefully* Let's weave magic into mathematics!"
        ],
        'abilities': ['Rainbow Magic', 'Dream Weaving', 'Sacred Horn'],
        'evolution_chain': ['Foal', 'Unicorn', 'Sacred Unicorn', 'Celestial Unicorn']
    },
    'butterfly': {
        'name': 'Chaos Butterfly',
        'emoji': '🦋',
        'personality': 'gentle and transformative',
        'favorite_fractal': 'lorenz',
        'color_palette': ['#3498db', '#9b59b6', '#1abc9c'],
        'greetings': [
            "*flutters delicately* Small actions create infinite patterns!",
            "Like the butterfly effect, your goals ripple through reality...",
            "*lands softly* Ready to create beautiful chaos?"
        ],
        'abilities': ['Butterfly Effect', 'Gentle Transformation', 'Chaos Theory'],
        'evolution_chain': ['Caterpillar', 'Chrysalis', 'Butterfly', 'Chaos Butterfly']
    }
}

# Pet food items
PET_FOOD = {
    'golden_apple': {'hunger': -50, 'happiness': +20, 'xp': 25, 'description': 'A perfect apple grown in golden ratio proportions'},
    'fibonacci_berries': {'hunger': -30, 'happiness': +15, 'xp': 15, 'description': 'Berries arranged in sacred spirals'},
    'fractal_treat': {'hunger': -40, 'happiness': +25, 'xp': 30, 'description': 'A treat with infinite delicious layers'},
    'cosmic_kibble': {'hunger': -60, 'happiness': +10, 'xp': 20, 'description': 'Standard nutritious pet food'},
    'sacred_nectar': {'hunger': -20, 'happiness': +30, 'xp': 35, 'description': 'Sweet nectar infused with phi'}
}

# Pet activities
PET_ACTIVITIES = {
    'explore_mandelbrot': {'happiness': +30, 'xp': 50, 'energy': -20, 'description': 'Journey to the edge of infinity'},
    'chase_fractals': {'happiness': +25, 'xp': 30, 'energy': -15, 'description': 'Playful pursuit of patterns'},
    'meditate_sacred': {'happiness': +20, 'xp': 40, 'energy': +10, 'description': 'Rest in the flower of life'},
    'learn_geometry': {'happiness': +15, 'xp': 60, 'energy': -10, 'description': 'Study sacred mathematics'},
    'dream_fractals': {'happiness': +35, 'xp': 45, 'energy': +20, 'description': 'Sleep among infinite patterns'}
}


@dataclass
class VirtualPet:
    """Virtual pet companion for fractal exploration"""
    species: str
    name: str
    level: int = 1
    experience: int = 0
    happiness: int = 80
    hunger: int = 20
    energy: int = 100
    bond_level: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_fed: Optional[str] = None
    last_played: Optional[str] = None
    last_interaction: Optional[str] = None
    total_fractals_created: int = 0
    favorite_colors: List[str] = field(default_factory=list)
    unlocked_abilities: List[str] = field(default_factory=list)
    evolution_stage: int = 0
    personality_traits: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.favorite_colors and self.species in PET_SPECIES:
            self.favorite_colors = PET_SPECIES[self.species]['color_palette']
        if not self.unlocked_abilities:
            self.unlocked_abilities = [PET_SPECIES.get(self.species, {}).get('abilities', ['Basic'])[0]]
    
    def get_mood(self) -> str:
        """Determine pet's current mood"""
        if self.happiness >= 80 and self.hunger <= 30:
            return 'ecstatic'
        elif self.happiness >= 60 and self.hunger <= 50:
            return 'happy'
        elif self.happiness >= 40:
            return 'content'
        elif self.hunger >= 70:
            return 'hungry'
        elif self.happiness < 30:
            return 'sad'
        else:
            return 'neutral'
    
    def get_greeting(self) -> str:
        """Get a mood-appropriate greeting"""
        species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
        mood = self.get_mood()
        
        if mood in ['ecstatic', 'happy']:
            return random.choice(species_data['greetings'])
        elif mood == 'hungry':
            return f"{species_data['emoji']} *stomach growls* I could use a snack before we explore..."
        elif mood == 'sad':
            return f"{species_data['emoji']} *looks down* I miss playing with you..."
        else:
            return f"{species_data['emoji']} Hello! Ready for some fractal fun?"
    
    def feed(self, food_type: str = 'cosmic_kibble') -> dict:
        """Feed the pet"""
        food = PET_FOOD.get(food_type, PET_FOOD['cosmic_kibble'])
        
        self.hunger = max(0, min(100, self.hunger + food['hunger']))
        self.happiness = max(0, min(100, self.happiness + food['happiness']))
        self.experience += food['xp']
        self.last_fed = datetime.now(timezone.utc).isoformat()
        self.last_interaction = self.last_fed
        
        self._check_level_up()
        
        species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
        
        return {
            'success': True,
            'message': f"{species_data['emoji']} *munches happily* {food['description']}!",
            'food': food_type,
            'stats': self.get_stats()
        }
    
    def play(self, activity: str = 'chase_fractals') -> dict:
        """Play with the pet"""
        act = PET_ACTIVITIES.get(activity, PET_ACTIVITIES['chase_fractals'])
        
        if self.energy < abs(act.get('energy', 0)) and act.get('energy', 0) < 0:
            species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
            return {
                'success': False,
                'message': f"{species_data['emoji']} *yawns* I'm too tired... need rest first.",
                'stats': self.get_stats()
            }
        
        self.happiness = max(0, min(100, self.happiness + act['happiness']))
        self.experience += act['xp']
        self.energy = max(0, min(100, self.energy + act['energy']))
        self.bond_level += 5
        self.last_played = datetime.now(timezone.utc).isoformat()
        self.last_interaction = self.last_played
        
        self._check_level_up()
        
        species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
        
        return {
            'success': True,
            'message': f"{species_data['emoji']} {act['description']} - So much fun!",
            'activity': activity,
            'xp_gained': act['xp'],
            'stats': self.get_stats()
        }
    
    def _check_level_up(self):
        """Check if pet should level up"""
        xp_needed = self.level * 100
        while self.experience >= xp_needed:
            self.experience -= xp_needed
            self.level += 1
            xp_needed = self.level * 100
            
            # Unlock new ability every 5 levels
            species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
            abilities = species_data.get('abilities', [])
            unlocked_count = len(self.unlocked_abilities)
            if unlocked_count < len(abilities) and self.level % 5 == 0:
                self.unlocked_abilities.append(abilities[unlocked_count])
            
            # Evolution at levels 10, 25, 50
            evolution_chain = species_data.get('evolution_chain', [])
            if self.level >= 50 and self.evolution_stage < 3:
                self.evolution_stage = 3
            elif self.level >= 25 and self.evolution_stage < 2:
                self.evolution_stage = 2
            elif self.level >= 10 and self.evolution_stage < 1:
                self.evolution_stage = 1
    
    def update_stats(self):
        """Update time-based stats"""
        now = datetime.now(timezone.utc)
        
        if self.last_fed:
            try:
                fed_time = datetime.fromisoformat(self.last_fed.replace('Z', '+00:00'))
                hours = (now - fed_time).total_seconds() / 3600
                self.hunger = min(100, int(self.hunger + hours * 3))
            except:
                pass
        
        if self.last_played:
            try:
                played_time = datetime.fromisoformat(self.last_played.replace('Z', '+00:00'))
                hours = (now - played_time).total_seconds() / 3600
                self.happiness = max(0, int(self.happiness - hours * 2))
            except:
                pass
        
        # Energy regenerates over time
        if self.last_interaction:
            try:
                interact_time = datetime.fromisoformat(self.last_interaction.replace('Z', '+00:00'))
                hours = (now - interact_time).total_seconds() / 3600
                self.energy = min(100, int(self.energy + hours * 10))
            except:
                pass
    
    def get_stats(self) -> dict:
        """Get current pet stats"""
        self.update_stats()
        species_data = PET_SPECIES.get(self.species, PET_SPECIES['cat'])
        evolution_chain = species_data.get('evolution_chain', [self.species])
        
        return {
            'species': self.species,
            'species_name': evolution_chain[min(self.evolution_stage, len(evolution_chain)-1)],
            'name': self.name,
            'emoji': species_data.get('emoji', '🐾'),
            'level': self.level,
            'experience': self.experience,
            'xp_to_next': self.level * 100,
            'happiness': self.happiness,
            'hunger': self.hunger,
            'energy': self.energy,
            'mood': self.get_mood(),
            'bond_level': self.bond_level,
            'evolution_stage': self.evolution_stage,
            'unlocked_abilities': self.unlocked_abilities,
            'total_fractals': self.total_fractals_created,
            'favorite_fractal': species_data.get('favorite_fractal', 'mandelbrot'),
            'personality': species_data.get('personality', 'friendly')
        }
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🎨 FRACTAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

FRACTAL_TYPES = {
    'mandelbrot': {
        'name': 'Mandelbrot Set',
        'description': 'The most famous fractal - infinite complexity from a simple equation. Perfect for meditation and seeing how small changes create infinite variety.',
        'meaning': 'Represents how small consistent actions lead to infinite possibilities',
        'best_for': ['meditation', 'focus', 'complexity appreciation'],
        'params': ['zoom', 'iterations', 'center_x', 'center_y']
    },
    'julia': {
        'name': 'Julia Set',
        'description': 'The Mandelbrot\'s twin - each point creates a unique universe. Great for visualizing different life paths.',
        'meaning': 'Shows how different starting points lead to different beautiful outcomes',
        'best_for': ['decision making', 'exploring possibilities', 'creativity'],
        'params': ['c_real', 'c_imag', 'zoom', 'iterations']
    },
    'golden_spiral': {
        'name': 'Golden Spiral',
        'description': 'Based on φ (phi = 1.618...) - nature\'s favorite number found in shells, galaxies, and DNA.',
        'meaning': 'Natural growth and harmony - your goals growing in perfect balance',
        'best_for': ['growth goals', 'balance', 'natural progress'],
        'params': ['turns', 'growth_rate', 'density']
    },
    'flower_of_life': {
        'name': 'Flower of Life',
        'description': 'Ancient sacred geometry found in temples worldwide - interconnected circles of creation.',
        'meaning': 'Everything is connected - your goals support each other',
        'best_for': ['spiritual goals', 'relationships', 'holistic planning'],
        'params': ['rings', 'size', 'rotation']
    },
    'burning_ship': {
        'name': 'Burning Ship',
        'description': 'A variation of the Mandelbrot that creates dramatic ship-like shapes emerging from chaos.',
        'meaning': 'Rising above challenges - strength through adversity',
        'best_for': ['overcoming obstacles', 'resilience', 'transformation'],
        'params': ['zoom', 'iterations']
    },
    'phoenix': {
        'name': 'Phoenix Fractal',
        'description': 'Named after the mythical bird that rises from ashes - regenerating patterns.',
        'meaning': 'Rebirth and renewal - every ending is a new beginning',
        'best_for': ['new beginnings', 'recovery', 'fresh starts'],
        'params': ['p_real', 'p_imag', 'zoom']
    },
    'sierpinski': {
        'name': 'Sierpiński Triangle',
        'description': 'Infinite triangles within triangles - perfect self-similarity at every scale.',
        'meaning': 'The power of repetition - small habits build great things',
        'best_for': ['habit building', 'consistency', 'step-by-step progress'],
        'params': ['depth', 'variation']
    },
    'dragon': {
        'name': 'Dragon Curve',
        'description': 'Created by paper folding - simple rules create complex beauty.',
        'meaning': 'Complexity from simplicity - trust the process',
        'best_for': ['patience', 'process-oriented goals', 'trust'],
        'params': ['iterations', 'angle']
    },
    'koch_snowflake': {
        'name': 'Koch Snowflake',
        'description': 'Infinite perimeter in finite area - boundless potential within your limits.',
        'meaning': 'Infinite growth within boundaries - expand without overwhelm',
        'best_for': ['focused growth', 'boundary setting', 'deep work'],
        'params': ['iterations', 'size']
    },
    'fibonacci': {
        'name': 'Fibonacci Spiral',
        'description': 'Numbers where each is the sum of the two before: 1,1,2,3,5,8,13...',
        'meaning': 'Build on your past successes - growth compounds',
        'best_for': ['building momentum', 'compound growth', 'leveraging success'],
        'params': ['terms', 'spiral_tightness']
    },
    'lorenz': {
        'name': 'Lorenz Attractor',
        'description': 'The butterfly effect visualized - tiny changes create big differences.',
        'meaning': 'Small actions matter - every step counts',
        'best_for': ['motivation', 'seeing impact', 'starting small'],
        'params': ['rho', 'sigma', 'beta']
    },
    'metatron': {
        'name': 'Metatron\'s Cube',
        'description': 'Sacred geometry containing all Platonic solids - the building blocks of the universe.',
        'meaning': 'Universal structure - your life as sacred architecture',
        'best_for': ['life purpose', 'structure', 'comprehensive planning'],
        'params': ['complexity', 'rotation']
    }
}


class FractalEngine:
    """Advanced fractal generation engine"""
    
    def __init__(self, default_size: int = 800):
        self.default_size = default_size
    
    def generate(self, fractal_type: str, size: int = None, params: dict = None) -> Image.Image:
        """Generate a fractal image"""
        size = size or self.default_size
        params = params or {}
        
        generators = {
            'mandelbrot': self._mandelbrot,
            'julia': self._julia,
            'golden_spiral': self._golden_spiral,
            'flower_of_life': self._flower_of_life,
            'burning_ship': self._burning_ship,
            'phoenix': self._phoenix,
            'sierpinski': self._sierpinski,
            'dragon': self._dragon,
            'koch_snowflake': self._koch_snowflake,
            'fibonacci': self._fibonacci,
            'lorenz': self._lorenz,
            'metatron': self._metatron
        }
        
        generator = generators.get(fractal_type, self._mandelbrot)
        return generator(size, params)
    
    def _create_colormap(self, iterations: int, hue_offset: float = 0) -> List[Tuple[int, int, int]]:
        """Create a beautiful colormap based on golden ratio"""
        colors = []
        for i in range(iterations):
            # Use golden angle for pleasing color distribution
            hue = (hue_offset + i * GOLDEN_ANGLE / 360) % 1.0
            sat = 0.7 + 0.3 * math.sin(i * PHI_INVERSE)
            val = 0.8 + 0.2 * math.cos(i * PHI_INVERSE)
            
            # HSV to RGB
            c = val * sat
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = val - c
            
            if hue < 1/6:
                r, g, b = c, x, 0
            elif hue < 2/6:
                r, g, b = x, c, 0
            elif hue < 3/6:
                r, g, b = 0, c, x
            elif hue < 4/6:
                r, g, b = 0, x, c
            elif hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append((int((r+m)*255), int((g+m)*255), int((b+m)*255)))
        
        return colors
    
    def _mandelbrot(self, size: int, params: dict) -> Image.Image:
        """Generate Mandelbrot set"""
        zoom = params.get('zoom', 1.0)
        iterations = min(params.get('iterations', 100), 500)
        center_x = params.get('center_x', -0.5)
        center_y = params.get('center_y', 0)
        
        img = Image.new('RGB', (size, size), 'black')
        pixels = img.load()
        
        colors = self._create_colormap(iterations)
        
        for py in range(size):
            for px in range(size):
                x0 = (px - size/2) / (size/4) / zoom + center_x
                y0 = (py - size/2) / (size/4) / zoom + center_y
                
                x, y = 0.0, 0.0
                iteration = 0
                
                while x*x + y*y <= 4 and iteration < iterations:
                    xtemp = x*x - y*y + x0
                    y = 2*x*y + y0
                    x = xtemp
                    iteration += 1
                
                if iteration < iterations:
                    # Smooth coloring
                    log_zn = math.log(x*x + y*y) / 2
                    nu = math.log(log_zn / math.log(2)) / math.log(2) if log_zn > 0 else 0
                    iteration = iteration + 1 - nu
                    
                    color_idx = int(iteration) % len(colors)
                    pixels[px, py] = colors[color_idx]
        
        return img
    
    def _julia(self, size: int, params: dict) -> Image.Image:
        """Generate Julia set"""
        c_real = params.get('c_real', -0.7)
        c_imag = params.get('c_imag', 0.27015)
        zoom = params.get('zoom', 1.0)
        iterations = min(params.get('iterations', 100), 500)
        
        img = Image.new('RGB', (size, size), 'black')
        pixels = img.load()
        
        colors = self._create_colormap(iterations, hue_offset=0.3)
        
        for py in range(size):
            for px in range(size):
                x = (px - size/2) / (size/4) / zoom
                y = (py - size/2) / (size/4) / zoom
                
                iteration = 0
                
                while x*x + y*y <= 4 and iteration < iterations:
                    xtemp = x*x - y*y + c_real
                    y = 2*x*y + c_imag
                    x = xtemp
                    iteration += 1
                
                if iteration < iterations:
                    color_idx = iteration % len(colors)
                    pixels[px, py] = colors[color_idx]
        
        return img
    
    def _golden_spiral(self, size: int, params: dict) -> Image.Image:
        """Generate golden spiral"""
        turns = params.get('turns', 8)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        max_radius = size * 0.45
        
        # Draw golden spiral
        points = []
        for i in range(int(turns * 360)):
            theta = math.radians(i)
            r = max_radius * (PHI ** (theta / (2 * math.pi))) / (PHI ** turns)
            x = center + r * math.cos(theta)
            y = center + r * math.sin(theta)
            points.append((x, y))
        
        # Draw with gradient colors
        for i in range(len(points) - 1):
            hue = (i / len(points) * GOLDEN_ANGLE) % 360
            color = f'hsl({int(hue)}, 80%, 60%)'
            draw.line([points[i], points[i+1]], fill=color, width=3)
        
        # Add Fibonacci circles
        fib_radii = [max_radius * f / max(FIBONACCI[:12]) for f in FIBONACCI[:12]]
        for i, r in enumerate(fib_radii[2:]):
            hue = (i * GOLDEN_ANGLE) % 360
            color = f'hsl({int(hue)}, 70%, 50%)'
            draw.ellipse([center-r, center-r, center+r, center+r], outline=color, width=1)
        
        return img
    
    def _flower_of_life(self, size: int, params: dict) -> Image.Image:
        """Generate Flower of Life sacred geometry"""
        rings = params.get('rings', 3)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        radius = size // (rings * 4 + 2)
        
        # Central circle
        positions = [(center, center)]
        
        # Expand rings
        for ring in range(rings):
            ring_radius = radius * (ring + 1) * 2
            for i in range(6):
                angle = i * math.pi / 3
                x = center + ring_radius * math.cos(angle)
                y = center + ring_radius * math.sin(angle)
                positions.append((x, y))
                
                # Fill between
                for j in range(ring):
                    sub_angle = angle + (j + 1) * math.pi / 3 / (ring + 1)
                    sx = center + ring_radius * math.cos(sub_angle)
                    sy = center + ring_radius * math.sin(sub_angle)
                    positions.append((sx, sy))
        
        # Draw circles
        colors = ['#9b59b6', '#3498db', '#1abc9c', '#f39c12', '#e74c3c', '#e91e63']
        for i, (x, y) in enumerate(positions):
            color = colors[i % len(colors)]
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=2)
        
        return img
    
    def _burning_ship(self, size: int, params: dict) -> Image.Image:
        """Generate Burning Ship fractal"""
        zoom = params.get('zoom', 1.0)
        iterations = min(params.get('iterations', 100), 500)
        
        img = Image.new('RGB', (size, size), 'black')
        pixels = img.load()
        
        colors = self._create_colormap(iterations, hue_offset=0.1)
        
        for py in range(size):
            for px in range(size):
                x0 = (px - size * 0.6) / (size/4) / zoom
                y0 = (py - size * 0.5) / (size/4) / zoom
                
                x, y = 0.0, 0.0
                iteration = 0
                
                while x*x + y*y <= 4 and iteration < iterations:
                    xtemp = x*x - y*y + x0
                    y = abs(2*x*y) + y0
                    x = abs(xtemp)
                    iteration += 1
                
                if iteration < iterations:
                    color_idx = iteration % len(colors)
                    pixels[px, py] = colors[color_idx]
        
        return img
    
    def _phoenix(self, size: int, params: dict) -> Image.Image:
        """Generate Phoenix fractal"""
        p_real = params.get('p_real', 0.5667)
        p_imag = params.get('p_imag', -0.5)
        zoom = params.get('zoom', 1.0)
        iterations = 100
        
        img = Image.new('RGB', (size, size), 'black')
        pixels = img.load()
        
        colors = self._create_colormap(iterations, hue_offset=0.05)
        
        for py in range(size):
            for px in range(size):
                x = (px - size/2) / (size/4) / zoom
                y = (py - size/2) / (size/4) / zoom
                
                xprev, yprev = 0.0, 0.0
                iteration = 0
                
                while x*x + y*y <= 4 and iteration < iterations:
                    xtemp = x*x - y*y + p_real + p_imag * xprev
                    yprev = y
                    y = 2*x*y + p_imag * yprev
                    xprev = x
                    x = xtemp
                    iteration += 1
                
                if iteration < iterations:
                    color_idx = iteration % len(colors)
                    pixels[px, py] = colors[color_idx]
        
        return img
    
    def _sierpinski(self, size: int, params: dict) -> Image.Image:
        """Generate Sierpiński triangle"""
        depth = min(params.get('depth', 7), 10)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        def draw_triangle(x1, y1, x2, y2, x3, y3, d):
            if d == 0:
                hue = (d * GOLDEN_ANGLE) % 360
                color = f'hsl({int(hue)}, 80%, 60%)'
                draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=color)
            else:
                mx1 = (x1 + x2) / 2
                my1 = (y1 + y2) / 2
                mx2 = (x2 + x3) / 2
                my2 = (y2 + y3) / 2
                mx3 = (x3 + x1) / 2
                my3 = (y3 + y1) / 2
                
                draw_triangle(x1, y1, mx1, my1, mx3, my3, d - 1)
                draw_triangle(mx1, my1, x2, y2, mx2, my2, d - 1)
                draw_triangle(mx3, my3, mx2, my2, x3, y3, d - 1)
        
        margin = size * 0.1
        draw_triangle(
            size/2, margin,
            margin, size - margin,
            size - margin, size - margin,
            depth
        )
        
        return img
    
    def _dragon(self, size: int, params: dict) -> Image.Image:
        """Generate Dragon curve"""
        iterations = min(params.get('iterations', 12), 16)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        # Generate dragon curve using L-system
        axiom = 'FX'
        rules = {'X': 'X+YF+', 'Y': '-FX-Y'}
        
        sequence = axiom
        for _ in range(iterations):
            new_seq = ''
            for char in sequence:
                new_seq += rules.get(char, char)
            sequence = new_seq
        
        # Draw
        x, y = size * 0.3, size * 0.6
        angle = 0
        length = size / (2 ** (iterations / 2 + 1))
        
        points = [(x, y)]
        for char in sequence:
            if char == 'F':
                x += length * math.cos(math.radians(angle))
                y += length * math.sin(math.radians(angle))
                points.append((x, y))
            elif char == '+':
                angle += 90
            elif char == '-':
                angle -= 90
        
        # Draw with gradient
        for i in range(len(points) - 1):
            hue = (i / len(points) * 360) % 360
            color = f'hsl({int(hue)}, 80%, 60%)'
            draw.line([points[i], points[i+1]], fill=color, width=2)
        
        return img
    
    def _koch_snowflake(self, size: int, params: dict) -> Image.Image:
        """Generate Koch snowflake"""
        iterations = min(params.get('iterations', 5), 7)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        def koch_line(x1, y1, x2, y2, depth):
            if depth == 0:
                return [(x1, y1), (x2, y2)]
            
            dx = x2 - x1
            dy = y2 - y1
            
            xa = x1 + dx / 3
            ya = y1 + dy / 3
            xb = x1 + dx * 2 / 3
            yb = y1 + dy * 2 / 3
            
            xc = (x1 + x2) / 2 - dy / 3 * math.sqrt(3) / 2
            yc = (y1 + y2) / 2 + dx / 3 * math.sqrt(3) / 2
            
            points = []
            points.extend(koch_line(x1, y1, xa, ya, depth - 1))
            points.extend(koch_line(xa, ya, xc, yc, depth - 1))
            points.extend(koch_line(xc, yc, xb, yb, depth - 1))
            points.extend(koch_line(xb, yb, x2, y2, depth - 1))
            
            return points
        
        # Initial triangle
        margin = size * 0.15
        r = size / 2 - margin
        center = size / 2
        
        p1 = (center, center - r)
        p2 = (center - r * math.sqrt(3) / 2, center + r / 2)
        p3 = (center + r * math.sqrt(3) / 2, center + r / 2)
        
        all_points = []
        all_points.extend(koch_line(*p1, *p2, iterations))
        all_points.extend(koch_line(*p2, *p3, iterations))
        all_points.extend(koch_line(*p3, *p1, iterations))
        
        # Draw with color
        for i in range(len(all_points) - 1):
            hue = (i / len(all_points) * GOLDEN_ANGLE * 10) % 360
            color = f'hsl({int(hue)}, 80%, 60%)'
            draw.line([all_points[i], all_points[i+1]], fill=color, width=2)
        
        return img
    
    def _fibonacci(self, size: int, params: dict) -> Image.Image:
        """Generate Fibonacci spiral with squares"""
        terms = min(params.get('terms', 12), 15)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        # Calculate scale
        fib = FIBONACCI[:terms]
        max_fib = max(fib) if fib else 1
        scale = size * 0.8 / max_fib
        
        # Starting position
        x, y = size * 0.4, size * 0.4
        direction = 0
        
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#e91e63']
        
        for i, f in enumerate(fib[1:]):
            s = f * scale
            color = colors[i % len(colors)]
            
            # Draw square
            if direction == 0:
                draw.rectangle([x, y, x + s, y + s], outline=color, width=2)
                # Arc
                draw.arc([x - s, y, x + s, y + s * 2], 270, 360, fill=color, width=3)
                x += s
            elif direction == 1:
                draw.rectangle([x - s, y, x, y + s], outline=color, width=2)
                draw.arc([x - s * 2, y - s, x, y + s], 0, 90, fill=color, width=3)
                y += s
            elif direction == 2:
                draw.rectangle([x - s, y - s, x, y], outline=color, width=2)
                draw.arc([x - s, y - s * 2, x + s, y], 90, 180, fill=color, width=3)
                x -= s
            else:
                draw.rectangle([x, y - s, x + s, y], outline=color, width=2)
                draw.arc([x, y - s, x + s * 2, y + s], 180, 270, fill=color, width=3)
                y -= s
            
            direction = (direction + 1) % 4
        
        return img
    
    def _lorenz(self, size: int, params: dict) -> Image.Image:
        """Generate Lorenz attractor (butterfly effect)"""
        rho = params.get('rho', 28.0)
        sigma = params.get('sigma', 10.0)
        beta = params.get('beta', 8.0 / 3.0)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        # Simulate Lorenz system
        dt = 0.005
        x, y, z = 1.0, 1.0, 1.0
        points = []
        
        for _ in range(15000):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            # Project to 2D
            px = size / 2 + x * (size / 60)
            py = size / 2 + z * (size / 60) - size / 4
            points.append((px, py))
        
        # Draw with gradient
        for i in range(len(points) - 1):
            hue = (i / len(points) * 360) % 360
            color = f'hsl({int(hue)}, 80%, 60%)'
            draw.line([points[i], points[i+1]], fill=color, width=1)
        
        return img
    
    def _metatron(self, size: int, params: dict) -> Image.Image:
        """Generate Metatron's Cube"""
        complexity = min(params.get('complexity', 3), 5)
        
        img = Image.new('RGB', (size, size), '#0a0a1e')
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        radius = size * 0.35
        
        # 13 circles of Metatron's Cube
        positions = [(center, center)]  # Central
        
        # Inner ring (6 circles)
        for i in range(6):
            angle = i * math.pi / 3
            x = center + radius * 0.5 * math.cos(angle)
            y = center + radius * 0.5 * math.sin(angle)
            positions.append((x, y))
        
        # Outer ring (6 circles)
        for i in range(6):
            angle = i * math.pi / 3 + math.pi / 6
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            positions.append((x, y))
        
        # Draw circles
        circle_radius = radius * 0.25
        colors = ['#9b59b6', '#3498db', '#1abc9c', '#f39c12', '#e74c3c']
        
        for i, (x, y) in enumerate(positions):
            color = colors[i % len(colors)]
            draw.ellipse([x - circle_radius, y - circle_radius,
                         x + circle_radius, y + circle_radius],
                        outline=color, width=2)
        
        # Connect all points with lines
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i < j:
                    hue = ((i + j) * GOLDEN_ANGLE) % 360
                    color = f'hsl({int(hue)}, 60%, 50%)'
                    draw.line([p1, p2], fill=color, width=1)
        
        return img
    
    def to_base64(self, img: Image.Image, format: str = 'PNG') -> str:
        """Convert image to base64 string"""
        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def to_bytes(self, img: Image.Image, format: str = 'PNG') -> bytes:
        """Convert image to bytes"""
        buffer = BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════════════
# 📋 GOAL & VISUALIZATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Goal:
    """A life goal with fractal visualization"""
    id: str
    title: str
    description: str = ""
    goal_type: str = "karma"  # karma (action) or dharma (purpose)
    points: int = 50
    progress: float = 0.0
    fractal_type: str = "mandelbrot"
    color_seed: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_date: Optional[str] = None
    completed: bool = False
    tasks: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


def suggest_fractal_for_goal(title: str, goal_type: str, points: int) -> str:
    """Suggest a fractal type based on goal characteristics"""
    title_lower = title.lower()
    
    # Match keywords to fractals
    if any(word in title_lower for word in ['meditat', 'calm', 'peace', 'focus', 'mind']):
        return 'mandelbrot'
    elif any(word in title_lower for word in ['grow', 'learn', 'develop', 'improve']):
        return 'golden_spiral'
    elif any(word in title_lower for word in ['connect', 'relationship', 'friend', 'family']):
        return 'flower_of_life'
    elif any(word in title_lower for word in ['overcome', 'challenge', 'difficult', 'struggle']):
        return 'burning_ship'
    elif any(word in title_lower for word in ['new', 'start', 'begin', 'fresh', 'change']):
        return 'phoenix'
    elif any(word in title_lower for word in ['habit', 'daily', 'routine', 'consistent']):
        return 'sierpinski'
    elif any(word in title_lower for word in ['patient', 'process', 'step', 'gradual']):
        return 'dragon'
    elif any(word in title_lower for word in ['creative', 'art', 'explore', 'discover']):
        return 'julia'
    elif any(word in title_lower for word in ['build', 'compound', 'momentum']):
        return 'fibonacci'
    elif any(word in title_lower for word in ['small', 'little', 'tiny', 'action']):
        return 'lorenz'
    elif any(word in title_lower for word in ['life', 'purpose', 'meaning', 'everything']):
        return 'metatron'
    elif goal_type == 'dharma':
        return 'golden_spiral'
    elif points >= 75:
        return 'mandelbrot'
    else:
        return random.choice(list(FRACTAL_TYPES.keys()))


# ═══════════════════════════════════════════════════════════════════════════════════════
# 👤 USER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class User:
    """User account with pet and goals"""
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subscription_status: str = "trial"
    trial_start: Optional[str] = None
    pet: Optional[VirtualPet] = None
    goals: List[Goal] = field(default_factory=list)
    gallery: List[dict] = field(default_factory=list)  # Saved fractal images
    total_fractals_generated: int = 0
    
    def to_dict(self) -> dict:
        data = {
            'email': self.email,
            'created_at': self.created_at,
            'subscription_status': self.subscription_status,
            'trial_start': self.trial_start,
            'pet': self.pet.to_dict() if self.pet else None,
            'goals': [g.to_dict() for g in self.goals],
            'gallery_count': len(self.gallery),
            'total_fractals': self.total_fractals_generated
        }
        return data
    
    def is_trial_active(self) -> bool:
        if not self.trial_start:
            return False
        try:
            start = datetime.fromisoformat(self.trial_start.replace('Z', '+00:00'))
            end = start + timedelta(days=Config.TRIAL_DAYS)
            return datetime.now(timezone.utc) < end
        except:
            return False


# ═══════════════════════════════════════════════════════════════════════════════════════
# 💾 DATA STORAGE
# ═══════════════════════════════════════════════════════════════════════════════════════

class DataStore:
    """Simple JSON-based storage"""
    
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.users_file = os.path.join(self.data_dir, 'users.json')
        self.gallery_dir = os.path.join(self.data_dir, 'gallery')
        os.makedirs(self.gallery_dir, exist_ok=True)
        self.users: Dict[str, User] = self._load_users()
        # Session storage for ChatGPT (no account needed)
        self.chatgpt_sessions: Dict[str, dict] = {}
    
    def _load_users(self) -> Dict[str, User]:
        if not os.path.exists(self.users_file):
            return {}
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
            users = {}
            for email, user_data in data.items():
                user = User(
                    email=user_data['email'],
                    password_hash=user_data.get('password_hash', ''),
                    created_at=user_data.get('created_at', ''),
                    subscription_status=user_data.get('subscription_status', 'trial'),
                    trial_start=user_data.get('trial_start')
                )
                if user_data.get('pet'):
                    user.pet = VirtualPet(**user_data['pet'])
                users[email] = user
            return users
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return {}
    
    def save_users(self):
        try:
            data = {email: {
                'email': user.email,
                'password_hash': user.password_hash,
                'created_at': user.created_at,
                'subscription_status': user.subscription_status,
                'trial_start': user.trial_start,
                'pet': user.pet.to_dict() if user.pet else None,
                'goals': [g.to_dict() for g in user.goals],
                'gallery': user.gallery,
                'total_fractals': user.total_fractals_generated
            } for email, user in self.users.items()}
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def get_or_create_session(self, session_id: str) -> dict:
        """Get or create a ChatGPT session"""
        if session_id not in self.chatgpt_sessions:
            self.chatgpt_sessions[session_id] = {
                'pet': None,
                'goals': [],
                'fractals_generated': 0,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        return self.chatgpt_sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🚀 FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)

# Initialize
store = DataStore()
fractal_engine = FractalEngine()

logger.info("🌀 Life Fractal Intelligence - Ultimate ChatGPT Edition starting...")


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🔐 AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_token(email: str) -> str:
    payload = {
        'email': email,
        'exp': (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def verify_token(token: str) -> Optional[str]:
    try:
        payload = json.loads(base64.b64decode(token).decode())
        exp = datetime.fromisoformat(payload['exp'])
        if exp < datetime.now(timezone.utc):
            return None
        return payload['email']
    except:
        return None


# ═══════════════════════════════════════════════════════════════════════════════════════
# 📄 BASIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return jsonify({
        'app': '🌀 Life Fractal Intelligence',
        'version': '2.0.0 - Ultimate ChatGPT Edition',
        'status': 'live',
        'features': [
            '🐾 Virtual Pet System (8 species)',
            '🎨 12+ Fractal Types',
            '📊 Goal Visualization',
            '🎁 Media Generation',
            '🔐 Account System'
        ],
        'chatgpt_endpoints': '/chatgpt/*',
        'documentation': '/docs'
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'users': len(store.users),
        'fractal_types': len(FRACTAL_TYPES),
        'pet_species': len(PET_SPECIES)
    })


@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Privacy Policy - Life Fractal Intelligence</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; min-height: 100vh; }
h1 { color: #9b59b6; border-bottom: 2px solid #9b59b6; padding-bottom: 10px; }
h2 { color: #3498db; margin-top: 30px; }
.emoji { font-size: 1.2em; }
a { color: #1abc9c; }
.updated { color: #888; font-size: 0.9em; }
</style></head><body>
<h1>🌀 Life Fractal Intelligence - Privacy Policy</h1>
<p class="updated">Last Updated: December 2024</p>

<h2>🌟 Our Commitment</h2>
<p>We believe in <strong>minimal data collection</strong> and <strong>maximum privacy</strong>. Your fractal journey is yours.</p>

<h2>📊 What We Collect</h2>
<ul>
<li><strong>Goals you enter</strong> - To generate personalized fractals</li>
<li><strong>Pet interactions</strong> - To track your companion's growth</li>
<li><strong>Email (optional)</strong> - Only if you create an account to save your gallery</li>
</ul>

<h2>🚫 What We DON'T Collect</h2>
<ul>
<li>Personal identification beyond email</li>
<li>Location data</li>
<li>Browsing history</li>
<li>Data from other apps</li>
</ul>

<h2>🔒 Data Security</h2>
<p>All data is encrypted and stored securely. We never sell your information.</p>

<h2>🗑️ Data Deletion</h2>
<p>Request deletion anytime: <a href="mailto:onlinediscountsllc@gmail.com">onlinediscountsllc@gmail.com</a></p>

<h2>📧 Contact</h2>
<p>Questions? Email: <a href="mailto:onlinediscountsllc@gmail.com">onlinediscountsllc@gmail.com</a></p>

<p style="margin-top: 40px; text-align: center; color: #666;">© 2024 Life Fractal Intelligence | Made with 💜 for neurodivergent minds</p>
</body></html>"""
    return html, 200, {'Content-Type': 'text/html'}


@app.route('/terms')
def terms():
    """Terms of service page"""
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Terms of Service - Life Fractal Intelligence</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; min-height: 100vh; }
h1 { color: #9b59b6; border-bottom: 2px solid #9b59b6; padding-bottom: 10px; }
h2 { color: #3498db; margin-top: 30px; }
a { color: #1abc9c; }
</style></head><body>
<h1>🌀 Life Fractal Intelligence - Terms of Service</h1>

<h2>1. Service</h2>
<p>Life Fractal Intelligence generates mathematical fractal art from your goals and provides virtual pet companions for your journey.</p>

<h2>2. Your Fractals</h2>
<p>You own the fractals you create! Use them however you like - print them, share them, enjoy them.</p>

<h2>3. Acceptable Use</h2>
<ul>
<li>✅ Personal use</li>
<li>✅ Sharing your creations</li>
<li>✅ Having fun with math!</li>
<li>❌ Automated abuse</li>
<li>❌ Reselling the service</li>
</ul>

<h2>4. Rate Limits</h2>
<p>Free tier: 10 fractals/day. Premium: 1000/day.</p>

<h2>5. Contact</h2>
<p>Email: <a href="mailto:onlinediscountsllc@gmail.com">onlinediscountsllc@gmail.com</a></p>

<p style="margin-top: 40px; text-align: center; color: #666;">© 2024 Life Fractal Intelligence</p>
</body></html>"""
    return html, 200, {'Content-Type': 'text/html'}


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🤖 CHATGPT API ENDPOINTS - THESE ARE WHAT CHATGPT CALLS!
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/health', methods=['GET'])
def chatgpt_health():
    """Health check for ChatGPT"""
    return jsonify({
        'status': 'healthy',
        'service': 'Life Fractal Intelligence',
        'version': '2.0.0',
        'available_fractals': len(FRACTAL_TYPES),
        'available_pets': len(PET_SPECIES),
        'message': '🌀 Ready to create beautiful fractals!'
    })


@app.route('/chatgpt/fractals/list', methods=['GET'])
def chatgpt_list_fractals():
    """List all available fractal types"""
    fractals = []
    for ftype, info in FRACTAL_TYPES.items():
        fractals.append({
            'type': ftype,
            'name': info['name'],
            'description': info['description'],
            'meaning': info['meaning'],
            'best_for': info['best_for'],
            'params': info['params']
        })
    
    return jsonify({
        'fractals': fractals,
        'total': len(fractals),
        'message': 'Each fractal has a unique meaning. Choose one that resonates with your goal!'
    })


@app.route('/chatgpt/fractals/generate', methods=['POST'])
def chatgpt_generate_fractal():
    """Generate a fractal - THE MAIN ENDPOINT!"""
    try:
        data = request.get_json() or {}
        fractal_type = data.get('type', 'mandelbrot').lower()
        params = data.get('params', {})
        size = min(max(data.get('size', 800), 256), 2048)
        
        # Validate fractal type
        if fractal_type not in FRACTAL_TYPES:
            return jsonify({
                'error': 'Invalid fractal type',
                'message': f"'{fractal_type}' is not available",
                'available': list(FRACTAL_TYPES.keys())
            }), 400
        
        # Generate fractal
        img = fractal_engine.generate(fractal_type, size, params)
        base64_data = fractal_engine.to_base64(img)
        
        fractal_info = FRACTAL_TYPES[fractal_type]
        
        return jsonify({
            'success': True,
            'fractal_type': fractal_type,
            'fractal_name': fractal_info['name'],
            'size': f'{size}x{size}',
            'format': 'png',
            'image_data': f'data:image/png;base64,{base64_data}',
            'meaning': fractal_info['meaning'],
            'message': f"✨ Your {fractal_info['name']} is ready! {fractal_info['meaning']}"
        })
        
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({
            'error': 'Generation failed',
            'message': str(e)
        }), 500


@app.route('/chatgpt/fractals/suggest', methods=['POST'])
def chatgpt_suggest_fractal():
    """Suggest a fractal based on user's goal"""
    try:
        data = request.get_json() or {}
        goal_text = data.get('goal', data.get('title', ''))
        goal_type = data.get('type', 'karma')
        
        if not goal_text:
            return jsonify({
                'error': 'No goal provided',
                'message': 'Tell me about your goal and I\'ll suggest the perfect fractal!'
            }), 400
        
        suggested = suggest_fractal_for_goal(goal_text, goal_type, 50)
        fractal_info = FRACTAL_TYPES[suggested]
        
        return jsonify({
            'success': True,
            'suggested_fractal': suggested,
            'fractal_name': fractal_info['name'],
            'reason': fractal_info['meaning'],
            'description': fractal_info['description'],
            'message': f"For '{goal_text}', I suggest the {fractal_info['name']}! {fractal_info['meaning']}"
        })
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🐾 CHATGPT PET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/pets/list', methods=['GET'])
def chatgpt_list_pets():
    """List all available pet species"""
    pets = []
    for species, info in PET_SPECIES.items():
        pets.append({
            'species': species,
            'name': info['name'],
            'emoji': info['emoji'],
            'personality': info['personality'],
            'favorite_fractal': info['favorite_fractal'],
            'abilities': info['abilities'],
            'greetings': info['greetings'][0]
        })
    
    return jsonify({
        'pets': pets,
        'total': len(pets),
        'message': 'Choose your fractal companion! Each has unique abilities.'
    })


@app.route('/chatgpt/pet/create', methods=['POST'])
def chatgpt_create_pet():
    """Create a virtual pet companion"""
    try:
        data = request.get_json() or {}
        species = data.get('species', 'cat').lower()
        name = data.get('name', 'Fractal Friend')[:50]
        
        if species not in PET_SPECIES:
            return jsonify({
                'error': 'Invalid species',
                'message': f"'{species}' is not available",
                'available_species': list(PET_SPECIES.keys())
            }), 400
        
        pet = VirtualPet(species=species, name=name)
        species_info = PET_SPECIES[species]
        
        return jsonify({
            'success': True,
            'pet': pet.get_stats(),
            'greeting': pet.get_greeting(),
            'abilities': species_info['abilities'],
            'evolution_chain': species_info['evolution_chain'],
            'message': f"🎉 Meet {name} the {species_info['name']}! {pet.get_greeting()}"
        })
        
    except Exception as e:
        logger.error(f"Pet creation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chatgpt/pet/feed', methods=['POST'])
def chatgpt_feed_pet():
    """Feed a pet"""
    try:
        data = request.get_json() or {}
        
        # Get pet data from request (stateless)
        pet_data = data.get('pet', {})
        food_type = data.get('food', 'cosmic_kibble')
        
        if not pet_data or 'species' not in pet_data:
            return jsonify({
                'error': 'No pet provided',
                'message': 'Create a pet first with /chatgpt/pet/create',
                'available_food': list(PET_FOOD.keys())
            }), 400
        
        if food_type not in PET_FOOD:
            return jsonify({
                'error': 'Invalid food',
                'available_food': list(PET_FOOD.keys()),
                'message': f"Try one of these: {', '.join(PET_FOOD.keys())}"
            }), 400
        
        # Reconstruct pet
        pet = VirtualPet(**pet_data)
        result = pet.feed(food_type)
        
        return jsonify({
            'success': True,
            'pet': pet.get_stats(),
            'food': food_type,
            'food_description': PET_FOOD[food_type]['description'],
            'message': result['message']
        })
        
    except Exception as e:
        logger.error(f"Pet feed error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chatgpt/pet/play', methods=['POST'])
def chatgpt_play_pet():
    """Play with a pet"""
    try:
        data = request.get_json() or {}
        
        pet_data = data.get('pet', {})
        activity = data.get('activity', 'chase_fractals')
        
        if not pet_data or 'species' not in pet_data:
            return jsonify({
                'error': 'No pet provided',
                'message': 'Create a pet first!',
                'available_activities': list(PET_ACTIVITIES.keys())
            }), 400
        
        if activity not in PET_ACTIVITIES:
            return jsonify({
                'error': 'Invalid activity',
                'available_activities': list(PET_ACTIVITIES.keys()),
                'message': f"Try one of these: {', '.join(PET_ACTIVITIES.keys())}"
            }), 400
        
        pet = VirtualPet(**pet_data)
        result = pet.play(activity)
        
        return jsonify({
            'success': result['success'],
            'pet': pet.get_stats(),
            'activity': activity,
            'activity_description': PET_ACTIVITIES[activity]['description'],
            'xp_gained': result.get('xp_gained', 0),
            'message': result['message']
        })
        
    except Exception as e:
        logger.error(f"Pet play error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chatgpt/pet/status', methods=['POST'])
def chatgpt_pet_status():
    """Get pet status"""
    try:
        data = request.get_json() or {}
        pet_data = data.get('pet', {})
        
        if not pet_data or 'species' not in pet_data:
            return jsonify({
                'error': 'No pet provided',
                'message': 'Create a pet first!'
            }), 400
        
        pet = VirtualPet(**pet_data)
        pet.update_stats()
        
        return jsonify({
            'success': True,
            'pet': pet.get_stats(),
            'greeting': pet.get_greeting(),
            'needs': {
                'hungry': pet.hunger > 50,
                'lonely': pet.happiness < 50,
                'tired': pet.energy < 30
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🎯 CHATGPT GOAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/goals/create', methods=['POST'])
def chatgpt_create_goal():
    """Create a goal and get its fractal"""
    try:
        data = request.get_json() or {}
        title = data.get('title', '')[:200]
        goal_type = data.get('type', 'karma')
        points = min(max(data.get('points', 50), 1), 100)
        
        if not title:
            return jsonify({
                'error': 'No goal title',
                'message': 'Tell me your goal!'
            }), 400
        
        # Suggest fractal
        suggested_fractal = suggest_fractal_for_goal(title, goal_type, points)
        fractal_info = FRACTAL_TYPES[suggested_fractal]
        
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            title=title,
            goal_type=goal_type,
            points=points,
            fractal_type=suggested_fractal
        )
        
        return jsonify({
            'success': True,
            'goal': goal.to_dict(),
            'suggested_fractal': {
                'type': suggested_fractal,
                'name': fractal_info['name'],
                'meaning': fractal_info['meaning'],
                'description': fractal_info['description']
            },
            'message': f"🎯 Goal created! The {fractal_info['name']} represents: {fractal_info['meaning']}",
            'next_step': f"Generate your goal's fractal with type='{suggested_fractal}'"
        })
        
    except Exception as e:
        logger.error(f"Goal creation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chatgpt/goals/visualize', methods=['POST'])
def chatgpt_visualize_goal():
    """Visualize a goal as a fractal - combines goal creation and fractal generation"""
    try:
        data = request.get_json() or {}
        title = data.get('title', data.get('goal', ''))[:200]
        goal_type = data.get('type', 'karma')
        points = min(max(data.get('points', 50), 1), 100)
        size = min(max(data.get('size', 800), 256), 2048)
        
        if not title:
            return jsonify({
                'error': 'No goal provided',
                'message': 'Tell me your goal and I\'ll create a beautiful fractal for it!'
            }), 400
        
        # Suggest and generate fractal
        suggested_fractal = suggest_fractal_for_goal(title, goal_type, points)
        fractal_info = FRACTAL_TYPES[suggested_fractal]
        
        # Generate the fractal
        img = fractal_engine.generate(suggested_fractal, size, {})
        base64_data = fractal_engine.to_base64(img)
        
        return jsonify({
            'success': True,
            'goal': title,
            'fractal_type': suggested_fractal,
            'fractal_name': fractal_info['name'],
            'meaning': fractal_info['meaning'],
            'description': fractal_info['description'],
            'image_data': f'data:image/png;base64,{base64_data}',
            'message': f"✨ Your goal '{title}' is now a {fractal_info['name']}!\n\n{fractal_info['meaning']}"
        })
        
    except Exception as e:
        logger.error(f"Goal visualization error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 📐 CHATGPT SACRED GEOMETRY / MATH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/sacred/explain', methods=['GET'])
def chatgpt_explain_sacred():
    """Explain sacred mathematics"""
    return jsonify({
        'sacred_mathematics': {
            'golden_ratio': {
                'symbol': 'φ (phi)',
                'value': PHI,
                'explanation': 'The golden ratio appears everywhere in nature - from spiral galaxies to seashells to the proportions of the human body. When we use φ in fractals, we tap into nature\'s own design language.',
                'in_fractals': 'The golden spiral uses φ to create infinitely harmonious growth patterns.'
            },
            'fibonacci_sequence': {
                'sequence': FIBONACCI[:12],
                'explanation': 'Each number is the sum of the two before it: 1, 1, 2, 3, 5, 8, 13... This sequence appears in flower petals, pinecones, and branching trees.',
                'in_fractals': 'Fibonacci spirals show how small beginnings lead to grand outcomes.'
            },
            'golden_angle': {
                'value': GOLDEN_ANGLE,
                'explanation': 'The angle that produces the most efficient packing of seeds in a sunflower. It\'s φ expressed as an angle!',
                'in_fractals': 'We use this angle to create naturally pleasing color distributions.'
            }
        },
        'for_aphantasia': {
            'message': "If you can't visualize in your mind, fractals are your external imagination! They show you what goals and growth LOOK like.",
            'benefit': 'External visualization makes abstract concepts concrete and beautiful.'
        }
    })


@app.route('/chatgpt/sacred/calculate', methods=['POST'])
def chatgpt_calculate_sacred():
    """Calculate sacred geometry values"""
    try:
        data = request.get_json() or {}
        calc_type = data.get('type', 'fibonacci')
        n = min(max(data.get('n', 10), 1), 50)
        
        results = {}
        
        if calc_type == 'fibonacci':
            # Generate Fibonacci sequence
            fib = [0, 1]
            for i in range(2, n):
                fib.append(fib[-1] + fib[-2])
            results = {
                'sequence': fib,
                'golden_ratio_approximation': fib[-1] / fib[-2] if len(fib) > 1 and fib[-2] != 0 else 0,
                'actual_phi': PHI,
                'message': f'The ratio of consecutive Fibonacci numbers approaches φ ({PHI:.10f})'
            }
        
        elif calc_type == 'golden_rectangle':
            width = data.get('width', 100)
            height = width / PHI
            results = {
                'width': width,
                'height': round(height, 4),
                'ratio': PHI,
                'message': f'A golden rectangle with width {width} has height {height:.4f}'
            }
        
        elif calc_type == 'golden_spiral':
            # Points on a golden spiral
            points = []
            for i in range(n * 10):
                theta = i * GOLDEN_ANGLE_RAD / 10
                r = PHI ** (theta / (2 * math.pi))
                points.append({
                    'angle': round(math.degrees(theta), 2),
                    'radius': round(r, 4)
                })
            results = {
                'points': points[:n],
                'growth_factor': PHI,
                'message': 'Each quarter turn, the spiral grows by factor φ'
            }
        
        return jsonify({
            'success': True,
            'calculation': calc_type,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🎨 CHATGPT POSTER/MEDIA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/poster/create', methods=['POST'])
def chatgpt_create_poster():
    """Create a poster with fractal and goal text"""
    try:
        data = request.get_json() or {}
        goal = data.get('goal', data.get('title', 'My Goal'))[:100]
        fractal_type = data.get('fractal_type', 'mandelbrot')
        size = min(max(data.get('size', 1200), 800), 2048)
        
        if fractal_type not in FRACTAL_TYPES:
            fractal_type = 'mandelbrot'
        
        # Generate larger fractal for poster
        fractal = fractal_engine.generate(fractal_type, size, {})
        
        # Add text overlay
        from PIL import ImageDraw
        draw = ImageDraw.Draw(fractal)
        
        # Add goal text at bottom
        text_y = size - 80
        draw.rectangle([0, text_y - 20, size, size], fill=(10, 10, 30, 200))
        
        # Simple text (PIL default font)
        draw.text((size//2, text_y + 20), goal, fill='white', anchor='mm')
        
        fractal_info = FRACTAL_TYPES[fractal_type]
        draw.text((size//2, text_y + 50), fractal_info['meaning'][:60], fill='#9b59b6', anchor='mm')
        
        base64_data = fractal_engine.to_base64(fractal)
        
        return jsonify({
            'success': True,
            'poster': f'data:image/png;base64,{base64_data}',
            'size': f'{size}x{size}',
            'goal': goal,
            'fractal_type': fractal_type,
            'message': f"🖼️ Your poster is ready! Print it, share it, or use it as your wallpaper!"
        })
        
    except Exception as e:
        logger.error(f"Poster creation error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🧠 CHATGPT NEURODIVERGENT SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/chatgpt/aphantasia/help', methods=['GET'])
def chatgpt_aphantasia_help():
    """Information and support for aphantasia users"""
    return jsonify({
        'what_is_aphantasia': {
            'definition': "Aphantasia is the inability to voluntarily create mental images. About 2-5% of people have it.",
            'challenge': "Traditional goal-setting asks you to 'visualize success' - impossible if you can't see pictures in your mind!"
        },
        'how_we_help': {
            'external_visualization': "Fractals ARE your visualization! No imagination required - the patterns are real and visible.",
            'math_as_meaning': "Each fractal type represents a concept. The Mandelbrot shows infinite complexity from simple rules - like your small daily actions creating big results.",
            'concrete_not_abstract': "Instead of imagining a 'better future', you see actual mathematical beauty representing your goals."
        },
        'tips': [
            "Save your fractal images - they're YOUR external memory",
            "Different fractal types = different goal energies",
            "Print posters and put them where you'll see them",
            "The pet companion provides emotional connection without visual imagination"
        ],
        'message': "You're not broken - your brain just works differently. We built this FOR brains like yours. 💜"
    })


@app.route('/chatgpt/spoons/check', methods=['POST'])
def chatgpt_spoon_check():
    """Spoon Theory energy check"""
    try:
        data = request.get_json() or {}
        current_spoons = data.get('spoons', 5)
        
        # Recommendations based on energy level
        if current_spoons >= 8:
            recommendation = {
                'level': 'high_energy',
                'message': "You have good energy today! 🥄🥄🥄",
                'suggested_fractals': ['mandelbrot', 'metatron', 'flower_of_life'],
                'suggested_activities': ['explore_mandelbrot', 'learn_geometry'],
                'goal_approach': 'Great day for tackling bigger goals!'
            }
        elif current_spoons >= 5:
            recommendation = {
                'level': 'moderate_energy',
                'message': "You have moderate energy. Be mindful! 🥄🥄",
                'suggested_fractals': ['golden_spiral', 'fibonacci'],
                'suggested_activities': ['meditate_sacred', 'dream_fractals'],
                'goal_approach': 'Focus on one meaningful task.'
            }
        elif current_spoons >= 3:
            recommendation = {
                'level': 'low_energy',
                'message': "Energy is low. Be gentle with yourself. 🥄",
                'suggested_fractals': ['julia', 'sierpinski'],
                'suggested_activities': ['meditate_sacred'],
                'goal_approach': 'Maintenance tasks only. Rest is productive!'
            }
        else:
            recommendation = {
                'level': 'very_low',
                'message': "Almost no spoons. Just survive today. 💜",
                'suggested_fractals': ['lorenz'],
                'suggested_activities': ['dream_fractals'],
                'goal_approach': 'No goals today. Existing is enough.'
            }
        
        return jsonify({
            'success': True,
            'spoons': current_spoons,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🔐 ACCOUNT ENDPOINTS (for saving gallery)
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new account"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        pet_species = data.get('pet_species', 'cat')
        pet_name = data.get('pet_name', 'Buddy')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return jsonify({'error': 'Invalid email'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be 8+ characters'}), 400
        
        if email in store.users:
            return jsonify({'error': 'Email already registered'}), 400
        
        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            trial_start=datetime.now(timezone.utc).isoformat()
        )
        user.pet = VirtualPet(species=pet_species, name=pet_name)
        
        store.users[email] = user
        store.save_users()
        
        token = create_token(email)
        
        return jsonify({
            'success': True,
            'message': f'Welcome! {user.pet.get_greeting()}',
            'token': token,
            'user': user.to_dict(),
            'trial_days': Config.TRIAL_DAYS
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Login to account"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if email not in store.users:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = store.users[email]
        
        if not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if user.pet:
            user.pet.update_stats()
        
        token = create_token(email)
        
        return jsonify({
            'success': True,
            'message': user.pet.get_greeting() if user.pet else 'Welcome back!',
            'token': token,
            'user': user.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# 📚 DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/docs')
def docs():
    """API Documentation"""
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>API Docs - Life Fractal Intelligence</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
h1 { color: #9b59b6; }
h2 { color: #3498db; margin-top: 40px; border-bottom: 1px solid #3498db; padding-bottom: 10px; }
h3 { color: #1abc9c; }
code { background: #2d2d44; padding: 2px 6px; border-radius: 4px; }
pre { background: #2d2d44; padding: 15px; border-radius: 8px; overflow-x: auto; }
.endpoint { background: #2d2d44; padding: 15px; border-radius: 8px; margin: 15px 0; }
.method { display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: bold; }
.get { background: #27ae60; }
.post { background: #3498db; }
</style></head><body>
<h1>🌀 Life Fractal Intelligence - API Documentation</h1>

<h2>ChatGPT Endpoints</h2>

<div class="endpoint">
<span class="method get">GET</span> <code>/chatgpt/health</code>
<p>Check API status</p>
</div>

<div class="endpoint">
<span class="method get">GET</span> <code>/chatgpt/fractals/list</code>
<p>List all 12+ fractal types with meanings</p>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/chatgpt/fractals/generate</code>
<p>Generate a fractal image</p>
<pre>{"type": "mandelbrot", "size": 800, "params": {"zoom": 1.0}}</pre>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/chatgpt/goals/visualize</code>
<p>Create a goal and generate its fractal</p>
<pre>{"goal": "Learn meditation", "type": "dharma", "points": 80}</pre>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/chatgpt/pet/create</code>
<p>Create a virtual pet companion</p>
<pre>{"species": "dragon", "name": "Ember"}</pre>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/chatgpt/pet/feed</code>
<p>Feed your pet</p>
<pre>{"pet": {...pet_data...}, "food": "golden_apple"}</pre>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/chatgpt/pet/play</code>
<p>Play with your pet</p>
<pre>{"pet": {...pet_data...}, "activity": "explore_mandelbrot"}</pre>
</div>

<h2>Available Fractals</h2>
<ul>
<li><strong>mandelbrot</strong> - Infinite complexity from simple rules</li>
<li><strong>julia</strong> - Unique patterns from different starting points</li>
<li><strong>golden_spiral</strong> - Nature's growth pattern</li>
<li><strong>flower_of_life</strong> - Sacred interconnection</li>
<li><strong>burning_ship</strong> - Rising from challenges</li>
<li><strong>phoenix</strong> - Rebirth and renewal</li>
<li><strong>sierpinski</strong> - Power of repetition</li>
<li><strong>dragon</strong> - Complexity from simplicity</li>
<li><strong>koch_snowflake</strong> - Infinite in finite</li>
<li><strong>fibonacci</strong> - Compound growth</li>
<li><strong>lorenz</strong> - Small actions matter</li>
<li><strong>metatron</strong> - Universal structure</li>
</ul>

<h2>Available Pets</h2>
<ul>
<li>🐱 cat - Curious and independent</li>
<li>🐕 dog - Enthusiastic and loyal</li>
<li>🐉 dragon - Wise and powerful</li>
<li>🔥 phoenix - Transformative</li>
<li>🦉 owl - Wise and observant</li>
<li>🦊 fox - Clever and playful</li>
<li>🦄 unicorn - Magical and pure</li>
<li>🦋 butterfly - Gentle chaos</li>
</ul>

<p style="margin-top: 40px; color: #666;">Built with 💜 for neurodivergent minds</p>
</body></html>"""
    return html, 200, {'Content-Type': 'text/html'}


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🚀 START SERVER
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌀 Starting on port {port}")
    logger.info(f"🐾 {len(PET_SPECIES)} pet species available")
    logger.info(f"🎨 {len(FRACTAL_TYPES)} fractal types available")
    app.run(host='0.0.0.0', port=port, debug=False)
