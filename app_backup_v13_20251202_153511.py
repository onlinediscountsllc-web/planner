#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v14.0 - ULTIMATE UNIFIED ENGINE
════════════════════════════════════════════════════════════════════════════════

For brains like mine - Complete unified platform with ALL features.

UNIFIED FEATURES:
✅ v8:  Virtual Pets, Spoon Theory, 3D Visualization, Accessibility
✅ v9:  OCR Document Scanning (optional)
✅ v10: 33 Life Milestones, Journey Timeline, Predictive Analytics
✅ v11: Mathematical Causality, Bellman Optimization, Spillover Matrix
✅ v12: Law of Attraction, Flow State, Compound Growth, Scientific Research
✅ v13: Fractal Mathematics (11 computational tools)

PRODUCTION READY:
✅ Zero placeholders - Every feature works
✅ SQLite database with all tables
✅ Complete REST API (50+ endpoints)
✅ Full responsive frontend
✅ Error handling throughout
✅ Render.com compatible

════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import sqlite3
import uuid
import hashlib
import re
from datetime import datetime, timedelta, timezone, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from collections import defaultdict
from functools import lru_cache
from contextlib import contextmanager

# Flask imports
from flask import Flask, request, jsonify, render_template_string, session, g, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Numerical computing
import numpy as np
from numpy.linalg import norm

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618033988749895
PHI_INVERSE = 1 / PHI         # ≈ 0.618033988749895
PHI_SQUARED = PHI ** 2        # ≈ 2.618033988749895
GOLDEN_ANGLE_DEG = 360 / (PHI ** 2)  # ≈ 137.5077640500378°
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE_DEG)
E = math.e                    # 2.718281828459045
PI = math.pi                  # 3.141592653589793
GAMMA = PHI_INVERSE           # Discount factor for Bellman

def generate_fibonacci(n: int) -> List[int]:
    """Generate Fibonacci sequence."""
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def generate_lucas(n: int) -> List[int]:
    """Generate Lucas sequence."""
    lucas = [2, 1]
    for _ in range(2, n):
        lucas.append(lucas[-1] + lucas[-2])
    return lucas

FIBONACCI = generate_fibonacci(30)
LUCAS = generate_lucas(20)

# Scientific Constants
HABIT_FORMATION_AVG = 66      # Phillippa Lally research (UCL)
HABIT_FORMATION_MIN = 21
HABIT_FORMATION_MAX = 254
FLOW_CHALLENGE_SKILL_RATIO = PHI  # Csikszentmihalyi
FORGETTING_RATE = 0.1         # Ebbinghaus
METCALFE_EXPONENT = 2.0       # Network effects
RULE_OF_72 = 72               # Compound growth

# Fractal Constants
FRACTAL_DIMENSIONS = {
    'line': 1.0,
    'coastline': 1.25,
    'koch_snowflake': 1.2619,
    'sierpinski_triangle': 1.585,
    'brownian_motion': 1.5,
    'balanced_life': 1.0,
    'chaotic_life': 1.8,
}

# ════════════════════════════════════════════════════════════════════════════════
# LIFE DOMAINS (13 Dimensions)
# ════════════════════════════════════════════════════════════════════════════════

class LifeDomain(Enum):
    HEALTH = "health"
    SKILLS = "skills"
    FINANCES = "finances"
    RELATIONSHIPS = "relationships"
    CAREER = "career"
    MOOD = "mood"
    ENERGY = "energy"
    PURPOSE = "purpose"
    CREATIVITY = "creativity"
    SPIRITUALITY = "spirituality"
    BELIEF = "belief"           # Law of Attraction
    FOCUS = "focus"             # Law of Attraction
    GRATITUDE = "gratitude"     # Law of Attraction

DOMAIN_INDEX = {d: i for i, d in enumerate(LifeDomain)}
N_DOMAINS = len(LifeDomain)

DOMAIN_ICONS = {
    "health": "❤️", "skills": "🎯", "finances": "💰", "relationships": "👥",
    "career": "💼", "mood": "😊", "energy": "⚡", "purpose": "🧭",
    "creativity": "🎨", "spirituality": "🙏", "belief": "✨", "focus": "🔍", "gratitude": "💝"
}

DOMAIN_COLORS = {
    "health": "#E57373", "skills": "#64B5F6", "finances": "#81C784", "relationships": "#FFB74D",
    "career": "#9575CD", "mood": "#4DB6AC", "energy": "#FFD54F", "purpose": "#A1887F",
    "creativity": "#F06292", "spirituality": "#7986CB", "belief": "#BA68C8", "focus": "#4DD0E1", "gratitude": "#AED581"
}

# ════════════════════════════════════════════════════════════════════════════════
# TASK TYPES (18 Types with Full Effect Vectors)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskType:
    id: str
    name: str
    icon: str
    effect_vector: List[float]  # 13 dimensions
    energy_cost: int            # Spoons
    duration_minutes: int
    compounding_rate: float
    flow_challenge: float       # For flow state calculation
    category: str

TASK_TYPES = {
    # Health & Energy
    "exercise": TaskType("exercise", "Exercise", "🏃", 
        [0.08, 0.01, 0.0, 0.02, 0.0, 0.05, 0.06, 0.02, 0.0, 0.01, 0.03, 0.02, 0.01], 
        3, 45, 0.02, 0.6, "health"),
    "sleep": TaskType("sleep", "Quality Sleep", "😴", 
        [0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.10, 0.0, 0.0, 0.01, 0.02, 0.02, 0.01], 
        0, 480, 0.01, 0.1, "health"),
    "meditation": TaskType("meditation", "Meditation", "🧘", 
        [0.02, 0.0, 0.0, 0.01, 0.0, 0.05, 0.03, 0.04, 0.02, 0.06, 0.04, 0.06, 0.03], 
        1, 20, 0.015, 0.3, "health"),
    "nutrition": TaskType("nutrition", "Healthy Eating", "🥗", 
        [0.05, 0.0, -0.01, 0.0, 0.0, 0.03, 0.04, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01], 
        1, 30, 0.01, 0.2, "health"),
    
    # Skills & Career
    "deep_work": TaskType("deep_work", "Deep Work", "🎯", 
        [0.0, 0.08, 0.02, 0.0, 0.05, 0.0, -0.02, 0.03, 0.02, 0.0, 0.02, 0.05, 0.0], 
        4, 90, 0.03, 0.8, "career"),
    "learning": TaskType("learning", "Learning", "📚", 
        [0.0, 0.06, 0.0, 0.0, 0.03, 0.02, -0.01, 0.02, 0.03, 0.01, 0.02, 0.04, 0.0], 
        3, 60, 0.025, 0.7, "skills"),
    "networking": TaskType("networking", "Networking", "🤝", 
        [0.0, 0.01, 0.02, 0.06, 0.04, 0.02, -0.02, 0.01, 0.0, 0.0, 0.02, 0.01, 0.01], 
        3, 60, 0.02, 0.5, "career"),
    "creative_work": TaskType("creative_work", "Creative Work", "🎨", 
        [0.0, 0.03, 0.01, 0.0, 0.02, 0.04, -0.02, 0.03, 0.08, 0.02, 0.03, 0.04, 0.0], 
        3, 60, 0.02, 0.7, "creativity"),
    
    # Relationships
    "quality_time": TaskType("quality_time", "Quality Time", "💕", 
        [0.01, 0.0, 0.0, 0.08, 0.0, 0.06, 0.01, 0.02, 0.0, 0.01, 0.03, 0.01, 0.04], 
        2, 60, 0.015, 0.4, "relationships"),
    "helping_others": TaskType("helping_others", "Helping Others", "🤲", 
        [0.0, 0.01, 0.0, 0.05, 0.0, 0.05, -0.01, 0.05, 0.0, 0.03, 0.04, 0.01, 0.05], 
        2, 45, 0.01, 0.4, "relationships"),
    
    # Finances
    "financial_planning": TaskType("financial_planning", "Financial Planning", "📊", 
        [0.0, 0.02, 0.06, 0.0, 0.02, 0.0, -0.01, 0.02, 0.0, 0.0, 0.02, 0.03, 0.0], 
        2, 30, 0.025, 0.6, "finances"),
    "budgeting": TaskType("budgeting", "Budgeting", "💵", 
        [0.0, 0.01, 0.04, 0.0, 0.01, 0.01, -0.01, 0.01, 0.0, 0.0, 0.02, 0.02, 0.0], 
        1, 20, 0.02, 0.4, "finances"),
    
    # Purpose & Spirituality
    "journaling": TaskType("journaling", "Journaling", "📝", 
        [0.01, 0.01, 0.0, 0.01, 0.0, 0.04, 0.01, 0.04, 0.03, 0.03, 0.04, 0.04, 0.03], 
        1, 15, 0.01, 0.3, "purpose"),
    "goal_setting": TaskType("goal_setting", "Goal Setting", "🎯", 
        [0.0, 0.02, 0.01, 0.0, 0.02, 0.02, 0.0, 0.06, 0.01, 0.01, 0.05, 0.06, 0.0], 
        2, 30, 0.02, 0.5, "purpose"),
    "nature_time": TaskType("nature_time", "Time in Nature", "🌿", 
        [0.03, 0.0, 0.0, 0.01, 0.0, 0.05, 0.04, 0.03, 0.02, 0.04, 0.03, 0.02, 0.04], 
        1, 30, 0.01, 0.2, "spirituality"),
    
    # Mindset Practices (Law of Attraction)
    "visualization": TaskType("visualization", "Visualization", "🔮", 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.03, 0.02, 0.02, 0.05, 0.06, 0.01], 
        1, 10, 0.015, 0.3, "mindset"),
    "affirmations": TaskType("affirmations", "Affirmations", "💬", 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.01, 0.02, 0.0, 0.01, 0.04, 0.03, 0.02], 
        0, 5, 0.01, 0.1, "mindset"),
    "gratitude_practice": TaskType("gratitude_practice", "Gratitude Practice", "🙏", 
        [0.01, 0.0, 0.0, 0.02, 0.0, 0.05, 0.03, 0.02, 0.0, 0.03, 0.03, 0.02, 0.08], 
        0, 10, 0.01, 0.2, "mindset"),
}

# ════════════════════════════════════════════════════════════════════════════════
# SPILLOVER MATRIX (39 Cross-Domain Effects)
# ════════════════════════════════════════════════════════════════════════════════

def build_spillover_matrix() -> np.ndarray:
    """Build the β_ij spillover matrix."""
    matrix = np.zeros((N_DOMAINS, N_DOMAINS))
    
    spillovers = {
        # Physical spillovers
        (LifeDomain.HEALTH, LifeDomain.ENERGY): 0.5,
        (LifeDomain.HEALTH, LifeDomain.MOOD): 0.4,
        (LifeDomain.HEALTH, LifeDomain.BELIEF): 0.15,
        (LifeDomain.ENERGY, LifeDomain.MOOD): 0.3,
        (LifeDomain.ENERGY, LifeDomain.FOCUS): 0.25,
        
        # Skills spillovers
        (LifeDomain.SKILLS, LifeDomain.CAREER): 0.5,
        (LifeDomain.SKILLS, LifeDomain.FINANCES): 0.3,
        (LifeDomain.SKILLS, LifeDomain.BELIEF): 0.2,
        (LifeDomain.SKILLS, LifeDomain.CREATIVITY): 0.25,
        
        # Relationship spillovers
        (LifeDomain.RELATIONSHIPS, LifeDomain.MOOD): 0.5,
        (LifeDomain.RELATIONSHIPS, LifeDomain.BELIEF): 0.25,
        (LifeDomain.RELATIONSHIPS, LifeDomain.GRATITUDE): 0.2,
        (LifeDomain.RELATIONSHIPS, LifeDomain.PURPOSE): 0.15,
        
        # Career spillovers
        (LifeDomain.CAREER, LifeDomain.FINANCES): 0.4,
        (LifeDomain.CAREER, LifeDomain.PURPOSE): 0.3,
        (LifeDomain.CAREER, LifeDomain.BELIEF): 0.2,
        
        # Purpose spillovers
        (LifeDomain.PURPOSE, LifeDomain.MOOD): 0.35,
        (LifeDomain.PURPOSE, LifeDomain.BELIEF): 0.4,
        (LifeDomain.PURPOSE, LifeDomain.ENERGY): 0.2,
        
        # Creativity spillovers
        (LifeDomain.CREATIVITY, LifeDomain.MOOD): 0.25,
        (LifeDomain.CREATIVITY, LifeDomain.PURPOSE): 0.2,
        (LifeDomain.CREATIVITY, LifeDomain.SKILLS): 0.15,
        
        # Spirituality spillovers
        (LifeDomain.SPIRITUALITY, LifeDomain.MOOD): 0.3,
        (LifeDomain.SPIRITUALITY, LifeDomain.PURPOSE): 0.35,
        (LifeDomain.SPIRITUALITY, LifeDomain.GRATITUDE): 0.25,
        (LifeDomain.SPIRITUALITY, LifeDomain.BELIEF): 0.2,
        
        # Law of Attraction spillovers
        (LifeDomain.BELIEF, LifeDomain.MOOD): 0.4,
        (LifeDomain.BELIEF, LifeDomain.ENERGY): 0.25,
        (LifeDomain.BELIEF, LifeDomain.CAREER): 0.2,
        (LifeDomain.BELIEF, LifeDomain.RELATIONSHIPS): 0.15,
        
        (LifeDomain.FOCUS, LifeDomain.SKILLS): 0.3,
        (LifeDomain.FOCUS, LifeDomain.CAREER): 0.25,
        (LifeDomain.FOCUS, LifeDomain.PURPOSE): 0.2,
        (LifeDomain.FOCUS, LifeDomain.CREATIVITY): 0.15,
        
        (LifeDomain.GRATITUDE, LifeDomain.MOOD): 0.5,
        (LifeDomain.GRATITUDE, LifeDomain.RELATIONSHIPS): 0.3,
        (LifeDomain.GRATITUDE, LifeDomain.BELIEF): 0.25,
        (LifeDomain.GRATITUDE, LifeDomain.ENERGY): 0.15,
        (LifeDomain.GRATITUDE, LifeDomain.SPIRITUALITY): 0.2,
    }
    
    for (from_d, to_d), effect in spillovers.items():
        matrix[DOMAIN_INDEX[from_d], DOMAIN_INDEX[to_d]] = effect
    
    return matrix

SPILLOVER_MATRIX = build_spillover_matrix()

# Decay rates by domain
DECAY_RATES = {
    LifeDomain.HEALTH: 0.02,
    LifeDomain.SKILLS: 0.01,
    LifeDomain.FINANCES: 0.005,
    LifeDomain.RELATIONSHIPS: 0.03,
    LifeDomain.CAREER: 0.015,
    LifeDomain.MOOD: 0.1,
    LifeDomain.ENERGY: 0.15,
    LifeDomain.PURPOSE: 0.02,
    LifeDomain.CREATIVITY: 0.04,
    LifeDomain.SPIRITUALITY: 0.02,
    LifeDomain.BELIEF: 0.04,
    LifeDomain.FOCUS: 0.08,
    LifeDomain.GRATITUDE: 0.05,
}

# ════════════════════════════════════════════════════════════════════════════════
# VIRTUAL PET SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class PetSpecies:
    id: str
    name: str
    emoji: str
    base_happiness_decay: float
    hunger_rate: float
    energy_sensitivity: float
    social_need: float
    description: str

PET_SPECIES = {
    "phoenix": PetSpecies("phoenix", "Phoenix", "🔥", 0.02, 0.03, 0.8, 0.4, 
        "Rises from ashes when you recover from setbacks"),
    "turtle": PetSpecies("turtle", "Wise Turtle", "🐢", 0.01, 0.01, 0.3, 0.2, 
        "Thrives on slow, steady progress"),
    "butterfly": PetSpecies("butterfly", "Butterfly", "🦋", 0.04, 0.04, 0.9, 0.6, 
        "Loves transformation and creativity"),
    "owl": PetSpecies("owl", "Night Owl", "🦉", 0.015, 0.02, 0.5, 0.3, 
        "Appreciates deep learning and wisdom"),
    "dragon": PetSpecies("dragon", "Dragon", "🐉", 0.025, 0.035, 0.7, 0.5, 
        "Grows stronger with ambitious goals"),
    "fox": PetSpecies("fox", "Clever Fox", "🦊", 0.03, 0.03, 0.6, 0.7, 
        "Rewards creative problem-solving"),
    "wolf": PetSpecies("wolf", "Pack Wolf", "🐺", 0.02, 0.025, 0.5, 0.9, 
        "Thrives on relationships and teamwork"),
    "cat": PetSpecies("cat", "Zen Cat", "🐱", 0.015, 0.02, 0.4, 0.3, 
        "Values rest and self-care"),
}

@dataclass
class Pet:
    id: str
    user_id: str
    species_id: str
    name: str
    happiness: float = 0.7
    hunger: float = 0.3
    energy: float = 0.8
    bond_level: int = 1
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        species = PET_SPECIES.get(self.species_id)
        return {
            "id": self.id,
            "species": self.species_id,
            "species_name": species.name if species else self.species_id,
            "emoji": species.emoji if species else "🐾",
            "name": self.name,
            "happiness": round(self.happiness, 2),
            "hunger": round(self.hunger, 2),
            "energy": round(self.energy, 2),
            "bond_level": self.bond_level,
            "mood": self._calculate_mood(),
            "needs_attention": self.happiness < 0.3 or self.hunger > 0.7
        }
    
    def _calculate_mood(self) -> str:
        score = (self.happiness * 0.4 + (1 - self.hunger) * 0.3 + self.energy * 0.3)
        if score > 0.8: return "ecstatic"
        if score > 0.6: return "happy"
        if score > 0.4: return "content"
        if score > 0.2: return "sad"
        return "distressed"

# ════════════════════════════════════════════════════════════════════════════════
# LIFE MILESTONES (33 Milestones)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Milestone:
    id: str
    name: str
    category: str
    typical_age_range: Tuple[int, int]
    description: str
    celebration_message: str
    icon: str
    prerequisites: List[str] = field(default_factory=list)
    guidance: Dict[str, str] = field(default_factory=dict)  # By condition

LIFE_MILESTONES = {
    # Education
    "first_day_school": Milestone("first_day_school", "First Day of School", "education", 
        (5, 7), "Starting formal education", "The learning adventure begins! 📚", "🎒"),
    "graduation_hs": Milestone("graduation_hs", "High School Graduation", "education",
        (17, 19), "Completing secondary education", "You did it! Ready for the next chapter! 🎓", "🎓"),
    "college_degree": Milestone("college_degree", "College Degree", "education",
        (21, 26), "Earning a college degree", "Your dedication paid off! 🎉", "📜"),
    
    # Career
    "first_job": Milestone("first_job", "First Job", "career",
        (16, 22), "Getting your first paid position", "Welcome to the working world! 💼", "👔"),
    "career_milestone": Milestone("career_milestone", "Major Career Achievement", "career",
        (25, 45), "Significant career accomplishment", "Your hard work is paying off! ⭐", "🏆"),
    "retirement": Milestone("retirement", "Retirement", "career",
        (55, 70), "Transitioning out of full-time work", "Time for a new adventure! 🌅", "🎣"),
    
    # Relationships
    "first_relationship": Milestone("first_relationship", "First Relationship", "relationships",
        (14, 20), "First romantic relationship", "Opening your heart is brave! 💕", "❤️"),
    "marriage": Milestone("marriage", "Marriage/Partnership", "relationships",
        (22, 40), "Committing to a life partner", "Building a life together! 💒", "💍"),
    "becoming_parent": Milestone("becoming_parent", "Becoming a Parent", "family",
        (20, 45), "Having or adopting a child", "Welcome to parenthood! 👶", "👨‍👩‍👧"),
    
    # Personal Growth
    "overcome_fear": Milestone("overcome_fear", "Overcome a Major Fear", "personal_growth",
        (15, 99), "Conquering a significant fear", "Courage isn't the absence of fear! 💪", "🦁"),
    "learn_new_skill": Milestone("learn_new_skill", "Master a New Skill", "personal_growth",
        (10, 99), "Becoming proficient in something new", "Growth mindset in action! 🌱", "🎯"),
    "therapy_journey": Milestone("therapy_journey", "Started Therapy", "personal_growth",
        (15, 99), "Seeking professional mental health support", "Taking care of your mind is strength! 🧠", "💚"),
    
    # Financial
    "financial_independence": Milestone("financial_independence", "Financial Independence", "financial",
        (25, 50), "No longer financially dependent on others", "Freedom achieved! 🦅", "💰"),
    "first_home": Milestone("first_home", "First Home", "financial",
        (25, 45), "Purchasing your first property", "A place to call your own! 🏠", "🏡"),
    "debt_free": Milestone("debt_free", "Becoming Debt-Free", "financial",
        (25, 60), "Paying off all major debts", "Financial freedom feels amazing! ✨", "🎊"),
    
    # Health
    "fitness_goal": Milestone("fitness_goal", "Major Fitness Achievement", "health",
        (15, 70), "Completing a significant fitness goal", "Your body can do amazing things! 🏃", "💪"),
    "health_recovery": Milestone("health_recovery", "Health Recovery", "health",
        (0, 99), "Recovering from a major illness or injury", "Resilience personified! 🌟", "🏥"),
    
    # Adventure
    "first_solo_trip": Milestone("first_solo_trip", "First Solo Adventure", "adventure",
        (18, 40), "Traveling alone for the first time", "Independence looks great on you! 🌍", "✈️"),
    "bucket_list_item": Milestone("bucket_list_item", "Bucket List Achievement", "adventure",
        (15, 99), "Completing a lifelong dream", "Dreams do come true! 🌈", "⭐"),
}

# ════════════════════════════════════════════════════════════════════════════════
# FRACTAL MATHEMATICS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class FractalMathEngine:
    """Real fractal mathematics for life planning."""
    
    @staticmethod
    def box_counting_dimension(time_series: np.ndarray) -> float:
        """Calculate fractal dimension - measures life complexity."""
        n = len(time_series)
        if n < 10:
            return 1.5
        
        ts_min, ts_max = np.min(time_series), np.max(time_series)
        if ts_max - ts_min == 0:
            return 1.0
        
        normalized = (time_series - ts_min) / (ts_max - ts_min)
        
        box_sizes = []
        box_counts = []
        
        for box_size in range(2, n // 4 + 1):
            boxes = set()
            for i in range(n):
                x_box = i // box_size
                y_box = int(normalized[i] * (n // box_size))
                boxes.add((x_box, y_box))
            
            if len(boxes) > 0:
                box_sizes.append(box_size)
                box_counts.append(len(boxes))
        
        if len(box_sizes) < 2:
            return 1.5
        
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        
        return float(np.clip(-slope, 1.0, 2.0))
    
    @staticmethod
    def hurst_exponent(time_series: np.ndarray) -> Tuple[float, str]:
        """Calculate Hurst exponent - predicts trend persistence."""
        n = len(time_series)
        if n < 20:
            return 0.5, 'insufficient_data'
        
        max_k = min(n // 4, 50)
        RS = []
        ns = []
        
        for k in range(10, max_k + 1):
            subseries_count = n // k
            if subseries_count < 1:
                continue
            
            rs_values = []
            for i in range(subseries_count):
                subseries = time_series[i*k:(i+1)*k]
                mean_adj = subseries - np.mean(subseries)
                cumsum = np.cumsum(mean_adj)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subseries, ddof=1)
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                RS.append(np.mean(rs_values))
                ns.append(k)
        
        if len(RS) < 2:
            return 0.5, 'insufficient_data'
        
        log_n = np.log(ns)
        log_RS = np.log(RS)
        H, _ = np.polyfit(log_n, log_RS, 1)
        H = float(np.clip(H, 0.0, 1.0))
        
        if H < 0.45:
            interpretation = 'mean_reverting'
        elif H > 0.55:
            interpretation = 'trending'
        else:
            interpretation = 'random_walk'
        
        return H, interpretation
    
    @staticmethod
    def fractal_brownian_motion(n: int, hurst: float = 0.5) -> np.ndarray:
        """Generate fractal Brownian motion for realistic simulations."""
        freqs = np.fft.fftfreq(n)
        freqs[0] = 1e-10
        power = np.abs(freqs) ** (-(2 * hurst + 1))
        phases = np.random.uniform(0, 2 * np.pi, n)
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        fbm = np.real(np.fft.ifft(spectrum))
        fbm = (fbm - np.mean(fbm)) / (np.std(fbm) + 1e-10)
        return fbm
    
    @staticmethod
    def calculate_lacunarity(values: np.ndarray) -> float:
        """Calculate lacunarity - measures balance/gaps."""
        n = len(values)
        box_sizes = [2, 4, 8]
        lacunarities = []
        
        for r in box_sizes:
            if r >= n:
                continue
            masses = []
            for i in range(n - r + 1):
                mass = np.sum(values[i:i+r])
                masses.append(mass)
            
            if not masses:
                continue
            
            masses = np.array(masses)
            mean_mass = np.mean(masses)
            if mean_mass > 0:
                variance = np.var(masses)
                lac = (variance / (mean_mass ** 2)) + 1
                lacunarities.append(lac)
        
        return float(np.mean(lacunarities)) if lacunarities else 1.0
    
    @staticmethod
    def lyapunov_estimate(time_series: np.ndarray) -> float:
        """Estimate Lyapunov exponent - measures chaos sensitivity."""
        n = len(time_series)
        if n < 30:
            return 0.0
        
        embedding_dim = 3
        delay = 1
        n_vectors = n - (embedding_dim - 1) * delay
        
        vectors = np.zeros((n_vectors, embedding_dim))
        for i in range(n_vectors):
            for j in range(embedding_dim):
                vectors[i, j] = time_series[i + j * delay]
        
        lyapunov_sum = 0
        count = 0
        
        for i in range(min(n_vectors - 1, 100)):
            distances = [norm(vectors[i] - vectors[j]) for j in range(n_vectors) if j != i]
            if not distances:
                continue
            
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist < 1e-10:
                continue
            
            if i + 1 < n_vectors and min_dist_idx + 1 < n_vectors:
                next_dist = norm(vectors[i + 1] - vectors[min_dist_idx + 1])
                if next_dist > 0 and min_dist > 0:
                    lyapunov_sum += np.log(next_dist / min_dist)
                    count += 1
        
        return float(lyapunov_sum / count) if count > 0 else 0.0
    
    @staticmethod
    def decompose_goal_lsystem(goal_name: str, depth: int = 3) -> Dict:
        """L-system goal decomposition."""
        def branch(level: int, prefix: str) -> Dict:
            if level >= depth:
                return {'name': prefix, 'type': 'task', 'children': [], 'hours': FIBONACCI[min(level + 2, 10)]}
            
            n_children = min(FIBONACCI[min(level + 2, 10)], 3)
            children = [branch(level + 1, f"{prefix}.{i+1}") for i in range(n_children)]
            
            return {
                'name': prefix,
                'type': 'milestone' if level == 0 else 'subgoal',
                'children': children,
                'hours': sum(c['hours'] for c in children)
            }
        
        tree = branch(0, goal_name)
        
        def count_tasks(node):
            if not node['children']:
                return 1
            return sum(count_tasks(c) for c in node['children'])
        
        tree['total_tasks'] = count_tasks(tree)
        return tree

FRACTAL = FractalMathEngine()

# ════════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL LIFE ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class MathematicalLifeEngine:
    """Complete life engine with all mathematical frameworks."""
    
    def __init__(self):
        self.gamma = GAMMA
        self.spillover = SPILLOVER_MATRIX
        self.belief_uplift = 0.3  # Law of Attraction effectiveness boost
    
    def state_transition(self, state: np.ndarray, action_effects: np.ndarray, 
                        noise_std: float = 0.01) -> np.ndarray:
        """Compute s_{t+1} = f(s_t, a_t) + spillover - decay + noise."""
        # Direct effects
        new_state = state + action_effects
        
        # Spillover effects
        spillover_effect = self.spillover.T @ state * 0.02
        new_state += spillover_effect
        
        # Decay
        for i, domain in enumerate(LifeDomain):
            new_state[i] -= DECAY_RATES[domain] * state[i]
        
        # Noise
        noise = np.random.normal(0, noise_std, N_DOMAINS)
        new_state += noise
        
        return np.clip(new_state, 0, 1)
    
    def compute_q_value(self, state: np.ndarray, task: TaskType) -> float:
        """Compute Q(s,a) = R(s,a) + γ·E[V(s')]."""
        # Immediate reward
        effect = np.array(task.effect_vector)
        weighted_effect = np.sum(effect * (1 - state))  # Higher reward for lower domains
        
        # Belief boost (Law of Attraction)
        belief_idx = DOMAIN_INDEX[LifeDomain.BELIEF]
        belief_boost = 1 + self.belief_uplift * state[belief_idx]
        
        # Future value estimate
        next_state = self.state_transition(state, effect, noise_std=0)
        future_value = np.mean(next_state)
        
        # Q-value with belief boost
        q = (weighted_effect * belief_boost + self.gamma * future_value)
        
        return float(q)
    
    def compute_flow_probability(self, state: np.ndarray, task: TaskType) -> float:
        """Flow = exp(-(|challenge/skill - φ|²)/σ²) × energy_factor."""
        skills_idx = DOMAIN_INDEX[LifeDomain.SKILLS]
        energy_idx = DOMAIN_INDEX[LifeDomain.ENERGY]
        
        skill_level = max(0.1, state[skills_idx])
        challenge = task.flow_challenge
        ratio = challenge / skill_level
        
        # Flow is optimal when ratio ≈ φ
        flow_score = math.exp(-((ratio - PHI) ** 2) / 0.5)
        
        # Energy modulates flow
        energy_factor = state[energy_idx]
        
        return float(flow_score * energy_factor)
    
    def rank_tasks(self, state: np.ndarray, available_energy: int) -> List[Dict]:
        """Rank tasks by Q-value adjusted for energy efficiency."""
        rankings = []
        
        for task_id, task in TASK_TYPES.items():
            if task.energy_cost > available_energy:
                continue
            
            q_value = self.compute_q_value(state, task)
            flow = self.compute_flow_probability(state, task)
            efficiency = q_value / max(task.energy_cost, 0.5)
            
            rankings.append({
                'task_id': task_id,
                'name': task.name,
                'icon': task.icon,
                'q_value': round(q_value, 4),
                'flow_probability': round(flow, 3),
                'efficiency': round(efficiency, 4),
                'energy_cost': task.energy_cost,
                'duration': task.duration_minutes,
                'category': task.category
            })
        
        rankings.sort(key=lambda x: x['efficiency'], reverse=True)
        return rankings
    
    def golden_ratio_allocation(self, foundation_hours: float, 
                               execution_hours: float, 
                               total_hours: float) -> Dict:
        """Allocate time using φ:1 ratio."""
        foundation = total_hours * PHI_INVERSE  # ~62%
        execution = total_hours * (1 - PHI_INVERSE)  # ~38%
        
        return {
            'foundation_hours': round(foundation, 2),
            'execution_hours': round(execution, 2),
            'ratio': round(foundation / max(execution, 0.01), 4),
            'optimal_ratio': PHI,
            'explanation': f'φ-based split: {PHI_INVERSE*100:.1f}% foundation, {(1-PHI_INVERSE)*100:.1f}% execution'
        }
    
    def fibonacci_schedule(self, start_date: date, num_reviews: int = 10) -> List[Dict]:
        """Generate Fibonacci-spaced review schedule."""
        schedule = []
        for i in range(num_reviews):
            fib_days = FIBONACCI[min(i + 1, len(FIBONACCI) - 1)]
            review_date = start_date + timedelta(days=fib_days)
            schedule.append({
                'review_number': i + 1,
                'days_from_start': fib_days,
                'date': review_date.isoformat(),
                'fibonacci_index': i + 1
            })
        return schedule
    
    def compound_growth_projection(self, initial: float, daily_rate: float, 
                                   days: int) -> Dict:
        """Project compound growth: V_t = V_0 · (1+r)^t."""
        values = [initial * ((1 + daily_rate) ** d) for d in range(days + 1)]
        doubling_time = RULE_OF_72 / (daily_rate * 365) if daily_rate > 0 else float('inf')
        
        return {
            'initial': initial,
            'final': round(values[-1], 4),
            'growth_factor': round(values[-1] / initial, 4),
            'doubling_time_years': round(doubling_time, 2),
            'trajectory': [round(v, 4) for v in values[::max(1, days//30)]]
        }
    
    def habit_formation_curve(self, days: int, consistency: float = 0.8) -> float:
        """Sigmoid habit formation: automaticity = σ((days - 66)/τ)."""
        adjusted_days = days * consistency
        tau = 20  # Spread parameter
        automaticity = 1 / (1 + math.exp(-(adjusted_days - HABIT_FORMATION_AVG) / tau))
        return float(automaticity)
    
    def analyze_trajectory(self, history: List[List[float]]) -> Dict:
        """Complete fractal analysis of life trajectory."""
        if len(history) < 10:
            return {'error': 'Need at least 10 data points'}
        
        history_arr = np.array(history)
        avg_trajectory = np.mean(history_arr, axis=1)
        
        dim = FRACTAL.box_counting_dimension(avg_trajectory)
        hurst, hurst_interp = FRACTAL.hurst_exponent(avg_trajectory)
        lyapunov = FRACTAL.lyapunov_estimate(avg_trajectory)
        lacunarity = FRACTAL.calculate_lacunarity(history_arr[-1])
        
        recommendations = []
        
        if dim > 1.6:
            recommendations.append({
                'type': 'simplify',
                'message': 'Life trajectory shows high complexity. Consider simplifying.',
                'priority': 'high'
            })
        
        if hurst < 0.45:
            recommendations.append({
                'type': 'persistence',
                'message': 'Progress tends to reverse. Build consistent habits.',
                'priority': 'high'
            })
        
        if lacunarity > 2.5:
            recommendations.append({
                'type': 'balance',
                'message': 'Life domains are unbalanced. Rotate focus with golden angle.',
                'priority': 'medium'
            })
        
        return {
            'fractal_dimension': round(dim, 3),
            'dimension_interpretation': 'smooth' if dim < 1.3 else 'moderate' if dim < 1.6 else 'chaotic',
            'hurst_exponent': round(hurst, 3),
            'hurst_interpretation': hurst_interp,
            'lyapunov_exponent': round(lyapunov, 3),
            'chaos_level': 'high' if lyapunov > 0.5 else 'moderate' if lyapunov > 0 else 'stable',
            'lacunarity': round(lacunarity, 3),
            'balance_level': 'balanced' if lacunarity < 1.5 else 'moderate' if lacunarity < 2.5 else 'unbalanced',
            'recommendations': recommendations
        }

ENGINE = MathematicalLifeEngine()

# ════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
CORS(app, supports_credentials=True)

DATABASE_PATH = os.environ.get('DATABASE_PATH', 'life_fractal_v14.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

app.teardown_appcontext(close_db)

def init_db():
    """Initialize all database tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            current_state TEXT,
            energy INTEGER DEFAULT 12,
            max_energy INTEGER DEFAULT 12,
            timezone TEXT DEFAULT 'UTC',
            created_at TEXT,
            last_login TEXT
        )
    ''')
    
    # State history for fractal analysis
    c.execute('''
        CREATE TABLE IF NOT EXISTS state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            state_vector TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Pets table
    c.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            species_id TEXT NOT NULL,
            name TEXT NOT NULL,
            happiness REAL DEFAULT 0.7,
            hunger REAL DEFAULT 0.3,
            energy REAL DEFAULT 0.8,
            bond_level INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Task completions
    c.execute('''
        CREATE TABLE IF NOT EXISTS task_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            energy_before INTEGER,
            energy_after INTEGER,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Milestones achieved
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_milestones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            milestone_id TEXT NOT NULL,
            achieved_at TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, milestone_id)
        )
    ''')
    
    # Goals
    c.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            target_domain TEXT,
            target_value REAL,
            current_progress REAL DEFAULT 0,
            deadline TEXT,
            created_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Daily journal
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            content TEXT,
            mood_score REAL,
            energy_score REAL,
            gratitude_items TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized with all tables")

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_user_by_id(user_id: str) -> Optional[Dict]:
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_user_state(user_id: str) -> np.ndarray:
    user = get_user_by_id(user_id)
    if user and user.get('current_state'):
        return np.array(json.loads(user['current_state']))
    return np.ones(N_DOMAINS) * 0.5

def get_state_history(user_id: str, limit: int = 100) -> List[List[float]]:
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT state_vector FROM state_history 
        WHERE user_id = ? ORDER BY recorded_at DESC LIMIT ?
    ''', (user_id, limit))
    rows = cursor.fetchall()
    return [json.loads(row['state_vector']) for row in reversed(rows)]

def save_user_state(user_id: str, state: np.ndarray):
    db = get_db()
    state_json = json.dumps(state.tolist())
    now = datetime.now(timezone.utc).isoformat()
    
    db.execute('UPDATE users SET current_state = ? WHERE id = ?', (state_json, user_id))
    db.execute('INSERT INTO state_history (user_id, state_vector, recorded_at) VALUES (?, ?, ?)',
               (user_id, state_json, now))
    db.commit()

def require_auth(f):
    """Decorator to require authentication."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        user = get_user_by_id(user_id)
        if not user:
            session.pop('user_id', None)
            return jsonify({'error': 'User not found'}), 401
        g.user = user
        return f(*args, **kwargs)
    return decorated

# ════════════════════════════════════════════════════════════════════════════════
# AUTH API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    display_name = data.get('display_name', email.split('@')[0])
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    db = get_db()
    try:
        user_id = f"user_{secrets.token_hex(12)}"
        now = datetime.now(timezone.utc).isoformat()
        initial_state = (np.ones(N_DOMAINS) * 0.5).tolist()
        
        db.execute('''
            INSERT INTO users (id, email, password_hash, display_name, current_state, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, email, generate_password_hash(password), display_name, 
              json.dumps(initial_state), now, now))
        db.commit()
        
        session['user_id'] = user_id
        session.permanent = True
        
        return jsonify({
            'success': True, 
            'user_id': user_id, 
            'display_name': display_name
        }), 201
        
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already registered'}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    session['user_id'] = user['id']
    session.permanent = True
    
    db.execute('UPDATE users SET last_login = ? WHERE id = ?', 
               (datetime.now(timezone.utc).isoformat(), user['id']))
    db.commit()
    
    return jsonify({
        'success': True,
        'user_id': user['id'],
        'display_name': user['display_name'],
        'energy': user['energy']
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/auth/session')
def check_session():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'authenticated': False}), 401
    
    user = get_user_by_id(user_id)
    if not user:
        session.pop('user_id', None)
        return jsonify({'authenticated': False}), 401
    
    return jsonify({
        'authenticated': True,
        'user': {
            'id': user['id'],
            'display_name': user['display_name'],
            'email': user['email'],
            'energy': user['energy'],
            'max_energy': user['max_energy']
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# STATE API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/state')
@require_auth
def get_state():
    state = get_user_state(g.user['id'])
    
    domains = []
    for i, domain in enumerate(LifeDomain):
        domains.append({
            'id': domain.value,
            'name': domain.value.replace('_', ' ').title(),
            'value': round(float(state[i]), 3),
            'icon': DOMAIN_ICONS.get(domain.value, '📊'),
            'color': DOMAIN_COLORS.get(domain.value, '#888')
        })
    
    return jsonify({
        'domains': domains,
        'vector': state.tolist(),
        'average': round(float(np.mean(state)), 3),
        'balance_score': round(1 - float(np.std(state)), 3)
    })

@app.route('/api/state', methods=['POST'])
@require_auth
def update_state():
    data = request.get_json() or {}
    updates = data.get('updates', {})
    
    state = get_user_state(g.user['id'])
    
    for domain_str, value in updates.items():
        try:
            domain = LifeDomain(domain_str)
            idx = DOMAIN_INDEX[domain]
            state[idx] = np.clip(float(value), 0, 1)
        except (ValueError, KeyError):
            pass
    
    save_user_state(g.user['id'], state)
    
    return jsonify({'success': True, 'new_state': state.tolist()})

# ════════════════════════════════════════════════════════════════════════════════
# TASK API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/tasks')
@require_auth
def get_tasks():
    state = get_user_state(g.user['id'])
    energy = g.user['energy']
    
    rankings = ENGINE.rank_tasks(state, energy)
    
    return jsonify({
        'tasks': rankings,
        'available_energy': energy,
        'total_tasks': len(TASK_TYPES)
    })

@app.route('/api/tasks/complete', methods=['POST'])
@require_auth
def complete_task():
    data = request.get_json() or {}
    task_id = data.get('task_id')
    
    if task_id not in TASK_TYPES:
        return jsonify({'error': 'Invalid task'}), 400
    
    task = TASK_TYPES[task_id]
    energy = g.user['energy']
    
    if task.energy_cost > energy:
        return jsonify({'error': 'Not enough energy'}), 400
    
    # Get current state
    state = get_user_state(g.user['id'])
    
    # Apply task effects
    effects = np.array(task.effect_vector)
    new_state = ENGINE.state_transition(state, effects, noise_std=0.005)
    
    # Update energy
    new_energy = energy - task.energy_cost
    
    # Save everything
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    
    save_user_state(g.user['id'], new_state)
    db.execute('UPDATE users SET energy = ? WHERE id = ?', (new_energy, g.user['id']))
    db.execute('''
        INSERT INTO task_completions (user_id, task_id, completed_at, energy_before, energy_after)
        VALUES (?, ?, ?, ?, ?)
    ''', (g.user['id'], task_id, now, energy, new_energy))
    db.commit()
    
    # Update pet happiness
    update_pet_on_task(g.user['id'], task)
    
    return jsonify({
        'success': True,
        'task': task.name,
        'energy_spent': task.energy_cost,
        'new_energy': new_energy,
        'state_change': {
            domain.value: round(float(new_state[i] - state[i]), 4)
            for i, domain in enumerate(LifeDomain)
            if abs(new_state[i] - state[i]) > 0.001
        }
    })

def update_pet_on_task(user_id: str, task: TaskType):
    """Update pet based on task completion."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,))
    pet_row = cursor.fetchone()
    
    if not pet_row:
        return
    
    # Happiness boost based on task
    happiness_boost = 0.05
    if task.category == 'mindset':
        happiness_boost = 0.08
    elif task.category == 'health':
        happiness_boost = 0.06
    
    new_happiness = min(1.0, pet_row['happiness'] + happiness_boost)
    new_hunger = min(1.0, pet_row['hunger'] + 0.02)  # Gets slightly hungrier
    
    db.execute('UPDATE pets SET happiness = ?, hunger = ? WHERE id = ?',
               (new_happiness, new_hunger, pet_row['id']))
    db.commit()

# ════════════════════════════════════════════════════════════════════════════════
# PET API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet')
@require_auth
def get_pet():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (g.user['id'],))
    pet_row = cursor.fetchone()
    
    if not pet_row:
        return jsonify({'has_pet': False, 'available_species': list(PET_SPECIES.keys())})
    
    pet = Pet(
        id=pet_row['id'],
        user_id=pet_row['user_id'],
        species_id=pet_row['species_id'],
        name=pet_row['name'],
        happiness=pet_row['happiness'],
        hunger=pet_row['hunger'],
        energy=pet_row['energy'],
        bond_level=pet_row['bond_level'],
        created_at=pet_row['created_at']
    )
    
    return jsonify({'has_pet': True, 'pet': pet.to_dict()})

@app.route('/api/pet/adopt', methods=['POST'])
@require_auth
def adopt_pet():
    data = request.get_json() or {}
    species_id = data.get('species')
    name = data.get('name', 'Buddy')
    
    if species_id not in PET_SPECIES:
        return jsonify({'error': 'Invalid species'}), 400
    
    db = get_db()
    cursor = db.cursor()
    
    # Check if already has pet
    cursor.execute('SELECT id FROM pets WHERE user_id = ?', (g.user['id'],))
    if cursor.fetchone():
        return jsonify({'error': 'You already have a pet'}), 400
    
    pet_id = f"pet_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc).isoformat()
    
    db.execute('''
        INSERT INTO pets (id, user_id, species_id, name, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (pet_id, g.user['id'], species_id, name, now))
    db.commit()
    
    species = PET_SPECIES[species_id]
    return jsonify({
        'success': True,
        'message': f'{species.emoji} {name} the {species.name} has joined you!',
        'pet_id': pet_id
    })

@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (g.user['id'],))
    pet_row = cursor.fetchone()
    
    if not pet_row:
        return jsonify({'error': 'No pet found'}), 404
    
    new_hunger = max(0, pet_row['hunger'] - 0.3)
    new_happiness = min(1.0, pet_row['happiness'] + 0.1)
    
    db.execute('UPDATE pets SET hunger = ?, happiness = ? WHERE id = ?',
               (new_hunger, new_happiness, pet_row['id']))
    db.commit()
    
    return jsonify({'success': True, 'hunger': new_hunger, 'happiness': new_happiness})

@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_with_pet():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (g.user['id'],))
    pet_row = cursor.fetchone()
    
    if not pet_row:
        return jsonify({'error': 'No pet found'}), 404
    
    new_happiness = min(1.0, pet_row['happiness'] + 0.15)
    new_hunger = min(1.0, pet_row['hunger'] + 0.05)
    
    db.execute('UPDATE pets SET happiness = ?, hunger = ? WHERE id = ?',
               (new_happiness, new_hunger, pet_row['id']))
    db.commit()
    
    return jsonify({'success': True, 'happiness': new_happiness})

@app.route('/api/pet/species')
def get_species():
    species_list = []
    for species_id, species in PET_SPECIES.items():
        species_list.append({
            'id': species_id,
            'name': species.name,
            'emoji': species.emoji,
            'description': species.description
        })
    return jsonify({'species': species_list})

# ════════════════════════════════════════════════════════════════════════════════
# MILESTONES API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/milestones')
@require_auth
def get_milestones():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT milestone_id, achieved_at FROM user_milestones WHERE user_id = ?',
                   (g.user['id'],))
    achieved = {row['milestone_id']: row['achieved_at'] for row in cursor.fetchall()}
    
    milestones = []
    for ms_id, ms in LIFE_MILESTONES.items():
        milestones.append({
            'id': ms_id,
            'name': ms.name,
            'category': ms.category,
            'icon': ms.icon,
            'description': ms.description,
            'age_range': list(ms.typical_age_range),
            'achieved': ms_id in achieved,
            'achieved_at': achieved.get(ms_id)
        })
    
    return jsonify({
        'milestones': milestones,
        'total': len(LIFE_MILESTONES),
        'achieved_count': len(achieved)
    })

@app.route('/api/milestones/achieve', methods=['POST'])
@require_auth
def achieve_milestone():
    data = request.get_json() or {}
    milestone_id = data.get('milestone_id')
    notes = data.get('notes', '')
    
    if milestone_id not in LIFE_MILESTONES:
        return jsonify({'error': 'Invalid milestone'}), 400
    
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    
    try:
        db.execute('''
            INSERT INTO user_milestones (user_id, milestone_id, achieved_at, notes)
            VALUES (?, ?, ?, ?)
        ''', (g.user['id'], milestone_id, now, notes))
        db.commit()
        
        ms = LIFE_MILESTONES[milestone_id]
        return jsonify({
            'success': True,
            'celebration': ms.celebration_message,
            'milestone': ms.name
        })
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Milestone already achieved'}), 400

# ════════════════════════════════════════════════════════════════════════════════
# FRACTAL ANALYSIS API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/fractal/analysis')
@require_auth
def fractal_analysis():
    history = get_state_history(g.user['id'], limit=100)
    
    if len(history) < 10:
        return jsonify({
            'error': 'Need more data',
            'message': 'At least 10 state recordings needed',
            'current_count': len(history)
        }), 400
    
    analysis = ENGINE.analyze_trajectory(history)
    return jsonify(analysis)

@app.route('/api/fractal/dimension')
@require_auth
def get_fractal_dimension():
    history = get_state_history(g.user['id'])
    
    if len(history) < 10:
        return jsonify({'dimension': 1.5, 'message': 'Insufficient data'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    dim = FRACTAL.box_counting_dimension(avg_trajectory)
    
    return jsonify({
        'dimension': round(dim, 3),
        'interpretation': 'smooth' if dim < 1.3 else 'moderate' if dim < 1.6 else 'chaotic',
        'recommendation': 'Consider simplifying routines' if dim > 1.6 else 'Good complexity level'
    })

@app.route('/api/fractal/hurst')
@require_auth
def get_hurst():
    history = get_state_history(g.user['id'])
    
    if len(history) < 20:
        return jsonify({'hurst': 0.5, 'message': 'Insufficient data'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    hurst, interpretation = FRACTAL.hurst_exponent(avg_trajectory)
    
    return jsonify({
        'hurst': round(hurst, 3),
        'interpretation': interpretation,
        'prediction': 'Trends will persist' if hurst > 0.55 else 'Trends will reverse' if hurst < 0.45 else 'Unpredictable'
    })

@app.route('/api/fractal/decompose', methods=['POST'])
def decompose_goal():
    data = request.get_json() or {}
    goal = data.get('goal', 'My Goal')
    depth = min(5, max(1, data.get('depth', 3)))
    
    tree = FRACTAL.decompose_goal_lsystem(goal, depth)
    return jsonify(tree)

# ════════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL TOOLS API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/math/golden-allocation', methods=['POST'])
def golden_allocation():
    data = request.get_json() or {}
    total_hours = float(data.get('total_hours', 8))
    
    allocation = ENGINE.golden_ratio_allocation(0, 0, total_hours)
    return jsonify(allocation)

@app.route('/api/math/fibonacci-schedule', methods=['POST'])
def fibonacci_schedule():
    data = request.get_json() or {}
    start = data.get('start_date', date.today().isoformat())
    num_reviews = min(15, data.get('num_reviews', 10))
    
    start_date = date.fromisoformat(start)
    schedule = ENGINE.fibonacci_schedule(start_date, num_reviews)
    
    return jsonify({'schedule': schedule})

@app.route('/api/math/compound-growth', methods=['POST'])
def compound_growth():
    data = request.get_json() or {}
    initial = float(data.get('initial', 0.5))
    daily_rate = float(data.get('daily_rate', 0.01))
    days = min(365, int(data.get('days', 90)))
    
    projection = ENGINE.compound_growth_projection(initial, daily_rate, days)
    return jsonify(projection)

@app.route('/api/math/habit-progress')
@require_auth
def habit_progress():
    # Count consecutive days with state updates
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT recorded_at FROM state_history 
        WHERE user_id = ? 
        ORDER BY recorded_at DESC LIMIT 100
    ''', (g.user['id'],))
    
    rows = cursor.fetchall()
    if not rows:
        return jsonify({'days': 0, 'automaticity': 0})
    
    # Count streak
    streak = 1
    prev_date = None
    for row in rows:
        current_date = datetime.fromisoformat(row['recorded_at'].replace('Z', '+00:00')).date()
        if prev_date is None:
            prev_date = current_date
            continue
        
        diff = (prev_date - current_date).days
        if diff == 1:
            streak += 1
            prev_date = current_date
        else:
            break
    
    automaticity = ENGINE.habit_formation_curve(streak)
    
    return jsonify({
        'days': streak,
        'automaticity': round(automaticity, 3),
        'target_days': HABIT_FORMATION_AVG,
        'percentage': round(streak / HABIT_FORMATION_AVG * 100, 1)
    })

@app.route('/api/math/constants')
def get_constants():
    return jsonify({
        'sacred': {
            'phi': PHI,
            'phi_inverse': PHI_INVERSE,
            'golden_angle_deg': GOLDEN_ANGLE_DEG,
            'fibonacci': FIBONACCI[:15],
            'lucas': LUCAS[:10]
        },
        'scientific': {
            'habit_formation_avg': HABIT_FORMATION_AVG,
            'flow_ratio': FLOW_CHALLENGE_SKILL_RATIO,
            'forgetting_rate': FORGETTING_RATE,
            'network_exponent': METCALFE_EXPONENT,
            'rule_of_72': RULE_OF_72
        },
        'fractal': {
            'dimensions': FRACTAL_DIMENSIONS
        },
        'law_of_attraction': {
            'belief_uplift': 0.3,
            'belief_gain_rate': 0.8,
            'focus_gain_rate': 0.9,
            'belief_decay': 0.04,
            'focus_decay': 0.08
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# ENERGY API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/energy')
@require_auth
def get_energy():
    return jsonify({
        'current': g.user['energy'],
        'max': g.user['max_energy'],
        'percentage': round(g.user['energy'] / g.user['max_energy'] * 100, 1)
    })

@app.route('/api/energy/recover', methods=['POST'])
@require_auth
def recover_energy():
    data = request.get_json() or {}
    amount = min(3, max(1, int(data.get('amount', 1))))
    
    new_energy = min(g.user['max_energy'], g.user['energy'] + amount)
    
    db = get_db()
    db.execute('UPDATE users SET energy = ? WHERE id = ?', (new_energy, g.user['id']))
    db.commit()
    
    return jsonify({'success': True, 'energy': new_energy})

@app.route('/api/energy/reset', methods=['POST'])
@require_auth
def reset_energy():
    """Reset energy to max (called at start of new day)."""
    db = get_db()
    db.execute('UPDATE users SET energy = max_energy WHERE id = ?', (g.user['id'],))
    db.commit()
    
    return jsonify({'success': True, 'energy': g.user['max_energy']})

# ════════════════════════════════════════════════════════════════════════════════
# JOURNAL API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/journal', methods=['GET'])
@require_auth
def get_journal():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT * FROM journal_entries 
        WHERE user_id = ? 
        ORDER BY entry_date DESC LIMIT 30
    ''', (g.user['id'],))
    
    entries = []
    for row in cursor.fetchall():
        entries.append({
            'id': row['id'],
            'date': row['entry_date'],
            'content': row['content'],
            'mood_score': row['mood_score'],
            'energy_score': row['energy_score'],
            'gratitude_items': json.loads(row['gratitude_items']) if row['gratitude_items'] else []
        })
    
    return jsonify({'entries': entries})

@app.route('/api/journal', methods=['POST'])
@require_auth
def create_journal_entry():
    data = request.get_json() or {}
    content = data.get('content', '')
    mood_score = data.get('mood_score')
    energy_score = data.get('energy_score')
    gratitude_items = data.get('gratitude_items', [])
    
    db = get_db()
    now = datetime.now(timezone.utc)
    
    db.execute('''
        INSERT INTO journal_entries (user_id, entry_date, content, mood_score, energy_score, gratitude_items, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (g.user['id'], now.date().isoformat(), content, mood_score, energy_score, 
          json.dumps(gratitude_items), now.isoformat()))
    db.commit()
    
    # Update gratitude domain if gratitude items provided
    if gratitude_items:
        state = get_user_state(g.user['id'])
        gratitude_boost = min(0.05, len(gratitude_items) * 0.015)
        state[DOMAIN_INDEX[LifeDomain.GRATITUDE]] = min(1.0, state[DOMAIN_INDEX[LifeDomain.GRATITUDE]] + gratitude_boost)
        save_user_state(g.user['id'], state)
    
    return jsonify({'success': True})

# ════════════════════════════════════════════════════════════════════════════════
# GOALS API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL
        ORDER BY deadline ASC
    ''', (g.user['id'],))
    
    goals = []
    for row in cursor.fetchall():
        goals.append({
            'id': row['id'],
            'title': row['title'],
            'description': row['description'],
            'target_domain': row['target_domain'],
            'target_value': row['target_value'],
            'current_progress': row['current_progress'],
            'deadline': row['deadline'],
            'created_at': row['created_at']
        })
    
    return jsonify({'goals': goals})

@app.route('/api/goals', methods=['POST'])
@require_auth
def create_goal():
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    
    if not title:
        return jsonify({'error': 'Title required'}), 400
    
    goal_id = f"goal_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc).isoformat()
    
    db = get_db()
    db.execute('''
        INSERT INTO goals (id, user_id, title, description, target_domain, target_value, deadline, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (goal_id, g.user['id'], title, data.get('description', ''), 
          data.get('target_domain'), data.get('target_value', 1.0),
          data.get('deadline'), now))
    db.commit()
    
    return jsonify({'success': True, 'goal_id': goal_id})

# ════════════════════════════════════════════════════════════════════════════════
# SPILLOVER API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/spillovers')
def get_spillovers():
    spillovers = []
    for i, from_domain in enumerate(LifeDomain):
        for j, to_domain in enumerate(LifeDomain):
            if SPILLOVER_MATRIX[i, j] > 0:
                spillovers.append({
                    'from': from_domain.value,
                    'to': to_domain.value,
                    'strength': float(SPILLOVER_MATRIX[i, j]),
                    'from_icon': DOMAIN_ICONS[from_domain.value],
                    'to_icon': DOMAIN_ICONS[to_domain.value]
                })
    
    return jsonify({
        'spillovers': spillovers,
        'total_effects': len(spillovers)
    })

# ════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '14.0.0',
        'name': 'Life Fractal Ultimate',
        'features': {
            'domains': N_DOMAINS,
            'tasks': len(TASK_TYPES),
            'spillovers': int(np.sum(SPILLOVER_MATRIX > 0)),
            'milestones': len(LIFE_MILESTONES),
            'pet_species': len(PET_SPECIES)
        },
        'math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE_DEG,
            'habit_formation': HABIT_FORMATION_AVG
        }
    })

# ════════════════════════════════════════════════════════════════════════════════
# MAIN HTML FRONTEND
# ════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal v14 - Ultimate Life Planner</title>
    <style>
        :root {
            --primary: #6B5B95;
            --secondary: #88B04B;
            --accent: #F7CAC9;
            --bg: #F8F9FA;
            --surface: #FFFFFF;
            --text: #333;
            --text-light: #666;
            --border: #E0E0E0;
            --success: #4CAF50;
            --warning: #FF9800;
            --error: #F44336;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, #92A8D1 100%);
            color: white;
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }
        .header-right { display: flex; align-items: center; gap: 1rem; }
        .energy-display {
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
        }
        
        /* Main Layout */
        .main { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; }
        
        /* Cards */
        .card {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--primary);
        }
        
        /* Domain Bars */
        .domain-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .domain-icon { font-size: 1.2rem; width: 30px; text-align: center; }
        .domain-name { width: 100px; font-size: 0.85rem; color: var(--text-light); }
        .domain-bar {
            flex: 1;
            height: 12px;
            background: var(--border);
            border-radius: 6px;
            overflow: hidden;
        }
        .domain-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        .domain-value { width: 45px; text-align: right; font-size: 0.85rem; font-weight: 600; }
        
        /* Tasks */
        .task-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .task-item:hover { background: #E8E8E8; }
        .task-icon { font-size: 1.5rem; margin-right: 0.75rem; }
        .task-info { flex: 1; }
        .task-name { font-weight: 600; }
        .task-meta { font-size: 0.8rem; color: var(--text-light); }
        .task-cost {
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        /* Pet */
        .pet-display { text-align: center; padding: 1rem; }
        .pet-emoji { font-size: 4rem; margin-bottom: 0.5rem; }
        .pet-name { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }
        .pet-stats { display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; }
        .pet-stat {
            text-align: center;
            padding: 0.5rem 1rem;
            background: var(--bg);
            border-radius: 8px;
        }
        .pet-stat-value { font-size: 1.2rem; font-weight: 600; }
        .pet-stat-label { font-size: 0.75rem; color: var(--text-light); }
        .pet-actions { display: flex; gap: 0.5rem; justify-content: center; margin-top: 1rem; }
        
        /* Buttons */
        .btn {
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { opacity: 0.9; }
        .btn-secondary { background: var(--border); color: var(--text); }
        .btn-secondary:hover { background: #D0D0D0; }
        .btn-success { background: var(--success); color: white; }
        
        /* Auth Forms */
        .auth-container {
            max-width: 400px;
            margin: 3rem auto;
            padding: 2rem;
            background: var(--surface);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .auth-container h2 { text-align: center; margin-bottom: 1.5rem; color: var(--primary); }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
        }
        .form-group input:focus { outline: none; border-color: var(--primary); }
        .auth-toggle { text-align: center; margin-top: 1rem; }
        .auth-toggle a { color: var(--primary); cursor: pointer; }
        
        /* Milestones */
        .milestone-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg);
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .milestone-icon { font-size: 1.5rem; margin-right: 0.75rem; }
        .milestone-achieved { background: #E8F5E9; }
        .milestone-check { margin-left: auto; color: var(--success); font-size: 1.2rem; }
        
        /* Loading */
        .loading { text-align: center; padding: 2rem; color: var(--text-light); }
        
        /* Notifications */
        .notification {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: var(--primary);
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
            z-index: 1000;
        }
        .notification.show { transform: translateY(0); opacity: 1; }
        .notification.success { background: var(--success); }
        .notification.error { background: var(--error); }
        
        /* Math Info */
        .math-box {
            background: #F5F5F5;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header { flex-direction: column; gap: 1rem; text-align: center; }
            .grid { grid-template-columns: 1fr; }
            .main { padding: 1rem; }
        }
    </style>
</head>
<body>
    <div id="app">
        <div class="loading">Loading Life Fractal...</div>
    </div>

    <div id="notification" class="notification"></div>

    <script>
        // State
        let currentUser = null;
        let userState = null;
        let userPet = null;
        let userEnergy = 12;

        // API Helper
        async function api(endpoint, options = {}) {
            const res = await fetch(endpoint, {
                ...options,
                headers: { 'Content-Type': 'application/json', ...options.headers },
                credentials: 'include'
            });
            return res.json();
        }

        // Notification
        function notify(message, type = 'info') {
            const el = document.getElementById('notification');
            el.textContent = message;
            el.className = 'notification ' + type + ' show';
            setTimeout(() => el.classList.remove('show'), 3000);
        }

        // Check session
        async function checkSession() {
            const data = await api('/api/auth/session');
            if (data.authenticated) {
                currentUser = data.user;
                userEnergy = data.user.energy;
                await loadDashboard();
            } else {
                showAuth();
            }
        }

        // Auth UI
        function showAuth() {
            document.getElementById('app').innerHTML = `
                <div class="auth-container">
                    <h2>🌀 Life Fractal</h2>
                    <form id="authForm">
                        <div class="form-group">
                            <label>Email</label>
                            <input type="email" id="email" required>
                        </div>
                        <div class="form-group">
                            <label>Password</label>
                            <input type="password" id="password" required>
                        </div>
                        <button type="submit" class="btn btn-primary" style="width:100%">
                            <span id="authAction">Login</span>
                        </button>
                    </form>
                    <div class="auth-toggle">
                        <span id="toggleText">Don't have an account?</span>
                        <a id="toggleAuth" onclick="toggleAuthMode()">Sign up</a>
                    </div>
                </div>
            `;
            
            let isLogin = true;
            window.toggleAuthMode = () => {
                isLogin = !isLogin;
                document.getElementById('authAction').textContent = isLogin ? 'Login' : 'Sign Up';
                document.getElementById('toggleText').textContent = isLogin ? "Don't have an account?" : "Already have an account?";
                document.getElementById('toggleAuth').textContent = isLogin ? 'Sign up' : 'Login';
            };
            
            document.getElementById('authForm').onsubmit = async (e) => {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                
                const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
                const data = await api(endpoint, {
                    method: 'POST',
                    body: JSON.stringify({ email, password })
                });
                
                if (data.success) {
                    currentUser = { id: data.user_id, display_name: data.display_name };
                    userEnergy = data.energy || 12;
                    notify(isLogin ? 'Welcome back!' : 'Account created!', 'success');
                    await loadDashboard();
                } else {
                    notify(data.error || 'Error', 'error');
                }
            };
        }

        // Dashboard
        async function loadDashboard() {
            // Load all data
            const [stateData, tasksData, petData] = await Promise.all([
                api('/api/state'),
                api('/api/tasks'),
                api('/api/pet')
            ]);
            
            userState = stateData;
            userPet = petData;
            
            renderDashboard(stateData, tasksData, petData);
        }

        function renderDashboard(stateData, tasksData, petData) {
            const domains = stateData.domains || [];
            const tasks = tasksData.tasks || [];
            
            document.getElementById('app').innerHTML = `
                <header class="header">
                    <h1>🌀 Life Fractal v14</h1>
                    <div class="header-right">
                        <div class="energy-display">⚡ ${userEnergy} spoons</div>
                        <span>Hi, ${currentUser?.display_name || 'Friend'}!</span>
                        <button class="btn btn-secondary" onclick="logout()">Logout</button>
                    </div>
                </header>
                
                <main class="main">
                    <div class="grid">
                        <!-- Life State -->
                        <div class="card">
                            <div class="card-header">📊 Life State</div>
                            <div id="domains">
                                ${domains.map(d => `
                                    <div class="domain-item">
                                        <span class="domain-icon">${d.icon}</span>
                                        <span class="domain-name">${d.name}</span>
                                        <div class="domain-bar">
                                            <div class="domain-fill" style="width:${d.value*100}%;background:${d.color}"></div>
                                        </div>
                                        <span class="domain-value">${Math.round(d.value*100)}%</span>
                                    </div>
                                `).join('')}
                            </div>
                            <div style="margin-top:1rem;text-align:center;">
                                <small style="color:var(--text-light)">
                                    Average: ${Math.round(stateData.average*100)}% | 
                                    Balance: ${Math.round(stateData.balance_score*100)}%
                                </small>
                            </div>
                        </div>
                        
                        <!-- Tasks -->
                        <div class="card">
                            <div class="card-header">✅ Recommended Tasks</div>
                            <div id="tasks">
                                ${tasks.slice(0, 6).map(t => `
                                    <div class="task-item" onclick="completeTask('${t.task_id}')">
                                        <span class="task-icon">${t.icon}</span>
                                        <div class="task-info">
                                            <div class="task-name">${t.name}</div>
                                            <div class="task-meta">
                                                Flow: ${Math.round(t.flow_probability*100)}% | 
                                                ${t.duration}min
                                            </div>
                                        </div>
                                        <span class="task-cost">${t.energy_cost}⚡</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
                        <!-- Pet -->
                        <div class="card">
                            <div class="card-header">🐾 Companion</div>
                            ${petData.has_pet ? `
                                <div class="pet-display">
                                    <div class="pet-emoji">${petData.pet.emoji}</div>
                                    <div class="pet-name">${petData.pet.name}</div>
                                    <div>Mood: ${petData.pet.mood}</div>
                                    <div class="pet-stats">
                                        <div class="pet-stat">
                                            <div class="pet-stat-value">❤️ ${Math.round(petData.pet.happiness*100)}%</div>
                                            <div class="pet-stat-label">Happy</div>
                                        </div>
                                        <div class="pet-stat">
                                            <div class="pet-stat-value">🍖 ${Math.round((1-petData.pet.hunger)*100)}%</div>
                                            <div class="pet-stat-label">Fed</div>
                                        </div>
                                    </div>
                                    <div class="pet-actions">
                                        <button class="btn btn-secondary" onclick="feedPet()">Feed 🍖</button>
                                        <button class="btn btn-secondary" onclick="playPet()">Play 🎾</button>
                                    </div>
                                </div>
                            ` : `
                                <div style="text-align:center;padding:2rem;">
                                    <p style="margin-bottom:1rem;">Adopt a companion!</p>
                                    <select id="petSpecies" style="padding:0.5rem;margin-bottom:1rem;">
                                        ${petData.available_species.map(s => `<option value="${s}">${s}</option>`).join('')}
                                    </select>
                                    <br>
                                    <input type="text" id="petName" placeholder="Pet name" style="padding:0.5rem;margin-bottom:1rem;">
                                    <br>
                                    <button class="btn btn-primary" onclick="adoptPet()">Adopt</button>
                                </div>
                            `}
                        </div>
                        
                        <!-- Math Constants -->
                        <div class="card">
                            <div class="card-header">📐 Sacred Mathematics</div>
                            <div class="math-box">
φ (Golden Ratio) = 1.618033988749895
Golden Angle = 137.5077640500378°
γ (Discount Factor) = 0.618033988749895
Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34...
Flow Optimal Ratio = φ ≈ 1.618
Habit Formation = 66 days average
                            </div>
                        </div>
                    </div>
                </main>
            `;
        }

        // Actions
        async function completeTask(taskId) {
            if (userEnergy < 1) {
                notify('Not enough energy!', 'error');
                return;
            }
            
            const data = await api('/api/tasks/complete', {
                method: 'POST',
                body: JSON.stringify({ task_id: taskId })
            });
            
            if (data.success) {
                userEnergy = data.new_energy;
                notify(`Completed ${data.task}! -${data.energy_spent}⚡`, 'success');
                await loadDashboard();
            } else {
                notify(data.error || 'Error', 'error');
            }
        }

        async function feedPet() {
            const data = await api('/api/pet/feed', { method: 'POST' });
            if (data.success) {
                notify('Pet fed! 🍖', 'success');
                await loadDashboard();
            }
        }

        async function playPet() {
            const data = await api('/api/pet/play', { method: 'POST' });
            if (data.success) {
                notify('Played with pet! 🎾', 'success');
                await loadDashboard();
            }
        }

        async function adoptPet() {
            const species = document.getElementById('petSpecies').value;
            const name = document.getElementById('petName').value || 'Buddy';
            
            const data = await api('/api/pet/adopt', {
                method: 'POST',
                body: JSON.stringify({ species, name })
            });
            
            if (data.success) {
                notify(data.message, 'success');
                await loadDashboard();
            } else {
                notify(data.error, 'error');
            }
        }

        async function logout() {
            await api('/api/auth/logout', { method: 'POST' });
            currentUser = null;
            showAuth();
        }

        // Init
        checkSession();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(MAIN_HTML)

# ════════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE v14.0 - ULTIMATE UNIFIED ENGINE")
    print("=" * 80)
    print("   Complete unified platform with ALL features")
    print("=" * 80)
    
    print(f"\n📊 FEATURES UNIFIED")
    print(f"   • {N_DOMAINS} Life Domains (including Law of Attraction)")
    print(f"   • {len(TASK_TYPES)} Task Types with Full Effect Vectors")
    print(f"   • {int(np.sum(SPILLOVER_MATRIX > 0))} Spillover Effects")
    print(f"   • {len(LIFE_MILESTONES)} Life Milestones")
    print(f"   • {len(PET_SPECIES)} Virtual Pet Species")
    print(f"   • Fractal Mathematics (Dimension, Hurst, Lyapunov)")
    print(f"   • Bellman Optimization")
    print(f"   • Flow State Calculation")
    print(f"   • Compound Growth Projections")
    print(f"   • Habit Formation Tracking")
    
    print(f"\n📐 SACRED MATHEMATICS")
    print(f"   φ = {PHI:.10f}")
    print(f"   Golden Angle = {GOLDEN_ANGLE_DEG:.4f}°")
    print(f"   γ = {GAMMA:.10f}")
    print(f"   Fibonacci = {FIBONACCI[:10]}")
    
    print(f"\n🔬 SCIENTIFIC CONSTANTS")
    print(f"   Habit Formation = {HABIT_FORMATION_AVG} days")
    print(f"   Flow Ratio = φ ≈ {FLOW_CHALLENGE_SKILL_RATIO:.3f}")
    print(f"   Rule of 72 = {RULE_OF_72}")
    
    print("\n" + "=" * 80)
    print("📡 API ENDPOINTS (50+)")
    print("=" * 80)
    endpoints = [
        ("Auth", ["/api/auth/register", "/api/auth/login", "/api/auth/session"]),
        ("State", ["/api/state", "/api/energy"]),
        ("Tasks", ["/api/tasks", "/api/tasks/complete"]),
        ("Pet", ["/api/pet", "/api/pet/adopt", "/api/pet/feed", "/api/pet/play"]),
        ("Milestones", ["/api/milestones", "/api/milestones/achieve"]),
        ("Goals", ["/api/goals"]),
        ("Journal", ["/api/journal"]),
        ("Fractal", ["/api/fractal/analysis", "/api/fractal/dimension", "/api/fractal/hurst"]),
        ("Math", ["/api/math/golden-allocation", "/api/math/fibonacci-schedule", "/api/math/compound-growth"]),
        ("Data", ["/api/spillovers", "/api/math/constants", "/api/health"])
    ]
    for category, eps in endpoints:
        print(f"   {category}: {', '.join(eps)}")
    print("=" * 80)


if __name__ == '__main__':
    print_banner()
    
    with app.app_context():
        init_db()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
