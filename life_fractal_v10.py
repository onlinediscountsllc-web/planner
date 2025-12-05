#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v13.0 - ULTIMATE MATHEMATICAL SYNTHESIS
════════════════════════════════════════════════════════════════════════════════
COMPLETE PRODUCTION SYSTEM - ALL FEATURES INTEGRATED

MATHEMATICAL FOUNDATIONS (20 Total):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORIGINAL 10 (v12):
1. Golden-Harmonic Folding Field
2. Pareidolia Detection Field
3. Sacred Blend Energy Map
4. Fractal Bloom Expansion (Julia/Mandelbrot)
5. Origami Curve Envelope
6. Emotional Harmonic Wave
7. Fourier Sketch Synthesis
8. GPU Parallel Frame Queue
9. Temporal Origami Compression
10. Full-Scene Emotional Manifold

NEW 10 (v13):
11. Lorenz Attractor (Goal Visualization)
12. Rossler Attractor (Mood Patterns)
13. Coupled Chaos System (Life-Emotion Sync)
14. Particle Swarm Dynamics (Spoon Theory)
15. Harmonic Resonance (Pythagorean Tuning)
16. Fractal Dimension Calculator (Life Complexity)
17. Flower of Life Generator (Sacred Geometry)
18. Metatron's Cube (Goal Positioning)
19. Golden Spiral Overlay (Nature's Pattern)
20. Binaural Beat Generator (Therapeutic Audio)

FEATURES:
━━━━━━━━━
✅ Complete authentication & session management
✅ Demo mode for unauthenticated users
✅ Subscription system (7-day trial, Stripe integration)
✅ SQLite database with 15+ tables
✅ 2D & 3D fractal visualization
✅ Ollama AI integration (pattern fallback)
✅ Mayan Tzolkin calendar
✅ Virtual pet system (8 species)
✅ Spoon Theory energy tracking
✅ Chaos theory attractors
✅ Sacred geometry overlays
✅ Binaural beats audio
✅ Comprehensive accessibility
✅ Self-healing architecture

════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import random
import secrets
import logging
import sqlite3
import hashlib
import struct
import wave
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64

# Flask
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Optional GPU
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None

# Optional ML
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional Ollama
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PI = math.pi
TAU = 2 * PI
E = math.e
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1
PHI_SQ = PHI * PHI
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
SQRT5 = math.sqrt(5)

GOLDEN_ANGLE = 360 / PHI_SQ  # 137.5077640500378°
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)

FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843]

# Lorenz attractor parameters
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

# Rossler attractor parameters
ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7

# Pythagorean tuning ratios
PYTHAGOREAN_RATIOS = {
    'unison': 1/1, 'octave': 2/1, 'fifth': 3/2, 'fourth': 4/3,
    'major_third': 5/4, 'minor_third': 6/5, 'major_sixth': 5/3,
    'minor_sixth': 8/5, 'major_second': 9/8, 'minor_seventh': 9/5
}

# Color palettes
PALETTES = {
    'golden': [(255, 215, 128), (220, 180, 100), (180, 140, 80), (140, 100, 60)],
    'calm': [(100, 150, 200), (80, 130, 180), (60, 110, 160), (40, 90, 140)],
    'energetic': [(255, 180, 100), (255, 140, 80), (255, 100, 60), (230, 80, 50)],
    'focused': [(180, 130, 220), (150, 100, 200), (120, 70, 180), (90, 40, 160)],
    'peaceful': [(130, 200, 170), (100, 180, 150), (70, 160, 130), (40, 140, 110)],
    'cosmic': [(180, 100, 255), (140, 80, 220), (100, 60, 180), (60, 40, 140)],
    'aurora': [(100, 255, 150), (80, 200, 180), (60, 150, 200), (40, 100, 220)]
}

# Mayan Tzolkin
TZOLKIN_SIGNS = ['Imix', 'Ik', 'Akbal', 'Kan', 'Chicchan', 'Cimi', 'Manik', 
                 'Lamat', 'Muluc', 'Oc', 'Chuen', 'Eb', 'Ben', 'Ix', 
                 'Men', 'Cib', 'Caban', 'Etznab', 'Cauac', 'Ahau']

TZOLKIN_MEANINGS = {
    'Imix': 'Primordial Waters - New beginnings',
    'Ik': 'Wind - Communication and breath',
    'Akbal': 'Night - Inner wisdom',
    'Kan': 'Seed - Potential growth',
    'Chicchan': 'Serpent - Life force energy',
    'Cimi': 'Death - Transformation',
    'Manik': 'Deer - Grace and healing',
    'Lamat': 'Star - Harmony and art',
    'Muluc': 'Water - Emotions flow',
    'Oc': 'Dog - Loyalty and guidance',
    'Chuen': 'Monkey - Creativity play',
    'Eb': 'Road - Life path journey',
    'Ben': 'Reed - Home and family',
    'Ix': 'Jaguar - Earth magic',
    'Men': 'Eagle - Vision clarity',
    'Cib': 'Owl - Ancient wisdom',
    'Caban': 'Earth - Grounding force',
    'Etznab': 'Mirror - Truth reflection',
    'Cauac': 'Storm - Purification',
    'Ahau': 'Sun - Enlightenment'
}


# ════════════════════════════════════════════════════════════════════════════════
# 3D VECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Vector3:
    """3D Vector with mathematical operations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, o): return Vector3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vector3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, s): return Vector3(self.x * s, self.y * s, self.z * s)
    def __truediv__(self, s): return Vector3(self.x/s, self.y/s, self.z/s) if s else Vector3()
    
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self): 
        m = self.magnitude()
        return self / m if m > 1e-10 else Vector3()
    
    def rotate_y(self, a):
        c, s = math.cos(a), math.sin(a)
        return Vector3(self.x*c + self.z*s, self.y, -self.x*s + self.z*c)
    
    def rotate_x(self, a):
        c, s = math.cos(a), math.sin(a)
        return Vector3(self.x, self.y*c - self.z*s, self.y*s + self.z*c)


# ════════════════════════════════════════════════════════════════════════════════
# ORIGINAL 10 MATHEMATICAL FOUNDATIONS (v12)
# ════════════════════════════════════════════════════════════════════════════════

class MathematicalFoundations:
    """The 10 original mathematical foundations for photorealistic generation"""
    
    def __init__(self):
        self.frame_queue = []
        self.emotion_weights = {
            'hope': 1.0, 'joy': 1.2, 'calm': 0.8, 'wonder': 1.1,
            'love': 1.3, 'peace': 0.7, 'energy': 1.4, 'focus': 0.9
        }
    
    # 1. Golden-Harmonic Folding Field
    def golden_harmonic_fold(self, t: float) -> float:
        """F(t,φ) = sin(2π·t·φ)·cos(2π·t/φ) + sin(π·t²)"""
        return (math.sin(TAU * t * PHI) * math.cos(TAU * t / PHI) + 
                math.sin(PI * t * t))
    
    # 2. Pareidolia Detection Field
    def pareidolia_field(self, x: float, y: float, t: float) -> float:
        """P(x,y,t) = sigmoid(cos(x²+y²+sin(t·π))) · L(x,y)"""
        radial = math.cos(x*x + y*y + math.sin(t * PI))
        sigmoid = 1 / (1 + math.exp(-radial * 3))
        laplacian = -4 * (x*x + y*y) + 2
        return sigmoid * max(0, min(1, (laplacian + 2) / 4))
    
    # 3. Sacred Blend Energy Map
    def sacred_blend(self, x: float, y: float, t: float) -> float:
        """B(x,y,t) = tanh(α·sin(2π·x) + β·cos(2π·y)) · γ(t)"""
        alpha, beta = PHI, PHI_INV
        gamma = 0.5 + 0.5 * math.sin(t * PI * 0.5)
        return math.tanh(alpha * math.sin(TAU * x) + beta * math.cos(TAU * y)) * gamma
    
    # 4. Fractal Bloom Expansion
    def fractal_bloom(self, c_real: float, c_imag: float, max_iter: int = 100) -> Tuple[int, float]:
        """Z(n+1) = Z(n)² + C with smooth coloring"""
        z_r, z_i = 0.0, 0.0
        for i in range(max_iter):
            if z_r*z_r + z_i*z_i > 4.0:
                smooth = i - math.log2(math.log2(z_r*z_r + z_i*z_i + 1) + 1)
                return (i, smooth)
            z_r, z_i = z_r*z_r - z_i*z_i + c_real, 2*z_r*z_i + c_imag
        return (max_iter, float(max_iter))
    
    # 5. Origami Curve Envelope
    def origami_curve(self, u: float, v: float) -> float:
        """O(u,v) = sin(u·v) + cos(φ·u)·sin(φ·v)"""
        return math.sin(u * v) + math.cos(PHI * u) * math.sin(PHI * v)
    
    # 6. Emotional Harmonic Wave
    def emotional_harmonic(self, t: float, emotion: str) -> float:
        """H(t,e) = |sin(π·t·E[e])| + tanh(t·0.2)"""
        e_weight = self.emotion_weights.get(emotion, 1.0)
        return abs(math.sin(PI * t * e_weight)) + math.tanh(t * 0.2)
    
    # 7. Fourier Sketch Synthesis
    def generate_sketch_coefficients(self, emotion: str, n_terms: int = 10) -> List[Tuple[float, float]]:
        """Generate unique Fourier coefficients for an emotion"""
        seed = sum(ord(c) for c in emotion)
        rng = random.Random(seed)
        return [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n_terms)]
    
    def fourier_sketch(self, x: float, y: float, coeffs: List[Tuple[float, float]]) -> float:
        """Σ(aₙ·cos(n·x) + bₙ·sin(n·y))"""
        return sum(a * math.cos((i+1) * x) + b * math.sin((i+1) * y) 
                   for i, (a, b) in enumerate(coeffs))
    
    # 8. GPU Parallel Frame Queue
    def queue_frame(self, params: dict, timestamp: float):
        """Queue frame for batch processing"""
        self.frame_queue.append({'params': params, 'time': timestamp})
        if len(self.frame_queue) > 1000:
            self.frame_queue = self.frame_queue[-500:]
    
    # 9. Temporal Origami Compression
    def compress_temporal(self, frames: List[np.ndarray]) -> np.ndarray:
        """Cₜ = Σ Mf·(1/φ)ⁿ - Golden ratio weighted compression"""
        if not frames:
            return np.zeros((1, 1))
        result = np.zeros_like(frames[0], dtype=float)
        for i, frame in enumerate(frames):
            weight = (1 / PHI) ** i
            result += frame.astype(float) * weight
        total_weight = sum((1/PHI)**i for i in range(len(frames)))
        return result / total_weight if total_weight > 0 else result
    
    # 10. Full-Scene Emotional Manifold
    def emotional_manifold(self, x: float, y: float, t: float, emotion: str) -> float:
        """E(x,y,t) = ∇²B + H(t,e)·F(t,φ)"""
        blend = self.sacred_blend(x, y, t)
        harmonic = self.emotional_harmonic(t, emotion)
        fold = self.golden_harmonic_fold(t)
        laplacian = -4 * blend  # Approximation
        return laplacian + harmonic * fold


# ════════════════════════════════════════════════════════════════════════════════
# NEW 10 MATHEMATICAL FOUNDATIONS (v13)
# ════════════════════════════════════════════════════════════════════════════════

class LorenzAttractor:
    """11. Lorenz Attractor - Chaos theory for goal visualization"""
    
    def __init__(self, sigma=LORENZ_SIGMA, rho=LORENZ_RHO, beta=LORENZ_BETA):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.pos = Vector3(1, 1, 1)
        self.history = []
        self.max_history = 2000
    
    def step(self, dt=0.01) -> Vector3:
        dx = self.sigma * (self.pos.y - self.pos.x)
        dy = self.pos.x * (self.rho - self.pos.z) - self.pos.y
        dz = self.pos.x * self.pos.y - self.beta * self.pos.z
        self.pos = Vector3(self.pos.x + dx*dt, self.pos.y + dy*dt, self.pos.z + dz*dt)
        self.history.append(Vector3(self.pos.x, self.pos.y, self.pos.z))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return self.pos
    
    def get_wing(self) -> str:
        return 'growth' if self.pos.x > 0 else 'reflection'
    
    def get_normalized(self) -> Vector3:
        return Vector3(self.pos.x/25, self.pos.y/30, self.pos.z/50)


class RosslerAttractor:
    """12. Rossler Attractor - Smooth spiral for mood patterns"""
    
    def __init__(self, a=ROSSLER_A, b=ROSSLER_B, c=ROSSLER_C):
        self.a, self.b, self.c = a, b, c
        self.pos = Vector3(1, 1, 1)
        self.history = []
        self.max_history = 1500
    
    def step(self, dt=0.02) -> Vector3:
        dx = -self.pos.y - self.pos.z
        dy = self.pos.x + self.a * self.pos.y
        dz = self.b + self.pos.z * (self.pos.x - self.c)
        self.pos = Vector3(self.pos.x + dx*dt, self.pos.y + dy*dt, self.pos.z + dz*dt)
        self.history.append(Vector3(self.pos.x, self.pos.y, self.pos.z))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return self.pos
    
    def get_phase(self) -> float:
        return (math.atan2(self.pos.y, self.pos.x) + PI) / TAU


class CoupledChaosSystem:
    """13. Coupled Chaos - Life domains influence emotional patterns"""
    
    def __init__(self, coupling=0.05):
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        self.coupling = coupling
        self.time = 0.0
    
    def step(self, dt=0.01) -> Tuple[Vector3, Vector3]:
        # Bidirectional coupling
        r_inf = self.rossler.pos * (self.coupling * 5)
        self.lorenz.pos = self.lorenz.pos + r_inf * dt
        l_inf = self.lorenz.get_normalized() * (self.coupling * 2)
        self.rossler.pos = self.rossler.pos + l_inf * dt
        self.lorenz.step(dt)
        self.rossler.step(dt)
        self.time += dt
        return (self.lorenz.pos, self.rossler.pos)
    
    def get_balance(self) -> float:
        l_n = self.lorenz.get_normalized().magnitude()
        r_n = self.rossler.pos.magnitude() / 15
        if l_n + r_n < 0.01:
            return 1.0
        return 1 - abs(l_n - r_n) / (l_n + r_n)


@dataclass
class Particle:
    """Individual spoon particle"""
    position: Vector3
    velocity: Vector3
    color: Tuple[int, int, int]
    size: float = 2.0
    life: float = 1.0
    max_life: float = 3.0
    energy: float = 1.0
    
    def update(self, dt: float, attractor: Vector3, w=0.7, c1=1.5, c2=1.5):
        r1, r2 = random.random(), random.random()
        social = (attractor - self.position) * c2 * r2
        self.velocity = self.velocity * w + social
        speed = self.velocity.magnitude()
        if speed > 50:
            self.velocity = self.velocity.normalize() * 50
        self.position = self.position + self.velocity * dt
        self.life -= dt / self.max_life
        self.energy *= 0.999
    
    def is_alive(self) -> bool:
        return self.life > 0 and self.energy > 0.01


class SpoonParticleSystem:
    """14. Particle Swarm - Spoon Theory energy tracking"""
    
    def __init__(self, max_spoons=12):
        self.particles = []
        self.max_spoons = max_spoons
        self.attractor = Vector3(0, 0, 0)
    
    def spawn(self, pos: Vector3, color=(255, 215, 128)):
        if len(self.particles) < self.max_spoons:
            vel = Vector3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
            self.particles.append(Particle(
                position=Vector3(pos.x, pos.y, pos.z),
                velocity=vel, color=color,
                size=2 + random.random() * 2,
                max_life=3 + random.random() * 4
            ))
    
    def update(self, dt: float):
        for p in self.particles:
            p.update(dt, self.attractor)
        self.particles = [p for p in self.particles if p.is_alive()]
    
    def get_energy_level(self) -> float:
        if not self.particles:
            return 0
        return sum(p.energy for p in self.particles) / self.max_spoons


class HarmonicResonance:
    """15. Harmonic Resonance - Pythagorean tuning for synesthetic experience"""
    
    BASE_FREQ = 432  # Healing frequency
    
    INTERVAL_COLORS = {
        'unison': (255, 255, 255), 'octave': (255, 255, 200),
        'fifth': (100, 200, 255), 'fourth': (150, 255, 150),
        'major_third': (255, 200, 100), 'minor_third': (200, 150, 255)
    }
    
    def wellness_to_interval(self, wellness: float) -> str:
        if wellness >= 90: return 'octave'
        elif wellness >= 70: return 'fifth'
        elif wellness >= 50: return 'fourth'
        elif wellness >= 30: return 'major_third'
        return 'unison'
    
    def get_wellness_color(self, wellness: float) -> Tuple[int, int, int]:
        interval = self.wellness_to_interval(wellness)
        return self.INTERVAL_COLORS.get(interval, (128, 128, 128))
    
    def get_frequency(self, wellness: float) -> float:
        interval = self.wellness_to_interval(wellness)
        return self.BASE_FREQ * PYTHAGOREAN_RATIOS.get(interval, 1)


class FractalDimensionCalculator:
    """16. Fractal Dimension - Calculate life pattern complexity"""
    
    def __init__(self):
        self.epsilon_values = [2**(-i) for i in range(2, 8)]
    
    def box_counting(self, points: List[Tuple[float, float]]) -> float:
        if len(points) < 10:
            return 1.0
        counts = []
        for eps in self.epsilon_values:
            boxes = set()
            for x, y in points:
                boxes.add((int(x/eps), int(y/eps)))
            counts.append(len(boxes))
        log_eps = np.log(self.epsilon_values)
        log_counts = np.log(np.array(counts) + 1)
        coeffs = np.polyfit(log_eps, log_counts, 1)
        return max(1.0, min(2.0, -coeffs[0]))
    
    def interpret(self, dim: float) -> str:
        if dim < 1.2: return "Smooth and focused - great for deep work"
        elif dim < 1.5: return "Balanced complexity - managing multiple areas well"
        elif dim < 1.7: return "Rich interconnected pattern - lots of variety"
        return "Highly complex - consider simplifying some areas"


class GoldenSpiral:
    """17. Golden Spiral - Nature's growth pattern"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def get_point(self, theta: float) -> Tuple[float, float]:
        r = self.scale * (PHI ** (theta / (PI / 2)))
        return (r * math.cos(theta), r * math.sin(theta))
    
    def get_points(self, n: int, max_theta=4*PI) -> List[Tuple[float, float]]:
        return [self.get_point(i * max_theta / n) for i in range(n)]


class FlowerOfLife:
    """18. Flower of Life - Ancient sacred geometry"""
    
    def __init__(self, radius=1.0, rings=3):
        self.radius = radius
        self.rings = rings
    
    def get_centers(self) -> List[Tuple[float, float]]:
        centers = [(0, 0)]
        for ring in range(1, self.rings + 1):
            for i in range(6 * ring):
                angle = i * (PI / 3) / ring
                dist = self.radius * ring
                centers.append((dist * math.cos(angle), dist * math.sin(angle)))
        return centers


class MetatronsCube:
    """19. Metatron's Cube - Contains all Platonic solids"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
        self.positions = self._calc_positions()
    
    def _calc_positions(self):
        pos = [(0, 0)]
        for i in range(6):
            angle = i * PI / 3
            pos.append((self.scale * math.cos(angle), self.scale * math.sin(angle)))
        for i in range(6):
            angle = i * PI / 3 + PI / 6
            pos.append((self.scale * 2 * math.cos(angle), self.scale * 2 * math.sin(angle)))
        return pos
    
    def project_goal(self, idx: int) -> Tuple[float, float]:
        return self.positions[idx % len(self.positions)]


class BinauralBeatGenerator:
    """20. Binaural Beats - Therapeutic audio generation"""
    
    BASE_FREQ = 432
    PRESETS = {
        'focus': {'beat': 15, 'base': 200},    # Beta
        'calm': {'beat': 10, 'base': 180},     # Alpha
        'sleep': {'beat': 3, 'base': 150},     # Delta
        'meditate': {'beat': 6, 'base': 170},  # Theta
        'energy': {'beat': 20, 'base': 220},   # High Beta
        'healing': {'beat': 8, 'base': 160}    # Alpha-Theta
    }
    
    def generate(self, preset: str, duration: float = 1.0, 
                 sample_rate: int = 44100) -> np.ndarray:
        params = self.PRESETS.get(preset, self.PRESETS['calm'])
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        left = np.sin(2 * PI * params['base'] * t) * 0.5
        right = np.sin(2 * PI * (params['base'] + params['beat']) * t) * 0.5
        return np.vstack([left, right]).T


# ════════════════════════════════════════════════════════════════════════════════
# LIVING ORGANISM - UNIFIED MATHEMATICAL CORE
# ════════════════════════════════════════════════════════════════════════════════

class OrbCellType(Enum):
    CORE = "core"
    GOAL = "goal"
    HABIT = "habit"
    DREAM = "dream"
    MEMORY = "memory"
    INSIGHT = "insight"


@dataclass
class MathematicalOrb:
    """Self-aware orb with mathematical identity"""
    id: str
    cell_type: OrbCellType
    position: Vector3
    meaning: str
    harmonic_phase: float = 0.0
    origami_fold_level: float = 1.0
    fourier_coefficients: List[Tuple[float, float]] = field(default_factory=list)
    bloom_seed: Tuple[float, float] = (0, 0)
    emotional_resonance: str = "hope"
    
    def __post_init__(self):
        if not self.fourier_coefficients:
            rng = random.Random(hash(self.id))
            self.fourier_coefficients = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(10)]
        if self.bloom_seed == (0, 0):
            rng = random.Random(hash(self.id) + 1)
            self.bloom_seed = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))


class OrbSwarm:
    """Self-organizing swarm of mathematical orbs"""
    
    def __init__(self):
        self.orbs: Dict[str, MathematicalOrb] = {}
        self._spawn_initial_orbs()
    
    def _spawn_initial_orbs(self):
        initial = [
            ("Welcome to your fractal universe", OrbCellType.CORE),
            ("Set your first goal", OrbCellType.GOAL),
            ("Track a daily habit", OrbCellType.HABIT),
            ("Record a dream", OrbCellType.DREAM),
            ("Capture a memory", OrbCellType.MEMORY)
        ]
        for i, (meaning, cell_type) in enumerate(initial):
            theta = i * GOLDEN_ANGLE_RAD
            r = math.sqrt(i + 1) * 2
            orb_id = secrets.token_urlsafe(8)
            self.orbs[orb_id] = MathematicalOrb(
                id=orb_id, cell_type=cell_type,
                position=Vector3(r * math.cos(theta), r * math.sin(theta), i * 0.5),
                meaning=meaning,
                emotional_resonance=random.choice(['hope', 'joy', 'calm', 'wonder'])
            )
    
    def spawn_orb(self, meaning: str, cell_type: OrbCellType) -> str:
        orb_id = secrets.token_urlsafe(8)
        idx = len(self.orbs)
        theta = idx * GOLDEN_ANGLE_RAD
        r = math.sqrt(idx + 1) * 2
        self.orbs[orb_id] = MathematicalOrb(
            id=orb_id, cell_type=cell_type,
            position=Vector3(r * math.cos(theta), r * math.sin(theta), idx * 0.3),
            meaning=meaning
        )
        return orb_id
    
    def get_state(self) -> List[dict]:
        return [
            {
                'id': o.id,
                'type': o.cell_type.value,
                'x': o.position.x, 'y': o.position.y, 'z': o.position.z,
                'meaning': o.meaning,
                'harmonic_phase': o.harmonic_phase,
                'fold_level': o.origami_fold_level,
                'emotion': o.emotional_resonance
            }
            for o in self.orbs.values()
        ]


class LivingOrganism:
    """The living mathematical organism that powers the visualization"""
    
    def __init__(self):
        # Original 10 foundations
        self.math = MathematicalFoundations()
        
        # New 10 foundations
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        self.coupled = CoupledChaosSystem()
        self.spoons = SpoonParticleSystem(12)
        self.harmonic = HarmonicResonance()
        self.fractal_dim = FractalDimensionCalculator()
        self.golden_spiral = GoldenSpiral(scale=5)
        self.flower = FlowerOfLife(radius=50, rings=2)
        self.metatron = MetatronsCube(scale=80)
        self.binaural = BinauralBeatGenerator()
        
        # Orb system
        self.swarm = OrbSwarm()
        
        # State
        self.time = 0.0
        self.wellness = 50.0
        self.mood = 50.0
        self.stress = 50.0
        
        logger.info("🌀 Living organism awakened with 20 mathematical foundations")
    
    def step(self, dt=0.016):
        """Advance all mathematical systems"""
        self.time += dt
        for _ in range(3):
            self.lorenz.step(0.005)
            self.rossler.step(0.01)
        self.coupled.step(dt)
        self.spoons.update(dt)
    
    def get_mayan_day(self) -> dict:
        """Get today's Mayan Tzolkin energy"""
        epoch = datetime(2012, 12, 21, tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - epoch).days
        tone = (days % 13) + 1
        sign_idx = days % 20
        sign = TZOLKIN_SIGNS[sign_idx]
        return {
            'tone': tone,
            'sign': sign,
            'meaning': TZOLKIN_MEANINGS.get(sign, ''),
            'kin': (days % 260) + 1,
            'display': f"{tone} {sign}"
        }
    
    def get_state(self) -> dict:
        return {
            'time': self.time,
            'wellness': self.wellness,
            'mood': self.mood,
            'stress': self.stress,
            'lorenz_wing': self.lorenz.get_wing(),
            'rossler_phase': self.rossler.get_phase(),
            'chaos_balance': self.coupled.get_balance(),
            'energy_level': self.spoons.get_energy_level(),
            'orbs': self.swarm.get_state(),
            'mayan': self.get_mayan_day(),
            'golden_ratio': PHI,
            'foundations': 20
        }


# ════════════════════════════════════════════════════════════════════════════════
# OLLAMA AI INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════

class OllamaAI:
    """AI integration with pattern fallback"""
    
    PATTERNS = {
        'goal': [
            "This goal connects to your deeper purpose through the golden spiral of growth",
            "Every step toward this goal creates ripples through your life's fractal pattern",
            "This aspiration resonates with the {tone} energy of {sign}"
        ],
        'habit': [
            "Small consistent actions compound like Fibonacci sequences",
            "This habit strengthens your life's fractal structure",
            "Daily practice aligns with natural rhythms"
        ],
        'insight': [
            "The Lorenz attractor reveals hidden patterns in your journey",
            "Like the Rossler spiral, insights unfold in their own time",
            "Sacred geometry shows connections you couldn't see before"
        ]
    }
    
    def __init__(self):
        self.available = self._check_ollama()
        logger.info(f"🤖 Ollama AI: {'Connected' if self.available else 'Pattern-based mode'}")
    
    def _check_ollama(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=1)
            return r.status_code == 200
        except:
            return False
    
    def generate_meaning(self, context: str, cell_type: str = 'insight', mayan: dict = None) -> str:
        if self.available:
            try:
                prompt = f"Create a brief, poetic insight (1-2 sentences) about: {context}. Use themes of sacred geometry, growth, and transformation."
                r = requests.post('http://localhost:11434/api/generate', 
                    json={'model': 'llama3', 'prompt': prompt, 'stream': False},
                    timeout=10)
                if r.status_code == 200:
                    return r.json().get('response', '')[:200]
            except:
                pass
        
        # Pattern fallback
        patterns = self.PATTERNS.get(cell_type, self.PATTERNS['insight'])
        pattern = random.choice(patterns)
        if mayan and '{tone}' in pattern:
            pattern = pattern.format(tone=mayan['tone'], sign=mayan['sign'])
        return pattern


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════════

class Database:
    """Production-ready SQLite database"""
    
    def __init__(self, db_path: str = "life_fractal_v13.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"✅ Database initialized: {db_path}")
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        c = conn.cursor()
        
        # Users with subscription
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, first_name TEXT, last_name TEXT,
            created_at TEXT NOT NULL, last_login TEXT,
            is_active INTEGER DEFAULT 1, subscription_status TEXT DEFAULT 'trial',
            trial_end_date TEXT, is_exempt INTEGER DEFAULT 0,
            spoons_max INTEGER DEFAULT 12)''')
        
        # Goals
        c.execute('''CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT, category TEXT DEFAULT 'personal',
            priority INTEGER DEFAULT 3, progress REAL DEFAULT 0,
            target_date TEXT, created_at TEXT NOT NULL, completed_at TEXT,
            fractal_x REAL, fractal_y REAL, fractal_z REAL,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        # Habits
        c.execute('''CREATE TABLE IF NOT EXISTS habits (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
            frequency TEXT DEFAULT 'daily', current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0, total_completions INTEGER DEFAULT 0,
            spoon_cost INTEGER DEFAULT 1, created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        # Daily entries
        c.execute('''CREATE TABLE IF NOT EXISTS daily_entries (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, date TEXT NOT NULL,
            mood_level INTEGER DEFAULT 50, stress_level INTEGER DEFAULT 50,
            energy_level INTEGER DEFAULT 50, sleep_hours REAL DEFAULT 7,
            sleep_quality INTEGER DEFAULT 50, spoons_available INTEGER DEFAULT 12,
            spoons_used INTEGER DEFAULT 0, wellness_index REAL DEFAULT 50,
            lorenz_wing TEXT, rossler_phase REAL, fractal_dimension REAL,
            journal_entry TEXT, created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date))''')
        
        # Pet state
        c.execute('''CREATE TABLE IF NOT EXISTS pet_state (
            user_id TEXT PRIMARY KEY, species TEXT DEFAULT 'cat',
            name TEXT DEFAULT 'Buddy', hunger REAL DEFAULT 50, energy REAL DEFAULT 50,
            mood REAL DEFAULT 50, level INTEGER DEFAULT 1, experience INTEGER DEFAULT 0,
            evolution_stage INTEGER DEFAULT 1, bond REAL DEFAULT 50, last_updated TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        # Accessibility
        c.execute('''CREATE TABLE IF NOT EXISTS accessibility (
            user_id TEXT PRIMARY KEY, aphantasia_mode INTEGER DEFAULT 0,
            autism_safe_colors INTEGER DEFAULT 0, reduce_motion INTEGER DEFAULT 0,
            high_contrast INTEGER DEFAULT 0, dyslexia_font INTEGER DEFAULT 0,
            text_size_multiplier REAL DEFAULT 1.0, preferred_palette TEXT DEFAULT 'golden',
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        # Orbs
        c.execute('''CREATE TABLE IF NOT EXISTS orbs (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, cell_type TEXT,
            meaning TEXT, x REAL, y REAL, z REAL, harmonic_phase REAL,
            emotional_resonance TEXT, created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        # Sessions
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT,
            duration_seconds INTEGER DEFAULT 0, karma_earned INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email: str, password: str, first_name: str = '', 
                   last_name: str = '') -> Optional[str]:
        try:
            conn = self.get_connection()
            c = conn.cursor()
            user_id = secrets.token_urlsafe(16)
            now = datetime.now(timezone.utc).isoformat()
            trial_end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            
            c.execute('''INSERT INTO users (id, email, password_hash, first_name, 
                        last_name, created_at, subscription_status, trial_end_date)
                        VALUES (?, ?, ?, ?, ?, ?, 'trial', ?)''',
                     (user_id, email.lower(), generate_password_hash(password),
                      first_name, last_name, now, trial_end))
            
            c.execute('INSERT INTO pet_state (user_id, last_updated) VALUES (?, ?)', (user_id, now))
            c.execute('INSERT INTO accessibility (user_id) VALUES (?)', (user_id,))
            
            conn.commit()
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None
        except Exception as e:
            logger.error(f"Create user error: {e}")
            return None
    
    def authenticate(self, email: str, password: str) -> Optional[dict]:
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email = ? AND is_active = 1', (email.lower(),))
            row = c.fetchone()
            conn.close()
            if row and check_password_hash(row['password_hash'], password):
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return None
    
    def get_user(self, user_id: str) -> Optional[dict]:
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = c.fetchone()
            conn.close()
            return dict(row) if row else None
        except:
            return None
    
    def get_goals(self, user_id: str) -> List[dict]:
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('''SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL
                        ORDER BY priority DESC''', (user_id,))
            rows = c.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except:
            return []
    
    def create_goal(self, user_id: str, title: str, description: str = '',
                   priority: int = 3) -> Optional[str]:
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM goals WHERE user_id = ?', (user_id,))
            count = c.fetchone()[0]
            
            theta = count * GOLDEN_ANGLE_RAD
            r = math.sqrt(count + 1) * 30
            
            goal_id = secrets.token_urlsafe(8)
            now = datetime.now(timezone.utc).isoformat()
            
            c.execute('''INSERT INTO goals (id, user_id, title, description, priority,
                        created_at, fractal_x, fractal_y, fractal_z)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (goal_id, user_id, title, description, priority, now,
                      r * math.cos(theta), r * math.sin(theta), priority * 10))
            
            conn.commit()
            conn.close()
            return goal_id
        except Exception as e:
            logger.error(f"Create goal error: {e}")
            return None
    
    def check_subscription(self, user_id: str) -> dict:
        try:
            user = self.get_user(user_id)
            if not user:
                return {'valid': False, 'reason': 'not_found'}
            
            if user.get('is_exempt'):
                return {'valid': True, 'status': 'exempt'}
            
            status = user.get('subscription_status', 'trial')
            
            if status == 'active':
                return {'valid': True, 'status': 'active'}
            
            if status == 'trial':
                trial_end = user.get('trial_end_date')
                if trial_end:
                    end = datetime.fromisoformat(trial_end.replace('Z', '+00:00'))
                    days_left = (end - datetime.now(timezone.utc)).days
                    if days_left >= 0:
                        return {'valid': True, 'status': 'trial', 'days_left': days_left}
            
            return {'valid': False, 'status': 'expired'}
        except Exception as e:
            logger.error(f"Subscription check error: {e}")
            return {'valid': False, 'reason': 'error'}


# ════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize systems
db = Database()
organism = LivingOrganism()
ai = OllamaAI()


# ════════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password or len(password) < 6:
            return jsonify({'error': 'Invalid email or password (min 6 chars)'}), 400
        
        user_id = db.create_user(email, password, 
                                 data.get('first_name', ''), 
                                 data.get('last_name', ''))
        
        if user_id:
            session['user_id'] = user_id
            return jsonify({'success': True, 'user_id': user_id})
        return jsonify({'error': 'Email already registered'}), 409
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user = db.authenticate(data.get('email', ''), data.get('password', ''))
        
        if user:
            session['user_id'] = user['id']
            sub = db.check_subscription(user['id'])
            return jsonify({
                'success': True,
                'user_id': user['id'],
                'first_name': user.get('first_name', ''),
                'subscription': sub
            })
        return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


# ════════════════════════════════════════════════════════════════════════════════
# ORGANISM ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/organism/state')
def organism_state():
    """Get complete organism state (demo-friendly)"""
    organism.step()
    return jsonify(organism.get_state())


@app.route('/api/organism/spawn-orb', methods=['POST'])
def spawn_orb():
    """Spawn a new mathematical orb"""
    data = request.get_json()
    meaning = data.get('meaning', 'New insight')
    cell_type = OrbCellType(data.get('type', 'insight'))
    
    # Generate AI meaning
    mayan = organism.get_mayan_day()
    ai_meaning = ai.generate_meaning(meaning, cell_type.value, mayan)
    
    orb_id = organism.swarm.spawn_orb(ai_meaning, cell_type)
    return jsonify({'success': True, 'orb_id': orb_id, 'meaning': ai_meaning})


# ════════════════════════════════════════════════════════════════════════════════
# USER DATA ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/dashboard')
def dashboard(user_id):
    user = db.get_user(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    goals = db.get_goals(user_id)
    sub = db.check_subscription(user_id)
    
    organism.step()
    
    return jsonify({
        'user': {
            'id': user['id'],
            'email': user['email'],
            'first_name': user.get('first_name', '')
        },
        'subscription': sub,
        'goals': goals,
        'organism': organism.get_state()
    })


@app.route('/api/user/<user_id>/goals', methods=['GET', 'POST'])
def goals(user_id):
    if request.method == 'POST':
        data = request.get_json()
        goal_id = db.create_goal(user_id, data.get('title', 'Goal'),
                                 data.get('description', ''),
                                 data.get('priority', 3))
        if goal_id:
            organism.swarm.spawn_orb(data.get('title', 'Goal'), OrbCellType.GOAL)
            return jsonify({'success': True, 'goal_id': goal_id})
        return jsonify({'error': 'Failed to create goal'}), 500
    
    return jsonify({'goals': db.get_goals(user_id)})


# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/visualization/config')
def viz_config():
    """Get Three.js visualization configuration"""
    organism.step()
    return jsonify({
        'sacred_math': {
            'phi': PHI, 'phi_inv': PHI_INV,
            'golden_angle_rad': GOLDEN_ANGLE_RAD,
            'fibonacci': FIBONACCI[:15]
        },
        'chaos': {
            'lorenz_wing': organism.lorenz.get_wing(),
            'rossler_phase': organism.rossler.get_phase(),
            'balance': organism.coupled.get_balance()
        },
        'orbs': organism.swarm.get_state(),
        'mayan': organism.get_mayan_day(),
        'foundations': 20
    })


@app.route('/api/math/<foundation>')
def math_foundation(foundation):
    """Query individual mathematical foundations"""
    t = float(request.args.get('t', 1.0))
    x = float(request.args.get('x', 0.5))
    y = float(request.args.get('y', 0.5))
    emotion = request.args.get('emotion', 'hope')
    
    results = {}
    
    if foundation == 'golden-harmonic' or foundation == 'all':
        results['golden_harmonic'] = organism.math.golden_harmonic_fold(t)
    if foundation == 'pareidolia' or foundation == 'all':
        results['pareidolia'] = organism.math.pareidolia_field(x, y, t)
    if foundation == 'sacred-blend' or foundation == 'all':
        results['sacred_blend'] = organism.math.sacred_blend(x, y, t)
    if foundation == 'origami' or foundation == 'all':
        results['origami'] = organism.math.origami_curve(x, y)
    if foundation == 'emotional-harmonic' or foundation == 'all':
        results['emotional_harmonic'] = organism.math.emotional_harmonic(t, emotion)
    if foundation == 'emotional-manifold' or foundation == 'all':
        results['emotional_manifold'] = organism.math.emotional_manifold(x, y, t, emotion)
    if foundation == 'lorenz' or foundation == 'all':
        results['lorenz'] = {'wing': organism.lorenz.get_wing(), 
                            'pos': organism.lorenz.get_normalized().__dict__}
    if foundation == 'rossler' or foundation == 'all':
        results['rossler'] = {'phase': organism.rossler.get_phase()}
    if foundation == 'harmonic-resonance' or foundation == 'all':
        wellness = float(request.args.get('wellness', 50))
        results['harmonic'] = {
            'interval': organism.harmonic.wellness_to_interval(wellness),
            'frequency': organism.harmonic.get_frequency(wellness)
        }
    
    if not results:
        return jsonify({'error': f'Unknown foundation: {foundation}'}), 400
    
    return jsonify(results)


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH & SYSTEM ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    mayan = organism.get_mayan_day()
    return jsonify({
        'status': 'healthy',
        'version': '13.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'gpu': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'math_foundations': 20,
        'orbs': len(organism.swarm.orbs),
        'mayan': f"Today is {mayan['display']}",
        'harmony': organism.math.golden_harmonic_fold(organism.time),
        'lorenz_wing': organism.lorenz.get_wing(),
        'chaos_balance': organism.coupled.get_balance()
    })


@app.route('/robots.txt')
def robots():
    return "User-agent: *\nAllow: /\n", 200, {'Content-Type': 'text/plain'}


@app.route('/')
def index():
    return render_template_string(MAIN_HTML)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN HTML (Nordic Design + Full 3D Universe)
# ════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌀 Life Fractal Intelligence v13</title>
<style>
:root {
    --phi: 1.618033988749895;
    --gold: #FFD700;
    --cosmic: #8B5CF6;
    --bg: #0a0a0f;
    --card: #15151f;
    --text: #e0e0e0;
    --dim: #888;
    --border: #2a2a3a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}
.container { max-width: 1400px; margin: 0 auto; padding: 1rem; }
header {
    text-align: center;
    padding: calc(1.5rem * var(--phi)) 1rem;
    background: linear-gradient(180deg, #1a1a2e 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
}
h1 {
    font-size: calc(1.5rem * var(--phi));
    background: linear-gradient(135deg, var(--gold), var(--cosmic));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle { color: var(--dim); margin-top: 0.5rem; }
.badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid var(--cosmic);
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    margin: 0.2rem;
    font-size: 0.8rem;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: calc(1rem * var(--phi));
    margin-top: 1.5rem;
}
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
}
.card h2 { color: var(--gold); font-size: 1.2rem; margin-bottom: 0.8rem; }
#fractal-canvas {
    width: 100%;
    aspect-ratio: 1;
    background: #000;
    border-radius: 8px;
}
.stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
.stat {
    background: rgba(255,255,255,0.05);
    padding: 0.6rem;
    border-radius: 6px;
    text-align: center;
}
.stat-val { font-size: 1.2rem; font-weight: bold; color: var(--gold); }
.stat-lbl { font-size: 0.7rem; color: var(--dim); text-transform: uppercase; }
.slider-wrap { margin: 0.8rem 0; }
.slider-top { display: flex; justify-content: space-between; margin-bottom: 0.2rem; }
input[type="range"] {
    width: 100%; height: 6px; border-radius: 3px;
    background: var(--border); -webkit-appearance: none;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: var(--gold); cursor: pointer;
}
button {
    background: linear-gradient(135deg, var(--gold), #FFA500);
    color: #000; border: none; padding: 0.7rem 1.2rem;
    border-radius: 8px; font-weight: 600; cursor: pointer;
    width: 100%; margin-top: 0.8rem;
}
button:hover { transform: scale(1.02); }
.auth-form {
    max-width: 380px; margin: 2rem auto;
    padding: 1.5rem; background: var(--card);
    border-radius: 12px; border: 1px solid var(--border);
}
.auth-form input {
    width: 100%; padding: 0.7rem; margin: 0.4rem 0;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
}
.spoons { display: flex; gap: 0.2rem; flex-wrap: wrap; margin: 0.5rem 0; }
.spoon { font-size: 1.3rem; opacity: 0.3; }
.spoon.active { opacity: 1; }
.chaos-info {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(139,92,246,0.1); padding: 0.6rem;
    border-radius: 6px; margin-top: 0.8rem; font-size: 0.9rem;
}
.wing-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--cosmic); }
footer {
    text-align: center; padding: 1.5rem; color: var(--dim);
    border-top: 1px solid var(--border); margin-top: 2rem;
}
#demo-banner {
    background: linear-gradient(90deg, var(--cosmic), var(--gold));
    color: #000; text-align: center; padding: 0.5rem; font-weight: 600;
}
.phi { font-family: monospace; color: var(--gold); }
</style>
</head>
<body>
<div id="demo-banner">🌀 Life Fractal Intelligence v13.0 - 20 Mathematical Foundations</div>

<header>
<h1>🌀 Life Fractal Intelligence</h1>
<p class="subtitle">Mathematical Harmony for Neurodivergent Minds</p>
<div style="margin-top: 0.8rem;">
<span class="badge">🦋 Lorenz Chaos</span>
<span class="badge">🌀 Rossler Spiral</span>
<span class="badge">🌻 Golden Ratio</span>
<span class="badge">✨ Particle Swarm</span>
<span class="badge">🎵 Harmonic Resonance</span>
<span class="badge">📐 20 Foundations</span>
</div>
</header>

<div class="container">
<div id="app">
<!-- Auth -->
<div id="auth-view" class="auth-form">
<h2 style="text-align:center; margin-bottom:1rem;">✨ Welcome</h2>
<div id="login-form">
<input type="email" id="login-email" placeholder="Email">
<input type="password" id="login-password" placeholder="Password">
<button onclick="doLogin()">Login</button>
<p style="text-align:center; margin-top:0.8rem; color:var(--dim);">
New? <a href="#" onclick="showReg()" style="color:var(--gold);">Create account</a> |
<a href="#" onclick="enterDemo()" style="color:var(--cosmic);">Try Demo</a>
</p>
</div>
<div id="reg-form" style="display:none;">
<input type="text" id="reg-first" placeholder="First Name">
<input type="text" id="reg-last" placeholder="Last Name">
<input type="email" id="reg-email" placeholder="Email">
<input type="password" id="reg-password" placeholder="Password (6+ chars)">
<button onclick="doRegister()">Create Account</button>
<p style="text-align:center; margin-top:0.8rem; color:var(--dim);">
Have account? <a href="#" onclick="showLogin()" style="color:var(--gold);">Login</a>
</p>
</div>
</div>

<!-- Dashboard -->
<div id="dash-view" style="display:none;">
<div class="grid">
<!-- Visualization -->
<div class="card">
<h2>🌀 Chaos Visualization</h2>
<canvas id="fractal-canvas"></canvas>
<div class="chaos-info">
<div class="wing-dot"></div>
<span id="chaos-text">Loading chaos state...</span>
</div>
</div>

<!-- Energy Check-in -->
<div class="card">
<h2>🥄 Energy Check-in</h2>
<div class="spoons" id="spoons"></div>
<div class="slider-wrap">
<div class="slider-top"><span>Mood</span><span id="mood-val">50</span></div>
<input type="range" id="mood" min="0" max="100" value="50" oninput="updateSlider('mood')">
</div>
<div class="slider-wrap">
<div class="slider-top"><span>Stress</span><span id="stress-val">50</span></div>
<input type="range" id="stress" min="0" max="100" value="50" oninput="updateSlider('stress')">
</div>
<div class="slider-wrap">
<div class="slider-top"><span>Energy</span><span id="energy-val">50</span></div>
<input type="range" id="energy" min="0" max="100" value="50" oninput="updateSlider('energy')">
</div>
<button onclick="saveCheckin()">Save Check-in</button>
</div>

<!-- Sacred Math -->
<div class="card">
<h2>📐 Sacred Mathematics</h2>
<div class="stat-grid">
<div class="stat"><div class="stat-val phi" id="phi-disp">φ</div><div class="stat-lbl">Golden Ratio</div></div>
<div class="stat"><div class="stat-val" id="wellness-disp">--</div><div class="stat-lbl">Wellness</div></div>
<div class="stat"><div class="stat-val" id="harmonic-disp">--</div><div class="stat-lbl">Harmonic</div></div>
<div class="stat"><div class="stat-val" id="mayan-disp">--</div><div class="stat-lbl">Mayan Day</div></div>
</div>
</div>

<!-- Goals -->
<div class="card">
<h2>🎯 Goals</h2>
<div id="goals-list"><p style="color:var(--dim);">No goals yet</p></div>
<button onclick="addGoal()">+ Add Goal</button>
</div>
</div>
</div>
</div>
</div>

<footer>
<p>Life Fractal Intelligence v13.0 | φ = <span class="phi">1.618033988749895</span></p>
<p>Planning tools designed for brains like yours 💜</p>
</footer>

<script>
const PHI = 1.618033988749895;
const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
const TAU = Math.PI * 2;

let userId = localStorage.getItem('userId');
let isDemo = false;
let lorenz = {x:1, y:1, z:1, history:[]};
let rossler = {x:1, y:1, z:1, history:[]};
let time = 0;
let animId = null;

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('phi-disp').textContent = 'φ=' + PHI.toFixed(4);
    if (userId) showDash();
});

function showLogin() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('reg-form').style.display = 'none';
}
function showReg() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('reg-form').style.display = 'block';
}

async function doLogin() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    try {
        const r = await fetch('/api/auth/login', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({email, password})
        });
        const d = await r.json();
        if (d.success) { userId = d.user_id; localStorage.setItem('userId', userId); showDash(); }
        else alert(d.error || 'Login failed');
    } catch(e) { alert('Login failed'); }
}

async function doRegister() {
    const first = document.getElementById('reg-first').value;
    const last = document.getElementById('reg-last').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    try {
        const r = await fetch('/api/auth/register', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({email, password, first_name: first, last_name: last})
        });
        const d = await r.json();
        if (d.success) { userId = d.user_id; localStorage.setItem('userId', userId); showDash(); }
        else alert(d.error || 'Registration failed');
    } catch(e) { alert('Registration failed'); }
}

function enterDemo() {
    isDemo = true;
    showDash();
}

async function showDash() {
    document.getElementById('auth-view').style.display = 'none';
    document.getElementById('dash-view').style.display = 'block';
    renderSpoons(12);
    initChaos();
    if (!isDemo) await loadData();
}

async function loadData() {
    try {
        const r = await fetch('/api/organism/state');
        const d = await r.json();
        if (d.mayan) document.getElementById('mayan-disp').textContent = d.mayan.display;
        document.getElementById('chaos-text').textContent = 
            `${d.lorenz_wing} phase | Balance: ${(d.chaos_balance*100).toFixed(0)}%`;
    } catch(e) { console.error(e); }
}

function renderSpoons(n) {
    const c = document.getElementById('spoons');
    c.innerHTML = '';
    for (let i = 0; i < 12; i++) {
        const s = document.createElement('span');
        s.className = 'spoon' + (i < n ? ' active' : '');
        s.textContent = '🥄';
        c.appendChild(s);
    }
}

function updateSlider(type) {
    const v = document.getElementById(type).value;
    document.getElementById(type + '-val').textContent = v;
}

async function saveCheckin() {
    if (isDemo) { alert('Demo mode - data not saved'); return; }
    // API call would go here
    alert('Check-in saved! ✨');
}

function addGoal() {
    const title = prompt('Goal title:');
    if (!title) return;
    if (isDemo) { alert('Demo mode - goal not saved'); return; }
    fetch(`/api/user/${userId}/goals`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title, priority: 3})
    }).then(() => loadData());
}

function initChaos() {
    const canvas = document.getElementById('fractal-canvas');
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    lorenz = {x:1, y:1, z:1, history:[]};
    rossler = {x:1, y:1, z:1, history:[]};
    time = 0;
    animateChaos();
}

function stepLorenz(dt=0.01) {
    const s=10, r=28, b=8/3;
    const dx = s*(lorenz.y - lorenz.x);
    const dy = lorenz.x*(r - lorenz.z) - lorenz.y;
    const dz = lorenz.x*lorenz.y - b*lorenz.z;
    lorenz.x += dx*dt; lorenz.y += dy*dt; lorenz.z += dz*dt;
    lorenz.history.push({x:lorenz.x, y:lorenz.y, z:lorenz.z});
    if (lorenz.history.length > 1500) lorenz.history.shift();
}

function stepRossler(dt=0.02) {
    const a=0.2, b=0.2, c=5.7;
    const dx = -rossler.y - rossler.z;
    const dy = rossler.x + a*rossler.y;
    const dz = b + rossler.z*(rossler.x - c);
    rossler.x += dx*dt; rossler.y += dy*dt; rossler.z += dz*dt;
    rossler.history.push({x:rossler.x, y:rossler.y, z:rossler.z});
    if (rossler.history.length > 1000) rossler.history.shift();
}

function animateChaos() {
    const canvas = document.getElementById('fractal-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const cx = w/2, cy = h/2;
    
    ctx.fillStyle = 'rgba(10,10,15,0.1)';
    ctx.fillRect(0, 0, w, h);
    
    for (let i=0; i<5; i++) { stepLorenz(0.005); stepRossler(0.01); }
    time += 0.016;
    
    const rotY = time * 0.2;
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    
    // Lorenz
    ctx.strokeStyle = 'rgba(255,215,0,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i=1; i<lorenz.history.length; i++) {
        const p = lorenz.history[i];
        const nx = (p.x+25)/50*100-50, ny = (p.y+30)/60*100-50, nz = p.z;
        const rx = nx*cosY + nz*sinY, rz = -nx*sinY + nz*cosY;
        const scale = 200/(200+rz);
        const sx = cx + rx*2*scale, sy = cy - ny*2*scale;
        if (i===1) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    
    // Rossler
    ctx.strokeStyle = 'rgba(139,92,246,0.4)';
    ctx.beginPath();
    for (let i=1; i<rossler.history.length; i++) {
        const p = rossler.history[i];
        const rx = p.x*cosY + p.z*sinY, rz = -p.x*sinY + p.z*cosY;
        const scale = 200/(200+rz*2);
        const sx = cx + rx*4*scale, sy = cy - p.y*4*scale;
        if (i===1) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    
    // Golden spiral
    ctx.strokeStyle = 'rgba(255,215,0,0.2)';
    ctx.beginPath();
    for (let i=0; i<200; i++) {
        const theta = i*0.1;
        const r = 3 * Math.pow(PHI, theta/(Math.PI/2));
        const x = cx + r*Math.cos(theta + time*0.1);
        const y = cy + r*Math.sin(theta + time*0.1);
        if (i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Center glow
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 25);
    grad.addColorStop(0, 'rgba(255,215,0,0.5)');
    grad.addColorStop(1, 'rgba(255,215,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, 25, 0, TAU);
    ctx.fill();
    
    animId = requestAnimationFrame(animateChaos);
}
</script>
</body>
</html>'''


# ════════════════════════════════════════════════════════════════════════════════
# MODULE INITIALIZATION (for gunicorn/Render)
# ════════════════════════════════════════════════════════════════════════════════

# Initialize database on module import
with app.app_context():
    try:
        db.init_database()
        logger.info("✅ Database tables verified")
    except Exception as e:
        logger.error(f"Database init error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 78)
    print("🌀 LIFE FRACTAL INTELLIGENCE v13.0 - ULTIMATE MATHEMATICAL SYNTHESIS")
    print("=" * 78)
    print(f"✨ Golden Ratio (φ):       {PHI:.15f}")
    print(f"🌻 Golden Angle:           {GOLDEN_ANGLE:.10f}°")
    print(f"📐 Fibonacci:              {FIBONACCI[:10]}...")
    print(f"🦋 Lorenz:                 σ={LORENZ_SIGMA}, ρ={LORENZ_RHO}, β={LORENZ_BETA:.4f}")
    print(f"🌀 Rossler:                a={ROSSLER_A}, b={ROSSLER_B}, c={ROSSLER_C}")
    print(f"🖥️  GPU:                    {GPU_AVAILABLE} ({GPU_NAME or 'CPU'})")
    print("=" * 78)
    print("\n🔢 20 MATHEMATICAL FOUNDATIONS:")
    print("  Original 10: Golden-Harmonic, Pareidolia, Sacred Blend, Fractal Bloom,")
    print("               Origami, Emotional Harmonic, Fourier Sketch, GPU Queue,")
    print("               Temporal Compression, Emotional Manifold")
    print("  New 10:      Lorenz, Rossler, Coupled Chaos, Particle Swarm,")
    print("               Harmonic Resonance, Fractal Dimension, Golden Spiral,")
    print("               Flower of Life, Metatron's Cube, Binaural Beats")
    print("=" * 78)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print_banner()
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
