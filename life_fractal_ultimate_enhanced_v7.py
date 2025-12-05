#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE - ULTIMATE ENHANCED EDITION v7.0
═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL SYNTHESIS: Combining Chaos Theory, Sacred Geometry, and 
Particle Dynamics for Revolutionary Life Planning Visualization

NEW MATHEMATICAL INTEGRATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. LORENZ ATTRACTOR FOR GOAL VISUALIZATION
   - Goals orbit chaotic attractors showing interconnected nature
   - Butterfly effect: small habits create large life changes
   - Two-wing structure represents life balance

2. ROSSLER ATTRACTOR FOR MOOD PATTERNS
   - Smooth spiral chaos models emotional rhythms
   - Predicts mood cycles with mathematical precision
   - Calming meditative visualization for anxiety

3. COMBINED CHAOS + SACRED GEOMETRY
   - Julia sets morph based on wellness index
   - Golden spiral overlays on chaotic attractors
   - Fibonacci-timed particle spawning

4. PARTICLE SWARM WELLNESS SYSTEM
   - Particles represent daily energy units (spoons)
   - PSO-inspired attraction to goals
   - Visual feedback for neurodivergent users

5. FRACTAL DIMENSION WELLNESS SCORE
   - Calculate life "complexity" using box-counting
   - Higher engagement = richer fractal patterns
   - Mathematical meaning behind progress

6. HARMONIC RESONANCE VISUALIZATION
   - Pythagorean tuning for visual frequencies
   - Colors mapped to harmonic ratios
   - Synesthetic experience design

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64
import struct
import wave

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    torch = None

# ML Support
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS - EXPANDED
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
TAU = 2 * PI
E = math.e
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
PHI_SQUARED = PHI * PHI
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
SQRT5 = math.sqrt(5)

# Golden Angle - the mathematical key to natural beauty
GOLDEN_ANGLE = 360 / (PHI * PHI)  # 137.5077640500378 degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)

# Fibonacci sequence extended
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

# Lucas numbers (related to Fibonacci)
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843]

# Platonic solids for 3D sacred geometry
PLATONIC_SOLIDS = {
    'tetrahedron': {'faces': 4, 'vertices': 4, 'edges': 6, 'dihedral': 70.528779},
    'cube': {'faces': 6, 'vertices': 8, 'edges': 12, 'dihedral': 90},
    'octahedron': {'faces': 8, 'vertices': 6, 'edges': 12, 'dihedral': 109.471221},
    'dodecahedron': {'faces': 12, 'vertices': 20, 'edges': 30, 'dihedral': 116.565051},
    'icosahedron': {'faces': 20, 'vertices': 12, 'edges': 30, 'dihedral': 138.189685}
}

# Pythagorean tuning ratios for harmonic color/sound
PYTHAGOREAN_RATIOS = {
    'unison': 1/1,
    'octave': 2/1,
    'fifth': 3/2,
    'fourth': 4/3,
    'major_third': 5/4,
    'minor_third': 6/5,
    'major_sixth': 5/3,
    'minor_sixth': 8/5,
    'major_second': 9/8,
    'minor_seventh': 9/5
}

# Chaos theory parameters
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7

# Color palettes for different emotional states
MOOD_PALETTES = {
    'calm': [(100, 150, 200), (80, 130, 180), (60, 110, 160), (40, 90, 140)],
    'energetic': [(255, 180, 100), (255, 140, 80), (255, 100, 60), (230, 80, 50)],
    'focused': [(180, 130, 220), (150, 100, 200), (120, 70, 180), (90, 40, 160)],
    'anxious': [(220, 100, 100), (200, 80, 90), (180, 60, 80), (150, 40, 70)],
    'peaceful': [(130, 200, 170), (100, 180, 150), (70, 160, 130), (40, 140, 110)],
    'golden': [(255, 215, 128), (220, 180, 100), (180, 140, 80), (140, 100, 60)],
    'cosmic': [(180, 100, 255), (140, 80, 220), (100, 60, 180), (60, 40, 140)],
    'aurora': [(100, 255, 150), (80, 200, 180), (60, 150, 200), (40, 100, 220)]
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3D VECTOR MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Vector3:
    """3D Vector with comprehensive mathematical operations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        if scalar == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3(0, 0, 0)
        return self / mag
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def lerp(self, other: 'Vector3', t: float) -> 'Vector3':
        """Linear interpolation"""
        t = max(0, min(1, t))
        return Vector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )
    
    def rotate_x(self, angle: float) -> 'Vector3':
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Vector3(
            self.x,
            self.y * cos_a - self.z * sin_a,
            self.y * sin_a + self.z * cos_a
        )
    
    def rotate_y(self, angle: float) -> 'Vector3':
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Vector3(
            self.x * cos_a + self.z * sin_a,
            self.y,
            -self.x * sin_a + self.z * cos_a
        )
    
    def rotate_z(self, angle: float) -> 'Vector3':
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Vector3(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
            self.z
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_2d(self, scale: float = 1.0, perspective: float = 100) -> Tuple[float, float]:
        """Project to 2D with perspective"""
        z_factor = perspective / (perspective + self.z) if perspective + self.z > 0 else 0.1
        return (self.x * scale * z_factor, self.y * scale * z_factor)
    
    @staticmethod
    def from_spherical(r: float, theta: float, phi: float) -> 'Vector3':
        """Create from spherical coordinates"""
        return Vector3(
            r * math.sin(theta) * math.cos(phi),
            r * math.sin(theta) * math.sin(phi),
            r * math.cos(theta)
        )
    
    @staticmethod
    def random_unit() -> 'Vector3':
        """Random unit vector (uniform on sphere)"""
        import random
        theta = random.random() * TAU
        phi = math.acos(2 * random.random() - 1)
        return Vector3.from_spherical(1, theta, phi)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAOS THEORY ATTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

class LorenzAttractor:
    """
    The Lorenz Attractor - A system exhibiting chaotic behavior (butterfly effect)
    
    Used for: Visualizing goal interconnections and life balance
    The two "wings" represent different life domains that influence each other
    """
    
    def __init__(self, sigma: float = LORENZ_SIGMA, rho: float = LORENZ_RHO, 
                 beta: float = LORENZ_BETA):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.position = Vector3(1.0, 1.0, 1.0)
        self.history: List[Vector3] = []
        self.max_history = 3000
        
        # Normalization bounds
        self.bounds = {
            'x': (-25, 25), 'y': (-30, 30), 'z': (0, 50)
        }
    
    def step(self, dt: float = 0.01) -> Vector3:
        """Advance the system using 4th-order Runge-Kutta"""
        def derivatives(pos: Vector3) -> Vector3:
            return Vector3(
                self.sigma * (pos.y - pos.x),
                pos.x * (self.rho - pos.z) - pos.y,
                pos.x * pos.y - self.beta * pos.z
            )
        
        # RK4 integration for smoother trajectories
        k1 = derivatives(self.position)
        k2 = derivatives(self.position + k1 * (dt / 2))
        k3 = derivatives(self.position + k2 * (dt / 2))
        k4 = derivatives(self.position + k3 * dt)
        
        self.position = self.position + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)
        
        self.history.append(Vector3(self.position.x, self.position.y, self.position.z))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return self.position
    
    def step_multiple(self, steps: int = 5, dt: float = 0.01) -> Vector3:
        """Take multiple steps for smoother animation"""
        for _ in range(steps):
            self.step(dt)
        return self.position
    
    def get_normalized(self) -> Vector3:
        """Get position normalized to [-1, 1]"""
        return Vector3(
            (self.position.x - self.bounds['x'][0]) / (self.bounds['x'][1] - self.bounds['x'][0]) * 2 - 1,
            (self.position.y - self.bounds['y'][0]) / (self.bounds['y'][1] - self.bounds['y'][0]) * 2 - 1,
            (self.position.z - self.bounds['z'][0]) / (self.bounds['z'][1] - self.bounds['z'][0]) * 2 - 1
        )
    
    def get_wing(self) -> str:
        """Determine which 'wing' the attractor is in"""
        return 'right' if self.position.x > 0 else 'left'
    
    def reset(self, x: float = 1.0, y: float = 1.0, z: float = 1.0):
        self.position = Vector3(x, y, z)
        self.history.clear()


class RosslerAttractor:
    """
    The Rossler Attractor - Smooth spiral chaos
    
    Used for: Mood pattern visualization and prediction
    Creates calming, meditative spiral patterns
    """
    
    def __init__(self, a: float = ROSSLER_A, b: float = ROSSLER_B, c: float = ROSSLER_C):
        self.a = a
        self.b = b
        self.c = c
        self.position = Vector3(1.0, 1.0, 1.0)
        self.history: List[Vector3] = []
        self.max_history = 2000
        self.scale = 15.0
    
    def step(self, dt: float = 0.02) -> Vector3:
        """Advance using Euler integration (stable for Rossler)"""
        dx = -self.position.y - self.position.z
        dy = self.position.x + self.a * self.position.y
        dz = self.b + self.position.z * (self.position.x - self.c)
        
        self.position = Vector3(
            self.position.x + dx * dt,
            self.position.y + dy * dt,
            self.position.z + dz * dt
        )
        
        self.history.append(Vector3(self.position.x, self.position.y, self.position.z))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return self.position
    
    def step_multiple(self, steps: int = 3, dt: float = 0.02) -> Vector3:
        for _ in range(steps):
            self.step(dt)
        return self.position
    
    def get_normalized(self) -> Vector3:
        return self.position / self.scale
    
    def get_spiral_phase(self) -> float:
        """Get current phase in the spiral (0-1)"""
        return (math.atan2(self.position.y, self.position.x) + PI) / TAU


class CoupledChaosSystem:
    """
    Combined Lorenz + Rossler system with bidirectional coupling
    
    Used for: Showing how life domains (Lorenz) influence emotional patterns (Rossler)
    Coupling strength determines how interconnected these systems are
    """
    
    def __init__(self, coupling: float = 0.1):
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        self.coupling = coupling
        self.time = 0.0
    
    def step(self, dt: float = 0.01) -> Tuple[Vector3, Vector3]:
        """Step both systems with coupling"""
        # Lorenz influenced by Rossler mood
        rossler_influence = self.rossler.get_normalized() * self.coupling * 5
        self.lorenz.position = self.lorenz.position + rossler_influence * dt
        
        # Rossler influenced by Lorenz life-balance
        lorenz_influence = self.lorenz.get_normalized() * self.coupling * 2
        self.rossler.position = self.rossler.position + lorenz_influence * dt
        
        # Step both
        l_pos = self.lorenz.step(dt)
        r_pos = self.rossler.step(dt)
        
        self.time += dt
        return (l_pos, r_pos)
    
    def get_combined_energy(self) -> float:
        """Calculate combined system energy for wellness visualization"""
        l_energy = self.lorenz.position.magnitude_squared() / 1000
        r_energy = self.rossler.position.magnitude_squared() / 100
        return (l_energy + r_energy) / 2
    
    def get_balance_ratio(self) -> float:
        """How balanced are the two systems (0 = imbalanced, 1 = balanced)"""
        l_norm = self.lorenz.get_normalized().magnitude()
        r_norm = self.rossler.get_normalized().magnitude()
        if l_norm + r_norm < 0.01:
            return 1.0
        return 1 - abs(l_norm - r_norm) / (l_norm + r_norm)


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED FRACTAL ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

class MandelbrotEngine:
    """Enhanced Mandelbrot with smooth iteration counting"""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def iterate(self, c_real: float, c_imag: float) -> Tuple[int, float]:
        """Iterate with smooth escape-time coloring"""
        z_real = 0.0
        z_imag = 0.0
        
        for i in range(self.max_iterations):
            z_mag_sq = z_real * z_real + z_imag * z_imag
            if z_mag_sq > 4.0:
                # Smooth coloring using log
                smooth = i - math.log2(math.log2(z_mag_sq + 1) + 1)
                return (i, smooth)
            
            new_real = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = new_real
        
        return (self.max_iterations, float(self.max_iterations))
    
    def get_wellness_point(self, wellness: float) -> Tuple[float, float]:
        """Convert wellness score (0-100) to interesting Mandelbrot coordinate"""
        # Map wellness to the Mandelbrot cardioid boundary
        t = (wellness / 100) * TAU
        # Parametric cardioid with golden ratio modulation
        r = (1 - math.cos(t)) / 2 * PHI_INVERSE
        return (
            r * math.cos(t) - 0.5,
            r * math.sin(t) * PHI_INVERSE
        )


class JuliaEngine:
    """Julia set that morphs based on user state"""
    
    # Interesting Julia set parameters for different moods
    MOOD_C_VALUES = {
        'calm': (-0.7, 0.27),
        'energetic': (-0.8, 0.156),
        'focused': (-0.4, 0.6),
        'anxious': (-0.12, 0.74),
        'peaceful': (0.285, 0.01),
        'golden': (-0.835, -0.2321),  # Golden ratio related
        'cosmic': (-0.7269, 0.1889)
    }
    
    def __init__(self, c_real: float = -0.7, c_imag: float = 0.27, max_iterations: int = 100):
        self.c_real = c_real
        self.c_imag = c_imag
        self.max_iterations = max_iterations
    
    def set_mood(self, mood: str):
        """Set Julia parameters based on mood"""
        if mood in self.MOOD_C_VALUES:
            self.c_real, self.c_imag = self.MOOD_C_VALUES[mood]
    
    def set_from_wellness(self, wellness: float, stress: float):
        """Dynamically set c based on user state"""
        # Map wellness/stress to Julia parameter space
        t = wellness / 100
        s = stress / 100
        
        # Interpolate between calm and anxious Julia sets
        calm_c = self.MOOD_C_VALUES['calm']
        anxious_c = self.MOOD_C_VALUES['anxious']
        
        self.c_real = calm_c[0] * t + anxious_c[0] * s
        self.c_imag = calm_c[1] * t + anxious_c[1] * (1 - t)
    
    def iterate(self, z_real: float, z_imag: float) -> Tuple[int, float]:
        """Iterate Julia formula"""
        for i in range(self.max_iterations):
            z_mag_sq = z_real * z_real + z_imag * z_imag
            if z_mag_sq > 4.0:
                smooth = i - math.log2(math.log2(z_mag_sq + 1) + 1)
                return (i, smooth)
            
            new_real = z_real * z_real - z_imag * z_imag + self.c_real
            z_imag = 2 * z_real * z_imag + self.c_imag
            z_real = new_real
        
        return (self.max_iterations, float(self.max_iterations))


class FractalDimensionCalculator:
    """Calculate fractal dimension of user's life pattern"""
    
    def __init__(self):
        self.epsilon_values = [2**(-i) for i in range(2, 8)]
    
    def box_counting(self, points: List[Tuple[float, float]]) -> float:
        """Estimate fractal dimension using box-counting method"""
        if len(points) < 10:
            return 1.0
        
        counts = []
        for eps in self.epsilon_values:
            boxes = set()
            for x, y in points:
                box_x = int(x / eps)
                box_y = int(y / eps)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        
        # Linear regression on log-log plot
        log_eps = np.log(self.epsilon_values)
        log_counts = np.log(np.array(counts) + 1)
        
        if len(log_eps) < 2:
            return 1.0
        
        # Fit line to get dimension
        coeffs = np.polyfit(log_eps, log_counts, 1)
        dimension = -coeffs[0]
        
        return max(1.0, min(2.0, dimension))
    
    def life_complexity_score(self, dimension: float) -> str:
        """Convert fractal dimension to meaningful life insight"""
        if dimension < 1.2:
            return "Your life pattern is smooth and focused - great for deep work!"
        elif dimension < 1.5:
            return "Balanced complexity - you're managing multiple areas well"
        elif dimension < 1.7:
            return "Rich, interconnected life pattern - lots of variety!"
        else:
            return "Highly complex pattern - consider simplifying some areas"


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED GEOMETRY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenSpiral:
    """Golden Spiral - appears everywhere in nature"""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def get_point(self, theta: float) -> Tuple[float, float]:
        """Point on golden spiral at angle theta"""
        r = self.scale * (PHI ** (theta / (PI / 2)))
        return (r * math.cos(theta), r * math.sin(theta))
    
    def get_points(self, n: int, max_theta: float = 4 * PI) -> List[Tuple[float, float]]:
        """Generate n points along the spiral"""
        return [self.get_point(i * max_theta / n) for i in range(n)]
    
    def map_to_progress(self, progress: float) -> Tuple[float, float]:
        """Map progress (0-1) to position on spiral"""
        theta = progress * 4 * PI
        return self.get_point(theta)


class FlowerOfLife:
    """Flower of Life - ancient sacred geometry pattern"""
    
    def __init__(self, radius: float = 1.0, rings: int = 3):
        self.radius = radius
        self.rings = rings
    
    def get_circle_centers(self) -> List[Tuple[float, float]]:
        """Get centers of all circles in the pattern"""
        centers = [(0, 0)]
        
        for ring in range(1, self.rings + 1):
            for i in range(6 * ring):
                angle = i * (PI / 3) / ring
                distance = self.radius * ring
                centers.append((
                    distance * math.cos(angle),
                    distance * math.sin(angle)
                ))
        
        return centers
    
    def contains_point(self, x: float, y: float) -> List[int]:
        """Which circles contain this point?"""
        centers = self.get_circle_centers()
        containing = []
        for i, (cx, cy) in enumerate(centers):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= self.radius:
                containing.append(i)
        return containing


class MetatronsCube:
    """Metatron's Cube - contains all Platonic solids"""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        # 13 circles: 1 center, 6 inner, 6 outer
        self.circle_positions = self._calculate_positions()
    
    def _calculate_positions(self) -> List[Tuple[float, float]]:
        positions = [(0, 0)]  # Center
        
        # Inner ring of 6
        for i in range(6):
            angle = i * PI / 3
            positions.append((
                self.scale * math.cos(angle),
                self.scale * math.sin(angle)
            ))
        
        # Outer ring of 6
        for i in range(6):
            angle = i * PI / 3 + PI / 6
            positions.append((
                self.scale * 2 * math.cos(angle),
                self.scale * 2 * math.sin(angle)
            ))
        
        return positions
    
    def get_connecting_lines(self) -> List[Tuple[int, int]]:
        """Get all lines connecting the circles"""
        lines = []
        n = len(self.circle_positions)
        for i in range(n):
            for j in range(i + 1, n):
                lines.append((i, j))
        return lines
    
    def project_goal_to_circle(self, goal_index: int, total_goals: int) -> Tuple[float, float]:
        """Map a goal to a position in Metatron's Cube"""
        circle_idx = goal_index % len(self.circle_positions)
        return self.circle_positions[circle_idx]


class SriYantra:
    """Sri Yantra - complex sacred geometry for meditation"""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def get_triangle_vertices(self, layer: int, pointing_up: bool) -> List[Tuple[float, float]]:
        """Get vertices for a triangle in the yantra"""
        r = self.scale * (1 - layer * 0.15)
        rotation = 0 if pointing_up else PI
        
        vertices = []
        for i in range(3):
            angle = rotation + i * TAU / 3
            vertices.append((r * math.cos(angle), r * math.sin(angle)))
        
        return vertices
    
    def get_all_triangles(self) -> List[List[Tuple[float, float]]]:
        """Get all 9 interlocking triangles"""
        triangles = []
        for i in range(5):  # 5 downward pointing
            triangles.append(self.get_triangle_vertices(i, False))
        for i in range(4):  # 4 upward pointing
            triangles.append(self.get_triangle_vertices(i, True))
        return triangles


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE SYSTEM WITH PSO DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    """Individual particle with physics and state"""
    position: Vector3
    velocity: Vector3
    color: Tuple[int, int, int]
    size: float = 2.0
    life: float = 1.0
    max_life: float = 3.0
    personal_best: Optional[Vector3] = None
    energy: float = 1.0  # Represents a "spoon" unit
    
    def __post_init__(self):
        if self.personal_best is None:
            self.personal_best = Vector3(self.position.x, self.position.y, self.position.z)
    
    def update(self, dt: float, global_attractor: Vector3, 
               w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Update using PSO-inspired dynamics
        w: inertia weight
        c1: cognitive weight (personal best)
        c2: social weight (global best/attractor)
        """
        import random
        
        r1 = random.random()
        r2 = random.random()
        
        # PSO velocity update
        cognitive = (self.personal_best - self.position) * c1 * r1
        social = (global_attractor - self.position) * c2 * r2
        
        self.velocity = self.velocity * w + cognitive + social
        
        # Limit velocity
        max_speed = 50.0
        speed = self.velocity.magnitude()
        if speed > max_speed:
            self.velocity = self.velocity.normalize() * max_speed
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Update personal best if closer to attractor
        curr_dist = (self.position - global_attractor).magnitude()
        best_dist = (self.personal_best - global_attractor).magnitude()
        if curr_dist < best_dist:
            self.personal_best = Vector3(self.position.x, self.position.y, self.position.z)
        
        # Decay
        self.life -= dt / self.max_life
        self.energy *= 0.999
    
    def is_alive(self) -> bool:
        return self.life > 0 and self.energy > 0.01


class SpoonParticleSystem:
    """
    Particle system representing energy units ("spoons" from Spoon Theory)
    
    Each particle represents a unit of energy available for tasks
    Particles are attracted to goals based on priority
    Visual feedback shows energy expenditure
    """
    
    def __init__(self, max_spoons: int = 12):
        self.particles: List[Particle] = []
        self.max_spoons = max_spoons
        self.current_spoons = max_spoons
        self.global_best: Vector3 = Vector3(0, 0, 0)
        self.goal_attractors: List[Tuple[Vector3, float]] = []  # (position, priority)
    
    def spawn_spoon(self, position: Vector3, color: Tuple[int, int, int] = (255, 215, 128)):
        """Spawn a new energy particle"""
        import random
        
        if len(self.particles) >= self.max_spoons:
            return
        
        vel = Vector3(
            (random.random() - 0.5) * 2,
            (random.random() - 0.5) * 2,
            (random.random() - 0.5) * 2
        )
        
        self.particles.append(Particle(
            position=Vector3(position.x, position.y, position.z),
            velocity=vel,
            color=color,
            size=3.0 + random.random() * 2,
            life=1.0,
            max_life=5.0 + random.random() * 5,
            energy=1.0
        ))
        self.current_spoons = len(self.particles)
    
    def add_goal_attractor(self, position: Vector3, priority: float):
        """Add a goal as an attractor for particles"""
        self.goal_attractors.append((position, priority))
    
    def clear_goal_attractors(self):
        """Clear all goal attractors"""
        self.goal_attractors.clear()
    
    def update(self, dt: float):
        """Update all particles"""
        # Calculate weighted global attractor from goals
        if self.goal_attractors:
            total_priority = sum(p for _, p in self.goal_attractors)
            if total_priority > 0:
                self.global_best = Vector3(0, 0, 0)
                for pos, priority in self.goal_attractors:
                    weight = priority / total_priority
                    self.global_best = self.global_best + pos * weight
        
        # Update particles
        for p in self.particles:
            p.update(dt, self.global_best)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]
        self.current_spoons = len(self.particles)
    
    def use_spoon(self) -> bool:
        """Use one spoon of energy"""
        if self.particles:
            # Find the particle with most energy remaining
            best_idx = max(range(len(self.particles)), 
                          key=lambda i: self.particles[i].energy)
            self.particles[best_idx].life *= 0.5
            self.particles[best_idx].energy *= 0.5
            return True
        return False
    
    def get_energy_level(self) -> float:
        """Get total energy level (0-1)"""
        if not self.particles:
            return 0
        return sum(p.energy for p in self.particles) / self.max_spoons


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC RESONANCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicResonance:
    """
    Maps wellness states to harmonic frequencies and colors
    Based on Pythagorean tuning and synesthetic principles
    """
    
    # Base frequencies (A = 432 Hz for "healing" frequency)
    BASE_FREQ = 432
    
    # Map intervals to colors (synesthetic correspondence)
    INTERVAL_COLORS = {
        'unison': (255, 255, 255),      # White - pure
        'octave': (255, 255, 200),      # Light gold
        'fifth': (100, 200, 255),       # Sky blue - perfect
        'fourth': (150, 255, 150),      # Mint green
        'major_third': (255, 200, 100), # Warm yellow
        'minor_third': (200, 150, 255), # Lavender
        'major_sixth': (255, 180, 200), # Pink
        'minor_sixth': (180, 200, 255), # Periwinkle
    }
    
    def __init__(self):
        self.frequencies = {}
        self._calculate_frequencies()
    
    def _calculate_frequencies(self):
        """Calculate frequencies for all intervals"""
        for interval, ratio in PYTHAGOREAN_RATIOS.items():
            self.frequencies[interval] = self.BASE_FREQ * ratio
    
    def wellness_to_interval(self, wellness: float) -> str:
        """Map wellness score (0-100) to musical interval"""
        if wellness >= 90:
            return 'octave'
        elif wellness >= 80:
            return 'fifth'
        elif wellness >= 70:
            return 'major_sixth'
        elif wellness >= 60:
            return 'fourth'
        elif wellness >= 50:
            return 'major_third'
        elif wellness >= 40:
            return 'minor_third'
        elif wellness >= 30:
            return 'minor_sixth'
        else:
            return 'unison'
    
    def get_wellness_color(self, wellness: float) -> Tuple[int, int, int]:
        """Get color corresponding to wellness level"""
        interval = self.wellness_to_interval(wellness)
        return self.INTERVAL_COLORS.get(interval, (128, 128, 128))
    
    def get_wellness_frequency(self, wellness: float) -> float:
        """Get frequency corresponding to wellness level"""
        interval = self.wellness_to_interval(wellness)
        return self.frequencies.get(interval, self.BASE_FREQ)
    
    def generate_binaural_beat(self, wellness: float, duration: float = 1.0, 
                                sample_rate: int = 44100) -> np.ndarray:
        """Generate binaural beat audio for wellness state"""
        base_freq = self.get_wellness_frequency(wellness)
        
        # Binaural beat difference (alpha waves: 8-12 Hz for relaxation)
        if wellness < 50:
            beat_freq = 10  # Alpha for stress relief
        elif wellness < 70:
            beat_freq = 15  # Low beta for focus
        else:
            beat_freq = 8   # Alpha for peaceful state
        
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        
        # Left ear
        left = np.sin(2 * PI * base_freq * t) * 0.5
        # Right ear (slightly different frequency creates beat)
        right = np.sin(2 * PI * (base_freq + beat_freq) * t) * 0.5
        
        # Combine to stereo
        stereo = np.vstack([left, right]).T
        return stereo


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED FRACTAL RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedFractalRenderer:
    """
    Production-ready fractal renderer combining all mathematical systems
    """
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Mathematical engines
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        self.coupled = CoupledChaosSystem(coupling=0.05)
        self.mandelbrot = MandelbrotEngine(max_iterations=100)
        self.julia = JuliaEngine()
        self.spiral = GoldenSpiral(scale=5.0)
        self.flower = FlowerOfLife(radius=50, rings=2)
        self.metatron = MetatronsCube(scale=80)
        self.harmonic = HarmonicResonance()
        self.fractal_dim = FractalDimensionCalculator()
        
        # Particle systems
        self.spoon_system = SpoonParticleSystem(max_spoons=12)
        
        # State
        self.time = 0.0
        self.frame = 0
        
        logger.info(f"✅ Enhanced Fractal Renderer initialized ({width}x{height})")
    
    def lerp_color(self, c1: Tuple[int, int, int], c2: Tuple[int, int, int], 
                   t: float) -> Tuple[int, int, int]:
        """Linear interpolation between colors"""
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )
    
    def get_palette_color(self, t: float, palette: str = 'golden') -> Tuple[int, int, int]:
        """Get color from palette based on t (0-1)"""
        colors = MOOD_PALETTES.get(palette, MOOD_PALETTES['golden'])
        n = len(colors)
        idx = t * (n - 1)
        lower = int(max(0, min(n - 2, idx)))
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return self.lerp_color(colors[lower], colors[upper], frac)
    
    def project_3d_to_2d(self, pos: Vector3, scale: float = 1.0,
                         rot_y: float = 0.0, rot_x: float = 0.0) -> Tuple[int, int]:
        """Project 3D point to 2D screen coordinates"""
        rotated = pos.rotate_y(rot_y).rotate_x(rot_x)
        
        z_offset = rotated.z + 100
        if z_offset <= 0:
            z_offset = 0.1
        
        perspective = 200 / z_offset
        
        x = int(self.center_x + rotated.x * scale * perspective)
        y = int(self.center_y - rotated.y * scale * perspective)
        
        return (x, y)
    
    def draw_point(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                   color: Tuple[int, int, int], size: float = 1.0, alpha: float = 1.0):
        """Draw a point with optional size and alpha"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        
        adjusted = (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha)
        )
        
        if size <= 1:
            draw.point((x, y), fill=adjusted)
        else:
            r = int(size / 2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=adjusted)
    
    def draw_glow(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                  color: Tuple[int, int, int], radius: float = 10.0, intensity: float = 1.0):
        """Draw a glowing point with falloff"""
        for r in range(int(radius), 0, -1):
            alpha = (1 - r / radius) * intensity * 0.3
            self.draw_point(draw, x, y, color, size=r * 2, alpha=alpha)
    
    def render_lorenz_attractor(self, draw: ImageDraw.ImageDraw,
                                 palette: str = 'golden',
                                 scale: float = 4.0):
        """Render the Lorenz attractor trail"""
        if len(self.lorenz.history) < 2:
            return
        
        rot_y = self.time * 0.2
        rot_x = math.sin(self.time * 0.1) * 0.3
        
        for i in range(1, len(self.lorenz.history)):
            pos = self.lorenz.history[i]
            normalized = Vector3(
                (pos.x - self.lorenz.bounds['x'][0]) / 
                (self.lorenz.bounds['x'][1] - self.lorenz.bounds['x'][0]) * 50 - 25,
                (pos.y - self.lorenz.bounds['y'][0]) /
                (self.lorenz.bounds['y'][1] - self.lorenz.bounds['y'][0]) * 60 - 30,
                (pos.z - self.lorenz.bounds['z'][0]) /
                (self.lorenz.bounds['z'][1] - self.lorenz.bounds['z'][0]) * 50
            )
            
            x, y = self.project_3d_to_2d(normalized, scale, rot_y, rot_x)
            
            t = i / len(self.lorenz.history)
            color = self.get_palette_color(t, palette)
            alpha = t * 0.8
            
            self.draw_point(draw, x, y, color, size=1.5, alpha=alpha)
    
    def render_rossler_attractor(self, draw: ImageDraw.ImageDraw,
                                  palette: str = 'calm',
                                  scale: float = 6.0):
        """Render the Rossler attractor trail"""
        if len(self.rossler.history) < 2:
            return
        
        rot_y = self.time * 0.15
        rot_x = math.cos(self.time * 0.08) * 0.2
        
        for i in range(1, len(self.rossler.history)):
            pos = self.rossler.history[i]
            x, y = self.project_3d_to_2d(pos, scale, rot_y, rot_x)
            
            t = i / len(self.rossler.history)
            color = self.get_palette_color(t, palette)
            alpha = t * 0.6
            
            self.draw_point(draw, x, y, color, size=1.0, alpha=alpha)
    
    def render_golden_spiral(self, draw: ImageDraw.ImageDraw,
                              palette: str = 'golden',
                              alpha_base: float = 0.4):
        """Render golden spiral overlay"""
        spiral_alpha = alpha_base + 0.2 * math.sin(self.time * 0.5)
        points = self.spiral.get_points(150, max_theta=8 * PI)
        
        for i, (sx, sy) in enumerate(points):
            px = int(self.center_x + sx * 3)
            py = int(self.center_y + sy * 3)
            
            if 0 <= px < self.width and 0 <= py < self.height:
                t = i / len(points)
                color = self.get_palette_color(t, palette)
                self.draw_point(draw, px, py, color, size=1.5, alpha=spiral_alpha * t)
    
    def render_flower_of_life(self, draw: ImageDraw.ImageDraw,
                               color: Tuple[int, int, int] = (100, 150, 200),
                               alpha: float = 0.3):
        """Render Flower of Life sacred geometry"""
        centers = self.flower.get_circle_centers()
        radius = self.flower.radius * 20  # Scale for display
        
        for cx, cy in centers:
            screen_x = int(self.center_x + cx * 20)
            screen_y = int(self.center_y + cy * 20)
            
            # Draw circle outline
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                px = int(screen_x + radius * math.cos(rad))
                py = int(screen_y + radius * math.sin(rad))
                
                if 0 <= px < self.width and 0 <= py < self.height:
                    self.draw_point(draw, px, py, color, size=1, alpha=alpha)
    
    def render_metatrons_cube(self, draw: ImageDraw.ImageDraw,
                               color: Tuple[int, int, int] = (180, 130, 220),
                               alpha: float = 0.4):
        """Render Metatron's Cube"""
        positions = self.metatron.circle_positions
        lines = self.metatron.get_connecting_lines()
        
        # Draw connecting lines
        for i, j in lines:
            x1 = int(self.center_x + positions[i][0])
            y1 = int(self.center_y + positions[i][1])
            x2 = int(self.center_x + positions[j][0])
            y2 = int(self.center_y + positions[j][1])
            
            # Draw line as series of points
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for s in range(steps + 1):
                t = s / steps
                px = int(x1 + (x2 - x1) * t)
                py = int(y1 + (y2 - y1) * t)
                if 0 <= px < self.width and 0 <= py < self.height:
                    self.draw_point(draw, px, py, color, size=0.5, alpha=alpha * 0.3)
        
        # Draw circles at nodes
        for x, y in positions:
            screen_x = int(self.center_x + x)
            screen_y = int(self.center_y + y)
            self.draw_glow(draw, screen_x, screen_y, color, radius=8, intensity=alpha)
    
    def render_particles(self, draw: ImageDraw.ImageDraw, scale: float = 4.0):
        """Render spoon particles"""
        rot_y = self.time * 0.2
        rot_x = math.sin(self.time * 0.1) * 0.3
        
        for p in self.spoon_system.particles:
            x, y = self.project_3d_to_2d(p.position, scale, rot_y, rot_x)
            alpha = p.life * p.energy
            self.draw_glow(draw, x, y, p.color, radius=p.size * 3, intensity=alpha)
    
    def render_goal_orbs(self, draw: ImageDraw.ImageDraw, goals: List[dict],
                          scale: float = 4.0, palette: str = 'golden'):
        """Render goals as glowing orbs positioned by Fibonacci"""
        if not goals:
            return
        
        rot_y = self.time * 0.1
        
        for i, goal in enumerate(goals):
            # Position using golden angle
            theta = i * GOLDEN_ANGLE_RAD
            r = math.sqrt(i + 1) * 30
            
            # Add 3D depth based on progress
            progress = goal.get('progress', 0) / 100
            z = (1 - progress) * 50
            
            pos = Vector3(
                r * math.cos(theta),
                r * math.sin(theta),
                z
            )
            
            x, y = self.project_3d_to_2d(pos, scale, rot_y, 0)
            
            # Color based on progress
            color = self.get_palette_color(progress, palette)
            
            # Size based on priority
            priority = goal.get('priority', 3)
            size = 10 + priority * 3
            
            # Glow intensity based on progress
            intensity = 0.3 + progress * 0.7
            
            self.draw_glow(draw, x, y, color, radius=size, intensity=intensity)
    
    def render_wellness_julia(self, wellness: float, stress: float) -> np.ndarray:
        """Generate Julia set based on wellness state"""
        self.julia.set_from_wellness(wellness, stress)
        
        x = np.linspace(-1.5, 1.5, self.width)
        y = np.linspace(-1.5, 1.5, self.height)
        X, Y = np.meshgrid(x, y)
        
        iterations = np.zeros((self.height, self.width))
        
        z_real = X.flatten()
        z_imag = Y.flatten()
        
        for i in range(len(z_real)):
            iters, _ = self.julia.iterate(z_real[i], z_imag[i])
            iterations.flat[i] = iters
        
        return iterations
    
    def render_combined_visualization(self, user_data: dict) -> Image.Image:
        """
        Create a complete visualization combining all mathematical systems
        
        user_data should contain:
        - wellness: 0-100
        - stress: 0-100
        - mood_score: 0-100
        - goals: list of goal dicts
        - spoons: current energy level
        """
        # Create base image
        img = Image.new('RGB', (self.width, self.height), (10, 10, 15))
        draw = ImageDraw.Draw(img)
        
        dt = 0.016
        self.time += dt
        self.frame += 1
        
        # Extract user data
        wellness = user_data.get('wellness', 50)
        stress = user_data.get('stress', 50)
        mood = user_data.get('mood_score', 50)
        goals = user_data.get('goals', [])
        spoons = user_data.get('spoons', 6)
        
        # Determine palette based on mood
        if mood >= 70:
            palette = 'energetic'
        elif mood >= 50:
            palette = 'golden'
        elif mood >= 30:
            palette = 'calm'
        else:
            palette = 'peaceful'
        
        # Step chaos systems
        self.lorenz.step_multiple(5, 0.005)
        self.rossler.step_multiple(3, 0.015)
        
        # Update particle system with goal attractors
        self.spoon_system.clear_goal_attractors()
        for i, goal in enumerate(goals[:5]):  # Top 5 goals
            theta = i * GOLDEN_ANGLE_RAD
            r = 30 * math.sqrt(i + 1)
            pos = Vector3(r * math.cos(theta), r * math.sin(theta), 0)
            priority = goal.get('priority', 3) / 5
            self.spoon_system.add_goal_attractor(pos, priority)
        
        # Spawn particles if needed
        while len(self.spoon_system.particles) < spoons:
            spawn_pos = Vector3(
                (np.random.random() - 0.5) * 100,
                (np.random.random() - 0.5) * 100,
                (np.random.random() - 0.5) * 50
            )
            color = self.get_palette_color(np.random.random(), palette)
            self.spoon_system.spawn_spoon(spawn_pos, color)
        
        self.spoon_system.update(dt)
        
        # === LAYER 1: Background sacred geometry ===
        if self.frame % 2 == 0:  # Optimize by updating less frequently
            # Flower of Life (very subtle)
            fol_color = self.harmonic.get_wellness_color(wellness)
            self.render_flower_of_life(draw, fol_color, alpha=0.1)
        
        # === LAYER 2: Chaos attractors ===
        # Lorenz for life-balance visualization
        self.render_lorenz_attractor(draw, palette, scale=3.5)
        
        # Rossler for mood patterns (more prominent when stressed)
        rossler_alpha = 0.3 + (stress / 100) * 0.4
        if stress > 40:
            self.render_rossler_attractor(draw, 'calm', scale=5.0)
        
        # === LAYER 3: Golden spiral overlay ===
        spiral_alpha = 0.2 + (wellness / 100) * 0.3
        self.render_golden_spiral(draw, 'golden', spiral_alpha)
        
        # === LAYER 4: Goal orbs ===
        self.render_goal_orbs(draw, goals, scale=3.5, palette=palette)
        
        # === LAYER 5: Spoon particles ===
        self.render_particles(draw, scale=3.5)
        
        # === LAYER 6: Central focus point (wellness indicator) ===
        wellness_color = self.harmonic.get_wellness_color(wellness)
        pulse = 0.8 + 0.2 * math.sin(self.time * 2)
        self.draw_glow(draw, self.center_x, self.center_y, wellness_color, 
                      radius=15 + wellness / 10, intensity=pulse * 0.6)
        
        # Add subtle vignette
        for edge in range(30):
            alpha = edge / 30 * 0.3
            for x in range(self.width):
                self.draw_point(draw, x, edge, (0, 0, 0), 1, alpha)
                self.draw_point(draw, x, self.height - 1 - edge, (0, 0, 0), 1, alpha)
            for y in range(self.height):
                self.draw_point(draw, edge, y, (0, 0, 0), 1, alpha)
                self.draw_point(draw, self.width - 1 - edge, y, (0, 0, 0), 1, alpha)
        
        return img
    
    def generate_2d_fractal(self, fractal_type: str = 'mandelbrot',
                            wellness: float = 50,
                            stress: float = 50,
                            zoom: float = 1.0,
                            center: Tuple[float, float] = (-0.5, 0)) -> np.ndarray:
        """Generate 2D fractal based on type and user state"""
        if fractal_type == 'julia':
            self.julia.set_from_wellness(wellness, stress)
            return self.render_wellness_julia(wellness, stress)
        else:
            x = np.linspace(-2/zoom + center[0], 2/zoom + center[0], self.width)
            y = np.linspace(-2/zoom + center[1], 2/zoom + center[1], self.height)
            X, Y = np.meshgrid(x, y)
            
            iterations = np.zeros((self.height, self.width))
            
            c_real = X.flatten()
            c_imag = Y.flatten()
            
            for i in range(len(c_real)):
                iters, _ = self.mandelbrot.iterate(c_real[i], c_imag[i])
                iterations.flat[i] = iters
            
            return iterations
    
    def iterations_to_image(self, iterations: np.ndarray, 
                            palette: str = 'golden') -> Image.Image:
        """Convert iteration counts to colored image"""
        max_iter = iterations.max()
        if max_iter == 0:
            max_iter = 1
        
        normalized = iterations / max_iter
        
        # Create RGB image
        img = Image.new('RGB', (self.width, self.height))
        pixels = img.load()
        
        for y in range(self.height):
            for x in range(self.width):
                t = normalized[y, x]
                color = self.get_palette_color(t, palette)
                pixels[x, y] = color
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class Database:
    """Production-ready SQLite database with enhanced schema"""
    
    def __init__(self, db_path: str = "life_fractal_ultimate.db"):
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
                spoons_max INTEGER DEFAULT 12,
                accessibility_mode TEXT DEFAULT 'standard'
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
                fractal_position_x REAL,
                fractal_position_y REAL,
                fractal_position_z REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
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
                longest_streak INTEGER DEFAULT 0,
                total_completions INTEGER DEFAULT 0,
                spoon_cost INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Daily entries with chaos metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_entries (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                mood_level INTEGER DEFAULT 50,
                stress_level INTEGER DEFAULT 50,
                energy_level INTEGER DEFAULT 50,
                sleep_hours REAL DEFAULT 7.0,
                sleep_quality INTEGER DEFAULT 50,
                spoons_available INTEGER DEFAULT 12,
                spoons_used INTEGER DEFAULT 0,
                wellness_index REAL DEFAULT 50.0,
                lorenz_wing TEXT,
                rossler_phase REAL,
                fractal_dimension REAL,
                journal_entry TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)
            )
        ''')
        
        # Pet state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pet_state (
                user_id TEXT PRIMARY KEY,
                species TEXT DEFAULT 'cat',
                name TEXT DEFAULT 'Buddy',
                hunger REAL DEFAULT 50.0,
                energy REAL DEFAULT 50.0,
                mood REAL DEFAULT 50.0,
                level INTEGER DEFAULT 1,
                experience INTEGER DEFAULT 0,
                evolution_stage INTEGER DEFAULT 1,
                bond REAL DEFAULT 50.0,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Chaos state tracking (for visualization continuity)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chaos_state (
                user_id TEXT PRIMARY KEY,
                lorenz_x REAL DEFAULT 1.0,
                lorenz_y REAL DEFAULT 1.0,
                lorenz_z REAL DEFAULT 1.0,
                rossler_x REAL DEFAULT 1.0,
                rossler_y REAL DEFAULT 1.0,
                rossler_z REAL DEFAULT 1.0,
                coupling_strength REAL DEFAULT 0.05,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Accessibility preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accessibility (
                user_id TEXT PRIMARY KEY,
                aphantasia_mode INTEGER DEFAULT 0,
                autism_safe_colors INTEGER DEFAULT 0,
                reduce_motion INTEGER DEFAULT 0,
                high_contrast INTEGER DEFAULT 0,
                screen_reader_friendly INTEGER DEFAULT 0,
                dysgraphia_support INTEGER DEFAULT 1,
                text_size_multiplier REAL DEFAULT 1.0,
                preferred_palette TEXT DEFAULT 'golden',
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email: str, password: str, 
                   first_name: str = '', last_name: str = '') -> Optional[str]:
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            user_id = secrets.token_urlsafe(16)
            now = datetime.now(timezone.utc).isoformat()
            trial_end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            
            cursor.execute('''
                INSERT INTO users (id, email, password_hash, first_name, last_name,
                                  created_at, subscription_status, trial_end_date)
                VALUES (?, ?, ?, ?, ?, ?, 'trial', ?)
            ''', (user_id, email.lower(), generate_password_hash(password),
                  first_name, last_name, now, trial_end))
            
            # Initialize related records
            cursor.execute('''
                INSERT INTO pet_state (user_id, last_updated) VALUES (?, ?)
            ''', (user_id, now))
            
            cursor.execute('''
                INSERT INTO chaos_state (user_id, last_updated) VALUES (?, ?)
            ''', (user_id, now))
            
            cursor.execute('''
                INSERT INTO accessibility (user_id) VALUES (?)
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Created user: {email}")
            return user_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"User already exists: {email}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def authenticate(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE email = ? AND is_active = 1
            ''', (email.lower(),))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and check_password_hash(row['password_hash'], password):
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return None
    
    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Get user error: {e}")
            return None
    
    def get_goals(self, user_id: str) -> List[dict]:
        """Get all goals for user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL
                ORDER BY priority DESC, created_at DESC
            ''', (user_id,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Get goals error: {e}")
            return []
    
    def create_goal(self, user_id: str, title: str, description: str = '',
                   category: str = 'personal', term: str = 'medium',
                   priority: int = 3, target_date: str = None) -> Optional[str]:
        """Create a new goal with fractal positioning"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Calculate fractal position
            cursor.execute('SELECT COUNT(*) as count FROM goals WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()['count']
            
            theta = count * GOLDEN_ANGLE_RAD
            r = math.sqrt(count + 1) * 30
            
            goal_id = secrets.token_urlsafe(8)
            now = datetime.now(timezone.utc).isoformat()
            
            cursor.execute('''
                INSERT INTO goals (id, user_id, title, description, category, term,
                                  priority, target_date, created_at,
                                  fractal_position_x, fractal_position_y, fractal_position_z)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (goal_id, user_id, title, description, category, term,
                  priority, target_date, now,
                  r * math.cos(theta), r * math.sin(theta), priority * 10))
            
            conn.commit()
            conn.close()
            return goal_id
            
        except Exception as e:
            logger.error(f"Create goal error: {e}")
            return None
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """Update goal progress"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE goals SET progress = ? WHERE id = ?
            ''', (min(100, max(0, progress)), goal_id))
            
            # Check if completed
            if progress >= 100:
                cursor.execute('''
                    UPDATE goals SET completed_at = ? WHERE id = ?
                ''', (datetime.now(timezone.utc).isoformat(), goal_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Update progress error: {e}")
            return False
    
    def save_daily_entry(self, user_id: str, entry_data: dict) -> bool:
        """Save daily wellness entry"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            entry_id = secrets.token_urlsafe(8)
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            now = datetime.now(timezone.utc).isoformat()
            
            # Calculate wellness index using golden ratio weighting
            mood = entry_data.get('mood_level', 50)
            stress = entry_data.get('stress_level', 50)
            energy = entry_data.get('energy_level', 50)
            sleep_q = entry_data.get('sleep_quality', 50)
            
            wellness = (
                mood * PHI_INVERSE +
                (100 - stress) * (PHI_INVERSE ** 2) +
                energy * (PHI_INVERSE ** 3) +
                sleep_q * (PHI_INVERSE ** 4)
            ) / (PHI_INVERSE + PHI_INVERSE**2 + PHI_INVERSE**3 + PHI_INVERSE**4)
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_entries 
                (id, user_id, date, mood_level, stress_level, energy_level,
                 sleep_hours, sleep_quality, spoons_available, spoons_used,
                 wellness_index, journal_entry, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (entry_id, user_id, today,
                  mood, stress, energy,
                  entry_data.get('sleep_hours', 7),
                  sleep_q,
                  entry_data.get('spoons_available', 12),
                  entry_data.get('spoons_used', 0),
                  wellness,
                  entry_data.get('journal_entry', ''),
                  now))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Save entry error: {e}")
            return False
    
    def get_today_entry(self, user_id: str) -> Optional[dict]:
        """Get today's entry"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT * FROM daily_entries WHERE user_id = ? AND date = ?
            ''', (user_id, today))
            
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Get today error: {e}")
            return None
    
    def get_entry_history(self, user_id: str, days: int = 30) -> List[dict]:
        """Get entry history"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM daily_entries 
                WHERE user_id = ? 
                ORDER BY date DESC LIMIT ?
            ''', (user_id, days))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
            
        except Exception as e:
            logger.error(f"Get history error: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize components
db = Database()
renderer = EnhancedFractalRenderer(512, 512)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        user_id = db.create_user(email, password, first_name, last_name)
        
        if user_id:
            session['user_id'] = user_id
            return jsonify({
                'success': True,
                'user_id': user_id,
                'message': 'Welcome to Life Fractal Intelligence!'
            })
        else:
            return jsonify({'error': 'Email already registered'}), 409
            
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        user = db.authenticate(email, password)
        
        if user:
            session['user_id'] = user['id']
            return jsonify({
                'success': True,
                'user_id': user['id'],
                'first_name': user['first_name'],
                'subscription_status': user['subscription_status']
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════
# USER DATA ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/dashboard')
def dashboard(user_id):
    """Get complete dashboard data"""
    try:
        user = db.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        goals = db.get_goals(user_id)
        today = db.get_today_entry(user_id)
        history = db.get_entry_history(user_id, 7)
        
        # Calculate chaos metrics
        if today:
            wellness = today.get('wellness_index', 50)
            stress = today.get('stress_level', 50)
        else:
            wellness = 50
            stress = 50
        
        # Determine which "wing" of life the user is in
        lorenz_wing = renderer.lorenz.get_wing()
        rossler_phase = renderer.rossler.get_spiral_phase()
        
        return jsonify({
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'subscription_status': user['subscription_status']
            },
            'goals': goals,
            'today': today or {
                'mood_level': 50,
                'stress_level': 50,
                'energy_level': 50,
                'spoons_available': 12
            },
            'history_summary': {
                'days': len(history),
                'avg_wellness': sum(h.get('wellness_index', 50) for h in history) / max(1, len(history)),
                'avg_mood': sum(h.get('mood_level', 50) for h in history) / max(1, len(history))
            },
            'chaos_insight': {
                'life_balance_wing': lorenz_wing,
                'mood_cycle_phase': rossler_phase,
                'balance_interpretation': (
                    'Currently focused on growth and expansion' if lorenz_wing == 'right'
                    else 'Currently focused on reflection and consolidation'
                )
            },
            'sacred_math': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE,
                'wellness_harmonic': renderer.harmonic.wellness_to_interval(wellness)
            }
        })
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500


@app.route('/api/user/<user_id>/goals', methods=['GET', 'POST'])
def goals(user_id):
    """Get or create goals"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            goal_id = db.create_goal(
                user_id,
                data.get('title', 'New Goal'),
                data.get('description', ''),
                data.get('category', 'personal'),
                data.get('term', 'medium'),
                data.get('priority', 3),
                data.get('target_date')
            )
            
            if goal_id:
                return jsonify({'success': True, 'goal_id': goal_id})
            return jsonify({'error': 'Failed to create goal'}), 500
            
        except Exception as e:
            logger.error(f"Create goal error: {e}")
            return jsonify({'error': 'Failed to create goal'}), 500
    
    else:
        return jsonify({'goals': db.get_goals(user_id)})


@app.route('/api/user/<user_id>/today', methods=['GET', 'POST'])
def today_entry(user_id):
    """Get or update today's entry"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            success = db.save_daily_entry(user_id, data)
            
            if success:
                return jsonify({'success': True})
            return jsonify({'error': 'Failed to save entry'}), 500
            
        except Exception as e:
            logger.error(f"Save entry error: {e}")
            return jsonify({'error': 'Failed to save entry'}), 500
    
    else:
        entry = db.get_today_entry(user_id)
        return jsonify({'today': entry or {}})


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/user/<user_id>/visualization')
def visualization(user_id):
    """Generate combined visualization"""
    try:
        goals = db.get_goals(user_id)
        today = db.get_today_entry(user_id)
        
        user_data = {
            'wellness': today.get('wellness_index', 50) if today else 50,
            'stress': today.get('stress_level', 50) if today else 50,
            'mood_score': today.get('mood_level', 50) if today else 50,
            'goals': goals,
            'spoons': today.get('spoons_available', 12) if today else 12
        }
        
        img = renderer.render_combined_visualization(user_data)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': 'Visualization failed'}), 500


@app.route('/api/user/<user_id>/fractal')
def fractal(user_id):
    """Generate fractal based on user state"""
    try:
        fractal_type = request.args.get('type', 'mandelbrot')
        zoom = float(request.args.get('zoom', 1.0))
        
        today = db.get_today_entry(user_id)
        wellness = today.get('wellness_index', 50) if today else 50
        stress = today.get('stress_level', 50) if today else 50
        
        iterations = renderer.generate_2d_fractal(
            fractal_type=fractal_type,
            wellness=wellness,
            stress=stress,
            zoom=zoom
        )
        
        # Determine palette based on mood
        mood = today.get('mood_level', 50) if today else 50
        if mood >= 70:
            palette = 'energetic'
        elif mood >= 50:
            palette = 'golden'
        else:
            palette = 'calm'
        
        img = renderer.iterations_to_image(iterations, palette)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Fractal error: {e}")
        return jsonify({'error': 'Fractal generation failed'}), 500


@app.route('/api/user/<user_id>/visualization/config')
def visualization_config(user_id):
    """Get Three.js visualization configuration"""
    try:
        goals = db.get_goals(user_id)
        today = db.get_today_entry(user_id)
        
        wellness = today.get('wellness_index', 50) if today else 50
        stress = today.get('stress_level', 50) if today else 50
        mood = today.get('mood_level', 50) if today else 50
        
        # Calculate Julia set parameters
        renderer.julia.set_from_wellness(wellness, stress)
        
        return jsonify({
            'sacred_math': {
                'phi': PHI,
                'phi_inverse': PHI_INVERSE,
                'golden_angle_rad': GOLDEN_ANGLE_RAD,
                'fibonacci': FIBONACCI[:15]
            },
            'chaos_params': {
                'lorenz': {
                    'sigma': LORENZ_SIGMA,
                    'rho': LORENZ_RHO,
                    'beta': LORENZ_BETA,
                    'current_wing': renderer.lorenz.get_wing()
                },
                'rossler': {
                    'a': ROSSLER_A,
                    'b': ROSSLER_B,
                    'c': ROSSLER_C,
                    'phase': renderer.rossler.get_spiral_phase()
                }
            },
            'julia_params': {
                'c_real': renderer.julia.c_real,
                'c_imag': renderer.julia.c_imag
            },
            'goals': [
                {
                    'id': g['id'],
                    'title': g['title'],
                    'progress': g['progress'],
                    'priority': g['priority'],
                    'position': {
                        'x': g.get('fractal_position_x', 0),
                        'y': g.get('fractal_position_y', 0),
                        'z': g.get('fractal_position_z', 0)
                    }
                }
                for g in goals
            ],
            'user_state': {
                'wellness': wellness,
                'stress': stress,
                'mood': mood,
                'spoons': today.get('spoons_available', 12) if today else 12
            },
            'palette': {
                'name': 'golden' if mood >= 50 else 'calm',
                'colors': MOOD_PALETTES['golden'] if mood >= 50 else MOOD_PALETTES['calm']
            }
        })
        
    except Exception as e:
        logger.error(f"Config error: {e}")
        return jsonify({'error': 'Config failed'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/sacred-math')
def sacred_math():
    """Get all sacred mathematics constants"""
    return jsonify({
        'phi': PHI,
        'phi_inverse': PHI_INVERSE,
        'phi_squared': PHI_SQUARED,
        'golden_angle_degrees': GOLDEN_ANGLE,
        'golden_angle_radians': GOLDEN_ANGLE_RAD,
        'fibonacci': FIBONACCI,
        'lucas': LUCAS,
        'platonic_solids': PLATONIC_SOLIDS,
        'pythagorean_ratios': PYTHAGOREAN_RATIOS,
        'chaos_params': {
            'lorenz': {'sigma': LORENZ_SIGMA, 'rho': LORENZ_RHO, 'beta': LORENZ_BETA},
            'rossler': {'a': ROSSLER_A, 'b': ROSSLER_B, 'c': ROSSLER_C}
        },
        'constants': {
            'pi': PI,
            'tau': TAU,
            'e': E,
            'sqrt2': SQRT2,
            'sqrt3': SQRT3,
            'sqrt5': SQRT5
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '7.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'features': {
            'lorenz_attractor': True,
            'rossler_attractor': True,
            'coupled_chaos': True,
            'particle_system': True,
            'sacred_geometry': True,
            'harmonic_resonance': True,
            'fractal_dimension': True,
            'spoon_theory': True
        }
    })


@app.route('/')
def index():
    """Serve main page"""
    return render_template_string(MAIN_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence v7.0</title>
    <style>
        :root {
            --phi: 1.618033988749895;
            --golden: #FFD700;
            --cosmic: #8B5CF6;
            --bg-dark: #0a0a0f;
            --bg-card: #15151f;
            --text: #e0e0e0;
            --text-dim: #888;
            --border: #2a2a3a;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Golden ratio based spacing */
        .container {
            max-width: calc(1000px * var(--phi));
            margin: 0 auto;
            padding: calc(1rem * var(--phi));
        }
        
        header {
            text-align: center;
            padding: calc(2rem * var(--phi)) 1rem;
            background: linear-gradient(180deg, #1a1a2e 0%, var(--bg-dark) 100%);
            border-bottom: 1px solid var(--border);
        }
        
        h1 {
            font-size: calc(2rem * var(--phi));
            background: linear-gradient(135deg, var(--golden), var(--cosmic));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 1.1rem;
        }
        
        .math-badge {
            display: inline-block;
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid var(--cosmic);
            border-radius: 20px;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            font-size: 0.85rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: calc(1rem * var(--phi));
            margin-top: 2rem;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: calc(1rem * var(--phi));
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
        }
        
        .card h2 {
            color: var(--golden);
            font-size: 1.3rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .visualization {
            width: 100%;
            aspect-ratio: 1;
            background: #000;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        }
        
        .visualization img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .visualization canvas {
            width: 100%;
            height: 100%;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }
        
        .stat {
            background: rgba(255, 255, 255, 0.05);
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--golden);
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .slider-container {
            margin: 1rem 0;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.3rem;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: var(--border);
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--golden);
            cursor: pointer;
        }
        
        button {
            background: linear-gradient(135deg, var(--golden), #FFA500);
            color: #000;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
            margin-top: 1rem;
        }
        
        button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
        }
        
        .chaos-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.8rem;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .wing-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--cosmic);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        .spoon-container {
            display: flex;
            gap: 0.3rem;
            flex-wrap: wrap;
            margin: 1rem 0;
        }
        
        .spoon {
            font-size: 1.5rem;
            opacity: 0.3;
            transition: opacity 0.3s, transform 0.3s;
        }
        
        .spoon.active {
            opacity: 1;
            animation: float 2s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }
        
        .auth-form {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .auth-form input {
            width: 100%;
            padding: 0.8rem;
            margin: 0.5rem 0;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
        }
        
        .auth-form input:focus {
            outline: none;
            border-color: var(--golden);
        }
        
        .sacred-geometry {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.03;
            z-index: -1;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-dim);
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }
        
        .phi-value {
            font-family: monospace;
            color: var(--golden);
        }
    </style>
</head>
<body>
    <svg class="sacred-geometry" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice">
        <defs>
            <pattern id="flowerOfLife" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="10" cy="2" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="10" cy="18" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="3" cy="6" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="17" cy="6" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="3" cy="14" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
                <circle cx="17" cy="14" r="8" fill="none" stroke="currentColor" stroke-width="0.2"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#flowerOfLife)"/>
    </svg>
    
    <header>
        <h1>🌀 Life Fractal Intelligence</h1>
        <p class="subtitle">Mathematical Harmony for Neurodivergent Minds</p>
        <div style="margin-top: 1rem;">
            <span class="math-badge">🦋 Lorenz Attractor</span>
            <span class="math-badge">🌀 Rossler Spiral</span>
            <span class="math-badge">🌻 Golden Ratio</span>
            <span class="math-badge">✨ Particle Swarm</span>
            <span class="math-badge">🎵 Harmonic Resonance</span>
        </div>
    </header>
    
    <div class="container">
        <div id="app">
            <!-- Auth View -->
            <div id="auth-view" class="auth-form">
                <h2 style="text-align: center; margin-bottom: 1.5rem;">✨ Welcome</h2>
                <div id="login-form">
                    <input type="email" id="login-email" placeholder="Email">
                    <input type="password" id="login-password" placeholder="Password">
                    <button onclick="login()">Login</button>
                    <p style="text-align: center; margin-top: 1rem; color: var(--text-dim);">
                        New? <a href="#" onclick="showRegister()" style="color: var(--golden);">Create account</a>
                    </p>
                </div>
                <div id="register-form" style="display: none;">
                    <input type="text" id="reg-first" placeholder="First Name">
                    <input type="text" id="reg-last" placeholder="Last Name">
                    <input type="email" id="reg-email" placeholder="Email">
                    <input type="password" id="reg-password" placeholder="Password (6+ chars)">
                    <button onclick="register()">Create Account</button>
                    <p style="text-align: center; margin-top: 1rem; color: var(--text-dim);">
                        Have account? <a href="#" onclick="showLogin()" style="color: var(--golden);">Login</a>
                    </p>
                </div>
            </div>
            
            <!-- Dashboard View -->
            <div id="dashboard-view" style="display: none;">
                <div class="grid">
                    <!-- Chaos Visualization Card -->
                    <div class="card">
                        <h2>🦋 Chaos Visualization</h2>
                        <div class="visualization" id="viz-container">
                            <canvas id="chaos-canvas"></canvas>
                        </div>
                        <div class="chaos-indicator">
                            <div class="wing-indicator"></div>
                            <span id="chaos-insight">Loading chaos state...</span>
                        </div>
                    </div>
                    
                    <!-- Daily Check-in Card -->
                    <div class="card">
                        <h2>🥄 Energy Check-in</h2>
                        <div class="spoon-container" id="spoons">
                            <!-- Spoons rendered by JS -->
                        </div>
                        
                        <div class="slider-container">
                            <div class="slider-label">
                                <span>Mood</span>
                                <span id="mood-value">50</span>
                            </div>
                            <input type="range" id="mood-slider" min="0" max="100" value="50" 
                                   oninput="updateSlider('mood')">
                        </div>
                        
                        <div class="slider-container">
                            <div class="slider-label">
                                <span>Stress</span>
                                <span id="stress-value">50</span>
                            </div>
                            <input type="range" id="stress-slider" min="0" max="100" value="50"
                                   oninput="updateSlider('stress')">
                        </div>
                        
                        <div class="slider-container">
                            <div class="slider-label">
                                <span>Energy</span>
                                <span id="energy-value">50</span>
                            </div>
                            <input type="range" id="energy-slider" min="0" max="100" value="50"
                                   oninput="updateSlider('energy')">
                        </div>
                        
                        <button onclick="saveCheckin()">Save Check-in</button>
                    </div>
                    
                    <!-- Goals Card -->
                    <div class="card">
                        <h2>🎯 Goals (Fibonacci Positioned)</h2>
                        <div id="goals-list">
                            <!-- Goals rendered by JS -->
                        </div>
                        <button onclick="showAddGoal()">+ Add Goal</button>
                    </div>
                    
                    <!-- Sacred Math Stats -->
                    <div class="card">
                        <h2>📐 Sacred Mathematics</h2>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value phi-value" id="phi-display">φ</div>
                                <div class="stat-label">Golden Ratio</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="wellness-display">--</div>
                                <div class="stat-label">Wellness Index</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="harmonic-display">--</div>
                                <div class="stat-label">Harmonic</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="dimension-display">--</div>
                                <div class="stat-label">Life Complexity</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>Life Fractal Intelligence v7.0 | φ = <span class="phi-value">1.618033988749895</span></p>
        <p style="margin-top: 0.5rem;">Planning tools designed for brains like yours 💜</p>
    </footer>
    
    <script>
        // ═══════════════════════════════════════════════════════════════════════
        // SACRED MATHEMATICS CONSTANTS
        // ═══════════════════════════════════════════════════════════════════════
        
        const PHI = 1.618033988749895;
        const PHI_INV = 0.618033988749895;
        const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
        const TAU = Math.PI * 2;
        
        // ═══════════════════════════════════════════════════════════════════════
        // STATE
        // ═══════════════════════════════════════════════════════════════════════
        
        let userId = localStorage.getItem('userId');
        let dashboardData = null;
        let animationFrame = null;
        
        // Chaos system state
        let lorenz = { x: 1, y: 1, z: 1, history: [] };
        let rossler = { x: 1, y: 1, z: 1, history: [] };
        let time = 0;
        
        // ═══════════════════════════════════════════════════════════════════════
        // INITIALIZATION
        // ═══════════════════════════════════════════════════════════════════════
        
        document.addEventListener('DOMContentLoaded', () => {
            if (userId) {
                showDashboard();
            }
            document.getElementById('phi-display').textContent = 'φ = ' + PHI.toFixed(6);
        });
        
        // ═══════════════════════════════════════════════════════════════════════
        // AUTH FUNCTIONS
        // ═══════════════════════════════════════════════════════════════════════
        
        function showLogin() {
            document.getElementById('login-form').style.display = 'block';
            document.getElementById('register-form').style.display = 'none';
        }
        
        function showRegister() {
            document.getElementById('login-form').style.display = 'none';
            document.getElementById('register-form').style.display = 'block';
        }
        
        async function login() {
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    userId = data.user_id;
                    localStorage.setItem('userId', userId);
                    showDashboard();
                } else {
                    alert(data.error || 'Login failed');
                }
            } catch (e) {
                console.error('Login error:', e);
                alert('Login failed');
            }
        }
        
        async function register() {
            const firstName = document.getElementById('reg-first').value;
            const lastName = document.getElementById('reg-last').value;
            const email = document.getElementById('reg-email').value;
            const password = document.getElementById('reg-password').value;
            
            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        email, 
                        password, 
                        first_name: firstName, 
                        last_name: lastName 
                    })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    userId = data.user_id;
                    localStorage.setItem('userId', userId);
                    showDashboard();
                } else {
                    alert(data.error || 'Registration failed');
                }
            } catch (e) {
                console.error('Register error:', e);
                alert('Registration failed');
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // DASHBOARD FUNCTIONS
        // ═══════════════════════════════════════════════════════════════════════
        
        async function showDashboard() {
            document.getElementById('auth-view').style.display = 'none';
            document.getElementById('dashboard-view').style.display = 'block';
            
            await loadDashboard();
            initChaosVisualization();
        }
        
        async function loadDashboard() {
            try {
                const res = await fetch(`/api/user/${userId}/dashboard`);
                dashboardData = await res.json();
                
                renderSpoons(dashboardData.today?.spoons_available || 12);
                renderGoals(dashboardData.goals || []);
                updateStats();
                updateChaosInsight();
                
                // Set slider values
                if (dashboardData.today) {
                    document.getElementById('mood-slider').value = dashboardData.today.mood_level || 50;
                    document.getElementById('stress-slider').value = dashboardData.today.stress_level || 50;
                    document.getElementById('energy-slider').value = dashboardData.today.energy_level || 50;
                    updateSlider('mood');
                    updateSlider('stress');
                    updateSlider('energy');
                }
            } catch (e) {
                console.error('Load dashboard error:', e);
            }
        }
        
        function renderSpoons(count) {
            const container = document.getElementById('spoons');
            container.innerHTML = '';
            for (let i = 0; i < 12; i++) {
                const spoon = document.createElement('span');
                spoon.className = 'spoon' + (i < count ? ' active' : '');
                spoon.textContent = '🥄';
                spoon.style.animationDelay = (i * 0.1) + 's';
                container.appendChild(spoon);
            }
        }
        
        function renderGoals(goals) {
            const container = document.getElementById('goals-list');
            if (!goals.length) {
                container.innerHTML = '<p style="color: var(--text-dim);">No goals yet. Add your first goal!</p>';
                return;
            }
            
            container.innerHTML = goals.slice(0, 5).map((g, i) => {
                const theta = i * GOLDEN_ANGLE;
                const r = Math.sqrt(i + 1) * 20;
                return `
                    <div class="stat" style="margin-bottom: 0.5rem; position: relative;">
                        <div style="position: absolute; left: 10px; top: 50%; transform: translateY(-50%); 
                                    width: 8px; height: 8px; border-radius: 50%; 
                                    background: hsl(${(g.progress / 100) * 120}, 80%, 50%);"></div>
                        <div style="padding-left: 20px;">
                            <strong>${g.title}</strong>
                            <div style="width: 100%; height: 4px; background: var(--border); border-radius: 2px; margin-top: 5px;">
                                <div style="width: ${g.progress}%; height: 100%; background: var(--golden); border-radius: 2px;"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function updateStats() {
            if (!dashboardData) return;
            
            const wellness = dashboardData.history_summary?.avg_wellness || 50;
            document.getElementById('wellness-display').textContent = wellness.toFixed(1);
            
            const harmonic = dashboardData.sacred_math?.wellness_harmonic || 'unison';
            document.getElementById('harmonic-display').textContent = harmonic;
            
            // Fake fractal dimension for now (would be calculated from real data)
            const dimension = 1.2 + (wellness / 100) * 0.5;
            document.getElementById('dimension-display').textContent = dimension.toFixed(2);
        }
        
        function updateChaosInsight() {
            if (!dashboardData?.chaos_insight) return;
            
            const insight = dashboardData.chaos_insight;
            document.getElementById('chaos-insight').textContent = insight.balance_interpretation;
        }
        
        function updateSlider(type) {
            const value = document.getElementById(type + '-slider').value;
            document.getElementById(type + '-value').textContent = value;
        }
        
        async function saveCheckin() {
            const mood = parseInt(document.getElementById('mood-slider').value);
            const stress = parseInt(document.getElementById('stress-slider').value);
            const energy = parseInt(document.getElementById('energy-slider').value);
            
            try {
                await fetch(`/api/user/${userId}/today`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        mood_level: mood,
                        stress_level: stress,
                        energy_level: energy,
                        spoons_available: 12 - Math.floor(stress / 10)
                    })
                });
                
                await loadDashboard();
                alert('Check-in saved! ✨');
            } catch (e) {
                console.error('Save error:', e);
                alert('Failed to save');
            }
        }
        
        function showAddGoal() {
            const title = prompt('Goal title:');
            if (!title) return;
            
            fetch(`/api/user/${userId}/goals`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, priority: 3 })
            }).then(() => loadDashboard());
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // CHAOS VISUALIZATION
        // ═══════════════════════════════════════════════════════════════════════
        
        function initChaosVisualization() {
            const canvas = document.getElementById('chaos-canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            const container = document.getElementById('viz-container');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            
            // Reset chaos systems
            lorenz = { x: 1, y: 1, z: 1, history: [] };
            rossler = { x: 1, y: 1, z: 1, history: [] };
            time = 0;
            
            animate();
        }
        
        function stepLorenz(dt = 0.01) {
            const sigma = 10, rho = 28, beta = 8/3;
            
            const dx = sigma * (lorenz.y - lorenz.x);
            const dy = lorenz.x * (rho - lorenz.z) - lorenz.y;
            const dz = lorenz.x * lorenz.y - beta * lorenz.z;
            
            lorenz.x += dx * dt;
            lorenz.y += dy * dt;
            lorenz.z += dz * dt;
            
            lorenz.history.push({ x: lorenz.x, y: lorenz.y, z: lorenz.z });
            if (lorenz.history.length > 1500) lorenz.history.shift();
        }
        
        function stepRossler(dt = 0.02) {
            const a = 0.2, b = 0.2, c = 5.7;
            
            const dx = -rossler.y - rossler.z;
            const dy = rossler.x + a * rossler.y;
            const dz = b + rossler.z * (rossler.x - c);
            
            rossler.x += dx * dt;
            rossler.y += dy * dt;
            rossler.z += dz * dt;
            
            rossler.history.push({ x: rossler.x, y: rossler.y, z: rossler.z });
            if (rossler.history.length > 1000) rossler.history.shift();
        }
        
        function animate() {
            const canvas = document.getElementById('chaos-canvas');
            const ctx = canvas.getContext('2d');
            
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;
            
            // Clear with fade effect
            ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
            ctx.fillRect(0, 0, w, h);
            
            // Step chaos systems
            for (let i = 0; i < 5; i++) {
                stepLorenz(0.005);
                stepRossler(0.01);
            }
            
            time += 0.016;
            
            // Rotation
            const rotY = time * 0.2;
            const rotX = Math.sin(time * 0.1) * 0.3;
            
            // Draw Lorenz attractor
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            
            for (let i = 1; i < lorenz.history.length; i++) {
                const p = lorenz.history[i];
                
                // Normalize and project
                const nx = (p.x + 25) / 50 * 100 - 50;
                const ny = (p.y + 30) / 60 * 100 - 50;
                const nz = p.z;
                
                // Rotate
                const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
                const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
                
                let rx = nx * cosY + nz * sinY;
                let ry = ny;
                let rz = -nx * sinY + nz * cosY;
                
                const ty = ry * cosX - rz * sinX;
                rz = ry * sinX + rz * cosX;
                ry = ty;
                
                // Perspective
                const scale = 200 / (200 + rz);
                const sx = cx + rx * 2 * scale;
                const sy = cy - ry * 2 * scale;
                
                if (i === 1) {
                    ctx.moveTo(sx, sy);
                } else {
                    ctx.lineTo(sx, sy);
                }
            }
            ctx.stroke();
            
            // Draw Rossler attractor
            ctx.strokeStyle = 'rgba(139, 92, 246, 0.4)';
            ctx.beginPath();
            
            for (let i = 1; i < rossler.history.length; i++) {
                const p = rossler.history[i];
                
                // Rotate
                const cosY = Math.cos(rotY * 0.7), sinY = Math.sin(rotY * 0.7);
                
                const rx = p.x * cosY + p.z * sinY;
                const ry = p.y;
                const rz = -p.x * sinY + p.z * cosY;
                
                const scale = 200 / (200 + rz * 2);
                const sx = cx + rx * 4 * scale;
                const sy = cy - ry * 4 * scale;
                
                if (i === 1) {
                    ctx.moveTo(sx, sy);
                } else {
                    ctx.lineTo(sx, sy);
                }
            }
            ctx.stroke();
            
            // Draw golden spiral overlay
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.2)';
            ctx.beginPath();
            for (let i = 0; i < 200; i++) {
                const theta = i * 0.1;
                const r = 3 * Math.pow(PHI, theta / (Math.PI / 2));
                const x = cx + r * Math.cos(theta + time * 0.1);
                const y = cy + r * Math.sin(theta + time * 0.1);
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            
            // Center glow
            const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 30);
            gradient.addColorStop(0, 'rgba(255, 215, 0, 0.5)');
            gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(cx, cy, 30, 0, TAU);
            ctx.fill();
            
            animationFrame = requestAnimationFrame(animate);
        }
    </script>
</body>
</html>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 78)
    print("🌀 LIFE FRACTAL INTELLIGENCE - ULTIMATE ENHANCED EDITION v7.0")
    print("=" * 78)
    print(f"✨ Golden Ratio (φ):       {PHI:.15f}")
    print(f"🌻 Golden Angle:           {GOLDEN_ANGLE:.10f}°")
    print(f"📐 Fibonacci:              {FIBONACCI[:10]}...")
    print(f"🦋 Lorenz Params:          σ={LORENZ_SIGMA}, ρ={LORENZ_RHO}, β={LORENZ_BETA:.4f}")
    print(f"🌀 Rossler Params:         a={ROSSLER_A}, b={ROSSLER_B}, c={ROSSLER_C}")
    print(f"🖥️  GPU Available:          {GPU_AVAILABLE} ({GPU_NAME or 'CPU Only'})")
    print("=" * 78)
    print("\n🎯 MATHEMATICAL ENHANCEMENTS:")
    print("  • Lorenz Attractor - Goal visualization (butterfly effect)")
    print("  • Rossler Attractor - Mood pattern prediction")
    print("  • Coupled Chaos - Life-emotion interconnection")
    print("  • Particle Swarm - Spoon Theory energy tracking")
    print("  • Harmonic Resonance - Synesthetic color/sound mapping")
    print("  • Fractal Dimension - Life complexity scoring")
    print("  • Sacred Geometry - Flower of Life, Metatron's Cube")
    print("=" * 78)
    print("\n📡 API Endpoints:")
    print("  POST /api/auth/register, /api/auth/login")
    print("  GET  /api/user/<id>/dashboard")
    print("  GET  /api/user/<id>/visualization")
    print("  GET  /api/user/<id>/visualization/config")
    print("  GET  /api/user/<id>/fractal")
    print("  GET  /api/sacred-math")
    print("=" * 78)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print_banner()
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
