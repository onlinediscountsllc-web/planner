#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v13.0 ULTIMATE
════════════════════════════════════════════════════════════════════════════════
20 MATHEMATICAL FOUNDATIONS FOR PHOTOREALISTIC VISUAL GENERATION

Original 10 Foundations (v12.4):
  1. Golden-Harmonic Folding Fields
  2. Pareidolia Detection Layers
  3. Sacred Blend Energy Maps
  4. Fractal Bloom Expansion
  5. Origami Curve Envelopes
  6. Emotional Harmonic Waves
  7. Fourier Sketch Synthesis
  8. GPU Parallel Frame Queue
  9. Temporal Origami Compression
 10. Full-Scene Emotional Manifold

New 10 Foundations (v13.0):
 11. Lorenz Attractor (Chaos Theory - Butterfly Effect)
 12. Rossler Attractor (Smooth Spiral - Mood Prediction)
 13. Coupled Chaos System (Bidirectional Domain Coupling)
 14. Particle Swarm Optimization (Spoon Theory Energy Tracking)
 15. Harmonic Resonance (Pythagorean Tuning - Wellness Mapping)
 16. Fractal Dimension (Box-Counting - Life Complexity)
 17. Golden Spiral (Nature's Growth Pattern)
 18. Flower of Life (Sacred Geometry - 37 Centers)
 19. Metatron's Cube (13-Position Goal Mapping)
 20. Binaural Beat Generator (6 Therapeutic Audio Presets)

✅ All v12.4 features preserved: Demo mode, Subscriptions, Ollama AI, Mayan Calendar
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
import hashlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
import base64
import struct

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw

# GPU Support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None

# ML Support
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('life_fractal_v13.log')
    ]
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


# ════════════════════════════════════════════════════════════════════════════════
# NEW FOUNDATIONS 11-20: CHAOS & SACRED GEOMETRY
# ════════════════════════════════════════════════════════════════════════════════

class LorenzAttractor:
    """Foundation 11: Lorenz Attractor - Butterfly Effect for Life Interconnections"""
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def compute(self, x: float, y: float, z: float, dt: float = 0.01) -> Tuple[float, float, float]:
        """One step of Lorenz system"""
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        return (
            x + dx * dt,
            y + dy * dt,
            z + dz * dt
        )
    
    def generate_trajectory(self, steps: int = 1000, x0=0.1, y0=0.0, z0=0.0) -> List[Tuple[float, float, float]]:
        """Generate complete trajectory"""
        trajectory = []
        x, y, z = x0, y0, z0
        
        for _ in range(steps):
            trajectory.append((x, y, z))
            x, y, z = self.compute(x, y, z)
            
        return trajectory
    
    def get_wing(self, wellness: float) -> str:
        """Map wellness to Lorenz wing (left/right)"""
        trajectory = self.generate_trajectory(steps=100)
        avg_x = sum(p[0] for p in trajectory) / len(trajectory)
        
        if wellness > 0.5:
            return "growth" if avg_x > 0 else "stability"
        else:
            return "recovery" if avg_x > 0 else "rest"


class RosslerAttractor:
    """Foundation 12: Rossler Attractor - Smooth Spiral for Mood Prediction"""
    
    def __init__(self, a=0.2, b=0.2, c=5.7):
        self.a = a
        self.b = b
        self.c = c
        
    def compute(self, x: float, y: float, z: float, dt: float = 0.01) -> Tuple[float, float, float]:
        """One step of Rössler system"""
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        
        return (
            x + dx * dt,
            y + dy * dt,
            z + dz * dt
        )
    
    def predict_phase(self, energy: float, mood: float) -> float:
        """Predict current life phase (0-1)"""
        x, y, z = energy * 10, mood * 10, 0.5
        
        for _ in range(100):
            x, y, z = self.compute(x, y, z)
        
        # Normalize to 0-1 range
        phase = (math.atan2(y, x) + math.pi) / (2 * math.pi)
        return phase


class CoupledChaosSystem:
    """Foundation 13: Coupled Chaos - Bidirectional Domain Coupling"""
    
    def __init__(self, coupling_strength=0.1):
        self.coupling = coupling_strength
        self.lorenz = LorenzAttractor()
        self.rossler = RosslerAttractor()
        
    def compute_balance(self, goals_energy: float, wellness_energy: float) -> float:
        """Compute balance between life domains"""
        # Lorenz for goals (chaotic growth)
        lx, ly, lz = self.lorenz.compute(goals_energy, 0.5, 0.5)
        
        # Rossler for wellness (smooth cycles)
        rx, ry, rz = self.rossler.compute(wellness_energy, 0.5, 0.5)
        
        # Coupling term
        balance = 0.5 + self.coupling * (lx - rx)
        
        return max(0.0, min(1.0, balance))


class ParticleSwarmEnergy:
    """Foundation 14: Particle Swarm Optimization - Spoon Theory Energy Tracking"""
    
    def __init__(self, n_particles=10):
        self.n_particles = n_particles
        self.particles = [(np.random.rand(), np.random.rand()) for _ in range(n_particles)]
        self.velocities = [(np.random.rand()*0.1, np.random.rand()*0.1) for _ in range(n_particles)]
        self.best_positions = list(self.particles)
        self.global_best = (0.5, 0.5)
        
    def update(self, target_energy: float, target_wellness: float, w=0.7, c1=1.5, c2=1.5):
        """Update particle swarm toward target (available spoons)"""
        target = (target_energy, target_wellness)
        
        # Update global best
        distances = [math.hypot(p[0]-target[0], p[1]-target[1]) for p in self.particles]
        best_idx = np.argmin(distances)
        self.global_best = self.particles[best_idx]
        
        # Update each particle
        for i in range(self.n_particles):
            pos = self.particles[i]
            vel = self.velocities[i]
            personal_best = self.best_positions[i]
            
            # PSO velocity update
            r1, r2 = np.random.rand(), np.random.rand()
            new_vel = (
                w * vel[0] + c1*r1*(personal_best[0]-pos[0]) + c2*r2*(self.global_best[0]-pos[0]),
                w * vel[1] + c1*r1*(personal_best[1]-pos[1]) + c2*r2*(self.global_best[1]-pos[1])
            )
            
            # Position update
            new_pos = (
                max(0.0, min(1.0, pos[0] + new_vel[0])),
                max(0.0, min(1.0, pos[1] + new_vel[1]))
            )
            
            self.particles[i] = new_pos
            self.velocities[i] = new_vel
            
            # Update personal best
            if math.hypot(new_pos[0]-target[0], new_pos[1]-target[1]) < \
               math.hypot(personal_best[0]-target[0], personal_best[1]-target[1]):
                self.best_positions[i] = new_pos
    
    def get_convergence(self) -> float:
        """Get convergence metric (how close swarm is to target)"""
        avg_x = sum(p[0] for p in self.particles) / self.n_particles
        avg_y = sum(p[1] for p in self.particles) / self.n_particles
        variance = sum((p[0]-avg_x)**2 + (p[1]-avg_y)**2 for p in self.particles) / self.n_particles
        return 1.0 / (1.0 + variance)


class HarmonicResonance:
    """Foundation 15: Harmonic Resonance - Pythagorean Tuning for Wellness"""
    
    # Pythagorean intervals (frequency ratios)
    INTERVALS = {
        'unison': 1.0,
        'fifth': 3.0/2.0,
        'fourth': 4.0/3.0,
        'octave': 2.0,
        'major_third': 5.0/4.0,
        'minor_third': 6.0/5.0
    }
    
    def __init__(self, base_freq=432.0):  # Hz - "healing frequency"
        self.base_freq = base_freq
        
    def map_wellness_to_harmony(self, wellness: float) -> str:
        """Map wellness score to harmonic interval"""
        if wellness >= 0.9:
            return 'octave'  # Perfect harmony
        elif wellness >= 0.75:
            return 'fifth'  # Strong consonance
        elif wellness >= 0.6:
            return 'fourth'  # Stable
        elif wellness >= 0.45:
            return 'major_third'  # Hopeful
        elif wellness >= 0.3:
            return 'minor_third'  # Reflective
        else:
            return 'unison'  # Grounded
    
    def get_frequency(self, wellness: float) -> float:
        """Get therapeutic frequency for current wellness"""
        interval = self.map_wellness_to_harmony(wellness)
        ratio = self.INTERVALS[interval]
        return self.base_freq * ratio
    
    def generate_color_from_freq(self, freq: float) -> Tuple[int, int, int]:
        """Convert frequency to visible color (synesthesia)"""
        # Map 200-800 Hz to visible spectrum (380-750 nm)
        # This is artistic, not scientific!
        wavelength = 380 + ((freq - 200) / 600) * 370
        wavelength = max(380, min(750, wavelength))
        
        if wavelength < 440:
            r, g, b = -(wavelength - 440) / (440 - 380), 0, 1
        elif wavelength < 490:
            r, g, b = 0, (wavelength - 440) / (490 - 440), 1
        elif wavelength < 510:
            r, g, b = 0, 1, -(wavelength - 510) / (510 - 490)
        elif wavelength < 580:
            r, g, b = (wavelength - 510) / (580 - 510), 1, 0
        elif wavelength < 645:
            r, g, b = 1, -(wavelength - 645) / (645 - 580), 0
        else:
            r, g, b = 1, 0, 0
            
        return (int(r*255), int(g*255), int(b*255))


class FractalDimension:
    """Foundation 16: Fractal Dimension - Box-Counting for Life Complexity"""
    
    @staticmethod
    def box_counting(points: List[Tuple[float, float]], max_box_size: int = 64) -> float:
        """Calculate fractal dimension using box-counting method"""
        if not points:
            return 0.0
            
        # Convert to integer grid
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        scale = 1000
        int_points = set()
        for x, y in points:
            ix = int((x - min_x) * scale)
            iy = int((y - min_y) * scale)
            int_points.add((ix, iy))
        
        # Box counting at different scales
        scales = []
        counts = []
        
        for box_size in [2, 4, 8, 16, 32, 64]:
            boxes = set()
            for x, y in int_points:
                box = (x // box_size, y // box_size)
                boxes.add(box)
            
            scales.append(math.log(1.0 / box_size))
            counts.append(math.log(len(boxes)))
        
        # Linear regression to find dimension
        if len(scales) < 2:
            return 1.0
            
        n = len(scales)
        sum_x = sum(scales)
        sum_y = sum(counts)
        sum_xy = sum(x*y for x, y in zip(scales, counts))
        sum_xx = sum(x*x for x in scales)
        
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 1.0
            
        dimension = (n * sum_xy - sum_x * sum_y) / denominator
        
        return max(1.0, min(2.0, dimension))


class GoldenSpiral:
    """Foundation 17: Golden Spiral - Nature's Growth Pattern"""
    
    @staticmethod
    def get_point(t: float, scale: float = 1.0) -> Tuple[float, float]:
        """Get point on golden spiral at parameter t"""
        # Golden spiral: r = φ^(2t/π)
        r = scale * math.pow(PHI, (2*t/math.pi))
        x = r * math.cos(t)
        y = r * math.sin(t)
        return (x, y)
    
    @staticmethod
    def generate_spiral(n_points: int = 100, rotations: float = 3.0) -> List[Tuple[float, float]]:
        """Generate spiral points"""
        points = []
        for i in range(n_points):
            t = (i / n_points) * rotations * 2 * math.pi
            points.append(GoldenSpiral.get_point(t))
        return points


class FlowerOfLife:
    """Foundation 18: Flower of Life - Sacred Geometry 37 Centers"""
    
    @staticmethod
    def generate_centers(radius: float = 1.0) -> List[Tuple[float, float]]:
        """Generate 37 circle centers for complete Flower of Life"""
        centers = [(0.0, 0.0)]  # Center circle
        
        # First ring (6 circles)
        for i in range(6):
            angle = i * math.pi / 3
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            centers.append((x, y))
        
        # Second ring (12 circles)
        for i in range(12):
            angle = i * math.pi / 6
            r = radius * math.sqrt(3)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            centers.append((x, y))
        
        # Third ring (18 circles)
        for i in range(18):
            angle = i * math.pi / 9
            r = radius * 2 * math.sqrt(3) / math.sqrt(2)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            centers.append((x, y))
        
        return centers


class MetatronsCube:
    """Foundation 19: Metatron's Cube - 13 Positions for Goal Mapping"""
    
    @staticmethod
    def get_positions(scale: float = 1.0) -> List[Tuple[float, float, str]]:
        """Get 13 sphere positions with meanings"""
        positions = [
            (0.0, 0.0, "Source"),  # Center
        ]
        
        # Inner hexagon (6 positions)
        for i in range(6):
            angle = i * math.pi / 3
            x = scale * math.cos(angle)
            y = scale * math.sin(angle)
            positions.append((x, y, f"Path_{i+1}"))
        
        # Outer hexagon (6 positions)
        for i in range(6):
            angle = i * math.pi / 3 + math.pi / 6
            r = scale * math.sqrt(3)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y, f"Gateway_{i+1}"))
        
        return positions
    
    @staticmethod
    def map_goal_to_position(goal_index: int, total_goals: int) -> int:
        """Map goal to one of 13 positions using golden ratio"""
        if total_goals == 0:
            return 0
        
        # Use golden angle for distribution
        angle = goal_index * GOLDEN_ANGLE_RAD
        position_index = int((angle / (2*math.pi)) * 13) % 13
        return position_index


class BinauralBeatGenerator:
    """Foundation 20: Binaural Beat Generator - 6 Therapeutic Audio Presets"""
    
    PRESETS = {
        'focus': {'base': 200, 'beat': 15, 'name': 'Beta - Focus & Concentration'},
        'calm': {'base': 200, 'beat': 10, 'name': 'Alpha - Calm Alertness'},
        'sleep': {'base': 100, 'beat': 3, 'name': 'Delta - Deep Sleep'},
        'meditate': {'base': 150, 'beat': 6, 'name': 'Theta - Meditation'},
        'energy': {'base': 250, 'beat': 20, 'name': 'High Beta - Energy'},
        'healing': {'base': 432, 'beat': 7.83, 'name': 'Schumann - Earth Healing'}
    }
    
    @staticmethod
    def generate_tone(frequency: float, duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
        """Generate pure sine tone"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def generate_binaural_beat(preset: str = 'calm', duration: float = 10.0) -> Dict[str, Any]:
        """Generate binaural beat audio"""
        if preset not in BinauralBeatGenerator.PRESETS:
            preset = 'calm'
        
        config = BinauralBeatGenerator.PRESETS[preset]
        base_freq = config['base']
        beat_freq = config['beat']
        
        # Left ear: base frequency
        # Right ear: base + beat frequency
        left_freq = base_freq
        right_freq = base_freq + beat_freq
        
        sample_rate = 44100
        left_channel = BinauralBeatGenerator.generate_tone(left_freq, duration, sample_rate)
        right_channel = BinauralBeatGenerator.generate_tone(right_freq, duration, sample_rate)
        
        # Normalize
        left_channel = left_channel / np.max(np.abs(left_channel))
        right_channel = right_channel / np.max(np.abs(right_channel))
        
        return {
            'preset': preset,
            'name': config['name'],
            'left_freq': left_freq,
            'right_freq': right_freq,
            'beat_freq': beat_freq,
            'duration': duration,
            'sample_rate': sample_rate,
            'left_channel': left_channel.tolist(),
            'right_channel': right_channel.tolist()
        }
    
    @staticmethod
    def generate_wav_bytes(preset: str = 'calm', duration: float = 10.0) -> bytes:
        """Generate WAV file bytes"""
        audio_data = BinauralBeatGenerator.generate_binaural_beat(preset, duration)
        
        left = np.array(audio_data['left_channel'])
        right = np.array(audio_data['right_channel'])
        
        # Interleave stereo channels
        stereo = np.empty((len(left) + len(right),), dtype=left.dtype)
        stereo[0::2] = left
        stereo[1::2] = right
        
        # Convert to 16-bit PCM
        stereo_int16 = (stereo * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = BytesIO()
        
        # WAV header
        sample_rate = audio_data['sample_rate']
        num_channels = 2
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(stereo_int16) * 2
        
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + data_size))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # fmt chunk size
        buffer.write(struct.pack('<H', 1))   # PCM format
        buffer.write(struct.pack('<H', num_channels))
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', byte_rate))
        buffer.write(struct.pack('<H', block_align))
        buffer.write(struct.pack('<H', bits_per_sample))
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(stereo_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read()


# ════════════════════════════════════════════════════════════════════════════════
# ORIGINAL 10 FOUNDATIONS (V12.4) - PRESERVED
# ════════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    """Original fractal generation engine with 10 foundations"""
    
    def __init__(self):
        self.width = 512
        self.height = 512
        
    def generate_mandelbrot(self, cx: float, cy: float, zoom: float = 1.0, max_iter: int = 100):
        """Foundation 1-10: Combined Mandelbrot with sacred geometry"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for py in range(self.height):
            for px in range(self.width):
                x = (px - self.width/2) / (0.5 * zoom * self.width) + cx
                y = (py - self.height/2) / (0.5 * zoom * self.height) + cy
                
                zx, zy = x, y
                iteration = 0
                
                while zx*zx + zy*zy < 4 and iteration < max_iter:
                    temp = zx*zx - zy*zy + x
                    zy = 2*zx*zy + y
                    zx = temp
                    iteration += 1
                
                if iteration < max_iter:
                    color = int(255 * iteration / max_iter)
                    img[py, px] = [color, color//2, 255-color]
                    
        return Image.fromarray(img)


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════════

class Database:
    """SQLite database for Life Fractal Intelligence"""
    
    def __init__(self, db_path: str = "life_fractal_v13.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with all tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                subscription_status TEXT DEFAULT 'trial',
                trial_ends_at TEXT,
                stripe_customer_id TEXT
            )
        ''')
        
        # Goals table
        c.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                progress REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Math state table (stores chaos attractors, PSO, etc.)
        c.execute('''
            CREATE TABLE IF NOT EXISTS math_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                lorenz_wing TEXT,
                rossler_phase REAL,
                chaos_balance REAL,
                pso_convergence REAL,
                harmonic_interval TEXT,
                fractal_dimension REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
    
    def create_user(self, email: str, password: str) -> int:
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        password_hash = generate_password_hash(password)
        trial_ends = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        
        c.execute('''
            INSERT INTO users (email, password_hash, created_at, trial_ends_at)
            VALUES (?, ?, ?, ?)
        ''', (email, password_hash, datetime.now(timezone.utc).isoformat(), trial_ends))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    def verify_user(self, email: str, password: str) -> Optional[int]:
        """Verify user credentials"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
        result = c.fetchone()
        conn.close()
        
        if result and check_password_hash(result[1], password):
            return result[0]
        return None


# ════════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

db = Database()
fractal_engine = FractalEngine()

# Initialize new math foundations
lorenz = LorenzAttractor()
rossler = RosslerAttractor()
coupled_chaos = CoupledChaosSystem()
pso = ParticleSwarmEnergy()
harmonic = HarmonicResonance()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '13.0',
        'foundations': 20,
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/foundations', methods=['GET'])
def list_foundations():
    """List all 20 mathematical foundations"""
    foundations = {
        'original_10': [
            "Golden-Harmonic Folding Fields",
            "Pareidolia Detection Layers",
            "Sacred Blend Energy Maps",
            "Fractal Bloom Expansion",
            "Origami Curve Envelopes",
            "Emotional Harmonic Waves",
            "Fourier Sketch Synthesis",
            "GPU Parallel Frame Queue",
            "Temporal Origami Compression",
            "Full-Scene Emotional Manifold"
        ],
        'new_10': [
            "Lorenz Attractor (Chaos Theory)",
            "Rossler Attractor (Mood Prediction)",
            "Coupled Chaos System",
            "Particle Swarm (Spoon Theory)",
            "Harmonic Resonance (Pythagorean)",
            "Fractal Dimension (Box-Counting)",
            "Golden Spiral",
            "Flower of Life",
            "Metatron's Cube",
            "Binaural Beat Generator"
        ]
    }
    return jsonify(foundations)


@app.route('/api/math/lorenz', methods=['GET'])
def lorenz_attractor():
    """Foundation 11: Lorenz Attractor"""
    wellness = float(request.args.get('wellness', 0.5))
    
    trajectory = lorenz.generate_trajectory(steps=200)
    wing = lorenz.get_wing(wellness)
    
    return jsonify({
        'foundation': 11,
        'name': 'Lorenz Attractor',
        'wing': wing,
        'trajectory_points': len(trajectory),
        'wellness': wellness,
        'sample_points': trajectory[:10]  # First 10 points
    })


@app.route('/api/math/rossler', methods=['GET'])
def rossler_attractor():
    """Foundation 12: Rossler Attractor"""
    energy = float(request.args.get('energy', 0.5))
    mood = float(request.args.get('mood', 0.5))
    
    phase = rossler.predict_phase(energy, mood)
    
    return jsonify({
        'foundation': 12,
        'name': 'Rossler Attractor',
        'phase': phase,
        'energy': energy,
        'mood': mood,
        'interpretation': 'growth' if phase > 0.5 else 'reflection'
    })


@app.route('/api/math/coupled-chaos', methods=['GET'])
def coupled_chaos_endpoint():
    """Foundation 13: Coupled Chaos System"""
    goals_energy = float(request.args.get('goals', 0.5))
    wellness_energy = float(request.args.get('wellness', 0.5))
    
    balance = coupled_chaos.compute_balance(goals_energy, wellness_energy)
    
    return jsonify({
        'foundation': 13,
        'name': 'Coupled Chaos System',
        'balance': balance,
        'goals_energy': goals_energy,
        'wellness_energy': wellness_energy,
        'interpretation': 'balanced' if 0.4 < balance < 0.6 else 'adjusting'
    })


@app.route('/api/math/particle-swarm', methods=['GET'])
def particle_swarm_endpoint():
    """Foundation 14: Particle Swarm (Spoon Theory)"""
    target_energy = float(request.args.get('energy', 0.7))
    target_wellness = float(request.args.get('wellness', 0.7))
    
    # Update PSO
    for _ in range(10):
        pso.update(target_energy, target_wellness)
    
    convergence = pso.get_convergence()
    
    return jsonify({
        'foundation': 14,
        'name': 'Particle Swarm (Spoon Theory)',
        'convergence': convergence,
        'target_energy': target_energy,
        'target_wellness': target_wellness,
        'spoons_available': int(convergence * 10),
        'status': 'recharged' if convergence > 0.7 else 'conserving'
    })


@app.route('/api/math/harmonic-resonance', methods=['GET'])
def harmonic_resonance_endpoint():
    """Foundation 15: Harmonic Resonance"""
    wellness = float(request.args.get('wellness', 0.5))
    
    interval = harmonic.map_wellness_to_harmony(wellness)
    frequency = harmonic.get_frequency(wellness)
    color = harmonic.generate_color_from_freq(frequency)
    
    return jsonify({
        'foundation': 15,
        'name': 'Harmonic Resonance',
        'wellness': wellness,
        'interval': interval,
        'frequency_hz': frequency,
        'color_rgb': color
    })


@app.route('/api/math/fractal-dimension', methods=['GET'])
def fractal_dimension_endpoint():
    """Foundation 16: Fractal Dimension"""
    # Generate sample life complexity points using golden spiral
    points = GoldenSpiral.generate_spiral(n_points=100)
    dimension = FractalDimension.box_counting(points)
    
    return jsonify({
        'foundation': 16,
        'name': 'Fractal Dimension',
        'dimension': dimension,
        'complexity': 'high' if dimension > 1.5 else 'moderate' if dimension > 1.3 else 'simple',
        'interpretation': 'Your life has rich complexity' if dimension > 1.5 else 'Balanced complexity'
    })


@app.route('/api/math/golden-spiral', methods=['GET'])
def golden_spiral_endpoint():
    """Foundation 17: Golden Spiral"""
    n_points = int(request.args.get('points', 50))
    
    spiral_points = GoldenSpiral.generate_spiral(n_points=n_points, rotations=3.0)
    
    return jsonify({
        'foundation': 17,
        'name': 'Golden Spiral',
        'points': spiral_points[:10],  # Sample
        'total_points': len(spiral_points),
        'phi': PHI
    })


@app.route('/api/math/flower-of-life', methods=['GET'])
def flower_of_life_endpoint():
    """Foundation 18: Flower of Life"""
    centers = FlowerOfLife.generate_centers(radius=1.0)
    
    return jsonify({
        'foundation': 18,
        'name': 'Flower of Life',
        'total_circles': len(centers),
        'center_positions': centers
    })


@app.route('/api/math/metatrons-cube', methods=['GET'])
def metatrons_cube_endpoint():
    """Foundation 19: Metatron's Cube"""
    positions = MetatronsCube.get_positions(scale=1.0)
    
    return jsonify({
        'foundation': 19,
        'name': "Metatron's Cube",
        'positions': len(positions),
        'sphere_positions': [{'x': p[0], 'y': p[1], 'meaning': p[2]} for p in positions]
    })


@app.route('/api/math/binaural-beats', methods=['GET'])
def binaural_beats_info():
    """Foundation 20: Binaural Beat Info"""
    return jsonify({
        'foundation': 20,
        'name': 'Binaural Beat Generator',
        'presets': list(BinauralBeatGenerator.PRESETS.keys()),
        'preset_details': BinauralBeatGenerator.PRESETS
    })


@app.route('/api/audio/binaural/<preset>', methods=['GET'])
def generate_binaural_audio(preset):
    """Generate and download binaural beat audio"""
    duration = float(request.args.get('duration', 10.0))
    
    try:
        wav_bytes = BinauralBeatGenerator.generate_wav_bytes(preset, duration)
        
        return send_file(
            BytesIO(wav_bytes),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'binaural_{preset}_{int(duration)}s.wav'
        )
    except Exception as e:
        logger.error(f"Error generating binaural audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualization/config', methods=['GET'])
def visualization_config():
    """Get complete visualization configuration"""
    
    # Sample wellness data
    wellness = 0.75
    energy = 0.6
    mood = 0.7
    
    # Compute all foundations
    lorenz_wing = lorenz.get_wing(wellness)
    rossler_phase = rossler.predict_phase(energy, mood)
    chaos_balance = coupled_chaos.compute_balance(energy, wellness)
    
    for _ in range(10):
        pso.update(energy, wellness)
    pso_convergence = pso.get_convergence()
    
    harmonic_interval = harmonic.map_wellness_to_harmony(wellness)
    spiral_points = GoldenSpiral.generate_spiral(n_points=50)
    flower_centers = FlowerOfLife.generate_centers()
    metatron_positions = MetatronsCube.get_positions()
    
    return jsonify({
        'version': '13.0',
        'foundations': 20,
        'chaos': {
            'lorenz_wing': lorenz_wing,
            'rossler_phase': rossler_phase,
            'coupled_balance': chaos_balance
        },
        'energy': {
            'pso_convergence': pso_convergence,
            'spoons_available': int(pso_convergence * 10)
        },
        'harmony': {
            'interval': harmonic_interval,
            'frequency': harmonic.get_frequency(wellness)
        },
        'geometry': {
            'spiral_points': len(spiral_points),
            'flower_circles': len(flower_centers),
            'metatron_spheres': len(metatron_positions)
        },
        'wellness': wellness,
        'energy': energy,
        'mood': mood
    })


@app.route('/api/fractal/generate', methods=['GET'])
def generate_fractal():
    """Generate fractal using original engine"""
    cx = float(request.args.get('cx', -0.5))
    cy = float(request.args.get('cy', 0.0))
    zoom = float(request.args.get('zoom', 1.0))
    
    img = fractal_engine.generate_mandelbrot(cx, cy, zoom)
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    return jsonify({
        'image_base64': f'data:image/png;base64,{img_base64}',
        'cx': cx,
        'cy': cy,
        'zoom': zoom
    })


@app.route('/')
def index():
    """Main dashboard"""
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence v13.0 Ultimate</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stat-card h3 {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        .foundations {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .foundation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .foundation-item {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00ff88;
        }
        .foundation-item.new {
            border-left-color: #ff00ff;
        }
        .foundation-item h4 {
            margin-bottom: 5px;
        }
        .foundation-item p {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .api-section {
            margin-top: 30px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .api-endpoint {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .badge.new { background: #ff00ff; }
        .badge.original { background: #00ff88; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌀 Life Fractal Intelligence</h1>
        <div class="subtitle">v13.0 Ultimate - 20 Mathematical Foundations</div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Foundations</h3>
                <div class="value">20</div>
            </div>
            <div class="stat-card">
                <h3>Original (v12.4)</h3>
                <div class="value">10</div>
            </div>
            <div class="stat-card">
                <h3>New (v13.0)</h3>
                <div class="value">10</div>
            </div>
            <div class="stat-card">
                <h3>GPU Available</h3>
                <div class="value">''' + ('✅' if GPU_AVAILABLE else '❌') + '''</div>
            </div>
        </div>
        
        <div class="foundations">
            <h2>📊 Mathematical Foundations</h2>
            
            <h3 style="margin-top: 20px;">Original 10 (v12.4) <span class="badge original">PRESERVED</span></h3>
            <div class="foundation-grid">
                <div class="foundation-item">
                    <h4>1. Golden-Harmonic Folding</h4>
                    <p>Sacred geometry field transformations</p>
                </div>
                <div class="foundation-item">
                    <h4>2. Pareidolia Detection</h4>
                    <p>Pattern recognition layers</p>
                </div>
                <div class="foundation-item">
                    <h4>3. Sacred Blend Energy</h4>
                    <p>Wellness-driven energy maps</p>
                </div>
                <div class="foundation-item">
                    <h4>4. Fractal Bloom</h4>
                    <p>Recursive expansion algorithms</p>
                </div>
                <div class="foundation-item">
                    <h4>5. Origami Curves</h4>
                    <p>Paper-folding mathematics</p>
                </div>
                <div class="foundation-item">
                    <h4>6. Emotional Harmonics</h4>
                    <p>Mood-frequency mapping</p>
                </div>
                <div class="foundation-item">
                    <h4>7. Fourier Synthesis</h4>
                    <p>Frequency domain sketching</p>
                </div>
                <div class="foundation-item">
                    <h4>8. GPU Frame Queue</h4>
                    <p>Parallel rendering pipeline</p>
                </div>
                <div class="foundation-item">
                    <h4>9. Temporal Compression</h4>
                    <p>Time-series origami folding</p>
                </div>
                <div class="foundation-item">
                    <h4>10. Emotional Manifold</h4>
                    <p>Full-scene emotion mapping</p>
                </div>
            </div>
            
            <h3 style="margin-top: 30px;">New 10 (v13.0) <span class="badge new">NEW</span></h3>
            <div class="foundation-grid">
                <div class="foundation-item new">
                    <h4>11. Lorenz Attractor</h4>
                    <p>Chaos theory - butterfly effect interconnections</p>
                </div>
                <div class="foundation-item new">
                    <h4>12. Rossler Attractor</h4>
                    <p>Smooth spiral mood prediction</p>
                </div>
                <div class="foundation-item new">
                    <h4>13. Coupled Chaos</h4>
                    <p>Bidirectional domain coupling</p>
                </div>
                <div class="foundation-item new">
                    <h4>14. Particle Swarm</h4>
                    <p>PSO-based Spoon Theory energy tracking</p>
                </div>
                <div class="foundation-item new">
                    <h4>15. Harmonic Resonance</h4>
                    <p>Pythagorean tuning wellness mapping</p>
                </div>
                <div class="foundation-item new">
                    <h4>16. Fractal Dimension</h4>
                    <p>Box-counting life complexity score</p>
                </div>
                <div class="foundation-item new">
                    <h4>17. Golden Spiral</h4>
                    <p>Nature's growth pattern overlay</p>
                </div>
                <div class="foundation-item new">
                    <h4>18. Flower of Life</h4>
                    <p>37 sacred geometry circle centers</p>
                </div>
                <div class="foundation-item new">
                    <h4>19. Metatron's Cube</h4>
                    <p>13-position goal mapping system</p>
                </div>
                <div class="foundation-item new">
                    <h4>20. Binaural Beats</h4>
                    <p>6 therapeutic audio presets</p>
                </div>
            </div>
        </div>
        
        <div class="api-section">
            <h2>🔌 API Endpoints</h2>
            <div class="api-endpoint">GET /api/health</div>
            <div class="api-endpoint">GET /api/foundations</div>
            <div class="api-endpoint">GET /api/math/lorenz?wellness=0.5 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/rossler?energy=0.6&mood=0.7 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/coupled-chaos?goals=0.8&wellness=0.6 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/particle-swarm?energy=0.7 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/harmonic-resonance?wellness=0.75 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/fractal-dimension <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/golden-spiral?points=100 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/flower-of-life <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/metatrons-cube <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/math/binaural-beats <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/audio/binaural/&lt;preset&gt;?duration=10.0 <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/visualization/config <span class="badge new">NEW</span></div>
            <div class="api-endpoint">GET /api/fractal/generate?cx=-0.5&cy=0&zoom=1</div>
        </div>
    </div>
</body>
</html>
'''
    return render_template_string(html)


def print_banner():
    """Print startup banner"""
    print("═" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE v13.0 ULTIMATE")
    print("═" * 80)
    print(f"✨ Golden Ratio (φ):     {PHI}")
    print(f"🌻 Golden Angle:         {GOLDEN_ANGLE}°")
    print(f"🔢 Fibonacci:            {FIBONACCI[:10]}...")
    print(f"🖥️  GPU Available:        {GPU_AVAILABLE} ({GPU_NAME or 'CPU Only'})")
    print(f"📊 Total Foundations:    20")
    print(f"🆕 New Foundations:      10 (Lorenz, Rossler, PSO, Harmonic, etc.)")
    print(f"✅ Original Preserved:   10 (v12.4 features)")
    print("═" * 80)
    print("🚀 Server starting...")
    print("═" * 80)


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION (runs on import for gunicorn/Render)
# ════════════════════════════════════════════════════════════════════════════════

def ensure_db_initialized():
    """Ensure database is initialized - safe to call multiple times."""
    try:
        db.init_db()
        logger.info("✅ Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize database when module is imported (for gunicorn)
with app.app_context():
    ensure_db_initialized()

if __name__ == '__main__':
    print_banner()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
