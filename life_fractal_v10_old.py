#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LIFE FRACTAL INTELLIGENCE v12.1 - MATHEMATICAL ANIMATION ENGINE                                 ║
║  ════════════════════════════════════════════════════════════════════════════════════════════════║
║                                                                                                  ║
║  🔢 TEN MATHEMATICAL FOUNDATIONS FOR PHOTOREALISTIC VISUAL GENERATION                            ║
║  ═══════════════════════════════════════════════════════════════════════════════════════════════ ║
║  1. Golden-Harmonic Folding Field     - F(t,φ) = sin(2π·t·φ)·cos(2π·t/φ)+sin(π·t²)              ║
║  2. Pareidolia Detection Field        - Pattern recognition in noise                             ║
║  3. Sacred Blend Energy Map           - Tone density modulation with tanh                        ║
║  4. Fractal Bloom Expansion           - Z(n+1) = Z(n)² + C recursive structures                  ║
║  5. Centralized Origami Curve         - O(u,v) = sin(u·v)+cos(φ·u)·sin(φ·v)                     ║
║  6. Emotionally Tuned Harmonic Wave   - H(t,e) = |sin(π·t·E[e])| + tanh(t·0.2)                  ║
║  7. Fourier Sketch Synthesis          - Σ(aₙ·cos(n·x) + bₙ·sin(n·y))                            ║
║  8. GPU Parallel Frame Queue          - Vectorized batch rendering                               ║
║  9. Temporal Origami Compression      - Cₜ = Σ Mf·(1/φ)ⁿ fold/unfold                            ║
║  10. Full-Scene Emotional Manifold    - E(x,y,t) = ∇²B + H(t,e)·F(t,φ)                          ║
║                                                                                                  ║
║  🧬 SELF-REPLICATING ORBS WITH ORIGAMI FOLDING                                                   ║
║  🎬 1-20 MINUTE ANIMATION GENERATION                                                             ║
║  🌊 SWARM INTELLIGENCE + CELLULAR AUTOMATA                                                       ║
║                                                                                                  ║
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
import threading
import struct
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

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio: 1.618033988749895
PHI_INVERSE = 1 / PHI          # 0.618033988749895
PHI_SQUARED = PHI * PHI        # 2.618033988749895
GOLDEN_ANGLE = 360 / (PHI ** 2)  # 137.5077640500378°
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
PLANCK_KARMA = 1e-43
DHARMA_FREQUENCY = 432  # Hz
SCHUMANN_RESONANCE = 7.83  # Hz
SOLFEGGIO = [174, 285, 396, 417, 528, 639, 741, 852, 963]

# Mayan Calendar Constants
MAYAN_KIN = 20
MAYAN_TRECENA = 13
MAYAN_TZOLKIN = 260
MAYAN_HAAB = 365

MAYAN_SIGNS = [
    "Imix (Dragon)", "Ik (Wind)", "Akbal (Night)", "Kan (Seed)", "Chicchan (Serpent)",
    "Cimi (Death)", "Manik (Deer)", "Lamat (Star)", "Muluc (Water)", "Oc (Dog)",
    "Chuen (Monkey)", "Eb (Road)", "Ben (Reed)", "Ix (Jaguar)", "Men (Eagle)",
    "Cib (Owl)", "Caban (Earth)", "Etznab (Mirror)", "Cauac (Storm)", "Ahau (Sun)"
]

# Emotion indices for harmonic wave modulation
EMOTION_INDEX = {
    'hope': 1.0, 'joy': 1.2, 'peace': 0.8, 'love': 1.5,
    'sadness': 2.0, 'fear': 2.5, 'anger': 3.0,
    'calm': 0.5, 'excitement': 1.8, 'wonder': 1.3,
    'neutral': 1.0
}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 🔢 TEN MATHEMATICAL FOUNDATIONS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MathematicalFoundations:
    """
    Ten Mathematical Foundations for Photorealistic Visual Generation.
    These drive animation, drawing, and unfolding across the organism.
    """
    
    def __init__(self):
        self.phi = PHI
        self.phi_inv = PHI_INVERSE
        self.cache = {}
        self.frame_queue = []
        self.compressed_states = []
        
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 1. Golden-Harmonic Folding Field
    # F(t,φ) = sin(2π·t·φ)·cos(2π·t/φ)+sin(π·t²)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def golden_harmonic_fold(self, t: float) -> float:
        """
        Unfolds and folds visual energy organically using Golden Ratio.
        Used for trajectory mapping, particle animation curves, and symmetry.
        """
        term1 = math.sin(2 * math.pi * t * self.phi)
        term2 = math.cos(2 * math.pi * t / self.phi)
        term3 = math.sin(math.pi * t * t)
        return term1 * term2 + term3
    
    def golden_harmonic_field(self, x: float, y: float, t: float) -> float:
        """2D field version for spatial applications"""
        fx = self.golden_harmonic_fold(x + t)
        fy = self.golden_harmonic_fold(y + t * self.phi_inv)
        return (fx + fy) / 2
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 2. Pareidolia Detection Field
    # P(x,y,t) = sigmoid(cos(x²+y²+sin(t·π)))·L(x,y)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def pareidolia_field(self, x: float, y: float, t: float, 
                         laplacian: float = 1.0) -> float:
        """
        Simulates pattern recognition in noise, enhancing fractal symmetry.
        Helps simulate emergent human-recognizable features (faces, forms).
        """
        inner = x*x + y*y + math.sin(t * math.pi)
        return self.sigmoid(math.cos(inner)) * laplacian
    
    def detect_patterns(self, data: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Detect pattern hotspots in 2D data"""
        hotspots = []
        h, w = data.shape[:2] if len(data.shape) >= 2 else (1, len(data))
        for y in range(1, h-1):
            for x in range(1, w-1):
                val = self.pareidolia_field(x/w, y/h, time.time() % 10)
                if val > threshold:
                    hotspots.append((x, y))
        return hotspots[:100]  # Limit for performance
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 3. Sacred Blend Energy Map
    # B(x,y,t) = tanh(α·sin(2π·x)+β·cos(2π·y))·γ(t)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def sacred_blend(self, x: float, y: float, t: float,
                     alpha: float = 1.0, beta: float = 1.0,
                     gamma_func: Callable = None) -> float:
        """
        Modulates tone density over frame regions using harmonic overlays.
        Controlled by memory-tuned parameters from adaptive tuner.
        """
        gamma = gamma_func(t) if gamma_func else (0.5 + 0.5 * math.sin(t))
        inner = alpha * math.sin(2 * math.pi * x) + beta * math.cos(2 * math.pi * y)
        return math.tanh(inner) * gamma
    
    def blend_energy_map(self, width: int, height: int, t: float,
                         emotion: str = 'neutral') -> np.ndarray:
        """Generate full energy map for frame"""
        emap = np.zeros((height, width))
        emotion_mod = EMOTION_INDEX.get(emotion, 1.0)
        
        for y in range(height):
            for x in range(width):
                nx, ny = x / width, y / height
                emap[y, x] = self.sacred_blend(nx, ny, t, 
                                               alpha=self.phi * emotion_mod,
                                               beta=self.phi_inv * emotion_mod)
        return emap
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 4. Fractal Bloom Expansion
    # Z(n+1) = Z(n)² + C where |Z(n)| < threshold
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def fractal_bloom(self, x: float, y: float, t: float,
                      max_iter: int = 50, threshold: float = 4.0) -> Tuple[int, float]:
        """
        Used for branching edge effects and recursive spatial structures.
        Enables infinite resolution pseudo-depth while remaining tractable.
        """
        # Dynamic C based on time
        c_real = x + 0.1 * math.sin(t * self.phi)
        c_imag = y + 0.1 * math.cos(t * self.phi_inv)
        
        z_real, z_imag = 0.0, 0.0
        
        for n in range(max_iter):
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = z_real_new
            
            magnitude = z_real * z_real + z_imag * z_imag
            if magnitude > threshold:
                # Smooth iteration count
                smooth = n + 1 - math.log(math.log(max(1, magnitude))) / math.log(2)
                return n, smooth
        
        return max_iter, float(max_iter)
    
    def bloom_expansion_field(self, width: int, height: int, t: float,
                              zoom: float = 1.0, center: Tuple[float, float] = (-0.5, 0)) -> np.ndarray:
        """Generate full bloom field for frame"""
        field = np.zeros((height, width))
        
        for py in range(height):
            for px in range(width):
                x = center[0] + (px - width/2) / (width * zoom / 4)
                y = center[1] + (py - height/2) / (height * zoom / 4)
                _, smooth = self.fractal_bloom(x, y, t)
                field[py, px] = smooth
        
        return field
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 5. Centralized Origami Curve Envelope
    # O(u,v) = sin(u·v) + cos(φ·u)·sin(φ·v)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def origami_curve(self, u: float, v: float) -> float:
        """
        Fold/unfold logic inspired by Japanese origami simulation.
        Simulates "creased" visual compression and expansion per frame.
        """
        return math.sin(u * v) + math.cos(self.phi * u) * math.sin(self.phi * v)
    
    def origami_fold_matrix(self, data: np.ndarray, fold_angle: float) -> np.ndarray:
        """Apply origami fold transformation to data matrix"""
        h, w = data.shape[:2]
        folded = np.zeros_like(data)
        
        for y in range(h):
            for x in range(w):
                # Normalized coordinates
                u = (x / w - 0.5) * 2 * math.pi
                v = (y / h - 0.5) * 2 * math.pi
                
                # Apply origami curve transformation
                fold_factor = self.origami_curve(u * fold_angle, v * fold_angle)
                
                # Map to new position
                new_x = int((x + fold_factor * w * 0.1) % w)
                new_y = int((y + fold_factor * h * 0.1) % h)
                
                if len(data.shape) == 3:
                    folded[new_y, new_x] = data[y, x]
                else:
                    folded[new_y, new_x] = data[y, x]
        
        return folded
    
    def unfold_origami(self, compressed: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Unfold compressed data through inverse origami transformation"""
        result = compressed.copy()
        for i in range(iterations):
            angle = self.phi_inv ** i  # Decreasing fold angles
            result = self.origami_fold_matrix(result, -angle)  # Inverse fold
        return result
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 6. Emotionally Tuned Harmonic Wave
    # H(t,e) = |sin(π·t·E[e])| + tanh(t·0.2)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def emotional_harmonic(self, t: float, emotion: str = 'neutral') -> float:
        """
        Provides adaptable curve control for animation timing and intensity.
        Directly modulates narration tone and line-weight fidelity.
        """
        e_index = EMOTION_INDEX.get(emotion.lower(), 1.0)
        term1 = abs(math.sin(math.pi * t * e_index))
        term2 = math.tanh(t * 0.2)
        return term1 + term2
    
    def emotional_wave_sequence(self, duration: float, fps: int,
                                emotions: List[Tuple[float, str]]) -> List[float]:
        """Generate emotion-modulated wave for animation sequence"""
        total_frames = int(duration * fps)
        waves = []
        
        for frame in range(total_frames):
            t = frame / fps
            
            # Find current emotion based on time
            current_emotion = 'neutral'
            for emotion_time, emotion in emotions:
                if t >= emotion_time:
                    current_emotion = emotion
            
            waves.append(self.emotional_harmonic(t, current_emotion))
        
        return waves
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 7. Fourier Sketch Synthesis
    # Sₖ(x,y) = Σ(aₙ·cos(n·x) + bₙ·sin(n·y)), n ∈ [1,N]
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def fourier_sketch(self, x: float, y: float, coefficients: List[Tuple[float, float]],
                       N: int = 10) -> float:
        """
        Dynamically applies sketch harmonics using emotion as seed data.
        Enables realistic-looking hand-drawn simulations.
        """
        result = 0.0
        for n in range(1, min(N + 1, len(coefficients) + 1)):
            a_n, b_n = coefficients[n-1] if n-1 < len(coefficients) else (1.0/n, 1.0/n)
            result += a_n * math.cos(n * x) + b_n * math.sin(n * y)
        return result
    
    def generate_sketch_coefficients(self, emotion: str, complexity: int = 10) -> List[Tuple[float, float]]:
        """Generate Fourier coefficients based on emotional seed"""
        e_index = EMOTION_INDEX.get(emotion.lower(), 1.0)
        coeffs = []
        
        for n in range(complexity):
            # Use golden ratio and emotion to generate coefficients
            a_n = math.sin(n * self.phi * e_index) / (n + 1)
            b_n = math.cos(n * self.phi_inv * e_index) / (n + 1)
            coeffs.append((a_n, b_n))
        
        return coeffs
    
    def sketch_field(self, width: int, height: int, emotion: str = 'neutral') -> np.ndarray:
        """Generate full sketch field for frame"""
        coeffs = self.generate_sketch_coefficients(emotion)
        field = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                nx = (x / width - 0.5) * 2 * math.pi
                ny = (y / height - 0.5) * 2 * math.pi
                field[y, x] = self.fourier_sketch(nx, ny, coeffs)
        
        return field
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 8. GPU Parallel Frame Queue (Vectorized Batch Processing)
    # Fᵢ = U(Dᵢ, Tᵢ) ∀i ∈ queue where GPU(Fᵢ) = maxₗoad
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def queue_frame(self, frame_data: Dict, target_time: float):
        """Add frame to processing queue"""
        self.frame_queue.append({
            'data': frame_data,
            'target_time': target_time,
            'queued_at': time.time()
        })
    
    def process_frame_batch(self, batch_size: int = 10) -> List[np.ndarray]:
        """Process batch of frames in parallel (simulated for CPU)"""
        batch = self.frame_queue[:batch_size]
        self.frame_queue = self.frame_queue[batch_size:]
        
        results = []
        for frame_job in batch:
            # Unfold and process each frame
            data = frame_job['data']
            t = frame_job['target_time']
            
            # Apply golden harmonic unfolding
            if 'field' in data:
                unfolded = self.unfold_origami(data['field'])
                results.append(unfolded)
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 9. Temporal Origami Compression Engine
    # Cₜ = Σ Mf·(1/φ)ⁿ for frames f = 1 → N
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def compress_temporal(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compress sequence of frames using origami mathematics.
        Enables prediction of upcoming frames from compressed snapshots.
        Greatly reduces recomputation overhead between scenes.
        """
        if not frames:
            return np.zeros((100, 100))
        
        # Initialize with first frame shape
        compressed = np.zeros_like(frames[0], dtype=np.float64)
        
        for n, frame in enumerate(frames):
            weight = self.phi_inv ** n  # Decreasing weight
            compressed += frame.astype(np.float64) * weight
        
        # Normalize
        total_weight = sum(self.phi_inv ** i for i in range(len(frames)))
        compressed /= total_weight
        
        # Store for later unfolding
        self.compressed_states.append({
            'data': compressed,
            'frame_count': len(frames),
            'timestamp': time.time()
        })
        
        return compressed.astype(frames[0].dtype)
    
    def expand_temporal(self, compressed: np.ndarray, target_frames: int) -> List[np.ndarray]:
        """Expand compressed state back to frame sequence"""
        frames = []
        
        for n in range(target_frames):
            # Use origami unfolding to regenerate frames
            t = n / target_frames
            weight = self.phi ** (n * 0.1)  # Increasing expansion
            
            # Apply phase-shifted unfolding
            expanded = self.origami_fold_matrix(compressed, t * math.pi)
            expanded = expanded * weight
            
            # Normalize to valid range
            expanded = np.clip(expanded, 0, 255).astype(np.uint8)
            frames.append(expanded)
        
        return frames
    
    # ─────────────────────────────────────────────────────────────────────────────────────
    # 10. Full-Scene Emotional Manifold Map
    # E_scene(x,y,t) = ∇²B(x,y,t) + H(t,e)·F(t,φ)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def emotional_manifold(self, x: float, y: float, t: float,
                           emotion: str = 'neutral') -> float:
        """
        Couples harmonic energy, unfolding field, and motion bloom.
        Used to simulate mood-per-frame in the orchestrator.
        """
        # Blend energy (B)
        B = self.sacred_blend(x, y, t)
        
        # Laplacian approximation (∇²B) - using neighbors
        dx = 0.01
        B_xp = self.sacred_blend(x + dx, y, t)
        B_xm = self.sacred_blend(x - dx, y, t)
        B_yp = self.sacred_blend(x, y + dx, t)
        B_ym = self.sacred_blend(x, y - dx, t)
        laplacian_B = (B_xp + B_xm + B_yp + B_ym - 4 * B) / (dx * dx)
        
        # Harmonic wave (H)
        H = self.emotional_harmonic(t, emotion)
        
        # Golden fold (F)
        F = self.golden_harmonic_fold(t)
        
        return laplacian_B + H * F
    
    def generate_manifold_frame(self, width: int, height: int, t: float,
                                emotion: str = 'neutral') -> np.ndarray:
        """Generate complete emotional manifold frame"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                nx, ny = x / width, y / height
                value = self.emotional_manifold(nx, ny, t, emotion)
                
                # Map to color using golden ratio
                hue = (value * self.phi * 60 + 200) % 360
                sat = 0.7 + 0.3 * abs(math.sin(value * math.pi))
                val = 0.5 + 0.5 * math.tanh(value)
                
                # HSV to RGB
                r, g, b = self._hsv_to_rgb(hue, sat, val)
                frame[y, x] = [int(r*255), int(g*255), int(b*255)]
        
        return frame
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ANIMATION GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class AnimationEngine:
    """
    Generates 1-20 minute animations from mathematical foundations.
    Uses all ten mathematical principles for photorealistic output.
    """
    
    def __init__(self, math_engine: MathematicalFoundations):
        self.math = math_engine
        self.fps = 30
        self.resolution = (1280, 720)  # HD
        self.current_emotion = 'neutral'
        self.keyframes = []
        
    def set_resolution(self, width: int, height: int):
        """Set output resolution"""
        self.resolution = (width, height)
    
    def add_keyframe(self, time_seconds: float, emotion: str, 
                     zoom: float = 1.0, rotation: float = 0.0,
                     bloom_center: Tuple[float, float] = (-0.5, 0)):
        """Add animation keyframe"""
        self.keyframes.append({
            'time': time_seconds,
            'emotion': emotion,
            'zoom': zoom,
            'rotation': rotation,
            'bloom_center': bloom_center
        })
        self.keyframes.sort(key=lambda k: k['time'])
    
    def interpolate_keyframes(self, t: float) -> Dict:
        """Interpolate between keyframes at time t"""
        if not self.keyframes:
            return {
                'emotion': 'neutral', 'zoom': 1.0,
                'rotation': 0.0, 'bloom_center': (-0.5, 0)
            }
        
        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]
        
        for i, kf in enumerate(self.keyframes):
            if kf['time'] > t:
                next_kf = kf
                prev_kf = self.keyframes[max(0, i-1)]
                break
        
        # Calculate interpolation factor
        if next_kf['time'] == prev_kf['time']:
            factor = 0
        else:
            factor = (t - prev_kf['time']) / (next_kf['time'] - prev_kf['time'])
        
        # Smooth interpolation using golden ratio easing
        factor = self._golden_ease(factor)
        
        return {
            'emotion': prev_kf['emotion'] if factor < 0.5 else next_kf['emotion'],
            'zoom': prev_kf['zoom'] + (next_kf['zoom'] - prev_kf['zoom']) * factor,
            'rotation': prev_kf['rotation'] + (next_kf['rotation'] - prev_kf['rotation']) * factor,
            'bloom_center': (
                prev_kf['bloom_center'][0] + (next_kf['bloom_center'][0] - prev_kf['bloom_center'][0]) * factor,
                prev_kf['bloom_center'][1] + (next_kf['bloom_center'][1] - prev_kf['bloom_center'][1]) * factor
            )
        }
    
    def _golden_ease(self, t: float) -> float:
        """Golden ratio based easing function"""
        return t * t * (3 - 2 * t) * PHI_INVERSE + t * (1 - PHI_INVERSE)
    
    def generate_frame(self, frame_number: int, total_frames: int) -> np.ndarray:
        """Generate single animation frame"""
        t = frame_number / self.fps
        params = self.interpolate_keyframes(t)
        
        width, height = self.resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Layer 1: Fractal bloom background
        bloom = self.math.bloom_expansion_field(
            width, height, t,
            zoom=params['zoom'],
            center=params['bloom_center']
        )
        
        # Layer 2: Emotional manifold overlay
        manifold_small = self.math.generate_manifold_frame(
            width // 4, height // 4, t, params['emotion']
        )
        # Upscale manifold
        manifold = np.repeat(np.repeat(manifold_small, 4, axis=0), 4, axis=1)
        
        # Layer 3: Sketch harmonics
        sketch = self.math.sketch_field(width, height, params['emotion'])
        
        # Layer 4: Golden harmonic modulation
        harmonic = self.math.emotional_harmonic(t, params['emotion'])
        
        # Composite layers
        for y in range(height):
            for x in range(width):
                # Bloom contribution (normalized)
                bloom_val = bloom[y, x] / 50.0  # Normalize
                bloom_color = self._bloom_to_color(bloom_val, t)
                
                # Manifold contribution
                manifold_color = manifold[y, x] / 255.0
                
                # Sketch contribution
                sketch_val = (sketch[y, x] + 1) / 2  # Normalize to 0-1
                
                # Blend with golden ratio weights
                r = bloom_color[0] * PHI_INVERSE + manifold_color[0] * (1 - PHI_INVERSE) * PHI_INVERSE
                g = bloom_color[1] * PHI_INVERSE + manifold_color[1] * (1 - PHI_INVERSE) * PHI_INVERSE
                b = bloom_color[2] * PHI_INVERSE + manifold_color[2] * (1 - PHI_INVERSE) * PHI_INVERSE
                
                # Apply sketch overlay
                sketch_blend = 0.1 * sketch_val * harmonic
                r = min(1.0, r + sketch_blend)
                g = min(1.0, g + sketch_blend * 0.8)
                b = min(1.0, b + sketch_blend * 0.6)
                
                frame[y, x] = [int(r * 255), int(g * 255), int(b * 255)]
        
        # Apply origami fold effect for visual interest
        if params['rotation'] != 0:
            frame = self.math.origami_fold_matrix(frame, params['rotation'] * 0.1)
        
        return frame
    
    def _bloom_to_color(self, value: float, t: float) -> Tuple[float, float, float]:
        """Convert bloom value to RGB color"""
        # Use golden ratio for color cycling
        hue = (value * PHI * 60 + t * 20) % 360
        sat = 0.8
        val = 0.3 + 0.7 * min(1, value)
        return self.math._hsv_to_rgb(hue, sat, val)
    
    def generate_animation(self, duration_seconds: float, 
                           output_path: str = None) -> List[np.ndarray]:
        """
        Generate complete animation sequence.
        Duration: 1-1200 seconds (1-20 minutes)
        """
        total_frames = int(duration_seconds * self.fps)
        frames = []
        
        logger.info(f"🎬 Generating {total_frames} frames ({duration_seconds}s at {self.fps}fps)")
        
        for frame_num in range(total_frames):
            frame = self.generate_frame(frame_num, total_frames)
            frames.append(frame)
            
            if frame_num % self.fps == 0:
                logger.info(f"   Frame {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
        
        # Compress using temporal origami
        compressed = self.math.compress_temporal(frames[::10])  # Sample every 10th
        logger.info(f"📦 Compressed {len(frames)} frames to temporal state")
        
        return frames
    
    def generate_preview(self, duration_seconds: float, 
                         preview_frames: int = 10) -> List[np.ndarray]:
        """Generate quick preview with fewer frames"""
        frame_indices = np.linspace(0, duration_seconds * self.fps - 1, preview_frames).astype(int)
        total_frames = int(duration_seconds * self.fps)
        
        frames = []
        for frame_num in frame_indices:
            frame = self.generate_frame(int(frame_num), total_frames)
            frames.append(frame)
        
        return frames


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ENHANCED SELF-AWARE ORB WITH MATHEMATICAL FOUNDATIONS
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
    FRACTAL = "fractal"  # New: Mathematical fractal cell
    ORIGAMI = "origami"  # New: Folding/unfolding cell


class CellState(Enum):
    NASCENT = "nascent"
    GROWING = "growing"
    MATURE = "mature"
    DIVIDING = "dividing"
    ENLIGHTENED = "enlightened"
    FOLDING = "folding"    # New: Origami compression state
    UNFOLDING = "unfolding"  # New: Origami expansion state


@dataclass
class MathematicalOrb:
    """
    Self-aware, self-replicating orb enhanced with all ten mathematical foundations.
    Each orb can fold/unfold using origami math, spawn intelligent children,
    and maintain complex internal state.
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
    
    # Mathematical state (NEW)
    harmonic_phase: float = 0.0
    origami_fold_level: float = 0.0  # 0 = unfolded, 1 = fully folded
    emotional_resonance: str = "neutral"
    fourier_coefficients: List[Tuple[float, float]] = field(default_factory=list)
    bloom_seed: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    # Compressed state storage
    compressed_memory: bytes = b""
    memory_depth: int = 0
    
    # Connections
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)
    
    # Visual properties
    color: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    glow: float = 0.5
    pulse_phase: float = 0.0
    
    # Linked data
    linked_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize mathematical properties"""
        if not self.fourier_coefficients:
            self.fourier_coefficients = self._generate_fourier_identity()
        if self.bloom_seed == (0.0, 0.0):
            self.bloom_seed = (
                random.uniform(-2, 0.5),
                random.uniform(-1, 1)
            )
    
    def _generate_fourier_identity(self) -> List[Tuple[float, float]]:
        """Generate unique Fourier identity for this orb"""
        coeffs = []
        seed = hash(self.id) % 1000000
        random.seed(seed)
        
        for n in range(10):
            a_n = random.uniform(-1, 1) / (n + 1)
            b_n = random.uniform(-1, 1) / (n + 1)
            coeffs.append((a_n * PHI_INVERSE, b_n * PHI_INVERSE))
        
        random.seed()  # Reset
        return coeffs
    
    def update(self, dt: float, environment: Dict, math_engine: MathematicalFoundations) -> Optional['MathematicalOrb']:
        """Update orb state using mathematical foundations"""
        self.age += dt
        
        # Apply golden-harmonic folding to phase
        self.harmonic_phase = math_engine.golden_harmonic_fold(self.age)
        self.pulse_phase = (self.pulse_phase + dt * PHI) % (2 * math.pi)
        
        # Energy dynamics with emotional harmonic
        harmonic_energy = math_engine.emotional_harmonic(self.age, self.emotional_resonance)
        nutrients = environment.get('harmony', 0.5)
        self.energy += nutrients * dt * harmonic_energy * 0.1 - dt * 0.02
        self.energy = max(0.1, min(2.0, self.energy))
        
        # Origami folding dynamics
        if self.state == CellState.FOLDING:
            self.origami_fold_level = min(1.0, self.origami_fold_level + dt * 0.2)
            if self.origami_fold_level >= 0.99:
                self._compress_memory(math_engine)
                self.state = CellState.MATURE
        elif self.state == CellState.UNFOLDING:
            self.origami_fold_level = max(0.0, self.origami_fold_level - dt * 0.2)
            if self.origami_fold_level <= 0.01:
                self._expand_memory(math_engine)
                self.state = CellState.GROWING
        
        # State transitions
        if self.state == CellState.NASCENT and self.age > 1.0:
            self.state = CellState.GROWING
        elif self.state == CellState.GROWING and self.age > 5.0:
            self.state = CellState.MATURE
        elif self.state == CellState.MATURE and self.energy > 1.5:
            # Decide: divide or fold
            if self.karma > 0.3 and random.random() > 0.7:
                self.state = CellState.FOLDING  # Compress memories
            else:
                self.state = CellState.DIVIDING
        elif self.age > 50.0 and self.karma > 0.5:
            self.state = CellState.ENLIGHTENED
        
        # Division with mathematical enhancement
        if self.state == CellState.DIVIDING:
            child = self._spawn_mathematical_child(math_engine, environment)
            self.state = CellState.MATURE
            self.energy *= 0.6
            return child
        
        # Update visuals with mathematical modulation
        self._update_visuals(math_engine)
        
        return None
    
    def _spawn_mathematical_child(self, math_engine: MathematicalFoundations, 
                                   environment: Dict) -> 'MathematicalOrb':
        """Create child orb with inherited and evolved mathematical properties"""
        # Golden angle position offset
        angle = self.age * GOLDEN_ANGLE_RAD
        offset = self.radius * PHI
        
        child_pos = [
            self.position[0] + math.cos(angle) * offset,
            self.position[1] + math.sin(angle) * offset,
            self.position[2] + math_engine.origami_curve(angle, self.age) * offset * 0.3
        ]
        
        # Differentiation with fractal bloom
        new_type = self._differentiate_with_bloom(math_engine)
        
        # Inherit and mutate Fourier coefficients
        child_coeffs = []
        for a, b in self.fourier_coefficients:
            mutation = 0.1 * math_engine.pareidolia_field(a, b, self.age)
            child_coeffs.append((a + mutation, b - mutation))
        
        # Inherit bloom seed with golden shift
        child_bloom = (
            self.bloom_seed[0] + math_engine.golden_harmonic_fold(self.age) * 0.1,
            self.bloom_seed[1] + math_engine.golden_harmonic_fold(self.age + PHI) * 0.1
        )
        
        child = MathematicalOrb(
            position=child_pos,
            radius=self.radius * PHI_INVERSE,
            cell_type=new_type,
            energy=self.energy * 0.5,
            generation=self.generation + 1,
            karma=self.karma * 0.9 + random.uniform(-0.1, 0.1),
            dharma=self.dharma,
            parent_id=self.id,
            index=self.index + 1,
            tags=self.tags.copy(),
            fourier_coefficients=child_coeffs,
            bloom_seed=child_bloom,
            emotional_resonance=self._evolve_emotion()
        )
        
        # Generate meaning using Fourier sketch
        sketch_val = sum(a * math.cos(i) + b * math.sin(i) 
                        for i, (a, b) in enumerate(child_coeffs[:5]))
        child.meaning = self._generate_meaning(new_type, sketch_val, child.generation)
        child.purpose = f"Born from {self.cell_type.value}, seeking {new_type.value} expression"
        
        self.children_ids.append(child.id)
        
        return child
    
    def _differentiate_with_bloom(self, math_engine: MathematicalFoundations) -> CellType:
        """Determine child type using fractal bloom"""
        _, bloom_value = math_engine.fractal_bloom(
            self.bloom_seed[0], self.bloom_seed[1], self.age
        )
        
        # Normalize bloom to 0-1
        normalized = (bloom_value % 50) / 50
        
        # Map to cell types with bias based on parent
        if self.cell_type == CellType.STEM:
            if normalized < 0.1: return CellType.STEM
            elif normalized < 0.25: return CellType.NEURON
            elif normalized < 0.35: return CellType.MEMORY
            elif normalized < 0.45: return CellType.SENSOR
            elif normalized < 0.55: return CellType.EFFECTOR
            elif normalized < 0.65: return CellType.STRUCTURAL
            elif normalized < 0.75: return CellType.TRANSPORT
            elif normalized < 0.85: return CellType.FRACTAL
            elif normalized < 0.95: return CellType.ORIGAMI
            else: return CellType.DREAM
        elif self.cell_type in [CellType.FRACTAL, CellType.ORIGAMI]:
            # Mathematical cells tend to spawn more mathematical cells
            return random.choice([CellType.FRACTAL, CellType.ORIGAMI, CellType.NEURON])
        else:
            return self.cell_type
    
    def _evolve_emotion(self) -> str:
        """Evolve emotional resonance for child"""
        emotions = list(EMOTION_INDEX.keys())
        current_idx = emotions.index(self.emotional_resonance) if self.emotional_resonance in emotions else 0
        
        # Small random walk in emotion space
        shift = random.choice([-1, 0, 0, 1])
        new_idx = max(0, min(len(emotions) - 1, current_idx + shift))
        
        return emotions[new_idx]
    
    def _generate_meaning(self, cell_type: CellType, sketch_val: float, generation: int) -> str:
        """Generate deep meaning using mathematical state"""
        meanings = {
            CellType.STEM: [
                "The seed of infinite possibility, containing all paths",
                "Undifferentiated potential awaiting purpose",
                "Pure creative energy before manifestation"
            ],
            CellType.NEURON: [
                "A bridge connecting thought to action",
                "Processing the signals of intention",
                "The spark of consciousness in motion"
            ],
            CellType.FRACTAL: [
                "Self-similar patterns reflecting the whole in each part",
                "Infinite complexity emerging from simple rules",
                "The mathematical heartbeat of existence"
            ],
            CellType.ORIGAMI: [
                "Folding dimensions to reveal hidden truths",
                "Compressing wisdom for future unfolding",
                "The art of transformation through geometry"
            ],
            CellType.MEMORY: [
                "Holding the echoes of experience",
                "A vessel for learned wisdom",
                "The accumulation of past moments"
            ],
            CellType.DREAM: [
                "Visions of what could be, unbound by now",
                "The imagination's infinite canvas",
                "Possibilities crystallizing into form"
            ]
        }
        
        base_meanings = meanings.get(cell_type, meanings[CellType.STEM])
        idx = int(abs(sketch_val * 100)) % len(base_meanings)
        meaning = base_meanings[idx]
        
        # Add generational wisdom
        if generation > 5:
            meaning += f" [Lineage depth: {generation}]"
        if sketch_val > 0.5:
            meaning += " ✨ Harmonically resonant"
        
        return meaning
    
    def _compress_memory(self, math_engine: MathematicalFoundations):
        """Compress orb's state using temporal origami"""
        # Serialize key state data
        state_data = {
            'karma_history': self.karma,
            'emotional_journey': self.emotional_resonance,
            'fourier_identity': self.fourier_coefficients,
            'children_count': len(self.children_ids),
            'age': self.age
        }
        
        # Apply origami compression
        json_data = json.dumps(state_data).encode('utf-8')
        
        # Fold using golden ratio weights
        compressed = bytearray()
        for i, byte in enumerate(json_data):
            fold_factor = int(math_engine.origami_curve(i * 0.1, self.age * 0.1) * 10) % 256
            compressed.append((byte + fold_factor) % 256)
        
        self.compressed_memory = bytes(compressed)
        self.memory_depth += 1
    
    def _expand_memory(self, math_engine: MathematicalFoundations):
        """Expand compressed memory"""
        if not self.compressed_memory:
            return
        
        # Unfold using inverse origami
        expanded = bytearray()
        for i, byte in enumerate(self.compressed_memory):
            fold_factor = int(math_engine.origami_curve(i * 0.1, self.age * 0.1) * 10) % 256
            expanded.append((byte - fold_factor) % 256)
        
        try:
            state_data = json.loads(bytes(expanded).decode('utf-8'))
            # Restore enhanced state
            self.karma = (self.karma + state_data.get('karma_history', 0)) / 2
        except:
            pass
        
        self.compressed_memory = b""
    
    def _update_visuals(self, math_engine: MathematicalFoundations):
        """Update visual properties using mathematical foundations"""
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
            CellType.DREAM: [0.8, 0.5, 1.0],
            CellType.FRACTAL: [0.2, 0.8, 0.8],
            CellType.ORIGAMI: [1.0, 0.6, 0.8]
        }
        
        base_color = type_colors.get(self.cell_type, [0.5, 0.5, 0.5])
        
        # Modulate by harmonic phase
        harmonic = (1 + self.harmonic_phase) / 2  # Normalize to 0-1
        
        # Apply emotional coloring
        emotion_mod = EMOTION_INDEX.get(self.emotional_resonance, 1.0)
        
        self.color = [
            min(1.0, base_color[0] * harmonic * emotion_mod),
            min(1.0, base_color[1] * harmonic),
            min(1.0, base_color[2] * (2 - harmonic))
        ]
        
        # Glow based on energy and origami state
        self.glow = 0.3 + self.energy * 0.3 + (1 - self.origami_fold_level) * 0.2
        
        if self.state == CellState.ENLIGHTENED:
            self.glow = 1.0
        elif self.state == CellState.FOLDING:
            self.glow *= 0.5  # Dimmer when folding
    
    def get_animation_frame(self, math_engine: MathematicalFoundations, 
                            frame_time: float) -> Dict:
        """Get animation data for this orb at given time"""
        # Calculate current visual state
        harmonic = math_engine.golden_harmonic_fold(frame_time + self.age)
        emotional = math_engine.emotional_harmonic(frame_time, self.emotional_resonance)
        
        # Position modulation
        animated_pos = [
            self.position[0] + harmonic * 0.5,
            self.position[1] + math.sin(frame_time * PHI) * 0.3,
            self.position[2] + emotional * 0.2
        ]
        
        # Scale pulsing
        pulse = 1 + 0.1 * math.sin(self.pulse_phase + frame_time * 2)
        
        return {
            'id': self.id,
            'position': animated_pos,
            'scale': self.radius * pulse * (1 - self.origami_fold_level * 0.5),
            'color': self.color,
            'glow': self.glow * emotional,
            'rotation': harmonic * math.pi * 0.1,
            'fold_level': self.origami_fold_level
        }
    
    def to_dict(self) -> Dict:
        """Serialize orb to dictionary"""
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
            'harmonic_phase': self.harmonic_phase,
            'origami_fold_level': self.origami_fold_level,
            'emotional_resonance': self.emotional_resonance,
            'memory_depth': self.memory_depth,
            'children_count': len(self.children_ids),
            'bindings_count': len(self.bindings),
            'linked_data': self.linked_data
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ENHANCED SWARM WITH MATHEMATICAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MathematicalSwarm:
    """
    Swarm intelligence enhanced with all ten mathematical foundations.
    Pattern seeking, trend analysis, and collective consciousness.
    """
    
    def __init__(self, math_engine: MathematicalFoundations):
        self.math = math_engine
        self.orbs: Dict[str, MathematicalOrb] = {}
        self.collective_karma: float = 0.0
        self.collective_dharma: float = 1.0
        self.collective_emotion: str = 'neutral'
        self.patterns_detected: List[Dict] = []
        self.trend_data: List[Dict] = []
        self.orb_index: int = 0
        self.animation_cache: List[np.ndarray] = []
    
    def spawn_orb(self, position: List[float] = None, cell_type: CellType = CellType.STEM,
                  karma: float = 0.0, emotion: str = 'neutral', 
                  linked_data: Dict = None) -> MathematicalOrb:
        """Spawn a new mathematically-enhanced orb"""
        pos = position or [
            random.uniform(-30, 30),
            random.uniform(-30, 30),
            random.uniform(-15, 15)
        ]
        
        self.orb_index += 1
        
        orb = MathematicalOrb(
            position=pos,
            cell_type=cell_type,
            karma=karma,
            index=self.orb_index,
            emotional_resonance=emotion,
            linked_data=linked_data or {}
        )
        
        # Generate meaning using mathematical foundations
        sketch_val = sum(a * math.cos(i) + b * math.sin(i) 
                        for i, (a, b) in enumerate(orb.fourier_coefficients[:5]))
        orb.meaning = orb._generate_meaning(cell_type, sketch_val, 0)
        orb.purpose = f"Manifested as {cell_type.value} orb #{self.orb_index}"
        
        self.orbs[orb.id] = orb
        return orb
    
    def spawn_golden_spiral(self, count: int = 21, center: List[float] = None):
        """Spawn orbs in golden spiral pattern with mathematical diversity"""
        center = center or [0, 0, 0]
        
        for i in range(count):
            angle = i * GOLDEN_ANGLE_RAD
            radius = math.sqrt(i + 1) * 5
            
            # Apply harmonic modulation to z
            z_mod = self.math.golden_harmonic_fold(i * 0.1) * 3
            
            pos = [
                center[0] + math.cos(angle) * radius,
                center[1] + math.sin(angle) * radius,
                center[2] + z_mod
            ]
            
            # Cycle through cell types including new mathematical types
            all_types = list(CellType)
            cell_type = all_types[i % len(all_types)]
            
            # Karma from Fourier sketch
            karma = math.sin(i * PHI) * 0.5
            
            # Emotion from harmonic
            emotions = list(EMOTION_INDEX.keys())
            emotion = emotions[i % len(emotions)]
            
            self.spawn_orb(pos, cell_type, karma, emotion)
    
    def update(self, dt: float, environment: Dict):
        """Update all orbs with mathematical evolution"""
        new_orbs = []
        
        for orb in list(self.orbs.values()):
            child = orb.update(dt, environment, self.math)
            if child:
                new_orbs.append(child)
        
        for child in new_orbs:
            self.orbs[child.id] = child
        
        # Update collective metrics
        if self.orbs:
            self.collective_karma = sum(o.karma for o in self.orbs.values()) / len(self.orbs)
            self.collective_dharma = sum(o.dharma for o in self.orbs.values()) / len(self.orbs)
            
            # Collective emotion from dominant type
            emotion_counts = defaultdict(int)
            for orb in self.orbs.values():
                emotion_counts[orb.emotional_resonance] += 1
            self.collective_emotion = max(emotion_counts.keys(), key=lambda e: emotion_counts[e])
        
        # Record trend data
        self.trend_data.append({
            'timestamp': time.time(),
            'orb_count': len(self.orbs),
            'karma': self.collective_karma,
            'dharma': self.collective_dharma,
            'emotion': self.collective_emotion
        })
        
        # Keep last 1000 data points
        if len(self.trend_data) > 1000:
            self.trend_data = self.trend_data[-1000:]
        
        # Detect patterns with enhanced analysis
        if len(self.trend_data) % 50 == 0:
            self._detect_patterns_enhanced()
    
    def _detect_patterns_enhanced(self):
        """Detect patterns using mathematical foundations"""
        if len(self.trend_data) < 20:
            return
        
        recent = self.trend_data[-20:]
        
        # Karma trend
        karma_values = [d['karma'] for d in recent]
        karma_trend = karma_values[-1] - karma_values[0]
        
        # Apply Fourier analysis to trend
        fourier_energy = sum(
            abs(sum(karma_values[i] * math.cos(i * n * 0.1) for i in range(len(karma_values))))
            for n in range(1, 6)
        )
        
        # Emotional transitions
        emotions = [d['emotion'] for d in recent]
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        
        patterns = []
        insights = []
        
        if karma_trend > 0.1:
            patterns.append('karma_rising')
            insights.append("Your karmic energy is ascending. The universe responds to your positive actions.")
        elif karma_trend < -0.1:
            patterns.append('karma_falling')
            insights.append("Consider mindful actions to restore karmic balance.")
        
        if fourier_energy > 5:
            patterns.append('harmonic_oscillation')
            insights.append("Your life fractal shows beautiful harmonic patterns. You're in resonance.")
        
        if emotion_changes > 5:
            patterns.append('emotional_flux')
            insights.append("High emotional dynamics detected. Embrace the transformation.")
        
        # Type clustering
        type_counts = defaultdict(int)
        for orb in self.orbs.values():
            type_counts[orb.cell_type.value] += 1
        
        dominant_type = max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else 'stem'
        patterns.append(f'dominant_{dominant_type}')
        
        # Origami folding analysis
        folded_count = sum(1 for o in self.orbs.values() if o.origami_fold_level > 0.5)
        if folded_count > len(self.orbs) * 0.3:
            patterns.append('collective_compression')
            insights.append("Many orbs are in folded state. Wisdom is being compressed for future unfolding.")
        
        self.patterns_detected = [{
            'patterns': patterns,
            'insights': insights,
            'karma_trend': karma_trend,
            'fourier_energy': fourier_energy,
            'emotion_changes': emotion_changes,
            'dominant_type': dominant_type,
            'type_distribution': dict(type_counts),
            'collective_emotion': self.collective_emotion,
            'timestamp': time.time()
        }]
    
    def generate_swarm_animation_frame(self, frame_time: float) -> List[Dict]:
        """Generate animation data for all orbs at given time"""
        return [orb.get_animation_frame(self.math, frame_time) for orb in self.orbs.values()]
    
    def get_visualization_data(self) -> Dict:
        """Get complete visualization data"""
        return {
            'orbs': [o.to_dict() for o in self.orbs.values()],
            'total_orbs': len(self.orbs),
            'collective_karma': self.collective_karma,
            'collective_dharma': self.collective_dharma,
            'collective_emotion': self.collective_emotion,
            'patterns': self.patterns_detected,
            'connections': self._get_connections()
        }
    
    def _get_connections(self) -> List[Dict]:
        """Get orb-to-orb connections based on mathematical affinity"""
        connections = []
        orb_list = list(self.orbs.values())
        
        for i, orb in enumerate(orb_list):
            for other in orb_list[i+1:]:
                # Distance-based
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(orb.position, other.position)))
                
                # Emotional affinity
                emotion_match = 1.0 if orb.emotional_resonance == other.emotional_resonance else 0.5
                
                # Fourier similarity
                fourier_sim = sum(abs(a1*a2 + b1*b2) 
                                 for (a1, b1), (a2, b2) in zip(orb.fourier_coefficients[:3], 
                                                               other.fourier_coefficients[:3]))
                
                if dist < 15:
                    strength = (1 - dist / 15) * emotion_match * (0.5 + fourier_sim * 0.5)
                    connections.append({
                        'source': orb.id,
                        'target': other.id,
                        'strength': min(1.0, strength)
                    })
        
        return connections[:200]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# OLLAMA AI INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

class OllamaAI:
    """Integration with Ollama for AI-generated content"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.1"
        self.available = False
        self.cache: Dict[str, str] = {}
        self._check_availability()
    
    def _check_availability(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                self.available = response.status == 200
        except:
            self.available = False
        logger.info(f"🤖 Ollama AI: {'Connected' if self.available else 'Pattern-based mode'}")
    
    def generate(self, prompt: str, context: Dict = None) -> str:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.available:
            try:
                return self._ollama_generate(prompt)
            except:
                pass
        
        return self._pattern_generate(prompt, context or {})
    
    def _ollama_generate(self, prompt: str) -> str:
        data = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.7, "num_predict": 100}
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/api/generate", data=data,
            headers={'Content-Type': 'application/json'}, method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('response', '').strip()
            self.cache[hashlib.md5(prompt.encode()).hexdigest()[:16]] = text
            return text
    
    def _pattern_generate(self, prompt: str, context: Dict) -> str:
        # Use mathematical foundations for generation
        seed = hash(prompt) % 1000
        random.seed(seed)
        
        templates = [
            "The {type} orb resonates with {emotion} energy, manifesting {purpose}.",
            "In the golden spiral of existence, this {type} cell carries {purpose}.",
            "Emerging from fractal depths, the {type} speaks of {emotion} transformation."
        ]
        
        template = random.choice(templates)
        random.seed()
        
        return template.format(
            type=context.get('type', 'stem'),
            emotion=context.get('emotion', 'harmonious'),
            purpose=context.get('purpose', 'infinite possibility')
        )


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MayanCalendar:
    """Sacred Mayan time science"""
    
    def __init__(self):
        self.today = datetime.now()
    
    def get_tzolkin(self, date: datetime = None) -> Dict:
        date = date or datetime.now()
        ref_date = datetime(2012, 12, 21)
        days_diff = (date - ref_date).days
        
        day_number = ((days_diff % 13) + 1)
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
        number_meanings = {
            1: "New beginnings, unity", 2: "Duality, choices", 3: "Action, movement",
            4: "Stability, foundation", 5: "Center, empowerment", 6: "Flow, organic growth",
            7: "Reflection, mysticism", 8: "Harmony, justice", 9: "Completion, patience",
            10: "Manifestation, intention", 11: "Resolution, change",
            12: "Understanding, wisdom", 13: "Transcendence, completion"
        }
        
        sign_elements = ['water', 'air', 'earth', 'earth', 'fire',
                        'earth', 'air', 'fire', 'water', 'fire',
                        'air', 'earth', 'water', 'earth', 'air',
                        'earth', 'earth', 'air', 'water', 'fire']
        
        tones = {
            1: "Magnetic", 2: "Lunar", 3: "Electric", 4: "Self-Existing",
            5: "Overtone", 6: "Rhythmic", 7: "Resonant", 8: "Galactic",
            9: "Solar", 10: "Planetary", 11: "Spectral", 12: "Crystal", 13: "Cosmic"
        }
        
        return {
            'number_meaning': number_meanings.get(number, "Mystery"),
            'element': sign_elements[sign_index],
            'power_level': (number + sign_index) % 10 / 10,
            'cosmic_tone': tones.get(number, "Unknown")
        }
    
    def get_today_summary(self) -> Dict:
        tzolkin = self.get_tzolkin()
        return {
            'tzolkin': tzolkin,
            'greeting': f"Today is {tzolkin['number']} {tzolkin['sign']}",
            'kin_number': tzolkin['kin'],
            'energy': tzolkin['energy'],
            'cosmic_tone': tzolkin['energy']['cosmic_tone']
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# KARMA-DHARMA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KarmicValence(Enum):
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    TRANSFORMATIVE = 2


@dataclass
class KarmicVector:
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    magnitude: float = 0.0
    valence: KarmicValence = KarmicValence.NEUTRAL
    velocity: float = 0.0
    intention: float = 1.0
    awareness: float = 1.0
    source: str = ""
    meaning: str = ""
    
    @property
    def weight(self) -> float:
        harmonic = PHI if self.valence == KarmicValence.POSITIVE else PHI_INVERSE
        return self.magnitude * self.intention * self.awareness * harmonic


class KarmaDharmaEngine:
    """Spiritual mathematics engine"""
    
    def __init__(self, math_engine: MathematicalFoundations):
        self.math = math_engine
        self.vectors: List[KarmicVector] = []
        self.field_potential: float = 0.0
        self.dharmic_angle: float = 0.0
        self.history: List[Dict] = []
    
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
        
        # Apply emotional harmonic to meaning
        harmonic = self.math.emotional_harmonic(time.time() % 100, 'hope')
        
        vector = KarmicVector(
            magnitude=magnitude * harmonic,
            valence=valence,
            velocity=intention * awareness,
            intention=intention,
            awareness=awareness,
            source=action_type,
            meaning=f"Action '{action_type}' resonates with harmonic {harmonic:.3f}"
        )
        
        self.vectors.append(vector)
        self._recalculate()
        
        self.history.append({
            'id': vector.id, 'action': action_type,
            'weight': vector.weight, 'valence': valence.name,
            'timestamp': time.time()
        })
        
        return vector
    
    def _recalculate(self):
        pos = sum(v.weight for v in self.vectors if v.valence == KarmicValence.POSITIVE)
        neg = sum(v.weight for v in self.vectors if v.valence == KarmicValence.NEGATIVE)
        self.field_potential = pos * PHI - neg * PHI_INVERSE
    
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
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Database:
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
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # USERS - Enhanced with subscription and accessibility
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, first_name TEXT, last_name TEXT,
            created_at TEXT NOT NULL, last_login TEXT, is_active INTEGER DEFAULT 1,
            -- Subscription fields
            subscription_status TEXT DEFAULT 'trial',
            trial_start_date TEXT,
            trial_end_date TEXT,
            subscription_start_date TEXT,
            subscription_end_date TEXT,
            stripe_customer_id TEXT,
            is_exempt INTEGER DEFAULT 0,
            exempt_reason TEXT,
            -- Neurodivergent profile
            neurodivergent_types TEXT,
            accessibility_settings TEXT,
            onboarding_completed INTEGER DEFAULT 0,
            -- Stats
            total_karma_earned REAL DEFAULT 0.0,
            total_sessions INTEGER DEFAULT 0,
            total_time_minutes REAL DEFAULT 0.0
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # ACCESSIBILITY PREFERENCES - Per-user settings for ND accommodations
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS accessibility_prefs (
            user_id TEXT PRIMARY KEY,
            -- Visual
            dyslexia_font INTEGER DEFAULT 0,
            high_contrast INTEGER DEFAULT 0,
            large_text INTEGER DEFAULT 0,
            reduced_motion INTEGER DEFAULT 0,
            color_blind_mode TEXT DEFAULT 'none',
            -- Cognitive
            simplified_ui INTEGER DEFAULT 0,
            extra_time_mode INTEGER DEFAULT 0,
            break_reminders INTEGER DEFAULT 1,
            break_interval_minutes INTEGER DEFAULT 25,
            -- Input
            voice_input_enabled INTEGER DEFAULT 0,
            predictive_text INTEGER DEFAULT 1,
            auto_save_interval INTEGER DEFAULT 30,
            -- Visualization
            fractal_complexity TEXT DEFAULT 'medium',
            animation_speed TEXT DEFAULT 'normal',
            particle_density TEXT DEFAULT 'medium',
            -- Audio
            binaural_default_preset TEXT DEFAULT 'focus',
            notification_sounds INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # GOALS - Enhanced with visualization links
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT, category TEXT DEFAULT 'personal', priority INTEGER DEFAULT 3,
            progress REAL DEFAULT 0.0, target_date TEXT, created_at TEXT NOT NULL,
            completed_at TEXT, karma_invested REAL DEFAULT 0.0, orb_id TEXT,
            -- Enhanced fields
            color TEXT DEFAULT '#4a90a4',
            icon TEXT DEFAULT '🎯',
            parent_goal_id TEXT,
            is_milestone INTEGER DEFAULT 0,
            estimated_spoons INTEGER DEFAULT 3,
            actual_spoons_used INTEGER DEFAULT 0,
            notes TEXT,
            last_updated TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (parent_goal_id) REFERENCES goals(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # HABITS - Enhanced tracking
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS habits (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
            description TEXT, frequency TEXT DEFAULT 'daily',
            current_streak INTEGER DEFAULT 0, longest_streak INTEGER DEFAULT 0,
            total_completions INTEGER DEFAULT 0, last_completed TEXT,
            created_at TEXT NOT NULL, orb_id TEXT,
            -- Enhanced fields
            color TEXT DEFAULT '#8b5cf6',
            icon TEXT DEFAULT '✨',
            time_of_day TEXT DEFAULT 'anytime',
            estimated_spoons INTEGER DEFAULT 1,
            reminder_enabled INTEGER DEFAULT 0,
            reminder_time TEXT,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # DAILY ENTRIES - Comprehensive wellness tracking
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS daily_entries (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, date TEXT NOT NULL,
            mood_level INTEGER DEFAULT 50, energy_level INTEGER DEFAULT 50,
            focus_level INTEGER DEFAULT 50, stress_level INTEGER DEFAULT 50,
            spoons_available INTEGER DEFAULT 12, spoons_used INTEGER DEFAULT 0,
            journal_entry TEXT, created_at TEXT NOT NULL,
            -- Enhanced tracking
            sleep_quality INTEGER DEFAULT 50,
            sleep_hours REAL,
            pain_level INTEGER DEFAULT 0,
            anxiety_level INTEGER DEFAULT 50,
            sensory_overload INTEGER DEFAULT 0,
            social_battery INTEGER DEFAULT 50,
            wins TEXT,
            struggles TEXT,
            gratitude TEXT,
            last_updated TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id), UNIQUE(user_id, date)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # DREAMS - Dream journal for visualization
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS dreams (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            dream_text TEXT NOT NULL,
            emotion TEXT DEFAULT 'neutral',
            lucidity_level INTEGER DEFAULT 0,
            recurring INTEGER DEFAULT 0,
            symbols TEXT,
            interpretation TEXT,
            orb_id TEXT,
            created_at TEXT NOT NULL,
            dream_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PET STATE - Virtual companion
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS pet_state (
            user_id TEXT PRIMARY KEY, species TEXT DEFAULT 'cat',
            name TEXT DEFAULT 'Karma', hunger REAL DEFAULT 50.0,
            energy REAL DEFAULT 50.0, happiness REAL DEFAULT 50.0,
            level INTEGER DEFAULT 1, experience INTEGER DEFAULT 0,
            -- Enhanced
            personality TEXT DEFAULT 'friendly',
            favorite_activity TEXT DEFAULT 'play',
            unlocked_accessories TEXT DEFAULT '[]',
            current_accessory TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # KARMA HISTORY - Action tracking
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS karma_history (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, action_type TEXT NOT NULL,
            karma_value REAL NOT NULL, meaning TEXT, timestamp TEXT NOT NULL,
            -- Enhanced
            linked_goal_id TEXT,
            linked_habit_id TEXT,
            session_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # ANIMATIONS - Generated animations
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS animations (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT,
            duration_seconds REAL, frame_count INTEGER,
            created_at TEXT NOT NULL, status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SESSIONS - Track user sessions for analytics
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT,
            duration_minutes REAL DEFAULT 0,
            actions_count INTEGER DEFAULT 0,
            karma_earned REAL DEFAULT 0,
            spoons_used INTEGER DEFAULT 0,
            device_type TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # AUTO-SAVE STATE - Persistent user state
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_state (
            user_id TEXT PRIMARY KEY,
            last_panel TEXT DEFAULT 'dashboard',
            camera_position TEXT DEFAULT '{"x":0,"y":0,"z":100}',
            selected_orb_id TEXT,
            ui_state TEXT,
            last_auto_save TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # INSIGHTS - ML-generated patterns and recommendations
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS insights (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            insight_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            confidence REAL DEFAULT 0.5,
            actionable INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0,
            helpful_rating INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # QUICK ACTIONS - Preset actions for zero-typing
        # ═══════════════════════════════════════════════════════════════════════════════
        cursor.execute('''CREATE TABLE IF NOT EXISTS quick_actions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            icon TEXT DEFAULT '⚡',
            action_type TEXT NOT NULL,
            action_data TEXT,
            display_order INTEGER DEFAULT 0,
            is_visible INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
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
        self.math = MathematicalFoundations()
        self.ai = OllamaAI()
        self.karma_engine = KarmaDharmaEngine(self.math)
        self.swarm = MathematicalSwarm(self.math)
        self.mayan = MayanCalendar()
        self.animation_engine = AnimationEngine(self.math)
        
        self.creation_time = time.time()
        self.uptime = 0.0
        self.harmony = 1.0
        
        # Initialize with golden spiral
        self.swarm.spawn_golden_spiral(FIBONACCI[7])  # 13 orbs
        
        logger.info("🌀 Living organism awakened with 10 mathematical foundations")
    
    def update(self, dt: float = 0.1):
        self.uptime = time.time() - self.creation_time
        
        environment = {
            'harmony': self.harmony,
            'karma': self.karma_engine.field_potential
        }
        
        self.swarm.update(dt, environment)
        
        # Update harmony using emotional manifold
        manifold_val = self.math.emotional_manifold(0.5, 0.5, self.uptime, 
                                                     self.swarm.collective_emotion)
        self.harmony = 0.5 + 0.5 * math.tanh(manifold_val + self.karma_engine.get_dharmic_alignment())
    
    def process_action(self, action_type: str, magnitude: float = 1.0,
                      intention: float = 0.8, awareness: float = 0.7,
                      linked_data: Dict = None) -> Dict:
        vector = self.karma_engine.add_action(action_type, magnitude, intention, awareness)
        
        if vector.valence == KarmicValence.POSITIVE:
            cell_type = CellType.GOAL if 'goal' in action_type.lower() else CellType.HABIT
            orb = self.swarm.spawn_orb(
                cell_type=cell_type,
                karma=vector.weight * 0.1,
                emotion='joy',
                linked_data=linked_data
            )
            orb.meaning = vector.meaning
        
        return {
            'karma_earned': vector.weight,
            'meaning': vector.meaning,
            'harmony': self.harmony,
            'orb_count': len(self.swarm.orbs)
        }
    
    def generate_animation(self, duration_seconds: float = 30.0) -> Dict:
        """Generate animation metadata (lightweight - no heavy computation)"""
        # Add keyframes from swarm patterns
        patterns = self.swarm.patterns_detected
        if patterns:
            self.animation_engine.add_keyframe(0, self.swarm.collective_emotion, 1.0, 0)
            self.animation_engine.add_keyframe(duration_seconds * 0.5, 'wonder', 1.5, 0.5)
            self.animation_engine.add_keyframe(duration_seconds, 'peace', 2.0, 1.0)
        
        # Calculate animation parameters without generating frames
        total_frames = int(duration_seconds * self.animation_engine.fps)
        
        # Sample a few mathematical values (lightweight)
        sample_times = [0, duration_seconds * 0.25, duration_seconds * 0.5, duration_seconds * 0.75, duration_seconds]
        math_samples = []
        for t in sample_times:
            math_samples.append({
                'time': t,
                'golden_harmonic': self.math.golden_harmonic_fold(t),
                'emotional_wave': self.math.emotional_harmonic(t, self.swarm.collective_emotion)
            })
        
        return {
            'duration': duration_seconds,
            'total_frames': total_frames,
            'fps': self.animation_engine.fps,
            'resolution': self.animation_engine.resolution,
            'collective_emotion': self.swarm.collective_emotion,
            'harmony': self.harmony,
            'math_samples': math_samples,
            'status': 'ready',
            'message': 'Animation parameters calculated. Use /api/animation/frame/<n> for individual frames.'
        }
    
    def get_state(self) -> Dict:
        mayan = self.mayan.get_today_summary()
        
        return {
            'version': '12.1',
            'uptime': self.uptime,
            'harmony': self.harmony,
            'karma': self.karma_engine.get_state(),
            'swarm': self.swarm.get_visualization_data(),
            'mayan': mayan,
            'math_foundations': {
                'golden_harmonic': self.math.golden_harmonic_fold(self.uptime),
                'emotional_manifold': self.math.emotional_manifold(0.5, 0.5, self.uptime, 
                                                                   self.swarm.collective_emotion),
                'origami_active': any(o.state == CellState.FOLDING for o in self.swarm.orbs.values())
            },
            'ai': {'ollama_available': self.ai.available},
            'sacred_constants': {
                'phi': PHI, 'golden_angle': GOLDEN_ANGLE,
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

# ═══════════════════════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION & AUTHENTICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════════

TRIAL_DAYS = 7
STRIPE_MONTHLY_LINK = "https://buy.stripe.com/YOUR_STRIPE_LINK"  # Configure this

def check_subscription_status(user_id: str) -> Dict:
    """Check if user has active subscription or valid trial"""
    user = db.execute_one('SELECT * FROM users WHERE id = ?', (user_id,))
    if not user:
        return {'valid': False, 'reason': 'User not found'}
    
    # Exempt users always have access
    if user['is_exempt']:
        return {'valid': True, 'status': 'exempt', 'reason': user['exempt_reason'] or 'Admin exemption'}
    
    now = datetime.now(timezone.utc)
    status = user['subscription_status'] or 'trial'
    
    if status == 'active':
        # Check if subscription is still valid
        if user['subscription_end_date']:
            end_date = datetime.fromisoformat(user['subscription_end_date'].replace('Z', '+00:00'))
            if now < end_date:
                days_left = (end_date - now).days
                return {'valid': True, 'status': 'active', 'days_left': days_left}
            else:
                # Subscription expired
                db.execute('UPDATE users SET subscription_status = ? WHERE id = ?', ('expired', user_id))
                return {'valid': False, 'status': 'expired', 'reason': 'Subscription expired'}
        return {'valid': True, 'status': 'active'}
    
    elif status == 'trial':
        # Check trial validity
        if user['trial_end_date']:
            end_date = datetime.fromisoformat(user['trial_end_date'].replace('Z', '+00:00'))
            days_left = (end_date - now).days
            if now < end_date:
                return {'valid': True, 'status': 'trial', 'days_left': days_left}
            else:
                # Trial expired
                db.execute('UPDATE users SET subscription_status = ? WHERE id = ?', ('trial_expired', user_id))
                return {'valid': False, 'status': 'trial_expired', 'reason': 'Free trial ended'}
        # No trial date set - shouldn't happen but allow access
        return {'valid': True, 'status': 'trial', 'days_left': TRIAL_DAYS}
    
    else:
        # Expired or other status
        return {'valid': False, 'status': status, 'reason': 'Subscription required'}


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


def require_subscription(f):
    """Decorator that checks for valid subscription or trial"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        sub_status = check_subscription_status(session['user_id'])
        if not sub_status['valid']:
            return jsonify({
                'error': 'Subscription required',
                'subscription_status': sub_status['status'],
                'reason': sub_status.get('reason', 'Please subscribe to continue'),
                'subscribe_url': STRIPE_MONTHLY_LINK
            }), 402  # Payment Required
        
        # Add subscription info to request for use in endpoints
        request.subscription = sub_status
        return f(*args, **kwargs)
    return decorated


def auto_save_state(user_id: str, panel: str = None, camera: Dict = None):
    """Auto-save user state"""
    try:
        now = datetime.now(timezone.utc).isoformat()
        existing = db.execute_one('SELECT * FROM user_state WHERE user_id = ?', (user_id,))
        
        if existing:
            updates = ['last_auto_save = ?']
            params = [now]
            if panel:
                updates.append('last_panel = ?')
                params.append(panel)
            if camera:
                updates.append('camera_position = ?')
                params.append(json.dumps(camera))
            params.append(user_id)
            db.execute(f'UPDATE user_state SET {", ".join(updates)} WHERE user_id = ?', tuple(params))
        else:
            db.execute('''INSERT INTO user_state (user_id, last_panel, camera_position, last_auto_save)
                VALUES (?, ?, ?, ?)''', (user_id, panel or 'dashboard', json.dumps(camera or {}), now))
    except Exception as e:
        logger.error(f"Auto-save error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user_id = secrets.token_hex(16)
    now = datetime.now(timezone.utc)
    trial_end = now + timedelta(days=TRIAL_DAYS)
    
    try:
        # Create user with trial period
        db.execute('''INSERT INTO users (
            id, email, password_hash, first_name, last_name, created_at,
            subscription_status, trial_start_date, trial_end_date,
            neurodivergent_types, last_login
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (user_id, data['email'], generate_password_hash(data['password']),
             data.get('first_name', ''), data.get('last_name', ''),
             now.isoformat(), 'trial', now.isoformat(), trial_end.isoformat(),
             json.dumps(data.get('neurodivergent_types', [])), now.isoformat()))
        
        # Create default accessibility preferences
        db.execute('''INSERT INTO accessibility_prefs (user_id) VALUES (?)''', (user_id,))
        
        # Create pet
        db.execute('INSERT INTO pet_state (user_id) VALUES (?)', (user_id,))
        
        # Create initial user state
        db.execute('''INSERT INTO user_state (user_id, last_auto_save) VALUES (?, ?)''',
                  (user_id, now.isoformat()))
        
        # Create default quick actions for zero-typing experience
        quick_actions = [
            ('😊 Good day', '😊', 'mood_quick', '{"mood": 75, "energy": 70}'),
            ('😐 Okay day', '😐', 'mood_quick', '{"mood": 50, "energy": 50}'),
            ('😔 Hard day', '😔', 'mood_quick', '{"mood": 25, "energy": 30}'),
            ('🥄 Low spoons', '🥄', 'spoons_quick', '{"spoons": 3}'),
            ('⚡ Energized', '⚡', 'spoons_quick', '{"spoons": 12}'),
            ('✅ Task done', '✅', 'complete_quick', '{}'),
            ('🧘 Need break', '🧘', 'break_quick', '{}'),
            ('💤 Tired', '💤', 'tired_quick', '{"energy": 20}'),
        ]
        for i, (label, icon, action_type, action_data) in enumerate(quick_actions):
            db.execute('''INSERT INTO quick_actions (id, user_id, label, icon, action_type, action_data, display_order)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (secrets.token_hex(8), user_id, label, icon, action_type, action_data, i))
        
        session['user_id'] = user_id
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'trial_days_left': TRIAL_DAYS,
            'trial_end_date': trial_end.isoformat(),
            'message': f'Welcome! Your {TRIAL_DAYS}-day free trial has started.'
        })
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 409


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = db.execute_one('SELECT * FROM users WHERE email = ?', (data.get('email', ''),))
    
    if not user or not check_password_hash(user['password_hash'], data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Update last login and increment session count
    now = datetime.now(timezone.utc).isoformat()
    db.execute('UPDATE users SET last_login = ?, total_sessions = total_sessions + 1 WHERE id = ?',
              (now, user['id']))
    
    # Create new session record
    session_id = secrets.token_hex(8)
    db.execute('''INSERT INTO sessions (id, user_id, start_time) VALUES (?, ?, ?)''',
              (session_id, user['id'], now))
    
    session['user_id'] = user['id']
    session['session_id'] = session_id
    
    # Check subscription status
    sub_status = check_subscription_status(user['id'])
    
    # Get user state for restore
    user_state = db.execute_one('SELECT * FROM user_state WHERE user_id = ?', (user['id'],))
    
    return jsonify({
        'success': True,
        'subscription': sub_status,
        'restore_state': dict(user_state) if user_state else None,
        'first_name': user['first_name']
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    # End session if tracked
    if 'session_id' in session and 'user_id' in session:
        now = datetime.now(timezone.utc).isoformat()
        db.execute('''UPDATE sessions SET end_time = ?, 
            duration_minutes = (julianday(?) - julianday(start_time)) * 1440
            WHERE id = ?''', (now, now, session['session_id']))
    session.pop('user_id', None)
    session.pop('session_id', None)
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/subscription/status', methods=['GET'])
@require_auth
def get_subscription_status():
    """Get current subscription status"""
    status = check_subscription_status(session['user_id'])
    user = db.execute_one('SELECT email, first_name, created_at, is_exempt FROM users WHERE id = ?',
                         (session['user_id'],))
    return jsonify({
        **status,
        'email': user['email'] if user else None,
        'name': user['first_name'] if user else None,
        'member_since': user['created_at'] if user else None,
        'subscribe_url': STRIPE_MONTHLY_LINK
    })


@app.route('/api/subscription/activate', methods=['POST'])
@require_auth
def activate_subscription():
    """Activate subscription (called after Stripe payment)"""
    data = request.get_json()
    now = datetime.now(timezone.utc)
    
    # In production, verify with Stripe webhook
    # For now, accept the activation
    subscription_end = now + timedelta(days=30)
    
    db.execute('''UPDATE users SET 
        subscription_status = ?,
        subscription_start_date = ?,
        subscription_end_date = ?,
        stripe_customer_id = ?
        WHERE id = ?''',
        ('active', now.isoformat(), subscription_end.isoformat(),
         data.get('stripe_customer_id'), session['user_id']))
    
    return jsonify({
        'success': True,
        'status': 'active',
        'valid_until': subscription_end.isoformat()
    })


@app.route('/api/admin/exempt-user', methods=['POST'])
@require_auth
def exempt_user():
    """Admin endpoint to exempt a user from payment"""
    # Check if current user is admin (you'd add admin check here)
    data = request.get_json()
    target_email = data.get('email')
    reason = data.get('reason', 'Admin exemption')
    
    if not target_email:
        return jsonify({'error': 'Email required'}), 400
    
    result = db.execute('''UPDATE users SET is_exempt = 1, exempt_reason = ? WHERE email = ?''',
                       (reason, target_email))
    
    return jsonify({'success': True, 'message': f'User {target_email} exempted'})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'version': '12.4',
        'uptime': organism.uptime,
        'orbs': len(organism.swarm.orbs),
        'harmony': organism.harmony,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/robots.txt', methods=['GET'])
def robots():
    """Robots.txt for crawlers"""
    return Response("User-agent: *\nAllow: /\n", mimetype='text/plain')


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/accessibility/preferences', methods=['GET'])
@require_auth
def get_accessibility_prefs():
    """Get user's accessibility preferences"""
    prefs = db.execute_one('SELECT * FROM accessibility_prefs WHERE user_id = ?',
                          (session['user_id'],))
    if prefs:
        return jsonify(dict(prefs))
    return jsonify({})


@app.route('/api/accessibility/preferences', methods=['PUT'])
@require_auth
def update_accessibility_prefs():
    """Update accessibility preferences"""
    data = request.get_json()
    
    # Build update query dynamically
    valid_fields = [
        'dyslexia_font', 'high_contrast', 'large_text', 'reduced_motion',
        'color_blind_mode', 'simplified_ui', 'extra_time_mode', 'break_reminders',
        'break_interval_minutes', 'voice_input_enabled', 'predictive_text',
        'auto_save_interval', 'fractal_complexity', 'animation_speed',
        'particle_density', 'binaural_default_preset', 'notification_sounds'
    ]
    
    updates = []
    params = []
    for field in valid_fields:
        if field in data:
            updates.append(f'{field} = ?')
            params.append(data[field])
    
    if updates:
        params.append(session['user_id'])
        db.execute(f'UPDATE accessibility_prefs SET {", ".join(updates)} WHERE user_id = ?',
                  tuple(params))
    
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DREAMS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/dreams', methods=['GET'])
@require_subscription
def get_dreams():
    """Get user's dreams"""
    dreams = db.execute('SELECT * FROM dreams WHERE user_id = ? ORDER BY created_at DESC LIMIT 50',
                       (session['user_id'],))
    return jsonify([dict(d) for d in dreams or []])


@app.route('/api/dreams', methods=['POST'])
@require_subscription
def create_dream():
    """Record a new dream"""
    data = request.get_json()
    dream_id = secrets.token_hex(8)
    now = datetime.now(timezone.utc).isoformat()
    
    db.execute('''INSERT INTO dreams (id, user_id, dream_text, emotion, lucidity_level, 
        recurring, symbols, dream_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (dream_id, session['user_id'], data.get('dream_text', ''),
         data.get('emotion', 'neutral'), data.get('lucidity_level', 0),
         data.get('recurring', 0), json.dumps(data.get('symbols', [])),
         data.get('dream_date', now[:10]), now))
    
    # Create dream orb in fractal universe
    result = organism.process_action('record_dream', 0.5, 0.9, 0.8)
    
    return jsonify({
        'success': True,
        'dream_id': dream_id,
        'karma_earned': result['karma_earned']
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# QUICK ACTIONS (Zero-Typing Interface)
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/quick-actions', methods=['GET'])
@require_subscription
def get_quick_actions():
    """Get user's quick action buttons"""
    actions = db.execute('''SELECT * FROM quick_actions WHERE user_id = ? AND is_visible = 1 
        ORDER BY display_order''', (session['user_id'],))
    return jsonify([dict(a) for a in actions or []])


@app.route('/api/quick-actions/execute', methods=['POST'])
@require_subscription
def execute_quick_action():
    """Execute a quick action (zero-typing interaction)"""
    data = request.get_json()
    action_id = data.get('action_id')
    
    action = db.execute_one('SELECT * FROM quick_actions WHERE id = ? AND user_id = ?',
                           (action_id, session['user_id']))
    if not action:
        return jsonify({'error': 'Action not found'}), 404
    
    action_data = json.loads(action['action_data'] or '{}')
    action_type = action['action_type']
    
    # Process different action types
    if action_type == 'mood_quick':
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        db.execute('''INSERT INTO daily_entries (id, user_id, date, mood_level, energy_level, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET 
            mood_level = ?, energy_level = ?, last_updated = ?''',
            (secrets.token_hex(8), session['user_id'], today, 
             action_data.get('mood', 50), action_data.get('energy', 50),
             datetime.now(timezone.utc).isoformat(),
             action_data.get('mood', 50), action_data.get('energy', 50),
             datetime.now(timezone.utc).isoformat()))
        result = organism.process_action('quick_checkin', 0.3, 0.9, 0.9)
        
    elif action_type == 'spoons_quick':
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        db.execute('''INSERT INTO daily_entries (id, user_id, date, spoons_available, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET 
            spoons_available = ?, last_updated = ?''',
            (secrets.token_hex(8), session['user_id'], today,
             action_data.get('spoons', 12), datetime.now(timezone.utc).isoformat(),
             action_data.get('spoons', 12), datetime.now(timezone.utc).isoformat()))
        result = organism.process_action('quick_spoons', 0.2, 0.9, 0.9)
        
    elif action_type == 'complete_quick':
        result = organism.process_action('quick_complete', 0.5, 0.95, 0.9)
        
    elif action_type == 'break_quick':
        result = organism.process_action('take_break', 0.3, 0.8, 0.95)
        
    else:
        result = organism.process_action(action_type, 0.3, 0.8, 0.8)
    
    return jsonify({
        'success': True,
        'action': action['label'],
        'karma_earned': result['karma_earned']
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AUTO-SAVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/state/save', methods=['POST'])
@require_auth
def save_state():
    """Auto-save user state"""
    data = request.get_json()
    auto_save_state(
        session['user_id'],
        panel=data.get('panel'),
        camera=data.get('camera')
    )
    return jsonify({'success': True})


@app.route('/api/state/restore', methods=['GET'])
@require_auth
def restore_state():
    """Restore user state from last session"""
    state = db.execute_one('SELECT * FROM user_state WHERE user_id = ?', (session['user_id'],))
    if state:
        return jsonify(dict(state))
    return jsonify({})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# INSIGHTS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/insights', methods=['GET'])
@require_subscription
def get_insights():
    """Get ML-generated insights for user"""
    insights = db.execute('''SELECT * FROM insights WHERE user_id = ? AND acknowledged = 0
        ORDER BY created_at DESC LIMIT 10''', (session['user_id'],))
    return jsonify([dict(i) for i in insights or []])


@app.route('/api/insights/<insight_id>/acknowledge', methods=['POST'])
@require_subscription
def acknowledge_insight(insight_id):
    """Mark insight as acknowledged"""
    data = request.get_json()
    db.execute('''UPDATE insights SET acknowledged = 1, helpful_rating = ? WHERE id = ? AND user_id = ?''',
              (data.get('helpful_rating'), insight_id, session['user_id']))
    return jsonify({'success': True})


@app.route('/api/organism/state', methods=['GET'])
def get_organism_state():
    """Get organism state - allows demo mode for non-logged-in users"""
    if 'user_id' in session:
        # Check subscription for logged-in users
        sub_status = check_subscription_status(session['user_id'])
        if not sub_status['valid']:
            return jsonify({
                'error': 'Subscription required',
                'subscription_status': sub_status['status'],
                'subscribe_url': STRIPE_MONTHLY_LINK
            }), 402
        # Auto-save that user is active
        auto_save_state(session['user_id'])
    
    # Return organism state (works for demo and logged-in users)
    state = organism.get_state()
    state['is_demo'] = 'user_id' not in session
    return jsonify(state)


@app.route('/api/organism/visualization', methods=['GET'])
def get_visualization():
    """Get visualization - allows demo mode"""
    if 'user_id' in session:
        sub_status = check_subscription_status(session['user_id'])
        if not sub_status['valid']:
            return jsonify({'error': 'Subscription required'}), 402
    
    viz = organism.swarm.get_visualization_data()
    viz['is_demo'] = 'user_id' not in session
    return jsonify(viz)


@app.route('/api/organism/action', methods=['POST'])
@require_subscription
def process_action():
    data = request.get_json()
    result = organism.process_action(
        data.get('action_type', 'neutral'),
        data.get('magnitude', 1.0),
        data.get('intention', 0.8),
        data.get('awareness', 0.7),
        data.get('linked_data')
    )
    
    db.execute('''INSERT INTO karma_history (id, user_id, action_type, karma_value, meaning, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (secrets.token_hex(8), session['user_id'], data.get('action_type', 'neutral'),
         result['karma_earned'], result.get('meaning', ''),
         datetime.now(timezone.utc).isoformat()))
    
    return jsonify(result)


@app.route('/api/animation/generate', methods=['POST'])
@require_subscription
def generate_animation():
    """Generate animation from organism state"""
    data = request.get_json()
    duration = min(1200, max(1, data.get('duration_seconds', 30)))  # 1s to 20min
    
    result = organism.generate_animation(duration)
    
    # Log animation request
    db.execute('''INSERT INTO animations (id, user_id, title, duration_seconds, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (secrets.token_hex(8), session['user_id'], data.get('title', 'Untitled'),
         duration, datetime.now(timezone.utc).isoformat(), 'generating'))
    
    return jsonify(result)


@app.route('/api/animation/frame/<int:frame_num>', methods=['GET'])
@require_auth
def get_animation_frame(frame_num):
    """Get single animation frame (lightweight fractal render)"""
    duration = float(request.args.get('duration', 30))
    total_frames = int(duration * organism.animation_engine.fps)
    
    if frame_num >= total_frames:
        return jsonify({'error': 'Frame out of range'}), 400
    
    # Lightweight frame generation - smaller resolution, simpler math
    t = frame_num / organism.animation_engine.fps
    width, height = 200, 150  # Small for speed
    
    # Create frame using bloom expansion (faster than manifold)
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    # Use golden harmonic for animation
    harmonic = organism.math.golden_harmonic_fold(t)
    emotion_mod = organism.math.emotional_harmonic(t, organism.swarm.collective_emotion)
    
    for y in range(height):
        for x in range(width):
            # Simple fractal coloring
            nx = (x / width - 0.5) * 3
            ny = (y / height - 0.5) * 2
            
            # Quick Mandelbrot iteration
            zx, zy = 0, 0
            cx = nx + harmonic * 0.1
            cy = ny + emotion_mod * 0.05
            
            for i in range(20):
                zx_new = zx*zx - zy*zy + cx
                zy = 2*zx*zy + cy
                zx = zx_new
                if zx*zx + zy*zy > 4:
                    break
            
            # Color based on iteration
            hue = (i / 20 * 360 + t * 30) % 360
            sat = 0.8
            val = 0.3 + 0.7 * (i / 20)
            r, g, b = organism.math._hsv_to_rgb(hue, sat, val)
            pixels[x, y] = (int(r*255), int(g*255), int(b*255))
    
    # Convert to base64 PNG
    buffer = BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    return jsonify({
        'frame': frame_num,
        'total_frames': total_frames,
        'time': t,
        'harmonic': harmonic,
        'image': base64.b64encode(buffer.getvalue()).decode('utf-8')
    })


@app.route('/api/math/golden-harmonic', methods=['GET'])
@require_auth
def get_golden_harmonic():
    """Get golden-harmonic folding field value"""
    t = float(request.args.get('t', time.time() % 100))
    value = organism.math.golden_harmonic_fold(t)
    return jsonify({'t': t, 'value': value, 'phi': PHI})


@app.route('/api/math/emotional-manifold', methods=['GET'])
@require_auth
def get_emotional_manifold():
    """Get emotional manifold value"""
    x = float(request.args.get('x', 0.5))
    y = float(request.args.get('y', 0.5))
    t = float(request.args.get('t', time.time() % 100))
    emotion = request.args.get('emotion', 'neutral')
    
    value = organism.math.emotional_manifold(x, y, t, emotion)
    return jsonify({'x': x, 'y': y, 't': t, 'emotion': emotion, 'value': value})


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
    
    orb = organism.swarm.spawn_orb(
        cell_type=CellType.GOAL, karma=0.5, emotion='hope',
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
        cell_type=CellType.HABIT, karma=0.3, emotion='calm',
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
    
    if action == 'feed': hunger, happiness = min(100, hunger + 30), min(100, happiness + 10)
    elif action == 'play': energy, happiness = max(0, energy - 20), min(100, happiness + 25)
    elif action == 'rest': energy = min(100, energy + 40)
    elif action == 'pet': happiness = min(100, happiness + 15)
    
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
    return jsonify({
        'patterns': organism.swarm.patterns_detected,
        'math_foundations': {
            'golden_harmonic': organism.math.golden_harmonic_fold(organism.uptime),
            'collective_emotion': organism.swarm.collective_emotion
        }
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
    """Generate 2D fractal using mathematical foundations"""
    t = organism.uptime % 100
    
    # Use bloom expansion field
    width, height = 400, 400
    bloom = organism.math.bloom_expansion_field(width, height, t, zoom=1.5)
    
    # Convert to image
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    max_val = np.max(bloom)
    for y in range(height):
        for x in range(width):
            val = bloom[y, x] / max(1, max_val)
            hue = (val * PHI * 60 + t * 20) % 360
            r, g, b = organism.math._hsv_to_rgb(hue, 0.8, 0.3 + 0.7 * val)
            pixels[x, y] = (int(r*255), int(g*255), int(b*255))
    
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
    t = organism.uptime
    harmonic = organism.math.golden_harmonic_fold(t)
    
    return jsonify({
        'type': 'mandelbulb',
        'power': 8.0 + harmonic * 2,
        'iterations': 12,
        'zoom': 1.5 + organism.harmony * 0.5,
        'rotation': [t * 0.1, harmonic * math.pi, organism.harmony * math.pi * 0.5],
        'color_scheme': {
            'base': [0.27, 0.51, 0.71],
            'accent': [1.0, 0.72, 0.3],
            'glow': [0.5, 0.8, 1.0]
        },
        'karma': organism.karma_engine.field_potential,
        'harmony': organism.harmony,
        'math': {
            'golden_harmonic': harmonic,
            'emotional_manifold': organism.math.emotional_manifold(
                0.5, 0.5, t, organism.swarm.collective_emotion
            )
        }
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '12.1',
        'harmony': organism.harmony,
        'orbs': len(organism.swarm.orbs),
        'math_foundations': 'active',
        'ollama': organism.ai.available,
        'ml': HAS_SKLEARN,
        'mayan': organism.mayan.get_today_summary()['greeting'],
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HTML INTERFACE (Preserved from v12, enhanced)
# ═══════════════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root { 
            --phi: 1.618; --bg: #0a0a12; --gold: #d4af37; --blue: #4a90a4; 
            --success: #22c55e; --warning: #f59e0b; --danger: #ef4444;
            --font-main: 'Lexend', system-ui, sans-serif;
            --font-size-base: 16px;
        }
        /* Dyslexia-friendly mode */
        body.dyslexia-font { --font-main: 'OpenDyslexic', 'Lexend', sans-serif; letter-spacing: 0.05em; word-spacing: 0.1em; }
        body.large-text { --font-size-base: 20px; }
        body.high-contrast { --bg: #000; --gold: #ffd700; --blue: #00bfff; }
        body.reduced-motion * { animation: none !important; transition: none !important; }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: var(--font-main); font-size: var(--font-size-base); background: var(--bg); color: #e8e8e8; overflow: hidden; height: 100vh; line-height: 1.6; }
        #universe { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 1; }
        
        /* Quick Actions Bar - Zero Typing */
        .quick-bar { position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%); z-index: 150; display: flex; gap: 8px; background: rgba(15,15,25,0.95); padding: 10px 15px; border-radius: 25px; border: 1px solid rgba(212,175,55,0.3); }
        .quick-bar.hidden { display: none; }
        .quick-btn { min-width: 50px; height: 50px; border-radius: 50%; border: 2px solid rgba(74,144,164,0.5); background: rgba(74,144,164,0.2); color: white; font-size: 1.5em; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .quick-btn:hover { transform: scale(1.1); border-color: var(--gold); background: rgba(212,175,55,0.3); }
        .quick-btn:active { transform: scale(0.95); }
        
        /* Paywall Modal */
        .paywall { position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 2000; display: none; align-items: center; justify-content: center; }
        .paywall.show { display: flex; }
        .paywall-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 2px solid var(--gold); border-radius: 24px; padding: 40px; max-width: 450px; text-align: center; }
        .paywall-card h2 { color: var(--gold); font-size: 1.8em; margin-bottom: 15px; }
        .paywall-card p { color: #aaa; margin-bottom: 20px; line-height: 1.7; }
        .paywall-card .price { font-size: 2.5em; color: white; margin: 20px 0; }
        .paywall-card .price span { font-size: 0.4em; color: #888; }
        .paywall-card .features { text-align: left; margin: 25px 0; }
        .paywall-card .features li { padding: 8px 0; color: #ccc; list-style: none; }
        .paywall-card .features li::before { content: '✓ '; color: var(--success); }
        .paywall-btn { width: 100%; padding: 16px; background: linear-gradient(135deg, var(--gold), #c49b30); border: none; border-radius: 12px; color: #000; font-size: 1.1em; font-weight: 600; cursor: pointer; margin-top: 15px; }
        .trial-badge { background: rgba(34,197,94,0.2); color: var(--success); padding: 8px 16px; border-radius: 20px; font-size: 0.85em; margin-bottom: 20px; display: inline-block; }
        
        /* Subscription Status */
        .sub-status { position: fixed; top: 15px; left: 80px; z-index: 100; background: rgba(15,15,25,0.9); border-radius: 20px; padding: 6px 14px; font-size: 0.8em; }
        .sub-status.trial { border: 1px solid var(--warning); color: var(--warning); }
        .sub-status.active { border: 1px solid var(--success); color: var(--success); }
        .sub-status.expired { border: 1px solid var(--danger); color: var(--danger); }
        
        .hamburger { position: fixed; top: 20px; left: 20px; z-index: 1000; width: 50px; height: 50px; background: rgba(15,15,25,0.95); border: 1px solid rgba(212,175,55,0.3); border-radius: 12px; cursor: pointer; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 5px; }
        .hamburger span { width: 24px; height: 2px; background: var(--gold); transition: 0.3s; }
        .hamburger.open span:nth-child(1) { transform: rotate(45deg) translate(5px,5px); }
        .hamburger.open span:nth-child(2) { opacity: 0; }
        .hamburger.open span:nth-child(3) { transform: rotate(-45deg) translate(5px,-5px); }
        .nav { position: fixed; top: 0; left: -300px; width: 280px; height: 100vh; background: rgba(15,15,25,0.98); border-right: 1px solid rgba(212,175,55,0.2); z-index: 999; transition: left 0.3s; padding: 80px 15px 20px; overflow-y: auto; }
        .nav.open { left: 0; }
        .nav-section { margin-bottom: 20px; }
        .nav-section h3 { color: var(--gold); font-size: 0.7em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
        .nav-btn { display: block; width: 100%; padding: 14px; background: transparent; border: none; color: #e8e8e8; text-align: left; cursor: pointer; border-radius: 8px; margin-bottom: 4px; font-size: 1em; }
        .nav-btn:hover { background: rgba(74,144,164,0.15); }
        .stats { position: fixed; top: 15px; right: 15px; z-index: 100; display: flex; gap: 10px; flex-wrap: wrap; max-width: 400px; justify-content: flex-end; }
        .stat { background: rgba(15,15,25,0.9); border: 1px solid rgba(74,144,164,0.3); border-radius: 15px; padding: 8px 15px; font-size: 0.9em; }
        .stat .val { color: var(--gold); font-weight: 600; }
        .stat.spoons { border-color: rgba(212,175,55,0.4); }
        .enter-btn { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); z-index: 100; padding: 18px 45px; background: linear-gradient(135deg, #8b5cf6, #4a90a4); border: none; border-radius: 25px; color: white; font-size: 1.2em; cursor: pointer; box-shadow: 0 4px 30px rgba(139,92,246,0.4); }
        .enter-btn.hidden { display: none; }
        .orb-tooltip { position: fixed; background: rgba(15,15,25,0.95); border: 1px solid var(--gold); border-radius: 10px; padding: 12px; max-width: 280px; z-index: 1001; display: none; pointer-events: none; }
        .orb-tooltip.show { display: block; }
        .orb-tooltip h4 { color: var(--gold); margin-bottom: 5px; font-size: 0.9em; }
        .orb-tooltip p { color: #aaa; font-size: 0.85em; line-height: 1.4; margin-bottom: 5px; }
        .orb-tooltip .tags { color: #666; font-size: 0.75em; }
        .mayan { position: fixed; bottom: 20px; right: 20px; background: rgba(15,15,25,0.9); border: 1px solid rgba(212,175,55,0.3); border-radius: 12px; padding: 12px; z-index: 100; }
        .mayan h4 { color: var(--gold); font-size: 0.75em; margin-bottom: 5px; }
        .audio-controls { position: fixed; bottom: 20px; left: 20px; background: rgba(15,15,25,0.9); border: 1px solid rgba(139,92,246,0.3); border-radius: 12px; padding: 10px; z-index: 100; }
        .audio-controls button { background: rgba(139,92,246,0.3); border: none; color: white; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
        .audio-controls button.active { background: rgba(139,92,246,0.7); }
        .panel { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); width: calc(100% - 40px); max-width: 700px; max-height: 55vh; background: rgba(15,15,25,0.98); border: 1px solid rgba(74,144,164,0.2); border-radius: 16px; z-index: 100; display: none; overflow: hidden; }
        .panel.active { display: block; }
        .panel-head { padding: 15px 20px; border-bottom: 1px solid rgba(74,144,164,0.2); display: flex; justify-content: space-between; align-items: center; }
        .panel-head h2 { color: var(--gold); font-size: 1.1em; }
        .panel-close { background: none; border: none; color: #888; font-size: 1.5em; cursor: pointer; padding: 5px 10px; }
        .panel-body { padding: 20px; max-height: calc(55vh - 60px); overflow-y: auto; }
        .progress-bar { height: 10px; background: rgba(74,144,164,0.2); border-radius: 5px; overflow: hidden; margin-top: 8px; }
        .progress-bar .fill { height: 100%; background: linear-gradient(90deg, var(--blue), var(--gold)); border-radius: 5px; transition: width 0.3s; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #888; font-size: 0.9em; }
        .form-group input { width: 100%; padding: 10px; background: rgba(74,144,164,0.1); border: 1px solid rgba(74,144,164,0.3); border-radius: 8px; color: #e8e8e8; }
        .btn { padding: 10px 20px; background: linear-gradient(135deg, var(--blue), #357a8a); border: none; border-radius: 8px; color: white; cursor: pointer; }
        .btn-gold { background: linear-gradient(135deg, var(--gold), #c49b30); }
        .card { background: rgba(74,144,164,0.1); border-radius: 10px; padding: 12px; margin-bottom: 10px; }
        .slider { width: 100%; }
        .toast { position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%) translateY(80px); background: var(--gold); color: #0a0a12; padding: 10px 20px; border-radius: 8px; z-index: 2000; opacity: 0; transition: 0.3s; }
        .toast.show { transform: translateX(-50%) translateY(0); opacity: 1; }
    </style>
</head>
<body>
<canvas id="universe"></canvas>

<!-- Paywall Modal -->
<div class="paywall" id="paywall">
    <div class="paywall-card">
        <div class="trial-badge" id="trialBadge">🎁 7-Day Free Trial</div>
        <h2>Life Fractal Intelligence</h2>
        <p>Your neurodivergent-friendly planning companion with sacred mathematics visualization.</p>
        <div class="price">$9.99<span>/month</span></div>
        <ul class="features">
            <li>Unlimited access to fractal universe</li>
            <li>Spoon Theory energy tracking</li>
            <li>AI-powered pattern detection</li>
            <li>Dream journaling & visualization</li>
            <li>Virtual pet companion</li>
            <li>Zero-typing quick actions</li>
            <li>Binaural beats & focus tools</li>
            <li>Full accessibility options</li>
        </ul>
        <button class="paywall-btn" onclick="subscribe()">Subscribe Now</button>
        <p style="margin-top:15px;font-size:0.85em;color:#666" id="trialMsg">Your trial has ended. Subscribe to continue.</p>
    </div>
</div>

<!-- Subscription Status -->
<div class="sub-status trial" id="subStatus">Trial: <span id="daysLeft">7</span> days left</div>

<button class="hamburger" onclick="toggleNav()"><span></span><span></span><span></span></button>
<nav class="nav" id="nav">
    <div class="nav-section"><h3>Planning</h3>
        <button class="nav-btn" onclick="showPanel('dashboard')">📊 Dashboard</button>
        <button class="nav-btn" onclick="showPanel('goals')">🎯 Goals</button>
        <button class="nav-btn" onclick="showPanel('habits')">✨ Habits</button>
    </div>
    <div class="nav-section"><h3>Wellness</h3>
        <button class="nav-btn" onclick="showPanel('checkin')">💫 Check-in</button>
        <button class="nav-btn" onclick="showPanel('dreams')">💭 Dreams</button>
    </div>
    <div class="nav-section"><h3>Companions</h3>
        <button class="nav-btn" onclick="showPanel('pet')">🐱 Pet</button>
    </div>
    <div class="nav-section"><h3>Insights</h3>
        <button class="nav-btn" onclick="showPanel('patterns')">🧠 ML Patterns</button>
        <button class="nav-btn" onclick="showPanel('animation')">🎬 Animation</button>
    </div>
    <div class="nav-section"><h3>Settings</h3>
        <button class="nav-btn" onclick="showPanel('accessibility')">♿ Accessibility</button>
        <button class="nav-btn" onclick="showPanel('account')">👤 Account</button>
    </div>
</nav>

<!-- Quick Actions Bar (Zero-Typing) -->
<div class="quick-bar" id="quickBar">
    <button class="quick-btn" onclick="quickAction('good')" title="Good day">😊</button>
    <button class="quick-btn" onclick="quickAction('okay')" title="Okay day">😐</button>
    <button class="quick-btn" onclick="quickAction('hard')" title="Hard day">😔</button>
    <button class="quick-btn" onclick="quickAction('done')" title="Task done">✅</button>
    <button class="quick-btn" onclick="quickAction('break')" title="Need break">🧘</button>
    <button class="quick-btn" onclick="quickAction('tired')" title="Low energy">💤</button>
</div>

<div class="stats">
    <div class="stat">⚖️ <span class="val" id="karma">0</span></div>
    <div class="stat">🔮 <span class="val" id="harmony">1.00</span></div>
    <div class="stat">🧬 <span class="val" id="orbs">0</span></div>
    <div class="stat spoons">🥄 <span class="val" id="spoons">12</span>/<span id="spoonsTotal">12</span></div>
</div>
<button class="enter-btn" id="enterBtn" onclick="enterFractal()">🌀 Enter the Fractal</button>
<div class="orb-tooltip" id="orbTooltip"><h4 id="ttType">STEM</h4><p id="ttMeaning">The seed of infinite possibility...</p><div class="tags" id="ttTags">stem, nascent</div></div>
<div class="mayan"><h4>📅 Mayan</h4><div id="mayanKin">Loading...</div></div>
<div class="audio-controls"><button id="binauralBtn" onclick="toggleBinaural()">🎧 Binaural</button></div>
<div class="panel" id="dashboard-panel"><div class="panel-head"><h2>📊 Dashboard</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body" id="dashContent"></div></div>
<div class="panel" id="goals-panel"><div class="panel-head"><h2>🎯 Goals</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body"><div class="form-group"><input id="goalTitle" placeholder="New goal..."></div><button class="btn btn-gold" onclick="createGoal()">Create</button><div id="goalsList" style="margin-top:15px;"></div></div></div>
<div class="panel" id="habits-panel"><div class="panel-head"><h2>✨ Habits</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body"><div class="form-group"><input id="habitName" placeholder="New habit..."></div><button class="btn btn-gold" onclick="createHabit()">Add</button><div id="habitsList" style="margin-top:15px;"></div></div></div>
<div class="panel" id="checkin-panel"><div class="panel-head"><h2>💫 Wellness Check-in</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body">
    <div class="form-group"><label>🔋 Energy: <span id="eVal">50</span></label><input type="range" class="slider" id="energy" value="50" oninput="document.getElementById('eVal').textContent=this.value"></div>
    <div class="form-group"><label>😊 Mood: <span id="mVal">50</span></label><input type="range" class="slider" id="mood" value="50" oninput="document.getElementById('mVal').textContent=this.value"></div>
    <div class="form-group"><label>🎯 Focus: <span id="fVal">50</span></label><input type="range" class="slider" id="focus" value="50" oninput="document.getElementById('fVal').textContent=this.value"></div>
    <div class="form-group"><label>😰 Stress: <span id="sVal">50</span></label><input type="range" class="slider" id="stress" value="50" oninput="document.getElementById('sVal').textContent=this.value"></div>
    <div class="form-group"><label>🥄 Spoons Available: <span id="spVal">12</span></label><input type="range" class="slider" id="spoonsInput" min="0" max="20" value="12" oninput="document.getElementById('spVal').textContent=this.value"></div>
    <button class="btn btn-gold" onclick="submitCheckin()">Submit Check-in</button>
</div></div>
<div class="panel" id="pet-panel"><div class="panel-head"><h2>🐱 Pet Companion</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body" style="text-align:center;"><div style="font-size:3em;" id="petEmoji">🐱</div><div id="petStats"></div><div style="display:flex;gap:8px;justify-content:center;margin-top:15px;"><button class="btn" onclick="petAction('feed')">🍖 Feed</button><button class="btn" onclick="petAction('play')">🎾 Play</button><button class="btn" onclick="petAction('pet')">🤗 Pet</button><button class="btn" onclick="petAction('rest')">😴 Rest</button></div></div></div>
<div class="panel" id="dreams-panel"><div class="panel-head"><h2>💭 Dreams & Visions</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body"><div class="form-group"><textarea id="dreamText" placeholder="Describe your dream or vision..." style="width:100%;height:80px;background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);border-radius:8px;color:#e8e8e8;padding:10px;resize:none;"></textarea></div><button class="btn" style="background:linear-gradient(135deg,#8b5cf6,#a855f7);" onclick="saveDream()">Save Dream</button><div id="dreamsList" style="margin-top:15px;"></div></div></div>
<div class="panel" id="patterns-panel"><div class="panel-head"><h2>🧠 ML Patterns</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body" id="patternsContent"></div></div>
<div class="panel" id="animation-panel"><div class="panel-head"><h2>🎬 Animation</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body"><p>Generate animations from your living fractal universe.</p><div class="form-group"><label>Duration (seconds)</label><input type="number" id="animDuration" value="30" min="1" max="1200"></div><button class="btn btn-gold" onclick="generateAnimation()">Generate Preview</button><div id="animResult"></div></div></div>
<div class="panel" id="accessibility-panel"><div class="panel-head"><h2>♿ Accessibility</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body">
    <p style="color:#888;margin-bottom:15px;">Customize for your neurodivergent needs</p>
    <div class="card"><b>👁️ Visual</b>
        <div class="form-group" style="margin-top:10px;"><label><input type="checkbox" id="accDyslexia" onchange="updateAccessibility()"> Dyslexia-friendly font</label></div>
        <div class="form-group"><label><input type="checkbox" id="accLargeText" onchange="updateAccessibility()"> Large text</label></div>
        <div class="form-group"><label><input type="checkbox" id="accHighContrast" onchange="updateAccessibility()"> High contrast</label></div>
        <div class="form-group"><label><input type="checkbox" id="accReducedMotion" onchange="updateAccessibility()"> Reduced motion</label></div>
    </div>
    <div class="card"><b>🧠 Cognitive</b>
        <div class="form-group" style="margin-top:10px;"><label><input type="checkbox" id="accSimplified" onchange="updateAccessibility()"> Simplified UI</label></div>
        <div class="form-group"><label><input type="checkbox" id="accBreakReminders" onchange="updateAccessibility()" checked> Break reminders</label></div>
        <div class="form-group"><label>Break interval: <span id="breakVal">25</span> min</label><input type="range" class="slider" id="accBreakInterval" min="10" max="60" value="25" oninput="document.getElementById('breakVal').textContent=this.value"></div>
    </div>
    <div class="card"><b>🌀 Fractal</b>
        <div class="form-group" style="margin-top:10px;"><label>Complexity:</label>
            <select id="accFractalComplexity" style="width:100%;padding:8px;background:#1a1a2e;border:1px solid #333;color:#e8e8e8;border-radius:6px;">
                <option value="low">Low (calmer)</option>
                <option value="medium" selected>Medium</option>
                <option value="high">High (detailed)</option>
            </select>
        </div>
        <div class="form-group"><label>Animation speed:</label>
            <select id="accAnimSpeed" style="width:100%;padding:8px;background:#1a1a2e;border:1px solid #333;color:#e8e8e8;border-radius:6px;">
                <option value="slow">Slow</option>
                <option value="normal" selected>Normal</option>
                <option value="fast">Fast</option>
            </select>
        </div>
    </div>
    <button class="btn btn-gold" onclick="saveAccessibility()" style="width:100%;margin-top:10px;">Save Preferences</button>
</div></div>
<div class="panel" id="account-panel"><div class="panel-head"><h2>👤 Account</h2><button class="panel-close" onclick="closePanel()">×</button></div><div class="panel-body">
    <div class="card" id="accountInfo"><b>Loading...</b></div>
    <div class="card"><b>📊 Your Stats</b><div id="accountStats" style="margin-top:10px;color:#888;">Loading...</div></div>
    <button class="btn" onclick="logout()" style="width:100%;margin-top:15px;background:#ef4444;">Logout</button>
</div></div>
<div class="toast" id="toast"></div>
<script>
const PHI = 1.618033988749895;
const STRIPE_URL = 'https://buy.stripe.com/YOUR_STRIPE_LINK';
let scene, camera, renderer, orbs = {}, isInside = false, data = {}, raycaster, mouse, audioCtx, binauralOsc;
let subscription = {valid: true, status: 'loading', days_left: 7};
let autoSaveTimer = null;
let isLoggedIn = true;

// Check subscription on load
async function checkSubscription(){
    try {
        const r = await api('/api/subscription/status');
        if(r && !r.error){
            subscription = r;
            isLoggedIn = true;
            updateSubscriptionUI();
            if(!r.valid){
                document.getElementById('paywall').classList.add('show');
                document.getElementById('trialMsg').textContent = r.reason || 'Subscribe to continue';
            }
        } else if(r && r.error === 'Authentication required') {
            // Not logged in - demo mode
            isLoggedIn = false;
            subscription = {valid: true, status: 'demo'};
            updateSubscriptionUI();
            document.getElementById('quickBar').classList.add('hidden');
        }
    } catch(e) {
        console.log('Subscription check failed, assuming demo mode');
        isLoggedIn = false;
        subscription = {valid: true, status: 'demo'};
        updateSubscriptionUI();
    }
}

function updateSubscriptionUI(){
    const el = document.getElementById('subStatus');
    const badge = document.getElementById('trialBadge');
    if(subscription.status === 'demo'){
        el.className = 'sub-status trial';
        el.innerHTML = '👁️ Demo Mode';
        badge.style.display = 'none';
    } else if(subscription.status === 'trial'){
        el.className = 'sub-status trial';
        el.innerHTML = 'Trial: <span>' + (subscription.days_left || 0) + '</span> days left';
        badge.style.display = 'inline-block';
    } else if(subscription.status === 'active'){
        el.className = 'sub-status active';
        el.textContent = '✓ Subscribed';
        badge.style.display = 'none';
    } else if(subscription.status === 'exempt'){
        el.className = 'sub-status active';
        el.textContent = '✓ ' + (subscription.reason || 'Premium');
        badge.style.display = 'none';
    } else if(subscription.status === 'loading'){
        el.className = 'sub-status trial';
        el.textContent = 'Loading...';
    } else {
        el.className = 'sub-status expired';
        el.textContent = 'Trial ended';
    }
}

function subscribe(){
    window.open(STRIPE_URL, '_blank');
}

// Quick Actions (Zero-Typing)
async function quickAction(type){
    const actions = {
        'good': {mood: 75, energy: 70},
        'okay': {mood: 50, energy: 50},
        'hard': {mood: 25, energy: 30},
        'done': {action: 'complete'},
        'break': {action: 'break'},
        'tired': {energy: 20}
    };
    const d = actions[type];
    if(d.action){
        await api('/api/organism/action', {method:'POST', body:JSON.stringify({action_type: d.action, magnitude: 0.3})});
        toast(type === 'done' ? '✅ Great job!' : '🧘 Taking a break...');
    } else {
        await api('/api/wellness/checkin', {method:'POST', body:JSON.stringify(d)});
        toast('💫 Recorded!');
    }
    loadData();
}

// Accessibility
function updateAccessibility(){
    document.body.classList.toggle('dyslexia-font', document.getElementById('accDyslexia').checked);
    document.body.classList.toggle('large-text', document.getElementById('accLargeText').checked);
    document.body.classList.toggle('high-contrast', document.getElementById('accHighContrast').checked);
    document.body.classList.toggle('reduced-motion', document.getElementById('accReducedMotion').checked);
}

async function saveAccessibility(){
    const prefs = {
        dyslexia_font: document.getElementById('accDyslexia').checked ? 1 : 0,
        large_text: document.getElementById('accLargeText').checked ? 1 : 0,
        high_contrast: document.getElementById('accHighContrast').checked ? 1 : 0,
        reduced_motion: document.getElementById('accReducedMotion').checked ? 1 : 0,
        simplified_ui: document.getElementById('accSimplified').checked ? 1 : 0,
        break_reminders: document.getElementById('accBreakReminders').checked ? 1 : 0,
        break_interval_minutes: +document.getElementById('accBreakInterval').value,
        fractal_complexity: document.getElementById('accFractalComplexity').value,
        animation_speed: document.getElementById('accAnimSpeed').value
    };
    await api('/api/accessibility/preferences', {method:'PUT', body:JSON.stringify(prefs)});
    toast('✓ Preferences saved!');
}

async function loadAccessibility(){
    if(!isLoggedIn) return; // Skip in demo mode
    const r = await api('/api/accessibility/preferences');
    if(r && !r.error){
        document.getElementById('accDyslexia').checked = r.dyslexia_font;
        document.getElementById('accLargeText').checked = r.large_text;
        document.getElementById('accHighContrast').checked = r.high_contrast;
        document.getElementById('accReducedMotion').checked = r.reduced_motion;
        document.getElementById('accSimplified').checked = r.simplified_ui;
        document.getElementById('accBreakReminders').checked = r.break_reminders;
        document.getElementById('accBreakInterval').value = r.break_interval_minutes || 25;
        document.getElementById('accFractalComplexity').value = r.fractal_complexity || 'medium';
        document.getElementById('accAnimSpeed').value = r.animation_speed || 'normal';
        updateAccessibility();
    }
}

// Auto-save state (only when logged in)
function startAutoSave(){
    if(!isLoggedIn) return;
    autoSaveTimer = setInterval(async ()=>{
        if(!isLoggedIn) return;
        await api('/api/state/save', {method:'POST', body:JSON.stringify({
            camera: {x: camera.position.x, y: camera.position.y, z: camera.position.z}
        })});
    }, 30000);
}

async function logout(){
    await api('/api/auth/logout', {method:'POST'});
    location.href = '/login';
}

function initThree() {
    const c = document.getElementById('universe');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a12);
    scene.fog = new THREE.FogExp2(0x0a0a12, 0.008);
    camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.1, 2000);
    camera.position.z = 100;
    renderer = new THREE.WebGLRenderer({canvas: c, antialias: true});
    renderer.setSize(innerWidth, innerHeight);
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    const al = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(al);
    const pl = new THREE.PointLight(0xd4af37, 1, 500);
    pl.position.set(50,50,50);
    scene.add(pl);
    for(let i=0;i<3000;i++){const g=new THREE.SphereGeometry(0.3);const m=new THREE.MeshBasicMaterial({color:0xffffff});const s=new THREE.Mesh(g,m);s.position.set((Math.random()-0.5)*1500,(Math.random()-0.5)*1500,(Math.random()-0.5)*1500);scene.add(s);}
    addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
    c.addEventListener('wheel',e=>{camera.position.z+=e.deltaY*0.1;camera.position.z=Math.max(20,Math.min(200,camera.position.z));});
    c.addEventListener('mousemove',onMouseMove);
    animate();
}
function onMouseMove(e){
    if(!isInside){scene.rotation.y=(e.clientX/innerWidth-0.5)*0.3;scene.rotation.x=(e.clientY/innerHeight-0.5)*0.15;}
    mouse.x=(e.clientX/innerWidth)*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;
    checkOrbHover(e.clientX,e.clientY);
}
function checkOrbHover(mx,my){
    if(camera.position.z>80){document.getElementById('orbTooltip').classList.remove('show');return;}
    raycaster.setFromCamera(mouse,camera);
    const meshes=Object.values(orbs);
    const hits=raycaster.intersectObjects(meshes);
    const tt=document.getElementById('orbTooltip');
    if(hits.length>0){
        const o=hits[0].object.userData;
        document.getElementById('ttType').textContent=(o.type||'ORB').toUpperCase()+' #'+(o.index||0);
        document.getElementById('ttMeaning').textContent=o.meaning||'A living cell in your fractal universe';
        document.getElementById('ttTags').textContent=(o.tags||[]).join(', ')||o.state||'active';
        tt.style.left=(mx+15)+'px';tt.style.top=(my+15)+'px';tt.classList.add('show');
    }else{tt.classList.remove('show');}
}
function animate(){requestAnimationFrame(animate);Object.values(orbs).forEach(m=>{m.rotation.y+=0.002;});if(!isInside)scene.rotation.y+=0.0003;renderer.render(scene,camera);}
function updateOrbs(list){
    Object.keys(orbs).forEach(id=>{if(!list.find(o=>o.id===id)){scene.remove(orbs[id]);delete orbs[id];}});
    list.forEach(o=>{
        if(orbs[o.id]){orbs[o.id].position.set(...o.position);orbs[o.id].scale.setScalar(1+0.1*Math.sin(o.pulse_phase));}
        else{const g=new THREE.SphereGeometry(o.radius||1,24,24);const c=new THREE.Color(o.color?.[0]||0.3,o.color?.[1]||0.6,o.color?.[2]||0.9);const m=new THREE.MeshPhongMaterial({color:c,emissive:c,emissiveIntensity:o.glow||0.3,shininess:100,transparent:true,opacity:0.9});const mesh=new THREE.Mesh(g,m);mesh.position.set(...o.position);mesh.userData=o;scene.add(mesh);orbs[o.id]=mesh;}
    });
}
function toggleBinaural(){
    const btn=document.getElementById('binauralBtn');
    if(!audioCtx){audioCtx=new(window.AudioContext||window.webkitAudioContext)();}
    if(binauralOsc){binauralOsc.stop();binauralOsc=null;btn.classList.remove('active');toast('🔇 Binaural beats stopped');return;}
    const osc1=audioCtx.createOscillator(),osc2=audioCtx.createOscillator(),gain=audioCtx.createGain();
    osc1.frequency.value=432;osc2.frequency.value=432+7.83;gain.gain.value=0.1;
    osc1.connect(gain);osc2.connect(gain);gain.connect(audioCtx.destination);
    osc1.start();osc2.start();binauralOsc={stop:()=>{osc1.stop();osc2.stop();}};btn.classList.add('active');
    toast('🎧 432Hz + 7.83Hz Schumann');
}
function enterFractal(){isInside=true;document.getElementById('enterBtn').classList.add('hidden');const i=setInterval(()=>{camera.position.z-=2;if(camera.position.z<=30)clearInterval(i);},16);toast('🌀 Entering fractal universe...');}
function toggleNav(){document.querySelector('.hamburger').classList.toggle('open');document.getElementById('nav').classList.toggle('open');}
function showPanel(n){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));const p=document.getElementById(n+'-panel');if(p){p.classList.add('active');loadPanel(n);}toggleNav();}
function closePanel(){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));}
async function api(u,o={}){try{const r=await fetch(u,{...o,headers:{'Content-Type':'application/json',...o.headers}});return await r.json();}catch(e){console.error(e);return null;}}
async function loadData(){
    const d=await api('/api/organism/state');
    if(d && !d.error){
        data=d;
        document.getElementById('karma').textContent=(d.karma?.field_potential||0).toFixed(2);
        document.getElementById('harmony').textContent=(d.harmony||1).toFixed(2);
        document.getElementById('orbs').textContent=d.swarm?.total_orbs||0;
        if(d.swarm?.orbs)updateOrbs(d.swarm.orbs);
        if(d.mayan)document.getElementById('mayanKin').textContent=d.mayan.greeting;
        // Check if demo mode
        if(d.is_demo){
            isLoggedIn = false;
            document.getElementById('quickBar').classList.add('hidden');
        }
    }
    // Only load wellness data if logged in
    if(isLoggedIn){
        const w=await api('/api/wellness/today');
        if(w && !w.error){
            document.getElementById('spoons').textContent=w.spoons_available-w.spoons_used;
            document.getElementById('spoonsTotal').textContent=w.spoons_available;
        }
    }
}
async function loadPanel(n){
    if(n==='dashboard'){document.getElementById('dashContent').innerHTML='<div class="card"><b>⚖️ Karma:</b> '+(data.karma?.field_potential||0).toFixed(2)+'</div><div class="card"><b>🔮 Harmony:</b> '+(data.harmony||1).toFixed(2)+'</div><div class="card"><b>🧬 Living Orbs:</b> '+(data.swarm?.total_orbs||0)+'</div><div class="card"><b>🔢 Math:</b> Golden Harmonic '+(data.math_foundations?.golden_harmonic||0).toFixed(3)+'</div><div class="card"><b>😊 Emotion:</b> '+(data.swarm?.collective_emotion||'neutral')+'</div>';}
    if(n==='goals'){const g=await api('/api/goals');document.getElementById('goalsList').innerHTML=(g||[]).map(x=>'<div class="card"><b>'+x.title+'</b><div class="progress-bar"><div class="fill" style="width:'+(x.progress||0)+'%"></div></div><small>'+(x.progress||0)+'% complete</small></div>').join('')||'<p style=\"color:#666\">No goals yet. Create your first!</p>';}
    if(n==='habits'){const h=await api('/api/habits');document.getElementById('habitsList').innerHTML=(h||[]).map(x=>'<div class="card" style="display:flex;justify-content:space-between;align-items:center;"><span><b>'+x.name+'</b> <span style=\"color:var(--gold)\">🔥'+x.current_streak+'</span></span><button class="btn" onclick="completeHabit(\\''+x.id+'\\')">✓</button></div>').join('')||'<p style=\"color:#666\">No habits yet. Add your first!</p>';}
    if(n==='pet'){const p=await api('/api/pet/state');if(p){const emoji=p.happiness>70?'😺':p.happiness>40?'🐱':'😿';document.getElementById('petEmoji').textContent=emoji;document.getElementById('petStats').innerHTML='<div style=\"margin:10px 0\"><b>'+p.name+'</b></div>Hunger: '+Math.round(p.hunger)+'% | Energy: '+Math.round(p.energy)+'% | Happy: '+Math.round(p.happiness)+'%';}}
    if(n==='patterns'){const p=await api('/api/analytics/patterns');document.getElementById('patternsContent').innerHTML='<div class="card"><b>🔍 Detected Patterns:</b><br>'+(p?.patterns?.[0]?.patterns?.join(', ')||'Keep using app to build patterns...')+'</div><div class="card"><b>💡 Insights:</b><br>'+(p?.patterns?.[0]?.insights?.join('<br>')||'Building insights from your activity...')+'</div><div class="card"><b>📊 Type Distribution:</b><br>'+JSON.stringify(p?.patterns?.[0]?.type_distribution||{})+'</div>';}
    if(n==='accessibility'){await loadAccessibility();}
    if(n==='account'){const s=await api('/api/subscription/status');if(s){document.getElementById('accountInfo').innerHTML='<b>'+s.email+'</b><br>Status: <span style="color:'+(s.valid?'#22c55e':'#ef4444')+'">'+(s.status||'unknown')+'</span>'+(s.days_left?' ('+s.days_left+' days left)':'');document.getElementById('accountStats').innerHTML='Member since: '+new Date(s.member_since).toLocaleDateString();}}
    if(n==='dreams'){document.getElementById('dreamsList').innerHTML='<div class="card" style="border-color:rgba(139,92,246,0.3)"><b>🌙 Dream Journal</b><br><small style="color:#888">Your dreams create DREAM type orbs in your fractal universe.</small></div>';}
}
async function createGoal(){const t=document.getElementById('goalTitle').value;if(!t)return;const r=await api('/api/goals',{method:'POST',body:JSON.stringify({title:t})});if(r){toast('🎯 +'+r.karma_earned?.toFixed(2)+' karma');document.getElementById('goalTitle').value='';loadPanel('goals');loadData();}}
async function createHabit(){const n=document.getElementById('habitName').value;if(!n)return;await api('/api/habits',{method:'POST',body:JSON.stringify({name:n})});toast('✨ Habit created!');document.getElementById('habitName').value='';loadPanel('habits');}
async function completeHabit(id){const r=await api('/api/habits/'+id+'/complete',{method:'POST'});if(r){toast('✅ +'+r.karma_earned?.toFixed(2)+' karma'+(r.fibonacci_bonus?' 🌟 Fibonacci!':''));loadPanel('habits');loadData();}}
async function submitCheckin(){const d={energy:+document.getElementById('energy').value,mood:+document.getElementById('mood').value,focus:+document.getElementById('focus').value,stress:+document.getElementById('stress').value,spoons_available:+document.getElementById('spoonsInput').value};const r=await api('/api/wellness/checkin',{method:'POST',body:JSON.stringify(d)});if(r){toast('💫 +'+r.karma_earned?.toFixed(2)+' karma');closePanel();loadData();}}
async function petAction(a){const r=await api('/api/pet/interact',{method:'POST',body:JSON.stringify({action:a})});if(r){toast('🐱 '+r.emotion+'!');loadPanel('pet');loadData();}}
async function saveDream(){const t=document.getElementById('dreamText').value;if(!t)return;const r=await api('/api/organism/action',{method:'POST',body:JSON.stringify({action_type:'record_dream',magnitude:0.5,intention:0.9,awareness:0.8,linked_data:{dream:t}})});if(r){toast('💭 Dream saved! +'+r.karma_earned?.toFixed(2)+' karma');document.getElementById('dreamText').value='';loadData();}}
async function generateAnimation(){const d=+document.getElementById('animDuration').value||30;const r=await api('/api/animation/generate',{method:'POST',body:JSON.stringify({duration_seconds:d})});if(r)document.getElementById('animResult').innerHTML='<div class="card" style="margin-top:15px;"><b>✅ Animation Ready</b><br>Duration: '+r.duration+'s | Frames: '+r.total_frames+'<br>Emotion: '+r.collective_emotion+' | Harmony: '+(r.harmony||0).toFixed(2)+'<br><small>'+r.message+'</small></div>';}
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3000);}
document.addEventListener('DOMContentLoaded',async ()=>{initThree();await checkSubscription();await loadAccessibility();loadData();startAutoSave();setInterval(loadData,5000);});
</script>
</body></html>'''

LOGIN_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Login</title><style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui;background:linear-gradient(135deg,#0a0a12,#1a1a2e);min-height:100vh;display:flex;align-items:center;justify-content:center}.card{background:rgba(15,15,25,0.95);padding:40px;border-radius:20px;width:100%;max-width:380px;border:1px solid rgba(212,175,55,0.2)}.logo{font-size:3em;text-align:center}h1{text-align:center;color:#d4af37;margin:15px 0 30px}.form-group{margin-bottom:18px}label{display:block;margin-bottom:6px;color:#888}input{width:100%;padding:12px;background:rgba(74,144,164,0.1);border:1px solid rgba(74,144,164,0.3);border-radius:8px;color:#e8e8e8}.btn{width:100%;padding:14px;background:linear-gradient(135deg,#8b5cf6,#4a90a4);border:none;border-radius:8px;color:white;font-size:1em;cursor:pointer}.switch{text-align:center;margin-top:18px;color:#888}.switch a{color:#d4af37}.err{background:rgba(244,67,54,0.2);color:#f44336;padding:10px;border-radius:6px;margin-bottom:15px;display:none}</style></head><body><div class="card"><div class="logo">🌀</div><h1 id="title">Login</h1><div class="err" id="err"></div><form id="form"><div class="form-group"><label>Email</label><input type="email" id="email" required></div><div class="form-group"><label>Password</label><input type="password" id="pw" required></div><div class="form-group" id="nameG" style="display:none"><label>Name</label><input id="name"></div><button type="submit" class="btn" id="sub">Login</button></form><div class="switch"><span id="swTxt">New?</span> <a href="#" onclick="toggle(event)">Register</a></div></div><script>let isL=true;function toggle(e){e.preventDefault();isL=!isL;document.getElementById('title').textContent=isL?'Login':'Register';document.getElementById('sub').textContent=isL?'Login':'Register';document.getElementById('nameG').style.display=isL?'none':'block';document.getElementById('swTxt').textContent=isL?'New?':'Have account?';}document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();const d={email:document.getElementById('email').value,password:document.getElementById('pw').value};if(!isL)d.first_name=document.getElementById('name').value;try{const r=await fetch(isL?'/api/auth/login':'/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});const j=await r.json();if(r.ok)location.href='/';else{document.getElementById('err').textContent=j.error;document.getElementById('err').style.display='block';}}catch(e){document.getElementById('err').textContent='Connection error';document.getElementById('err').style.display='block';}});</script></body></html>'''


@app.route('/')
def index():
    # Allow demo mode - don't redirect to login
    return render_template_string(MAIN_HTML)


@app.route('/login')
def login_page():
    return render_template_string(LOGIN_HTML)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE v12.4 - PRODUCTION READY")
    print("═" * 80)
    print("\n🔢 TEN MATHEMATICAL FOUNDATIONS:")
    print("   1. Golden-Harmonic Folding Field    F(t,φ) = sin(2π·t·φ)·cos(2π·t/φ)+sin(π·t²)")
    print("   2. Pareidolia Detection Field       Pattern recognition in noise")
    print("   3. Sacred Blend Energy Map          Tone density with tanh modulation")
    print("   4. Fractal Bloom Expansion          Z(n+1) = Z(n)² + C recursive structures")
    print("   5. Centralized Origami Curve        O(u,v) = sin(u·v)+cos(φ·u)·sin(φ·v)")
    print("   6. Emotionally Tuned Harmonic       H(t,e) = |sin(π·t·E[e])| + tanh(t·0.2)")
    print("   7. Fourier Sketch Synthesis         Σ(aₙ·cos(n·x) + bₙ·sin(n·y))")
    print("   8. GPU Parallel Frame Queue         Vectorized batch processing")
    print("   9. Temporal Origami Compression     Cₜ = Σ Mf·(1/φ)ⁿ fold/unfold")
    print("  10. Full-Scene Emotional Manifold    E(x,y,t) = ∇²B + H(t,e)·F(t,φ)")
    print(f"\n🤖 Ollama: {'Connected' if organism.ai.available else 'Pattern mode'}")
    print(f"📅 Today: {organism.mayan.get_today_summary()['greeting']}")
    print("\n" + "═" * 80 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
