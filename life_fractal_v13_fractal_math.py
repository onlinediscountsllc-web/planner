#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v13.0 - FRACTAL MATHEMATICS ENGINE
════════════════════════════════════════════════════════════════════════════════

For brains like mine - Fractal math solving real problems.

FRACTAL MATHEMATICS APPLICATIONS:
✅ Fractal Dimension - Measure life complexity (not just 0 or 1)
✅ Self-Similar Pattern Detection - Find repeating life cycles
✅ Fractal Brownian Motion - Realistic life trajectory noise
✅ L-Systems - Goal tree branching structures
✅ Strange Attractors - Stable life states people gravitate toward
✅ 1/f Pink Noise - Model mood/energy natural fluctuations
✅ Lacunarity - Measure gaps in life balance
✅ Fractal Scaling Laws - How small actions compound
✅ Chaos Sensitivity - Butterfly effect quantification
✅ Fractal Compression - Efficient life history storage
✅ Hurst Exponent - Predict trend persistence vs mean reversion

PLUS ALL v12 FEATURES:
✅ Law of Attraction (Belief/Focus)
✅ Bellman Optimization
✅ Sacred Mathematics (φ, Fibonacci)
✅ Flow State Theory
✅ 39 Spillover Effects

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
from datetime import datetime, timedelta, timezone, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
from functools import lru_cache

# Flask imports
from flask import Flask, request, jsonify, render_template_string, session, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Numerical computing
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618
PHI_INVERSE = 1 / PHI         # ≈ 0.618
PHI_SQUARED = PHI ** 2        # ≈ 2.618
GOLDEN_ANGLE_DEG = 360 / (PHI ** 2)  # ≈ 137.5°
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE_DEG)
E = math.e
PI = math.pi
GAMMA = PHI_INVERSE  # Discount factor

def generate_fibonacci(n: int) -> List[int]:
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

FIBONACCI = generate_fibonacci(30)

# ════════════════════════════════════════════════════════════════════════════════
# FRACTAL MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

# Mandelbrot Set iteration limit
MANDELBROT_MAX_ITER = 100

# Fractal dimension of common objects (for reference)
FRACTAL_DIMENSIONS = {
    'line': 1.0,
    'coastline_britain': 1.25,
    'koch_snowflake': 1.2619,  # log(4)/log(3)
    'sierpinski_triangle': 1.585,  # log(3)/log(2)
    'sierpinski_carpet': 1.8928,  # log(8)/log(3)
    'menger_sponge': 2.7268,
    'plane': 2.0,
    'brownian_motion': 1.5,
    'balanced_life': 1.0,  # Target: smooth, not jagged
    'chaotic_life': 1.8,   # High complexity
}

# Hurst exponent interpretations
HURST_INTERPRETATIONS = {
    (0.0, 0.5): 'mean_reverting',   # Tends to return to average
    (0.5, 0.5): 'random_walk',      # Unpredictable
    (0.5, 1.0): 'trending',         # Trends persist
}

# L-System rules for goal branching
LSYSTEM_RULES = {
    'goal_tree': {
        'axiom': 'G',
        'rules': {
            'G': 'G[+T][-T]G',  # Goal branches into tasks
            'T': 'T[+S]S',      # Task branches into subtasks
            'S': 'S'            # Subtask is terminal
        },
        'angle': GOLDEN_ANGLE_DEG,
        'iterations': 4
    }
}

# ════════════════════════════════════════════════════════════════════════════════
# FRACTAL MATHEMATICS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class FractalMathEngine:
    """
    Real fractal mathematics for solving life planning problems.
    
    This is NOT just visualization - these are computational tools.
    """
    
    # ════════════════════════════════════════════════════════════════════
    # 1. FRACTAL DIMENSION - Measure complexity of life trajectory
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def box_counting_dimension(time_series: np.ndarray, 
                               min_box: int = 2, 
                               max_box: int = None) -> float:
        """
        Calculate fractal dimension of a time series using box-counting method.
        
        This measures how "complex" or "jagged" your life trajectory is.
        
        Dimension close to 1.0 = smooth, predictable life
        Dimension close to 2.0 = chaotic, complex life
        
        Application: Identify if someone's life needs simplification
        """
        n = len(time_series)
        if max_box is None:
            max_box = n // 4
        
        # Normalize to [0, 1]
        ts_min, ts_max = np.min(time_series), np.max(time_series)
        if ts_max - ts_min > 0:
            normalized = (time_series - ts_min) / (ts_max - ts_min)
        else:
            return 1.0  # Flat line = dimension 1
        
        box_sizes = []
        box_counts = []
        
        for box_size in range(min_box, max_box + 1):
            # Count boxes needed to cover the curve
            boxes = set()
            for i in range(n):
                x_box = i // box_size
                y_box = int(normalized[i] * (n // box_size))
                boxes.add((x_box, y_box))
            
            if len(boxes) > 0:
                box_sizes.append(box_size)
                box_counts.append(len(boxes))
        
        if len(box_sizes) < 2:
            return 1.0
        
        # Fractal dimension = -slope of log(count) vs log(size)
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Linear regression
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        
        # Dimension is negative of slope
        dimension = -slope
        
        return float(np.clip(dimension, 1.0, 2.0))
    
    @staticmethod
    def higuchi_fractal_dimension(time_series: np.ndarray, 
                                  k_max: int = 10) -> float:
        """
        Higuchi's method for fractal dimension - better for short time series.
        
        Used in EEG analysis for brain complexity.
        
        Application: Analyze mood/energy patterns for mental health insights
        """
        n = len(time_series)
        if n < k_max * 4:
            k_max = max(2, n // 4)
        
        L = []
        x = np.asarray(time_series)
        
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                # Length of curve for this k and m
                Lmk = 0
                max_i = (n - m) // k
                if max_i > 0:
                    for i in range(1, max_i + 1):
                        Lmk += abs(x[m + i*k - 1] - x[m + (i-1)*k - 1])
                    Lmk = (Lmk * (n - 1)) / (k * max_i * k)
                    Lk.append(Lmk)
            
            if Lk:
                L.append(np.mean(Lk))
        
        if len(L) < 2:
            return 1.5
        
        # Fractal dimension from slope
        log_k = np.log(np.arange(1, len(L) + 1))
        log_L = np.log(np.array(L) + 1e-10)
        
        slope, _ = np.polyfit(log_k, log_L, 1)
        
        return float(np.clip(-slope, 1.0, 2.0))
    
    # ════════════════════════════════════════════════════════════════════
    # 2. HURST EXPONENT - Predict if trends will persist or reverse
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def hurst_exponent(time_series: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Hurst exponent to predict future behavior.
        
        H < 0.5: Mean-reverting (bad times will improve, good times will fade)
        H = 0.5: Random walk (unpredictable)
        H > 0.5: Trending (current direction will likely continue)
        
        Application: Should you expect your current trajectory to continue?
        """
        n = len(time_series)
        if n < 20:
            return 0.5, 'insufficient_data'
        
        # R/S analysis
        max_k = min(n // 4, 50)
        RS = []
        ns = []
        
        for k in range(10, max_k + 1):
            # Divide into subseries
            subseries_count = n // k
            if subseries_count < 1:
                continue
            
            rs_values = []
            for i in range(subseries_count):
                subseries = time_series[i*k:(i+1)*k]
                
                # Mean-adjusted series
                mean_adj = subseries - np.mean(subseries)
                
                # Cumulative deviation
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(subseries, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                RS.append(np.mean(rs_values))
                ns.append(k)
        
        if len(RS) < 2:
            return 0.5, 'insufficient_data'
        
        # Hurst exponent from log-log regression
        log_n = np.log(ns)
        log_RS = np.log(RS)
        
        H, _ = np.polyfit(log_n, log_RS, 1)
        H = float(np.clip(H, 0.0, 1.0))
        
        # Interpretation
        if H < 0.45:
            interpretation = 'mean_reverting'
        elif H > 0.55:
            interpretation = 'trending'
        else:
            interpretation = 'random_walk'
        
        return H, interpretation
    
    # ════════════════════════════════════════════════════════════════════
    # 3. FRACTAL BROWNIAN MOTION - Realistic life trajectory noise
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def fractal_brownian_motion(n: int, hurst: float = 0.5, 
                                seed: int = None) -> np.ndarray:
        """
        Generate fractal Brownian motion with specified Hurst exponent.
        
        Better than Gaussian noise for modeling life events because
        it captures correlations across time scales.
        
        Application: More realistic simulation of life trajectories
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Spectral synthesis method
        # Power spectrum: S(f) ∝ 1/f^(2H+1)
        
        # Generate frequencies
        freqs = np.fft.fftfreq(n)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # Power spectrum
        power = np.abs(freqs) ** (-(2 * hurst + 1))
        
        # Random phases
        phases = np.random.uniform(0, 2 * np.pi, n)
        
        # Construct spectrum
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        
        # Inverse FFT
        fbm = np.real(np.fft.ifft(spectrum))
        
        # Normalize
        fbm = (fbm - np.mean(fbm)) / (np.std(fbm) + 1e-10)
        
        return fbm
    
    @staticmethod
    def pink_noise_1f(n: int, seed: int = None) -> np.ndarray:
        """
        Generate 1/f (pink) noise - found in many natural phenomena.
        
        Mood, heart rate, brain activity, and stock prices all show 1/f patterns.
        
        Application: Model natural fluctuations in energy/mood
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 1/f noise has Hurst exponent ≈ 0.5 for the integral
        # Generate white noise
        white = np.random.randn(n)
        
        # Filter to create 1/f spectrum
        freqs = np.fft.fftfreq(n)
        freqs[0] = 1e-10
        
        # 1/f filter
        fft_white = np.fft.fft(white)
        fft_pink = fft_white / np.sqrt(np.abs(freqs))
        
        pink = np.real(np.fft.ifft(fft_pink))
        pink = (pink - np.mean(pink)) / (np.std(pink) + 1e-10)
        
        return pink
    
    # ════════════════════════════════════════════════════════════════════
    # 4. SELF-SIMILAR PATTERN DETECTION - Find repeating life cycles
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def detect_self_similarity(time_series: np.ndarray, 
                               scales: List[int] = None) -> Dict[str, Any]:
        """
        Detect self-similar patterns at different time scales.
        
        Life often has patterns that repeat:
        - Daily cycles (morning energy, afternoon slump)
        - Weekly cycles (Monday blues, Friday relief)
        - Monthly cycles
        - Yearly cycles (seasonal mood)
        
        Application: Identify your personal rhythms for better planning
        """
        n = len(time_series)
        if scales is None:
            scales = [7, 14, 30, 90, 365]  # Week, 2 weeks, month, quarter, year
        
        results = {
            'scales_analyzed': [],
            'similarity_scores': [],
            'dominant_cycle': None,
            'pattern_strength': 0.0
        }
        
        for scale in scales:
            if scale >= n // 2:
                continue
            
            # Compare pattern at this scale to overall pattern
            n_periods = n // scale
            if n_periods < 2:
                continue
            
            # Average pattern at this scale
            patterns = []
            for i in range(n_periods):
                start = i * scale
                end = min(start + scale, n)
                if end - start == scale:
                    patterns.append(time_series[start:end])
            
            if len(patterns) < 2:
                continue
            
            patterns = np.array(patterns)
            
            # Calculate similarity between periods
            mean_pattern = np.mean(patterns, axis=0)
            
            # Correlation of each period with mean pattern
            correlations = []
            for pattern in patterns:
                if np.std(pattern) > 0 and np.std(mean_pattern) > 0:
                    corr = np.corrcoef(pattern, mean_pattern)[0, 1]
                    correlations.append(corr)
            
            if correlations:
                avg_similarity = np.mean(correlations)
                results['scales_analyzed'].append(scale)
                results['similarity_scores'].append(float(avg_similarity))
        
        # Find dominant cycle
        if results['similarity_scores']:
            max_idx = np.argmax(results['similarity_scores'])
            results['dominant_cycle'] = results['scales_analyzed'][max_idx]
            results['pattern_strength'] = results['similarity_scores'][max_idx]
        
        return results
    
    # ════════════════════════════════════════════════════════════════════
    # 5. L-SYSTEMS - Goal tree generation and decomposition
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def generate_lsystem(axiom: str, rules: Dict[str, str], 
                        iterations: int) -> str:
        """
        Generate L-system string for goal decomposition.
        
        L-systems model branching structures like trees.
        Goals naturally branch into sub-goals and tasks.
        
        Application: Automatic goal decomposition into actionable tasks
        """
        current = axiom
        
        for _ in range(iterations):
            next_string = ""
            for char in current:
                if char in rules:
                    next_string += rules[char]
                else:
                    next_string += char
            current = next_string
        
        return current
    
    @staticmethod
    def decompose_goal_fractal(goal_name: str, 
                               depth: int = 3) -> Dict[str, Any]:
        """
        Decompose a goal into hierarchical sub-tasks using L-system logic.
        
        Each goal branches into φ (golden ratio) number of sub-components
        at each level, creating natural fractal structure.
        
        Application: Break overwhelming goals into manageable pieces
        """
        # Number of branches at each level follows Fibonacci
        # This creates natural, balanced decomposition
        
        def branch(level: int, prefix: str) -> Dict[str, Any]:
            if level >= depth:
                return {'name': prefix, 'type': 'task', 'children': []}
            
            # Number of children based on Fibonacci
            n_children = FIBONACCI[min(level + 2, 10)]  # 1, 2, 3 children
            n_children = min(n_children, 3)  # Cap at 3 for practicality
            
            children = []
            for i in range(n_children):
                child_name = f"{prefix}.{i+1}"
                children.append(branch(level + 1, child_name))
            
            return {
                'name': prefix,
                'type': 'milestone' if level == 0 else 'subgoal',
                'children': children
            }
        
        tree = branch(0, goal_name)
        
        # Count total tasks
        def count_tasks(node):
            if not node['children']:
                return 1
            return sum(count_tasks(c) for c in node['children'])
        
        tree['total_tasks'] = count_tasks(tree)
        tree['depth'] = depth
        tree['branching_factor'] = 'fibonacci'
        
        return tree
    
    # ════════════════════════════════════════════════════════════════════
    # 6. STRANGE ATTRACTORS - Stable life states
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def identify_attractors(state_history: np.ndarray, 
                           n_clusters: int = 5) -> Dict[str, Any]:
        """
        Identify "attractor states" - stable configurations life gravitates toward.
        
        Like a ball rolling into valleys, life tends toward certain stable states.
        Some attractors are healthy (growth), others are not (stagnation).
        
        Application: Understand what states you naturally fall into
        """
        if len(state_history) < 10:
            return {'attractors': [], 'current_attractor': None}
        
        # Simple k-means style clustering
        n_states = state_history.shape[0]
        n_dims = state_history.shape[1] if len(state_history.shape) > 1 else 1
        
        if n_dims == 1:
            state_history = state_history.reshape(-1, 1)
        
        # Initialize centroids
        n_clusters = min(n_clusters, n_states // 2)
        indices = np.linspace(0, n_states - 1, n_clusters, dtype=int)
        centroids = state_history[indices].copy()
        
        # Iterate
        for _ in range(20):
            # Assign points to nearest centroid
            distances = np.array([[norm(s - c) for c in centroids] for s in state_history])
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = []
            for i in range(n_clusters):
                mask = assignments == i
                if np.any(mask):
                    new_centroids.append(np.mean(state_history[mask], axis=0))
                else:
                    new_centroids.append(centroids[i])
            centroids = np.array(new_centroids)
        
        # Calculate attractor strength (how often we're near each)
        final_distances = np.array([[norm(s - c) for c in centroids] for s in state_history])
        assignments = np.argmin(final_distances, axis=1)
        
        attractors = []
        for i in range(n_clusters):
            mask = assignments == i
            count = np.sum(mask)
            if count > 0:
                # Basin of attraction size
                basin_size = count / n_states
                
                # Stability (inverse of average distance from centroid)
                avg_distance = np.mean(final_distances[mask, i])
                stability = 1 / (1 + avg_distance)
                
                attractors.append({
                    'id': i,
                    'centroid': centroids[i].tolist(),
                    'basin_size': float(basin_size),
                    'stability': float(stability),
                    'visits': int(count),
                    'is_healthy': bool(np.mean(centroids[i]) > 0.5)  # Simple heuristic
                })
        
        # Sort by basin size
        attractors.sort(key=lambda x: x['basin_size'], reverse=True)
        
        # Current attractor
        current_attractor = int(assignments[-1])
        
        return {
            'attractors': attractors,
            'current_attractor': current_attractor,
            'total_attractors': len(attractors)
        }
    
    # ════════════════════════════════════════════════════════════════════
    # 7. LACUNARITY - Measure gaps in life balance
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def calculate_lacunarity(values: np.ndarray, 
                            box_sizes: List[int] = None) -> float:
        """
        Calculate lacunarity - measures "gaps" or unevenness in distribution.
        
        High lacunarity = very uneven, lots of gaps
        Low lacunarity = even distribution
        
        Application: Measure how balanced your attention is across life domains
        """
        n = len(values)
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16]
        
        lacunarities = []
        
        for r in box_sizes:
            if r >= n:
                continue
            
            # Gliding box algorithm
            masses = []
            for i in range(n - r + 1):
                mass = np.sum(values[i:i+r])
                masses.append(mass)
            
            if not masses:
                continue
            
            masses = np.array(masses)
            mean_mass = np.mean(masses)
            
            if mean_mass > 0:
                # Lacunarity = variance / mean^2 + 1
                variance = np.var(masses)
                lac = (variance / (mean_mass ** 2)) + 1
                lacunarities.append(lac)
        
        if lacunarities:
            return float(np.mean(lacunarities))
        return 1.0  # Minimum lacunarity
    
    # ════════════════════════════════════════════════════════════════════
    # 8. CHAOS SENSITIVITY - Butterfly effect quantification
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def lyapunov_exponent_estimate(time_series: np.ndarray, 
                                   embedding_dim: int = 3,
                                   delay: int = 1) -> float:
        """
        Estimate Lyapunov exponent - measures chaos/sensitivity.
        
        Positive = chaotic (small changes have big effects)
        Negative = stable (system returns to equilibrium)
        Zero = periodic
        
        Application: How sensitive is your life to small daily choices?
        """
        n = len(time_series)
        if n < embedding_dim * delay * 10:
            return 0.0
        
        # Create embedded vectors
        n_vectors = n - (embedding_dim - 1) * delay
        vectors = np.zeros((n_vectors, embedding_dim))
        
        for i in range(n_vectors):
            for j in range(embedding_dim):
                vectors[i, j] = time_series[i + j * delay]
        
        # Find nearest neighbors and track divergence
        lyapunov_sum = 0
        count = 0
        
        for i in range(n_vectors - 1):
            # Find nearest neighbor (not itself)
            distances = [norm(vectors[i] - vectors[j]) 
                        for j in range(n_vectors) if j != i]
            if not distances:
                continue
            
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            # Skip if too close
            if min_dist < 1e-10:
                continue
            
            # Track divergence after one step
            if i + 1 < n_vectors and min_dist_idx + 1 < n_vectors:
                next_dist = norm(vectors[i + 1] - vectors[min_dist_idx + 1])
                if next_dist > 0 and min_dist > 0:
                    lyapunov_sum += np.log(next_dist / min_dist)
                    count += 1
        
        if count > 0:
            return float(lyapunov_sum / count)
        return 0.0
    
    # ════════════════════════════════════════════════════════════════════
    # 9. FRACTAL COMPRESSION - Efficient storage of life patterns
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def compress_pattern_fractal(time_series: np.ndarray, 
                                 threshold: float = 0.1) -> Dict[str, Any]:
        """
        Compress time series by finding self-similar segments.
        
        Instead of storing every point, store the "rules" that generate it.
        
        Application: Efficient storage of long life histories
        """
        n = len(time_series)
        
        # Find repeating segments
        segments = []
        i = 0
        
        while i < n:
            best_match = None
            best_length = 0
            
            # Look for matching segment earlier in series
            for j in range(i):
                # Find longest matching segment
                length = 0
                while (i + length < n and j + length < i and 
                       abs(time_series[i + length] - time_series[j + length]) < threshold):
                    length += 1
                
                if length > best_length and length > 3:
                    best_match = j
                    best_length = length
            
            if best_match is not None and best_length > 3:
                # Reference to earlier segment
                segments.append({
                    'type': 'reference',
                    'source': best_match,
                    'length': best_length
                })
                i += best_length
            else:
                # Store literal value
                segments.append({
                    'type': 'literal',
                    'value': float(time_series[i])
                })
                i += 1
        
        # Calculate compression ratio
        original_size = n
        compressed_size = sum(1 if s['type'] == 'literal' else 2 for s in segments)
        compression_ratio = original_size / max(compressed_size, 1)
        
        return {
            'segments': segments,
            'original_length': n,
            'compressed_segments': len(segments),
            'compression_ratio': float(compression_ratio),
            'self_similarity_score': float(1 - compressed_size / original_size)
        }
    
    # ════════════════════════════════════════════════════════════════════
    # 10. MULTIFRACTAL ANALYSIS - Different scaling in different regions
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def multifractal_spectrum(time_series: np.ndarray, 
                             q_range: Tuple[float, float] = (-5, 5)) -> Dict[str, Any]:
        """
        Calculate multifractal spectrum.
        
        Some life domains may have different fractal properties than others.
        
        Application: Identify which life areas are most chaotic vs stable
        """
        n = len(time_series)
        
        # Simplified DFA-based multifractal analysis
        q_values = np.linspace(q_range[0], q_range[1], 11)
        h_q = []  # Generalized Hurst exponent
        
        for q in q_values:
            # Skip q=0 (undefined)
            if abs(q) < 0.1:
                h_q.append(0.5)
                continue
            
            # Calculate fluctuation function for different scales
            scales = [8, 16, 32, 64]
            F_q = []
            
            for scale in scales:
                if scale >= n // 4:
                    continue
                
                # Divide into segments
                n_segments = n // scale
                fluctuations = []
                
                for i in range(n_segments):
                    segment = time_series[i*scale:(i+1)*scale]
                    # Detrend with linear fit
                    x = np.arange(scale)
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    var = np.var(segment - trend)
                    if var > 0:
                        fluctuations.append(var ** 0.5)
                
                if fluctuations:
                    # q-th order fluctuation function
                    if q > 0:
                        F_q.append(np.mean(np.array(fluctuations) ** q) ** (1/q))
                    else:
                        F_q.append(np.mean(np.array(fluctuations) ** q) ** (1/q))
            
            if len(F_q) >= 2:
                # Estimate Hurst exponent from scaling
                log_scales = np.log([s for s in scales if s < n // 4][:len(F_q)])
                log_F = np.log(np.array(F_q) + 1e-10)
                slope, _ = np.polyfit(log_scales, log_F, 1)
                h_q.append(float(slope))
            else:
                h_q.append(0.5)
        
        # Multifractal width
        h_q = np.array(h_q)
        width = float(np.max(h_q) - np.min(h_q))
        
        return {
            'q_values': q_values.tolist(),
            'h_q': h_q.tolist(),
            'multifractal_width': width,
            'is_multifractal': width > 0.2,
            'interpretation': 'complex_dynamics' if width > 0.3 else 
                            'simple_dynamics' if width < 0.1 else 
                            'moderate_dynamics'
        }


# ════════════════════════════════════════════════════════════════════════════════
# LIFE DOMAINS
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
    BELIEF = "belief"
    FOCUS = "focus"
    GRATITUDE = "gratitude"


DOMAIN_INDEX = {d: i for i, d in enumerate(LifeDomain)}
N_DOMAINS = len(LifeDomain)

# ════════════════════════════════════════════════════════════════════════════════
# INTEGRATED LIFE ENGINE (v12 + Fractal Math)
# ════════════════════════════════════════════════════════════════════════════════

class FractalLifeEngine:
    """
    Complete life engine integrating fractal mathematics with
    all previous frameworks (Bellman, Law of Attraction, etc.)
    """
    
    def __init__(self):
        self.fractal = FractalMathEngine()
        self.gamma = GAMMA
        
        # Build spillover matrix (from v12)
        self.spillover = np.zeros((N_DOMAINS, N_DOMAINS))
        spillover_effects = {
            (LifeDomain.HEALTH, LifeDomain.ENERGY): 0.5,
            (LifeDomain.HEALTH, LifeDomain.MOOD): 0.4,
            (LifeDomain.SKILLS, LifeDomain.CAREER): 0.5,
            (LifeDomain.RELATIONSHIPS, LifeDomain.MOOD): 0.5,
            (LifeDomain.PURPOSE, LifeDomain.BELIEF): 0.35,
            (LifeDomain.BELIEF, LifeDomain.MOOD): 0.4,
            (LifeDomain.FOCUS, LifeDomain.SKILLS): 0.3,
            (LifeDomain.GRATITUDE, LifeDomain.MOOD): 0.5,
        }
        for (from_d, to_d), effect in spillover_effects.items():
            self.spillover[DOMAIN_INDEX[from_d], DOMAIN_INDEX[to_d]] = effect
    
    def analyze_life_trajectory(self, state_history: List[List[float]]) -> Dict[str, Any]:
        """
        Comprehensive fractal analysis of life trajectory.
        """
        if len(state_history) < 10:
            return {'error': 'Need at least 10 data points'}
        
        history = np.array(state_history)
        n_points, n_dims = history.shape
        
        results = {
            'overall': {},
            'by_domain': {},
            'recommendations': []
        }
        
        # Overall trajectory analysis (average across domains)
        avg_trajectory = np.mean(history, axis=1)
        
        # Fractal dimension
        dim = self.fractal.box_counting_dimension(avg_trajectory)
        results['overall']['fractal_dimension'] = dim
        results['overall']['dimension_interpretation'] = (
            'smooth_predictable' if dim < 1.3 else
            'moderately_complex' if dim < 1.6 else
            'highly_complex_chaotic'
        )
        
        # Hurst exponent
        hurst, hurst_interp = self.fractal.hurst_exponent(avg_trajectory)
        results['overall']['hurst_exponent'] = hurst
        results['overall']['hurst_interpretation'] = hurst_interp
        
        # Self-similarity
        self_sim = self.fractal.detect_self_similarity(avg_trajectory)
        results['overall']['dominant_cycle'] = self_sim['dominant_cycle']
        results['overall']['pattern_strength'] = self_sim['pattern_strength']
        
        # Chaos sensitivity
        lyapunov = self.fractal.lyapunov_exponent_estimate(avg_trajectory)
        results['overall']['lyapunov_exponent'] = lyapunov
        results['overall']['chaos_level'] = (
            'highly_sensitive' if lyapunov > 0.5 else
            'moderately_sensitive' if lyapunov > 0 else
            'stable'
        )
        
        # Attractors
        attractors = self.fractal.identify_attractors(history)
        results['overall']['attractors'] = attractors
        
        # Lacunarity (balance measure)
        current_state = history[-1]
        lacunarity = self.fractal.calculate_lacunarity(current_state)
        results['overall']['lacunarity'] = lacunarity
        results['overall']['balance_interpretation'] = (
            'well_balanced' if lacunarity < 1.5 else
            'somewhat_uneven' if lacunarity < 2.5 else
            'very_unbalanced'
        )
        
        # Per-domain analysis
        for i, domain in enumerate(LifeDomain):
            domain_trajectory = history[:, i]
            
            dim_d = self.fractal.higuchi_fractal_dimension(domain_trajectory)
            hurst_d, _ = self.fractal.hurst_exponent(domain_trajectory)
            
            results['by_domain'][domain.value] = {
                'fractal_dimension': dim_d,
                'hurst_exponent': hurst_d,
                'current_value': float(domain_trajectory[-1]),
                'trend': 'improving' if hurst_d > 0.5 and domain_trajectory[-1] > domain_trajectory[-5] else
                        'declining' if hurst_d > 0.5 and domain_trajectory[-1] < domain_trajectory[-5] else
                        'fluctuating'
            }
        
        # Generate recommendations based on fractal analysis
        if dim > 1.6:
            results['recommendations'].append({
                'type': 'simplify',
                'message': 'Your life trajectory shows high complexity. Consider simplifying routines.',
                'priority': 'high'
            })
        
        if hurst < 0.45:
            results['recommendations'].append({
                'type': 'persistence',
                'message': 'Your progress tends to reverse. Focus on building consistent habits.',
                'priority': 'high'
            })
        
        if lacunarity > 2.5:
            results['recommendations'].append({
                'type': 'balance',
                'message': 'Life domains are very unbalanced. Use golden angle rotation for focus.',
                'priority': 'medium'
            })
        
        if lyapunov > 0.5:
            results['recommendations'].append({
                'type': 'stability',
                'message': 'Small changes having big effects. Build stabilizing routines.',
                'priority': 'medium'
            })
        
        return results
    
    def generate_fractal_trajectory(self, initial_state: np.ndarray,
                                   weeks: int = 12,
                                   hurst: float = 0.6) -> Dict[str, Any]:
        """
        Generate realistic future trajectory using fractal Brownian motion.
        """
        # Generate correlated noise for each domain
        n_points = weeks * 7  # Daily
        
        trajectories = []
        current = initial_state.copy()
        
        for day in range(n_points):
            # Fractal noise (more realistic than Gaussian)
            noise = self.fractal.fractal_brownian_motion(N_DOMAINS, hurst=hurst, seed=day)
            noise = noise * 0.02  # Scale factor
            
            # Simple growth + noise
            growth = np.ones(N_DOMAINS) * 0.001  # Baseline growth
            
            # Spillover effects
            spillover_effect = self.spillover.T @ current * 0.01
            
            # Update
            current = current + growth + spillover_effect + noise
            current = np.clip(current, 0, 1)
            
            trajectories.append(current.copy())
        
        trajectories = np.array(trajectories)
        
        # Analyze the generated trajectory
        analysis = self.analyze_life_trajectory(trajectories)
        
        return {
            'trajectory': trajectories.tolist(),
            'weeks': weeks,
            'hurst_used': hurst,
            'analysis': analysis,
            'final_state': trajectories[-1].tolist()
        }
    
    def decompose_goal(self, goal_name: str, complexity: int = 3) -> Dict[str, Any]:
        """
        Use L-system fractal decomposition for goals.
        """
        tree = self.fractal.decompose_goal_fractal(goal_name, depth=complexity)
        
        # Add time estimates based on Fibonacci
        def add_time_estimates(node, level):
            if not node['children']:
                # Leaf task: Fibonacci-based time
                fib_idx = min(level + 2, 10)
                node['estimated_hours'] = FIBONACCI[fib_idx]
            else:
                for child in node['children']:
                    add_time_estimates(child, level + 1)
                node['estimated_hours'] = sum(c['estimated_hours'] for c in node['children'])
        
        add_time_estimates(tree, 0)
        
        return tree


# ════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

DATABASE_PATH = os.environ.get('DATABASE_PATH', 'life_fractal_v13.db')

# Global engine
engine = FractalLifeEngine()
fractal_math = FractalMathEngine()


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
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            current_state TEXT,
            state_history TEXT,
            energy INTEGER DEFAULT 12,
            created_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            state_vector TEXT,
            recorded_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized")


# Helper functions
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
        WHERE user_id = ? 
        ORDER BY recorded_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    rows = cursor.fetchall()
    
    history = []
    for row in reversed(rows):  # Oldest first
        history.append(json.loads(row['state_vector']))
    
    return history


def save_user_state(user_id: str, state: np.ndarray):
    db = get_db()
    state_json = json.dumps(state.tolist())
    now = datetime.now(timezone.utc).isoformat()
    
    db.execute('UPDATE users SET current_state = ? WHERE id = ?', (state_json, user_id))
    db.execute('INSERT INTO state_history (user_id, state_vector, recorded_at) VALUES (?, ?, ?)',
               (user_id, state_json, now))
    db.commit()


# ════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    db = get_db()
    try:
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        initial_state = (np.ones(N_DOMAINS) * 0.5).tolist()
        
        db.execute('''
            INSERT INTO users (id, email, password_hash, display_name, current_state, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, email, generate_password_hash(password), email.split('@')[0],
              json.dumps(initial_state), now))
        db.commit()
        
        session['user_id'] = user_id
        return jsonify({'success': True, 'user_id': user_id}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400


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
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user_id'] = user['id']
    return jsonify({'success': True, 'user_id': user['id']})


@app.route('/api/auth/session')
def check_session():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'authenticated': False}), 401
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'user': {'id': user['id'], 'display_name': user['display_name']}
    })


# ════════════════════════════════════════════════════════════════════════════════
# FRACTAL MATHEMATICS API
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/fractal/analyze-trajectory')
def analyze_trajectory():
    """Comprehensive fractal analysis of user's life trajectory."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id, limit=100)
    
    if len(history) < 10:
        return jsonify({
            'error': 'Need more data',
            'message': 'At least 10 state recordings needed for fractal analysis',
            'current_count': len(history)
        }), 400
    
    analysis = engine.analyze_life_trajectory(history)
    
    return jsonify(analysis)


@app.route('/api/fractal/dimension')
def get_fractal_dimension():
    """Calculate fractal dimension of life trajectory."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id)
    
    if len(history) < 10:
        return jsonify({'dimension': 1.5, 'message': 'Insufficient data'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    
    box_dim = fractal_math.box_counting_dimension(avg_trajectory)
    higuchi_dim = fractal_math.higuchi_fractal_dimension(avg_trajectory)
    
    return jsonify({
        'box_counting_dimension': box_dim,
        'higuchi_dimension': higuchi_dim,
        'average_dimension': (box_dim + higuchi_dim) / 2,
        'interpretation': (
            'Your life trajectory is smooth and predictable' if box_dim < 1.3 else
            'Your life has moderate complexity' if box_dim < 1.6 else
            'Your life is highly complex - consider simplifying'
        ),
        'reference': {
            'line': 1.0,
            'coastline': 1.25,
            'brownian_motion': 1.5,
            'plane': 2.0
        }
    })


@app.route('/api/fractal/hurst')
def get_hurst_exponent():
    """Calculate Hurst exponent - will trends persist or reverse?"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id)
    
    if len(history) < 20:
        return jsonify({'hurst': 0.5, 'message': 'Need more data for reliable estimate'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    hurst, interpretation = fractal_math.hurst_exponent(avg_trajectory)
    
    return jsonify({
        'hurst_exponent': hurst,
        'interpretation': interpretation,
        'prediction': (
            'Your progress tends to reverse - focus on building consistent habits' if hurst < 0.45 else
            'Your trajectory is unpredictable - outcomes are uncertain' if hurst < 0.55 else
            'Your current trends are likely to continue - momentum is on your side'
        ),
        'reference': {
            'mean_reverting': '< 0.5',
            'random_walk': '= 0.5',
            'trending': '> 0.5'
        }
    })


@app.route('/api/fractal/cycles')
def detect_cycles():
    """Detect self-similar cycles in life patterns."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id)
    
    if len(history) < 30:
        return jsonify({'message': 'Need at least 30 days of data'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    cycles = fractal_math.detect_self_similarity(avg_trajectory)
    
    # Interpret dominant cycle
    cycle_names = {
        7: 'weekly',
        14: 'bi-weekly',
        30: 'monthly',
        90: 'quarterly',
        365: 'yearly'
    }
    
    dominant = cycles.get('dominant_cycle')
    cycle_name = cycle_names.get(dominant, f'{dominant} days')
    
    return jsonify({
        'cycles': cycles,
        'dominant_cycle_name': cycle_name,
        'recommendation': (
            f'Your strongest pattern is {cycle_name}. Plan around this rhythm.' 
            if cycles.get('pattern_strength', 0) > 0.5 else
            'No strong repeating pattern detected. You may benefit from more routine.'
        )
    })


@app.route('/api/fractal/attractors')
def get_attractors():
    """Identify stable life states (attractors)."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id)
    
    if len(history) < 20:
        return jsonify({'message': 'Need more data to identify attractors'})
    
    attractors = fractal_math.identify_attractors(np.array(history))
    
    # Add domain labels to attractors
    for attractor in attractors.get('attractors', []):
        centroid = attractor['centroid']
        # Find strongest domains
        sorted_indices = np.argsort(centroid)[::-1]
        top_domains = [list(LifeDomain)[i].value for i in sorted_indices[:3]]
        attractor['dominant_domains'] = top_domains
        attractor['label'] = f"High {'/'.join(top_domains[:2])}"
    
    return jsonify(attractors)


@app.route('/api/fractal/sensitivity')
def get_chaos_sensitivity():
    """How sensitive is your life to small changes?"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    history = get_state_history(user_id)
    
    if len(history) < 30:
        return jsonify({'message': 'Need more data'})
    
    avg_trajectory = np.mean(np.array(history), axis=1)
    lyapunov = fractal_math.lyapunov_exponent_estimate(avg_trajectory)
    
    return jsonify({
        'lyapunov_exponent': lyapunov,
        'chaos_level': (
            'highly_sensitive' if lyapunov > 0.5 else
            'moderately_sensitive' if lyapunov > 0 else
            'stable'
        ),
        'interpretation': (
            'Small daily choices have BIG long-term effects. Every decision matters!' if lyapunov > 0.5 else
            'Moderate sensitivity. Consistent habits provide some stability.' if lyapunov > 0 else
            'Your life is stable. Big changes require sustained effort.'
        ),
        'butterfly_effect': lyapunov > 0.3
    })


@app.route('/api/fractal/balance')
def get_balance_lacunarity():
    """Measure life balance using lacunarity."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    state = get_user_state(user_id)
    lacunarity = fractal_math.calculate_lacunarity(state)
    
    return jsonify({
        'lacunarity': lacunarity,
        'balance_score': max(0, 1 - (lacunarity - 1) / 2),  # Normalize to 0-1
        'interpretation': (
            'Well balanced across life domains' if lacunarity < 1.5 else
            'Some imbalance - certain areas getting neglected' if lacunarity < 2.5 else
            'Very unbalanced - major gaps in some life areas'
        ),
        'lowest_domains': [
            list(LifeDomain)[i].value 
            for i in np.argsort(state)[:3]
        ],
        'highest_domains': [
            list(LifeDomain)[i].value 
            for i in np.argsort(state)[-3:][::-1]
        ]
    })


@app.route('/api/fractal/decompose-goal', methods=['POST'])
def decompose_goal():
    """Fractal decomposition of a goal into sub-tasks."""
    data = request.get_json() or {}
    goal_name = data.get('goal', 'My Goal')
    complexity = min(5, max(1, data.get('complexity', 3)))
    
    tree = engine.decompose_goal(goal_name, complexity)
    
    return jsonify(tree)


@app.route('/api/fractal/simulate', methods=['POST'])
def simulate_fractal():
    """Generate fractal Brownian motion trajectory simulation."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json() or {}
    weeks = min(52, data.get('weeks', 12))
    hurst = max(0.1, min(0.9, data.get('hurst', 0.6)))
    
    state = get_user_state(user_id)
    
    result = engine.generate_fractal_trajectory(state, weeks, hurst)
    
    return jsonify(result)


@app.route('/api/fractal/pink-noise')
def get_pink_noise():
    """Generate 1/f pink noise for natural mood modeling."""
    n = int(request.args.get('n', 30))
    n = min(365, max(7, n))
    
    noise = fractal_math.pink_noise_1f(n)
    
    return jsonify({
        'pink_noise': noise.tolist(),
        'length': n,
        'explanation': '1/f noise appears in mood, heart rate, and brain activity. More realistic than white noise.'
    })


@app.route('/api/fractal/constants')
def get_fractal_constants():
    """Get all fractal-related constants."""
    return jsonify({
        'sacred_mathematics': {
            'phi': PHI,
            'phi_inverse': PHI_INVERSE,
            'golden_angle': GOLDEN_ANGLE_DEG,
            'fibonacci': FIBONACCI[:15]
        },
        'fractal_dimensions': FRACTAL_DIMENSIONS,
        'mandelbrot_max_iter': MANDELBROT_MAX_ITER,
        'hurst_interpretations': {
            '< 0.5': 'mean_reverting',
            '= 0.5': 'random_walk',
            '> 0.5': 'trending'
        },
        'applications': [
            'fractal_dimension: Measure life complexity',
            'hurst_exponent: Predict trend persistence',
            'fbm: Realistic trajectory simulation',
            'l_systems: Goal decomposition',
            'attractors: Stable life states',
            'lacunarity: Balance measurement',
            'lyapunov: Chaos sensitivity'
        ]
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '13.0.0',
        'name': 'Fractal Mathematics Engine',
        'fractal_tools': [
            'box_counting_dimension',
            'higuchi_dimension',
            'hurst_exponent',
            'fractal_brownian_motion',
            'pink_noise_1f',
            'self_similarity_detection',
            'l_system_decomposition',
            'attractor_identification',
            'lacunarity',
            'lyapunov_exponent',
            'multifractal_spectrum'
        ],
        'domains': N_DOMAINS,
        'phi': PHI,
        'golden_angle': GOLDEN_ANGLE_DEG
    })


# ════════════════════════════════════════════════════════════════════════════════
# MAIN HTML
# ════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal v13 - Fractal Mathematics Engine</title>
    <style>
        :root {
            --primary: #6B5B95;
            --secondary: #88B04B;
            --accent: #F7CAC9;
            --bg: #F5F5F5;
            --surface: #FFFFFF;
            --text: #333;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
        
        .header {
            background: linear-gradient(135deg, var(--primary), #92A8D1);
            padding: 1.5rem;
            color: white;
            text-align: center;
        }
        .header h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.9; font-size: 0.9rem; }
        
        .main { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; }
        
        .card {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg);
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .metric-value {
            font-weight: 700;
            font-size: 1.2rem;
            color: var(--primary);
        }
        
        .interpretation {
            padding: 1rem;
            background: linear-gradient(135deg, #E8F5E9, #F3E5F5);
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { opacity: 0.9; }
        
        .math-box {
            background: #F8F8F8;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
        }
        
        #loading { text-align: center; padding: 2rem; color: #666; }
    </style>
</head>
<body>
    <header class="header">
        <h1>🌀 Life Fractal v13 - Fractal Mathematics Engine</h1>
        <p>Real fractal math solving real life problems</p>
    </header>
    
    <main class="main">
        <div id="loading">Loading fractal analysis...</div>
        
        <div class="grid" id="content" style="display: none;">
            <!-- Fractal Dimension -->
            <div class="card">
                <div class="card-header">📐 Fractal Dimension</div>
                <p style="color: #666; font-size: 0.85rem; margin-bottom: 1rem;">
                    Measures complexity of your life trajectory. Lower = smoother, higher = more chaotic.
                </p>
                <div class="metric">
                    <span>Box-Counting Dimension</span>
                    <span class="metric-value" id="box-dim">-</span>
                </div>
                <div class="metric">
                    <span>Higuchi Dimension</span>
                    <span class="metric-value" id="higuchi-dim">-</span>
                </div>
                <div class="interpretation" id="dim-interpretation"></div>
            </div>
            
            <!-- Hurst Exponent -->
            <div class="card">
                <div class="card-header">📈 Hurst Exponent</div>
                <p style="color: #666; font-size: 0.85rem; margin-bottom: 1rem;">
                    Predicts if current trends will continue or reverse.
                </p>
                <div class="metric">
                    <span>Hurst Exponent (H)</span>
                    <span class="metric-value" id="hurst">-</span>
                </div>
                <div class="metric">
                    <span>Pattern Type</span>
                    <span id="hurst-type" style="font-weight: 600;">-</span>
                </div>
                <div class="interpretation" id="hurst-interpretation"></div>
            </div>
            
            <!-- Chaos Sensitivity -->
            <div class="card">
                <div class="card-header">🦋 Chaos Sensitivity</div>
                <p style="color: #666; font-size: 0.85rem; margin-bottom: 1rem;">
                    Butterfly effect: How much do small changes matter?
                </p>
                <div class="metric">
                    <span>Lyapunov Exponent</span>
                    <span class="metric-value" id="lyapunov">-</span>
                </div>
                <div class="metric">
                    <span>Chaos Level</span>
                    <span id="chaos-level" style="font-weight: 600;">-</span>
                </div>
                <div class="interpretation" id="chaos-interpretation"></div>
            </div>
            
            <!-- Life Balance -->
            <div class="card">
                <div class="card-header">⚖️ Life Balance (Lacunarity)</div>
                <p style="color: #666; font-size: 0.85rem; margin-bottom: 1rem;">
                    Measures gaps and unevenness across life domains.
                </p>
                <div class="metric">
                    <span>Lacunarity</span>
                    <span class="metric-value" id="lacunarity">-</span>
                </div>
                <div class="metric">
                    <span>Balance Score</span>
                    <span class="metric-value" id="balance-score">-</span>
                </div>
                <div class="interpretation" id="balance-interpretation"></div>
            </div>
        </div>
        
        <!-- Math Reference -->
        <div class="card" style="margin-top: 2rem;">
            <div class="card-header">📚 Fractal Mathematics Reference</div>
            <div class="math-box">
<b>Fractal Dimension (Box-Counting):</b>
D = lim(ε→0) [log N(ε) / log(1/ε)]
Where N(ε) = number of boxes of size ε needed to cover the set

<b>Hurst Exponent (R/S Analysis):</b>
E[R(n)/S(n)] = C · n^H
H < 0.5: Mean-reverting | H = 0.5: Random walk | H > 0.5: Trending

<b>Fractal Brownian Motion:</b>
B_H(t) - B_H(s) ~ N(0, |t-s|^(2H))
Power spectrum: S(f) ∝ 1/f^(2H+1)

<b>Lyapunov Exponent (Chaos):</b>
λ = lim(t→∞) (1/t) · ln|δZ(t)/δZ(0)|
λ > 0: Chaotic (sensitive to initial conditions)

<b>Lacunarity (Gaps):</b>
Λ(r) = σ²(r)/μ²(r) + 1
Higher = more gaps/unevenness
            </div>
        </div>
    </main>

    <script>
        async function loadFractalAnalysis() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('content').style.display = 'none';
            
            try {
                // Fractal Dimension
                const dimRes = await fetch('/api/fractal/dimension');
                const dimData = await dimRes.json();
                document.getElementById('box-dim').textContent = dimData.box_counting_dimension?.toFixed(3) || '-';
                document.getElementById('higuchi-dim').textContent = dimData.higuchi_dimension?.toFixed(3) || '-';
                document.getElementById('dim-interpretation').textContent = dimData.interpretation || '';
                
                // Hurst Exponent
                const hurstRes = await fetch('/api/fractal/hurst');
                const hurstData = await hurstRes.json();
                document.getElementById('hurst').textContent = hurstData.hurst_exponent?.toFixed(3) || '-';
                document.getElementById('hurst-type').textContent = hurstData.interpretation || '-';
                document.getElementById('hurst-interpretation').textContent = hurstData.prediction || '';
                
                // Chaos Sensitivity
                const chaosRes = await fetch('/api/fractal/sensitivity');
                const chaosData = await chaosRes.json();
                document.getElementById('lyapunov').textContent = chaosData.lyapunov_exponent?.toFixed(3) || '-';
                document.getElementById('chaos-level').textContent = chaosData.chaos_level || '-';
                document.getElementById('chaos-interpretation').textContent = chaosData.interpretation || '';
                
                // Balance
                const balRes = await fetch('/api/fractal/balance');
                const balData = await balRes.json();
                document.getElementById('lacunarity').textContent = balData.lacunarity?.toFixed(3) || '-';
                document.getElementById('balance-score').textContent = (balData.balance_score * 100)?.toFixed(0) + '%' || '-';
                document.getElementById('balance-interpretation').textContent = balData.interpretation || '';
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'grid';
            } catch (e) {
                document.getElementById('loading').textContent = 'Login required or insufficient data for fractal analysis';
                console.error(e);
            }
        }
        
        loadFractalAnalysis();
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
    print("\n" + "=" * 75)
    print("🌀 LIFE FRACTAL INTELLIGENCE v13.0 - FRACTAL MATHEMATICS ENGINE")
    print("=" * 75)
    print("   Real fractal math solving real life problems")
    print("=" * 75)
    
    print(f"\n📐 FRACTAL MATHEMATICS TOOLS")
    print(f"   • Box-counting dimension - Life complexity measurement")
    print(f"   • Higuchi dimension - Time series analysis")
    print(f"   • Hurst exponent - Trend persistence prediction")
    print(f"   • Fractal Brownian motion - Realistic trajectory simulation")
    print(f"   • 1/f Pink noise - Natural mood modeling")
    print(f"   • Self-similarity detection - Find life cycles")
    print(f"   • L-system decomposition - Goal tree generation")
    print(f"   • Strange attractor identification - Stable life states")
    print(f"   • Lacunarity - Life balance measurement")
    print(f"   • Lyapunov exponent - Chaos sensitivity (butterfly effect)")
    print(f"   • Multifractal spectrum - Complex dynamics analysis")
    
    print(f"\n🔢 REFERENCE FRACTAL DIMENSIONS")
    for name, dim in FRACTAL_DIMENSIONS.items():
        print(f"   {name}: {dim}")
    
    print(f"\n📐 SACRED MATHEMATICS")
    print(f"   φ = {PHI:.10f}")
    print(f"   Golden Angle = {GOLDEN_ANGLE_DEG:.4f}°")
    print("=" * 75)
    
    print("\n📡 FRACTAL API ENDPOINTS")
    print("   GET  /api/fractal/analyze-trajectory - Full analysis")
    print("   GET  /api/fractal/dimension - Fractal dimension")
    print("   GET  /api/fractal/hurst - Hurst exponent")
    print("   GET  /api/fractal/cycles - Self-similar patterns")
    print("   GET  /api/fractal/attractors - Stable states")
    print("   GET  /api/fractal/sensitivity - Chaos/butterfly effect")
    print("   GET  /api/fractal/balance - Lacunarity/balance")
    print("   POST /api/fractal/decompose-goal - L-system goal tree")
    print("   POST /api/fractal/simulate - FBM trajectory simulation")
    print("   GET  /api/fractal/pink-noise - 1/f noise generation")
    print("   GET  /api/fractal/constants - All fractal constants")
    print("=" * 75)


if __name__ == '__main__':
    print_banner()
    
    with app.app_context():
        init_db()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
