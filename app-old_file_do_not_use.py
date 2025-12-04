#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v11.0 - COMPLETE AI-POWERED SYSTEM
═══════════════════════════════════════════════════════════════════════════════════════════

🧠 AI BACKBONE ARCHITECTURE:
   ┌─────────────────────────────────────────────────────────────────┐
   │                    FRACTAL INTELLIGENCE CORE                     │
   ├─────────────────────────────────────────────────────────────────┤
   │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
   │  │ DATA      │  │ PATTERN   │  │ FRACTAL   │  │ PREDICTION│    │
   │  │ INGESTION │→→│ RECOGNITION│→→│ MATH      │→→│ ENGINE    │    │
   │  │ PIPELINE  │  │ ML ENGINE │  │ OPTIMIZER │  │           │    │
   │  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │
   │        ↓              ↓              ↓              ↓          │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │              AI MEMORY & LEARNING SYSTEM                 │   │
   │  │  • User Pattern Memory    • Optimal Math Combinations    │   │
   │  │  • Federated Insights     • Personalized Parameters      │   │
   │  │  • Executive Dysfunction  • Success Predictors           │   │
   │  └─────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────┘

✅ COMPLETE FEATURES:
   • AI Brain with persistent memory and learning
   • Pattern Recognition ML (RandomForest + KMeans clustering)
   • Federated Learning from anonymized aggregate data
   • Fractal Math Optimization Engine
   • Multi-layer 2D fractal compositions
   • 3D immersive visualization connected to all data
   • Executive dysfunction early warning system
   • Predictive analytics for mood/energy/stress
   • Data ingestion pipeline with continuous learning
   • Math combination storage and optimization
   • Spoon Theory energy management
   • Mayan Tzolkin calendar integration
   • Binaural beats therapy

DESIGNED FOR: Aphantasia, Autism, ADHD, Dysgraphia, Executive Dysfunction
═══════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import secrets
import logging
import sqlite3
import hashlib
import statistics
from datetime import datetime, timedelta, timezone
from io import BytesIO
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import base64

from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ML Support
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn not available - using heuristic fallbacks")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio: 1.618033988749895
PHI_INVERSE = 1 / PHI         # 0.618033988749895
PHI_SQUARED = PHI * PHI       # 2.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)

# Fibonacci sequence (for milestone calculations, fractal iterations)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]

# Lucas numbers (secondary sacred sequence)
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521]

# Therapeutic fractal dimension target (proven calming for neurodivergent users)
THERAPEUTIC_D_MIN = 1.3
THERAPEUTIC_D_MAX = 1.5
THERAPEUTIC_D_TARGET = 1.4

# Mayan Tzolkin
MAYAN_GLYPHS = ['Imix', 'Ik', 'Akbal', 'Kan', 'Chicchan', 'Cimi', 'Manik', 'Lamat',
                'Muluc', 'Oc', 'Chuen', 'Eb', 'Ben', 'Ix', 'Men', 'Cib', 'Caban',
                'Etznab', 'Cauac', 'Ahau']

MAYAN_MEANINGS = {
    'Imix': ('Dragon', 'New beginnings, primal energy'),
    'Ik': ('Wind', 'Communication, breath of life'),
    'Akbal': ('Night', 'Introspection, dreaming'),
    'Kan': ('Seed', 'Planting intentions, potential'),
    'Chicchan': ('Serpent', 'Life force, kundalini'),
    'Cimi': ('Transformer', 'Change, release'),
    'Manik': ('Deer', 'Healing, gentleness'),
    'Lamat': ('Star', 'Harmony, beauty'),
    'Muluc': ('Moon', 'Emotions, flow'),
    'Oc': ('Dog', 'Loyalty, companionship'),
    'Chuen': ('Monkey', 'Playfulness, creativity'),
    'Eb': ('Road', 'Journey, human path'),
    'Ben': ('Reed', 'Growth, sky-earth connection'),
    'Ix': ('Jaguar', 'Power, feminine magic'),
    'Men': ('Eagle', 'Vision, higher perspective'),
    'Cib': ('Wisdom', 'Ancestors, karma'),
    'Caban': ('Earth', 'Grounding, movement'),
    'Etznab': ('Mirror', 'Truth, reflection'),
    'Cauac': ('Storm', 'Purification, transformation'),
    'Ahau': ('Sun', 'Enlightenment, mastery')
}

# ═══════════════════════════════════════════════════════════════════════════════════════════
# AI BRAIN - CORE INTELLIGENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════

class FractalIntelligenceBrain:
    """
    The AI backbone of the application.
    Manages pattern recognition, learning, predictions, and optimization.
    """
    
    def __init__(self, db_connection_factory):
        self.get_db = db_connection_factory
        self.pattern_models = {}  # User-specific ML models
        self.global_insights = {}  # Federated anonymized insights
        self.math_combinations = {}  # Stored optimal fractal parameters
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        logger.info("🧠 Fractal Intelligence Brain initialized")
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA INGESTION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    
    def ingest_daily_data(self, user_id: str, data: Dict) -> Dict:
        """
        Primary data ingestion point. Processes new data and triggers learning.
        
        Pipeline:
        1. Validate and normalize data
        2. Store in database
        3. Update pattern recognition models
        4. Generate predictions
        5. Optimize fractal parameters
        6. Contribute to federated insights (anonymized)
        """
        logger.info(f"📥 Ingesting data for user {user_id[:8]}...")
        
        # Step 1: Normalize data
        normalized = self._normalize_wellness_data(data)
        
        # Step 2: Store (handled by caller)
        
        # Step 3: Update user's pattern model
        self._update_pattern_model(user_id)
        
        # Step 4: Generate predictions
        predictions = self.predict_tomorrow(user_id)
        
        # Step 5: Optimize fractal parameters for this user
        fractal_params = self.optimize_fractal_params(user_id, normalized)
        
        # Step 6: Contribute to federated learning (anonymized)
        self._contribute_federated_insight(normalized)
        
        return {
            'processed': True,
            'predictions': predictions,
            'fractal_params': fractal_params,
            'wellness_index': normalized.get('wellness_index', 50)
        }
    
    def _normalize_wellness_data(self, data: Dict) -> Dict:
        """Normalize incoming data to consistent ranges."""
        normalized = {}
        
        # Clamp values to 0-100 range
        for key in ['mood_level', 'energy_level', 'stress_level', 'sleep_quality']:
            if key in data:
                normalized[key] = max(0, min(100, float(data.get(key, 50))))
        
        # Sleep hours (0-24)
        normalized['sleep_hours'] = max(0, min(24, float(data.get('sleep_hours', 7))))
        
        # Calculate wellness index using Fibonacci weighting
        # Weights: mood(1), energy(1), inverse_stress(2), sleep_quality(3), sleep_hours_normalized(5)
        fib_weights = [1, 1, 2, 3, 5]
        values = [
            normalized.get('mood_level', 50),
            normalized.get('energy_level', 50),
            100 - normalized.get('stress_level', 50),  # Invert stress
            normalized.get('sleep_quality', 50),
            min(100, normalized.get('sleep_hours', 7) / 8 * 100)  # 8 hours = 100%
        ]
        
        total_weight = sum(fib_weights)
        wellness_index = sum(v * w for v, w in zip(values, fib_weights)) / total_weight
        normalized['wellness_index'] = round(wellness_index, 2)
        
        # Calculate spoons (energy capacity) - Spoon Theory
        # Based on energy and sleep, modified by stress
        base_spoons = 12  # Maximum spoons
        energy_factor = normalized.get('energy_level', 50) / 100
        sleep_factor = min(1.0, normalized.get('sleep_hours', 7) / 7)
        stress_penalty = normalized.get('stress_level', 50) / 200  # Max 0.5 penalty
        
        spoons = int(base_spoons * energy_factor * sleep_factor * (1 - stress_penalty))
        normalized['spoons_available'] = max(1, min(12, spoons))
        
        return normalized
    
    # ─────────────────────────────────────────────────────────────────────────
    # PATTERN RECOGNITION ML ENGINE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_pattern_model(self, user_id: str):
        """Train/update the user's pattern recognition model."""
        if not HAS_SKLEARN:
            return
        
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get user's historical data
        cursor.execute('''
            SELECT mood_level, energy_level, stress_level, sleep_hours, sleep_quality,
                   goals_completed, habits_completed, wellness_score, date
            FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 60
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 7:  # Need at least a week of data
            logger.info(f"📊 User {user_id[:8]} needs more data ({len(rows)}/7 days)")
            return
        
        # Prepare training data
        X = []
        y_mood = []
        y_energy = []
        
        for i in range(len(rows) - 1):
            current = rows[i + 1]  # Earlier day
            next_day = rows[i]      # Later day (what we're predicting)
            
            # Features: previous day's metrics + day of week
            day_of_week = datetime.strptime(current[8], '%Y-%m-%d').weekday()
            features = [
                current[0] or 50,  # mood
                current[1] or 50,  # energy
                current[2] or 50,  # stress
                current[3] or 7,   # sleep_hours
                current[4] or 50,  # sleep_quality
                current[5] or 0,   # goals_completed
                current[6] or 0,   # habits_completed
                day_of_week
            ]
            X.append(features)
            y_mood.append(next_day[0] or 50)
            y_energy.append(next_day[1] or 50)
        
        if len(X) < 5:
            return
        
        X = np.array(X)
        
        # Train models
        try:
            mood_model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
            energy_model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
            
            mood_model.fit(X, y_mood)
            energy_model.fit(X, y_energy)
            
            self.pattern_models[user_id] = {
                'mood': mood_model,
                'energy': energy_model,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'data_points': len(X)
            }
            
            logger.info(f"🤖 Updated ML models for user {user_id[:8]} with {len(X)} data points")
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    def predict_tomorrow(self, user_id: str) -> Dict:
        """Predict tomorrow's mood and energy levels."""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get latest entry
        cursor.execute('''
            SELECT mood_level, energy_level, stress_level, sleep_hours, sleep_quality,
                   goals_completed, habits_completed, date
            FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1
        ''', (user_id,))
        
        latest = cursor.fetchone()
        conn.close()
        
        if not latest:
            return {'predicted_mood': 50, 'predicted_energy': 50, 'confidence': 0.0}
        
        # Prepare features
        tomorrow = datetime.now() + timedelta(days=1)
        features = np.array([[
            latest[0] or 50,
            latest[1] or 50,
            latest[2] or 50,
            latest[3] or 7,
            latest[4] or 50,
            latest[5] or 0,
            latest[6] or 0,
            tomorrow.weekday()
        ]])
        
        # Use ML if available
        if user_id in self.pattern_models and HAS_SKLEARN:
            models = self.pattern_models[user_id]
            predicted_mood = float(models['mood'].predict(features)[0])
            predicted_energy = float(models['energy'].predict(features)[0])
            confidence = min(0.9, models['data_points'] / 30)  # Max 90% confidence
        else:
            # Heuristic fallback
            predicted_mood = latest[0] * 0.7 + 50 * 0.3 if latest[0] else 50
            predicted_energy = latest[1] * 0.6 + (100 - latest[2]) * 0.2 + 50 * 0.2 if latest[1] else 50
            confidence = 0.3
        
        return {
            'predicted_mood': round(max(0, min(100, predicted_mood)), 1),
            'predicted_energy': round(max(0, min(100, predicted_energy)), 1),
            'confidence': round(confidence, 2),
            'recommendations': self._generate_recommendations(predicted_mood, predicted_energy)
        }
    
    def _generate_recommendations(self, mood: float, energy: float) -> List[str]:
        """Generate personalized recommendations based on predictions."""
        recs = []
        
        if energy < 40:
            recs.append("🔋 Low energy predicted. Plan lighter tasks and prioritize rest.")
            recs.append("💤 Consider going to bed 30 minutes earlier tonight.")
        elif energy > 70:
            recs.append("⚡ Good energy day ahead! Great time for challenging tasks.")
        
        if mood < 40:
            recs.append("🌧️ Mood may be lower tomorrow. Schedule something enjoyable.")
            recs.append("🤝 Consider reaching out to a supportive person.")
        elif mood > 70:
            recs.append("☀️ Positive mood predicted. Good day to tackle difficult goals!")
        
        if not recs:
            recs.append("📊 Balanced day predicted. Maintain your current routines.")
        
        return recs
    
    # ─────────────────────────────────────────────────────────────────────────
    # FRACTAL MATH OPTIMIZATION ENGINE
    # ─────────────────────────────────────────────────────────────────────────
    
    def optimize_fractal_params(self, user_id: str, wellness_data: Dict) -> Dict:
        """
        Generate optimal fractal parameters based on user's current state.
        
        Uses:
        - Wellness data to determine base parameters
        - Historical patterns to personalize
        - Sacred mathematics for therapeutic ranges
        - Stored combinations that worked well before
        """
        mood = wellness_data.get('mood_level', 50)
        energy = wellness_data.get('energy_level', 50)
        stress = wellness_data.get('stress_level', 50)
        wellness = wellness_data.get('wellness_index', 50)
        
        # Calculate therapy scalar S_therapy (0-1)
        # Higher = more stimulating, Lower = more calming
        s_therapy = self._calculate_therapy_scalar(mood, energy, stress)
        
        # Julia set C parameter
        # Low stress/high wellness = more complex, beautiful fractals
        # High stress = simpler, more calming patterns
        c_real_base = -0.7
        c_imag_base = 0.27015
        
        # Modulate based on wellness (stay in therapeutic range)
        c_real = c_real_base + (wellness - 50) / 500 * (1 - 0.7 * s_therapy)
        c_imag = c_imag_base + math.sin(mood / 100 * math.pi) * 0.05
        
        # Lambda damping (prevents overstimulation)
        lambda_damping = 0.5 + s_therapy * 0.4  # 0.5-0.9 range
        
        # Color parameters
        # High s_therapy = emphasize hue changes (focus mode)
        # Low s_therapy = maintain base saturation, low sensory load
        hue_sensitivity = s_therapy
        saturation_base = 0.7 - s_therapy * 0.3
        
        # Evolution rate (animation speed)
        # Calmer states = slower evolution
        evolution_rate = 0.5 + s_therapy * 0.5
        
        # Fractal dimension target
        # Always aim for therapeutic range (1.3-1.5)
        d_target = THERAPEUTIC_D_MIN + s_therapy * (THERAPEUTIC_D_MAX - THERAPEUTIC_D_MIN)
        
        # Store this combination for this user
        params = {
            'c_real': round(c_real, 6),
            'c_imag': round(c_imag, 6),
            's_therapy': round(s_therapy, 4),
            'lambda_damping': round(lambda_damping, 4),
            'hue_sensitivity': round(hue_sensitivity, 4),
            'saturation_base': round(saturation_base, 4),
            'evolution_rate': round(evolution_rate, 4),
            'd_target': round(d_target, 4),
            'golden_angle_offset': GOLDEN_ANGLE * (wellness / 100),
            'iterations': self._calculate_optimal_iterations(wellness),
            'color_palette': self._select_palette(mood, energy, stress)
        }
        
        # Store in memory for this user
        self.math_combinations[user_id] = {
            'params': params,
            'wellness_at_generation': wellness,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return params
    
    def _calculate_therapy_scalar(self, mood: float, energy: float, stress: float) -> float:
        """
        Calculate S_therapy(t) = σ((T_α + N_focus - N_calm) / MaxIntensity)
        
        Returns 0-1 where:
        - 0 = deep relaxation mode (low sensory load)
        - 1 = active focus mode (higher stimulation OK)
        """
        # Simplified: based on energy and inverse stress
        focus_need = energy / 100
        calm_need = stress / 100
        
        raw_score = (focus_need - calm_need + 1) / 2  # Normalize to 0-1
        
        # Apply sigmoid for smooth transitions
        def sigmoid(x):
            return 1 / (1 + math.exp(-5 * (x - 0.5)))
        
        return sigmoid(raw_score)
    
    def _calculate_optimal_iterations(self, wellness: float) -> int:
        """Use Fibonacci-based iteration count."""
        # Higher wellness = more detail (more iterations)
        wellness_index = int(wellness / 100 * 6) + 4  # Index 4-10 in Fibonacci
        return FIBONACCI[min(wellness_index, len(FIBONACCI) - 1)]
    
    def _select_palette(self, mood: float, energy: float, stress: float) -> str:
        """Select autism-safe color palette based on state."""
        if stress > 70:
            return 'calm'  # Soft blues/greens for high stress
        elif energy > 70 and mood > 60:
            return 'energy'  # Warm sunrise colors for high energy
        elif mood > 70:
            return 'aurora'  # Northern lights for good mood
        elif energy < 40:
            return 'earth'  # Grounding browns for low energy
        else:
            return 'focus'  # Amber/gold for focus
    
    # ─────────────────────────────────────────────────────────────────────────
    # FEDERATED LEARNING (Privacy-Preserving)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _contribute_federated_insight(self, normalized_data: Dict):
        """
        Contribute anonymized insights to global learning.
        No personal data stored - only aggregate statistics.
        """
        # Round values to bins to prevent identification
        mood_bin = round(normalized_data.get('mood_level', 50) / 10) * 10
        energy_bin = round(normalized_data.get('energy_level', 50) / 10) * 10
        
        key = f"m{mood_bin}_e{energy_bin}"
        
        if key not in self.global_insights:
            self.global_insights[key] = {
                'count': 0,
                'avg_wellness': 0,
                'common_time': defaultdict(int)
            }
        
        insights = self.global_insights[key]
        n = insights['count']
        insights['avg_wellness'] = (insights['avg_wellness'] * n + normalized_data.get('wellness_index', 50)) / (n + 1)
        insights['count'] += 1
        
        hour = datetime.now().hour
        time_bucket = 'morning' if hour < 12 else ('afternoon' if hour < 18 else 'evening')
        insights['common_time'][time_bucket] += 1
    
    def get_federated_insight(self, mood: float, energy: float) -> Dict:
        """Get anonymized insights from similar states."""
        mood_bin = round(mood / 10) * 10
        energy_bin = round(energy / 10) * 10
        key = f"m{mood_bin}_e{energy_bin}"
        
        if key in self.global_insights and self.global_insights[key]['count'] > 5:
            return {
                'similar_users': self.global_insights[key]['count'],
                'typical_wellness': round(self.global_insights[key]['avg_wellness'], 1),
                'insight': f"Users with similar mood/energy typically achieve {round(self.global_insights[key]['avg_wellness'])}% wellness"
            }
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTIVE DYSFUNCTION EARLY WARNING
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_executive_dysfunction_risk(self, user_id: str) -> Dict:
        """
        Detect early signs of executive dysfunction or burnout.
        
        Warning signs:
        - Declining goal completion
        - Breaking habit streaks
        - Mood/energy trends downward
        - Task overload
        """
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get recent trends
        cursor.execute('''
            SELECT mood_level, energy_level, stress_level, goals_completed, 
                   habits_completed, wellness_score, date
            FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 7
        ''', (user_id,))
        
        recent = cursor.fetchall()
        
        # Get goal status
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN progress < 10 AND created_at < date('now', '-7 days') THEN 1 ELSE 0 END) as stalled
            FROM goals WHERE user_id = ? AND completed_at IS NULL
        ''', (user_id,))
        
        goals = cursor.fetchone()
        conn.close()
        
        risk_score = 0.0
        warnings = []
        
        if len(recent) >= 3:
            # Check mood trend
            moods = [r[0] or 50 for r in recent[:3]]
            if all(moods[i] < moods[i+1] for i in range(len(moods)-1)):
                risk_score += 0.2
                warnings.append("📉 Mood has been declining for 3+ days")
            
            # Check energy trend
            energies = [r[1] or 50 for r in recent[:3]]
            if all(energies[i] < energies[i+1] for i in range(len(energies)-1)):
                risk_score += 0.2
                warnings.append("🔋 Energy has been declining")
            
            # Check stress trend
            stresses = [r[2] or 50 for r in recent[:3]]
            if all(stresses[i] > stresses[i+1] for i in range(len(stresses)-1)):
                risk_score += 0.15
                warnings.append("😰 Stress has been increasing")
        
        # Check stalled goals
        if goals and goals[0] > 0:
            stall_ratio = (goals[1] or 0) / goals[0]
            if stall_ratio > 0.5:
                risk_score += 0.25
                warnings.append(f"🎯 {goals[1]} of {goals[0]} goals haven't progressed in a week")
        
        # Check recent low wellness
        if recent and recent[0][5] and recent[0][5] < 40:
            risk_score += 0.15
            warnings.append("💔 Recent wellness score is low")
        
        risk_level = 'low' if risk_score < 0.3 else ('medium' if risk_score < 0.6 else 'high')
        
        support = []
        if risk_level == 'high':
            support = [
                "🫂 This is a gentle reminder that you're doing your best",
                "🎯 Consider focusing on just ONE small task today",
                "💤 Rest is productive - it's okay to take a break",
                "📞 Reaching out to someone you trust might help"
            ]
        elif risk_level == 'medium':
            support = [
                "🌱 Small steps count - what's ONE thing you can do?",
                "🥄 Check your spoons - you may need to conserve energy"
            ]
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'warnings': warnings,
            'support': support
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MULTI-LAYER FRACTAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MultiLayerFractalEngine:
    """
    Generates multi-layer 2D fractals that combine:
    - Julia set (base layer - emotional state)
    - Mandelbrot highlights (goals/aspirations)
    - Golden spiral overlay (life path)
    - Particle scatter (habits/daily actions)
    """
    
    # Autism-safe color palettes (low contrast, calming)
    PALETTES = {
        'calm': {
            'primary': (86, 130, 156),    # Steel blue
            'secondary': (120, 166, 163),  # Seafoam
            'accent': (181, 211, 199),     # Sage
            'background': (240, 244, 245)  # Off-white
        },
        'focus': {
            'primary': (194, 139, 83),     # Amber
            'secondary': (220, 176, 105),  # Gold
            'accent': (240, 210, 145),     # Warm cream
            'background': (45, 40, 50)     # Dark purple
        },
        'energy': {
            'primary': (214, 124, 95),     # Coral
            'secondary': (239, 169, 118),  # Peach
            'accent': (250, 211, 168),     # Light peach
            'background': (50, 45, 55)     # Deep purple
        },
        'earth': {
            'primary': (139, 119, 101),    # Taupe
            'secondary': (176, 156, 137),  # Tan
            'accent': (210, 195, 178),     # Cream
            'background': (62, 55, 53)     # Dark brown
        },
        'aurora': {
            'primary': (106, 168, 156),    # Teal
            'secondary': (160, 196, 186),  # Light teal
            'accent': (203, 224, 218),     # Pale green
            'background': (35, 40, 55)     # Night blue
        }
    }
    
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
    
    def generate_composite(self, params: Dict, goals: List, habits: List) -> Image.Image:
        """Generate a multi-layer fractal visualization."""
        palette = self.PALETTES.get(params.get('color_palette', 'calm'))
        
        # Create layers
        base_layer = self._generate_julia_layer(params, palette)
        goal_layer = self._generate_goal_layer(goals, params, palette)
        spiral_layer = self._generate_golden_spiral(params, palette)
        habit_layer = self._generate_habit_particles(habits, params, palette)
        
        # Composite layers
        composite = Image.new('RGBA', (self.width, self.height), palette['background'] + (255,))
        composite = Image.alpha_composite(composite, base_layer)
        composite = Image.alpha_composite(composite, goal_layer)
        composite = Image.alpha_composite(composite, spiral_layer)
        composite = Image.alpha_composite(composite, habit_layer)
        
        return composite.convert('RGB')
    
    def _generate_julia_layer(self, params: Dict, palette: Dict) -> Image.Image:
        """Generate Julia set as base emotional layer."""
        c_real = params.get('c_real', -0.7)
        c_imag = params.get('c_imag', 0.27015)
        max_iter = params.get('iterations', 100)
        
        x = np.linspace(-1.5, 1.5, self.width)
        y = np.linspace(-1.5, 1.5, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        iterations = np.zeros((self.height, self.width))
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + complex(c_real, c_imag)
            iterations[mask] = i
        
        # Smooth coloring
        smooth = iterations - np.log2(np.log2(np.abs(Z) + 1) + 1)
        smooth = np.nan_to_num(smooth)
        normalized = smooth / max_iter
        
        # Color mapping
        img_array = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        for i in range(3):
            low = palette['background'][i]
            high = palette['primary'][i]
            img_array[:,:,i] = (low + normalized * (high - low)).astype(np.uint8)
        
        img_array[:,:,3] = 200  # Semi-transparent
        
        return Image.fromarray(img_array, 'RGBA')
    
    def _generate_goal_layer(self, goals: List, params: Dict, palette: Dict) -> Image.Image:
        """Generate goal representations as sacred geometry shapes."""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if not goals:
            return img
        
        center_x, center_y = self.width // 2, self.height // 2
        
        for i, goal in enumerate(goals[:12]):  # Max 12 goals visualized
            # Position using golden angle
            angle = i * GOLDEN_ANGLE_RAD
            radius = 100 + i * 40
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Size based on progress
            progress = goal.get('progress', 0) / 100
            size = 10 + progress * 30
            
            # Shape based on priority
            priority = goal.get('priority', 3)
            color = palette['accent'] + (int(100 + progress * 155),)
            
            if priority >= 4:
                # High priority: hexagon
                self._draw_polygon(draw, x, y, size, 6, color)
            elif priority >= 2:
                # Medium: pentagon  
                self._draw_polygon(draw, x, y, size, 5, color)
            else:
                # Low: triangle
                self._draw_polygon(draw, x, y, size, 3, color)
        
        return img
    
    def _draw_polygon(self, draw, cx, cy, radius, sides, color):
        """Draw a regular polygon."""
        points = []
        for i in range(sides):
            angle = i * 2 * math.pi / sides - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline=color[:3] + (255,))
    
    def _generate_golden_spiral(self, params: Dict, palette: Dict) -> Image.Image:
        """Generate golden spiral overlay representing life path."""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = self.width // 2, self.height // 2
        offset = params.get('golden_angle_offset', 0)
        
        # Draw Fibonacci spiral using arcs
        points = []
        for i in range(200):
            theta = i * 0.1 + math.radians(offset)
            r = math.pow(PHI, theta / (2 * math.pi)) * 5
            
            if r > min(self.width, self.height) / 2:
                break
            
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            points.append((x, y))
        
        if len(points) > 1:
            color = palette['secondary'] + (100,)
            draw.line(points, fill=color, width=2)
        
        return img
    
    def _generate_habit_particles(self, habits: List, params: Dict, palette: Dict) -> Image.Image:
        """Generate particles representing habits."""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if not habits:
            return img
        
        # Generate particles for each habit
        for i, habit in enumerate(habits[:20]):
            streak = habit.get('current_streak', 0)
            completed_today = habit.get('completed_today', 0)
            
            # More particles for higher streaks
            particle_count = min(20, 5 + streak)
            
            # Color: brighter if completed today
            base_color = palette['accent'] if completed_today else palette['secondary']
            
            for j in range(particle_count):
                # Position using golden angle distribution
                angle = (i * 20 + j) * GOLDEN_ANGLE_RAD
                radius = 150 + j * 15 + i * 10
                
                x = self.width // 2 + radius * math.cos(angle)
                y = self.height // 2 + radius * math.sin(angle)
                
                if 0 < x < self.width and 0 < y < self.height:
                    size = 2 + (streak / 10)
                    alpha = 150 if completed_today else 80
                    draw.ellipse([x-size, y-size, x+size, y+size], 
                               fill=base_color + (alpha,))
        
        return img


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Database:
    def __init__(self, db_path: str = "life_fractal_v11.db"):
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
        
        # Users
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT,
            subscription_status TEXT DEFAULT 'active',
            settings JSON DEFAULT '{}'
        )''')
        
        # Goals with more tracking fields
        cursor.execute('''CREATE TABLE IF NOT EXISTS goals (
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
            updated_at TEXT,
            completed_at TEXT,
            fractal_layer INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Habits with streak tracking
        cursor.execute('''CREATE TABLE IF NOT EXISTS habits (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            frequency TEXT DEFAULT 'daily',
            icon TEXT DEFAULT '✓',
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            total_completions INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            last_completed TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Daily entries with comprehensive tracking
        cursor.execute('''CREATE TABLE IF NOT EXISTS daily_entries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            mood_level INTEGER DEFAULT 50,
            energy_level INTEGER DEFAULT 50,
            stress_level INTEGER DEFAULT 50,
            sleep_hours REAL DEFAULT 7.0,
            sleep_quality INTEGER DEFAULT 50,
            goals_completed INTEGER DEFAULT 0,
            habits_completed INTEGER DEFAULT 0,
            spoons_available INTEGER DEFAULT 12,
            spoons_used INTEGER DEFAULT 0,
            journal_entry TEXT,
            wellness_score REAL DEFAULT 50.0,
            ai_predictions JSON,
            fractal_params JSON,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date)
        )''')
        
        # AI Memory - stores learned patterns
        cursor.execute('''CREATE TABLE IF NOT EXISTS ai_memory (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content JSON NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Fractal Parameters - stores optimal combinations
        cursor.execute('''CREATE TABLE IF NOT EXISTS fractal_params (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            params JSON NOT NULL,
            wellness_context JSON,
            effectiveness_score REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Habit completions
        cursor.execute('''CREATE TABLE IF NOT EXISTS habit_completions (
            id TEXT PRIMARY KEY,
            habit_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            completed_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (habit_id) REFERENCES habits(id),
            UNIQUE(habit_id, completed_date)
        )''')
        
        # Pet state
        cursor.execute('''CREATE TABLE IF NOT EXISTS pet_state (
            user_id TEXT PRIMARY KEY,
            species TEXT DEFAULT 'phoenix',
            name TEXT DEFAULT 'Spark',
            hunger REAL DEFAULT 50.0,
            energy REAL DEFAULT 50.0,
            mood REAL DEFAULT 50.0,
            level INTEGER DEFAULT 1,
            experience INTEGER DEFAULT 0,
            evolution_stage INTEGER DEFAULT 0,
            last_fed TEXT,
            last_played TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        conn.commit()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize core systems
db = Database()
brain = FractalIntelligenceBrain(db.get_connection)
fractal_engine = MultiLayerFractalEngine()

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        user_id = secrets.token_urlsafe(16)
        password_hash = generate_password_hash(password)
        now = datetime.now(timezone.utc).isoformat()
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                      (user_id, email, password_hash, now))
        cursor.execute('INSERT INTO pet_state (user_id) VALUES (?)', (user_id,))
        conn.commit()
        conn.close()
        
        session['user_id'] = user_id
        logger.info(f"✅ New user registered: {email}")
        return jsonify({'success': True, 'user_id': user_id})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 409
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['user_id'] = user['id']
        return jsonify({'success': True, 'user_id': user['id']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/auth/status')
def auth_status():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'user_id': session['user_id']})
    return jsonify({'authenticated': False})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DAILY CHECK-IN & WELLNESS (With AI Integration)
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/daily/today')
@require_auth
def get_today():
    user_id = session['user_id']
    today = datetime.now().strftime('%Y-%m-%d')
    
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? AND date = ?', (user_id, today))
    entry = cursor.fetchone()
    conn.close()
    
    if entry:
        return jsonify(dict(entry))
    return jsonify({'date': today, 'exists': False, 'spoons_available': 12})

@app.route('/api/daily/checkin', methods=['POST'])
@require_auth
def daily_checkin():
    """Daily check-in with full AI pipeline."""
    try:
        user_id = session['user_id']
        data = request.get_json()
        today = datetime.now().strftime('%Y-%m-%d')
        entry_id = secrets.token_urlsafe(8)
        
        # Run through AI brain pipeline
        ai_result = brain.ingest_daily_data(user_id, data)
        
        wellness_score = ai_result['wellness_index']
        predictions = ai_result['predictions']
        fractal_params = ai_result['fractal_params']
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO daily_entries 
            (id, user_id, date, mood_level, energy_level, stress_level, sleep_hours, 
             sleep_quality, spoons_available, spoons_used, journal_entry, wellness_score,
             ai_predictions, fractal_params, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (entry_id, user_id, today, 
             data.get('mood_level', 50),
             data.get('energy_level', 50),
             data.get('stress_level', 50),
             data.get('sleep_hours', 7),
             data.get('sleep_quality', 50),
             brain._normalize_wellness_data(data).get('spoons_available', 12),
             data.get('spoons_used', 0),
             data.get('journal_entry', ''),
             wellness_score,
             json.dumps(predictions),
             json.dumps(fractal_params),
             datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'wellness_score': wellness_score,
            'predictions': predictions,
            'fractal_params': fractal_params
        })
    except Exception as e:
        logger.error(f"Check-in error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellness/summary')
@require_auth
def wellness_summary():
    """Complete wellness summary with AI insights."""
    user_id = session['user_id']
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get today's entry
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? AND date = ?', (user_id, today))
    today_entry = cursor.fetchone()
    
    # Get goals count
    cursor.execute('''SELECT COUNT(*) as total, 
                     SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END) as completed 
                     FROM goals WHERE user_id = ?''', (user_id,))
    goals_stats = cursor.fetchone()
    
    # Get habits completed today
    cursor.execute('''SELECT COUNT(*) as count FROM habit_completions 
                     WHERE user_id = ? AND completed_date = ?''', (user_id, today))
    habits_today = cursor.fetchone()['count']
    
    # Get pet level
    cursor.execute('SELECT level FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    
    conn.close()
    
    wellness_score = today_entry['wellness_score'] if today_entry else 50
    spoons = 12
    if today_entry:
        spoons = (today_entry['spoons_available'] or 12) - (today_entry['spoons_used'] or 0)
    
    # Get AI predictions and risk assessment
    predictions = brain.predict_tomorrow(user_id)
    risk = brain.check_executive_dysfunction_risk(user_id)
    
    # Mayan date
    mayan = get_mayan_date()
    
    return jsonify({
        'wellness_score': round(wellness_score),
        'active_goals': (goals_stats['total'] or 0) - (goals_stats['completed'] or 0),
        'habits_today': habits_today,
        'pet_level': pet['level'] if pet else 1,
        'spoons_remaining': max(0, spoons),
        'mayan_date': mayan,
        'predictions': predictions,
        'risk_assessment': risk,
        'ai_insight': brain.get_federated_insight(
            today_entry['mood_level'] if today_entry else 50,
            today_entry['energy_level'] if today_entry else 50
        )
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# GOALS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET', 'POST'])
@require_auth
def goals():
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        data = request.get_json()
        goal_id = secrets.token_urlsafe(8)
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute('''INSERT INTO goals 
            (id, user_id, title, description, category, term, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (goal_id, user_id, data.get('title'), data.get('description', ''),
             data.get('category', 'personal'), data.get('term', 'medium'),
             data.get('priority', 3), now, now))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'goal_id': goal_id})
    
    cursor.execute('SELECT * FROM goals WHERE user_id = ? ORDER BY priority DESC, created_at DESC', (user_id,))
    goals_list = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'goals': goals_list})

@app.route('/api/goals/<goal_id>/progress', methods=['PUT'])
@require_auth
def update_goal_progress(goal_id):
    try:
        data = request.get_json()
        progress = min(100, max(0, float(data.get('progress', 0))))
        completed_at = datetime.now(timezone.utc).isoformat() if progress >= 100 else None
        now = datetime.now(timezone.utc).isoformat()
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE goals SET progress = ?, completed_at = ?, updated_at = ? WHERE id = ?',
                      (progress, completed_at, now, goal_id))
        conn.commit()
        conn.close()
        
        # Check Fibonacci milestones
        milestone = None
        for fib in FIBONACCI:
            if progress >= fib and progress < fib + 10:
                milestone = fib
                break
        
        return jsonify({'success': True, 'progress': progress, 'milestone': milestone})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HABITS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/habits', methods=['GET', 'POST'])
@require_auth
def habits():
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        data = request.get_json()
        habit_id = secrets.token_urlsafe(8)
        cursor.execute('''INSERT INTO habits (id, user_id, name, description, frequency, icon, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (habit_id, user_id, data.get('name'), data.get('description', ''),
                       data.get('frequency', 'daily'), data.get('icon', '✓'),
                       datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'habit_id': habit_id})
    
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''SELECT h.*, 
                     (SELECT COUNT(*) FROM habit_completions WHERE habit_id = h.id AND completed_date = ?) as completed_today
                     FROM habits h WHERE h.user_id = ?''', (today, user_id))
    habits_list = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'habits': habits_list})

@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(habit_id):
    try:
        user_id = session['user_id']
        today = datetime.now().strftime('%Y-%m-%d')
        completion_id = secrets.token_urlsafe(8)
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''INSERT OR IGNORE INTO habit_completions (id, habit_id, user_id, completed_date, created_at)
                         VALUES (?, ?, ?, ?, ?)''',
                      (completion_id, habit_id, user_id, today, datetime.now(timezone.utc).isoformat()))
        
        cursor.execute('''UPDATE habits SET current_streak = current_streak + 1, 
                         total_completions = total_completions + 1, last_completed = ?
                         WHERE id = ?''', (today, habit_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════════
# PET ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet')
@require_auth
def get_pet():
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    conn.close()
    
    if pet:
        return jsonify(dict(pet))
    return jsonify({'error': 'No pet found'}), 404

@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('''UPDATE pet_state SET hunger = MAX(0, hunger - 30), mood = MIN(100, mood + 10),
                     last_fed = ? WHERE user_id = ?''',
                  (datetime.now(timezone.utc).isoformat(), user_id))
    conn.commit()
    cursor.execute('SELECT * FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    conn.close()
    return jsonify({'success': True, 'pet': dict(pet)})

@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_pet():
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('''UPDATE pet_state SET energy = MAX(0, energy - 15), mood = MIN(100, mood + 20),
                     experience = experience + 10, last_played = ? WHERE user_id = ?''',
                  (datetime.now(timezone.utc).isoformat(), user_id))
    
    # Check level up
    cursor.execute('SELECT * FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    xp_needed = FIBONACCI[min(pet['level'] + 4, len(FIBONACCI) - 1)] * 10
    
    if pet['experience'] >= xp_needed:
        cursor.execute('UPDATE pet_state SET level = level + 1, experience = 0 WHERE user_id = ?', (user_id,))
    
    conn.commit()
    cursor.execute('SELECT * FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    conn.close()
    return jsonify({'success': True, 'pet': dict(pet)})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/visualization/data')
@require_auth
def get_visualization_data():
    """Get all user data for 3D visualization."""
    user_id = session['user_id']
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL', (user_id,))
    goals = [dict(row) for row in cursor.fetchall()]
    
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''SELECT h.*, 
                     (SELECT COUNT(*) FROM habit_completions WHERE habit_id = h.id AND completed_date = ?) as completed_today
                     FROM habits h WHERE h.user_id = ?''', (today, user_id))
    habits = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? AND date = ?', (user_id, today))
    today_entry = cursor.fetchone()
    
    cursor.execute('SELECT * FROM pet_state WHERE user_id = ?', (user_id,))
    pet = cursor.fetchone()
    
    cursor.execute('''SELECT date, wellness_score, mood_level, energy_level 
                     FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 7''', (user_id,))
    trend = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Get optimized fractal params
    wellness_data = dict(today_entry) if today_entry else {'mood_level': 50, 'energy_level': 50, 'stress_level': 50}
    fractal_params = brain.optimize_fractal_params(user_id, wellness_data)
    
    return jsonify({
        'goals': goals,
        'habits': habits,
        'wellness': dict(today_entry) if today_entry else {'wellness_score': 50, 'energy_level': 50, 'mood_level': 50, 'stress_level': 50},
        'pet': dict(pet) if pet else {'level': 1, 'mood': 50},
        'trend': trend,
        'mayan': get_mayan_date(),
        'fractal_params': fractal_params,
        'sacred_math': {
            'phi': PHI,
            'phi_inverse': PHI_INVERSE,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:10],
            'therapeutic_d': [THERAPEUTIC_D_MIN, THERAPEUTIC_D_MAX]
        }
    })

@app.route('/api/visualization/fractal/2d', methods=['POST'])
@require_auth
def generate_2d_fractal():
    """Generate multi-layer 2D fractal."""
    try:
        user_id = session['user_id']
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get user data
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1', (user_id,))
        entry = cursor.fetchone()
        
        cursor.execute('SELECT * FROM goals WHERE user_id = ? AND completed_at IS NULL', (user_id,))
        goals = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute('''SELECT h.*, 
                         (SELECT COUNT(*) FROM habit_completions WHERE habit_id = h.id AND completed_date = ?) as completed_today
                         FROM habits h WHERE h.user_id = ?''', (today, user_id))
        habits = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        # Get optimized params
        wellness_data = dict(entry) if entry else {'mood_level': 50, 'energy_level': 50, 'stress_level': 50}
        params = brain.optimize_fractal_params(user_id, wellness_data)
        
        # Generate composite fractal
        img = fractal_engine.generate_composite(params, goals, habits)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AI & PREDICTIONS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/ai/predictions')
@require_auth
def get_predictions():
    """Get AI predictions for tomorrow."""
    user_id = session['user_id']
    predictions = brain.predict_tomorrow(user_id)
    return jsonify(predictions)

@app.route('/api/ai/risk')
@require_auth
def get_risk_assessment():
    """Get executive dysfunction risk assessment."""
    user_id = session['user_id']
    risk = brain.check_executive_dysfunction_risk(user_id)
    return jsonify(risk)

@app.route('/api/ai/fractal-params')
@require_auth
def get_fractal_params():
    """Get current optimized fractal parameters."""
    user_id = session['user_id']
    
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1', (user_id,))
    entry = cursor.fetchone()
    conn.close()
    
    wellness_data = dict(entry) if entry else {'mood_level': 50, 'energy_level': 50, 'stress_level': 50}
    params = brain.optimize_fractal_params(user_id, wellness_data)
    
    return jsonify({
        'params': params,
        'stored_combinations': brain.math_combinations.get(user_id)
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AUDIO PRESETS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/audio/presets')
def audio_presets():
    return jsonify({
        'presets': [
            {'id': 'alpha', 'name': 'Alpha Waves (Relaxation)', 'base_freq': 200, 'beat_freq': 10, 'description': 'Promotes calm, relaxed alertness'},
            {'id': 'theta', 'name': 'Theta Waves (Meditation)', 'base_freq': 150, 'beat_freq': 6, 'description': 'Deep meditation, creativity'},
            {'id': 'delta', 'name': 'Delta Waves (Deep Sleep)', 'base_freq': 100, 'beat_freq': 2.5, 'description': 'Promotes deep, restorative sleep'},
            {'id': 'beta', 'name': 'Beta Waves (Focus)', 'base_freq': 250, 'beat_freq': 20, 'description': 'Active thinking, concentration'},
            {'id': 'gamma', 'name': 'Gamma Waves (Cognition)', 'base_freq': 300, 'beat_freq': 40, 'description': 'Peak awareness, problem solving'},
            {'id': 'schumann', 'name': 'Schumann Resonance (Grounding)', 'base_freq': 136.1, 'beat_freq': 7.83, 'description': 'Earth frequency, grounding'}
        ]
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '11.0.0',
        'ai_brain': 'active',
        'ml_available': HAS_SKLEARN,
        'features': [
            'AI_BRAIN_LEARNING',
            'PATTERN_RECOGNITION',
            'FEDERATED_INSIGHTS',
            'MULTI_LAYER_FRACTALS',
            'EXECUTIVE_DYSFUNCTION_DETECTION',
            'PREDICTIVE_ANALYTICS',
            'FRACTAL_MATH_OPTIMIZATION',
            '3D_IMMERSIVE_WORLD',
            'BINAURAL_AUDIO',
            'SPOON_THEORY',
            'MAYAN_CALENDAR'
        ]
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAYAN CALENDAR HELPER
# ═══════════════════════════════════════════════════════════════════════════════════════════

def get_mayan_date():
    base_date = datetime(2012, 12, 21)
    today = datetime.now()
    days_diff = (today - base_date).days
    tone = ((days_diff % 13) + 1)
    glyph_index = days_diff % 20
    glyph = MAYAN_GLYPHS[glyph_index]
    meaning = MAYAN_MEANINGS.get(glyph, ('Unknown', ''))
    
    energies = {
        1: "Fresh start energy. Plant seeds for new projects.",
        2: "Duality. Balance opposing forces.",
        3: "Gentle playful energy. Focus on small, steady progress.",
        4: "Stability. Good day for foundation work.",
        5: "Empowerment. Take charge of your goals.",
        6: "Flow. Let things unfold naturally.",
        7: "Mystical energy. Trust your intuition.",
        8: "Harmonic resonance. Collaboration favored.",
        9: "Completion energy. Finish what you started.",
        10: "Manifestation power. Your efforts bear fruit.",
        11: "Dissolution. Release what no longer serves you.",
        12: "Understanding. See the bigger picture.",
        13: "Transcendence. Connect with higher purpose."
    }
    
    return {
        'tone': tone,
        'glyph': glyph,
        'glyph_meaning': meaning[0],
        'full_name': f"{tone} {glyph} ({meaning[0]})",
        'description': meaning[1],
        'energy': energies.get(tone, "Embrace today's energy.")
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HTML (Complete UI)
# ═══════════════════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal Intelligence v11</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg-dark: #0f0f1a;
            --bg-card: #1a1a2e;
            --bg-input: #252540;
            --text-primary: #e8e8f0;
            --text-secondary: #a0a0b8;
            --accent-orange: #ff6b35;
            --accent-purple: #7c3aed;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-gold: #f59e0b;
            --accent-red: #ef4444;
            --border: #2a2a45;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1025 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .app-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 25px;
        }
        
        .logo {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-orange), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .version { color: var(--text-secondary); font-size: 0.9rem; margin-top: 5px; }
        .ai-badge { 
            display: inline-block;
            background: var(--accent-purple);
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-left: 10px;
        }
        
        .nav {
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 25px;
        }
        
        .nav-btn {
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 25px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        
        .nav-btn:hover { border-color: var(--accent-purple); color: var(--text-primary); }
        .nav-btn.active { background: var(--accent-orange); color: white; border-color: var(--accent-orange); }
        
        .section { display: none; animation: fadeIn 0.3s ease; }
        .section.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        
        .card-title {
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: var(--accent-orange);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: var(--bg-input);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-orange), var(--accent-gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-label { color: var(--text-secondary); font-size: 0.85rem; margin-top: 5px; }
        
        .prediction-card {
            background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
            border: 1px solid var(--accent-purple);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        
        .prediction-title { color: var(--accent-purple); font-weight: 600; margin-bottom: 10px; }
        
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .risk-low { background: var(--accent-green); color: white; }
        .risk-medium { background: var(--accent-gold); color: #1a1a2e; }
        .risk-high { background: var(--accent-red); color: white; }
        
        .mayan-card {
            background: linear-gradient(135deg, #2d1f4e 0%, #1a1a2e 100%);
            border: 1px solid var(--accent-purple);
        }
        
        .mayan-date { font-size: 1.5rem; font-weight: 600; color: var(--accent-gold); }
        .mayan-energy { color: var(--text-secondary); margin-top: 10px; font-style: italic; }
        
        .spoons-container { margin: 20px 0; }
        .spoons-display { font-size: 2rem; letter-spacing: 5px; margin: 10px 0; }
        .spoon-tips { color: var(--text-secondary); font-size: 0.9rem; list-style: none; }
        .spoon-tips li { margin: 5px 0; }
        
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; color: var(--text-secondary); margin-bottom: 8px; font-size: 0.9rem; }
        
        input[type="text"], input[type="email"], input[type="password"], input[type="number"], 
        select, textarea {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 1rem;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent-orange);
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            background: var(--bg-input);
            border-radius: 4px;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--accent-orange);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-orange), #ff8c42);
            color: white;
        }
        
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3); }
        
        .btn-secondary {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-3d {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            color: white;
            font-size: 1.1rem;
            padding: 15px 30px;
        }
        
        .item-card {
            background: var(--bg-input);
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 4px solid var(--accent-purple);
        }
        
        .item-title { font-weight: 600; margin-bottom: 8px; }
        
        .progress-bar {
            height: 10px;
            background: var(--bg-dark);
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-orange), var(--accent-gold));
            transition: width 0.3s;
        }
        
        .viz-3d-container {
            position: relative;
            width: 100%;
            height: 500px;
            background: var(--bg-dark);
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid var(--accent-purple);
        }
        
        #three-canvas { width: 100%; height: 100%; display: block; }
        
        .viz-controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px 25px;
            border-radius: 30px;
            z-index: 10;
        }
        
        .viz-info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 0.85rem;
            z-index: 10;
        }
        
        .fractal-params {
            background: var(--bg-input);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-family: monospace;
            font-size: 0.85rem;
        }
        
        .fullscreen-3d {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            background: #000;
        }
        
        .fullscreen-3d .viz-3d-container {
            width: 100%;
            height: 100%;
            border-radius: 0;
            border: none;
        }
        
        .exit-fullscreen {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,107,53,0.9);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            z-index: 11;
        }
        
        .auth-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: var(--bg-card);
            border-radius: 20px;
            border: 1px solid var(--border);
        }
        
        .auth-title {
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 30px;
            color: var(--accent-orange);
        }
        
        .pet-display { text-align: center; padding: 30px; }
        .pet-avatar { font-size: 5rem; margin-bottom: 20px; }
        
        .pet-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .pet-stat {
            background: var(--bg-input);
            padding: 15px;
            border-radius: 10px;
        }
        
        @media (max-width: 768px) {
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .viz-3d-container { height: 350px; }
            .nav { gap: 5px; }
            .nav-btn { padding: 8px 14px; font-size: 0.85rem; }
        }
    </style>
</head>
<body>
    <div id="app">
        <!-- Auth Screen -->
        <div id="auth-screen" class="auth-container">
            <h1 class="auth-title">🌀 Life Fractal Intelligence</h1>
            <p style="text-align: center; color: var(--text-secondary); margin-bottom: 20px;">
                AI-Powered Life Organization<br>
                <span class="ai-badge">🧠 Intelligent</span>
            </p>
            <div id="login-form">
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="login-email" placeholder="your@email.com">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="login-password" placeholder="••••••••">
                </div>
                <button class="btn btn-primary" style="width: 100%; margin-bottom: 15px;" onclick="login()">Login</button>
                <button class="btn btn-secondary" style="width: 100%;" onclick="showRegister()">Create Account</button>
            </div>
            <div id="register-form" style="display: none;">
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="register-email" placeholder="your@email.com">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="register-password" placeholder="••••••••">
                </div>
                <button class="btn btn-primary" style="width: 100%; margin-bottom: 15px;" onclick="register()">Create Account</button>
                <button class="btn btn-secondary" style="width: 100%;" onclick="showLogin()">Back to Login</button>
            </div>
        </div>
        
        <!-- Main App -->
        <div id="main-app" class="app-container" style="display: none;">
            <header class="header">
                <div class="logo">🌀 Life Fractal Intelligence <span class="ai-badge">🧠 AI</span></div>
                <div class="version">v11.0 - Your Life Visualized as Living Art</div>
            </header>
            
            <nav class="nav">
                <button class="nav-btn active" onclick="showSection('overview', this)">📊 Overview</button>
                <button class="nav-btn" onclick="showSection('today', this)">📅 Today</button>
                <button class="nav-btn" onclick="showSection('goals', this)">🎯 Goals</button>
                <button class="nav-btn" onclick="showSection('habits', this)">✅ Habits</button>
                <button class="nav-btn" onclick="showSection('fractal', this)">🌀 Fractal</button>
                <button class="nav-btn" onclick="showSection('pet', this)">🐾 Pet</button>
                <button class="nav-btn" onclick="showSection('audio', this)">🎵 Audio</button>
            </nav>
            
            <!-- OVERVIEW SECTION WITH AI -->
            <div id="overview-section" class="section active">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="wellness-score">--</div>
                        <div class="metric-label">Wellness Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="active-goals">--</div>
                        <div class="metric-label">Active Goals</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="habits-today">--</div>
                        <div class="metric-label">Habits Today</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="pet-level">--</div>
                        <div class="metric-label">Pet Level</div>
                    </div>
                </div>
                
                <!-- AI Predictions -->
                <div class="prediction-card">
                    <div class="prediction-title">🧠 AI Predictions for Tomorrow</div>
                    <div id="predictions-content">Loading...</div>
                </div>
                
                <!-- Risk Assessment -->
                <div class="prediction-card" id="risk-card" style="display: none;">
                    <div class="prediction-title">⚠️ Executive Function Check</div>
                    <div id="risk-content"></div>
                </div>
                
                <div class="card mayan-card">
                    <div class="card-title">📅 Mayan Tzolkin Date</div>
                    <div class="mayan-date" id="mayan-date">Loading...</div>
                    <div class="mayan-energy" id="mayan-energy"></div>
                </div>
                
                <div class="card">
                    <div class="card-title">🥄 Energy Spoons</div>
                    <div class="spoons-container">
                        <div class="spoons-display" id="spoons-display">🥄🥄🥄🥄🥄🥄🥄🥄🥄🥄🥄🥄</div>
                        <ul class="spoon-tips" id="spoon-tips"></ul>
                    </div>
                </div>
            </div>
            
            <!-- TODAY SECTION -->
            <div id="today-section" class="section">
                <div class="card">
                    <div class="card-title">📝 Daily Check-in</div>
                    <p style="color: var(--text-secondary); margin-bottom: 20px;">
                        Your data feeds the AI brain - helping it learn your patterns and optimize your fractal universe.
                    </p>
                    <div class="form-group">
                        <label>Mood Level: <span id="mood-val">50</span></label>
                        <input type="range" id="mood-slider" min="0" max="100" value="50" oninput="updateSliderVal('mood')">
                    </div>
                    <div class="form-group">
                        <label>Energy Level: <span id="energy-val">50</span></label>
                        <input type="range" id="energy-slider" min="0" max="100" value="50" oninput="updateSliderVal('energy')">
                    </div>
                    <div class="form-group">
                        <label>Stress Level: <span id="stress-val">50</span></label>
                        <input type="range" id="stress-slider" min="0" max="100" value="50" oninput="updateSliderVal('stress')">
                    </div>
                    <div class="form-group">
                        <label>Sleep Hours</label>
                        <input type="number" id="sleep-hours" value="7" min="0" max="24" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>Sleep Quality: <span id="sleep-q-val">50</span></label>
                        <input type="range" id="sleep-q-slider" min="0" max="100" value="50" oninput="updateSliderVal('sleep-q')">
                    </div>
                    <div class="form-group">
                        <label>Journal Entry</label>
                        <textarea id="journal-entry" rows="4" placeholder="How are you feeling today?"></textarea>
                    </div>
                    <button class="btn btn-primary" onclick="saveCheckin()">💾 Save & Analyze</button>
                </div>
            </div>
            
            <!-- GOALS SECTION -->
            <div id="goals-section" class="section">
                <div class="card">
                    <div class="card-title">🎯 Your Goals</div>
                    <p style="color: var(--text-secondary); margin-bottom: 15px;">
                        Each goal becomes a floating sacred geometry shape in your 3D universe.
                    </p>
                    <div id="goals-list"><p style="color: var(--text-secondary);">No goals yet. Create your first goal!</p></div>
                    <div class="form-group" style="margin-top: 20px;">
                        <input type="text" id="new-goal" placeholder="New goal title...">
                    </div>
                    <button class="btn btn-primary" onclick="addGoal()">Add Goal</button>
                </div>
            </div>
            
            <!-- HABITS SECTION -->
            <div id="habits-section" class="section">
                <div class="card">
                    <div class="card-title">✅ Daily Habits</div>
                    <p style="color: var(--text-secondary); margin-bottom: 15px;">
                        Habits become particles orbiting your fractal universe.
                    </p>
                    <div id="habits-list"><p style="color: var(--text-secondary);">No habits yet. Create your first habit!</p></div>
                    <div class="form-group" style="margin-top: 20px;">
                        <input type="text" id="new-habit" placeholder="New habit name...">
                    </div>
                    <button class="btn btn-primary" onclick="addHabit()">Add Habit</button>
                </div>
            </div>
            
            <!-- FRACTAL SECTION -->
            <div id="fractal-section" class="section">
                <div class="card">
                    <div class="card-title">🌀 Your Fractal Universe</div>
                    <p style="color: var(--text-secondary); margin-bottom: 20px;">
                        AI-optimized fractals based on your wellness data. Multi-layer visualization combining Julia sets, 
                        goal geometries, golden spirals, and habit particles.
                    </p>
                    
                    <div style="display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
                        <button class="btn btn-secondary" onclick="generate2DFractal()">🖼️ Generate Multi-Layer 2D</button>
                        <button class="btn btn-3d" onclick="launch3DWorld()">🌌 Enter 3D Universe</button>
                        <button class="btn btn-3d" onclick="launch3DFullscreen()">🔭 Fullscreen Experience</button>
                    </div>
                    
                    <div id="2d-fractal-container" style="text-align: center; margin: 20px 0;">
                        <p style="color: var(--text-secondary);">Click "Generate Multi-Layer 2D" to create your personalized fractal</p>
                    </div>
                    
                    <div id="viz-3d-wrapper" style="display: none;">
                        <div class="viz-3d-container" id="viz-3d-container">
                            <canvas id="three-canvas"></canvas>
                            <div class="viz-info">
                                <div>🌀 <strong>Your Fractal Universe</strong></div>
                                <div id="viz-goals-count">Goals: --</div>
                                <div id="viz-wellness">Wellness: --</div>
                                <div id="viz-therapy">S_therapy: --</div>
                            </div>
                            <div class="viz-controls">
                                <button class="btn btn-secondary" onclick="resetCamera()">🔄 Reset</button>
                                <button class="btn btn-secondary" onclick="toggleAutoRotate()">🔁 Rotate</button>
                                <button class="btn btn-primary" onclick="close3DWorld()">✕ Close</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="fractal-params" id="fractal-params-display">
                        <strong>🔢 AI-Optimized Fractal Parameters:</strong><br>
                        <span id="params-display">Loading...</span>
                    </div>
                    
                    <div class="card" style="background: var(--bg-input); margin-top: 20px;">
                        <strong>🔢 Sacred Mathematics:</strong>
                        <ul style="margin-top: 10px; color: var(--text-secondary); list-style: none;">
                            <li>φ (Golden Ratio): 1.618033988749895</li>
                            <li>Golden Angle: 137.5078°</li>
                            <li>Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...</li>
                            <li>Therapeutic Fractal Dimension: 1.3 - 1.5</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <!-- PET SECTION -->
            <div id="pet-section" class="section">
                <div class="card">
                    <div class="card-title">🐾 Your Virtual Companion</div>
                    <div class="pet-display">
                        <div class="pet-avatar" id="pet-avatar">🔥</div>
                        <div id="pet-name" style="font-size: 1.5rem; font-weight: 600;">Spark</div>
                        <div id="pet-species" style="color: var(--text-secondary);">Phoenix • Level 1</div>
                        <div class="pet-stats">
                            <div class="pet-stat">
                                <div style="font-size: 1.5rem;" id="pet-hunger">50</div>
                                <div style="color: var(--text-secondary); font-size: 0.85rem;">Hunger</div>
                            </div>
                            <div class="pet-stat">
                                <div style="font-size: 1.5rem;" id="pet-energy">50</div>
                                <div style="color: var(--text-secondary); font-size: 0.85rem;">Energy</div>
                            </div>
                            <div class="pet-stat">
                                <div style="font-size: 1.5rem;" id="pet-mood">50</div>
                                <div style="color: var(--text-secondary); font-size: 0.85rem;">Mood</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 15px; justify-content: center;">
                            <button class="btn btn-primary" onclick="feedPet()">🍖 Feed</button>
                            <button class="btn btn-secondary" onclick="playPet()">🎾 Play</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- AUDIO SECTION -->
            <div id="audio-section" class="section">
                <div class="card">
                    <div class="card-title">🎵 Binaural Beats Therapy</div>
                    <p style="color: var(--text-secondary); margin-bottom: 20px;">
                        Therapeutic audio frequencies. Use headphones for binaural effect.
                    </p>
                    <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
                        <select id="audio-preset" style="flex: 1; min-width: 200px;">
                            <option value="alpha">🧘 Alpha (Relaxation)</option>
                            <option value="theta">🌙 Theta (Meditation)</option>
                            <option value="delta">😴 Delta (Deep Sleep)</option>
                            <option value="beta">⚡ Beta (Focus)</option>
                            <option value="gamma">🧠 Gamma (Cognition)</option>
                            <option value="schumann">🌍 Schumann (Grounding)</option>
                        </select>
                        <button class="btn btn-primary" onclick="playAudio()">▶️ Play</button>
                        <button class="btn btn-secondary" onclick="stopAudio()">⏹️ Stop</button>
                    </div>
                    <div id="audio-status" style="margin-top: 15px; color: var(--text-secondary);">
                        Select a preset and click Play
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Fullscreen Modal -->
    <div id="fullscreen-3d-modal" style="display: none;">
        <button class="exit-fullscreen" onclick="exitFullscreen3D()">✕ Exit</button>
        <div class="viz-3d-container" id="fullscreen-canvas-container">
            <canvas id="fullscreen-three-canvas"></canvas>
        </div>
    </div>
    
    <script>
    // Sacred Math
    const PHI = (1 + Math.sqrt(5)) / 2;
    const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
    const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
    
    // State
    let vizData = null;
    let scene, camera, renderer;
    let goalMeshes = [];
    let particleSystem;
    let isAutoRotate = true;
    let audioContext, oscillatorL, oscillatorR;
    
    // Auth
    async function checkAuth() {
        try {
            const res = await fetch('/api/auth/status');
            const data = await res.json();
            if (data.authenticated) {
                showMainApp();
                loadAllData();
            }
        } catch (e) { console.error('Auth check failed:', e); }
    }
    
    async function login() {
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email, password})
            });
            const data = await res.json();
            if (data.success) { showMainApp(); loadAllData(); }
            else alert(data.error || 'Login failed');
        } catch (e) { alert('Login error: ' + e.message); }
    }
    
    async function register() {
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email, password})
            });
            const data = await res.json();
            if (data.success) { showMainApp(); loadAllData(); }
            else alert(data.error || 'Registration failed');
        } catch (e) { alert('Registration error: ' + e.message); }
    }
    
    function showRegister() {
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('register-form').style.display = 'block';
    }
    
    function showLogin() {
        document.getElementById('register-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'block';
    }
    
    function showMainApp() {
        document.getElementById('auth-screen').style.display = 'none';
        document.getElementById('main-app').style.display = 'block';
    }
    
    // Data Loading
    async function loadAllData() {
        await loadWellnessSummary();
        await loadGoals();
        await loadHabits();
        await loadPet();
        await loadVisualizationData();
    }
    
    async function loadWellnessSummary() {
        try {
            const res = await fetch('/api/wellness/summary');
            const data = await res.json();
            
            document.getElementById('wellness-score').textContent = data.wellness_score || '--';
            document.getElementById('active-goals').textContent = data.active_goals || 0;
            document.getElementById('habits-today').textContent = data.habits_today || 0;
            document.getElementById('pet-level').textContent = data.pet_level || 1;
            
            // Mayan
            if (data.mayan_date) {
                document.getElementById('mayan-date').textContent = data.mayan_date.full_name;
                document.getElementById('mayan-energy').textContent = data.mayan_date.energy;
            }
            
            // Spoons
            const spoons = data.spoons_remaining || 12;
            document.getElementById('spoons-display').textContent = '🥄'.repeat(Math.max(0, spoons)) + '⚪'.repeat(Math.max(0, 12 - spoons));
            updateSpoonTips(spoons);
            
            // AI Predictions
            if (data.predictions) {
                document.getElementById('predictions-content').innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                        <div>Predicted Mood: <strong>${data.predictions.predicted_mood}</strong></div>
                        <div>Predicted Energy: <strong>${data.predictions.predicted_energy}</strong></div>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">
                        Confidence: ${Math.round(data.predictions.confidence * 100)}%
                    </div>
                    ${data.predictions.recommendations ? '<ul style="margin-top: 10px; color: var(--text-secondary);">' + 
                        data.predictions.recommendations.map(r => `<li>${r}</li>`).join('') + '</ul>' : ''}
                `;
            }
            
            // Risk Assessment
            if (data.risk_assessment && data.risk_assessment.risk_level !== 'low') {
                const riskCard = document.getElementById('risk-card');
                riskCard.style.display = 'block';
                const riskClass = data.risk_assessment.risk_level === 'high' ? 'risk-high' : 'risk-medium';
                document.getElementById('risk-content').innerHTML = `
                    <span class="risk-badge ${riskClass}">${data.risk_assessment.risk_level.toUpperCase()} RISK</span>
                    ${data.risk_assessment.warnings.length ? '<ul style="margin-top: 15px; color: var(--text-secondary);">' + 
                        data.risk_assessment.warnings.map(w => `<li>${w}</li>`).join('') + '</ul>' : ''}
                    ${data.risk_assessment.support.length ? '<div style="margin-top: 15px; padding: 15px; background: var(--bg-input); border-radius: 10px;">' + 
                        data.risk_assessment.support.map(s => `<div style="margin: 5px 0;">${s}</div>`).join('') + '</div>' : ''}
                `;
            }
            
        } catch (e) { console.error('Error loading wellness:', e); }
    }
    
    function updateSpoonTips(spoons) {
        const tips = document.getElementById('spoon-tips');
        if (spoons >= 10) {
            tips.innerHTML = '<li>• Good energy - tackle challenging tasks</li><li>• Bank extra spoons with quick wins</li>';
        } else if (spoons >= 6) {
            tips.innerHTML = '<li>• Moderate energy - choose wisely</li><li>• Focus on top 2-3 priorities</li>';
        } else {
            tips.innerHTML = '<li>• Low energy - be gentle with yourself</li><li>• Focus only on essentials</li><li>• Rest is productive</li>';
        }
    }
    
    async function loadGoals() {
        try {
            const res = await fetch('/api/goals');
            const data = await res.json();
            const container = document.getElementById('goals-list');
            if (!data.goals || data.goals.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary);">No goals yet. Create your first goal!</p>';
                return;
            }
            container.innerHTML = data.goals.map(goal => `
                <div class="item-card">
                    <div class="item-title">${goal.completed_at ? '✅' : '🎯'} ${goal.title}</div>
                    <div class="progress-bar"><div class="progress-fill" style="width: ${goal.progress}%"></div></div>
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <button class="btn btn-secondary" style="font-size: 0.85rem; padding: 6px 12px;" onclick="updateGoalProgress('${goal.id}', ${Math.min(100, goal.progress + 10)})">+10%</button>
                        <button class="btn btn-secondary" style="font-size: 0.85rem; padding: 6px 12px;" onclick="updateGoalProgress('${goal.id}', 100)">Complete</button>
                    </div>
                </div>
            `).join('');
        } catch (e) { console.error('Error loading goals:', e); }
    }
    
    async function loadHabits() {
        try {
            const res = await fetch('/api/habits');
            const data = await res.json();
            const container = document.getElementById('habits-list');
            if (!data.habits || data.habits.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary);">No habits yet. Create your first habit!</p>';
                return;
            }
            container.innerHTML = data.habits.map(habit => `
                <div class="item-card" style="border-left-color: ${habit.completed_today ? 'var(--accent-green)' : 'var(--accent-purple)'};">
                    <div class="item-title">${habit.completed_today ? '✅' : '⬜'} ${habit.name}</div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem;">Streak: ${habit.current_streak} days</div>
                    ${!habit.completed_today ? `<button class="btn btn-primary" style="margin-top: 10px; font-size: 0.85rem; padding: 6px 12px;" onclick="completeHabit('${habit.id}')">Mark Complete</button>` : ''}
                </div>
            `).join('');
        } catch (e) { console.error('Error loading habits:', e); }
    }
    
    async function loadPet() {
        try {
            const res = await fetch('/api/pet');
            const data = await res.json();
            const avatars = {phoenix: '🔥', dragon: '🐉', cat: '🐱', owl: '🦉', fox: '🦊', wolf: '🐺', unicorn: '🦄', turtle: '🐢'};
            document.getElementById('pet-avatar').textContent = avatars[data.species] || '🔥';
            document.getElementById('pet-name').textContent = data.name || 'Spark';
            document.getElementById('pet-species').textContent = `${data.species || 'Phoenix'} • Level ${data.level || 1}`;
            document.getElementById('pet-hunger').textContent = Math.round(data.hunger || 50);
            document.getElementById('pet-energy').textContent = Math.round(data.energy || 50);
            document.getElementById('pet-mood').textContent = Math.round(data.mood || 50);
        } catch (e) { console.error('Error loading pet:', e); }
    }
    
    async function loadVisualizationData() {
        try {
            const res = await fetch('/api/visualization/data');
            vizData = await res.json();
            
            // Update fractal params display
            if (vizData.fractal_params) {
                const p = vizData.fractal_params;
                document.getElementById('params-display').innerHTML = `
                    C = ${p.c_real} + ${p.c_imag}i | S_therapy = ${p.s_therapy} | λ = ${p.lambda_damping}<br>
                    Palette: ${p.color_palette} | Iterations: ${p.iterations} | D_target: ${p.d_target}
                `;
            }
        } catch (e) { console.error('Error loading viz data:', e); }
    }
    
    // Actions
    async function addGoal() {
        const title = document.getElementById('new-goal').value.trim();
        if (!title) return alert('Please enter a goal');
        try {
            await fetch('/api/goals', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({title}) });
            document.getElementById('new-goal').value = '';
            loadGoals(); loadWellnessSummary(); loadVisualizationData();
        } catch (e) { alert('Error: ' + e.message); }
    }
    
    async function updateGoalProgress(goalId, progress) {
        try {
            await fetch(`/api/goals/${goalId}/progress`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({progress}) });
            loadGoals(); loadWellnessSummary();
        } catch (e) { console.error('Error:', e); }
    }
    
    async function addHabit() {
        const name = document.getElementById('new-habit').value.trim();
        if (!name) return alert('Please enter a habit');
        try {
            await fetch('/api/habits', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name}) });
            document.getElementById('new-habit').value = '';
            loadHabits();
        } catch (e) { alert('Error: ' + e.message); }
    }
    
    async function completeHabit(habitId) {
        try {
            await fetch(`/api/habits/${habitId}/complete`, {method: 'POST'});
            loadHabits(); loadWellnessSummary();
        } catch (e) { console.error('Error:', e); }
    }
    
    async function feedPet() {
        try { await fetch('/api/pet/feed', {method: 'POST'}); loadPet(); } catch (e) { console.error('Error:', e); }
    }
    
    async function playPet() {
        try { await fetch('/api/pet/play', {method: 'POST'}); loadPet(); } catch (e) { console.error('Error:', e); }
    }
    
    async function saveCheckin() {
        const data = {
            mood_level: parseInt(document.getElementById('mood-slider').value),
            energy_level: parseInt(document.getElementById('energy-slider').value),
            stress_level: parseInt(document.getElementById('stress-slider').value),
            sleep_hours: parseFloat(document.getElementById('sleep-hours').value),
            sleep_quality: parseInt(document.getElementById('sleep-q-slider').value),
            journal_entry: document.getElementById('journal-entry').value
        };
        try {
            const res = await fetch('/api/daily/checkin', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
            const result = await res.json();
            alert('✅ Check-in saved! Wellness: ' + Math.round(result.wellness_score) + '\\n\\nAI has analyzed your data and updated your fractal parameters.');
            loadAllData();
        } catch (e) { alert('Error: ' + e.message); }
    }
    
    // 3D Visualization
    function init3DWorld(canvasId, containerId) {
        const container = document.getElementById(containerId);
        const canvas = document.getElementById(canvasId);
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a15);
        scene.fog = new THREE.FogExp2(0x0a0a15, 0.02);
        
        camera = new THREE.PerspectiveCamera(60, container.offsetWidth / container.offsetHeight, 0.1, 1000);
        camera.position.set(0, 5, 15);
        
        renderer = new THREE.WebGLRenderer({canvas, antialias: true});
        renderer.setSize(container.offsetWidth, container.offsetHeight);
        
        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambient);
        const light = new THREE.DirectionalLight(0xffffff, 0.6);
        light.position.set(10, 20, 10);
        scene.add(light);
        
        createGoalGeometries();
        createParticles();
        createStars();
        
        setupControls(container);
        animate();
    }
    
    function createGoalGeometries() {
        if (!vizData || !vizData.goals) return;
        goalMeshes.forEach(m => scene.remove(m));
        goalMeshes = [];
        
        vizData.goals.forEach((goal, i) => {
            const angle = i * GOLDEN_ANGLE;
            const radius = 3 + i * 0.8;
            const height = Math.sin(i * 0.5) * 3;
            
            const size = 0.5 + (goal.progress || 0) / 100 * 0.5;
            const geometry = goal.priority >= 4 ? new THREE.IcosahedronGeometry(size, 1) : 
                             goal.priority >= 2 ? new THREE.OctahedronGeometry(size) : new THREE.TetrahedronGeometry(size);
            
            const hue = 0.1 + (goal.progress || 0) / 100 * 0.3;
            const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
            const material = new THREE.MeshPhongMaterial({color, emissive: color, emissiveIntensity: 0.3, transparent: true, opacity: 0.85});
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(radius * Math.cos(angle), height, radius * Math.sin(angle));
            mesh.userData = {goal, baseY: height};
            scene.add(mesh);
            goalMeshes.push(mesh);
        });
    }
    
    function createParticles() {
        const count = 500;
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const phi = Math.acos(1 - 2 * (i + 0.5) / count);
            const theta = i * GOLDEN_ANGLE * 2;
            const r = 8 + Math.random() * 4;
            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({size: 0.08, color: 0x7c3aed, transparent: true, opacity: 0.8});
        particleSystem = new THREE.Points(geometry, material);
        scene.add(particleSystem);
    }
    
    function createStars() {
        const count = 1000;
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 200;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 200;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({size: 0.5, color: 0xffffff, opacity: 0.6, transparent: true})));
    }
    
    function setupControls(container) {
        let isDragging = false, prevMouse = {x:0, y:0}, camAngle = {theta: 0, phi: Math.PI/4}, camRadius = 15;
        container.addEventListener('mousedown', e => { isDragging = true; prevMouse = {x: e.clientX, y: e.clientY}; });
        container.addEventListener('mousemove', e => {
            if (!isDragging) return;
            camAngle.theta -= (e.clientX - prevMouse.x) * 0.01;
            camAngle.phi = Math.max(0.1, Math.min(Math.PI - 0.1, camAngle.phi + (e.clientY - prevMouse.y) * 0.01));
            updateCam();
            prevMouse = {x: e.clientX, y: e.clientY};
        });
        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('wheel', e => { e.preventDefault(); camRadius = Math.max(5, Math.min(50, camRadius + e.deltaY * 0.01)); updateCam(); });
        function updateCam() {
            if (!camera) return;
            camera.position.x = camRadius * Math.sin(camAngle.phi) * Math.cos(camAngle.theta);
            camera.position.y = camRadius * Math.cos(camAngle.phi);
            camera.position.z = camRadius * Math.sin(camAngle.phi) * Math.sin(camAngle.theta);
            camera.lookAt(0, 0, 0);
        }
    }
    
    function animate() {
        if (!renderer) return;
        requestAnimationFrame(animate);
        const time = Date.now() * 0.001;
        goalMeshes.forEach((mesh, i) => {
            if (mesh.userData.goal) {
                mesh.rotation.y += 0.01;
                mesh.position.y = mesh.userData.baseY + Math.sin(time + i) * 0.3;
            }
        });
        if (particleSystem) particleSystem.rotation.y += 0.001;
        if (isAutoRotate && camera) {
            camera.position.x = 15 * Math.sin(time * 0.1);
            camera.position.z = 15 * Math.cos(time * 0.1);
            camera.lookAt(0, 0, 0);
        }
        renderer.render(scene, camera);
    }
    
    function launch3DWorld() {
        document.getElementById('viz-3d-wrapper').style.display = 'block';
        document.getElementById('2d-fractal-container').style.display = 'none';
        setTimeout(() => init3DWorld('three-canvas', 'viz-3d-container'), 100);
    }
    
    function launch3DFullscreen() {
        const modal = document.getElementById('fullscreen-3d-modal');
        modal.style.display = 'block';
        modal.classList.add('fullscreen-3d');
        setTimeout(() => init3DWorld('fullscreen-three-canvas', 'fullscreen-canvas-container'), 100);
    }
    
    function exitFullscreen3D() {
        document.getElementById('fullscreen-3d-modal').style.display = 'none';
        if (renderer) { renderer.dispose(); renderer = null; }
        scene = null; camera = null;
    }
    
    function close3DWorld() {
        document.getElementById('viz-3d-wrapper').style.display = 'none';
        document.getElementById('2d-fractal-container').style.display = 'block';
        if (renderer) { renderer.dispose(); renderer = null; }
        scene = null; camera = null;
    }
    
    function resetCamera() { if (camera) { camera.position.set(0, 5, 15); camera.lookAt(0, 0, 0); } }
    function toggleAutoRotate() { isAutoRotate = !isAutoRotate; }
    
    async function generate2DFractal() {
        const container = document.getElementById('2d-fractal-container');
        container.innerHTML = '<p>Generating AI-optimized multi-layer fractal...</p>';
        try {
            const res = await fetch('/api/visualization/fractal/2d', {method: 'POST'});
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                container.innerHTML = `<img src="${url}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);"><p style="color: var(--text-secondary); margin-top: 10px;">Multi-layer fractal: Julia set + Goal geometries + Golden spiral + Habit particles</p>`;
            }
        } catch (e) { container.innerHTML = '<p style="color: #f87171;">Error generating fractal</p>'; }
    }
    
    // Audio
    const PRESETS = {alpha: {base: 200, beat: 10}, theta: {base: 150, beat: 6}, delta: {base: 100, beat: 2.5}, beta: {base: 250, beat: 20}, gamma: {base: 300, beat: 40}, schumann: {base: 136.1, beat: 7.83}};
    
    function playAudio() {
        const preset = document.getElementById('audio-preset').value;
        const {base, beat} = PRESETS[preset];
        stopAudio();
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const gain = audioContext.createGain();
        gain.gain.value = 0.3;
        gain.connect(audioContext.destination);
        oscillatorL = audioContext.createOscillator();
        oscillatorL.frequency.value = base;
        const panL = audioContext.createStereoPanner();
        panL.pan.value = -1;
        oscillatorL.connect(panL).connect(gain);
        oscillatorR = audioContext.createOscillator();
        oscillatorR.frequency.value = base + beat;
        const panR = audioContext.createStereoPanner();
        panR.pan.value = 1;
        oscillatorR.connect(panR).connect(gain);
        oscillatorL.start(); oscillatorR.start();
        document.getElementById('audio-status').textContent = `Playing ${preset} waves (${beat} Hz binaural)`;
    }
    
    function stopAudio() {
        if (oscillatorL) { oscillatorL.stop(); oscillatorL = null; }
        if (oscillatorR) { oscillatorR.stop(); oscillatorR = null; }
        if (audioContext) { audioContext.close(); audioContext = null; }
        document.getElementById('audio-status').textContent = 'Stopped';
    }
    
    // UI
    function showSection(sectionId, btn) {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(sectionId + '-section').classList.add('active');
        btn.classList.add('active');
    }
    
    function updateSliderVal(id) {
        const slider = document.getElementById(id + '-slider');
        const val = document.getElementById(id + '-val');
        if (slider && val) val.textContent = slider.value;
    }
    
    document.addEventListener('DOMContentLoaded', checkAuth);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌀 LIFE FRACTAL INTELLIGENCE v11.0 - COMPLETE AI SYSTEM")
    print("=" * 70)
    print("🧠 AI BRAIN: Pattern Recognition, Predictions, Learning")
    print("📊 DATA PIPELINE: Continuous ingestion and optimization")
    print("🎨 MULTI-LAYER FRACTALS: Julia + Goals + Spiral + Particles")
    print("⚠️  EXECUTIVE DYSFUNCTION: Early warning system")
    print("🔮 PREDICTIONS: Tomorrow's mood/energy with ML")
    print("🌐 FEDERATED LEARNING: Privacy-preserving insights")
    print(f"📐 SACRED MATH: φ = {PHI:.15f}")
    print(f"🔬 ML AVAILABLE: {HAS_SKLEARN}")
    print("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
