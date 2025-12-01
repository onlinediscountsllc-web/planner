#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v10.0 - NEURODIVERGENT WELLNESS SYSTEM
═══════════════════════════════════════════════════════════════════════════════

COMPLETE UNIFIED INTELLIGENCE SYSTEM:

🥄 SPOON/ENERGY THEORY:
- Visual energy tracking (pebbles/spoons)
- Activity cost calculation
- Recovery time prediction
- Burnout prevention alerts
- Energy visualization in fractals

🧮 UNIFIED PREDICTIVE MATH ENGINE:
- Centralized algorithm for ALL predictions
- Golden Ratio / Fibonacci natural rhythms
- Logistic map for chaos/entropy modeling
- Exponential decay for recovery
- Weighted moving averages for trends
- Harmonic means for wellness scoring

🧠 EXECUTIVE DYSFUNCTION MODELING:
- Cognitive load scoring
- Decision fatigue tracking
- Task switching costs
- Recovery time prediction
- Deficit accumulation

🌅 MAYAN CALENDAR INTEGRATION:
- Tzolkin (260-day sacred calendar)
- Haab (365-day civil calendar)
- Seasonal energy patterns
- Personal day sign calculation

🌊 ENTROPY & CHAOS THEORY:
- Predict consequences of choices
- Model symbiotic relationship effects
- Butterfly effect visualization
- Trajectory forecasting

🐾 VIRTUAL PET WELLNESS MIRROR:
- Reflects user's energy level
- Shows stress/anxiety visually
- Warns of burnout risk
- Celebrates recovery

🎨 FRACTAL VISUALIZATION:
- 2D/3D driven by ALL metrics
- Energy level → brightness
- Stress → chaos parameter
- Recovery → zoom level
- Predictions → color gradients

💬 NLP & SENTIMENT:
- Basic sentiment analysis (no sklearn needed)
- Pattern-based mood detection
- Voice input support
- Conversational responses

✅ RENDER COMPATIBLE - No sklearn required!

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import re
import secrets
import logging
import colorsys
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from io import BytesIO
from functools import wraps
import base64

# Flask
from flask import Flask, request, jsonify, session, render_template_string, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Mayan Calendar Constants
TZOLKIN_DAYS = 260  # Sacred calendar
HAAB_DAYS = 365  # Civil calendar
CALENDAR_ROUND = 18980  # Days until both calendars align (52 Haab years)

# Mayan Day Signs (Tzolkin)
MAYAN_DAY_SIGNS = [
    ('Imix', '🐊', 'Crocodile - New beginnings, primordial energy'),
    ('Ik', '💨', 'Wind - Communication, breath of life'),
    ('Akbal', '🌙', 'Night - Dreams, introspection, mystery'),
    ('Kan', '🌽', 'Seed - Potential, fertility, new ideas'),
    ('Chicchan', '🐍', 'Serpent - Life force, kundalini, transformation'),
    ('Cimi', '💀', 'Death - Transformation, ancestors, rebirth'),
    ('Manik', '🦌', 'Deer - Healing, gentleness, forest energy'),
    ('Lamat', '⭐', 'Star - Harmony, beauty, abundance'),
    ('Muluc', '💧', 'Water - Emotions, purification, flow'),
    ('Oc', '🐕', 'Dog - Loyalty, guidance, companionship'),
    ('Chuen', '🐒', 'Monkey - Play, creativity, artistry'),
    ('Eb', '🌿', 'Grass - Path, journey, community'),
    ('Ben', '🎋', 'Reed - Authority, knowledge, pillars'),
    ('Ix', '🐆', 'Jaguar - Earth magic, feminine power'),
    ('Men', '🦅', 'Eagle - Vision, freedom, higher perspective'),
    ('Cib', '🦉', 'Owl - Wisdom, karma, ancestral knowledge'),
    ('Caban', '🌍', 'Earth - Grounding, evolution, synchronicity'),
    ('Etznab', '🔪', 'Flint - Truth, clarity, reflection'),
    ('Cauac', '⛈️', 'Storm - Catalysis, healing energy'),
    ('Ahau', '☀️', 'Sun - Enlightenment, mastery, completion'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 🛡️ SELF-HEALING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingSystem:
    def __init__(self):
        self.error_counts = {}
        self.component_status = {}
        self.start_time = datetime.now(timezone.utc)
    
    def record_error(self, component: str, error: str):
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        self.component_status[component] = 'error'
        logger.warning(f"🛡️ Error in {component}: {error}")
    
    def mark_healthy(self, component: str):
        self.component_status[component] = 'healthy'
    
    def get_health_report(self) -> dict:
        total_errors = sum(self.error_counts.values())
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {'overall_health': 'excellent' if total_errors == 0 else 'healthy',
                'uptime_seconds': uptime, 'error_counts': self.error_counts}

HEALER = SelfHealingSystem()

def safe_execute(fallback_value=None, component="unknown"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                HEALER.mark_healthy(component)
                return result
            except Exception as e:
                HEALER.record_error(component, str(e))
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value() if callable(fallback_value) else fallback_value
        return wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════════════════
# 🥄 SPOON/ENERGY THEORY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SpoonTheoryEngine:
    """
    Manages energy (spoons/pebbles) for neurodivergent users.
    Based on Christine Miserandino's Spoon Theory adapted for autism.
    """
    
    # Default daily spoon allocation
    DEFAULT_DAILY_SPOONS = 12
    
    # Activity costs (spoons)
    ACTIVITY_COSTS = {
        # Basic daily tasks
        'wake_up': 1,
        'shower': 2,
        'get_dressed': 1,
        'eat_meal': 1,
        'commute': 2,
        
        # Work/Cognitive
        'focused_work_hour': 2,
        'meeting': 2,
        'email': 1,
        'phone_call': 2,
        'decision_making': 2,
        'learning_new': 3,
        'task_switching': 1,
        
        # Social
        'social_small_talk': 2,
        'social_deep_conversation': 2,
        'social_group': 3,
        'social_new_people': 4,
        'masking': 3,
        
        # Sensory
        'noisy_environment': 2,
        'bright_lights': 1,
        'crowded_space': 3,
        'unexpected_change': 3,
        
        # Recovery activities (negative = restore spoons)
        'rest_quiet': -1,
        'special_interest': -2,
        'nature_walk': -1,
        'sleep_good': -4,
        'meditation': -1,
    }
    
    # Executive dysfunction multipliers
    ED_MULTIPLIERS = {
        'none': 1.0,
        'mild': 1.3,
        'moderate': 1.6,
        'severe': 2.0,
        'crisis': 2.5,
    }
    
    def __init__(self):
        pass
    
    def calculate_daily_spoons(self, user_data: dict) -> int:
        """Calculate available spoons based on user state."""
        base = user_data.get('base_daily_spoons', self.DEFAULT_DAILY_SPOONS)
        
        # Modifiers
        sleep_quality = user_data.get('sleep_quality', 50) / 100
        stress_level = user_data.get('stress_level', 50) / 100
        recovery_debt = user_data.get('recovery_debt', 0)
        
        # Sleep affects starting spoons
        sleep_modifier = 0.5 + sleep_quality * 0.5  # 50-100% of base
        
        # Stress reduces available spoons
        stress_modifier = 1.0 - (stress_level * 0.3)  # -0 to -30%
        
        # Recovery debt carries over
        debt_modifier = max(0, 1 - recovery_debt * 0.1)  # Each point = -10%
        
        available = int(base * sleep_modifier * stress_modifier * debt_modifier)
        return max(1, available)  # Always at least 1 spoon
    
    def calculate_activity_cost(self, activity: str, context: dict = None) -> float:
        """Calculate spoon cost for an activity with context modifiers."""
        base_cost = self.ACTIVITY_COSTS.get(activity, 1)
        
        if context is None:
            return base_cost
        
        # Executive dysfunction multiplier
        ed_level = context.get('executive_dysfunction', 'none')
        ed_mult = self.ED_MULTIPLIERS.get(ed_level, 1.0)
        
        # Anxiety multiplier
        anxiety = context.get('anxiety_level', 0) / 100
        anxiety_mult = 1.0 + anxiety * 0.5  # Up to 50% more
        
        # Sensory overload multiplier
        sensory = context.get('sensory_overload', 0) / 100
        sensory_mult = 1.0 + sensory * 0.3  # Up to 30% more
        
        # Time of day (afternoon slump)
        hour = context.get('hour', 12)
        if 14 <= hour <= 16:
            time_mult = 1.2  # 20% more in afternoon
        else:
            time_mult = 1.0
        
        total_cost = base_cost * ed_mult * anxiety_mult * sensory_mult * time_mult
        return round(total_cost, 1)
    
    def predict_recovery_time(self, current_spoons: int, deficit: int, 
                              sleep_hours: float = 7, rest_quality: float = 0.5) -> dict:
        """Predict how long to recover from energy deficit."""
        if deficit <= 0:
            return {'hours': 0, 'days': 0, 'status': 'no_deficit'}
        
        # Recovery rate per hour of good sleep
        base_recovery = 0.5  # spoons per hour of sleep
        recovery_per_sleep = base_recovery * rest_quality * sleep_hours
        
        # Waking rest recovery
        waking_recovery = 0.1  # spoons per hour of rest while awake
        
        # Daily recovery potential
        daily_recovery = recovery_per_sleep + (waking_recovery * 4)  # 4 hours rest
        
        # Time to recover
        days_to_recover = deficit / daily_recovery if daily_recovery > 0 else 999
        hours_to_recover = days_to_recover * 24
        
        # Status
        if days_to_recover <= 1:
            status = 'quick_recovery'
        elif days_to_recover <= 3:
            status = 'moderate_recovery'
        elif days_to_recover <= 7:
            status = 'extended_recovery'
        else:
            status = 'burnout_risk'
        
        return {
            'hours': round(hours_to_recover, 1),
            'days': round(days_to_recover, 1),
            'daily_recovery': round(daily_recovery, 1),
            'status': status
        }
    
    def get_burnout_risk(self, spoon_history: List[int], current_spoons: int) -> dict:
        """Calculate burnout risk based on spoon patterns."""
        if not spoon_history:
            return {'risk': 'unknown', 'score': 50, 'trend': 'stable'}
        
        # Calculate trend
        avg_recent = sum(spoon_history[-7:]) / len(spoon_history[-7:]) if len(spoon_history) >= 7 else sum(spoon_history) / len(spoon_history)
        avg_older = sum(spoon_history[:-7]) / len(spoon_history[:-7]) if len(spoon_history) > 7 else avg_recent
        
        trend_change = (avg_recent - avg_older) / max(avg_older, 1) * 100
        
        # Risk score (0-100)
        deficit_factor = max(0, (self.DEFAULT_DAILY_SPOONS - current_spoons) / self.DEFAULT_DAILY_SPOONS) * 50
        trend_factor = max(0, -trend_change) * 0.5  # Negative trend = higher risk
        
        risk_score = min(100, deficit_factor + trend_factor)
        
        if risk_score < 25:
            risk = 'low'
        elif risk_score < 50:
            risk = 'moderate'
        elif risk_score < 75:
            risk = 'high'
        else:
            risk = 'critical'
        
        trend = 'improving' if trend_change > 5 else ('declining' if trend_change < -5 else 'stable')
        
        return {'risk': risk, 'score': round(risk_score), 'trend': trend}

# ═══════════════════════════════════════════════════════════════════════════════
# 🧮 UNIFIED PREDICTIVE MATH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedPredictiveEngine:
    """
    Centralized mathematical engine for ALL predictions.
    Uses sacred geometry, chaos theory, and statistical methods.
    """
    
    def __init__(self):
        self.spoon_engine = SpoonTheoryEngine()
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE MATHEMATICAL FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def golden_ratio_weight(self, index: int) -> float:
        """Calculate weight using golden ratio decay."""
        return PHI_INVERSE ** index
    
    def fibonacci_nearest(self, value: float) -> int:
        """Find nearest Fibonacci number."""
        for i, f in enumerate(FIBONACCI):
            if f >= value:
                return f
        return FIBONACCI[-1]
    
    def logistic_map(self, r: float, x: float, iterations: int = 1) -> float:
        """
        Logistic map for chaos modeling.
        r: chaos parameter (3.57-4.0 for chaos)
        x: current state (0-1)
        """
        for _ in range(iterations):
            x = r * x * (1 - x)
        return x
    
    def exponential_decay(self, initial: float, decay_rate: float, time: float) -> float:
        """Calculate exponential decay for recovery modeling."""
        return initial * math.exp(-decay_rate * time)
    
    def harmonic_mean(self, values: List[float]) -> float:
        """Harmonic mean - penalizes low values more than arithmetic mean."""
        if not values or 0 in values:
            return 0
        return len(values) / sum(1/v for v in values if v > 0)
    
    def weighted_moving_average(self, values: List[float], weights: List[float] = None) -> float:
        """Calculate weighted moving average with golden ratio weights."""
        if not values:
            return 0
        
        if weights is None:
            # Use golden ratio decay for weights
            weights = [self.golden_ratio_weight(i) for i in range(len(values))]
        
        # Reverse so most recent has highest weight
        values = list(reversed(values))
        weights = weights[:len(values)]
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0
        
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    
    def entropy_score(self, values: List[float]) -> float:
        """Calculate entropy/chaos score from value distribution."""
        if not values or len(values) < 2:
            return 0
        
        # Normalize to probabilities
        total = sum(abs(v) for v in values)
        if total == 0:
            return 0
        
        probs = [abs(v) / total for v in values]
        
        # Shannon entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        
        # Normalize to 0-100
        max_entropy = math.log(len(values))
        return (entropy / max_entropy * 100) if max_entropy > 0 else 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAYAN CALENDAR CALCULATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_mayan_day_sign(self, date: datetime = None) -> dict:
        """Calculate Mayan day sign for a date."""
        if date is None:
            date = datetime.now(timezone.utc)
        
        # Mayan creation date correlation (GMT: August 11, 3114 BCE)
        mayan_epoch = datetime(year=1, month=1, day=1, tzinfo=timezone.utc)
        # We use a simplified correlation from a known date
        # December 21, 2012 was 13.0.0.0.0 (end of 13th Baktun) = 4 Ahau 3 Kankin
        
        reference_date = datetime(2012, 12, 21, tzinfo=timezone.utc)
        days_since_ref = (date - reference_date).days
        
        # Tzolkin calculation (260-day cycle)
        # Reference: 4 Ahau on Dec 21, 2012
        tzolkin_day_num = (4 + days_since_ref - 1) % 13 + 1  # 1-13
        tzolkin_sign_idx = (19 + days_since_ref) % 20  # Ahau is index 19
        
        sign = MAYAN_DAY_SIGNS[tzolkin_sign_idx]
        
        # Haab calculation (365-day cycle)
        # Reference: 3 Kankin on Dec 21, 2012
        haab_months = ['Pop', 'Uo', 'Zip', 'Zotz', 'Tzec', 'Xul', 'Yaxkin', 'Mol',
                       'Chen', 'Yax', 'Zac', 'Ceh', 'Mac', 'Kankin', 'Muan', 'Pax',
                       'Kayab', 'Cumku', 'Wayeb']
        
        haab_day_of_year = (3 + 13 * 20 + days_since_ref) % 365  # 3 Kankin + Kankin start
        haab_month_idx = haab_day_of_year // 20
        haab_day = haab_day_of_year % 20
        
        if haab_month_idx >= 18:
            haab_month = 'Wayeb'
            haab_day = haab_day_of_year - 360
        else:
            haab_month = haab_months[haab_month_idx]
        
        # Energy calculation based on day sign
        sign_energies = {
            0: {'creation': 80, 'transformation': 60},  # Imix
            1: {'communication': 90, 'clarity': 70},    # Ik
            2: {'introspection': 85, 'dreams': 90},     # Akbal
            3: {'potential': 80, 'growth': 85},         # Kan
            4: {'vitality': 90, 'transformation': 85},  # Chicchan
            5: {'release': 70, 'rebirth': 80},          # Cimi
            6: {'healing': 90, 'gentleness': 85},       # Manik
            7: {'harmony': 85, 'abundance': 80},        # Lamat
            8: {'emotions': 80, 'flow': 75},            # Muluc
            9: {'loyalty': 85, 'guidance': 80},         # Oc
            10: {'creativity': 90, 'play': 85},         # Chuen
            11: {'journey': 75, 'community': 80},       # Eb
            12: {'authority': 80, 'knowledge': 85},     # Ben
            13: {'magic': 85, 'intuition': 90},         # Ix
            14: {'vision': 90, 'freedom': 85},          # Men
            15: {'wisdom': 85, 'karma': 70},            # Cib
            16: {'grounding': 80, 'evolution': 85},     # Caban
            17: {'truth': 75, 'clarity': 90},           # Etznab
            18: {'catalysis': 70, 'healing': 85},       # Cauac
            19: {'enlightenment': 95, 'mastery': 90},   # Ahau
        }
        
        energy = sign_energies.get(tzolkin_sign_idx, {'base': 75})
        
        return {
            'tzolkin': f"{tzolkin_day_num} {sign[0]}",
            'haab': f"{haab_day} {haab_month}",
            'day_sign': sign[0],
            'day_sign_emoji': sign[1],
            'day_sign_meaning': sign[2],
            'day_number': tzolkin_day_num,
            'energy': energy,
            'cycle_day': days_since_ref % TZOLKIN_DAYS + 1
        }
    
    def get_mayan_energy_forecast(self, days: int = 7) -> List[dict]:
        """Get Mayan energy forecast for upcoming days."""
        forecasts = []
        today = datetime.now(timezone.utc)
        
        for i in range(days):
            date = today + timedelta(days=i)
            day_info = self.get_mayan_day_sign(date)
            
            # Calculate overall energy score
            energies = list(day_info['energy'].values())
            avg_energy = sum(energies) / len(energies) if energies else 75
            
            forecasts.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_name': date.strftime('%A'),
                'tzolkin': day_info['tzolkin'],
                'day_sign': day_info['day_sign'],
                'emoji': day_info['day_sign_emoji'],
                'meaning': day_info['day_sign_meaning'],
                'energy_score': round(avg_energy),
                'day_number': day_info['day_number']
            })
        
        return forecasts
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTIVE DYSFUNCTION MODELING
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_executive_load(self, tasks: List[dict], current_state: dict) -> dict:
        """Calculate total executive function load."""
        if not tasks:
            return {'total_load': 0, 'capacity_used': 0, 'status': 'clear'}
        
        # Base capacity (can be reduced by stress, sleep, etc.)
        base_capacity = 100
        
        # Modifiers
        stress_mod = 1 - (current_state.get('stress_level', 50) / 200)  # -50% at max stress
        sleep_mod = current_state.get('sleep_quality', 50) / 100  # 0-100%
        energy_mod = current_state.get('energy_level', 50) / 100
        
        effective_capacity = base_capacity * stress_mod * (0.5 + sleep_mod * 0.5) * (0.5 + energy_mod * 0.5)
        
        # Calculate task loads
        total_load = 0
        for task in tasks:
            complexity = task.get('complexity', 1)  # 1-5
            novelty = task.get('novelty', 1)  # 1-3 (new tasks cost more)
            switching = task.get('context_switch', False)  # Additional cost
            
            task_load = complexity * 5 * novelty
            if switching:
                task_load *= 1.5
            
            total_load += task_load
        
        capacity_used = (total_load / effective_capacity * 100) if effective_capacity > 0 else 100
        
        if capacity_used < 50:
            status = 'comfortable'
        elif capacity_used < 75:
            status = 'manageable'
        elif capacity_used < 100:
            status = 'strained'
        else:
            status = 'overloaded'
        
        return {
            'total_load': round(total_load),
            'effective_capacity': round(effective_capacity),
            'capacity_used': round(min(100, capacity_used)),
            'status': status,
            'overflow': max(0, round(total_load - effective_capacity))
        }
    
    def predict_executive_recovery(self, current_load: float, rest_activities: List[str]) -> dict:
        """Predict recovery from executive dysfunction."""
        if current_load <= 0:
            return {'hours': 0, 'activities_needed': [], 'status': 'recovered'}
        
        # Recovery rates per hour for different activities
        recovery_rates = {
            'sleep': 5.0,
            'rest_quiet': 2.0,
            'nature': 3.0,
            'special_interest': 4.0,
            'meditation': 2.5,
            'sensory_break': 2.0,
            'social_recovery': 1.5,  # For those who find social restorative
        }
        
        total_recovery_rate = sum(recovery_rates.get(act, 1.0) for act in rest_activities)
        if not rest_activities:
            total_recovery_rate = 1.0  # Passive recovery
        
        hours_needed = current_load / total_recovery_rate
        
        if hours_needed <= 2:
            status = 'quick_recovery'
        elif hours_needed <= 8:
            status = 'rest_day_needed'
        elif hours_needed <= 24:
            status = 'extended_recovery'
        else:
            status = 'burnout_protocol'
        
        return {
            'hours': round(hours_needed, 1),
            'recovery_rate': round(total_recovery_rate, 1),
            'status': status,
            'recommended_activities': ['sleep', 'special_interest', 'nature']
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def predict_all(self, user_state: dict, history: List[dict]) -> dict:
        """
        Generate comprehensive predictions using all mathematical models.
        This is the main entry point for the prediction engine.
        """
        predictions = {}
        
        # 1. Mood Prediction (using weighted moving average)
        mood_history = [h.get('mood_score', 50) for h in history[-14:]]
        if mood_history:
            predicted_mood = self.weighted_moving_average(mood_history)
            mood_trend = mood_history[-1] - mood_history[0] if len(mood_history) > 1 else 0
            predictions['mood'] = {
                'predicted': round(predicted_mood),
                'trend': 'improving' if mood_trend > 5 else ('declining' if mood_trend < -5 else 'stable'),
                'confidence': 'high' if len(mood_history) >= 7 else 'medium'
            }
        else:
            predictions['mood'] = {'predicted': 50, 'trend': 'unknown', 'confidence': 'low'}
        
        # 2. Energy/Spoon Prediction
        available_spoons = self.spoon_engine.calculate_daily_spoons(user_state)
        spoon_history = [h.get('energy_level', 50) // 10 for h in history[-7:]]
        burnout_risk = self.spoon_engine.get_burnout_risk(spoon_history, available_spoons)
        
        predictions['energy'] = {
            'available_spoons': available_spoons,
            'burnout_risk': burnout_risk['risk'],
            'risk_score': burnout_risk['score'],
            'trend': burnout_risk['trend']
        }
        
        # 3. Chaos/Entropy Score
        recent_moods = [h.get('mood_score', 50) for h in history[-7:]]
        recent_stress = [h.get('stress_level', 50) for h in history[-7:]]
        all_recent = recent_moods + recent_stress
        
        entropy = self.entropy_score(all_recent) if all_recent else 50
        
        # Calculate chaos parameter for fractal
        avg_stress = sum(recent_stress) / len(recent_stress) if recent_stress else 50
        chaos_r = 3.57 + (avg_stress / 100) * 0.43  # 3.57-4.0 range
        
        predictions['chaos'] = {
            'entropy_score': round(entropy),
            'chaos_parameter': round(chaos_r, 3),
            'stability': 'stable' if entropy < 30 else ('variable' if entropy < 60 else 'chaotic')
        }
        
        # 4. Mayan Calendar
        mayan = self.get_mayan_day_sign()
        predictions['mayan'] = {
            'today': mayan['tzolkin'],
            'day_sign': mayan['day_sign'],
            'emoji': mayan['day_sign_emoji'],
            'energy': mayan['energy'],
            'meaning': mayan['day_sign_meaning']
        }
        
        # 5. Recovery Prediction
        current_energy = user_state.get('energy_level', 50)
        target_energy = 80  # Target for "recovered"
        deficit = max(0, target_energy - current_energy)
        
        if deficit > 0:
            recovery = self.spoon_engine.predict_recovery_time(
                current_energy // 10,
                deficit // 10,
                user_state.get('sleep_hours', 7),
                user_state.get('sleep_quality', 50) / 100
            )
            predictions['recovery'] = recovery
        else:
            predictions['recovery'] = {'hours': 0, 'days': 0, 'status': 'recovered'}
        
        # 6. Goal Trajectory
        goals = user_state.get('goals', [])
        if goals:
            avg_progress = sum(g.get('progress', 0) for g in goals) / len(goals)
            progress_velocity = avg_progress / max(1, len(history))  # Progress per day
            
            predictions['goals'] = {
                'average_progress': round(avg_progress),
                'velocity': round(progress_velocity, 2),
                'projected_completion_days': round((100 - avg_progress) / max(0.1, progress_velocity)) if progress_velocity > 0 else 999
            }
        
        # 7. Overall Wellness Trajectory
        wellness_history = []
        for h in history[-14:]:
            mood = h.get('mood_score', 50)
            energy = h.get('energy_level', 50)
            stress = 100 - h.get('stress_level', 50)
            wellness_history.append((mood + energy + stress) / 3)
        
        if wellness_history:
            wellness_trend = self.weighted_moving_average(wellness_history)
            predictions['wellness'] = {
                'current': round(wellness_history[-1]) if wellness_history else 50,
                'predicted': round(wellness_trend),
                'trend_direction': 'up' if len(wellness_history) > 1 and wellness_history[-1] > wellness_history[0] else 'down'
            }
        
        # 8. Pet Wellness Correlation
        if 'pet' in user_state:
            pet = user_state['pet']
            user_wellness = predictions.get('wellness', {}).get('current', 50)
            
            # Pet reflects user state
            pet_happiness_target = int(user_wellness * 0.8 + 20)  # 20-100 range
            predictions['pet'] = {
                'happiness_target': pet_happiness_target,
                'stress_indicator': 'high' if burnout_risk['risk'] in ['high', 'critical'] else 'normal',
                'needs_attention': pet.get('happiness', 50) < 40 or pet.get('hunger', 0) > 60
            }
        
        return predictions

# ═══════════════════════════════════════════════════════════════════════════════
# 💬 NLP & SENTIMENT (No sklearn required!)
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """Basic sentiment analysis without ML dependencies."""
    
    # Emotion word lists
    POSITIVE_WORDS = set([
        'happy', 'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'joy', 'excited', 'peaceful', 'calm', 'relaxed', 'grateful',
        'hopeful', 'optimistic', 'proud', 'confident', 'energetic', 'motivated',
        'blessed', 'thankful', 'content', 'satisfied', 'cheerful', 'delighted',
        'accomplished', 'better', 'improving', 'progress', 'success', 'achieved'
    ])
    
    NEGATIVE_WORDS = set([
        'sad', 'bad', 'terrible', 'awful', 'horrible', 'depressed', 'anxious',
        'worried', 'stressed', 'tired', 'exhausted', 'overwhelmed', 'frustrated',
        'angry', 'upset', 'disappointed', 'lonely', 'scared', 'afraid', 'nervous',
        'hopeless', 'helpless', 'worthless', 'guilty', 'ashamed', 'regret',
        'struggling', 'failing', 'stuck', 'lost', 'confused', 'burnt', 'burnout'
    ])
    
    ENERGY_WORDS = {
        'high': ['energetic', 'motivated', 'pumped', 'ready', 'active', 'alive', 'vibrant'],
        'low': ['tired', 'exhausted', 'drained', 'depleted', 'fatigued', 'sleepy', 'weak']
    }
    
    EXECUTIVE_DYSFUNCTION_WORDS = set([
        'cant', 'stuck', 'frozen', 'paralyzed', 'overwhelmed', 'scattered',
        'foggy', 'confused', 'forget', 'forgot', 'distracted', 'unfocused',
        'procrastinating', 'avoiding', 'shutdown', 'meltdown', 'overload'
    ])
    
    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text."""
        if not text:
            return {'sentiment': 'neutral', 'score': 50, 'emotions': [], 'energy': 'neutral'}
        
        # Normalize text
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Count sentiment words
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        # Calculate sentiment score (0-100)
        total = positive_count + negative_count
        if total == 0:
            score = 50
            sentiment = 'neutral'
        else:
            score = int((positive_count / total) * 100)
            if score >= 60:
                sentiment = 'positive'
            elif score <= 40:
                sentiment = 'negative'
            else:
                sentiment = 'mixed'
        
        # Detect specific emotions
        emotions = []
        if words & {'happy', 'joy', 'excited', 'cheerful'}:
            emotions.append('happiness')
        if words & {'sad', 'depressed', 'down', 'unhappy'}:
            emotions.append('sadness')
        if words & {'anxious', 'worried', 'nervous', 'scared'}:
            emotions.append('anxiety')
        if words & {'angry', 'frustrated', 'upset', 'annoyed'}:
            emotions.append('anger')
        if words & {'tired', 'exhausted', 'drained', 'fatigued'}:
            emotions.append('fatigue')
        if words & {'grateful', 'thankful', 'blessed', 'appreciative'}:
            emotions.append('gratitude')
        if words & {'hopeful', 'optimistic', 'looking forward'}:
            emotions.append('hope')
        
        # Energy level
        high_energy = len(words & set(self.ENERGY_WORDS['high']))
        low_energy = len(words & set(self.ENERGY_WORDS['low']))
        
        if high_energy > low_energy:
            energy = 'high'
        elif low_energy > high_energy:
            energy = 'low'
        else:
            energy = 'neutral'
        
        # Executive dysfunction detection
        ed_count = len(words & self.EXECUTIVE_DYSFUNCTION_WORDS)
        executive_dysfunction = ed_count >= 2
        
        return {
            'sentiment': sentiment,
            'score': score,
            'emotions': emotions,
            'energy': energy,
            'executive_dysfunction_detected': executive_dysfunction,
            'word_count': len(words)
        }
    
    def generate_supportive_response(self, analysis: dict) -> str:
        """Generate supportive response based on sentiment analysis."""
        responses = []
        
        if analysis['sentiment'] == 'negative':
            if 'anxiety' in analysis.get('emotions', []):
                responses.append("I notice you might be feeling anxious. Remember: this feeling will pass. 🌬️")
            if 'fatigue' in analysis.get('emotions', []):
                responses.append("It sounds like you're running low on energy. Be gentle with yourself. 💜")
            if analysis.get('executive_dysfunction_detected'):
                responses.append("If you're feeling stuck, try the smallest possible next step. 🐢")
            if not responses:
                responses.append("I hear you. These feelings are valid. You're not alone. 💙")
        
        elif analysis['sentiment'] == 'positive':
            if 'happiness' in analysis.get('emotions', []):
                responses.append("Your positive energy is wonderful! Savor this moment. 🌟")
            if 'gratitude' in analysis.get('emotions', []):
                responses.append("Gratitude is a superpower. Keep noticing the good. ✨")
            if not responses:
                responses.append("You're doing great! Keep nurturing this energy. 🌱")
        
        else:
            responses.append("I'm here with you, whatever you're feeling. 🌈")
        
        return ' '.join(responses)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 JSON DATA STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    def __init__(self):
        self.data_dir = os.environ.get('DATA_DIR', '/tmp/life_fractal_data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.files = {k: os.path.join(self.data_dir, f'{k}.json') 
                     for k in ['users', 'goals', 'habits', 'pets', 'entries', 'spoons']}
        for f in self.files.values():
            if not os.path.exists(f):
                with open(f, 'w') as fp:
                    json.dump({}, fp)
        logger.info(f"✅ DataStore: {self.data_dir}")
    
    def _read(self, key: str) -> dict:
        try:
            with open(self.files[key], 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _write(self, key: str, data: dict):
        with open(self.files[key], 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # User methods
    def get_user(self, uid: str): return self._read('users').get(uid)
    def get_user_by_email(self, email: str):
        for uid, u in self._read('users').items():
            if u.get('email', '').lower() == email.lower():
                u['id'] = uid
                return u
        return None
    def save_user(self, uid: str, data: dict):
        users = self._read('users')
        users[uid] = data
        self._write('users', users)
    
    # Goals
    def get_goals(self, uid: str):
        return [dict(g, id=gid) for gid, g in self._read('goals').items() if g.get('user_id') == uid]
    def save_goal(self, gid: str, data: dict):
        goals = self._read('goals')
        goals[gid] = data
        self._write('goals', goals)
    def delete_goal(self, gid: str):
        goals = self._read('goals')
        goals.pop(gid, None)
        self._write('goals', goals)
    
    # Habits
    def get_habits(self, uid: str):
        return [dict(h, id=hid) for hid, h in self._read('habits').items() if h.get('user_id') == uid]
    def get_habit(self, hid: str):
        h = self._read('habits').get(hid)
        if h: h['id'] = hid
        return h
    def save_habit(self, hid: str, data: dict):
        habits = self._read('habits')
        habits[hid] = data
        self._write('habits', habits)
    
    # Pet
    def get_pet(self, uid: str):
        for pid, p in self._read('pets').items():
            if p.get('user_id') == uid:
                p['id'] = pid
                return p
        return None
    def save_pet(self, pid: str, data: dict):
        pets = self._read('pets')
        pets[pid] = data
        self._write('pets', pets)
    
    # Daily entries
    def get_daily_entry(self, uid: str, date: str):
        return self._read('entries').get(f"{uid}_{date}")
    def save_daily_entry(self, uid: str, date: str, data: dict):
        entries = self._read('entries')
        data.update({'user_id': uid, 'date': date})
        entries[f"{uid}_{date}"] = data
        self._write('entries', entries)
    def get_user_history(self, uid: str, days: int = 30):
        entries = self._read('entries')
        user_entries = [e for k, e in entries.items() if k.startswith(f"{uid}_")]
        user_entries.sort(key=lambda x: x.get('date', ''), reverse=True)
        return user_entries[:days]
    
    # Spoon tracking
    def get_spoon_history(self, uid: str, days: int = 30):
        spoons = self._read('spoons')
        user_spoons = spoons.get(uid, [])
        return user_spoons[-days:]
    def save_spoon_entry(self, uid: str, spoons_used: int, spoons_available: int, activities: List[str]):
        spoons = self._read('spoons')
        if uid not in spoons:
            spoons[uid] = []
        spoons[uid].append({
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'used': spoons_used,
            'available': spoons_available,
            'remaining': spoons_available - spoons_used,
            'activities': activities
        })
        # Keep last 90 days
        spoons[uid] = spoons[uid][-90:]
        self._write('spoons', spoons)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 FRACTAL ENGINE (Driven by predictions)
# ═══════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    def __init__(self, prediction_engine: UnifiedPredictiveEngine):
        self.predictor = prediction_engine
    
    def generate(self, user_state: dict, predictions: dict, size: int = 600) -> bytes:
        """Generate fractal visualization driven by prediction data."""
        
        # Extract parameters from predictions
        mood = predictions.get('mood', {}).get('predicted', 50) / 100
        energy = predictions.get('energy', {}).get('available_spoons', 6) / 12
        chaos_r = predictions.get('chaos', {}).get('chaos_parameter', 3.7)
        entropy = predictions.get('chaos', {}).get('entropy_score', 50) / 100
        burnout_score = predictions.get('energy', {}).get('risk_score', 50) / 100
        
        # Mayan energy influence
        mayan_energy = sum(predictions.get('mayan', {}).get('energy', {}).values()) / 200  # 0-1
        
        # Fractal parameters
        zoom = 1.0 + mood * 2.0 + energy
        max_iter = int(100 + (1 - entropy) * 150)  # More stable = more detail
        
        # Generate Mandelbrot
        x = np.linspace(-2.5/zoom, 1.0/zoom, size)
        y = np.linspace(-1.5/zoom, 1.5/zoom, size)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i
        
        M = M / max_iter
        
        # Create image
        img = Image.new('RGB', (size, size))
        pixels = img.load()
        
        # Color based on state
        hue_base = mood * 0.3 + mayan_energy * 0.1  # Mood affects base hue
        sat_mod = 0.5 + energy * 0.5  # Energy affects saturation
        
        for py in range(size):
            for px in range(size):
                v = M[py, px]
                if v >= 0.99:
                    # Inside set - color based on burnout risk
                    r = int(30 * burnout_score)
                    g = int(10 * (1 - burnout_score))
                    b = 30
                    pixels[px, py] = (r, g, b)
                else:
                    hue = (v + hue_base) % 1.0
                    sat = sat_mod * (0.6 + v * 0.4)
                    val = 0.4 + v * 0.6
                    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                    pixels[px, py] = (int(r*255), int(g*255), int(b*255))
        
        # Draw energy/spoon indicator
        draw = ImageDraw.Draw(img, 'RGBA')
        spoons = predictions.get('energy', {}).get('available_spoons', 6)
        
        # Spoon bar at bottom
        bar_height = 10
        bar_width = size - 40
        spoon_width = bar_width / 12
        
        for i in range(12):
            x1 = 20 + i * spoon_width
            y1 = size - 20 - bar_height
            x2 = x1 + spoon_width - 2
            y2 = size - 20
            
            if i < spoons:
                # Available spoon - golden
                color = (255, 215, 0, 200)
            else:
                # Used spoon - dim
                color = (100, 100, 100, 100)
            
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Golden spiral overlay (if energy is good)
        if energy > 0.4:
            cx, cy = size // 2, size // 2
            points = []
            for i in range(200):
                theta = i * GOLDEN_ANGLE_RAD * 0.1
                r = 5 * math.sqrt(i) * (1 + energy)
                px_s = cx + r * math.cos(theta)
                py_s = cy + r * math.sin(theta)
                if 0 <= px_s < size and 0 <= py_s < size:
                    points.append((px_s, py_s))
            if len(points) > 1:
                spiral_alpha = int(energy * 150)
                draw.line(points, fill=(255, 215, 0, spiral_alpha), width=2)
        
        # Mayan glyph indicator (top right)
        mayan_emoji = predictions.get('mayan', {}).get('emoji', '☀️')
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize engines
prediction_engine = UnifiedPredictiveEngine()
sentiment_analyzer = SentimentAnalyzer()
db = DataStore()
fractal_engine = FractalEngine(prediction_engine)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('ENVIRONMENT') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
CORS(app, supports_credentials=True)

logger.info("🌀 Life Fractal Intelligence v10.0 - Neurodivergent Wellness System")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/health')
def health():
    mayan = prediction_engine.get_mayan_day_sign()
    return jsonify({
        'status': 'healthy',
        'version': '10.0',
        'mayan_day': mayan['tzolkin'],
        'day_sign': f"{mayan['day_sign_emoji']} {mayan['day_sign']}",
        **HEALER.get_health_report()
    })

# ─────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/auth/register', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Registration failed'}), 500), component="register")
def register():
    data = request.get_json() or {}
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or len(password) < 8:
        return jsonify({'error': 'Valid email and password (8+ chars) required'}), 400
    if db.get_user_by_email(email):
        return jsonify({'error': 'Email already registered'}), 400
    
    uid = f"user_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc)
    
    # Calculate Mayan day sign for user
    mayan = prediction_engine.get_mayan_day_sign(now)
    
    db.save_user(uid, {
        'email': email,
        'password_hash': generate_password_hash(password),
        'created_at': now.isoformat(),
        'subscription_status': 'trial',
        'trial_end': (now + timedelta(days=7)).isoformat(),
        'base_daily_spoons': data.get('daily_spoons', 12),
        'mayan_birth_sign': mayan
    })
    
    # Create pet
    pid = f"pet_{secrets.token_hex(8)}"
    db.save_pet(pid, {
        'user_id': uid,
        'name': data.get('pet_name', 'Buddy'),
        'species': data.get('pet_species', 'cat'),
        'level': 1,
        'xp': 0,
        'happiness': 100,
        'hunger': 0,
        'energy': 100,
        'stress': 0,
        'tasks_completed': 0
    })
    
    session['user_id'] = uid
    session.permanent = True
    return jsonify({'success': True, 'user_id': uid, 'mayan_sign': mayan}), 201

@app.route('/api/auth/login', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Login failed'}), 500), component="login")
def login():
    data = request.get_json() or {}
    user = db.get_user_by_email(data.get('email', '').lower().strip())
    
    if not user or not check_password_hash(user['password_hash'], data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user_id'] = user['id']
    session.permanent = True
    return jsonify({'success': True, 'user_id': user['id']})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/me')
@require_auth
def get_me():
    user = db.get_user(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.pop('password_hash', None)
    user['id'] = session['user_id']
    return jsonify({'user': user})

# ─────────────────────────────────────────────────────────────────────────
# PREDICTIONS & INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/predictions')
@require_auth
def get_predictions():
    """Get comprehensive predictions for current user."""
    uid = session['user_id']
    user = db.get_user(uid)
    history = db.get_user_history(uid, 30)
    goals = db.get_goals(uid)
    pet = db.get_pet(uid)
    
    # Build current state
    latest = history[0] if history else {}
    user_state = {
        'mood_score': latest.get('mood_score', 50),
        'energy_level': latest.get('energy_level', 50),
        'stress_level': latest.get('stress_level', 50),
        'sleep_hours': latest.get('sleep_hours', 7),
        'sleep_quality': latest.get('sleep_quality', 50),
        'base_daily_spoons': user.get('base_daily_spoons', 12),
        'goals': goals,
        'pet': pet
    }
    
    predictions = prediction_engine.predict_all(user_state, history)
    
    return jsonify({'predictions': predictions})

@app.route('/api/mayan')
def get_mayan():
    """Get Mayan calendar information."""
    today = prediction_engine.get_mayan_day_sign()
    forecast = prediction_engine.get_mayan_energy_forecast(7)
    return jsonify({'today': today, 'forecast': forecast})

@app.route('/api/spoons')
@require_auth
def get_spoons():
    """Get current spoon/energy status."""
    uid = session['user_id']
    user = db.get_user(uid)
    history = db.get_user_history(uid, 7)
    spoon_history = db.get_spoon_history(uid, 30)
    
    # Calculate current state
    latest = history[0] if history else {}
    user_state = {
        'stress_level': latest.get('stress_level', 50),
        'sleep_quality': latest.get('sleep_quality', 50),
        'recovery_debt': len([s for s in spoon_history[-7:] if s.get('remaining', 0) < 0]),
        'base_daily_spoons': user.get('base_daily_spoons', 12)
    }
    
    available = prediction_engine.spoon_engine.calculate_daily_spoons(user_state)
    recent_spoons = [s.get('remaining', 6) for s in spoon_history[-7:]]
    burnout = prediction_engine.spoon_engine.get_burnout_risk(recent_spoons, available)
    
    return jsonify({
        'available_spoons': available,
        'max_spoons': user_state['base_daily_spoons'],
        'burnout_risk': burnout,
        'history': spoon_history[-7:]
    })

@app.route('/api/spoons/log', methods=['POST'])
@require_auth
def log_spoons():
    """Log spoon usage for activities."""
    uid = session['user_id']
    data = request.get_json() or {}
    
    activities = data.get('activities', [])
    context = data.get('context', {})
    
    # Calculate costs
    total_cost = 0
    activity_costs = []
    for activity in activities:
        cost = prediction_engine.spoon_engine.calculate_activity_cost(activity, context)
        total_cost += cost
        activity_costs.append({'activity': activity, 'cost': cost})
    
    # Get available spoons
    user = db.get_user(uid)
    user_state = {'base_daily_spoons': user.get('base_daily_spoons', 12)}
    available = prediction_engine.spoon_engine.calculate_daily_spoons(user_state)
    
    # Save entry
    db.save_spoon_entry(uid, int(total_cost), available, activities)
    
    remaining = available - total_cost
    
    return jsonify({
        'activities': activity_costs,
        'total_cost': round(total_cost, 1),
        'available': available,
        'remaining': round(remaining, 1),
        'warning': 'Low energy!' if remaining < 2 else None
    })

# ─────────────────────────────────────────────────────────────────────────
# DAILY ENTRY & SENTIMENT
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/daily-entry', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="entry")
def save_daily_entry():
    uid = session['user_id']
    data = request.get_json() or {}
    date = data.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    
    mood = float(data.get('mood_score', 50))
    energy = float(data.get('energy_level', 50))
    stress = float(data.get('stress_level', 50))
    notes = data.get('notes', '')
    
    # Sentiment analysis on notes
    sentiment = sentiment_analyzer.analyze(notes)
    supportive_response = sentiment_analyzer.generate_supportive_response(sentiment)
    
    # Calculate wellness using harmonic mean
    values = [mood, energy, 100 - stress]
    wellness = prediction_engine.harmonic_mean(values)
    
    entry = {
        'mood_score': mood,
        'energy_level': energy,
        'stress_level': stress,
        'sleep_hours': float(data.get('sleep_hours', 7)),
        'sleep_quality': float(data.get('sleep_quality', 50)),
        'notes': notes,
        'sentiment': sentiment,
        'wellness_index': round(wellness, 1)
    }
    
    db.save_daily_entry(uid, date, entry)
    
    # Get predictions
    history = db.get_user_history(uid, 30)
    user_state = {
        'mood_score': mood,
        'energy_level': energy,
        'stress_level': stress,
        'sleep_hours': entry['sleep_hours'],
        'sleep_quality': entry['sleep_quality']
    }
    predictions = prediction_engine.predict_all(user_state, history)
    
    # Update pet based on user state
    pet = db.get_pet(uid)
    if pet:
        # Pet mirrors user's emotional state
        pet['happiness'] = max(0, min(100, int(mood * 0.5 + pet['happiness'] * 0.5)))
        pet['stress'] = max(0, min(100, int(stress * 0.7)))
        pet['energy'] = max(0, min(100, int(energy * 0.6 + 40)))
        pet['xp'] = pet.get('xp', 0) + 10
        if pet['xp'] >= pet.get('level', 1) * 100:
            pet['level'] += 1
            pet['xp'] = 0
        db.save_pet(pet['id'], pet)
    
    return jsonify({
        'success': True,
        'entry': entry,
        'sentiment': sentiment,
        'supportive_message': supportive_response,
        'predictions': predictions,
        'mayan': predictions.get('mayan', {})
    })

# ─────────────────────────────────────────────────────────────────────────
# GOALS & HABITS
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals():
    return jsonify({'goals': db.get_goals(session['user_id'])})

@app.route('/api/goals', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="goal")
def create_goal():
    data = request.get_json() or {}
    gid = f"goal_{secrets.token_hex(8)}"
    goal = {
        'user_id': session['user_id'],
        'title': data.get('title', 'New Goal'),
        'progress': 0,
        'spoon_cost': data.get('spoon_cost', 2),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    db.save_goal(gid, goal)
    goal['id'] = gid
    return jsonify({'success': True, 'goal': goal}), 201

@app.route('/api/goals/<gid>', methods=['PUT'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="goal")
def update_goal(gid):
    data = request.get_json() or {}
    goals = db.get_goals(session['user_id'])
    goal = next((g for g in goals if g['id'] == gid), None)
    if not goal:
        return jsonify({'error': 'Not found'}), 404
    
    for k in ['title', 'progress', 'spoon_cost']:
        if k in data:
            goal[k] = data[k]
    
    db.save_goal(gid, goal)
    return jsonify({'success': True, 'goal': goal})

@app.route('/api/goals/<gid>', methods=['DELETE'])
@require_auth
def delete_goal(gid):
    db.delete_goal(gid)
    return jsonify({'success': True})

@app.route('/api/habits', methods=['GET'])
@require_auth
def get_habits():
    return jsonify({'habits': db.get_habits(session['user_id'])})

@app.route('/api/habits', methods=['POST'])
@safe_execute(fallback_value=lambda: (jsonify({'error': 'Failed'}), 500), component="habit")
def create_habit():
    data = request.get_json() or {}
    hid = f"habit_{secrets.token_hex(8)}"
    habit = {
        'user_id': session['user_id'],
        'name': data.get('name', 'New Habit'),
        'current_streak': 0,
        'longest_streak': 0,
        'spoon_cost': data.get('spoon_cost', 1),
        'completions': [],
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    db.save_habit(hid, habit)
    habit['id'] = hid
    return jsonify({'success': True, 'habit': habit}), 201

@app.route('/api/habits/<hid>/complete', methods=['POST'])
@require_auth
def complete_habit(hid):
    habit = db.get_habit(hid)
    if not habit or habit.get('user_id') != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if today not in habit.get('completions', []):
        habit.setdefault('completions', []).append(today)
        habit['current_streak'] = habit.get('current_streak', 0) + 1
        if habit['current_streak'] > habit.get('longest_streak', 0):
            habit['longest_streak'] = habit['current_streak']
        db.save_habit(hid, habit)
        
        # Pet XP
        pet = db.get_pet(session['user_id'])
        if pet:
            pet['xp'] = pet.get('xp', 0) + 5
            pet['happiness'] = min(100, pet.get('happiness', 50) + 3)
            if pet['xp'] >= pet.get('level', 1) * 100:
                pet['level'] += 1
                pet['xp'] = 0
            db.save_pet(pet['id'], pet)
    
    return jsonify({'success': True, 'habit': habit})

# ─────────────────────────────────────────────────────────────────────────
# PET
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/pet')
@require_auth
def get_pet():
    pet = db.get_pet(session['user_id'])
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    
    # Get predictions to inform pet state
    history = db.get_user_history(session['user_id'], 7)
    if history:
        latest = history[0]
        user_state = {
            'mood_score': latest.get('mood_score', 50),
            'energy_level': latest.get('energy_level', 50),
            'stress_level': latest.get('stress_level', 50)
        }
        predictions = prediction_engine.predict_all(user_state, history)
        pet['burnout_warning'] = predictions.get('energy', {}).get('burnout_risk', 'low') in ['high', 'critical']
        pet['mood_reflection'] = predictions.get('mood', {}).get('trend', 'stable')
    
    return jsonify({'pet': pet})

@app.route('/api/pet/feed', methods=['POST'])
@require_auth
def feed_pet():
    pet = db.get_pet(session['user_id'])
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    pet['hunger'] = max(0, pet.get('hunger', 0) - 30)
    pet['happiness'] = min(100, pet.get('happiness', 50) + 10)
    pet['energy'] = min(100, pet.get('energy', 50) + 15)
    pet['xp'] = pet.get('xp', 0) + 5
    if pet['xp'] >= pet.get('level', 1) * 100:
        pet['level'] += 1
        pet['xp'] = 0
    db.save_pet(pet['id'], pet)
    return jsonify({'success': True, 'pet': pet})

@app.route('/api/pet/play', methods=['POST'])
@require_auth
def play_pet():
    pet = db.get_pet(session['user_id'])
    if not pet:
        return jsonify({'error': 'No pet'}), 404
    pet['happiness'] = min(100, pet.get('happiness', 50) + 20)
    pet['stress'] = max(0, pet.get('stress', 0) - 10)
    pet['hunger'] = min(100, pet.get('hunger', 0) + 10)
    pet['xp'] = pet.get('xp', 0) + 10
    if pet['xp'] >= pet.get('level', 1) * 100:
        pet['level'] += 1
        pet['xp'] = 0
    db.save_pet(pet['id'], pet)
    return jsonify({'success': True, 'pet': pet})

# ─────────────────────────────────────────────────────────────────────────
# FRACTAL & STATS
# ─────────────────────────────────────────────────────────────────────────

@app.route('/api/fractal/generate')
def generate_fractal():
    uid = session.get('user_id')
    size = min(max(int(request.args.get('size', 600)), 200), 1200)
    
    if uid:
        history = db.get_user_history(uid, 14)
        user = db.get_user(uid)
        latest = history[0] if history else {}
        user_state = {
            'mood_score': latest.get('mood_score', 50),
            'energy_level': latest.get('energy_level', 50),
            'stress_level': latest.get('stress_level', 50),
            'base_daily_spoons': user.get('base_daily_spoons', 12) if user else 12
        }
        predictions = prediction_engine.predict_all(user_state, history)
    else:
        user_state = {}
        predictions = {
            'mood': {'predicted': 50},
            'energy': {'available_spoons': 6, 'risk_score': 30},
            'chaos': {'chaos_parameter': 3.7, 'entropy_score': 50},
            'mayan': prediction_engine.get_mayan_day_sign()
        }
    
    img_bytes = fractal_engine.generate(user_state, predictions, size)
    return send_file(BytesIO(img_bytes), mimetype='image/png')

@app.route('/api/stats')
@require_auth
def get_stats():
    uid = session['user_id']
    goals = db.get_goals(uid)
    habits = db.get_habits(uid)
    pet = db.get_pet(uid)
    spoon_history = db.get_spoon_history(uid, 7)
    
    avg_spoons = sum(s.get('remaining', 6) for s in spoon_history) / len(spoon_history) if spoon_history else 6
    
    return jsonify({
        'active_goals': len([g for g in goals if g.get('progress', 0) < 100]),
        'longest_streak': max((h.get('longest_streak', 0) for h in habits), default=0),
        'companion_level': pet.get('level', 1) if pet else 1,
        'avg_spoons': round(avg_spoons, 1),
        'total_habits': len(habits)
    })

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence v10</title>
    <style>
        :root { --primary: #667eea; --secondary: #764ba2; --gold: #f0c420; --success: #48c774; --danger: #ff6b6b; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #333; }
        .header { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; padding: 25px 20px; text-align: center; }
        .header h1 { font-size: 2em; margin-bottom: 5px; }
        .mayan-badge { background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; display: inline-block; margin-top: 10px; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .stat-card .value { font-size: 2em; font-weight: bold; color: var(--primary); }
        .stat-card .label { color: #666; font-size: 0.9em; }
        .spoon-bar { display: flex; gap: 5px; justify-content: center; margin: 15px 0; }
        .spoon { width: 30px; height: 40px; border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; transition: all 0.3s; }
        .spoon.available { background: linear-gradient(135deg, var(--gold), #ffd700); box-shadow: 0 2px 8px rgba(240,196,32,0.5); }
        .spoon.used { background: #ddd; opacity: 0.5; }
        .card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .card h2 { color: var(--primary); margin-bottom: 15px; font-size: 1.2em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 500; }
        input[type="range"] { width: 100%; height: 8px; -webkit-appearance: none; background: linear-gradient(to right, var(--primary), var(--secondary)); border-radius: 10px; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; background: white; border: 3px solid var(--primary); border-radius: 50%; cursor: pointer; }
        .slider-value { text-align: center; font-size: 1.4em; font-weight: bold; color: var(--primary); margin: 5px 0 12px; }
        textarea { width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 1em; min-height: 80px; resize: vertical; }
        .btn { padding: 12px 20px; border: none; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary { background: var(--primary); color: white; }
        .btn-success { background: var(--success); color: white; }
        .btn-gold { background: var(--gold); color: #333; }
        .btn:hover { transform: translateY(-2px); }
        .pet-section { text-align: center; }
        .pet-emoji { font-size: 4em; margin: 10px 0; }
        .pet-stats { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
        .pet-stat { text-align: center; }
        .pet-stat-value { font-size: 1.5em; font-weight: bold; color: var(--primary); }
        .progress-bar { background: #e5e7eb; height: 8px; border-radius: 10px; overflow: hidden; margin-top: 5px; }
        .progress-fill { background: linear-gradient(90deg, var(--primary), var(--secondary)); height: 100%; transition: width 0.3s; }
        .prediction-box { background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); padding: 15px; border-radius: 10px; margin: 10px 0; }
        .mayan-forecast { display: flex; gap: 10px; overflow-x: auto; padding: 10px 0; }
        .mayan-day { text-align: center; padding: 10px; background: rgba(255,255,255,0.8); border-radius: 10px; min-width: 80px; }
        .mayan-day .emoji { font-size: 1.5em; }
        .burnout-indicator { padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
        .burnout-low { background: #d4edda; color: #155724; }
        .burnout-moderate { background: #fff3cd; color: #856404; }
        .burnout-high { background: #f8d7da; color: #721c24; }
        .burnout-critical { background: #721c24; color: white; }
        .sentiment-box { padding: 15px; border-radius: 10px; margin-top: 15px; }
        .sentiment-positive { background: #d4edda; border-left: 4px solid var(--success); }
        .sentiment-negative { background: #f8d7da; border-left: 4px solid var(--danger); }
        .sentiment-neutral { background: #e2e8f0; border-left: 4px solid #718096; }
        .hidden { display: none; }
        .auth-container { max-width: 400px; margin: 50px auto; }
        .item { background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid var(--primary); }
        @media (max-width: 768px) { .header h1 { font-size: 1.5em; } .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div id="app"></div>
    <script>
        const petEmojis = {cat:'🐱', dragon:'🐉', phoenix:'🔥', owl:'🦉', fox:'🦊'};
        let currentUser = null, predictions = null, mayan = null;
        
        async function api(endpoint, method='GET', data=null) {
            const opts = { method, headers: {'Content-Type':'application/json'}, credentials:'include' };
            if (data) opts.body = JSON.stringify(data);
            try {
                const res = await fetch(endpoint, opts);
                return await res.json();
            } catch(e) { return {error: e.message}; }
        }
        
        async function checkAuth() {
            const res = await api('/api/auth/me');
            if (res.user) { currentUser = res.user; await loadData(); renderDashboard(); }
            else renderAuth();
        }
        
        async function loadData() {
            [predictions, mayan] = await Promise.all([api('/api/predictions'), api('/api/mayan')]);
        }
        
        function renderAuth() {
            document.getElementById('app').innerHTML = \`
                <div class="header"><h1>🌀 Life Fractal Intelligence</h1><p>Neurodivergent Wellness System v10</p></div>
                <div class="container auth-container">
                    <div class="card">
                        <div style="display:flex;margin-bottom:20px;">
                            <button class="btn btn-primary" style="flex:1;margin-right:5px" onclick="showTab('login')">Login</button>
                            <button class="btn" style="flex:1;background:#e5e7eb" onclick="showTab('register')">Register</button>
                        </div>
                        <div id="loginForm">
                            <input type="email" id="loginEmail" placeholder="Email" style="width:100%;padding:12px;margin-bottom:10px;border:2px solid #e5e7eb;border-radius:8px">
                            <input type="password" id="loginPassword" placeholder="Password" style="width:100%;padding:12px;margin-bottom:10px;border:2px solid #e5e7eb;border-radius:8px">
                            <button class="btn btn-primary" style="width:100%" onclick="doLogin()">Login</button>
                        </div>
                        <div id="registerForm" class="hidden">
                            <input type="email" id="regEmail" placeholder="Email" style="width:100%;padding:12px;margin-bottom:10px;border:2px solid #e5e7eb;border-radius:8px">
                            <input type="password" id="regPassword" placeholder="Password (8+ chars)" style="width:100%;padding:12px;margin-bottom:10px;border:2px solid #e5e7eb;border-radius:8px">
                            <label>Daily Energy (Spoons): <span id="spoonVal">12</span></label>
                            <input type="range" id="dailySpoons" min="6" max="18" value="12" oninput="document.getElementById('spoonVal').textContent=this.value" style="margin-bottom:15px">
                            <button class="btn btn-success" style="width:100%" onclick="doRegister()">Start Free Trial</button>
                        </div>
                        <div id="authMsg" style="margin-top:15px;color:var(--danger);text-align:center"></div>
                    </div>
                </div>\`;
        }
        
        function showTab(t) {
            document.getElementById('loginForm').classList.toggle('hidden', t!=='login');
            document.getElementById('registerForm').classList.toggle('hidden', t!=='register');
        }
        
        async function doLogin() {
            const res = await api('/api/auth/login', 'POST', {
                email: document.getElementById('loginEmail').value,
                password: document.getElementById('loginPassword').value
            });
            if (res.success) { await loadData(); renderDashboard(); }
            else document.getElementById('authMsg').textContent = res.error || 'Login failed';
        }
        
        async function doRegister() {
            const res = await api('/api/auth/register', 'POST', {
                email: document.getElementById('regEmail').value,
                password: document.getElementById('regPassword').value,
                daily_spoons: parseInt(document.getElementById('dailySpoons').value)
            });
            if (res.success) { await loadData(); renderDashboard(); }
            else document.getElementById('authMsg').textContent = res.error || 'Failed';
        }
        
        async function renderDashboard() {
            const [stats, goals, habits, pet, spoons] = await Promise.all([
                api('/api/stats'), api('/api/goals'), api('/api/habits'), api('/api/pet'), api('/api/spoons')
            ]);
            const pred = predictions?.predictions || {};
            const m = mayan?.today || {};
            const forecast = mayan?.forecast || [];
            
            const burnoutClass = {'low':'burnout-low','moderate':'burnout-moderate','high':'burnout-high','critical':'burnout-critical'}[pred.energy?.burnout_risk || 'low'];
            const availableSpoons = spoons?.available_spoons || 12;
            const maxSpoons = spoons?.max_spoons || 12;
            
            document.getElementById('app').innerHTML = \`
                <div class="header">
                    <h1>🌀 Life Fractal Intelligence</h1>
                    <div class="mayan-badge">\${m.day_sign_emoji || '☀️'} \${m.tzolkin || 'Loading...'}</div>
                </div>
                <div class="container">
                    <div class="stats-row">
                        <div class="stat-card"><div class="value">\${stats.active_goals || 0}</div><div class="label">Active Goals</div></div>
                        <div class="stat-card"><div class="value">\${stats.longest_streak || 0}</div><div class="label">Longest Streak</div></div>
                        <div class="stat-card"><div class="value">\${stats.companion_level || 1}</div><div class="label">Companion Lv</div></div>
                        <div class="stat-card"><div class="value">\${availableSpoons}</div><div class="label">🥄 Spoons</div></div>
                    </div>
                    
                    <div class="card">
                        <h2>🥄 Energy (Spoon Theory)</h2>
                        <div class="spoon-bar">
                            \${Array(maxSpoons).fill(0).map((_, i) => \`<div class="spoon \${i < availableSpoons ? 'available' : 'used'}"></div>\`).join('')}
                        </div>
                        <div class="\${burnoutClass} burnout-indicator">
                            Burnout Risk: \${(pred.energy?.burnout_risk || 'low').toUpperCase()} (\${pred.energy?.risk_score || 0}%)
                        </div>
                        \${pred.recovery?.status !== 'recovered' ? \`<p style="text-align:center;margin-top:10px">Recovery: ~\${pred.recovery?.hours || 0} hours</p>\` : ''}
                    </div>
                    
                    <div class="card">
                        <h2>\${m.day_sign_emoji || '🌅'} Mayan Calendar Energy</h2>
                        <p><strong>\${m.day_sign || ''}</strong>: \${m.day_sign_meaning || ''}</p>
                        <div class="mayan-forecast">
                            \${forecast.slice(0,7).map(d => \`
                                <div class="mayan-day">
                                    <div class="emoji">\${d.emoji}</div>
                                    <div style="font-size:0.8em">\${d.day_name?.slice(0,3)}</div>
                                    <div style="font-size:0.7em;color:#666">\${d.energy_score}%</div>
                                </div>
                            \`).join('')}
                        </div>
                    </div>
                    
                    <div class="grid">
                        <div class="card">
                            <h2>📊 Daily Check-In</h2>
                            <label>Mood (1-100)</label>
                            <input type="range" id="moodSlider" min="1" max="100" value="50" oninput="document.getElementById('moodVal').textContent=this.value">
                            <div class="slider-value" id="moodVal">50</div>
                            
                            <label>Energy (1-100)</label>
                            <input type="range" id="energySlider" min="1" max="100" value="50" oninput="document.getElementById('energyVal').textContent=this.value">
                            <div class="slider-value" id="energyVal">50</div>
                            
                            <label>Stress (1-100)</label>
                            <input type="range" id="stressSlider" min="1" max="100" value="30" oninput="document.getElementById('stressVal').textContent=this.value">
                            <div class="slider-value" id="stressVal">30</div>
                            
                            <label>Notes (voice supported 🎤)</label>
                            <textarea id="notesInput" placeholder="How are you feeling?"></textarea>
                            
                            <button class="btn btn-success" style="width:100%;margin-top:15px" onclick="saveCheckin()">Save Check-In</button>
                            
                            <div id="sentimentResult"></div>
                        </div>
                        
                        <div class="card pet-section">
                            <h2>🐾 \${pet.pet?.name || 'Companion'}</h2>
                            <div class="pet-emoji">\${petEmojis[pet.pet?.species] || '🐱'}</div>
                            \${pet.pet?.burnout_warning ? '<div style="color:var(--danger);font-weight:bold">⚠️ Sensing your stress!</div>' : ''}
                            <div class="pet-stats">
                                <div class="pet-stat"><div class="pet-stat-value">\${pet.pet?.level || 1}</div><div>Level</div></div>
                                <div class="pet-stat"><div class="pet-stat-value">\${pet.pet?.happiness || 0}%</div><div>Happy</div></div>
                                <div class="pet-stat"><div class="pet-stat-value">\${pet.pet?.energy || 0}%</div><div>Energy</div></div>
                            </div>
                            <div class="progress-bar" style="margin:15px 0"><div class="progress-fill" style="width:\${(pet.pet?.xp || 0) % 100}%"></div></div>
                            <div style="display:flex;gap:10px;justify-content:center">
                                <button class="btn btn-primary" onclick="feedPet()">🍖 Feed</button>
                                <button class="btn btn-primary" onclick="playPet()">🎾 Play</button>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h2>🎨 Your Life Fractal</h2>
                            <div style="text-align:center">
                                <img src="/api/fractal/generate?size=350&t=\${Date.now()}" style="max-width:100%;border-radius:12px">
                                <p style="margin-top:10px;color:#666">Chaos: \${pred.chaos?.stability || 'stable'} | Entropy: \${pred.chaos?.entropy_score || 50}%</p>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h2>🎯 Goals</h2>
                            \${(goals.goals || []).map(g => \`
                                <div class="item">
                                    <strong>\${g.title}</strong>
                                    <div class="progress-bar"><div class="progress-fill" style="width:\${g.progress || 0}%"></div></div>
                                    <small>\${g.progress || 0}% | 🥄 \${g.spoon_cost || 2} spoons</small>
                                </div>
                            \`).join('') || '<p style="color:#888">No goals yet</p>'}
                            <button class="btn btn-gold" style="width:100%;margin-top:10px" onclick="addGoal()">+ Add Goal</button>
                        </div>
                        
                        <div class="card">
                            <h2>✨ Habits</h2>
                            \${(habits.habits || []).map(h => \`
                                <div class="item">
                                    <strong>\${h.name}</strong> <span style="color:var(--gold)">🔥 \${h.current_streak || 0}</span>
                                    <button class="btn btn-success" style="float:right;padding:5px 10px" onclick="completeHabit('\${h.id}')">✓</button>
                                </div>
                            \`).join('') || '<p style="color:#888">No habits yet</p>'}
                            <button class="btn btn-gold" style="width:100%;margin-top:10px" onclick="addHabit()">+ Add Habit</button>
                        </div>
                    </div>
                    
                    <div style="text-align:center;margin-top:20px">
                        <a href="/3d"><button class="btn btn-primary" style="margin-right:10px">🌀 View 3D Universe</button></a>
                        <button class="btn" style="background:var(--danger);color:white" onclick="logout()">Logout</button>
                    </div>
                </div>\`;
        }
        
        async function saveCheckin() {
            const res = await api('/api/daily-entry', 'POST', {
                mood_score: parseInt(document.getElementById('moodSlider').value),
                energy_level: parseInt(document.getElementById('energySlider').value),
                stress_level: parseInt(document.getElementById('stressSlider').value),
                notes: document.getElementById('notesInput').value
            });
            
            if (res.success) {
                const s = res.sentiment || {};
                const sentClass = s.sentiment === 'positive' ? 'sentiment-positive' : (s.sentiment === 'negative' ? 'sentiment-negative' : 'sentiment-neutral');
                document.getElementById('sentimentResult').innerHTML = \`
                    <div class="\${sentClass} sentiment-box">
                        <strong>\${res.supportive_message || 'Saved!'}</strong>
                        \${s.emotions?.length ? '<br>Detected: ' + s.emotions.join(', ') : ''}
                        \${s.executive_dysfunction_detected ? '<br>💜 ED detected - be extra gentle with yourself' : ''}
                    </div>\`;
                await loadData();
                setTimeout(() => renderDashboard(), 3000);
            }
        }
        
        async function feedPet() { await api('/api/pet/feed', 'POST'); renderDashboard(); }
        async function playPet() { await api('/api/pet/play', 'POST'); renderDashboard(); }
        async function completeHabit(id) { await api('/api/habits/' + id + '/complete', 'POST'); renderDashboard(); }
        function addGoal() { const t = prompt('Goal:'); if(t) api('/api/goals', 'POST', {title:t}).then(renderDashboard); }
        function addHabit() { const n = prompt('Habit:'); if(n) api('/api/habits', 'POST', {name:n}).then(renderDashboard); }
        async function logout() { await api('/api/auth/logout', 'POST'); renderAuth(); }
        
        checkAuth();
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

# ═══════════════════════════════════════════════════════════════════════════════
# 3D VISUALIZATION ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

THREE_D_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Life Fractal Universe</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #050510; color: white; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; }
        #canvas-container { width: 100vw; height: 100vh; }
        #info-panel { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.85); padding: 20px; border-radius: 15px; border: 2px solid #667eea; max-width: 280px; z-index: 1000; }
        #info-panel h1 { font-size: 1.3em; color: #667eea; margin-bottom: 10px; }
        #goals-panel { position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 15px; border: 2px solid #764ba2; max-width: 260px; max-height: 70vh; overflow-y: auto; z-index: 1000; }
        #goals-panel h2 { color: #764ba2; margin-bottom: 10px; font-size: 1.1em; }
        .goal-item { padding: 10px; margin-bottom: 8px; background: rgba(102,126,234,0.1); border-radius: 8px; border-left: 3px solid #667eea; cursor: pointer; transition: all 0.3s; }
        .goal-item:hover { background: rgba(102,126,234,0.2); transform: translateX(5px); }
        .goal-title { font-weight: bold; margin-bottom: 5px; }
        .goal-progress { height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; margin-top: 8px; }
        .goal-progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #48c774); transition: width 0.5s; }
        #stats-panel { position: absolute; bottom: 20px; left: 20px; background: rgba(0,0,0,0.85); padding: 15px 25px; border-radius: 15px; border: 2px solid #f0c420; z-index: 1000; display: flex; gap: 30px; }
        .stat { text-align: center; }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #f0c420; }
        .stat-label { font-size: 0.8em; color: #aaa; }
        #controls { position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 15px; border: 2px solid #764ba2; z-index: 1000; }
        .control-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
        .control-row label { min-width: 100px; font-size: 0.9em; color: #aaa; }
        .control-row input[type="range"] { flex: 1; accent-color: #667eea; }
        button { padding: 10px 20px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 8px; color: white; cursor: pointer; transition: all 0.3s; margin: 5px; text-decoration: none; display: inline-block; }
        button:hover { transform: scale(1.05); box-shadow: 0 0 20px rgba(102,126,234,0.5); }
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 3000; }
        .spinner { width: 60px; height: 60px; border: 4px solid rgba(102,126,234,0.3); border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .sacred-info { margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85em; color: #888; }
        .sacred-info span { color: #f0c420; }
        #back-btn { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; }
        .spoon-indicator { display: flex; gap: 3px; margin-top: 10px; }
        .spoon { width: 15px; height: 20px; border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; }
        .spoon.available { background: linear-gradient(135deg, #f0c420, #ffd700); }
        .spoon.used { background: #333; }
        .mayan-badge { background: rgba(102,126,234,0.3); padding: 8px 12px; border-radius: 8px; margin-top: 10px; text-align: center; }
    </style>
</head>
<body>
    <div id="loading"><div class="spinner"></div><div>Loading 3D Fractal Universe...</div></div>
    <div id="canvas-container"></div>
    <a href="/" id="back-btn"><button>← Back to Dashboard</button></a>
    <div id="info-panel">
        <h1>🌀 3D Life Fractal</h1>
        <p style="color:#aaa;margin-bottom:10px;font-size:0.9em;">Your goals mapped in sacred geometry space.</p>
        <div style="font-size:0.85em;color:#aaa;">
            <div>🖱️ Drag: Rotate</div>
            <div>📜 Scroll: Zoom</div>
            <div>🎯 Click orbs for details</div>
        </div>
        <div class="sacred-info">
            <div>φ (Golden Ratio): <span>1.618033...</span></div>
            <div>Golden Angle: <span>137.5°</span></div>
            <div>Frame: <span id="frame-count">0</span></div>
        </div>
        <div class="mayan-badge" id="mayan-info">Loading Mayan day...</div>
        <div class="spoon-indicator" id="spoon-display"></div>
    </div>
    <div id="goals-panel">
        <h2>🎯 Goals</h2>
        <div id="goals-list"><div style="color:#aaa;">Loading...</div></div>
    </div>
    <div id="stats-panel">
        <div class="stat"><div class="stat-value" id="total-goals">0</div><div class="stat-label">Goals</div></div>
        <div class="stat"><div class="stat-value" id="completed-goals">0</div><div class="stat-label">Done</div></div>
        <div class="stat"><div class="stat-value" id="avg-progress">0%</div><div class="stat-label">Progress</div></div>
        <div class="stat"><div class="stat-value" id="spoons-left">--</div><div class="stat-label">🥄 Spoons</div></div>
    </div>
    <div id="controls">
        <div class="control-row"><label>Animation</label><input type="range" id="speed" min="0" max="200" value="100"></div>
        <div class="control-row"><label>Goal Scale</label><input type="range" id="scale" min="50" max="200" value="100"></div>
        <div style="margin-top:10px;text-align:center;">
            <button onclick="resetCamera()">Reset View</button>
        </div>
    </div>
    <script>
        const PHI = (1 + Math.sqrt(5)) / 2;
        const PHI_INV = PHI - 1;
        const GOLDEN_ANGLE = 137.5077640500378 * Math.PI / 180;
        const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        
        let scene, camera, renderer;
        let fractalMesh, goalOrbs = [], connectionLines = [];
        let animationSpeed = 1.0;
        let frameCount = 0;
        let isDragging = false, prevMouse = {x:0, y:0};
        let cameraAngle = {theta: 0, phi: Math.PI/4};
        let cameraDistance = 20;
        
        async function api(endpoint) {
            try {
                const res = await fetch(endpoint, {credentials: 'include'});
                return await res.json();
            } catch(e) { return {}; }
        }
        
        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050510);
            scene.fog = new THREE.FogExp2(0x050510, 0.012);
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            updateCameraPosition();
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            document.getElementById('canvas-container').appendChild(renderer.domElement);
            
            // Lights
            scene.add(new THREE.AmbientLight(0x404040, 0.5));
            const light1 = new THREE.PointLight(0x667eea, 2, 100);
            light1.position.set(10, 10, 10);
            scene.add(light1);
            const light2 = new THREE.PointLight(0x764ba2, 2, 100);
            light2.position.set(-10, -10, 10);
            scene.add(light2);
            const light3 = new THREE.PointLight(0xf0c420, 1.5, 100);
            light3.position.set(0, 15, -10);
            scene.add(light3);
            
            createFractal();
            createSacredGeometry();
            createStarfield();
            
            // Controls
            renderer.domElement.addEventListener('mousedown', e => { isDragging = true; prevMouse = {x: e.clientX, y: e.clientY}; });
            renderer.domElement.addEventListener('mouseup', () => isDragging = false);
            renderer.domElement.addEventListener('mousemove', e => {
                if (!isDragging) return;
                const dx = e.clientX - prevMouse.x;
                const dy = e.clientY - prevMouse.y;
                cameraAngle.theta -= dx * 0.005;
                cameraAngle.phi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraAngle.phi + dy * 0.005));
                updateCameraPosition();
                prevMouse = {x: e.clientX, y: e.clientY};
            });
            renderer.domElement.addEventListener('wheel', e => {
                cameraDistance = Math.max(5, Math.min(50, cameraDistance + e.deltaY * 0.02));
                updateCameraPosition();
            });
            
            document.getElementById('speed').addEventListener('input', e => animationSpeed = e.target.value / 100);
            
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            loadData();
            animate();
            setTimeout(() => document.getElementById('loading').style.display = 'none', 1500);
        }
        
        function updateCameraPosition() {
            camera.position.x = cameraDistance * Math.sin(cameraAngle.phi) * Math.cos(cameraAngle.theta);
            camera.position.y = cameraDistance * Math.cos(cameraAngle.phi);
            camera.position.z = cameraDistance * Math.sin(cameraAngle.phi) * Math.sin(cameraAngle.theta);
            camera.lookAt(0, 0, 0);
        }
        
        function resetCamera() {
            cameraAngle = {theta: 0, phi: Math.PI/4};
            cameraDistance = 20;
            updateCameraPosition();
        }
        
        function createFractal() {
            const geometry = new THREE.BufferGeometry();
            const vertices = [], colors = [];
            const resolution = 60, scale = 3;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const theta = (i / resolution) * Math.PI;
                    const phi = (j / resolution) * Math.PI * 2;
                    
                    let x = Math.sin(theta) * Math.cos(phi);
                    let y = Math.sin(theta) * Math.sin(phi);
                    let z = Math.cos(theta);
                    
                    // Mandelbulb iterations
                    for (let iter = 0; iter < 5; iter++) {
                        const r = Math.sqrt(x*x + y*y + z*z);
                        if (r > 2) break;
                        const theta_p = Math.acos(z / r) * 8;
                        const phi_p = Math.atan2(y, x) * 8;
                        const zr = Math.pow(r, 8);
                        x = zr * Math.sin(theta_p) * Math.cos(phi_p);
                        y = zr * Math.sin(theta_p) * Math.sin(phi_p);
                        z = zr * Math.cos(theta_p);
                    }
                    
                    const finalR = scale * (1 + 0.2 * Math.sin(theta * 5));
                    vertices.push(
                        finalR * Math.sin(theta) * Math.cos(phi),
                        finalR * Math.cos(theta),
                        finalR * Math.sin(theta) * Math.sin(phi)
                    );
                    
                    const hue = (i / resolution + j / resolution * PHI_INV) % 1;
                    const color = new THREE.Color().setHSL(hue * 0.3 + 0.6, 0.8, 0.5);
                    colors.push(color.r, color.g, color.b);
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            fractalMesh = new THREE.Points(geometry, new THREE.PointsMaterial({
                size: 0.08, vertexColors: true, transparent: true, opacity: 0.8,
                blending: THREE.AdditiveBlending
            }));
            scene.add(fractalMesh);
        }
        
        function createSacredGeometry() {
            // Golden Spiral
            const spiralVerts = [];
            for (let i = 0; i < 500; i++) {
                const angle = i * 0.1;
                const r = 0.1 * Math.pow(PHI, 2 * angle / Math.PI);
                if (r > 8) break;
                spiralVerts.push(r * Math.cos(angle), r * Math.sin(angle) * 0.3, r * Math.sin(angle));
            }
            const spiralGeom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(spiralVerts, 3));
            scene.add(new THREE.Line(spiralGeom, new THREE.LineBasicMaterial({ color: 0xf0c420, transparent: true, opacity: 0.4 })));
            
            // Flower of Life
            const flowerGroup = new THREE.Group();
            const circleGeom = new THREE.CircleGeometry(2, 64);
            const edges = new THREE.EdgesGeometry(circleGeom);
            const circleMat = new THREE.LineBasicMaterial({ color: 0x667eea, transparent: true, opacity: 0.15 });
            [[0,0], [2,0], [-2,0], [1,1.73], [-1,1.73], [1,-1.73], [-1,-1.73]].forEach(([x, y]) => {
                const circle = new THREE.LineSegments(edges.clone(), circleMat);
                circle.position.set(x, 0, y);
                circle.rotation.x = Math.PI / 2;
                flowerGroup.add(circle);
            });
            scene.add(flowerGroup);
        }
        
        function createStarfield() {
            const verts = [], cols = [];
            for (let i = 0; i < 2000; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 40 + Math.random() * 40;
                verts.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
                const b = 0.3 + Math.random() * 0.7;
                cols.push(b, b, b * 1.2);
            }
            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
            geom.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
            scene.add(new THREE.Points(geom, new THREE.PointsMaterial({ size: 0.3, vertexColors: true, transparent: true, opacity: 0.8 })));
        }
        
        function createGoalOrbs(goals) {
            goalOrbs.forEach(o => scene.remove(o));
            goalOrbs = [];
            connectionLines.forEach(l => scene.remove(l));
            connectionLines = [];
            if (!goals || !goals.length) return;
            
            goals.forEach((goal, i) => {
                const angle = i * GOLDEN_ANGLE;
                const radius = 4 + (goal.spoon_cost || 2) * 0.3;
                const height = (goal.progress / 100) * 4 - 2;
                const x = radius * Math.cos(angle);
                const z = radius * Math.sin(angle);
                const y = height + Math.sin(i * 0.5) * 1.5;
                
                const size = 0.3 + (goal.progress / 100) * 0.4;
                const color = goal.progress >= 100 ? 0x48c774 : (goal.progress >= 70 ? 0x3298dc : (goal.progress >= 40 ? 0xf0c420 : 0xff6b6b));
                
                const orb = new THREE.Mesh(
                    new THREE.SphereGeometry(size, 24, 24),
                    new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.3, transparent: true, opacity: 0.9 })
                );
                orb.position.set(x, y, z);
                orb.userData = { goal, index: i, baseY: y };
                
                // Glow
                orb.add(new THREE.Mesh(
                    new THREE.SphereGeometry(size * 1.5, 16, 16),
                    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.15 })
                ));
                
                scene.add(orb);
                goalOrbs.push(orb);
                
                // Connection to center
                const lineGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), orb.position]);
                scene.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2 })));
            });
        }
        
        async function loadData() {
            const [goals, mayan, spoons, stats] = await Promise.all([
                api('/api/goals'), api('/api/mayan'), api('/api/spoons'), api('/api/stats')
            ]);
            
            if (goals.goals) {
                createGoalOrbs(goals.goals);
                updateGoalsPanel(goals.goals);
                document.getElementById('total-goals').textContent = goals.goals.length;
                document.getElementById('completed-goals').textContent = goals.goals.filter(g => g.progress >= 100).length;
                const avg = goals.goals.length ? goals.goals.reduce((a,g) => a + (g.progress||0), 0) / goals.goals.length : 0;
                document.getElementById('avg-progress').textContent = Math.round(avg) + '%';
            }
            
            if (mayan.today) {
                document.getElementById('mayan-info').innerHTML = mayan.today.day_sign_emoji + ' ' + mayan.today.tzolkin + '<br><small>' + mayan.today.day_sign + '</small>';
            }
            
            if (spoons.available_spoons !== undefined) {
                document.getElementById('spoons-left').textContent = spoons.available_spoons;
                let spoonHtml = '';
                for (let i = 0; i < (spoons.max_spoons || 12); i++) {
                    spoonHtml += '<div class="spoon ' + (i < spoons.available_spoons ? 'available' : 'used') + '"></div>';
                }
                document.getElementById('spoon-display').innerHTML = spoonHtml;
            }
        }
        
        function updateGoalsPanel(goals) {
            const container = document.getElementById('goals-list');
            if (!goals.length) { container.innerHTML = '<div style="color:#888;">No goals</div>'; return; }
            container.innerHTML = goals.map((g, i) => {
                const color = g.progress >= 70 ? '#48c774' : (g.progress >= 40 ? '#f0c420' : '#ff6b6b');
                return '<div class="goal-item" onclick="focusGoal(' + i + ')"><div class="goal-title">' + (g.progress >= 100 ? '✅' : '🎯') + ' ' + g.title + '</div><div class="goal-progress"><div class="goal-progress-fill" style="width:' + g.progress + '%;background:' + color + '"></div></div><small>' + Math.round(g.progress) + '% | 🥄' + (g.spoon_cost || 2) + '</small></div>';
            }).join('');
        }
        
        function focusGoal(index) {
            if (!goalOrbs[index]) return;
            const pos = goalOrbs[index].position;
            cameraAngle.theta = Math.atan2(pos.z, pos.x);
            cameraAngle.phi = Math.PI / 3;
            cameraDistance = 10;
            updateCameraPosition();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            document.getElementById('frame-count').textContent = frameCount;
            
            if (fractalMesh) {
                fractalMesh.rotation.y += 0.001 * animationSpeed;
                fractalMesh.rotation.x += 0.0003 * animationSpeed;
            }
            
            goalOrbs.forEach((orb, i) => {
                orb.position.y = orb.userData.baseY + Math.sin(frameCount * 0.02 + i) * 0.3 * animationSpeed;
                orb.rotation.y += 0.01 * animationSpeed;
            });
            
            renderer.render(scene, camera);
        }
        
        init();
    </script>
</body>
</html>'''

@app.route('/3d')
def view_3d():
    return render_template_string(THREE_D_HTML)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    mayan = prediction_engine.get_mayan_day_sign()
    
    logger.info("═" * 70)
    logger.info("🌀 LIFE FRACTAL INTELLIGENCE v10.0 - NEURODIVERGENT WELLNESS SYSTEM")
    logger.info("═" * 70)
    logger.info("✅ Spoon Theory Engine: ACTIVE")
    logger.info("✅ Unified Predictive Math: ACTIVE")
    logger.info("✅ Executive Dysfunction Modeling: ACTIVE")
    logger.info(f"✅ Mayan Calendar: {mayan['tzolkin']} ({mayan['day_sign_emoji']} {mayan['day_sign']})")
    logger.info("✅ NLP Sentiment Analysis: ACTIVE")
    logger.info("✅ Fractal Visualization: ACTIVE")
    logger.info("✅ Self-Healing System: ACTIVE")
    logger.info(f"✅ Port: {port}")
    logger.info("═" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') != 'production')
