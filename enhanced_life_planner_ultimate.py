"""
Enhanced Life Planning System - Ultimate Edition
=================================================

This module extends the original life planning system with:
1. Advanced virtual pet with evolution, personalities, and mini-games
2. Additional mathematical methods (Markov chains, Bayesian inference, time series)
3. Gamification system with achievements, streaks, and rewards
4. Advanced analytics and pattern detection
5. Habit formation tracking with science-based curves
6. Life balance tools and SMART goal validation
7. Pomodoro integration and productivity analytics

Author: Enhanced by Claude
Date: 2025
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(
    filename="enhanced_life_planner.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# ENHANCED MATHEMATICAL UTILITIES
# ==============================================================================

class AdvancedMathUtil:
    """Advanced mathematical utilities for predictions and analysis"""
    
    @staticmethod
    def golden_ratio() -> float:
        """Return the golden ratio (φ ≈ 1.618)"""
        return (1.0 + math.sqrt(5.0)) / 2.0
    
    @staticmethod
    def fibonacci_sequence(n: int) -> List[int]:
        """Generate first n Fibonacci numbers"""
        if n <= 0:
            return []
        if n == 1:
            return [0]
        seq = [0, 1]
        for _ in range(n - 2):
            seq.append(seq[-1] + seq[-2])
        return seq
    
    @staticmethod
    def lucas_sequence(n: int) -> List[int]:
        """Generate first n Lucas numbers (variant of Fibonacci)"""
        if n <= 0:
            return []
        if n == 1:
            return [2]
        seq = [2, 1]
        for _ in range(n - 2):
            seq.append(seq[-1] + seq[-2])
        return seq
    
    @staticmethod
    def logistic_map(r: float, x0: float, n: int) -> List[float]:
        """Generate logistic map series for chaos modeling"""
        series = []
        x = x0
        for _ in range(n):
            series.append(x)
            x = r * x * (1.0 - x)
        return series
    
    @staticmethod
    def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
        """
        Apply exponential smoothing for time series forecasting
        
        Args:
            data: Historical data points
            alpha: Smoothing factor (0 < alpha < 1)
        """
        if not data:
            return []
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return smoothed
    
    @staticmethod
    def moving_average(data: List[float], window: int = 7) -> List[float]:
        """Calculate moving average with specified window"""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(sum(data[:i+1]) / (i+1))
            else:
                result.append(sum(data[i-window+1:i+1]) / window)
        return result
    
    @staticmethod
    def polynomial_trend(data: List[float], degree: int = 2) -> Tuple[List[float], List[float]]:
        """
        Fit polynomial trend to data
        
        Returns:
            (fitted_values, future_predictions)
        """
        if len(data) < degree + 1:
            return data, []
        
        x = np.array(range(len(data)))
        y = np.array(data)
        coeffs = np.polyfit(x, y, degree)
        fitted = np.polyval(coeffs, x)
        
        # Predict next 7 days
        future_x = np.array(range(len(data), len(data) + 7))
        future = np.polyval(coeffs, future_x)
        
        return fitted.tolist(), future.tolist()
    
    @staticmethod
    def calculate_entropy(data: List[float]) -> float:
        """Calculate Shannon entropy of data (measure of randomness)"""
        if not data:
            return 0.0
        # Discretize data into bins
        hist, _ = np.histogram(data, bins=10)
        hist = hist[hist > 0]  # Remove zero bins
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)
    
    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    
    @staticmethod
    def detect_anomalies(data: List[float], threshold: float = 2.0) -> List[int]:
        """
        Detect anomalies using z-score method
        
        Returns list of indices where anomalies detected
        """
        if len(data) < 3:
            return []
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return []
        
        z_scores = [(x - mean) / std for x in data]
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        return anomalies


# ==============================================================================
# MARKOV CHAIN FOR BEHAVIOR PREDICTION
# ==============================================================================

class MarkovChainPredictor:
    """Predict future states using Markov chains"""
    
    def __init__(self, states: List[str]):
        """
        Args:
            states: List of possible states (e.g., ['low', 'medium', 'high'])
        """
        self.states = states
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
    
    def train(self, sequence: List[str]) -> None:
        """Train on a sequence of observed states"""
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_state = sequence[i + 1]
            self.transition_matrix[current][next_state] += 1
            self.state_counts[current] += 1
    
    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        """Get probability of transitioning from one state to another"""
        if from_state not in self.state_counts or self.state_counts[from_state] == 0:
            return 1.0 / len(self.states)  # Uniform prior
        return self.transition_matrix[from_state][to_state] / self.state_counts[from_state]
    
    def predict_next(self, current_state: str) -> Tuple[str, float]:
        """
        Predict most likely next state
        
        Returns:
            (predicted_state, probability)
        """
        if current_state not in self.state_counts:
            return (self.states[0], 1.0 / len(self.states))
        
        best_state = None
        best_prob = 0.0
        
        for state in self.states:
            prob = self.get_transition_probability(current_state, state)
            if prob > best_prob:
                best_prob = prob
                best_state = state
        
        return (best_state or self.states[0], best_prob)
    
    def predict_sequence(self, start_state: str, length: int) -> List[str]:
        """Predict a sequence of future states"""
        sequence = [start_state]
        current = start_state
        
        for _ in range(length - 1):
            next_state, _ = self.predict_next(current)
            sequence.append(next_state)
            current = next_state
        
        return sequence


# ==============================================================================
# BAYESIAN INFERENCE ENGINE
# ==============================================================================

class BayesianInferenceEngine:
    """Use Bayesian reasoning for adaptive suggestions"""
    
    def __init__(self):
        self.priors = {}
        self.likelihoods = defaultdict(lambda: defaultdict(float))
        self.observations = defaultdict(int)
    
    def set_prior(self, hypothesis: str, probability: float) -> None:
        """Set prior probability for a hypothesis"""
        self.priors[hypothesis] = probability
    
    def add_observation(self, hypothesis: str, evidence: str, likelihood: float) -> None:
        """Add likelihood P(evidence|hypothesis)"""
        self.likelihoods[hypothesis][evidence] = likelihood
        self.observations[evidence] += 1
    
    def calculate_posterior(self, hypothesis: str, evidence: str) -> float:
        """
        Calculate posterior probability P(hypothesis|evidence) using Bayes' theorem
        
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        prior = self.priors.get(hypothesis, 0.5)
        likelihood = self.likelihoods[hypothesis].get(evidence, 0.1)
        
        # Calculate P(E) using law of total probability
        p_evidence = sum(
            self.likelihoods[h].get(evidence, 0.1) * self.priors.get(h, 0.5)
            for h in self.priors.keys()
        )
        
        if p_evidence == 0:
            return prior
        
        posterior = (likelihood * prior) / p_evidence
        return posterior
    
    def get_best_hypothesis(self, evidence: str) -> Tuple[str, float]:
        """Get hypothesis with highest posterior probability given evidence"""
        best_hypothesis = None
        best_posterior = 0.0
        
        for hypothesis in self.priors.keys():
            posterior = self.calculate_posterior(hypothesis, evidence)
            if posterior > best_posterior:
                best_posterior = posterior
                best_hypothesis = hypothesis
        
        return (best_hypothesis or "unknown", best_posterior)


# ==============================================================================
# ENHANCED VIRTUAL PET SYSTEM
# ==============================================================================

class PetPersonality(Enum):
    """Pet personality types affecting behavior and growth"""
    ENERGETIC = "energetic"  # Gains energy faster, needs more activity
    CALM = "calm"  # More stable mood, slower growth
    CURIOUS = "curious"  # Learns faster, gains XP quicker
    LOYAL = "loyal"  # Bond increases faster
    PLAYFUL = "playful"  # Prefers play activities

@dataclass
class PetStats:
    """Comprehensive pet statistics"""
    # Core stats
    level: int = 1
    experience: int = 0
    health: float = 100.0
    happiness: float = 100.0
    hunger: float = 0.0
    energy: float = 100.0
    bond: float = 0.0
    
    # Advanced stats
    intelligence: float = 10.0
    strength: float = 10.0
    charisma: float = 10.0
    wisdom: float = 10.0
    
    # State tracking
    mood_state: str = "content"
    last_fed: datetime = field(default_factory=datetime.now)
    last_played: datetime = field(default_factory=datetime.now)
    last_trained: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level,
            'experience': self.experience,
            'health': self.health,
            'happiness': self.happiness,
            'hunger': self.hunger,
            'energy': self.energy,
            'bond': self.bond,
            'intelligence': self.intelligence,
            'strength': self.strength,
            'charisma': self.charisma,
            'wisdom': self.wisdom,
            'mood_state': self.mood_state,
            'last_fed': self.last_fed.isoformat(),
            'last_played': self.last_played.isoformat(),
            'last_trained': self.last_trained.isoformat()
        }


class PetSpecies:
    """Define different pet species with unique characteristics"""
    
    SPECIES_DATA = {
        'dragon': {
            'emoji': '🐉',
            'base_stats': {'strength': 15, 'intelligence': 12},
            'growth_rate': 1.2,
            'special_ability': 'fire_breath',
            'evolution_stages': ['Egg', 'Wyrmling', 'Drake', 'Dragon', 'Ancient Dragon']
        },
        'phoenix': {
            'emoji': '🔥',
            'base_stats': {'wisdom': 15, 'charisma': 12},
            'growth_rate': 1.0,
            'special_ability': 'rebirth',
            'evolution_stages': ['Ash', 'Spark', 'Flame', 'Phoenix', 'Eternal Phoenix']
        },
        'unicorn': {
            'emoji': '🦄',
            'base_stats': {'charisma': 15, 'wisdom': 13},
            'growth_rate': 0.9,
            'special_ability': 'healing',
            'evolution_stages': ['Foal', 'Pony', 'Mare', 'Unicorn', 'Alicorn']
        },
        'owl': {
            'emoji': '🦉',
            'base_stats': {'intelligence': 18, 'wisdom': 15},
            'growth_rate': 1.1,
            'special_ability': 'wisdom_boost',
            'evolution_stages': ['Chick', 'Owlet', 'Fledgling', 'Owl', 'Great Owl']
        },
        'fox': {
            'emoji': '🦊',
            'base_stats': {'intelligence': 13, 'charisma': 13},
            'growth_rate': 1.3,
            'special_ability': 'quick_learn',
            'evolution_stages': ['Kit', 'Cub', 'Vixen', 'Fox', 'Kitsune']
        },
        'cat': {
            'emoji': '🐱',
            'base_stats': {'charisma': 12, 'intelligence': 11},
            'growth_rate': 1.0,
            'special_ability': 'independence',
            'evolution_stages': ['Kitten', 'Cat', 'Feline', 'Mystic Cat', 'Celestial Cat']
        }
    }
    
    @classmethod
    def get_species_info(cls, species: str) -> Dict:
        """Get information about a species"""
        return cls.SPECIES_DATA.get(species, cls.SPECIES_DATA['cat'])


class EnhancedVirtualPet:
    """Advanced virtual pet with personality, evolution, and mini-games"""
    
    def __init__(self, name: str, species: str = 'cat', personality: PetPersonality = PetPersonality.LOYAL):
        self.name = name
        self.species = species
        self.personality = personality
        self.stats = PetStats()
        self.birthday = datetime.now()
        self.achievements = []
        self.inventory = []
        self.skills = defaultdict(int)
        
        # Initialize species-specific stats
        species_info = PetSpecies.get_species_info(species)
        for stat, value in species_info['base_stats'].items():
            setattr(self.stats, stat, value)
        
        # Activity history
        self.activity_log = deque(maxlen=100)
        
        # Mini-game scores
        self.mini_game_scores = defaultdict(list)
    
    def get_evolution_stage(self) -> str:
        """Determine current evolution stage based on level"""
        species_info = PetSpecies.get_species_info(self.species)
        stages = species_info['evolution_stages']
        
        # Level thresholds: 1, 5, 15, 30, 50
        thresholds = [1, 5, 15, 30, 50]
        for i in range(len(thresholds) - 1, -1, -1):
            if self.stats.level >= thresholds[i]:
                return stages[min(i, len(stages) - 1)]
        return stages[0]
    
    def experience_to_level(self) -> int:
        """Calculate XP needed for next level"""
        species_info = PetSpecies.get_species_info(self.species)
        base = 100
        growth_rate = species_info['growth_rate']
        return int(base * (growth_rate ** (self.stats.level - 1)) * self.stats.level)
    
    def gain_experience(self, amount: int) -> List[str]:
        """Add experience and handle leveling"""
        messages = []
        self.stats.experience += amount
        
        while self.stats.experience >= self.experience_to_level():
            self.stats.experience -= self.experience_to_level()
            old_stage = self.get_evolution_stage()
            self.stats.level += 1
            new_stage = self.get_evolution_stage()
            
            messages.append(f"🎉 {self.name} leveled up to level {self.stats.level}!")
            
            # Stat increases
            self.stats.intelligence += random.randint(1, 3)
            self.stats.strength += random.randint(1, 3)
            self.stats.charisma += random.randint(1, 3)
            self.stats.wisdom += random.randint(1, 3)
            
            # Check for evolution
            if old_stage != new_stage:
                species_info = PetSpecies.get_species_info(self.species)
                messages.append(
                    f"✨ {self.name} evolved from {old_stage} to {new_stage}! {species_info['emoji']}"
                )
                self.unlock_achievement(f"evolution_{new_stage}")
        
        return messages
    
    def feed(self, food_quality: int = 50) -> Tuple[str, int]:
        """
        Feed the pet
        
        Returns:
            (message, xp_gained)
        """
        self.stats.hunger = max(0, self.stats.hunger - food_quality)
        self.stats.health = min(100, self.stats.health + food_quality * 0.2)
        happiness_gain = int(food_quality * 0.3)
        self.stats.happiness = min(100, self.stats.happiness + happiness_gain)
        self.stats.last_fed = datetime.now()
        
        xp = int(food_quality * 0.5)
        self.activity_log.append(('feed', datetime.now(), xp))
        
        messages = self.gain_experience(xp)
        msg = f"🍖 {self.name} enjoyed the meal! Happiness +{happiness_gain}"
        if messages:
            msg += " " + " ".join(messages)
        
        return (msg, xp)
    
    def play(self, activity: str = "fetch") -> Tuple[str, int]:
        """
        Play with the pet
        
        Returns:
            (message, xp_gained)
        """
        if self.stats.energy < 15:
            return (f"😴 {self.name} is too tired to play!", 0)
        
        self.stats.energy = max(0, self.stats.energy - 15)
        happiness_gain = random.randint(15, 30)
        
        # Personality affects happiness gain
        if self.personality == PetPersonality.PLAYFUL:
            happiness_gain = int(happiness_gain * 1.5)
        
        self.stats.happiness = min(100, self.stats.happiness + happiness_gain)
        bond_gain = random.randint(2, 8)
        
        if self.personality == PetPersonality.LOYAL:
            bond_gain = int(bond_gain * 1.5)
        
        self.stats.bond = min(100, self.stats.bond + bond_gain)
        self.stats.last_played = datetime.now()
        
        xp = random.randint(10, 25)
        self.skills[activity] += 1
        self.activity_log.append(('play', datetime.now(), xp))
        
        messages = self.gain_experience(xp)
        msg = f"🎮 You played {activity} with {self.name}! Happiness +{happiness_gain}, Bond +{bond_gain}"
        if messages:
            msg += " " + " ".join(messages)
        
        return (msg, xp)
    
    def train(self, skill: str = "intelligence") -> Tuple[str, int]:
        """
        Train a specific stat
        
        Returns:
            (message, xp_gained)
        """
        if self.stats.energy < 20:
            return (f"😓 {self.name} is too tired to train!", 0)
        
        self.stats.energy = max(0, self.stats.energy - 20)
        stat_gain = random.randint(1, 5)
        
        # Personality affects training
        if self.personality == PetPersonality.CURIOUS and skill == "intelligence":
            stat_gain = int(stat_gain * 1.5)
        
        if hasattr(self.stats, skill):
            current = getattr(self.stats, skill)
            setattr(self.stats, skill, current + stat_gain)
        
        self.stats.last_trained = datetime.now()
        self.skills[f"training_{skill}"] += 1
        
        xp = 15 + stat_gain * 2
        self.activity_log.append(('train', datetime.now(), xp))
        
        messages = self.gain_experience(xp)
        msg = f"💪 {self.name} trained {skill}! +{stat_gain} {skill}"
        if messages:
            msg += " " + " ".join(messages)
        
        return (msg, xp)
    
    def rest(self) -> str:
        """Let pet rest and recover"""
        energy_gain = 40
        if self.personality == PetPersonality.CALM:
            energy_gain = 50
        
        self.stats.energy = min(100, self.stats.energy + energy_gain)
        self.stats.health = min(100, self.stats.health + 10)
        
        return f"😴 {self.name} took a nap! Energy +{energy_gain}, Health +10"
    
    def use_special_ability(self) -> Tuple[str, Dict[str, float]]:
        """Activate species special ability"""
        species_info = PetSpecies.get_species_info(self.species)
        ability = species_info['special_ability']
        effects = {}
        
        if ability == 'fire_breath':
            # Dragon: Burn away stress
            effects = {'stress_reduction': 30, 'energy_cost': 25}
            msg = f"🔥 {self.name} breathes fire, burning away stress!"
        elif ability == 'rebirth':
            # Phoenix: Full health and happiness restore
            self.stats.health = 100
            self.stats.happiness = 100
            effects = {'health_restore': 100, 'happiness_restore': 100}
            msg = f"✨ {self.name} rises from the ashes, fully restored!"
        elif ability == 'healing':
            # Unicorn: Heal and boost mood
            effects = {'health_boost': 40, 'mood_boost': 25}
            msg = f"🌟 {self.name}'s magic heals and uplifts you!"
        elif ability == 'wisdom_boost':
            # Owl: Boost productivity and focus
            effects = {'productivity_boost': 35, 'focus_boost': 30}
            msg = f"🦉 {self.name}'s wisdom enhances your focus!"
        elif ability == 'quick_learn':
            # Fox: Double XP for next activity
            effects = {'xp_multiplier': 2.0, 'duration': 1}
            msg = f"🦊 {self.name}'s cleverness doubles your learning!"
        elif ability == 'independence':
            # Cat: Reduce negative effects
            effects = {'stress_resistance': 20, 'mood_stability': 15}
            msg = f"🐱 {self.name}'s independence strengthens your resolve!"
        else:
            msg = f"✨ {self.name} uses a mysterious ability!"
            effects = {'energy_boost': 20}
        
        self.stats.energy = max(0, self.stats.energy - effects.get('energy_cost', 20))
        self.unlock_achievement(f"ability_{ability}")
        
        return (msg, effects)
    
    def play_mini_game(self, game_type: str, score: int) -> Tuple[str, int]:
        """
        Record mini-game result and award XP
        
        Args:
            game_type: Type of mini-game
            score: Player's score
        
        Returns:
            (message, xp_gained)
        """
        self.mini_game_scores[game_type].append(score)
        
        # XP based on score and personal best
        personal_best = max(self.mini_game_scores[game_type][:-1]) if len(self.mini_game_scores[game_type]) > 1 else 0
        xp = int(score / 10)
        
        msg = f"🎯 Mini-game '{game_type}' score: {score}"
        
        if score > personal_best:
            bonus_xp = 50
            xp += bonus_xp
            msg += f" 🏆 NEW PERSONAL BEST! +{bonus_xp} bonus XP"
            self.unlock_achievement(f"minigame_{game_type}_best")
        
        messages = self.gain_experience(xp)
        if messages:
            msg += " " + " ".join(messages)
        
        return (msg, xp)
    
    def unlock_achievement(self, achievement_id: str) -> bool:
        """Unlock an achievement"""
        if achievement_id not in self.achievements:
            self.achievements.append(achievement_id)
            logger.info(f"Achievement unlocked: {achievement_id}")
            return True
        return False
    
    def get_mood_descriptor(self) -> str:
        """Get descriptive mood based on stats"""
        if self.stats.happiness >= 80:
            return "ecstatic"
        elif self.stats.happiness >= 60:
            return "happy"
        elif self.stats.happiness >= 40:
            return "content"
        elif self.stats.happiness >= 20:
            return "okay"
        else:
            return "sad"
    
    def update_daily(self) -> List[str]:
        """Daily maintenance - called once per day"""
        messages = []
        
        # Hunger increases
        self.stats.hunger = min(100, self.stats.hunger + 25)
        if self.stats.hunger > 80:
            messages.append(f"🍖 {self.name} is very hungry!")
        
        # Happiness decreases slightly
        happiness_decay = 10
        if self.personality == PetPersonality.CALM:
            happiness_decay = 5
        self.stats.happiness = max(0, self.stats.happiness - happiness_decay)
        
        # Energy regenerates
        energy_gain = 35
        if self.personality == PetPersonality.ENERGETIC:
            energy_gain = 50
        self.stats.energy = min(100, self.stats.energy + energy_gain)
        
        # Bond decreases if not interacted
        hours_since_interaction = (datetime.now() - max(
            self.stats.last_fed, self.stats.last_played, self.stats.last_trained
        )).total_seconds() / 3600
        
        if hours_since_interaction > 48:
            bond_loss = 5
            self.stats.bond = max(0, self.stats.bond - bond_loss)
            messages.append(f"💔 {self.name} misses you! Bond -{bond_loss}")
        
        return messages
    
    def get_status_display(self) -> str:
        """Get formatted status display"""
        species_info = PetSpecies.get_species_info(self.species)
        age_days = (datetime.now() - self.birthday).days
        stage = self.get_evolution_stage()
        mood = self.get_mood_descriptor()
        
        def bar(value: float, length: int = 10) -> str:
            filled = int(value / 10)
            return '█' * filled + '░' * (length - filled)
        
        status = f"""
╔══════════════════════════════════════════════════════════════════╗
║  {species_info['emoji']}  {self.name.upper()} - {self.species.title()} ({self.personality.value.title()})
╠══════════════════════════════════════════════════════════════════╣
║  Stage: {stage} | Level: {self.stats.level} | Age: {age_days} days
║  XP: {self.stats.experience}/{self.experience_to_level()} | Mood: {mood}
║  
║  ❤️  Health:    {bar(self.stats.health)} {self.stats.health:.0f}%
║  😊 Happiness: {bar(self.stats.happiness)} {self.stats.happiness:.0f}%
║  🍖 Hunger:    {bar(100-self.stats.hunger)} {(100-self.stats.hunger):.0f}%
║  ⚡ Energy:    {bar(self.stats.energy)} {self.stats.energy:.0f}%
║  💝 Bond:      {bar(self.stats.bond)} {self.stats.bond:.0f}%
║
║  📊 Attributes:
║    🧠 Intelligence: {self.stats.intelligence:.0f}  💪 Strength: {self.stats.strength:.0f}
║    ✨ Charisma: {self.stats.charisma:.0f}         🦉 Wisdom: {self.stats.wisdom:.0f}
║
║  🏆 Achievements: {len(self.achievements)}
║  🎮 Skills Learned: {len(self.skills)}
╚══════════════════════════════════════════════════════════════════╝
"""
        return status
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            'name': self.name,
            'species': self.species,
            'personality': self.personality.value,
            'stats': self.stats.to_dict(),
            'birthday': self.birthday.isoformat(),
            'achievements': self.achievements,
            'inventory': self.inventory,
            'skills': dict(self.skills),
            'mini_game_scores': {k: list(v) for k, v in self.mini_game_scores.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedVirtualPet':
        """Load from dictionary"""
        pet = cls(
            data['name'],
            data['species'],
            PetPersonality(data.get('personality', 'loyal'))
        )
        
        stats_data = data['stats']
        pet.stats.level = stats_data['level']
        pet.stats.experience = stats_data['experience']
        pet.stats.health = stats_data['health']
        pet.stats.happiness = stats_data['happiness']
        pet.stats.hunger = stats_data['hunger']
        pet.stats.energy = stats_data['energy']
        pet.stats.bond = stats_data['bond']
        pet.stats.intelligence = stats_data.get('intelligence', 10)
        pet.stats.strength = stats_data.get('strength', 10)
        pet.stats.charisma = stats_data.get('charisma', 10)
        pet.stats.wisdom = stats_data.get('wisdom', 10)
        pet.stats.mood_state = stats_data.get('mood_state', 'content')
        
        if 'last_fed' in stats_data:
            pet.stats.last_fed = datetime.fromisoformat(stats_data['last_fed'])
        if 'last_played' in stats_data:
            pet.stats.last_played = datetime.fromisoformat(stats_data['last_played'])
        if 'last_trained' in stats_data:
            pet.stats.last_trained = datetime.fromisoformat(stats_data['last_trained'])
        
        pet.birthday = datetime.fromisoformat(data['birthday'])
        pet.achievements = data.get('achievements', [])
        pet.inventory = data.get('inventory', [])
        pet.skills = defaultdict(int, data.get('skills', {}))
        
        mini_game_data = data.get('mini_game_scores', {})
        for game, scores in mini_game_data.items():
            pet.mini_game_scores[game] = scores
        
        return pet


# ==============================================================================
# GAMIFICATION SYSTEM
# ==============================================================================

class Achievement:
    """Represents an unlockable achievement"""
    
    def __init__(self, id: str, name: str, description: str, 
                 category: str, points: int, icon: str = "🏆"):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.points = points
        self.icon = icon
        self.unlocked_date: Optional[datetime] = None
    
    def unlock(self) -> int:
        """Unlock achievement and return points"""
        if self.unlocked_date is None:
            self.unlocked_date = datetime.now()
            return self.points
        return 0


class GamificationSystem:
    """Manage achievements, streaks, and rewards"""
    
    def __init__(self):
        self.achievements: Dict[str, Achievement] = {}
        self.streaks: Dict[str, int] = {}
        self.combo_multiplier: float = 1.0
        self.total_points: int = 0
        self.level: int = 1
        self.daily_streak: int = 0
        self.last_activity_date: Optional[datetime] = None
        
        self._initialize_achievements()
    
    def _initialize_achievements(self):
        """Create predefined achievements"""
        achievements_data = [
            # Streak achievements
            ("streak_7", "Week Warrior", "Maintain a 7-day streak", "streaks", 100, "🔥"),
            ("streak_30", "Month Master", "Maintain a 30-day streak", "streaks", 500, "⭐"),
            ("streak_100", "Centurion", "Maintain a 100-day streak", "streaks", 2000, "👑"),
            
            # Task achievements
            ("tasks_10", "Getting Started", "Complete 10 tasks", "tasks", 50, "✅"),
            ("tasks_100", "Task Master", "Complete 100 tasks", "tasks", 500, "💯"),
            ("tasks_1000", "Productivity Legend", "Complete 1000 tasks", "tasks", 5000, "🌟"),
            
            # Goal achievements
            ("goals_5", "Goal Getter", "Achieve 5 goals", "goals", 100, "🎯"),
            ("goals_25", "Dream Chaser", "Achieve 25 goals", "goals", 750, "🏆"),
            ("goals_100", "Visionary", "Achieve 100 goals", "goals", 5000, "👑"),
            
            # Pet achievements
            ("pet_bond_100", "Best Friends", "Reach 100% bond with pet", "pet", 200, "💕"),
            ("pet_level_25", "Pet Master", "Reach level 25 with pet", "pet", 500, "⭐"),
            ("pet_all_evolutions", "Evolution Expert", "See all evolution stages", "pet", 1000, "✨"),
            
            # Focus achievements
            ("focus_10h", "Focused Mind", "Complete 10 hours of focus time", "focus", 100, "🧘"),
            ("focus_100h", "Zen Master", "Complete 100 hours of focus time", "focus", 1000, "🧠"),
            ("focus_1000h", "Meditation Sage", "Complete 1000 hours of focus time", "focus", 10000, "☯️"),
            
            # Balance achievements
            ("balance_perfect_week", "Balanced Week", "Achieve perfect life balance for 7 days", "balance", 300, "⚖️"),
            ("all_categories_active", "Renaissance Person", "Be active in all life categories", "balance", 500, "🎨"),
        ]
        
        for id, name, desc, category, points, icon in achievements_data:
            self.achievements[id] = Achievement(id, name, desc, category, points, icon)
    
    def unlock_achievement(self, achievement_id: str) -> Tuple[bool, int]:
        """
        Unlock an achievement
        
        Returns:
            (was_newly_unlocked, points_earned)
        """
        if achievement_id in self.achievements:
            points = self.achievements[achievement_id].unlock()
            if points > 0:
                self.total_points += points
                self._check_level_up()
                logger.info(f"Achievement unlocked: {achievement_id} (+{points} points)")
                return (True, points)
        return (False, 0)
    
    def update_streak(self, activity: str) -> Tuple[int, bool]:
        """
        Update streak for an activity
        
        Returns:
            (current_streak, is_new_record)
        """
        today = datetime.now().date()
        
        # Check daily streak
        if self.last_activity_date:
            days_diff = (today - self.last_activity_date.date()).days
            if days_diff == 1:
                self.daily_streak += 1
            elif days_diff > 1:
                self.daily_streak = 1
        else:
            self.daily_streak = 1
        
        self.last_activity_date = datetime.now()
        
        # Update specific activity streak
        current_streak = self.streaks.get(activity, 0) + 1
        self.streaks[activity] = current_streak
        
        # Check for streak achievements
        if self.daily_streak == 7:
            self.unlock_achievement("streak_7")
        elif self.daily_streak == 30:
            self.unlock_achievement("streak_30")
        elif self.daily_streak == 100:
            self.unlock_achievement("streak_100")
        
        return (current_streak, True)  # Simplified - always new record for this update
    
    def calculate_combo_multiplier(self, actions_today: int) -> float:
        """Calculate XP multiplier based on consecutive actions"""
        # Multiplier increases with actions, caps at 2x
        self.combo_multiplier = min(2.0, 1.0 + (actions_today * 0.1))
        return self.combo_multiplier
    
    def _check_level_up(self):
        """Check if total points reached next level threshold"""
        # Level up every 1000 points
        new_level = (self.total_points // 1000) + 1
        if new_level > self.level:
            self.level = new_level
            logger.info(f"Gamification level up! Now level {self.level}")
    
    def get_progress_summary(self) -> str:
        """Get formatted progress summary"""
        unlocked = sum(1 for a in self.achievements.values() if a.unlocked_date)
        total = len(self.achievements)
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║  🎮 GAMIFICATION PROGRESS
╠══════════════════════════════════════════════════════════════════╣
║  Level: {self.level} | Total Points: {self.total_points}
║  Daily Streak: {self.daily_streak} days 🔥
║  Combo Multiplier: {self.combo_multiplier}x
║  
║  Achievements: {unlocked}/{total} unlocked
║  
║  Recent Unlocks:
"""
        
        # Show 5 most recent achievements
        recent = sorted(
            [a for a in self.achievements.values() if a.unlocked_date],
            key=lambda x: x.unlocked_date,
            reverse=True
        )[:5]
        
        for ach in recent:
            summary += f"║    {ach.icon} {ach.name} (+{ach.points} pts)\n"
        
        summary += "╚══════════════════════════════════════════════════════════════════╝"
        
        return summary


# ==============================================================================
# HABIT FORMATION TRACKER
# ==============================================================================

class HabitTracker:
    """Track habits using science-based formation curves"""
    
    def __init__(self, habit_name: str, target_days: int = 66):
        """
        Args:
            habit_name: Name of the habit
            target_days: Days to form habit (research suggests 66 days average)
        """
        self.habit_name = habit_name
        self.target_days = target_days
        self.completion_log: List[datetime] = []
        self.current_streak = 0
        self.longest_streak = 0
        self.formation_percentage = 0.0
    
    def mark_complete(self, date: Optional[datetime] = None) -> Tuple[str, float]:
        """
        Mark habit as complete for a date
        
        Returns:
            (message, formation_percentage)
        """
        if date is None:
            date = datetime.now()
        
        # Check if already logged today
        if any(d.date() == date.date() for d in self.completion_log):
            return ("Already completed today!", self.formation_percentage)
        
        self.completion_log.append(date)
        self.completion_log.sort()
        
        # Calculate streak
        self._update_streak()
        
        # Calculate formation percentage using logarithmic curve
        # Formation follows: P(t) = 100 * (1 - e^(-k*t)) where k = ln(2)/target_days
        days_practiced = len(self.completion_log)
        k = math.log(2) / self.target_days
        self.formation_percentage = 100 * (1 - math.exp(-k * days_practiced))
        
        msg = f"✅ {self.habit_name} completed! Streak: {self.current_streak} days"
        if self.current_streak > self.longest_streak:
            self.longest_streak = self.current_streak
            msg += " 🏆 NEW RECORD!"
        
        msg += f" | Formation: {self.formation_percentage:.1f}%"
        
        return (msg, self.formation_percentage)
    
    def _update_streak(self):
        """Update current streak based on completion log"""
        if not self.completion_log:
            self.current_streak = 0
            return
        
        # Count consecutive days from most recent
        today = datetime.now().date()
        streak = 0
        
        for i in range(len(self.completion_log) - 1, -1, -1):
            expected_date = today - timedelta(days=streak)
            if self.completion_log[i].date() == expected_date:
                streak += 1
            else:
                break
        
        self.current_streak = streak
    
    def get_formation_curve(self, days: int = 100) -> List[float]:
        """Get theoretical formation curve values"""
        k = math.log(2) / self.target_days
        return [100 * (1 - math.exp(-k * d)) for d in range(days)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get habit statistics"""
        total_completions = len(self.completion_log)
        if total_completions == 0:
            completion_rate = 0
        else:
            days_since_start = (datetime.now() - self.completion_log[0]).days + 1
            completion_rate = (total_completions / days_since_start) * 100
        
        return {
            'habit_name': self.habit_name,
            'total_completions': total_completions,
            'current_streak': self.current_streak,
            'longest_streak': self.longest_streak,
            'formation_percentage': self.formation_percentage,
            'completion_rate': completion_rate,
            'target_days': self.target_days
        }


# ==============================================================================
# LIFE BALANCE WHEEL
# ==============================================================================

class LifeBalanceWheel:
    """Track and visualize life balance across multiple dimensions"""
    
    DIMENSIONS = [
        'health', 'career', 'finances', 'relationships',
        'personal_growth', 'recreation', 'environment', 'spirituality'
    ]
    
    def __init__(self):
        self.scores: Dict[str, float] = {dim: 5.0 for dim in self.DIMENSIONS}
        self.history: List[Dict[str, float]] = []
        self.goals: Dict[str, float] = {dim: 8.0 for dim in self.DIMENSIONS}
    
    def update_dimension(self, dimension: str, score: float) -> str:
        """
        Update a dimension score (0-10)
        
        Returns:
            Feedback message
        """
        if dimension not in self.DIMENSIONS:
            return f"Unknown dimension: {dimension}"
        
        old_score = self.scores[dimension]
        self.scores[dimension] = max(0, min(10, score))
        
        # Record history
        self.history.append(self.scores.copy())
        
        change = self.scores[dimension] - old_score
        if change > 0:
            return f"📈 {dimension.title()} improved by {change:.1f} points!"
        elif change < 0:
            return f"📉 {dimension.title()} decreased by {abs(change):.1f} points"
        else:
            return f"➡️ {dimension.title()} unchanged"
    
    def calculate_balance_score(self) -> float:
        """
        Calculate overall balance (0-100)
        Higher is better balanced
        """
        # Balance is inverse of standard deviation
        # Perfectly balanced = all dimensions equal
        scores = list(self.scores.values())
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        # Convert to 0-100 scale where low std dev = high balance
        # Max possible std dev with 0-10 scale is 5, so:
        balance = 100 * (1 - min(std_dev / 5, 1))
        
        return balance
    
    def get_weakest_dimensions(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get n weakest dimensions"""
        sorted_dims = sorted(self.scores.items(), key=lambda x: x[1])
        return sorted_dims[:n]
    
    def get_recommendations(self) -> List[str]:
        """Get personalized recommendations"""
        recommendations = []
        weakest = self.get_weakest_dimensions(3)
        
        for dim, score in weakest:
            if score < 4:
                recommendations.append(
                    f"⚠️ {dim.title()} needs attention (score: {score:.1f}/10). "
                    f"Consider dedicating time to improve this area."
                )
            elif score < 6:
                recommendations.append(
                    f"💡 {dim.title()} could use improvement (score: {score:.1f}/10). "
                    f"Small consistent actions can make a big difference."
                )
        
        # Check balance
        balance = self.calculate_balance_score()
        if balance < 60:
            recommendations.append(
                "⚖️ Your life balance could be improved. Focus on neglected areas "
                "to create a more harmonious life."
            )
        
        return recommendations
    
    def visualize_text(self) -> str:
        """Create text-based visualization"""
        balance = self.calculate_balance_score()
        
        viz = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚖️ LIFE BALANCE WHEEL
╠══════════════════════════════════════════════════════════════════╣
║  Overall Balance: {balance:.1f}/100
║  
"""
        
        for dim in self.DIMENSIONS:
            score = self.scores[dim]
            goal = self.goals[dim]
            bar = '█' * int(score) + '░' * (10 - int(score))
            status = "✓" if score >= goal else "○"
            viz += f"║  {status} {dim.title():20} {bar} {score:.1f}/10\n"
        
        viz += "║\n"
        
        # Add recommendations
        recs = self.get_recommendations()
        if recs:
            viz += "║  💡 Recommendations:\n"
            for rec in recs[:3]:  # Show top 3
                # Wrap text
                words = rec.split()
                line = "║  "
                for word in words:
                    if len(line) + len(word) + 1 > 68:
                        viz += line + "\n"
                        line = "║  " + word + " "
                    else:
                        line += word + " "
                viz += line.rstrip() + "\n"
        
        viz += "╚══════════════════════════════════════════════════════════════════╝"
        
        return viz


# ==============================================================================
# SMART GOAL VALIDATOR
# ==============================================================================

class SMARTGoalValidator:
    """Validate and score goals against SMART criteria"""
    
    @staticmethod
    def validate_goal(goal_text: str, deadline: Optional[datetime] = None,
                     metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate a goal against SMART criteria
        
        Returns dict with scores for each criterion and overall score
        """
        scores = {
            'specific': 0,
            'measurable': 0,
            'achievable': 0,
            'relevant': 0,
            'time_bound': 0
        }
        
        feedback = []
        
        # Specific - check for concrete details
        specific_keywords = ['will', 'by', 'through', 'using', 'with']
        if any(keyword in goal_text.lower() for keyword in specific_keywords):
            scores['specific'] += 30
        if len(goal_text.split()) >= 10:
            scores['specific'] += 20
        if any(char.isdigit() for char in goal_text):
            scores['specific'] += 50
        else:
            feedback.append("💡 Add specific details or numbers to make goal more concrete")
        
        scores['specific'] = min(100, scores['specific'])
        
        # Measurable - check for metrics
        measurable_keywords = ['increase', 'decrease', 'achieve', 'reach', 'complete',
                              '%', 'percent', 'number', 'amount', 'times']
        if any(keyword in goal_text.lower() for keyword in measurable_keywords):
            scores['measurable'] += 50
        if metrics and len(metrics) > 0:
            scores['measurable'] += 50
        else:
            feedback.append("💡 Define how you'll measure progress")
        
        scores['measurable'] = min(100, scores['measurable'])
        
        # Achievable - basic checks
        if len(goal_text.split()) < 30:  # Not overly complex
            scores['achievable'] += 50
        if 'impossible' not in goal_text.lower() and 'never' not in goal_text.lower():
            scores['achievable'] += 50
        else:
            feedback.append("⚠️ Goal might be too ambitious - consider breaking it down")
        
        # Relevant - check for purpose words
        relevant_keywords = ['because', 'to', 'for', 'so that', 'in order to', 'improve']
        if any(keyword in goal_text.lower() for keyword in relevant_keywords):
            scores['relevant'] += 100
        else:
            feedback.append("💡 Add why this goal matters to you")
            scores['relevant'] = 50  # Assume somewhat relevant
        
        # Time-bound - check for deadline
        if deadline:
            scores['time_bound'] = 100
        else:
            time_keywords = ['by', 'until', 'before', 'within', 'january', 'february',
                           'march', 'april', 'may', 'june', 'july', 'august',
                           'september', 'october', 'november', 'december', 'week', 'month']
            if any(keyword in goal_text.lower() for keyword in time_keywords):
                scores['time_bound'] = 70
            else:
                scores['time_bound'] = 0
                feedback.append("⚠️ Set a clear deadline for this goal")
        
        # Calculate overall score
        overall = sum(scores.values()) / len(scores)
        
        return {
            'scores': scores,
            'overall': overall,
            'feedback': feedback,
            'grade': SMARTGoalValidator._get_grade(overall)
        }
    
    @staticmethod
    def _get_grade(score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A - Excellent SMART goal! 🌟"
        elif score >= 80:
            return "B - Good goal with minor improvements needed 👍"
        elif score >= 70:
            return "C - Fair goal, consider refining further 📝"
        elif score >= 60:
            return "D - Needs significant improvement ⚠️"
        else:
            return "F - Not yet a SMART goal, needs major revision ❌"


# ==============================================================================
# POMODORO TIMER INTEGRATION
# ==============================================================================

class PomodoroSession:
    """Track a Pomodoro focus session"""
    
    def __init__(self, duration_minutes: int = 25, break_minutes: int = 5):
        self.duration_minutes = duration_minutes
        self.break_minutes = break_minutes
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.completed = False
        self.interruptions = 0
        self.task_name = ""
    
    def start(self, task_name: str = "Focus Work") -> str:
        """Start a Pomodoro session"""
        self.start_time = datetime.now()
        self.task_name = task_name
        self.completed = False
        return f"🍅 Pomodoro started for '{task_name}' - {self.duration_minutes} minutes"
    
    def record_interruption(self) -> int:
        """Record an interruption"""
        self.interruptions += 1
        return self.interruptions
    
    def complete(self) -> Tuple[str, int]:
        """
        Complete the session
        
        Returns:
            (message, xp_earned)
        """
        self.end_time = datetime.now()
        self.completed = True
        
        # XP based on duration and interruptions
        base_xp = self.duration_minutes * 2
        interruption_penalty = self.interruptions * 5
        xp = max(10, base_xp - interruption_penalty)
        
        msg = f"✅ Pomodoro completed! {self.duration_minutes} min focused on '{self.task_name}'"
        if self.interruptions > 0:
            msg += f" ({self.interruptions} interruptions)"
        msg += f" | +{xp} XP"
        
        return (msg, xp)
    
    def get_elapsed_minutes(self) -> int:
        """Get elapsed time in minutes"""
        if not self.start_time:
            return 0
        elapsed = datetime.now() - self.start_time
        return int(elapsed.total_seconds() / 60)


class PomodoroTracker:
    """Track multiple Pomodoro sessions and statistics"""
    
    def __init__(self):
        self.sessions: List[PomodoroSession] = []
        self.current_session: Optional[PomodoroSession] = None
    
    def start_session(self, task_name: str = "Focus Work",
                     duration_minutes: int = 25) -> str:
        """Start a new Pomodoro session"""
        if self.current_session and not self.current_session.completed:
            return "⚠️ A session is already in progress!"
        
        self.current_session = PomodoroSession(duration_minutes)
        return self.current_session.start(task_name)
    
    def complete_current_session(self) -> Tuple[str, int]:
        """Complete the current session"""
        if not self.current_session:
            return ("No active session", 0)
        
        msg, xp = self.current_session.complete()
        self.sessions.append(self.current_session)
        self.current_session = None
        return (msg, xp)
    
    def record_interruption(self) -> str:
        """Record an interruption in current session"""
        if not self.current_session:
            return "No active session"
        
        count = self.current_session.record_interruption()
        return f"Interruption #{count} recorded"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Pomodoro statistics"""
        completed_sessions = [s for s in self.sessions if s.completed]
        
        if not completed_sessions:
            return {
                'total_sessions': 0,
                'total_focus_time': 0,
                'average_interruptions': 0,
                'completion_rate': 0
            }
        
        total_time = sum(s.duration_minutes for s in completed_sessions)
        total_interruptions = sum(s.interruptions for s in completed_sessions)
        
        return {
            'total_sessions': len(completed_sessions),
            'total_focus_time': total_time,
            'average_interruptions': total_interruptions / len(completed_sessions),
            'completion_rate': (len(completed_sessions) / len(self.sessions)) * 100 if self.sessions else 0,
            'sessions_today': len([s for s in completed_sessions 
                                  if s.end_time and s.end_time.date() == datetime.now().date()])
        }


# ==============================================================================
# PATTERN DETECTION AND ANALYTICS
# ==============================================================================

class PatternDetector:
    """Detect patterns in user behavior and provide insights"""
    
    def __init__(self):
        self.mood_history: List[Tuple[datetime, float]] = []
        self.productivity_history: List[Tuple[datetime, float]] = []
        self.activity_history: List[Tuple[datetime, str, float]] = []
    
    def add_mood_data(self, mood: float, timestamp: Optional[datetime] = None):
        """Add mood data point"""
        if timestamp is None:
            timestamp = datetime.now()
        self.mood_history.append((timestamp, mood))
    
    def add_productivity_data(self, productivity: float, timestamp: Optional[datetime] = None):
        """Add productivity data point"""
        if timestamp is None:
            timestamp = datetime.now()
        self.productivity_history.append((timestamp, productivity))
    
    def add_activity(self, activity_type: str, value: float, timestamp: Optional[datetime] = None):
        """Add activity data"""
        if timestamp is None:
            timestamp = datetime.now()
        self.activity_history.append((timestamp, activity_type, value))
    
    def detect_circadian_pattern(self) -> Dict[str, Any]:
        """Detect time-of-day patterns in mood and productivity"""
        if len(self.mood_history) < 7:
            return {'pattern_found': False, 'message': 'Insufficient data'}
        
        # Group by hour of day
        hour_moods = defaultdict(list)
        for timestamp, mood in self.mood_history:
            hour = timestamp.hour
            hour_moods[hour].append(mood)
        
        # Find best and worst hours
        hour_averages = {h: np.mean(moods) for h, moods in hour_moods.items() if moods}
        
        if not hour_averages:
            return {'pattern_found': False}
        
        best_hour = max(hour_averages.items(), key=lambda x: x[1])
        worst_hour = min(hour_averages.items(), key=lambda x: x[1])
        
        return {
            'pattern_found': True,
            'best_hour': best_hour[0],
            'best_hour_mood': best_hour[1],
            'worst_hour': worst_hour[0],
            'worst_hour_mood': worst_hour[1],
            'recommendation': f"You tend to feel best around {best_hour[0]}:00. "
                            f"Schedule important tasks during this time!"
        }
    
    def detect_correlations(self) -> List[Dict[str, Any]]:
        """Detect correlations between activities and mood"""
        correlations = []
        
        if len(self.mood_history) < 10 or len(self.activity_history) < 10:
            return [{'message': 'Insufficient data for correlation analysis'}]
        
        # Group activities by type
        activity_types = set(act[1] for act in self.activity_history)
        
        for activity_type in activity_types:
            # Get activity values and corresponding moods
            activity_data = []
            mood_data = []
            
            for act_time, act_type, act_value in self.activity_history:
                if act_type == activity_type:
                    # Find closest mood reading (within 4 hours)
                    closest_mood = None
                    min_diff = timedelta(hours=4)
                    
                    for mood_time, mood_value in self.mood_history:
                        diff = abs(mood_time - act_time)
                        if diff < min_diff:
                            min_diff = diff
                            closest_mood = mood_value
                    
                    if closest_mood is not None:
                        activity_data.append(act_value)
                        mood_data.append(closest_mood)
            
            if len(activity_data) >= 3:
                corr = AdvancedMathUtil.correlation(activity_data, mood_data)
                
                if abs(corr) > 0.3:  # Significant correlation
                    correlations.append({
                        'activity': activity_type,
                        'correlation': corr,
                        'impact': 'positive' if corr > 0 else 'negative',
                        'strength': 'strong' if abs(corr) > 0.6 else 'moderate',
                        'recommendation': f"{'Increase' if corr > 0 else 'Decrease'} {activity_type} "
                                        f"for better mood (correlation: {corr:.2f})"
                    })
        
        return correlations if correlations else [{'message': 'No significant correlations found yet'}]
    
    def forecast_mood(self, days_ahead: int = 7) -> List[float]:
        """Forecast future mood using trend analysis"""
        if len(self.mood_history) < 14:
            return []
        
        # Extract mood values
        moods = [m[1] for m in self.mood_history[-30:]]  # Last 30 data points
        
        # Use polynomial trend fitting
        _, forecast = AdvancedMathUtil.polynomial_trend(moods, degree=2)
        
        return forecast[:days_ahead]
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual patterns that might need attention"""
        anomalies = []
        
        if len(self.mood_history) < 14:
            return anomalies
        
        # Recent moods
        recent_moods = [m[1] for m in self.mood_history[-14:]]
        anomaly_indices = AdvancedMathUtil.detect_anomalies(recent_moods)
        
        for idx in anomaly_indices:
            timestamp, mood = self.mood_history[-(14-idx)]
            anomalies.append({
                'type': 'mood_anomaly',
                'timestamp': timestamp,
                'value': mood,
                'message': f"Unusual mood detected on {timestamp.strftime('%Y-%m-%d')}: {mood:.1f}"
            })
        
        return anomalies


# ==============================================================================
# INTEGRATING EVERYTHING: ULTIMATE LIFE PLANNING SYSTEM
# ==============================================================================

class UltimateLifePlanningSystem:
    """
    The ultimate life planning system integrating all components:
    - Enhanced virtual pet with evolution and personalities
    - Advanced mathematical predictive models
    - Gamification with achievements and streaks
    - Habit formation tracking
    - Life balance monitoring
    - SMART goal validation
    - Pomodoro integration
    - Pattern detection and analytics
    """
    
    def __init__(self, pet_name: str = "Buddy", pet_species: str = "cat",
                 pet_personality: PetPersonality = PetPersonality.LOYAL):
        # Core components
        self.pet = EnhancedVirtualPet(pet_name, pet_species, pet_personality)
        self.gamification = GamificationSystem()
        self.life_balance = LifeBalanceWheel()
        self.pattern_detector = PatternDetector()
        self.pomodoro = PomodoroTracker()
        
        # Mathematical models
        self.markov_mood = MarkovChainPredictor(['low', 'medium', 'high'])
        self.bayesian_engine = BayesianInferenceEngine()
        self._initialize_bayesian()
        
        # Tracking
        self.habits: Dict[str, HabitTracker] = {}
        self.goals: List[Dict[str, Any]] = []
        self.daily_log: List[Dict[str, Any]] = []
        
        # History for advanced analytics
        self.mood_history: List[float] = []
        self.stress_history: List[float] = []
        self.productivity_history: List[float] = []
        
        logger.info(f"Ultimate Life Planning System initialized with pet '{pet_name}' ({pet_species})")
    
    def _initialize_bayesian(self):
        """Initialize Bayesian inference engine with default hypotheses"""
        # Set up hypotheses for activity recommendations
        activities = ['exercise', 'meditation', 'social', 'work', 'rest']
        for activity in activities:
            self.bayesian_engine.set_prior(activity, 0.2)
        
        # Set up some likelihood examples
        self.bayesian_engine.add_observation('exercise', 'low_energy', 0.3)
        self.bayesian_engine.add_observation('exercise', 'high_energy', 0.8)
        self.bayesian_engine.add_observation('meditation', 'high_stress', 0.9)
        self.bayesian_engine.add_observation('rest', 'low_energy', 0.9)
        self.bayesian_engine.add_observation('social', 'low_mood', 0.7)
    
    def log_daily_data(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Log daily data and update all systems
        
        Expected keys in data:
        - mood (0-100)
        - stress (0-100)
        - energy (0-100)
        - productivity (0-100)
        - sleep_hours
        - exercise_minutes
        - social_time_minutes
        """
        timestamp = datetime.now()
        data['timestamp'] = timestamp.isoformat()
        self.daily_log.append(data)
        
        # Update histories
        if 'mood' in data:
            self.mood_history.append(data['mood'])
            self.pattern_detector.add_mood_data(data['mood'], timestamp)
            
            # Update Markov chain
            mood_state = self._discretize_mood(data['mood'])
            if len(self.mood_history) > 1:
                prev_state = self._discretize_mood(self.mood_history[-2])
                self.markov_mood.train([prev_state, mood_state])
        
        if 'stress' in data:
            self.stress_history.append(data['stress'])
        
        if 'productivity' in data:
            self.productivity_history.append(data['productivity'])
            self.pattern_detector.add_productivity_data(data['productivity'], timestamp)
        
        # Update pattern detector with activities
        if 'exercise_minutes' in data:
            self.pattern_detector.add_activity('exercise', data['exercise_minutes'], timestamp)
        if 'social_time_minutes' in data:
            self.pattern_detector.add_activity('social', data['social_time_minutes'], timestamp)
        
        # Update gamification
        actions_today = len([d for d in self.daily_log 
                           if datetime.fromisoformat(d['timestamp']).date() == timestamp.date()])
        multiplier = self.gamification.calculate_combo_multiplier(actions_today)
        
        # Update pet based on user data
        pet_messages = []
        
        # Feed pet if user had good nutrition
        if data.get('productivity', 0) > 70:
            msg, xp = self.pet.feed(80)
            pet_messages.append(msg)
        
        # Play with pet if user exercised
        if data.get('exercise_minutes', 0) > 30:
            msg, xp = self.pet.play('exercise')
            pet_messages.append(msg)
        
        return {
            'pet_messages': pet_messages,
            'xp_multiplier': multiplier,
            'data_logged': True
        }
    
    def _discretize_mood(self, mood: float) -> str:
        """Convert mood to discrete state"""
        if mood < 40:
            return 'low'
        elif mood < 70:
            return 'medium'
        else:
            return 'high'
    
    def add_habit(self, habit_name: str, target_days: int = 66) -> str:
        """Add a new habit to track"""
        if habit_name in self.habits:
            return f"Habit '{habit_name}' already exists"
        
        self.habits[habit_name] = HabitTracker(habit_name, target_days)
        return f"✅ Habit '{habit_name}' added! Target: {target_days} days"
    
    def complete_habit(self, habit_name: str) -> str:
        """Mark a habit as complete for today"""
        if habit_name not in self.habits:
            return f"Habit '{habit_name}' not found"
        
        msg, formation = self.habits[habit_name].mark_complete()
        
        # Reward XP
        xp = int(formation / 2)  # Max 50 XP for fully formed habit
        pet_msgs = self.pet.gain_experience(xp)
        
        if pet_msgs:
            msg += " " + " ".join(pet_msgs)
        
        # Check achievements
        if formation >= 50:
            unlocked, points = self.gamification.unlock_achievement("habit_halfway")
        if formation >= 90:
            unlocked, points = self.gamification.unlock_achievement("habit_formed")
        
        return msg
    
    def add_goal(self, title: str, description: str, deadline: datetime,
                 category: str = "personal", metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add a SMART goal"""
        # Validate goal
        validation = SMARTGoalValidator.validate_goal(description, deadline, metrics)
        
        goal = {
            'id': len(self.goals),
            'title': title,
            'description': description,
            'category': category,
            'deadline': deadline.isoformat(),
            'metrics': metrics or [],
            'progress': 0,
            'completed': False,
            'created': datetime.now().isoformat(),
            'validation': validation
        }
        
        self.goals.append(goal)
        
        return {
            'goal_id': goal['id'],
            'validation': validation,
            'message': f"Goal '{title}' added! SMART score: {validation['overall']:.1f}/100"
        }
    
    def update_goal_progress(self, goal_id: int, progress: int) -> str:
        """Update progress on a goal (0-100%)"""
        if goal_id >= len(self.goals):
            return "Goal not found"
        
        goal = self.goals[goal_id]
        old_progress = goal['progress']
        goal['progress'] = max(0, min(100, progress))
        
        msg = f"📊 Goal '{goal['title']}' progress: {goal['progress']}%"
        
        if goal['progress'] == 100 and not goal['completed']:
            goal['completed'] = True
            goal['completed_date'] = datetime.now().isoformat()
            
            # Big XP reward
            xp = 200
            pet_msgs = self.pet.gain_experience(xp)
            msg += f" 🎉 GOAL COMPLETED! +{xp} XP"
            if pet_msgs:
                msg += " " + " ".join(pet_msgs)
            
            # Achievement
            completed_count = sum(1 for g in self.goals if g['completed'])
            if completed_count == 5:
                self.gamification.unlock_achievement("goals_5")
            elif completed_count == 25:
                self.gamification.unlock_achievement("goals_25")
        
        return msg
    
    def start_pomodoro(self, task_name: str = "Focus Work", duration: int = 25) -> str:
        """Start a Pomodoro session"""
        return self.pomodoro.start_session(task_name, duration)
    
    def complete_pomodoro(self) -> Tuple[str, int]:
        """Complete current Pomodoro session"""
        msg, xp = self.pomodoro.complete_current_session()
        
        # Give XP to pet
        pet_msgs = self.pet.gain_experience(xp)
        if pet_msgs:
            msg += " " + " ".join(pet_msgs)
        
        # Update gamification
        stats = self.pomodoro.get_statistics()
        total_hours = stats['total_focus_time'] / 60
        
        if total_hours >= 10:
            self.gamification.unlock_achievement("focus_10h")
        elif total_hours >= 100:
            self.gamification.unlock_achievement("focus_100h")
        
        return (msg, xp)
    
    def get_recommendations(self) -> List[str]:
        """Get personalized recommendations based on all data"""
        recommendations = []
        
        # Pet recommendations
        if self.pet.stats.hunger > 70:
            recommendations.append(f"🍖 {self.pet.name} is very hungry! Feed your pet.")
        if self.pet.stats.happiness < 40:
            recommendations.append(f"😢 {self.pet.name} seems sad. Play with your pet to boost happiness!")
        if self.pet.stats.energy < 30:
            recommendations.append(f"😴 {self.pet.name} needs rest.")
        
        # Life balance recommendations
        balance_recs = self.life_balance.get_recommendations()
        recommendations.extend(balance_recs[:2])  # Add top 2
        
        # Pattern-based recommendations
        circadian = self.pattern_detector.detect_circadian_pattern()
        if circadian.get('pattern_found'):
            recommendations.append(circadian['recommendation'])
        
        correlations = self.pattern_detector.detect_correlations()
        for corr in correlations[:2]:  # Top 2 correlations
            if 'recommendation' in corr:
                recommendations.append(corr['recommendation'])
        
        # Bayesian recommendation
        if self.stress_history and self.mood_history:
            recent_stress = self.stress_history[-1]
            recent_mood = self.mood_history[-1]
            
            evidence = 'high_stress' if recent_stress > 70 else 'low_energy' if recent_mood < 40 else 'high_energy'
            best_activity, prob = self.bayesian_engine.get_best_hypothesis(evidence)
            
            recommendations.append(
                f"🎯 Based on your state, consider: {best_activity} "
                f"(confidence: {prob*100:.0f}%)"
            )
        
        # Markov prediction
        if len(self.mood_history) > 1:
            current_state = self._discretize_mood(self.mood_history[-1])
            predicted_state, prob = self.markov_mood.predict_next(current_state)
            
            if predicted_state == 'low':
                recommendations.append(
                    f"⚠️ Mood may dip soon. Consider preventive self-care activities."
                )
        
        # Habit reminders
        for habit_name, habit in self.habits.items():
            if habit.current_streak > 0 and not any(
                d.date() == datetime.now().date() for d in habit.completion_log
            ):
                recommendations.append(
                    f"🔥 Don't break your {habit.current_streak}-day streak for '{habit_name}'!"
                )
        
        return recommendations[:8]  # Return top 8
    
    def get_analytics_dashboard(self) -> str:
        """Generate comprehensive analytics dashboard"""
        dashboard = """
╔══════════════════════════════════════════════════════════════════╗
║  📊 ULTIMATE LIFE PLANNING DASHBOARD
╠══════════════════════════════════════════════════════════════════╣
"""
        
        # Pet status (condensed)
        dashboard += f"║  {self.pet.get_evolution_stage()} {self.pet.name} - Level {self.pet.stats.level}\n"
        dashboard += f"║  Happiness: {self.pet.stats.happiness:.0f}% | Bond: {self.pet.stats.bond:.0f}%\n"
        dashboard += "║\n"
        
        # Gamification
        dashboard += f"║  🎮 Gamification Level: {self.gamification.level}\n"
        dashboard += f"║  Total Points: {self.gamification.total_points} | Streak: {self.gamification.daily_streak} days\n"
        dashboard += "║\n"
        
        # Goals
        active_goals = [g for g in self.goals if not g['completed']]
        completed_goals = [g for g in self.goals if g['completed']]
        dashboard += f"║  🎯 Goals: {len(completed_goals)} completed, {len(active_goals)} active\n"
        
        if active_goals:
            next_goal = max(active_goals, key=lambda g: g['progress'])
            dashboard += f"║  Next: {next_goal['title']} ({next_goal['progress']}%)\n"
        dashboard += "║\n"
        
        # Habits
        dashboard += f"║  ✅ Habits: {len(self.habits)} tracked\n"
        for habit_name, habit in list(self.habits.items())[:3]:
            dashboard += f"║    {habit_name}: {habit.formation_percentage:.0f}% formed (streak: {habit.current_streak})\n"
        dashboard += "║\n"
        
        # Life balance
        balance_score = self.life_balance.calculate_balance_score()
        dashboard += f"║  ⚖️ Life Balance: {balance_score:.0f}/100\n"
        weakest = self.life_balance.get_weakest_dimensions(2)
        for dim, score in weakest:
            dashboard += f"║    ⚠️ {dim.title()}: {score:.1f}/10\n"
        dashboard += "║\n"
        
        # Pomodoro
        pomo_stats = self.pomodoro.get_statistics()
        dashboard += f"║  🍅 Pomodoro: {pomo_stats['total_sessions']} sessions, {pomo_stats['total_focus_time']} min total\n"
        dashboard += f"║  Today: {pomo_stats['sessions_today']} sessions\n"
        dashboard += "║\n"
        
        # Predictions
        if len(self.mood_history) >= 7:
            forecast = self.pattern_detector.forecast_mood(3)
            if forecast:
                dashboard += "║  🔮 Mood Forecast (next 3 days):\n"
                for i, mood in enumerate(forecast, 1):
                    emoji = "😊" if mood > 70 else "😐" if mood > 40 else "😔"
                    dashboard += f"║    Day {i}: {mood:.0f} {emoji}\n"
        
        dashboard += "╚══════════════════════════════════════════════════════════════════╝"
        
        return dashboard
    
    def save_to_file(self, filename: str = "life_planner_data.json") -> str:
        """Save all data to JSON file"""
        data = {
            'pet': self.pet.to_dict(),
            'gamification': {
                'level': self.gamification.level,
                'total_points': self.gamification.total_points,
                'daily_streak': self.gamification.daily_streak,
                'achievements': [
                    {'id': a.id, 'unlocked': a.unlocked_date.isoformat() if a.unlocked_date else None}
                    for a in self.gamification.achievements.values()
                ]
            },
            'life_balance': {
                'scores': self.life_balance.scores,
                'goals': self.life_balance.goals
            },
            'habits': {
                name: {
                    'completion_log': [d.isoformat() for d in habit.completion_log],
                    'current_streak': habit.current_streak,
                    'longest_streak': habit.longest_streak,
                    'target_days': habit.target_days
                }
                for name, habit in self.habits.items()
            },
            'goals': self.goals,
            'daily_log': self.daily_log,
            'mood_history': self.mood_history,
            'stress_history': self.stress_history,
            'productivity_history': self.productivity_history
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"✅ Data saved to {filename}"
    
    @classmethod
    def load_from_file(cls, filename: str = "life_planner_data.json") -> 'UltimateLifePlanningSystem':
        """Load system from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create new system
        pet_data = data['pet']
        system = cls(
            pet_data['name'],
            pet_data['species'],
            PetPersonality(pet_data.get('personality', 'loyal'))
        )
        
        # Load pet
        system.pet = EnhancedVirtualPet.from_dict(pet_data)
        
        # Load gamification
        gam_data = data.get('gamification', {})
        system.gamification.level = gam_data.get('level', 1)
        system.gamification.total_points = gam_data.get('total_points', 0)
        system.gamification.daily_streak = gam_data.get('daily_streak', 0)
        
        # Load life balance
        lb_data = data.get('life_balance', {})
        system.life_balance.scores = lb_data.get('scores', system.life_balance.scores)
        system.life_balance.goals = lb_data.get('goals', system.life_balance.goals)
        
        # Load habits
        habits_data = data.get('habits', {})
        for name, habit_info in habits_data.items():
            habit = HabitTracker(name, habit_info['target_days'])
            habit.completion_log = [datetime.fromisoformat(d) for d in habit_info['completion_log']]
            habit.current_streak = habit_info['current_streak']
            habit.longest_streak = habit_info['longest_streak']
            system.habits[name] = habit
        
        # Load goals and histories
        system.goals = data.get('goals', [])
        system.daily_log = data.get('daily_log', [])
        system.mood_history = data.get('mood_history', [])
        system.stress_history = data.get('stress_history', [])
        system.productivity_history = data.get('productivity_history', [])
        
        return system


# ==============================================================================
# DEMONSTRATION AND TESTING
# ==============================================================================

def run_demonstration():
    """Run a comprehensive demonstration of the system"""
    print("=" * 70)
    print("ULTIMATE LIFE PLANNING SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create system
    system = UltimateLifePlanningSystem(
        pet_name="Phoenix",
        pet_species="phoenix",
        pet_personality=PetPersonality.CURIOUS
    )
    
    print("✨ System initialized with Phoenix the Phoenix (Curious personality)")
    print()
    
    # Show initial pet status
    print(system.pet.get_status_display())
    print()
    
    # Add some habits
    print(system.add_habit("Morning Exercise", 66))
    print(system.add_habit("Meditation", 66))
    print(system.add_habit("Reading", 66))
    print()
    
    # Add a goal
    goal_result = system.add_goal(
        title="Learn Python",
        description="Complete a Python course with 100 coding exercises by December 2025",
        deadline=datetime(2025, 12, 31),
        category="learning",
        metrics=["exercises_completed", "projects_built"]
    )
    print(f"Goal added: {goal_result['message']}")
    print(f"SMART validation: {goal_result['validation']['grade']}")
    print()
    
    # Simulate a week of activity
    print("📅 Simulating a week of activities...")
    print()
    
    for day in range(7):
        print(f"--- Day {day + 1} ---")
        
        # Log daily data with some variation
        mood = 60 + random.randint(-10, 20)
        stress = 40 + random.randint(-15, 25)
        productivity = 65 + random.randint(-15, 20)
        
        result = system.log_daily_data({
            'mood': mood,
            'stress': stress,
            'energy': 70 + random.randint(-20, 20),
            'productivity': productivity,
            'sleep_hours': 7 + random.randint(-1, 1),
            'exercise_minutes': random.randint(20, 60),
            'social_time_minutes': random.randint(30, 120)
        })
        
        for msg in result['pet_messages']:
            print(f"  {msg}")
        
        # Complete habits
        if random.random() > 0.2:  # 80% chance
            print(f"  {system.complete_habit('Morning Exercise')}")
        if random.random() > 0.3:  # 70% chance
            print(f"  {system.complete_habit('Meditation')}")
        
        # Do a Pomodoro session
        system.start_pomodoro("Deep Work")
        msg, xp = system.complete_pomodoro()
        print(f"  {msg}")
        
        # Interact with pet
        if day % 2 == 0:
            msg, xp = system.pet.play("training")
            print(f"  {msg}")
        else:
            msg, xp = system.pet.train("intelligence")
            print(f"  {msg}")
        
        # Update life balance (random dimension)
        dimension = random.choice(LifeBalanceWheel.DIMENSIONS)
        score = random.randint(5, 9)
        msg = system.life_balance.update_dimension(dimension, score)
        print(f"  {msg}")
        
        print()
    
    # Update goal progress
    print(system.update_goal_progress(0, 45))
    print()
    
    # Use special ability
    msg, effects = system.pet.use_special_ability()
    print(msg)
    print(f"Effects: {effects}")
    print()
    
    # Show final dashboard
    print(system.get_analytics_dashboard())
    print()
    
    # Get recommendations
    print("💡 PERSONALIZED RECOMMENDATIONS:")
    for rec in system.get_recommendations():
        print(f"  • {rec}")
    print()
    
    # Show life balance
    print(system.life_balance.visualize_text())
    print()
    
    # Show gamification progress
    print(system.gamification.get_progress_summary())
    print()
    
    # Pattern analysis
    print("🔍 PATTERN ANALYSIS:")
    correlations = system.pattern_detector.detect_correlations()
    for corr in correlations[:3]:
        print(f"  • {corr.get('message', corr.get('recommendation', str(corr)))}")
    print()
    
    # Save data
    print(system.save_to_file("demo_life_planner.json"))
    print()
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
