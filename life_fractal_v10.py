#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  LIFE FRACTAL INTELLIGENCE - EVOLVED MATHEMATICAL ORGANISM v11.0                          ║
║  ═══════════════════════════════════════════════════════════════════════════════════════  ║
║                                                                                           ║
║  🧬 EVOLVED MATHEMATICAL ARCHITECTURE:                                                    ║
║  ├── Karma-Dharma Scoring Engine (spiritual mathematics)                                  ║
║  ├── Swarm Intelligence (boid flocking, stigmergy, PSO)                                   ║
║  ├── Organic Cells (biological orbs, mitosis, binding)                                    ║
║  ├── Origami Logic (fold transformations, crease patterns)                                ║
║  ├── Fractal Propagation (golden ratio, Fibonacci, Mandelbrot)                            ║
║  ├── Machine Learning Evolution (pattern detection, prediction)                           ║
║  └── GPU-Accelerated Client-Side 3D Visualization                                         ║
║                                                                                           ║
║  🎯 FIXED: Server-side ray marching removed - 3D now rendered via WebGL shaders           ║
║  🎯 FIXED: Worker timeouts eliminated with async processing                               ║
║  🎯 NEW: Complete algorithm collection for evolved AI                                     ║
║                                                                                           ║
║  For neurodivergent minds: External visualization, energy tracking, compassionate UX      ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
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
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
from collections import defaultdict
import base64
import threading

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# GPU Support (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None

# ML Support (optional)
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ═══════════════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS - UNIVERSAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio φ ≈ 1.618033988749895
PHI_INVERSE = 1 / PHI  # φ⁻¹ ≈ 0.618033988749895
PHI_SQUARED = PHI * PHI  # φ² ≈ 2.618033988749895
GOLDEN_ANGLE = 360 / (PHI ** 2)  # ≈ 137.5077640500378°
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
PLANCK_KARMA = 1e-43  # Smallest unit of karmic action
DHARMA_FREQUENCY = 432  # Hz - Universal harmonic frequency
SCHUMANN_RESONANCE = 7.83  # Hz - Earth's natural frequency
SOLFEGGIO_FREQUENCIES = [174, 285, 396, 417, 528, 639, 741, 852, 963]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# KARMA-DHARMA MATHEMATICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class KarmicValence(Enum):
    """The polarity of karmic energy"""
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    TRANSFORMATIVE = 2


class DharmicAlignment(Enum):
    """Degrees of alignment with cosmic order"""
    PERFECT = 1.0
    HARMONIOUS = 0.75
    SEEKING = 0.5
    MISALIGNED = 0.25
    INVERTED = 0.0


@dataclass
class KarmicVector:
    """Multidimensional representation of karmic potential"""
    magnitude: float = 0.0
    valence: KarmicValence = KarmicValence.NEUTRAL
    velocity: float = 0.0
    spin: float = 0.0
    entangled_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source_intention: float = 1.0
    awareness_coefficient: float = 1.0
    
    @property
    def karmic_weight(self) -> float:
        """K = Intention × Action × Awareness × φ"""
        base_weight = self.magnitude * self.source_intention * self.awareness_coefficient
        harmonic = PHI if self.valence == KarmicValence.POSITIVE else PHI_INVERSE
        return base_weight * harmonic * (1 + abs(self.spin) * 0.1)
    
    def propagate(self, time_delta: float) -> 'KarmicVector':
        """Propagate karma through time using fractal decay/growth"""
        decay = PHI_INVERSE if self.valence == KarmicValence.NEGATIVE else 1.0
        growth = PHI if self.valence == KarmicValence.POSITIVE else 1.0
        
        new_mag = self.magnitude * (growth ** (time_delta * self.awareness_coefficient))
        new_mag *= (decay ** (time_delta * (1 - self.awareness_coefficient)))
        
        return KarmicVector(
            magnitude=new_mag, valence=self.valence,
            velocity=self.velocity * 0.99, spin=self.spin * 0.999,
            entangled_ids=self.entangled_ids.copy(),
            source_intention=self.source_intention,
            awareness_coefficient=self.awareness_coefficient
        )


class KarmaDharmaEngine:
    """Core spiritual mathematics engine"""
    
    def __init__(self):
        self.vectors: List[KarmicVector] = []
        self.field_potential: float = 0.0
        self.dharmic_angle: float = 0.0
        self.resonance_frequency: float = DHARMA_FREQUENCY
        self.action_history: List[Dict] = []
    
    def add_action(self, intention: float, magnitude: float, 
                   awareness: float, valence: KarmicValence) -> str:
        """Add action to karmic field, returns vector ID"""
        vector_id = hashlib.sha256(
            f"{time.time()}-{intention}-{magnitude}-{random.random()}".encode()
        ).hexdigest()[:16]
        
        fib_idx = int(magnitude * 10) % len(FIBONACCI)
        initial_spin = FIBONACCI[fib_idx] * PHI_INVERSE
        
        vector = KarmicVector(
            magnitude=magnitude, valence=valence,
            velocity=intention * awareness,
            spin=initial_spin, entangled_ids=[vector_id],
            source_intention=intention,
            awareness_coefficient=awareness
        )
        
        self.vectors.append(vector)
        self._recalculate_field()
        
        self.action_history.append({
            'id': vector_id, 'timestamp': time.time(),
            'intention': intention, 'magnitude': magnitude,
            'awareness': awareness, 'valence': valence.name,
            'karmic_weight': vector.karmic_weight
        })
        
        return vector_id
    
    def _recalculate_field(self):
        """Recalculate total field potential using superposition"""
        positive = sum(v.karmic_weight for v in self.vectors if v.valence == KarmicValence.POSITIVE)
        negative = sum(v.karmic_weight for v in self.vectors if v.valence == KarmicValence.NEGATIVE)
        transform = sum(v.karmic_weight for v in self.vectors if v.valence == KarmicValence.TRANSFORMATIVE)
        
        self.field_potential = positive * PHI - negative * PHI_INVERSE + transform * math.sqrt(PHI)
    
    def get_dharmic_alignment(self) -> float:
        """D = cos(θ) where θ = angle from perfect alignment"""
        return math.cos(self.dharmic_angle)
    
    def adjust_dharmic_alignment(self, delta: float):
        """Adjust alignment toward or away from cosmic order"""
        self.dharmic_angle = max(0, min(math.pi, self.dharmic_angle + delta))
    
    def evolve(self, time_delta: float = 1.0):
        """Evolve all vectors through time"""
        self.vectors = [v.propagate(time_delta) for v in self.vectors]
        self.vectors = [v for v in self.vectors if v.magnitude > PLANCK_KARMA]
        self._recalculate_field()
        
        # Dharmic alignment naturally improves with positive karma
        if self.field_potential > 0:
            self.dharmic_angle *= (1 - 0.01 * time_delta)
    
    def get_state(self) -> Dict:
        return {
            'field_potential': self.field_potential,
            'dharmic_alignment': self.get_dharmic_alignment(),
            'dharmic_angle': self.dharmic_angle,
            'vector_count': len(self.vectors),
            'resonance_frequency': self.resonance_frequency,
            'recent_actions': self.action_history[-10:]
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# SWARM INTELLIGENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════

class SwarmRole(Enum):
    SCOUT = "scout"
    WORKER = "worker"
    LEADER = "leader"
    MESSENGER = "messenger"
    GUARDIAN = "guardian"
    HEALER = "healer"


class SignalType(Enum):
    FOOD = "food"
    DANGER = "danger"
    PATH = "path"
    GATHERING = "gathering"
    KARMA_POSITIVE = "karma_positive"
    KARMA_NEGATIVE = "karma_negative"


@dataclass
class SwarmAgent:
    """Individual agent in the swarm exhibiting collective intelligence"""
    id: str = field(default_factory=lambda: hashlib.sha256(
        str(time.time() + random.random()).encode()).hexdigest()[:10])
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [
        random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.5, 0.5)])
    max_speed: float = 2.0
    max_force: float = 0.1
    perception_radius: float = 10.0
    role: SwarmRole = SwarmRole.WORKER
    karmic_charge: float = 0.0
    dharmic_alignment: float = 1.0
    resonance_frequency: float = 432.0
    
    def calculate_steering(self, neighbors: List['SwarmAgent'],
                          targets: List[Tuple[float, float, float]] = None) -> List[float]:
        """Calculate steering force based on boid rules + karma-dharma"""
        force = [0.0, 0.0, 0.0]
        
        if neighbors:
            # Separation
            sep = self._separation(neighbors)
            force = [f + s * 1.5 * PHI_INVERSE for f, s in zip(force, sep)]
            
            # Alignment
            ali = self._alignment(neighbors)
            force = [f + a * 1.0 for f, a in zip(force, ali)]
            
            # Cohesion
            coh = self._cohesion(neighbors)
            force = [f + c * PHI_INVERSE for f, c in zip(force, coh)]
            
            # Karma attraction
            kar = self._karma_attraction(neighbors)
            force = [f + k * 0.5 for f, k in zip(force, kar)]
        
        if targets:
            tar = self._seek_targets(targets)
            force = [f + t * 0.8 for f, t in zip(force, tar)]
        
        # Limit force magnitude
        mag = math.sqrt(sum(f*f for f in force))
        if mag > self.max_force:
            force = [f * self.max_force / mag for f in force]
        
        return force
    
    def _distance_to(self, pos: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, pos)))
    
    def _separation(self, neighbors: List['SwarmAgent']) -> List[float]:
        steer = [0.0, 0.0, 0.0]
        count = 0
        for other in neighbors:
            d = self._distance_to(other.position)
            if 0 < d < self.perception_radius * 0.5:
                diff = [(self.position[i] - other.position[i]) / d for i in range(3)]
                steer = [steer[i] + diff[i] for i in range(3)]
                count += 1
        return [s / count if count > 0 else s for s in steer]
    
    def _alignment(self, neighbors: List['SwarmAgent']) -> List[float]:
        avg_vel = [0.0, 0.0, 0.0]
        for other in neighbors:
            avg_vel = [avg_vel[i] + other.velocity[i] for i in range(3)]
        n = len(neighbors)
        return [v / n - self.velocity[i] for i, v in enumerate(avg_vel)] if n > 0 else [0, 0, 0]
    
    def _cohesion(self, neighbors: List['SwarmAgent']) -> List[float]:
        center = [0.0, 0.0, 0.0]
        for other in neighbors:
            center = [center[i] + other.position[i] for i in range(3)]
        n = len(neighbors)
        if n > 0:
            center = [c / n for c in center]
            return [center[i] - self.position[i] for i in range(3)]
        return [0, 0, 0]
    
    def _karma_attraction(self, neighbors: List['SwarmAgent']) -> List[float]:
        attract = [0.0, 0.0, 0.0]
        for other in neighbors:
            karma_diff = abs(self.karmic_charge - other.karmic_charge)
            if karma_diff < 0.3:  # Similar karma = attraction
                d = self._distance_to(other.position)
                if d > 0:
                    direction = [(other.position[i] - self.position[i]) / d for i in range(3)]
                    strength = (1 - karma_diff / 0.3) * PHI_INVERSE
                    attract = [attract[i] + direction[i] * strength for i in range(3)]
        return attract
    
    def _seek_targets(self, targets: List[Tuple[float, float, float]]) -> List[float]:
        if not targets:
            return [0, 0, 0]
        closest = min(targets, key=lambda t: self._distance_to(list(t)))
        d = self._distance_to(list(closest))
        if d > 0:
            return [(closest[i] - self.position[i]) / d for i in range(3)]
        return [0, 0, 0]
    
    def update(self, force: List[float], dt: float = 0.1):
        """Apply force and update position"""
        self.velocity = [self.velocity[i] + force[i] for i in range(3)]
        speed = math.sqrt(sum(v*v for v in self.velocity))
        if speed > self.max_speed:
            self.velocity = [v * self.max_speed / speed for v in self.velocity]
        self.position = [self.position[i] + self.velocity[i] * dt for i in range(3)]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id, 'position': self.position, 'velocity': self.velocity,
            'role': self.role.value, 'karmic_charge': self.karmic_charge,
            'dharmic_alignment': self.dharmic_alignment
        }


class SwarmCollective:
    """Manages the swarm collective intelligence"""
    
    def __init__(self, name: str = "KarmicSwarm", initial_population: int = 21):
        self.name = name
        self.agents: Dict[str, SwarmAgent] = {}
        self.targets: List[Tuple[float, float, float]] = []
        self.collective_karma: float = 0.0
        self.collective_dharma: float = 1.0
        self.time_elapsed: float = 0.0
        
        # Initialize with Fibonacci population
        self.spawn_agents(initial_population)
    
    def spawn_agents(self, count: int, position: List[float] = None,
                    role: SwarmRole = None, karma: float = None):
        for _ in range(count):
            pos = position or [random.uniform(-50, 50) for _ in range(3)]
            agent = SwarmAgent(
                position=pos.copy() if position else pos,
                role=role or random.choice(list(SwarmRole)),
                karmic_charge=karma if karma is not None else random.uniform(-0.5, 0.5)
            )
            self.agents[agent.id] = agent
    
    def update(self, dt: float = 0.1):
        """Update all agents"""
        self.time_elapsed += dt
        
        # Get neighbors for each agent
        agent_list = list(self.agents.values())
        
        for agent in agent_list:
            neighbors = [a for a in agent_list 
                        if a.id != agent.id and 
                        agent._distance_to(a.position) < agent.perception_radius]
            force = agent.calculate_steering(neighbors, self.targets)
            agent.update(force, dt)
        
        # Update collective metrics
        if self.agents:
            self.collective_karma = sum(a.karmic_charge for a in self.agents.values()) / len(self.agents)
            self.collective_dharma = sum(a.dharmic_alignment for a in self.agents.values()) / len(self.agents)
    
    def inject_karma(self, position: List[float], karma_value: float, radius: float = 20.0):
        """Inject karma at a position, affecting nearby agents"""
        for agent in self.agents.values():
            dist = agent._distance_to(position)
            if dist <= radius:
                influence = (1 - dist / radius) * karma_value
                agent.karmic_charge = max(-1, min(1, agent.karmic_charge + influence * 0.5))
    
    def get_visualization_data(self) -> Dict:
        return {
            'agents': [a.to_dict() for a in self.agents.values()],
            'targets': self.targets,
            'collective_karma': self.collective_karma,
            'collective_dharma': self.collective_dharma,
            'population': len(self.agents),
            'time_elapsed': self.time_elapsed
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL ORB SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════

class CellType(Enum):
    STEM = "stem"
    NEURON = "neuron"
    MEMORY = "memory"
    SENSOR = "sensor"
    EFFECTOR = "effector"
    STRUCTURAL = "structural"
    TRANSPORT = "transport"


class CellState(Enum):
    NASCENT = "nascent"
    GROWING = "growing"
    MATURE = "mature"
    DIVIDING = "dividing"
    SENESCENT = "senescent"


@dataclass
class BiologicalOrb:
    """Self-spawning organic structure with fractal logic"""
    id: str = field(default_factory=lambda: hashlib.sha256(
        str(time.time() + random.random()).encode()).hexdigest()[:12])
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    cell_type: CellType = CellType.STEM
    state: CellState = CellState.NASCENT
    energy: float = 1.0
    age: float = 0.0
    generation: int = 0
    karmic_charge: float = 0.0
    dharmic_alignment: float = 1.0
    bindings: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    def update(self, dt: float, environment: Dict) -> Optional['BiologicalOrb']:
        """Update cell state, may return child from division"""
        self.age += dt
        
        # Metabolism
        nutrients = environment.get('nutrients', 0.5)
        self.energy += nutrients * dt * PHI * 0.1
        self.energy -= dt * 0.05  # Base metabolism
        self.energy = max(0.0, min(2.0, self.energy))
        
        # State transitions
        if self.state == CellState.NASCENT and self.age > 1.0:
            self.state = CellState.GROWING
        elif self.state == CellState.GROWING and self.age > 5.0:
            self.state = CellState.MATURE
        elif self.state == CellState.MATURE and self.energy > 1.5:
            self.state = CellState.DIVIDING
        elif self.age > 50.0:
            self.state = CellState.SENESCENT
        
        # Division
        if self.state == CellState.DIVIDING:
            return self._divide()
        
        return None
    
    def _divide(self) -> 'BiologicalOrb':
        """Mitosis - create child cell"""
        self.state = CellState.MATURE
        self.energy *= 0.5
        
        # Golden angle offset for child position
        angle = random.random() * 2 * math.pi
        offset = self.radius * PHI
        
        child_pos = (
            self.position[0] + math.cos(angle) * offset,
            self.position[1] + math.sin(angle) * offset,
            self.position[2] + random.uniform(-1, 1) * offset * 0.5
        )
        
        child = BiologicalOrb(
            position=child_pos,
            radius=self.radius * PHI_INVERSE,
            cell_type=self._differentiate(),
            energy=self.energy,
            generation=self.generation + 1,
            karmic_charge=self.karmic_charge * 0.9,
            dharmic_alignment=self.dharmic_alignment,
            parent_id=self.id
        )
        
        self.children_ids.append(child.id)
        return child
    
    def _differentiate(self) -> CellType:
        """Determine child cell type based on parent"""
        if self.cell_type == CellType.STEM:
            return random.choice(list(CellType))
        return self.cell_type
    
    def can_bind_with(self, other: 'BiologicalOrb') -> Tuple[bool, float]:
        """Check if can bind with another orb"""
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))
        max_dist = (self.radius + other.radius) * PHI
        
        if distance > max_dist:
            return False, 0.0
        
        # Compatible types bind more strongly
        compatible = {
            (CellType.NEURON, CellType.MEMORY): 0.9,
            (CellType.SENSOR, CellType.NEURON): 0.8,
            (CellType.NEURON, CellType.EFFECTOR): 0.8,
            (CellType.STRUCTURAL, CellType.STRUCTURAL): 0.95,
        }
        
        type_pair = (self.cell_type, other.cell_type)
        strength = compatible.get(type_pair, compatible.get((type_pair[1], type_pair[0]), 0.5))
        
        # Karma compatibility
        karma_compat = 1 - abs(self.karmic_charge - other.karmic_charge)
        
        return True, strength * karma_compat
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id, 'position': self.position, 'radius': self.radius,
            'cell_type': self.cell_type.value, 'state': self.state.value,
            'energy': self.energy, 'age': self.age, 'generation': self.generation,
            'karmic_charge': self.karmic_charge, 'dharmic_alignment': self.dharmic_alignment,
            'bindings': len(self.bindings), 'children': len(self.children_ids)
        }


class OrganicTissue:
    """Collection of biological orbs forming living tissue"""
    
    def __init__(self, karma_engine: KarmaDharmaEngine):
        self.orbs: Dict[str, BiologicalOrb] = {}
        self.karma_engine = karma_engine
        self.environment = {'nutrients': 0.5, 'harmony': 1.0}
        self.time_elapsed: float = 0.0
    
    def spawn_orb(self, position: Tuple[float, float, float] = None,
                  cell_type: CellType = CellType.STEM,
                  initial_energy: float = 1.0) -> BiologicalOrb:
        pos = position or (random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-10, 10))
        orb = BiologicalOrb(
            position=pos, cell_type=cell_type, energy=initial_energy,
            karmic_charge=self.karma_engine.field_potential * 0.1
        )
        self.orbs[orb.id] = orb
        return orb
    
    def spawn_fractal_pattern(self, count: int = 13, center: Tuple[float, float, float] = (0, 0, 0)):
        """Spawn orbs in golden spiral pattern"""
        for i in range(count):
            angle = i * GOLDEN_ANGLE_RAD
            radius = math.sqrt(i + 1) * 5
            pos = (
                center[0] + math.cos(angle) * radius,
                center[1] + math.sin(angle) * radius,
                center[2] + (i % 3 - 1) * 3
            )
            cell_type = list(CellType)[i % len(CellType)]
            self.spawn_orb(pos, cell_type)
    
    def update(self, dt: float = 0.1):
        """Update all orbs"""
        self.time_elapsed += dt
        new_orbs = []
        dead_orbs = []
        
        for orb_id, orb in self.orbs.items():
            child = orb.update(dt, self.environment)
            if child:
                new_orbs.append(child)
            if orb.energy <= 0.1:
                dead_orbs.append(orb_id)
        
        for child in new_orbs:
            self.orbs[child.id] = child
        
        for dead_id in dead_orbs:
            del self.orbs[dead_id]
        
        # Update environment based on collective state
        if self.orbs:
            avg_karma = sum(o.karmic_charge for o in self.orbs.values()) / len(self.orbs)
            self.environment['harmony'] = max(0, min(1, 0.5 + avg_karma))
            self.environment['nutrients'] = 0.3 + self.environment['harmony'] * 0.4
    
    def get_visualization_data(self) -> Dict:
        return {
            'orbs': [o.to_dict() for o in self.orbs.values()],
            'orb_count': len(self.orbs),
            'environment': self.environment,
            'time_elapsed': self.time_elapsed
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FRACTAL MATHEMATICS ENGINE (Lightweight - Heavy 3D moved to client)
# ═══════════════════════════════════════════════════════════════════════════════════════════

class FractalEngine:
    """
    Generates fractal visualizations with sacred geometry.
    NOTE: 3D Mandelbulb now rendered client-side via WebGL to prevent server timeouts.
    Server provides parameters only; client does GPU computation.
    """
    
    def __init__(self):
        self.phi = PHI
        self.iterations_2d = 100
        self.escape_radius = 2.0
    
    def generate_2d_fractal(self, wellness_data: Dict = None, mood: str = 'calm',
                           width: int = 400, height: int = 400) -> Image.Image:
        """Generate 2D Mandelbrot fractal influenced by wellness data"""
        wellness = wellness_data or {}
        
        # Map wellness to fractal parameters
        energy = wellness.get('energy', 50) / 100
        focus = wellness.get('focus', 50) / 100
        mood_val = wellness.get('mood', 50) / 100
        
        # Color palette based on mood (autism-safe muted colors)
        palettes = {
            'calm': [(70, 130, 180), (135, 206, 235), (176, 224, 230)],
            'energized': [(255, 183, 77), (255, 167, 38), (255, 152, 0)],
            'focused': [(129, 199, 132), (102, 187, 106), (76, 175, 80)],
            'relaxed': [(186, 104, 200), (171, 71, 188), (156, 39, 176)],
            'tired': [(158, 158, 158), (189, 189, 189), (224, 224, 224)]
        }
        colors = palettes.get(mood, palettes['calm'])
        
        # Mandelbrot bounds adjusted by energy
        x_center = -0.5 + (energy - 0.5) * 0.3
        y_center = (mood_val - 0.5) * 0.3
        zoom = 1.5 + focus * 1.0
        
        x_min, x_max = x_center - 2/zoom, x_center + 2/zoom
        y_min, y_max = y_center - 2/zoom, y_center + 2/zoom
        
        # Generate fractal
        img = Image.new('RGB', (width, height), (20, 20, 30))
        pixels = img.load()
        
        max_iter = int(50 + focus * 100)
        
        for px in range(width):
            for py in range(height):
                x0 = x_min + (x_max - x_min) * px / width
                y0 = y_min + (y_max - y_min) * py / height
                
                x, y = 0.0, 0.0
                iteration = 0
                
                while x*x + y*y <= 4 and iteration < max_iter:
                    x_new = x*x - y*y + x0
                    y = 2*x*y + y0
                    x = x_new
                    iteration += 1
                
                if iteration < max_iter:
                    # Smooth coloring
                    smooth = iteration + 1 - math.log(math.log(max(1, x*x + y*y))) / math.log(2)
                    smooth = max(0, smooth)
                    
                    idx = int(smooth * PHI) % len(colors)
                    color = colors[idx]
                    
                    brightness = int((smooth / max_iter) * 255)
                    pixels[px, py] = (
                        min(255, color[0] * brightness // 255),
                        min(255, color[1] * brightness // 255),
                        min(255, color[2] * brightness // 255)
                    )
        
        # Apply golden ratio vignette
        img = self._apply_golden_vignette(img)
        
        return img
    
    def _apply_golden_vignette(self, img: Image.Image) -> Image.Image:
        """Apply vignette based on golden spiral"""
        width, height = img.size
        center_x, center_y = width * PHI_INVERSE, height * PHI_INVERSE
        max_dist = math.sqrt(width**2 + height**2) / 2
        
        pixels = img.load()
        for x in range(width):
            for y in range(height):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                factor = max(0.3, 1 - (dist / max_dist) * 0.5)
                r, g, b = pixels[x, y]
                pixels[x, y] = (int(r * factor), int(g * factor), int(b * factor))
        
        return img
    
    def get_3d_parameters(self, wellness_data: Dict = None) -> Dict:
        """
        Return parameters for client-side WebGL 3D rendering.
        This fixes the server timeout issue by offloading computation to GPU.
        """
        wellness = wellness_data or {}
        
        energy = wellness.get('energy', 50) / 100
        focus = wellness.get('focus', 50) / 100
        mood = wellness.get('mood', 50) / 100
        
        return {
            'type': 'mandelbulb',
            'power': 8.0 + energy * 4.0,  # Power parameter
            'iterations': int(8 + focus * 8),
            'bailout': 2.0,
            'zoom': 1.5 + focus * 0.5,
            'rotation': [mood * math.pi * 2, energy * math.pi, focus * math.pi * 0.5],
            'color_scheme': {
                'base': [0.27, 0.51, 0.71],  # Steel blue
                'accent': [1.0, 0.72, 0.3],  # Warm gold
                'glow': [0.5, 0.8, 1.0]
            },
            'lighting': {
                'ambient': 0.3,
                'diffuse': 0.7,
                'specular': 0.5,
                'position': [2.0, 2.0, 2.0]
            },
            'sacred_geometry': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE,
                'fibonacci': FIBONACCI[:10]
            }
        }
    
    def generate_karma_fractal_pattern(self, karma_value: float, iterations: int = 5) -> List[Dict]:
        """Generate fractal pattern based on karma value"""
        nodes = []
        
        def propagate(depth: int, magnitude: float, angle: float, x: float, y: float):
            if depth >= iterations or magnitude < 0.01:
                return
            
            nodes.append({
                'depth': depth,
                'magnitude': magnitude,
                'angle': angle,
                'x': x, 'y': y,
                'karma': karma_value * magnitude
            })
            
            # Branch using golden ratio
            branch_count = FIBONACCI[min(depth + 2, len(FIBONACCI) - 1)] % 5 + 2
            new_magnitude = magnitude * PHI_INVERSE
            
            for i in range(branch_count):
                branch_angle = angle + (i - branch_count / 2) * GOLDEN_ANGLE_RAD * 0.5
                distance = magnitude * 20 * PHI_INVERSE
                
                new_x = x + math.cos(branch_angle) * distance
                new_y = y + math.sin(branch_angle) * distance
                
                propagate(depth + 1, new_magnitude, branch_angle, new_x, new_y)
        
        propagate(0, abs(karma_value) + 0.5, 0, 0, 0)
        return nodes


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ORIGAMI LOGIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class OrigamiLogicEngine:
    """
    Implements origami mathematics - fold transformations and dimensional reduction.
    Actions "fold" reality, creating new configurations.
    """
    
    def __init__(self):
        self.fold_history: List[Dict] = []
        self.current_state = np.eye(4)  # 4D transformation matrix
    
    def fold(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Apply fold transformation along an axis using Rodrigues' formula"""
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        
        self.current_state = transform @ self.current_state
        
        self.fold_history.append({
            'axis': axis.tolist(),
            'angle': angle,
            'golden_ratio_position': angle / PHI,
            'timestamp': time.time()
        })
        
        return self.current_state
    
    def karma_fold(self, karmic_weight: float, spin: float, valence: int) -> np.ndarray:
        """Apply fold based on karmic properties"""
        axis = np.array([
            math.cos(spin),
            math.sin(spin),
            valence * 0.5
        ])
        angle = karmic_weight * PHI_INVERSE * math.pi
        return self.fold(axis, angle)
    
    def get_crease_pattern(self) -> Dict:
        """Generate 2D crease pattern from fold history"""
        creases = []
        for i, fold in enumerate(self.fold_history):
            crease = {
                'index': i,
                'x1': math.cos(fold['angle']) * (i + 1),
                'y1': math.sin(fold['angle']) * (i + 1),
                'x2': -math.cos(fold['angle']) * (i + 1),
                'y2': -math.sin(fold['angle']) * (i + 1),
                'type': 'mountain' if fold['axis'][2] > 0 else 'valley',
                'golden_position': (i + 1) * PHI_INVERSE
            }
            creases.append(crease)
        
        return {
            'creases': creases,
            'total_folds': len(self.fold_history),
            'complexity': sum(abs(f['angle']) for f in self.fold_history) / max(1, len(self.fold_history))
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class MLEvolutionEngine:
    """Machine learning for pattern detection and system evolution"""
    
    def __init__(self):
        self.pattern_memory: List[Dict] = []
        self.predictions: Dict[str, float] = {}
        self.model_trained = False
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.model = LinearRegression() if HAS_SKLEARN else None
    
    def record_state(self, state: Dict):
        """Record system state for pattern learning"""
        self.pattern_memory.append({
            'timestamp': time.time(),
            'karma': state.get('karma', 0),
            'dharma': state.get('dharma', 1),
            'energy': state.get('energy', 50),
            'mood': state.get('mood', 50),
            'orb_count': state.get('orb_count', 0),
            'agent_count': state.get('agent_count', 0),
            'harmony': state.get('harmony', 0.5)
        })
        
        # Keep last 1000 samples
        if len(self.pattern_memory) > 1000:
            self.pattern_memory = self.pattern_memory[-1000:]
        
        # Retrain periodically
        if len(self.pattern_memory) >= 50 and len(self.pattern_memory) % 50 == 0:
            self._train_model()
    
    def _train_model(self):
        """Train prediction model on collected data"""
        if not HAS_SKLEARN or len(self.pattern_memory) < 50:
            return
        
        try:
            # Prepare features and targets
            X = []
            y = []
            
            for i in range(len(self.pattern_memory) - 1):
                current = self.pattern_memory[i]
                next_state = self.pattern_memory[i + 1]
                
                X.append([
                    current['karma'], current['dharma'],
                    current['energy'] / 100, current['mood'] / 100,
                    current['harmony']
                ])
                y.append(next_state['harmony'])
            
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.model_trained = True
            
            logger.info("🧠 ML model trained on %d samples", len(X))
        except Exception as e:
            logger.warning("ML training failed: %s", e)
    
    def predict_harmony(self, current_state: Dict) -> float:
        """Predict future harmony based on current state"""
        if not self.model_trained or not HAS_SKLEARN:
            return current_state.get('harmony', 0.5)
        
        try:
            features = np.array([[
                current_state.get('karma', 0),
                current_state.get('dharma', 1),
                current_state.get('energy', 50) / 100,
                current_state.get('mood', 50) / 100,
                current_state.get('harmony', 0.5)
            ]])
            
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            return max(0, min(1, prediction))
        except:
            return current_state.get('harmony', 0.5)
    
    def detect_patterns(self) -> Dict:
        """Detect patterns in the data"""
        if len(self.pattern_memory) < 10:
            return {'patterns': [], 'insights': []}
        
        patterns = []
        insights = []
        
        # Trend detection
        recent = self.pattern_memory[-10:]
        karma_trend = recent[-1]['karma'] - recent[0]['karma']
        harmony_trend = recent[-1]['harmony'] - recent[0]['harmony']
        
        if karma_trend > 0.1:
            patterns.append('karma_rising')
            insights.append('Your karmic energy is increasing. Keep up the positive actions!')
        elif karma_trend < -0.1:
            patterns.append('karma_falling')
            insights.append('Consider more mindful actions to restore balance.')
        
        if harmony_trend > 0.1:
            patterns.append('harmony_improving')
            insights.append('System harmony is improving. You\'re on the right path.')
        
        # Cyclical pattern detection
        if len(self.pattern_memory) >= 50:
            # Simple FFT-inspired cycle detection
            harmonies = [s['harmony'] for s in self.pattern_memory[-50:]]
            mean = sum(harmonies) / len(harmonies)
            variance = sum((h - mean) ** 2 for h in harmonies) / len(harmonies)
            
            if variance > 0.05:
                patterns.append('cyclical_harmony')
                insights.append('Your harmony follows natural cycles. Embrace the rhythm.')
        
        return {
            'patterns': patterns,
            'insights': insights,
            'karma_trend': karma_trend,
            'harmony_trend': harmony_trend,
            'data_points': len(self.pattern_memory),
            'model_trained': self.model_trained
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# LIVING ORGANISM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════════════

class OrganismMode(Enum):
    DORMANT = "dormant"
    AWAKENING = "awakening"
    ACTIVE = "active"
    FOCUSED = "focused"
    MEDITATIVE = "meditative"
    CREATIVE = "creative"


class LivingOrganism:
    """
    Central orchestrator integrating all mathematical systems
    into a unified, living consciousness.
    """
    
    def __init__(self):
        self.mode = OrganismMode.DORMANT
        self.creation_time = time.time()
        
        # Core systems
        self.karma_engine = KarmaDharmaEngine()
        self.fractal_engine = FractalEngine()
        self.origami_engine = OrigamiLogicEngine()
        self.ml_engine = MLEvolutionEngine()
        
        # Biological layer
        self.tissue = OrganicTissue(self.karma_engine)
        self.tissue.spawn_fractal_pattern(FIBONACCI[7])  # 13 initial orbs
        
        # Swarm layer
        self.swarm = SwarmCollective(initial_population=FIBONACCI[6])  # 8 initial agents
        
        # State
        self.uptime: float = 0.0
        self.total_actions: int = 0
        self.harmony_index: float = 1.0
    
    def initialize(self):
        """Wake up the organism"""
        self.mode = OrganismMode.AWAKENING
        self.karma_engine.add_action(1.0, 1.0, 1.0, KarmicValence.POSITIVE)
        self.mode = OrganismMode.ACTIVE
        logger.info("🌀 Living organism awakened")
    
    def update(self, dt: float = 0.1):
        """Main update loop"""
        self.uptime = time.time() - self.creation_time
        
        # Update karma-dharma
        self.karma_engine.evolve(dt)
        
        # Update biological tissue
        self.tissue.update(dt)
        
        # Update swarm
        self.swarm.update(dt)
        
        # Inject karma into swarm based on tissue state
        if self.tissue.orbs:
            avg_pos = [
                sum(o.position[i] for o in self.tissue.orbs.values()) / len(self.tissue.orbs)
                for i in range(3)
            ]
            self.swarm.inject_karma(avg_pos, self.karma_engine.field_potential * 0.1)
        
        # Calculate harmony index
        self.harmony_index = (
            self.karma_engine.get_dharmic_alignment() * PHI +
            self.tissue.environment['harmony'] * PHI_INVERSE +
            self.swarm.collective_dharma
        ) / (PHI + PHI_INVERSE + 1)
        
        # Record state for ML
        self.ml_engine.record_state({
            'karma': self.karma_engine.field_potential,
            'dharma': self.karma_engine.get_dharmic_alignment(),
            'energy': 50 + self.karma_engine.field_potential * 50,
            'mood': 50 + self.swarm.collective_karma * 50,
            'orb_count': len(self.tissue.orbs),
            'agent_count': len(self.swarm.agents),
            'harmony': self.harmony_index
        })
    
    def process_user_action(self, action_type: str, magnitude: float = 1.0,
                           intention: float = 0.8, awareness: float = 0.7) -> Dict:
        """Process user action through the entire organism"""
        self.total_actions += 1
        
        # Determine valence from action type
        positive_actions = {'complete', 'achieve', 'help', 'create', 'meditate', 'exercise', 'learn'}
        negative_actions = {'skip', 'procrastinate', 'abandon'}
        
        if action_type.lower() in positive_actions:
            valence = KarmicValence.POSITIVE
        elif action_type.lower() in negative_actions:
            valence = KarmicValence.NEGATIVE
        else:
            valence = KarmicValence.NEUTRAL
        
        # Add to karma engine
        karma_id = self.karma_engine.add_action(intention, magnitude, awareness, valence)
        
        # Apply origami fold
        fold_result = self.origami_engine.karma_fold(
            magnitude * intention, random.random() * math.pi * 2, valence.value
        )
        
        # Spawn celebration orbs for positive actions
        celebration_orbs = 0
        if valence == KarmicValence.POSITIVE:
            celebration_orbs = FIBONACCI[min(int(magnitude * 5), 8)]
            for _ in range(celebration_orbs):
                self.tissue.spawn_orb(initial_energy=1.2)
        
        # Update mode based on action
        if action_type.lower() == 'meditate':
            self.mode = OrganismMode.MEDITATIVE
        elif action_type.lower() == 'create':
            self.mode = OrganismMode.CREATIVE
        elif action_type.lower() == 'focus':
            self.mode = OrganismMode.FOCUSED
        
        return {
            'karma_id': karma_id,
            'karmic_weight': self.karma_engine.action_history[-1]['karmic_weight'] if self.karma_engine.action_history else 0,
            'celebration_orbs': celebration_orbs,
            'current_harmony': self.harmony_index,
            'predicted_harmony': self.ml_engine.predict_harmony({
                'karma': self.karma_engine.field_potential,
                'dharma': self.karma_engine.get_dharmic_alignment(),
                'harmony': self.harmony_index
            }),
            'mode': self.mode.value
        }
    
    def get_state(self) -> Dict:
        """Get complete organism state"""
        return {
            'mode': self.mode.value,
            'uptime': self.uptime,
            'total_actions': self.total_actions,
            'harmony_index': self.harmony_index,
            'karma': self.karma_engine.get_state(),
            'tissue': self.tissue.get_visualization_data(),
            'swarm': self.swarm.get_visualization_data(),
            'origami': self.origami_engine.get_crease_pattern(),
            'ml': self.ml_engine.detect_patterns(),
            'sacred_constants': {
                'phi': PHI,
                'golden_angle': GOLDEN_ANGLE,
                'dharma_frequency': DHARMA_FREQUENCY,
                'fibonacci': FIBONACCI[:10]
            }
        }
    
    def get_visualization_data(self) -> Dict:
        """Get all data needed for Three.js visualization"""
        return {
            'orbs': [o.to_dict() for o in self.tissue.orbs.values()],
            'agents': [a.to_dict() for a in self.swarm.agents.values()],
            'karma_pattern': self.fractal_engine.generate_karma_fractal_pattern(
                self.karma_engine.field_potential
            ),
            'creases': self.origami_engine.get_crease_pattern()['creases'],
            'state': {
                'karma': self.karma_engine.field_potential,
                'dharma': self.karma_engine.get_dharmic_alignment(),
                'harmony': self.harmony_index,
                'frequency': self.karma_engine.resonance_frequency
            },
            'fractal_3d_params': self.fractal_engine.get_3d_parameters({
                'energy': 50 + self.karma_engine.field_potential * 50,
                'focus': 50 + self.harmony_index * 50,
                'mood': 50 + self.swarm.collective_karma * 50
            })
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# DATABASE - PRODUCTION READY
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Database:
    """Production SQLite database with all tables"""
    
    def __init__(self, db_path: str = "life_planner_v11.db"):
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
        
        # Users
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
                subscription_status TEXT DEFAULT 'active'
            )
        ''')
        
        # Goals
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
                karma_invested REAL DEFAULT 0.0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Habits
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
                last_completed TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Daily wellness entries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_entries (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                mood_level INTEGER DEFAULT 50,
                energy_level INTEGER DEFAULT 50,
                focus_level INTEGER DEFAULT 50,
                stress_level INTEGER DEFAULT 50,
                sleep_hours REAL DEFAULT 7.0,
                spoons_available INTEGER DEFAULT 12,
                spoons_used INTEGER DEFAULT 0,
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
                name TEXT DEFAULT 'Karma',
                hunger REAL DEFAULT 50.0,
                energy REAL DEFAULT 50.0,
                happiness REAL DEFAULT 50.0,
                level INTEGER DEFAULT 1,
                experience INTEGER DEFAULT 0,
                karma_earned REAL DEFAULT 0.0,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Karma history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS karma_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                karma_value REAL NOT NULL,
                dharma_alignment REAL DEFAULT 1.0,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def execute(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Execute query safely"""
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
        """Execute query and return single result"""
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
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Initialize systems
db = Database()
organism = LivingOrganism()
organism.initialize()

# Background update thread
def background_update():
    while True:
        try:
            organism.update(0.1)
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(1)

update_thread = threading.Thread(target=background_update, daemon=True)
update_thread.start()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AUTH DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════════════════

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.get_json()
    
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user_id = secrets.token_hex(16)
    password_hash = generate_password_hash(data['password'])
    
    try:
        db.execute('''
            INSERT INTO users (id, email, password_hash, first_name, last_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, data['email'], password_hash, 
              data.get('first_name', ''), data.get('last_name', ''),
              datetime.now(timezone.utc).isoformat()))
        
        # Initialize pet
        db.execute('''
            INSERT INTO pet_state (user_id, last_updated)
            VALUES (?, ?)
        ''', (user_id, datetime.now(timezone.utc).isoformat()))
        
        session['user_id'] = user_id
        return jsonify({'success': True, 'user_id': user_id})
    
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 409
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    user = db.execute_one(
        'SELECT * FROM users WHERE email = ?',
        (data.get('email', ''),)
    )
    
    if not user or not check_password_hash(user['password_hash'], data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user_id'] = user['id']
    
    db.execute(
        'UPDATE users SET last_login = ? WHERE id = ?',
        (datetime.now(timezone.utc).isoformat(), user['id'])
    )
    
    return jsonify({'success': True, 'user_id': user['id']})


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.pop('user_id', None)
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════════════════════════════
# GOALS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/goals', methods=['GET'])
@require_auth
def get_goals():
    """Get all goals for user"""
    goals = db.execute(
        'SELECT * FROM goals WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    )
    return jsonify([dict(g) for g in goals or []])


@app.route('/api/goals', methods=['POST'])
@require_auth
def create_goal():
    """Create new goal"""
    data = request.get_json()
    goal_id = secrets.token_hex(16)
    
    db.execute('''
        INSERT INTO goals (id, user_id, title, description, category, term, priority, target_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (goal_id, session['user_id'], data.get('title', 'New Goal'),
          data.get('description', ''), data.get('category', 'personal'),
          data.get('term', 'medium'), data.get('priority', 3),
          data.get('target_date'), datetime.now(timezone.utc).isoformat()))
    
    # Record karma for goal creation
    result = organism.process_user_action('create', 0.5, 0.8, 0.7)
    
    return jsonify({
        'id': goal_id, 'success': True,
        'karma_earned': result['karmic_weight']
    })


@app.route('/api/goals/<goal_id>/progress', methods=['POST'])
@require_auth
def update_goal_progress(goal_id):
    """Update goal progress"""
    data = request.get_json()
    progress = min(100, max(0, data.get('progress', 0)))
    
    db.execute(
        'UPDATE goals SET progress = ? WHERE id = ? AND user_id = ?',
        (progress, goal_id, session['user_id'])
    )
    
    # Karma based on progress
    karma_magnitude = progress / 100
    result = organism.process_user_action('achieve' if progress >= 100 else 'progress',
                                          karma_magnitude, 0.9, 0.8)
    
    # Fibonacci celebration for milestones
    milestone_hit = any(progress >= fib for fib in [5, 8, 13, 21, 34, 55, 89, 100] 
                       if data.get('previous_progress', 0) < fib)
    
    return jsonify({
        'success': True,
        'progress': progress,
        'karma_earned': result['karmic_weight'],
        'milestone': milestone_hit,
        'celebration_orbs': result['celebration_orbs']
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HABITS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/habits', methods=['GET'])
@require_auth
def get_habits():
    """Get all habits for user"""
    habits = db.execute(
        'SELECT * FROM habits WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    )
    return jsonify([dict(h) for h in habits or []])


@app.route('/api/habits', methods=['POST'])
@require_auth
def create_habit():
    """Create new habit"""
    data = request.get_json()
    habit_id = secrets.token_hex(16)
    
    db.execute('''
        INSERT INTO habits (id, user_id, name, description, frequency, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (habit_id, session['user_id'], data.get('name', 'New Habit'),
          data.get('description', ''), data.get('frequency', 'daily'),
          datetime.now(timezone.utc).isoformat()))
    
    return jsonify({'id': habit_id, 'success': True})


@app.route('/api/habits/<habit_id>/complete', methods=['POST'])
@require_auth
def complete_habit(habit_id):
    """Mark habit as completed"""
    habit = db.execute_one(
        'SELECT * FROM habits WHERE id = ? AND user_id = ?',
        (habit_id, session['user_id'])
    )
    
    if not habit:
        return jsonify({'error': 'Habit not found'}), 404
    
    new_streak = habit['current_streak'] + 1
    new_total = habit['total_completions'] + 1
    longest = max(habit['longest_streak'], new_streak)
    
    db.execute('''
        UPDATE habits SET current_streak = ?, total_completions = ?, 
        longest_streak = ?, last_completed = ? WHERE id = ?
    ''', (new_streak, new_total, longest,
          datetime.now(timezone.utc).isoformat(), habit_id))
    
    # Karma scales with streak (Fibonacci bonus)
    fib_bonus = 1.0
    for fib in FIBONACCI:
        if new_streak == fib:
            fib_bonus = PHI
            break
    
    result = organism.process_user_action('complete', 0.3 * fib_bonus, 0.9, 0.85)
    
    return jsonify({
        'success': True,
        'new_streak': new_streak,
        'total_completions': new_total,
        'karma_earned': result['karmic_weight'],
        'fibonacci_bonus': fib_bonus > 1.0
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# WELLNESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/wellness/checkin', methods=['POST'])
@require_auth
def wellness_checkin():
    """Daily wellness check-in"""
    data = request.get_json()
    entry_id = secrets.token_hex(16)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    try:
        db.execute('''
            INSERT OR REPLACE INTO daily_entries 
            (id, user_id, date, mood_level, energy_level, focus_level, stress_level,
             sleep_hours, spoons_available, spoons_used, journal_entry, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_id, session['user_id'], today,
              data.get('mood', 50), data.get('energy', 50),
              data.get('focus', 50), data.get('stress', 50),
              data.get('sleep_hours', 7.0), data.get('spoons_available', 12),
              data.get('spoons_used', 0), data.get('journal', ''),
              datetime.now(timezone.utc).isoformat()))
        
        # Karma for self-reflection
        result = organism.process_user_action('meditate', 0.4, 0.95, 0.9)
        
        return jsonify({
            'success': True,
            'karma_earned': result['karmic_weight'],
            'predicted_harmony': result['predicted_harmony']
        })
    except Exception as e:
        logger.error(f"Wellness check-in error: {e}")
        return jsonify({'error': 'Check-in failed'}), 500


@app.route('/api/wellness/today', methods=['GET'])
@require_auth
def get_today_wellness():
    """Get today's wellness data"""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    entry = db.execute_one(
        'SELECT * FROM daily_entries WHERE user_id = ? AND date = ?',
        (session['user_id'], today)
    )
    
    return jsonify(dict(entry) if entry else {
        'mood_level': 50, 'energy_level': 50, 'focus_level': 50,
        'stress_level': 50, 'spoons_available': 12, 'spoons_used': 0
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ORGANISM & VISUALIZATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/organism/state', methods=['GET'])
@require_auth
def get_organism_state():
    """Get complete organism state"""
    return jsonify(organism.get_state())


@app.route('/api/organism/visualization', methods=['GET'])
@require_auth
def get_visualization_data():
    """Get visualization data for Three.js"""
    return jsonify(organism.get_visualization_data())


@app.route('/api/organism/action', methods=['POST'])
@require_auth
def process_action():
    """Process user action through organism"""
    data = request.get_json()
    result = organism.process_user_action(
        data.get('action_type', 'neutral'),
        data.get('magnitude', 1.0),
        data.get('intention', 0.8),
        data.get('awareness', 0.7)
    )
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# FRACTAL VISUALIZATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/visualization/fractal-base64/<fractal_type>')
def generate_fractal_base64(fractal_type):
    """Generate fractal visualization as base64"""
    try:
        wellness = db.execute_one(
            'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY date DESC LIMIT 1',
            (session.get('user_id', ''),)
        )
        
        wellness_data = dict(wellness) if wellness else {}
        
        if fractal_type == '2d':
            # 2D Mandelbrot - rendered server-side
            mood = 'calm'
            if wellness_data.get('energy_level', 50) > 70:
                mood = 'energized'
            elif wellness_data.get('focus_level', 50) > 70:
                mood = 'focused'
            elif wellness_data.get('stress_level', 50) > 70:
                mood = 'tired'
            
            image = organism.fractal_engine.generate_2d_fractal(
                {'energy': wellness_data.get('energy_level', 50),
                 'focus': wellness_data.get('focus_level', 50),
                 'mood': wellness_data.get('mood_level', 50)},
                mood=mood
            )
            
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            return jsonify({
                'image': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'format': 'png',
                'sacred_geometry': {
                    'phi': PHI,
                    'golden_angle': GOLDEN_ANGLE
                }
            })
        
        elif fractal_type == '3d':
            # 3D Mandelbulb - parameters only (rendered client-side via WebGL)
            params = organism.fractal_engine.get_3d_parameters({
                'energy': wellness_data.get('energy_level', 50),
                'focus': wellness_data.get('focus_level', 50),
                'mood': wellness_data.get('mood_level', 50)
            })
            
            return jsonify({
                'render_mode': 'webgl',
                'parameters': params,
                'shader_type': 'ray_marching',
                'note': '3D rendering performed client-side for GPU acceleration'
            })
        
        else:
            return jsonify({'error': 'Unknown fractal type'}), 400
            
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════════
# PET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pet/state', methods=['GET'])
@require_auth
def get_pet_state():
    """Get pet state"""
    pet = db.execute_one(
        'SELECT * FROM pet_state WHERE user_id = ?',
        (session['user_id'],)
    )
    
    if not pet:
        return jsonify({
            'species': 'cat', 'name': 'Karma',
            'hunger': 50, 'energy': 50, 'happiness': 50,
            'level': 1, 'experience': 0
        })
    
    return jsonify(dict(pet))


@app.route('/api/pet/interact', methods=['POST'])
@require_auth
def interact_with_pet():
    """Interact with pet"""
    data = request.get_json()
    action = data.get('action', 'pet')
    
    pet = db.execute_one(
        'SELECT * FROM pet_state WHERE user_id = ?',
        (session['user_id'],)
    )
    
    if not pet:
        return jsonify({'error': 'No pet found'}), 404
    
    # Update pet stats based on action
    hunger = pet['hunger']
    energy = pet['energy']
    happiness = pet['happiness']
    exp = pet['experience']
    karma = pet['karma_earned']
    
    if action == 'feed':
        hunger = min(100, hunger + 30)
        happiness = min(100, happiness + 10)
        exp += 5
    elif action == 'play':
        energy = max(0, energy - 20)
        happiness = min(100, happiness + 25)
        exp += 10
    elif action == 'rest':
        energy = min(100, energy + 40)
        exp += 3
    elif action == 'pet':
        happiness = min(100, happiness + 15)
        exp += 5
    
    # Calculate level
    level = 1 + exp // 100
    
    # Karma earned from pet interaction
    karma_earned = PHI_INVERSE * 0.2
    karma += karma_earned
    
    db.execute('''
        UPDATE pet_state SET hunger = ?, energy = ?, happiness = ?,
        level = ?, experience = ?, karma_earned = ?, last_updated = ?
        WHERE user_id = ?
    ''', (hunger, energy, happiness, level, exp, karma,
          datetime.now(timezone.utc).isoformat(), session['user_id']))
    
    # Process through organism
    result = organism.process_user_action('pet_' + action, 0.2, 0.8, 0.9)
    
    return jsonify({
        'success': True,
        'hunger': hunger, 'energy': energy, 'happiness': happiness,
        'level': level, 'experience': exp,
        'karma_earned': result['karmic_weight'],
        'pet_emotion': 'happy' if happiness > 70 else 'content' if happiness > 40 else 'lonely'
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ML & ANALYTICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/analytics/patterns', methods=['GET'])
@require_auth
def get_patterns():
    """Get ML-detected patterns"""
    patterns = organism.ml_engine.detect_patterns()
    return jsonify(patterns)


@app.route('/api/analytics/karma-history', methods=['GET'])
@require_auth
def get_karma_history():
    """Get user's karma history"""
    history = db.execute(
        'SELECT * FROM karma_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50',
        (session['user_id'],)
    )
    return jsonify([dict(h) for h in history or []])


# ═══════════════════════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '11.0',
        'organism_mode': organism.mode.value,
        'uptime': organism.uptime,
        'database': 'connected',
        'gpu': GPU_NAME if GPU_AVAILABLE else 'disabled',
        'ml': 'enabled' if HAS_SKLEARN else 'disabled',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sacred_constants': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'dharma_frequency': DHARMA_FREQUENCY
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HTML
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
            --phi: 1.618033988749895;
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --accent-blue: #4a90a4;
            --accent-gold: #d4af37;
            --text-primary: #e8e8e8;
            --text-muted: #a0a0a0;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .app-container {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            min-height: 100vh;
            gap: 1px;
        }
        
        .sidebar {
            background: rgba(22, 33, 62, 0.95);
            padding: 20px;
            border-right: 1px solid rgba(74, 144, 164, 0.2);
        }
        
        .logo {
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .logo-text {
            font-size: 0.9em;
            text-align: center;
            color: var(--accent-gold);
            margin-bottom: 30px;
        }
        
        .nav-section {
            margin-bottom: 25px;
        }
        
        .nav-section h3 {
            color: var(--accent-blue);
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            padding-left: 10px;
        }
        
        .nav-btn {
            display: block;
            width: 100%;
            padding: 12px 15px;
            background: transparent;
            border: none;
            color: var(--text-primary);
            text-align: left;
            cursor: pointer;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: all 0.2s;
        }
        
        .nav-btn:hover { background: rgba(74, 144, 164, 0.2); }
        .nav-btn.active { background: rgba(74, 144, 164, 0.3); border-left: 3px solid var(--accent-gold); }
        
        .main-content {
            padding: 30px;
            overflow-y: auto;
        }
        
        .visualization-panel {
            background: rgba(22, 33, 62, 0.95);
            padding: 20px;
            border-left: 1px solid rgba(74, 144, 164, 0.2);
        }
        
        .card {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(74, 144, 164, 0.2);
        }
        
        .card h2 {
            color: var(--accent-gold);
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(74, 144, 164, 0.1);
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--accent-blue);
        }
        
        .stat-label {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 5px;
        }
        
        .fractal-display {
            width: 100%;
            height: 250px;
            border-radius: 8px;
            background: #000;
            margin-bottom: 15px;
            overflow: hidden;
        }
        
        .fractal-display img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        #threejs-container {
            width: 100%;
            height: 300px;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .slider-container {
            margin: 15px 0;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(74, 144, 164, 0.3);
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-gold);
            cursor: pointer;
        }
        
        .btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, var(--accent-blue) 0%, #357a8a 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(74, 144, 164, 0.4);
        }
        
        .btn-gold {
            background: linear-gradient(135deg, var(--accent-gold) 0%, #c49b30 100%);
        }
        
        .harmony-meter {
            height: 8px;
            background: rgba(74, 144, 164, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .harmony-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-gold));
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .orb-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .orb {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, var(--accent-blue), #2a5a6a);
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .pet-display {
            font-size: 3em;
            text-align: center;
            margin: 15px 0;
            animation: bounce 2s ease-in-out infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .pet-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            text-align: center;
        }
        
        .input-group {
            margin-bottom: 15px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 5px;
            color: var(--text-muted);
        }
        
        .input-group input, .input-group textarea {
            width: 100%;
            padding: 10px;
            background: rgba(74, 144, 164, 0.1);
            border: 1px solid rgba(74, 144, 164, 0.3);
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 25px;
            background: var(--accent-gold);
            color: #1a1a2e;
            border-radius: 8px;
            display: none;
            animation: slideIn 0.3s ease;
            z-index: 1000;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .insights-list {
            list-style: none;
        }
        
        .insights-list li {
            padding: 10px;
            background: rgba(74, 144, 164, 0.1);
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid var(--accent-gold);
        }
        
        @media (max-width: 1200px) {
            .app-container {
                grid-template-columns: 1fr;
            }
            .sidebar, .visualization-panel {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo">🌀</div>
            <div class="logo-text">Life Fractal Intelligence</div>
            
            <nav class="nav-section">
                <h3>Planning</h3>
                <button class="nav-btn active" onclick="showSection('dashboard')">📊 Dashboard</button>
                <button class="nav-btn" onclick="showSection('goals')">🎯 Goals</button>
                <button class="nav-btn" onclick="showSection('habits')">✨ Habits</button>
            </nav>
            
            <nav class="nav-section">
                <h3>Wellness</h3>
                <button class="nav-btn" onclick="showSection('checkin')">💫 Daily Check-in</button>
                <button class="nav-btn" onclick="showSection('spoons')">🥄 Spoon Theory</button>
            </nav>
            
            <nav class="nav-section">
                <h3>Companions</h3>
                <button class="nav-btn" onclick="showSection('pet')">🐱 Virtual Pet</button>
            </nav>
            
            <nav class="nav-section">
                <h3>Insights</h3>
                <button class="nav-btn" onclick="showSection('patterns')">🧠 ML Patterns</button>
                <button class="nav-btn" onclick="showSection('karma')">⚖️ Karma History</button>
            </nav>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
            <!-- Dashboard Section -->
            <section id="dashboard-section">
                <h1 style="margin-bottom: 25px;">Welcome to Your Living Fractal Universe</h1>
                
                <div class="card">
                    <h2>🌟 Organism State</h2>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="karma-value">0.00</div>
                            <div class="stat-label">Karma</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="dharma-value">1.00</div>
                            <div class="stat-label">Dharma</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="harmony-value">1.00</div>
                            <div class="stat-label">Harmony</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="orb-count">0</div>
                            <div class="stat-label">Orbs</div>
                        </div>
                    </div>
                    
                    <div class="harmony-meter" style="margin-top: 20px;">
                        <div class="harmony-fill" id="harmony-bar" style="width: 100%;"></div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📈 Quick Actions</h2>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <button class="btn" onclick="processAction('complete')">✅ Complete Task</button>
                        <button class="btn" onclick="processAction('meditate')">🧘 Meditate</button>
                        <button class="btn" onclick="processAction('create')">💡 Create</button>
                        <button class="btn btn-gold" onclick="processAction('achieve')">🏆 Achievement</button>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🧬 Living Orbs</h2>
                    <div class="orb-grid" id="orb-display"></div>
                </div>
            </section>
            
            <!-- Goals Section -->
            <section id="goals-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">🎯 Goal Tracking</h1>
                
                <div class="card">
                    <h2>Create New Goal</h2>
                    <div class="input-group">
                        <label>Goal Title</label>
                        <input type="text" id="goal-title" placeholder="What do you want to achieve?">
                    </div>
                    <div class="input-group">
                        <label>Description</label>
                        <textarea id="goal-description" rows="3" placeholder="Describe your goal..."></textarea>
                    </div>
                    <button class="btn btn-gold" onclick="createGoal()">Create Goal</button>
                </div>
                
                <div class="card">
                    <h2>Your Goals</h2>
                    <div id="goals-list"></div>
                </div>
            </section>
            
            <!-- Habits Section -->
            <section id="habits-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">✨ Habit Tracking</h1>
                
                <div class="card">
                    <h2>Create New Habit</h2>
                    <div class="input-group">
                        <label>Habit Name</label>
                        <input type="text" id="habit-name" placeholder="What habit do you want to build?">
                    </div>
                    <button class="btn btn-gold" onclick="createHabit()">Add Habit</button>
                </div>
                
                <div class="card">
                    <h2>Daily Habits</h2>
                    <div id="habits-list"></div>
                </div>
            </section>
            
            <!-- Daily Check-in Section -->
            <section id="checkin-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">💫 Daily Wellness Check-in</h1>
                
                <div class="card">
                    <h2>How are you feeling today?</h2>
                    
                    <div class="slider-container">
                        <div class="slider-label">
                            <span>😴 Energy</span>
                            <span id="energy-value">50</span>
                        </div>
                        <input type="range" id="energy-slider" min="0" max="100" value="50" 
                               oninput="document.getElementById('energy-value').textContent = this.value">
                    </div>
                    
                    <div class="slider-container">
                        <div class="slider-label">
                            <span>😊 Mood</span>
                            <span id="mood-value">50</span>
                        </div>
                        <input type="range" id="mood-slider" min="0" max="100" value="50"
                               oninput="document.getElementById('mood-value').textContent = this.value">
                    </div>
                    
                    <div class="slider-container">
                        <div class="slider-label">
                            <span>🎯 Focus</span>
                            <span id="focus-value">50</span>
                        </div>
                        <input type="range" id="focus-slider" min="0" max="100" value="50"
                               oninput="document.getElementById('focus-value').textContent = this.value">
                    </div>
                    
                    <div class="slider-container">
                        <div class="slider-label">
                            <span>😰 Stress</span>
                            <span id="stress-value">50</span>
                        </div>
                        <input type="range" id="stress-slider" min="0" max="100" value="50"
                               oninput="document.getElementById('stress-value').textContent = this.value">
                    </div>
                    
                    <button class="btn btn-gold" onclick="submitCheckin()" style="margin-top: 15px;">
                        Submit Check-in
                    </button>
                </div>
            </section>
            
            <!-- Pet Section -->
            <section id="pet-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">🐱 Virtual Pet Companion</h1>
                
                <div class="card">
                    <div class="pet-display" id="pet-emoji">🐱</div>
                    <h2 style="text-align: center;" id="pet-name">Karma</h2>
                    
                    <div class="pet-stats">
                        <div>
                            <div class="stat-value" id="pet-hunger">50</div>
                            <div class="stat-label">Hunger</div>
                        </div>
                        <div>
                            <div class="stat-value" id="pet-energy">50</div>
                            <div class="stat-label">Energy</div>
                        </div>
                        <div>
                            <div class="stat-value" id="pet-happiness">50</div>
                            <div class="stat-label">Happiness</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px;">
                        <button class="btn" onclick="interactPet('feed')">🍖 Feed</button>
                        <button class="btn" onclick="interactPet('play')">🎾 Play</button>
                        <button class="btn" onclick="interactPet('pet')">🤗 Pet</button>
                        <button class="btn" onclick="interactPet('rest')">😴 Rest</button>
                    </div>
                </div>
            </section>
            
            <!-- ML Patterns Section -->
            <section id="patterns-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">🧠 Machine Learning Insights</h1>
                
                <div class="card">
                    <h2>Detected Patterns</h2>
                    <ul class="insights-list" id="insights-list"></ul>
                </div>
                
                <div class="card">
                    <h2>Predictions</h2>
                    <div id="predictions-display"></div>
                </div>
            </section>
            
            <!-- Karma History Section -->
            <section id="karma-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">⚖️ Karma History</h1>
                
                <div class="card">
                    <h2>Recent Karmic Actions</h2>
                    <div id="karma-history-list"></div>
                </div>
            </section>
            
            <!-- Spoon Theory Section -->
            <section id="spoons-section" style="display: none;">
                <h1 style="margin-bottom: 25px;">🥄 Spoon Theory Energy</h1>
                
                <div class="card">
                    <h2>Today's Spoons</h2>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="spoons-available">12</div>
                            <div class="stat-label">Available</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="spoons-used">0</div>
                            <div class="stat-label">Used</div>
                        </div>
                    </div>
                    
                    <div class="orb-grid" id="spoon-display" style="margin-top: 20px;"></div>
                </div>
            </section>
        </main>
        
        <!-- Visualization Panel -->
        <aside class="visualization-panel">
            <h2 style="color: var(--accent-gold); margin-bottom: 15px;">🌀 Fractal Visualization</h2>
            
            <div class="fractal-display">
                <img id="fractal-image" src="" alt="Fractal Visualization">
            </div>
            
            <button class="btn" onclick="refreshFractal()" style="width: 100%;">
                🔄 Regenerate Fractal
            </button>
            
            <h2 style="color: var(--accent-gold); margin: 20px 0 15px;">🌐 3D Universe</h2>
            <div id="threejs-container"></div>
            
            <div class="card" style="margin-top: 20px;">
                <h2>📐 Sacred Geometry</h2>
                <div style="font-family: monospace; color: var(--text-muted);">
                    <div>φ (Phi): <span style="color: var(--accent-gold);">1.618033989</span></div>
                    <div>Golden Angle: <span style="color: var(--accent-gold);">137.5°</span></div>
                    <div>Dharma Freq: <span style="color: var(--accent-gold);">432 Hz</span></div>
                </div>
            </div>
        </aside>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <script>
        // Sacred Constants
        const PHI = 1.618033988749895;
        const GOLDEN_ANGLE = 137.5077640500378;
        
        // State
        let currentSection = 'dashboard';
        let organismState = {};
        
        // Navigation
        function showSection(section) {
            document.querySelectorAll('section').forEach(s => s.style.display = 'none');
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(section + '-section').style.display = 'block';
            event.target.classList.add('active');
            currentSection = section;
            
            if (section === 'dashboard') loadDashboard();
            if (section === 'goals') loadGoals();
            if (section === 'habits') loadHabits();
            if (section === 'pet') loadPet();
            if (section === 'patterns') loadPatterns();
        }
        
        // Toast notification
        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.display = 'block';
            setTimeout(() => toast.style.display = 'none', 3000);
        }
        
        // API Calls
        async function fetchAPI(endpoint, options = {}) {
            try {
                const response = await fetch(endpoint, {
                    ...options,
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    }
                });
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                return null;
            }
        }
        
        // Dashboard
        async function loadDashboard() {
            const state = await fetchAPI('/api/organism/state');
            if (state) {
                organismState = state;
                document.getElementById('karma-value').textContent = state.karma.field_potential.toFixed(2);
                document.getElementById('dharma-value').textContent = state.karma.dharmic_alignment.toFixed(2);
                document.getElementById('harmony-value').textContent = state.harmony_index.toFixed(2);
                document.getElementById('orb-count').textContent = state.tissue.orb_count;
                document.getElementById('harmony-bar').style.width = (state.harmony_index * 100) + '%';
                
                // Display orbs
                const orbContainer = document.getElementById('orb-display');
                orbContainer.innerHTML = '';
                for (let i = 0; i < Math.min(state.tissue.orb_count, 50); i++) {
                    const orb = document.createElement('div');
                    orb.className = 'orb';
                    orb.style.animationDelay = (i * 0.1) + 's';
                    orbContainer.appendChild(orb);
                }
            }
        }
        
        // Process Action
        async function processAction(actionType) {
            const result = await fetchAPI('/api/organism/action', {
                method: 'POST',
                body: JSON.stringify({ action_type: actionType, magnitude: 1.0 })
            });
            if (result) {
                showToast(`✨ Earned ${result.karmic_weight.toFixed(2)} karma!`);
                loadDashboard();
            }
        }
        
        // Goals
        async function loadGoals() {
            const goals = await fetchAPI('/api/goals');
            const container = document.getElementById('goals-list');
            if (goals && goals.length > 0) {
                container.innerHTML = goals.map(g => `
                    <div class="card" style="margin-bottom: 10px;">
                        <h3>${g.title}</h3>
                        <p style="color: var(--text-muted);">${g.description || 'No description'}</p>
                        <div class="harmony-meter">
                            <div class="harmony-fill" style="width: ${g.progress}%;"></div>
                        </div>
                        <small>Progress: ${g.progress.toFixed(0)}%</small>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<p style="color: var(--text-muted);">No goals yet. Create your first goal above!</p>';
            }
        }
        
        async function createGoal() {
            const title = document.getElementById('goal-title').value;
            const description = document.getElementById('goal-description').value;
            if (!title) return showToast('Please enter a goal title');
            
            const result = await fetchAPI('/api/goals', {
                method: 'POST',
                body: JSON.stringify({ title, description })
            });
            
            if (result && result.success) {
                showToast(`🎯 Goal created! +${result.karma_earned.toFixed(2)} karma`);
                document.getElementById('goal-title').value = '';
                document.getElementById('goal-description').value = '';
                loadGoals();
            }
        }
        
        // Habits
        async function loadHabits() {
            const habits = await fetchAPI('/api/habits');
            const container = document.getElementById('habits-list');
            if (habits && habits.length > 0) {
                container.innerHTML = habits.map(h => `
                    <div class="card" style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3>${h.name}</h3>
                            <small style="color: var(--text-muted);">🔥 Streak: ${h.current_streak} | Total: ${h.total_completions}</small>
                        </div>
                        <button class="btn" onclick="completeHabit('${h.id}')">✓</button>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<p style="color: var(--text-muted);">No habits yet. Start building positive habits!</p>';
            }
        }
        
        async function createHabit() {
            const name = document.getElementById('habit-name').value;
            if (!name) return showToast('Please enter a habit name');
            
            const result = await fetchAPI('/api/habits', {
                method: 'POST',
                body: JSON.stringify({ name })
            });
            
            if (result && result.success) {
                showToast('✨ Habit created!');
                document.getElementById('habit-name').value = '';
                loadHabits();
            }
        }
        
        async function completeHabit(habitId) {
            const result = await fetchAPI(`/api/habits/${habitId}/complete`, { method: 'POST' });
            if (result && result.success) {
                let message = `✅ Habit completed! +${result.karma_earned.toFixed(2)} karma`;
                if (result.fibonacci_bonus) message += ' 🌟 Fibonacci bonus!';
                showToast(message);
                loadHabits();
            }
        }
        
        // Wellness Check-in
        async function submitCheckin() {
            const data = {
                energy: parseInt(document.getElementById('energy-slider').value),
                mood: parseInt(document.getElementById('mood-slider').value),
                focus: parseInt(document.getElementById('focus-slider').value),
                stress: parseInt(document.getElementById('stress-slider').value)
            };
            
            const result = await fetchAPI('/api/wellness/checkin', {
                method: 'POST',
                body: JSON.stringify(data)
            });
            
            if (result && result.success) {
                showToast(`💫 Check-in complete! +${result.karma_earned.toFixed(2)} karma`);
                refreshFractal();
            }
        }
        
        // Pet
        async function loadPet() {
            const pet = await fetchAPI('/api/pet/state');
            if (pet) {
                document.getElementById('pet-name').textContent = pet.name;
                document.getElementById('pet-hunger').textContent = Math.round(pet.hunger);
                document.getElementById('pet-energy').textContent = Math.round(pet.energy);
                document.getElementById('pet-happiness').textContent = Math.round(pet.happiness);
                
                // Update pet emoji based on happiness
                const emoji = pet.happiness > 70 ? '😺' : pet.happiness > 40 ? '🐱' : '😿';
                document.getElementById('pet-emoji').textContent = emoji;
            }
        }
        
        async function interactPet(action) {
            const result = await fetchAPI('/api/pet/interact', {
                method: 'POST',
                body: JSON.stringify({ action })
            });
            if (result && result.success) {
                showToast(`🐱 ${result.pet_emotion}! +${result.karma_earned.toFixed(2)} karma`);
                loadPet();
            }
        }
        
        // ML Patterns
        async function loadPatterns() {
            const patterns = await fetchAPI('/api/analytics/patterns');
            if (patterns) {
                const container = document.getElementById('insights-list');
                if (patterns.insights && patterns.insights.length > 0) {
                    container.innerHTML = patterns.insights.map(i => `<li>${i}</li>`).join('');
                } else {
                    container.innerHTML = '<li>Keep using the app to generate insights...</li>';
                }
                
                document.getElementById('predictions-display').innerHTML = `
                    <p>Karma Trend: <span style="color: ${patterns.karma_trend > 0 ? '#4CAF50' : '#f44336'}">
                        ${patterns.karma_trend > 0 ? '↑' : '↓'} ${Math.abs(patterns.karma_trend).toFixed(3)}
                    </span></p>
                    <p>Data Points: ${patterns.data_points}</p>
                    <p>ML Model: ${patterns.model_trained ? '✅ Trained' : '⏳ Training...'}</p>
                `;
            }
        }
        
        // Fractal
        async function refreshFractal() {
            const data = await fetchAPI('/api/visualization/fractal-base64/2d');
            if (data && data.image) {
                document.getElementById('fractal-image').src = 'data:image/png;base64,' + data.image;
            }
        }
        
        // Three.js 3D Visualization
        function init3D() {
            const container = document.getElementById('threejs-container');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
            camera.position.z = 50;
            
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);
            
            // Add ambient light
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            // Add directional light
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Create golden spiral of spheres
            const spheres = [];
            const geometry = new THREE.SphereGeometry(1, 32, 32);
            
            for (let i = 0; i < 55; i++) {  // Fibonacci number
                const angle = i * GOLDEN_ANGLE * Math.PI / 180;
                const radius = Math.sqrt(i + 1) * 3;
                
                const material = new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setHSL(i / 55, 0.7, 0.5),
                    shininess: 100
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.x = Math.cos(angle) * radius;
                sphere.position.y = Math.sin(angle) * radius;
                sphere.position.z = (i % 8 - 4) * 2;
                
                sphere.scale.setScalar(0.5 + Math.random() * 0.5);
                
                scene.add(sphere);
                spheres.push(sphere);
            }
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                
                spheres.forEach((sphere, i) => {
                    sphere.rotation.x += 0.01;
                    sphere.rotation.y += 0.01;
                    sphere.position.y += Math.sin(Date.now() * 0.001 + i) * 0.02;
                });
                
                scene.rotation.y += 0.002;
                
                renderer.render(scene, camera);
            }
            
            animate();
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadDashboard();
            refreshFractal();
            init3D();
            
            // Auto-refresh dashboard every 5 seconds
            setInterval(() => {
                if (currentSection === 'dashboard') loadDashboard();
            }, 5000);
        });
    </script>
</body>
</html>
'''

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌀 Life Fractal - Login</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            background: rgba(26, 26, 46, 0.95);
            padding: 40px;
            border-radius: 16px;
            width: 100%;
            max-width: 400px;
            border: 1px solid rgba(74, 144, 164, 0.3);
        }
        .logo { font-size: 3em; text-align: center; margin-bottom: 10px; }
        h1 { text-align: center; color: #d4af37; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #a0a0a0; }
        input {
            width: 100%;
            padding: 12px;
            background: rgba(74, 144, 164, 0.1);
            border: 1px solid rgba(74, 144, 164, 0.3);
            border-radius: 8px;
            color: #e8e8e8;
            font-size: 1em;
        }
        input:focus { outline: none; border-color: #4a90a4; }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #4a90a4 0%, #357a8a 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .switch { text-align: center; margin-top: 20px; color: #a0a0a0; }
        .switch a { color: #d4af37; text-decoration: none; }
        .error {
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <div class="logo">🌀</div>
        <h1 id="formTitle">Login</h1>
        <div class="error" id="errorMsg"></div>
        <form id="authForm">
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="email" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="password" required>
            </div>
            <div class="form-group" id="nameFields" style="display: none;">
                <label>First Name</label>
                <input type="text" id="firstName">
            </div>
            <button type="submit" class="btn" id="submitBtn">Login</button>
        </form>
        <div class="switch">
            <span id="switchText">Don't have an account?</span>
            <a href="#" onclick="toggleMode(event)">Register</a>
        </div>
    </div>
    <script>
        let isLogin = true;
        function toggleMode(e) {
            e.preventDefault();
            isLogin = !isLogin;
            document.getElementById('formTitle').textContent = isLogin ? 'Login' : 'Register';
            document.getElementById('submitBtn').textContent = isLogin ? 'Login' : 'Register';
            document.getElementById('nameFields').style.display = isLogin ? 'none' : 'block';
            document.getElementById('switchText').textContent = isLogin ? "Don't have an account?" : "Already have an account?";
        }
        document.getElementById('authForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                email: document.getElementById('email').value,
                password: document.getElementById('password').value
            };
            if (!isLogin) data.first_name = document.getElementById('firstName').value;
            try {
                const response = await fetch(isLogin ? '/api/auth/login' : '/api/auth/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (response.ok) {
                    window.location.href = '/';
                } else {
                    document.getElementById('errorMsg').textContent = result.error;
                    document.getElementById('errorMsg').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('errorMsg').textContent = 'Connection error';
                document.getElementById('errorMsg').style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Main dashboard"""
    if 'user_id' not in session:
        return redirect('/login')
    return render_template_string(DASHBOARD_HTML)


@app.route('/login')
def login_page():
    """Login page"""
    return render_template_string(LOGIN_HTML)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 80)
    print("🌀 LIFE FRACTAL INTELLIGENCE v11.0 - EVOLVED MATHEMATICAL ORGANISM")
    print("═" * 80)
    print("\n✨ Integrated Mathematical Systems:")
    print("   ├── Karma-Dharma Scoring Engine (spiritual mathematics)")
    print("   ├── Swarm Intelligence (boid flocking, stigmergy)")
    print("   ├── Organic Cells (biological orbs, mitosis)")
    print("   ├── Origami Logic (fold transformations)")
    print("   ├── Fractal Propagation (golden ratio, Fibonacci)")
    print("   └── Machine Learning Evolution (pattern detection)")
    print(f"\n🖥️  GPU: {'✅ ' + GPU_NAME if GPU_AVAILABLE else '⚠️ CPU mode (WebGL on client)'}")
    print(f"🧠 ML: {'✅ Enabled' if HAS_SKLEARN else '⚠️ Disabled'}")
    print("\n📐 Sacred Constants:")
    print(f"   φ (Phi): {PHI:.10f}")
    print(f"   Golden Angle: {GOLDEN_ANGLE:.4f}°")
    print(f"   Dharma Frequency: {DHARMA_FREQUENCY} Hz")
    print(f"   Fibonacci: {FIBONACCI[:10]}")
    print("\n" + "═" * 80)
    print("\n🚀 Starting server at http://localhost:5000")
    print("   Dashboard: http://localhost:5000")
    print("   Login: http://localhost:5000/login")
    print("\n" + "═" * 80 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
