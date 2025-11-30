"""
life_planning_unified.py
========================

This module combines several mathematical and behavioural engines into a
cohesive system that can be integrated into a life‑planning application.  The
goal of this system is to provide predictive insights, gentle guidance and
engaging feedback through a virtual pet and abstract artwork.  The design is
informed by classical mathematics (e.g. the golden ratio and Fibonacci
sequence), chaos theory (via the logistic map), modern predictive modelling
(decision trees and linear models), fuzzy logic and cognitive behavioural
techniques.  It also includes a hybrid multi‑fractal art generator that
translates behavioural data into intricate images.

The code is written to run on Windows PowerShell with ASCII‑only source and
relies only on common Python packages (numpy, scikit‑learn and Pillow).  All
classes include error handling and logging for robustness.  Comments are
extensive to aid understanding for users who may be new to these topics or
neurodivergent learners.

Usage:
    from life_planning_unified import LifePlanningSystem

    # construct system
    system = LifePlanningSystem(species="cat")

    # update with user inputs and generate guidance
    user_data = {
        "stress": 40,
        "mood": 70,
        "goals_completed": 3,
        "sleep_hours": 7,
        "nutrition_score": 60,
        "period": "daily"
    }
    system.update(user_data)
    guidance = system.generate_guidance()
    print(guidance["message"])
    # generate artwork
    image = system.generate_fractal_image()
    image.save("my_fractal.png")

The code intentionally avoids using any GUI frameworks.  It produces PIL
Image objects that can be displayed in your existing interface.  The
LifePlanningSystem class acts as a façade over the individual engines.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeRegressor

# -----------------------------------------------------------------------------
# Logging configuration
#
# Create a basic logger that writes to a file.  In a larger application the
# logging configuration could be customised further or integrated with your
# application's logging infrastructure.
logging.basicConfig(
    filename="pet_addon.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Ancient mathematics utilities
#
# These functions expose simple classical sequences and constants that are
# historically significant and can be used as inputs or modifiers within
# predictive models and artwork.  For example the golden ratio has been
# associated with aesthetically pleasing proportions and also appears in
# the limiting ratio of successive Fibonacci numbers.  The logistic map is a
# one‑dimensional discrete dynamical system which exhibits everything from
# stable equilibrium to chaos depending on a parameter.  See
# https://www.britannica.com/science/golden-ratio for the history of the
# golden ratio and its relation to Fibonacci numbers and fractals【898758732779967†L243-L261】.

class AncientMathUtil:
    """Utility class for classical mathematical sequences and maps."""

    @staticmethod
    def golden_ratio() -> float:
        """Return the value of the golden ratio (approximately 1.618)."""
        return (1.0 + math.sqrt(5.0)) / 2.0

    @staticmethod
    def fibonacci_sequence(n: int) -> List[int]:
        """
        Generate the first n Fibonacci numbers.

        Args:
            n: The number of terms to generate.
        Returns:
            A list of the first n Fibonacci numbers.
        """
        seq = [0, 1]
        for _ in range(max(0, n - 2)):
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

    @staticmethod
    def logistic_map_series(r: float, x0: float, n: int) -> List[float]:
        """
        Generate a series using the logistic map x_{n+1} = r * x_n * (1 - x_n).

        The logistic map is a simple nonlinear recurrence relation that can
        produce chaotic behaviour for certain values of r【315287541316333†L190-L203】.  It has
        applications in population dynamics and chaos theory.  Here it is used
        as a source of pseudo‑randomness and to inject natural variation into
        the artwork and pet behaviour.

        Args:
            r: The growth parameter (0 < r <= 4 yields bounded sequences).
            x0: The initial value of x (usually between 0 and 1).
            n: The number of terms to generate.
        Returns:
            A list of n values of the logistic map sequence.
        """
        series = []
        x = x0
        for _ in range(n):
            series.append(x)
            x = r * x * (1.0 - x)
        return series


# -----------------------------------------------------------------------------
# Decision tree predictor
#
# Decision trees are a classic supervised learning technique.  They are simple
# to understand and interpret, handle both numerical and categorical data, and
# require little data preparation【344428563639336†L834-L870】.  However, they are prone to
# overfitting and small changes in data can lead to very different trees【344428563639336†L874-L904】.
# This implementation uses scikit‑learn's DecisionTreeRegressor to predict the
# user's next mood given recent behavioural inputs.  In a real application you
# could extend this to multiple targets (stress, energy, productivity, etc.).

class DecisionTreePredictor:
    """Predict future mood using a decision tree regressor."""

    def __init__(self) -> None:
        self.model = DecisionTreeRegressor(random_state=42)
        self.trained = False

    def train(self, history: List[Dict[str, float]]) -> None:
        """
        Train the decision tree model on historical data.

        The history list should contain dictionaries with keys: 'stress',
        'mood', 'energy', 'goals_completed', 'sleep_hours' and 'mood_next'.

        Args:
            history: A list of past observations.
        """
        try:
            if not history:
                logger.warning("No history supplied to DecisionTreePredictor.train")
                return
            # Prepare feature matrix X and target vector y
            X: List[List[float]] = []
            y: List[float] = []
            for record in history:
                try:
                    features = [
                        float(record.get("stress", 50.0)),
                        float(record.get("mood", 50.0)),
                        float(record.get("energy", 50.0)),
                        float(record.get("goals_completed", 0.0)),
                        float(record.get("sleep_hours", 0.0)),
                    ]
                    target = float(record.get("mood_next", record.get("mood", 50.0)))
                    X.append(features)
                    y.append(target)
                except (TypeError, ValueError) as e:
                    logger.error("Invalid record in history: %s", e)
            if not X:
                logger.warning("No valid records to train on")
                return
            self.model.fit(X, y)
            self.trained = True
            logger.info("DecisionTreePredictor trained on %d samples", len(X))
        except Exception as e:
            logger.error("DecisionTreePredictor training failed: %s", e)

    def predict_next_mood(self, features: Dict[str, float]) -> float:
        """
        Predict the next mood value given current features.

        Args:
            features: A dictionary with keys 'stress', 'mood', 'energy',
                'goals_completed', and 'sleep_hours'.
        Returns:
            A predicted mood score (0–100).  If the model is untrained,
            returns the current mood.
        """
        try:
            if not self.trained:
                # Return current mood as baseline when untrained
                return float(features.get("mood", 50.0))
            X_test = [[
                float(features.get("stress", 50.0)),
                float(features.get("mood", 50.0)),
                float(features.get("energy", 50.0)),
                float(features.get("goals_completed", 0.0)),
                float(features.get("sleep_hours", 0.0)),
            ]]
            prediction = float(self.model.predict(X_test)[0])
            # Clamp prediction to valid range
            return max(0.0, min(100.0, prediction))
        except Exception as e:
            logger.error("DecisionTreePredictor prediction failed: %s", e)
            return float(features.get("mood", 50.0))


# -----------------------------------------------------------------------------
# Fuzzy logic engine
#
# Fuzzy logic allows reasoning with degrees of truth rather than binary true/false.
# It uses membership functions to express how strongly an input belongs to a
# linguistic category and IF–THEN rules to map inputs to outputs.  This mirrors
# human decision making and is effective for handling uncertainty【42040320637325†L85-L129】.  The
# architecture typically consists of fuzzification, rule evaluation, and
# defuzzification【601702499099764†L133-L148】.  Here we implement a simple fuzzy system
# that accepts stress and mood values (0–100) and outputs a support message.

class FuzzyLogicEngine:
    """
    Simple fuzzy logic system for generating supportive messages.

    This engine defines membership functions for the linguistic terms LOW,
    MEDIUM and HIGH on a 0–100 scale and a small rule base mapping stress and
    mood to qualitative advice.  It then performs defuzzification by
    selecting the rule with the highest degree of activation.  The design is
    intentionally basic but illustrates how fuzzy rules can be used for
    personalized guidance.
    """

    def __init__(self) -> None:
        # Define triangular membership functions as (a, b, c) tuples
        # where b is the peak membership point.  The functions return
        # membership values between 0 and 1.
        self.stress_mf = {
            "low": (0.0, 0.0, 40.0),
            "medium": (20.0, 50.0, 80.0),
            "high": (60.0, 100.0, 100.0),
        }
        self.mood_mf = {
            "low": (0.0, 0.0, 40.0),
            "medium": (20.0, 50.0, 80.0),
            "high": (60.0, 100.0, 100.0),
        }
        # Rule base: each rule is a tuple (stress_term, mood_term, message)
        self.rules = [
            ("high", "low", "It looks like you're overwhelmed. Try a grounding exercise."),
            ("medium", "low", "Take a short break and do something you enjoy."),
            ("high", "medium", "Prioritise rest; consider some deep breathing."),
            ("medium", "medium", "Stay balanced. Focus on one task at a time."),
            ("low", "low", "You're doing okay. Small steps forward help."),
            ("low", "high", "Great job! Keep up the positive momentum."),
            ("medium", "high", "Use this good mood to tackle a challenging task."),
            ("high", "high", "You're riding high! Celebrate your wins but watch your energy."),
        ]

    def _membership(self, x: float, mf: Tuple[float, float, float]) -> float:
        """
        Compute the membership of x in a triangular membership function.

        Args:
            x: The input value.
            mf: A tuple (a, b, c) defining the triangle.
        Returns:
            A membership degree between 0 and 1.
        """
        a, b, c = mf
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if (b - a) != 0 else 0.0
        else:
            return (c - x) / (c - b) if (c - b) != 0 else 0.0

    def _fuzzify(self, value: float, mfs: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Evaluate all membership functions for a given value.

        Args:
            value: The crisp input value.
            mfs: A dictionary of membership functions.
        Returns:
            A dictionary mapping each linguistic term to its membership degree.
        """
        return {term: self._membership(value, params) for term, params in mfs.items()}

    def infer(self, stress: float, mood: float) -> str:
        """
        Generate a support message based on stress and mood using fuzzy logic.

        Args:
            stress: A value between 0 and 100 representing stress level.
            mood: A value between 0 and 100 representing mood level.
        Returns:
            A textual message providing guidance.
        """
        # Fuzzify inputs
        stress_degrees = self._fuzzify(stress, self.stress_mf)
        mood_degrees = self._fuzzify(mood, self.mood_mf)

        # Evaluate rules and find the one with highest activation
        best_rule: Optional[Tuple[str, str, str]] = None
        best_activation = -1.0
        for s_term, m_term, message in self.rules:
            activation = min(stress_degrees.get(s_term, 0.0), mood_degrees.get(m_term, 0.0))
            if activation > best_activation:
                best_activation = activation
                best_rule = (s_term, m_term, message)

        if best_rule:
            logger.debug(
                "FuzzyLogicEngine selected rule (%s AND %s) with activation %.2f", best_rule[0], best_rule[1], best_activation
            )
            return best_rule[2]
        # Default message
        return "Unable to determine guidance. Please check your inputs."


# -----------------------------------------------------------------------------
# Fractal art engine
#
# The following classes generate abstract images whose structure and colour
# respond to behavioural data.  They are adapted from the earlier fractal
# generator with additional species‑specific modifiers.  Behavioural inputs are
# mapped to fractal parameters through FractalDataMapper.  Colour schemes are
# generated by PaletteGenerator.  MultiFractalGenerator builds the image by
# combining Mandelbrot, Julia and noise components, and SymmetryEngine applies
# radial symmetry to create mandala‑like patterns.  See the module
# documentation for background on these techniques.

class PaletteGenerator:
    """
    Generate colours based on a base palette and mood tilt.

    A palette is created by blending primary colour channels according to
    behavioural inputs.  Several base palettes are provided (fire, sky, soft,
    neon, balanced) inspired by common virtual pet species.  The mood_tilt
    parameter brightens or darkens the palette to reflect positive or
    negative mood.
    """

    def __init__(self, base_palette: str = "balanced", mood_tilt: float = 0.5) -> None:
        self.base_palette = base_palette
        self.mood_tilt = min(max(mood_tilt, 0.0), 1.0)

    def _clamp(self, v: float) -> int:
        return max(0, min(255, int(v)))

    def _fire(self, t: float) -> Tuple[int, int, int]:
        return (
            self._clamp(200 + 55 * t),
            self._clamp(50 + 100 * t),
            self._clamp(20 + 40 * (1.0 - t)),
        )

    def _sky(self, t: float) -> Tuple[int, int, int]:
        return (
            self._clamp(30 + 50 * t),
            self._clamp(100 + 100 * t),
            self._clamp(150 + 105 * t),
        )

    def _soft(self, t: float) -> Tuple[int, int, int]:
        return (
            self._clamp(180 + 50 * t),
            self._clamp(170 + 60 * t),
            self._clamp(190 + 40 * (1.0 - t)),
        )

    def _neon(self, t: float) -> Tuple[int, int, int]:
        return (
            self._clamp(50 + 205 * abs(math.sin(3.0 * t))),
            self._clamp(50 + 205 * abs(math.sin(2.0 * t))),
            self._clamp(50 + 205 * abs(math.sin(4.0 * t))),
        )

    def _balanced(self, t: float) -> Tuple[int, int, int]:
        return (
            self._clamp(60 + 150 * t),
            self._clamp(80 + 120 * (1.0 - t)),
            self._clamp(100 + 100 * math.sin(t * math.pi)),
        )

    def get_color(self, iteration: int, max_iterations: int) -> Tuple[int, int, int]:
        """
        Determine the colour for a given iteration value.

        Points that do not escape (i.e. inside the set) are rendered black to
        create a strong contrast with the fractal structure.  For points that
        escape, t is computed as the ratio of iteration to max_iterations and
        then adjusted by mood_tilt.  The chosen base palette is then used to
        produce a colour triple.
        """
        if iteration >= max_iterations:
            return (0, 0, 0)
        t = (iteration / float(max_iterations)) * (0.5 + 0.5 * self.mood_tilt)
        t = max(0.0, min(1.0, t))
        if self.base_palette == "fire":
            return self._fire(t)
        if self.base_palette == "sky":
            return self._sky(t)
        if self.base_palette == "soft":
            return self._soft(t)
        if self.base_palette == "neon":
            return self._neon(t)
        return self._balanced(t)


class MultiFractalGenerator:
    """
    Generate hybrid fractal images combining Mandelbrot, Julia and noise terms.

    This generator maps pixel coordinates to the complex plane, iterates them
    through two quadratic maps and adds a small random perturbation to
    introduce natural variation.  Behavioural parameters control the maximum
    number of iterations, the zoom level, the weightings of Mandelbrot vs
    Julia vs noise contributions and a centre shift that encodes pet
    behaviour.  Colour selection is delegated to the PaletteGenerator.  The
    algorithm is fully deterministic given the parameters and random seed.
    """

    def __init__(self, width: int = 512, height: int = 512) -> None:
        self.width = width
        self.height = height

    def _mandelbrot_escape(self, cx: float, cy: float, max_iter: int) -> int:
        c = complex(cx, cy)
        z = complex(0.0, 0.0)
        for i in range(max_iter):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 16.0:
                return i
        return max_iter

    def _julia_escape(self, x: float, y: float, c: complex, max_iter: int) -> int:
        z = complex(x, y)
        for i in range(max_iter):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 16.0:
                return i
        return max_iter

    def generate_image(self, params: Dict[str, float], palette: PaletteGenerator) -> Image.Image:
        """
        Build a PIL Image based on provided fractal parameters and palette.

        Args:
            params: A dictionary of parameters returned by FractalDataMapper.
            palette: An instance of PaletteGenerator to colour the image.
        Returns:
            A new PIL Image representing the fractal.
        """
        max_iterations = int(params.get("max_iterations", 200))
        zoom = float(params.get("zoom", 1.0))
        noise_strength = float(params.get("noise_strength", 0.0))
        mandelbrot_weight = float(params.get("mandelbrot_weight", 0.4))
        julia_weight = float(params.get("julia_weight", 0.4))
        noise_weight = float(params.get("noise_weight", 0.2))
        cx_shift, cy_shift = params.get("center_shift", (0.0, 0.0))
        symmetry_level = float(params.get("symmetry_level", 0.0))
        # Precompute Julia constant from centre shift
        julia_c = complex(cx_shift * 0.8, cy_shift * 0.8)

        img = Image.new("RGB", (self.width, self.height))
        pixels = img.load()

        random.seed(42)  # deterministic noise for reproducibility

        for px in range(self.width):
            for py in range(self.height):
                # Map pixel to complex plane with zoom and centre shift
                x = ((px - self.width / 2.0) / (self.width / 4.0)) / zoom + cx_shift
                y = ((py - self.height / 2.0) / (self.height / 4.0)) / zoom + cy_shift
                # Apply simple symmetry across vertical axis
                if symmetry_level > 0.5:
                    mix = (symmetry_level - 0.5) * 2.0
                    x = x * (1.0 - mix) + (-x) * mix
                # Compute contributions
                mandel_iter = self._mandelbrot_escape(x, y, max_iterations)
                julia_iter = self._julia_escape(x, y, julia_c, max_iterations)
                noise_iter = int(random.random() * max_iterations * noise_strength)
                combined = (
                    mandel_iter * mandelbrot_weight
                    + julia_iter * julia_weight
                    + noise_iter * noise_weight
                )
                combined_iter = int(combined)
                color = palette.get_color(combined_iter, max_iterations)
                pixels[px, py] = color
        return img


class SymmetryEngine:
    """
    Apply radial symmetry (kaleidoscope effect) to a PIL Image.

    Radial symmetry helps produce mandala‑like patterns which many users find
    calming and meditative.  The number of segments controls how many mirror
    copies around the circle are created.  A segment count of 8 produces an
    eight‑fold mandala.
    """

    def __init__(self, segments: int = 8) -> None:
        self.segments = max(1, segments)

    def apply(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        cx, cy = width // 2, height // 2
        out = img.copy()
        src = img.load()
        dst = out.load()
        for x in range(width):
            for y in range(height):
                dx = x - cx
                dy = y - cy
                r = math.hypot(dx, dy)
                theta = math.atan2(dy, dx) if r != 0 else 0.0
                segment_angle = 2.0 * math.pi / float(self.segments)
                folded_theta = theta % segment_angle
                sx = int(cx + r * math.cos(folded_theta))
                sy = int(cy + r * math.sin(folded_theta))
                if 0 <= sx < width and 0 <= sy < height:
                    dst[x, y] = src[sx, sy]
        return out


class FractalDataMapper:
    """
    Map behavioural and pet data into fractal parameters.

    This mapper scales user inputs (stress, mood, goals, sleep, nutrition) and
    pet state (growth, mood, stress, behaviour, species) into parameters used
    by the fractal generator.  It encodes intuitive relationships such as
    "more goals completed → deeper detail", "high stress → more noise" and
    species specific modifiers.  Species modifiers are derived from the
    multi‑species architecture: cat, blob creature, alien, frog, robot and
    dragon.  Each species shifts weights and symmetry according to its
    personality.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def map_data_to_params(self, user_data: Dict[str, float], pet_state: Dict[str, float]) -> Dict[str, float]:
        # Extract and clamp inputs
        stress = self._clamp(float(user_data.get("stress", 50.0)), 0.0, 100.0)
        mood = self._clamp(float(user_data.get("mood", 50.0)), 0.0, 100.0)
        goals = self._clamp(float(user_data.get("goals_completed", 0.0)), 0.0, 20.0)
        sleep = self._clamp(float(user_data.get("sleep_hours", 7.0)), 0.0, 24.0)
        nutrition = self._clamp(float(user_data.get("nutrition_score", 50.0)), 0.0, 100.0)

        pet_growth = self._clamp(float(pet_state.get("growth", 50.0)), 0.0, 100.0)
        pet_mood = self._clamp(float(pet_state.get("mood", 50.0)), 0.0, 100.0)
        pet_stress = self._clamp(float(pet_state.get("stress", 50.0)), 0.0, 100.0)
        pet_behavior = pet_state.get("behavior", "idle")
        species = pet_state.get("species", "cat")
        period = user_data.get("period", "daily")

        # Base iterations determined by goals and period
        base_iters = 200
        iter_boost = int(goals * 15)
        if period == "weekly":
            base_iters += 50
        elif period == "monthly":
            base_iters += 100
        elif period == "yearly":
            base_iters += 200
        max_iterations = base_iters + iter_boost

        # Zoom increases with pet growth (more progress yields deeper zoom)
        zoom = 1.0 + (pet_growth / 100.0) * 4.0

        # Noise strength increases with combined stress
        noise_strength = 0.0005 + (stress + pet_stress) / 100000.0

        # Choose a base palette based on species
        if species == "cat":
            base_palette = "balanced"
        elif species == "blob":
            base_palette = "soft"
        elif species == "alien":
            base_palette = "neon"
        elif species == "frog":
            base_palette = "sky"
        elif species == "robot":
            base_palette = "balanced"
        elif species == "dragon":
            base_palette = "fire"
        else:
            base_palette = "balanced"

        # Mood tilt influences brightness
        mood_tilt = (mood + pet_mood) / 200.0

        # Symmetry level increases with goals and sleep
        symmetry_level = self._clamp((goals / 20.0) + (sleep / 24.0), 0.1, 1.0)

        # Initial weights
        mandelbrot_weight = 0.4
        julia_weight = 0.4
        noise_weight = 0.2
        # Species specific modifiers (from previous plan)
        if species == "cat":
            mandelbrot_weight += 0.05
            julia_weight += 0.10
            noise_weight -= 0.05
            symmetry_level = min(1.0, symmetry_level + 0.15)
            mood_tilt = min(1.0, mood_tilt + 0.05)
        elif species == "blob":
            mandelbrot_weight += 0.05
            julia_weight -= 0.05
            noise_weight -= 0.10
            symmetry_level = min(1.0, symmetry_level + 0.20)
            zoom *= 0.9
        elif species == "alien":
            mandelbrot_weight -= 0.05
            julia_weight += 0.15
            noise_weight += 0.15
            mood_tilt = max(0.0, mood_tilt - 0.05)
        elif species == "frog":
            symmetry_level = min(1.0, symmetry_level + 0.30)
            mandelbrot_weight += 0.10
            noise_weight -= 0.05
            zoom *= 1.1
        elif species == "robot":
            noise_weight -= 0.15
            symmetry_level = min(1.0, symmetry_level + 0.25)
            mandelbrot_weight += 0.10
            julia_weight -= 0.05
        elif species == "dragon":
            noise_weight += 0.25
            julia_weight += 0.20
            mandelbrot_weight -= 0.10
            zoom *= 1.2
            mood_tilt = min(1.0, mood_tilt + 0.10)

        # Normalize weights
        total_weight = mandelbrot_weight + julia_weight + noise_weight
        mandelbrot_weight /= total_weight
        julia_weight /= total_weight
        noise_weight /= total_weight

        # Centre shift depends on behaviour
        if pet_behavior == "happy":
            center_shift = (0.0, 0.0)
        elif pet_behavior == "sad":
            center_shift = (-0.3, 0.5)
        elif pet_behavior == "tired":
            center_shift = (0.1, -0.4)
        else:
            center_shift = (0.0, 0.2)

        return {
            "max_iterations": max_iterations,
            "zoom": zoom,
            "noise_strength": noise_strength,
            "base_palette": base_palette,
            "mood_tilt": mood_tilt,
            "symmetry_level": symmetry_level,
            "mandelbrot_weight": mandelbrot_weight,
            "julia_weight": julia_weight,
            "noise_weight": noise_weight,
            "center_shift": center_shift,
            "period": period,
        }


class FractalEngine:
    """
    High level interface over fractal generation components.

    Provides convenience methods to generate a static fractal or a mandala with
    symmetry applied.  Instances of this class hold their own data mapper,
    generator and symmetry engine.  This class hides the complexity of the
    underlying components and prepares images for the user interface.
    """

    def __init__(self, width: int = 512, height: int = 512, segments: int = 8) -> None:
        self.mapper = FractalDataMapper()
        self.generator = MultiFractalGenerator(width, height)
        self.symmetry = SymmetryEngine(segments)

    def create_image(self, user_data: Dict[str, float], pet_state: Dict[str, float], apply_symmetry: bool = True) -> Image.Image:
        params = self.mapper.map_data_to_params(user_data, pet_state)
        palette = PaletteGenerator(base_palette=params["base_palette"], mood_tilt=params["mood_tilt"])
        img = self.generator.generate_image(params, palette)
        if apply_symmetry:
            img = self.symmetry.apply(img)
        return img


# -----------------------------------------------------------------------------
# Virtual pet state and behaviour
#
# The virtual pet holds a simple state (hunger, energy, mood, stress, growth,
# behaviour and species) and exposes methods to update that state based on
# user inputs.  Entropy causes the state to drift towards disorder unless
# maintained, simulating the need for regular care.  A behaviour engine
# determines the pet's mood and animates its actions.  In a larger system the
# pet could load and save its state from disk; here it is kept in memory for
# brevity.

@dataclass
class PetState:
    hunger: float = 50.0
    energy: float = 50.0
    mood: float = 50.0
    stress: float = 50.0
    growth: float = 1.0
    behavior: str = "idle"
    species: str = "cat"


class EntropyEngine:
    """
    Apply natural decay to the pet state.

    Entropy increases hunger and stress and decreases energy and mood unless
    countered by user inputs such as nutrition and rest.  The loss is
    influenced by the pet's current stress and energy levels.
    """

    def __init__(self, base_rate: float = 0.2) -> None:
        self.base_rate = base_rate

    def apply(self, state: PetState) -> None:
        try:
            loss = self.base_rate + (state.stress * 0.02) - (state.energy * 0.01)
            state.hunger = min(100.0, state.hunger + loss)
            state.energy = max(0.0, state.energy - loss * 0.5)
            state.mood = max(0.0, state.mood - loss * 0.3)
            logger.debug("Entropy applied: %.3f", loss)
        except Exception as e:
            logger.error("EntropyEngine error: %s", e)


class PhysiologyEngine:
    """
    Update pet physiology based on user inputs.

    Sleep replenishes energy and improves mood.  Nutrition reduces hunger and
    mitigates stress.  External stress inputs increase stress and depress mood.
    """

    def update(self, state: PetState, user_inputs: Dict[str, float]) -> None:
        try:
            sleep = float(user_inputs.get("sleep_hours", 0.0))
            nutrition = float(user_inputs.get("nutrition_score", 0.0))
            stress_input = float(user_inputs.get("stress", 0.0))
            state.energy = min(100.0, state.energy + sleep * 0.6)
            state.hunger = max(0.0, state.hunger - nutrition * 0.5)
            state.stress = self._clamp(state.stress + stress_input * 0.5, 0.0, 100.0)
            state.mood = self._clamp(state.mood + (sleep * 0.2) - (stress_input * 0.6), 0.0, 100.0)
        except Exception as e:
            logger.error("PhysiologyEngine error: %s", e)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


class BehaviorEngine:
    """
    Determine the pet's behaviour category based on its current state.

    Behaviours correspond to broad emotional states which in turn control the
    virtual pet's appearance and influence fractal parameters via the centre
    shift in FractalDataMapper.
    """

    def choose_behavior(self, state: PetState) -> str:
        if state.mood < 30.0:
            return "sad"
        if state.energy < 30.0:
            return "tired"
        if state.mood > 70.0:
            return "happy"
        return "idle"


class VirtualPet:
    """
    Manage the pet state and interface with the entropy, physiology and behaviour
    engines.  This class exposes methods to update from user data and to
    retrieve the current state as a dictionary for other modules (e.g. the
    fractal engine).
    """

    def __init__(self, species: str = "cat") -> None:
        self.state = PetState(species=species)
        self.entropy_engine = EntropyEngine()
        self.phys_engine = PhysiologyEngine()
        self.behavior_engine = BehaviorEngine()

    def update_from_user(self, user_data: Dict[str, float]) -> None:
        # Apply entropy first to simulate natural decay
        self.entropy_engine.apply(self.state)
        # Apply user inputs to influence state
        self.phys_engine.update(self.state, user_data)
        # Update growth: goals completed encourage growth
        goals = float(user_data.get("goals_completed", 0.0))
        self.state.growth = min(100.0, self.state.growth + goals * 2.0)
        # Determine behaviour
        self.state.behavior = self.behavior_engine.choose_behavior(self.state)
        logger.debug(
            "Pet updated: hunger=%.1f energy=%.1f mood=%.1f stress=%.1f growth=%.1f behavior=%s",
            self.state.hunger,
            self.state.energy,
            self.state.mood,
            self.state.stress,
            self.state.growth,
            self.state.behavior,
        )

    def get_state_dict(self) -> Dict[str, float]:
        return {
            "hunger": self.state.hunger,
            "energy": self.state.energy,
            "mood": self.state.mood,
            "stress": self.state.stress,
            "growth": self.state.growth,
            "behavior": self.state.behavior,
            "species": self.state.species,
        }


# -----------------------------------------------------------------------------
# Life planning system
#
# This class orchestrates the various engines to provide a simple API for
# updating user data, training models and generating outputs.  It is the
# primary entry point for the life planning tool.

class LifePlanningSystem:
    """
    High level coordinator for the virtual pet, predictive models, fuzzy logic
    and fractal art generation.

    Use this class within your life planning application to encapsulate all
    heavy computation.  The update() method should be called whenever new
    user inputs are available.  After updating, call generate_guidance() to
    obtain textual advice and generate_fractal_image() to obtain an image.
    """

    def __init__(self, species: str = "cat") -> None:
        self.pet = VirtualPet(species=species)
        self.predictor = DecisionTreePredictor()
        self.fuzzy_engine = FuzzyLogicEngine()
        self.fractal_engine = FractalEngine(width=512, height=512, segments=8)
        self.history: List[Dict[str, float]] = []

    def update(self, user_data: Dict[str, float]) -> None:
        """
        Update internal state with new user data.

        Args:
            user_data: Dictionary containing the same keys as required by
                FractalDataMapper and the physiology engine.  Should include
                'stress', 'mood', 'goals_completed', 'sleep_hours',
                'nutrition_score' and 'period'.
        """
        try:
            # Update pet state
            self.pet.update_from_user(user_data)
            # Append to history with a target (we must decide target for next
            # mood; for training we can approximate by using current mood as
            # next mood of the previous record).  If history not empty,
            # assign the current mood to the previous record's mood_next.
            if self.history:
                self.history[-1]["mood_next"] = float(user_data.get("mood", 50.0))
            # Store the current record (target to be filled on next update)
            record = {
                "stress": float(user_data.get("stress", 50.0)),
                "mood": float(user_data.get("mood", 50.0)),
                "energy": float(self.pet.state.energy),
                "goals_completed": float(user_data.get("goals_completed", 0.0)),
                "sleep_hours": float(user_data.get("sleep_hours", 0.0)),
                "mood_next": float(user_data.get("mood", 50.0)),  # placeholder
            }
            self.history.append(record)
            # Train predictor on history when enough samples collected
            if len(self.history) > 5:
                self.predictor.train(self.history[:-1])
        except Exception as e:
            logger.error("LifePlanningSystem.update error: %s", e)

    def generate_guidance(self) -> Dict[str, str]:
        """
        Produce guidance messages using decision tree predictions and fuzzy logic.

        Returns a dictionary with keys:
            'predicted_mood': The predicted mood for the next period.
            'fuzzy_message': A supportive message from the fuzzy engine.
            'message': A combined guidance string.
        """
        try:
            current_state = self.pet.get_state_dict()
            # Use last history record for predictor features
            if self.history:
                features = {
                    "stress": self.history[-1]["stress"],
                    "mood": self.history[-1]["mood"],
                    "energy": self.history[-1]["energy"],
                    "goals_completed": self.history[-1]["goals_completed"],
                    "sleep_hours": self.history[-1]["sleep_hours"],
                }
            else:
                features = {
                    "stress": current_state["stress"],
                    "mood": current_state["mood"],
                    "energy": current_state["energy"],
                    "goals_completed": 0.0,
                    "sleep_hours": 0.0,
                }
            predicted_mood = self.predictor.predict_next_mood(features)
            # Fuzzy logic uses current stress and mood
            fuzzy_message = self.fuzzy_engine.infer(current_state["stress"], current_state["mood"])
            # Combine guidance
            message = (
                f"Your predicted mood for the next period is {predicted_mood:.1f}. "
                f"{fuzzy_message}"
            )
            return {
                "predicted_mood": f"{predicted_mood:.1f}",
                "fuzzy_message": fuzzy_message,
                "message": message,
            }
        except Exception as e:
            logger.error("generate_guidance error: %s", e)
            return {
                "predicted_mood": "N/A",
                "fuzzy_message": "N/A",
                "message": "An error occurred while generating guidance.",
            }

    def generate_fractal_image(self) -> Image.Image:
        """
        Create a fractal image reflecting current user data and pet state.
        The caller is responsible for saving or displaying the image.
        """
        # Use the last user input stored in history along with pet state
        try:
            if not self.history:
                # Default neutral input
                user_data = {
                    "stress": 50.0,
                    "mood": 50.0,
                    "goals_completed": 0.0,
                    "sleep_hours": 0.0,
                    "nutrition_score": 50.0,
                    "period": "daily",
                }
            else:
                last = self.history[-1]
                user_data = {
                    "stress": last.get("stress", 50.0),
                    "mood": last.get("mood", 50.0),
                    "goals_completed": last.get("goals_completed", 0.0),
                    "sleep_hours": last.get("sleep_hours", 0.0),
                    "nutrition_score": 50.0,
                    "period": "daily",
                }
            pet_state = self.pet.get_state_dict()
            return self.fractal_engine.create_image(user_data, pet_state, apply_symmetry=True)
        except Exception as e:
            logger.error("generate_fractal_image error: %s", e)
            # Return a blank image to avoid crashing the UI
            return Image.new("RGB", (512, 512), color=(0, 0, 0))


if __name__ == "__main__":
    # Demonstration of usage
    system = LifePlanningSystem(species="cat")
    # Simulate a series of user inputs
    inputs = [
        {"stress": 40, "mood": 70, "goals_completed": 3, "sleep_hours": 7, "nutrition_score": 60, "period": "daily"},
        {"stress": 60, "mood": 50, "goals_completed": 1, "sleep_hours": 6, "nutrition_score": 50, "period": "daily"},
        {"stress": 20, "mood": 80, "goals_completed": 5, "sleep_hours": 8, "nutrition_score": 70, "period": "daily"},
    ]
    for entry in inputs:
        system.update(entry)
        guidance = system.generate_guidance()
        print(guidance["message"])
    # Generate final fractal image and save it
    img = system.generate_fractal_image()
    img_path = os.path.join(os.getcwd(), "demo_fractal.png")
    img.save(img_path)
    print(f"Fractal image saved to {img_path}")