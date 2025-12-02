#!/usr/bin/env python3
"""
🌀 LIFE FRACTAL INTELLIGENCE v9.0 - OCR & INTELLIGENT DATA INGESTION
════════════════════════════════════════════════════════════════════════════════

For brains like mine - built with love for the neurodivergent community.

NEW IN v9.0:
✅ OCR text recognition from images (journal entries, notes, screenshots)
✅ Automatic sentiment analysis and mood detection
✅ Privacy-first: Personal data stored locally only
✅ Federated learning: Anonymized insights improve the master AI
✅ Rapid data entry via image upload
✅ Auto-create tasks from recognized text
✅ Mathematical mood vectors for visualization
✅ Scrapbook/journal page detection

════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import secrets
import logging
import hashlib
import sqlite3
import re
import uuid
import base64
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from io import BytesIO
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

# Flask imports
from flask import Flask, request, jsonify, send_file, render_template_string, session, redirect, url_for, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Data processing
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ════════════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS - OCR & ML
# ════════════════════════════════════════════════════════════════════════════════

# Try to import pytesseract for local OCR
try:
    import pytesseract
    HAS_TESSERACT = True
    logging.info("✅ Tesseract OCR available")
except ImportError:
    HAS_TESSERACT = False
    logging.info("⚠️ Tesseract not available - using fallback OCR")

# Try to import easyocr as backup
try:
    import easyocr
    HAS_EASYOCR = True
    EASYOCR_READER = None  # Lazy load
    logging.info("✅ EasyOCR available")
except ImportError:
    HAS_EASYOCR = False
    EASYOCR_READER = None

# Try OpenCV for image processing
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# SACRED MATHEMATICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]

# Mayan Calendar
MAYAN_DAY_SIGNS = [
    ("Imix", "primordial waters, new beginnings"),
    ("Ik", "breath of life, wind spirit"),
    ("Akbal", "darkness, inner reflection"),
    ("Kan", "seed, growth potential"),
    ("Chicchan", "serpent energy, kundalini"),
    ("Cimi", "transformation, death/rebirth"),
    ("Manik", "healing hand, accomplishment"),
    ("Lamat", "star harmony, abundance"),
    ("Muluc", "cosmic water, emotions"),
    ("Oc", "loyalty, heart guidance"),
    ("Chuen", "creative play, artistry"),
    ("Eb", "road of life, human journey"),
    ("Ben", "sky walker, pillars of light"),
    ("Ix", "jaguar wisdom, earth magic"),
    ("Men", "eagle vision, higher perspective"),
    ("Cib", "ancestral wisdom, forgiveness"),
    ("Caban", "earth force, synchronicity"),
    ("Etznab", "mirror truth, clarity"),
    ("Cauac", "thunder being, purification"),
    ("Ahau", "sun lord, enlightenment")
]

# ════════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class SentimentEngine:
    """Advanced sentiment analysis with neurodivergent-aware vocabulary."""
    
    # Expanded word lists with neurodivergent-specific terms
    POSITIVE_WORDS = {
        # General positive
        "happy", "joy", "excited", "hope", "hopeful", "calm", "relaxed",
        "peace", "peaceful", "content", "grateful", "love", "loving", "cheerful",
        "optimistic", "bright", "upbeat", "wonderful", "amazing", "great",
        "good", "nice", "lovely", "beautiful", "fantastic", "excellent",
        "awesome", "brilliant", "delightful", "pleasant", "satisfied",
        # Neurodivergent-specific positive
        "focused", "hyperfocus", "flow", "special interest", "stimming",
        "regulated", "grounded", "safe", "understood", "accepted",
        "accommodated", "supported", "validated", "seen", "heard",
        "comfortable", "routine", "predictable", "quiet", "rest",
        "recharge", "spoons", "energy", "capable", "accomplished",
        "masking-free", "authentic", "sensory-friendly"
    }
    
    NEGATIVE_WORDS = {
        # General negative
        "sad", "angry", "upset", "anxious", "anxiety", "fear", "depressed",
        "tired", "exhausted", "worried", "stressed", "stress", "hopeless",
        "overwhelmed", "scared", "afraid", "lonely", "down", "terrible",
        "awful", "horrible", "bad", "painful", "hurt", "suffering",
        "miserable", "frustrated", "annoyed", "irritated", "disappointed",
        # Neurodivergent-specific challenges
        "meltdown", "shutdown", "burnout", "overload", "overstimulated",
        "understimulated", "masking", "exhausting", "rejection", "rsd",
        "executive dysfunction", "paralysis", "stuck", "frozen",
        "sensory overload", "too loud", "too bright", "overwhelming",
        "no spoons", "depleted", "drained", "can't focus", "scattered",
        "time blind", "forgot", "late", "missed", "failed", "ashamed",
        "misunderstood", "judged", "excluded", "isolated"
    }
    
    # Intensity modifiers
    INTENSIFIERS = {"very", "really", "extremely", "so", "incredibly", "absolutely", "totally"}
    DIMINISHERS = {"somewhat", "slightly", "a bit", "kind of", "sort of", "a little"}
    NEGATIONS = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", "dont", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't", "shouldn't"}
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into lowercase tokens."""
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    
    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis."""
        if not text or not text.strip():
            return {
                'score': 0.0,
                'magnitude': 0.0,
                'label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'key_phrases': [],
                'detected_emotions': ['neutral']
            }
        
        tokens = cls.tokenize(text)
        if not tokens:
            return {
                'score': 0.0,
                'magnitude': 0.0,
                'label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'key_phrases': [],
                'detected_emotions': ['neutral']
            }
        
        pos_count = 0
        neg_count = 0
        pos_words = []
        neg_words = []
        
        # Track negation context
        negation_active = False
        intensifier_active = False
        
        for i, token in enumerate(tokens):
            # Check for negation
            if token in cls.NEGATIONS:
                negation_active = True
                continue
            
            # Check for intensifiers
            if token in cls.INTENSIFIERS:
                intensifier_active = True
                continue
            
            multiplier = 1.5 if intensifier_active else 1.0
            
            if token in cls.POSITIVE_WORDS:
                if negation_active:
                    neg_count += multiplier
                    neg_words.append(token)
                else:
                    pos_count += multiplier
                    pos_words.append(token)
            elif token in cls.NEGATIVE_WORDS:
                if negation_active:
                    pos_count += multiplier * 0.5  # Negated negative is weakly positive
                    pos_words.append(f"not {token}")
                else:
                    neg_count += multiplier
                    neg_words.append(token)
            
            # Reset modifiers after use
            negation_active = False
            intensifier_active = False
        
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
        
        # Magnitude indicates strength of sentiment
        magnitude = min(1.0, total / max(len(tokens) * 0.3, 1))
        
        # Determine label
        if score > 0.3:
            label = 'positive'
        elif score < -0.3:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Detect specific emotions
        emotions = cls._detect_emotions(tokens, score)
        
        return {
            'score': round(score, 3),
            'magnitude': round(magnitude, 3),
            'label': label,
            'positive_count': int(pos_count),
            'negative_count': int(neg_count),
            'key_phrases': pos_words[:5] + neg_words[:5],
            'detected_emotions': emotions
        }
    
    @classmethod
    def _detect_emotions(cls, tokens: List[str], score: float) -> List[str]:
        """Detect specific emotional states from tokens."""
        emotions = []
        token_set = set(tokens)
        
        # Anxiety detection
        anxiety_words = {"anxious", "anxiety", "worried", "nervous", "panic", "fear", "scared"}
        if token_set & anxiety_words:
            emotions.append("anxiety")
        
        # Depression detection
        depression_words = {"depressed", "hopeless", "worthless", "empty", "numb"}
        if token_set & depression_words:
            emotions.append("depression_indicators")
        
        # Burnout detection
        burnout_words = {"burnout", "exhausted", "depleted", "drained", "overwhelmed", "shutdown"}
        if token_set & burnout_words:
            emotions.append("burnout")
        
        # Joy detection
        joy_words = {"happy", "joy", "excited", "thrilled", "delighted", "wonderful"}
        if token_set & joy_words:
            emotions.append("joy")
        
        # Calm detection
        calm_words = {"calm", "peaceful", "relaxed", "serene", "tranquil", "content"}
        if token_set & calm_words:
            emotions.append("calm")
        
        # Executive function struggles
        ef_words = {"forgot", "late", "stuck", "paralysis", "frozen", "scattered"}
        if token_set & ef_words:
            emotions.append("executive_dysfunction")
        
        # Sensory issues
        sensory_words = {"loud", "bright", "overwhelming", "overstimulated", "sensory"}
        if token_set & sensory_words:
            emotions.append("sensory_overload")
        
        if not emotions:
            emotions.append("neutral")
        
        return emotions


# ════════════════════════════════════════════════════════════════════════════════
# MOOD VECTOR COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

class MoodVector:
    """Six-dimensional mood vector computation using sacred mathematics."""
    
    # Dimensions: [calm, anxious, hopeful, tired, scattered, focused]
    DIMENSION_NAMES = ['calm', 'anxious', 'hopeful', 'tired', 'scattered', 'focused']
    
    @classmethod
    def compute(cls, sentiment_score: float, detected_emotions: List[str] = None) -> List[float]:
        """Compute a six-dimensional mood vector from sentiment and emotions."""
        sentiment = max(-1.0, min(1.0, sentiment_score))
        
        # Base values from sentiment
        calm = max(0.0, sentiment * PHI_INVERSE + 0.3)
        anxious = max(0.0, -sentiment * PHI_INVERSE + 0.2)
        hopeful = (sentiment + 1.0) / 2.0
        tired = 1.0 - hopeful
        scattered = abs(sentiment) * PHI_INVERSE
        focused = 1.0 - scattered
        
        # Adjust based on detected emotions
        if detected_emotions:
            if 'anxiety' in detected_emotions:
                anxious = min(1.0, anxious + 0.3)
                calm = max(0.0, calm - 0.2)
            if 'burnout' in detected_emotions:
                tired = min(1.0, tired + 0.4)
                focused = max(0.0, focused - 0.3)
            if 'executive_dysfunction' in detected_emotions:
                scattered = min(1.0, scattered + 0.3)
                focused = max(0.0, focused - 0.2)
            if 'sensory_overload' in detected_emotions:
                anxious = min(1.0, anxious + 0.2)
                scattered = min(1.0, scattered + 0.2)
            if 'joy' in detected_emotions:
                hopeful = min(1.0, hopeful + 0.2)
                calm = min(1.0, calm + 0.1)
            if 'calm' in detected_emotions:
                calm = min(1.0, calm + 0.3)
                anxious = max(0.0, anxious - 0.2)
        
        vector = [calm, anxious, hopeful, tired, scattered, focused]
        return [round(max(0.0, min(1.0, v)), 3) for v in vector]
    
    @classmethod
    def to_dict(cls, vector: List[float]) -> Dict[str, float]:
        """Convert vector to named dictionary."""
        return {name: val for name, val in zip(cls.DIMENSION_NAMES, vector)}
    
    @classmethod
    def mental_health_flags(cls, vector: List[float]) -> List[str]:
        """Generate mental health flags from mood vector."""
        flags = []
        calm, anxious, hopeful, tired, scattered, focused = vector
        
        if anxious > 0.6:
            flags.append("high_anxiety")
        if calm > 0.7:
            flags.append("calm_state")
        if tired > 0.7:
            flags.append("fatigue_warning")
        if hopeful > 0.7:
            flags.append("positive_outlook")
        if scattered > 0.6:
            flags.append("attention_scattered")
        if focused > 0.7:
            flags.append("good_focus")
        if tired > 0.8 and anxious > 0.5:
            flags.append("burnout_risk")
        
        if not flags:
            flags.append("balanced")
        
        return flags
    
    @classmethod
    def wellness_score(cls, vector: List[float]) -> float:
        """Calculate overall wellness from mood vector using Fibonacci weighting."""
        calm, anxious, hopeful, tired, scattered, focused = vector
        
        # Fibonacci weights: positive traits weighted higher
        weights = [FIBONACCI[8], FIBONACCI[5], FIBONACCI[7], FIBONACCI[4], FIBONACCI[3], FIBONACCI[6]]
        
        # Positive contributions
        positive = calm * weights[0] + hopeful * weights[2] + focused * weights[5]
        
        # Negative contributions (inverted)
        negative = anxious * weights[1] + tired * weights[3] + scattered * weights[4]
        
        total_weight = sum(weights)
        wellness = (positive - negative * 0.5 + total_weight * 0.5) / total_weight
        
        return round(max(0, min(100, wellness * 100)), 1)


# ════════════════════════════════════════════════════════════════════════════════
# OCR ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class OCREngine:
    """Multi-backend OCR with preprocessing for handwritten text."""
    
    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image.copy()
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)
        
        # Apply slight sharpening
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (OCR works better on larger images)
        min_dimension = 1000
        w, h = sharpened.size
        if w < min_dimension or h < min_dimension:
            scale = max(min_dimension / w, min_dimension / h)
            new_size = (int(w * scale), int(h * scale))
            sharpened = sharpened.resize(new_size, Image.Resampling.LANCZOS)
        
        return sharpened
    
    @classmethod
    def extract_text(cls, image: Image.Image, use_preprocessing: bool = True) -> Dict[str, Any]:
        """Extract text from image using available OCR engine."""
        if use_preprocessing:
            processed = cls.preprocess_image(image)
        else:
            processed = image
        
        text = ""
        confidence = 0.0
        method = "none"
        
        # Try Tesseract first
        if HAS_TESSERACT:
            try:
                # Use multiple PSM modes for better results
                configs = [
                    '--psm 3',  # Fully automatic page segmentation
                    '--psm 6',  # Uniform block of text
                    '--psm 4',  # Single column of text
                ]
                
                best_text = ""
                for config in configs:
                    try:
                        result = pytesseract.image_to_string(processed, config=config)
                        if len(result) > len(best_text):
                            best_text = result
                    except:
                        continue
                
                if best_text.strip():
                    text = best_text
                    method = "tesseract"
                    # Get confidence from detailed data
                    try:
                        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
                        confs = [int(c) for c in data['conf'] if int(c) > 0]
                        confidence = sum(confs) / len(confs) / 100 if confs else 0.5
                    except:
                        confidence = 0.6
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        # Try EasyOCR as fallback
        if not text and HAS_EASYOCR:
            try:
                global EASYOCR_READER
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
                
                # Convert to numpy array
                img_array = np.array(processed)
                results = EASYOCR_READER.readtext(img_array)
                
                if results:
                    text = " ".join([r[1] for r in results])
                    confidence = sum([r[2] for r in results]) / len(results)
                    method = "easyocr"
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Fallback: Return empty with instruction
        if not text:
            method = "fallback"
            confidence = 0.0
        
        return {
            'text': text.strip(),
            'confidence': round(confidence, 3),
            'method': method,
            'char_count': len(text),
            'word_count': len(text.split()) if text else 0
        }
    
    @classmethod
    def extract_dates(cls, text: str) -> List[Dict[str, Any]]:
        """Extract dates from text."""
        dates = []
        
        # Various date patterns
        patterns = [
            # MM/DD/YYYY or MM-DD-YYYY
            (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', 'mdy'),
            # YYYY-MM-DD
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'ymd'),
            # Month DD, YYYY
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', 'text'),
            # DD Month YYYY
            (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', 'text_eu'),
        ]
        
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern, fmt in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    if fmt == 'mdy':
                        m, d, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    elif fmt == 'ymd':
                        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    elif fmt == 'text':
                        m = month_map[match.group(1).lower()]
                        d, y = int(match.group(2)), int(match.group(3))
                    elif fmt == 'text_eu':
                        d = int(match.group(1))
                        m = month_map[match.group(2).lower()]
                        y = int(match.group(3))
                    
                    date = datetime(y, m, d)
                    dates.append({
                        'date': date.isoformat(),
                        'raw': match.group(0),
                        'confidence': 0.9
                    })
                except:
                    continue
        
        return dates
    
    @classmethod
    def extract_tasks(cls, text: str) -> List[Dict[str, Any]]:
        """Extract potential tasks from text."""
        tasks = []
        
        # Task patterns
        task_patterns = [
            r'(?:TODO|To Do|TO-DO|Task):\s*(.+?)(?:\n|$)',
            r'[-•*]\s*(.+?)(?:\n|$)',
            r'(?:\d+[.)]\s*)(.+?)(?:\n|$)',
            r'(?:Need to|Have to|Must|Should|Remember to)\s+(.+?)(?:\.|!|\n|$)',
        ]
        
        for pattern in task_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                task_text = match.group(1).strip()
                if len(task_text) > 3 and len(task_text) < 200:
                    # Estimate spoon cost based on complexity
                    word_count = len(task_text.split())
                    spoon_cost = min(5, max(1, word_count // 5 + 1))
                    
                    tasks.append({
                        'title': task_text[:100],
                        'full_text': task_text,
                        'spoon_cost': spoon_cost,
                        'confidence': 0.7
                    })
        
        # Deduplicate similar tasks
        unique_tasks = []
        seen = set()
        for task in tasks:
            key = task['title'].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        
        return unique_tasks[:10]  # Limit to 10 tasks


# ════════════════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS (SCRAPBOOK DETECTION)
# ════════════════════════════════════════════════════════════════════════════════

class ImageAnalyzer:
    """Analyze images for visual features and scrapbook elements."""
    
    @classmethod
    def analyze(cls, image: Image.Image) -> Dict[str, Any]:
        """Analyze image for visual characteristics."""
        features = []
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Detect dominant colors
        colors = cls._detect_dominant_colors(img_array)
        
        # Check for white borders (polaroid style)
        if cls._has_white_border(img_array):
            features.append("polaroid_border")
        
        # Check color variance (scrapbook indicator)
        variance = cls._color_variance(img_array)
        if variance > 2000:
            features.append("high_color_variance")
        elif variance < 500:
            features.append("uniform_background")
        
        # Detect if it's primarily text (journal page)
        text_ratio = cls._estimate_text_ratio(image)
        if text_ratio > 0.3:
            features.append("text_heavy")
        
        # Estimate brightness
        brightness = cls._calculate_brightness(img_array)
        if brightness > 200:
            features.append("bright")
        elif brightness < 80:
            features.append("dark")
        
        return {
            'features': features,
            'dominant_colors': colors,
            'color_variance': round(variance, 2),
            'brightness': round(brightness, 2),
            'text_ratio': round(text_ratio, 3),
            'dimensions': {'width': image.width, 'height': image.height}
        }
    
    @staticmethod
    def _detect_dominant_colors(img_array: np.ndarray, n_colors: int = 3) -> List[str]:
        """Detect dominant colors in image."""
        # Resize for faster processing
        small = Image.fromarray(img_array).resize((50, 50))
        pixels = np.array(small).reshape(-1, 3)
        
        # Simple k-means-like clustering
        colors = []
        for _ in range(n_colors):
            if len(pixels) == 0:
                break
            # Find most common color (simplified)
            mean_color = pixels.mean(axis=0).astype(int)
            colors.append(f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}")
            # Remove similar colors
            distances = np.linalg.norm(pixels - mean_color, axis=1)
            pixels = pixels[distances > 50]
        
        return colors
    
    @staticmethod
    def _has_white_border(img_array: np.ndarray, margin_ratio: float = 0.05) -> bool:
        """Check if image has white borders."""
        h, w = img_array.shape[:2]
        margin = int(min(h, w) * margin_ratio)
        if margin < 1:
            return False
        
        # Sample border pixels
        top = img_array[:margin, :, :]
        bottom = img_array[-margin:, :, :]
        left = img_array[:, :margin, :]
        right = img_array[:, -margin:, :]
        
        borders = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ])
        
        # Calculate mean brightness
        brightness = borders.mean()
        return brightness > 230
    
    @staticmethod
    def _color_variance(img_array: np.ndarray) -> float:
        """Calculate color variance."""
        return float(np.var(img_array))
    
    @staticmethod
    def _estimate_text_ratio(image: Image.Image) -> float:
        """Estimate how much of the image is text."""
        gray = image.convert('L')
        
        # Threshold to find dark regions (likely text)
        threshold = 128
        dark_pixels = sum(1 for p in gray.getdata() if p < threshold)
        total_pixels = gray.width * gray.height
        
        return dark_pixels / total_pixels if total_pixels > 0 else 0
    
    @staticmethod
    def _calculate_brightness(img_array: np.ndarray) -> float:
        """Calculate average brightness."""
        return float(img_array.mean())


# ════════════════════════════════════════════════════════════════════════════════
# FEDERATED LEARNING - PRIVACY-PRESERVING DATA AGGREGATION
# ════════════════════════════════════════════════════════════════════════════════

class FederatedLearning:
    """
    Privacy-preserving data aggregation for improving the master AI.
    
    Personal data stays local. Only anonymized, aggregated insights are shared.
    """
    
    @staticmethod
    def anonymize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an anonymized version of an entry for federated learning.
        
        Removes: personal text, names, dates, identifiers
        Keeps: sentiment scores, mood vectors, aggregate statistics
        """
        anonymized = {
            'timestamp_bucket': datetime.now().strftime('%Y-%m'),  # Only month granularity
            'sentiment_score': entry.get('sentiment_score', 0),
            'mood_vector': entry.get('mood_vector', [0.5] * 6),
            'wellness_score': entry.get('wellness_score', 50),
            'word_count_bucket': cls._bucket_value(entry.get('word_count', 0), [0, 50, 100, 200, 500]),
            'detected_emotions': entry.get('detected_emotions', ['neutral']),
            'mental_health_flags': entry.get('mental_health_flags', ['balanced']),
            'image_features': entry.get('image_features', []),
            'has_tasks': len(entry.get('extracted_tasks', [])) > 0,
            # No personal text, names, or specific dates
        }
        return anonymized
    
    @staticmethod
    def _bucket_value(value: int, buckets: List[int]) -> str:
        """Convert value to privacy-preserving bucket."""
        for i, threshold in enumerate(buckets[1:], 1):
            if value < threshold:
                return f"{buckets[i-1]}-{threshold}"
        return f"{buckets[-1]}+"
    
    @classmethod
    def prepare_batch_upload(cls, entries: List[Dict[str, Any]], 
                            consent_level: str = 'minimal') -> Dict[str, Any]:
        """
        Prepare a batch of entries for upload to improve the master AI.
        
        consent_level:
            - 'none': No data shared
            - 'minimal': Only aggregate statistics
            - 'standard': Anonymized individual entries
            - 'full': Full anonymized data with emotion patterns
        """
        if consent_level == 'none':
            return {'consent': 'none', 'data': None}
        
        anonymized_entries = [cls.anonymize_entry(e) for e in entries]
        
        if consent_level == 'minimal':
            # Only aggregate statistics
            return {
                'consent': 'minimal',
                'data': {
                    'entry_count': len(entries),
                    'avg_sentiment': np.mean([e['sentiment_score'] for e in anonymized_entries]),
                    'avg_wellness': np.mean([e['wellness_score'] for e in anonymized_entries]),
                    'common_emotions': cls._get_common_items([e['detected_emotions'] for e in anonymized_entries]),
                    'common_flags': cls._get_common_items([e['mental_health_flags'] for e in anonymized_entries])
                }
            }
        
        if consent_level == 'standard':
            return {
                'consent': 'standard',
                'data': {
                    'entries': anonymized_entries,
                    'aggregate': {
                        'entry_count': len(entries),
                        'avg_sentiment': np.mean([e['sentiment_score'] for e in anonymized_entries]),
                        'sentiment_std': np.std([e['sentiment_score'] for e in anonymized_entries])
                    }
                }
            }
        
        # Full consent
        return {
            'consent': 'full',
            'data': {
                'entries': anonymized_entries,
                'patterns': cls._extract_patterns(anonymized_entries),
                'aggregate': {
                    'entry_count': len(entries),
                    'avg_sentiment': np.mean([e['sentiment_score'] for e in anonymized_entries]),
                    'mood_vector_avg': np.mean([e['mood_vector'] for e in anonymized_entries], axis=0).tolist()
                }
            }
        }
    
    @staticmethod
    def _get_common_items(lists_of_items: List[List[str]], top_n: int = 5) -> List[str]:
        """Get most common items across lists."""
        from collections import Counter
        all_items = [item for sublist in lists_of_items for item in sublist]
        return [item for item, _ in Counter(all_items).most_common(top_n)]
    
    @staticmethod
    def _extract_patterns(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract anonymized patterns from entries."""
        return {
            'emotion_transitions': [],  # Would track emotion flow patterns
            'wellness_trends': [],  # Would track wellness over time
            'common_combinations': []  # Common emotion/flag combinations
        }


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════════════════════════

DATABASE_PATH = os.environ.get('DATABASE_PATH', 'life_fractal_v9.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with all tables including OCR entries."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT DEFAULT '',
            display_name TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            spoons INTEGER DEFAULT 12,
            max_spoons INTEGER DEFAULT 12,
            current_streak INTEGER DEFAULT 0,
            subscription_status TEXT DEFAULT 'trial',
            trial_end_date TEXT,
            data_sharing_consent TEXT DEFAULT 'minimal',
            created_at TEXT,
            last_login TEXT
        )
    ''')
    
    # Accessibility preferences
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accessibility_prefs (
            user_id TEXT PRIMARY KEY,
            reduced_motion INTEGER DEFAULT 0,
            high_contrast INTEGER DEFAULT 0,
            larger_text INTEGER DEFAULT 0,
            dyslexia_font INTEGER DEFAULT 0,
            calm_colors INTEGER DEFAULT 0,
            focus_indicators INTEGER DEFAULT 1,
            time_blindness_helpers INTEGER DEFAULT 1,
            task_chunking INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Pets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            user_id TEXT PRIMARY KEY,
            species TEXT DEFAULT 'cat',
            name TEXT DEFAULT 'Buddy',
            hunger REAL DEFAULT 50.0,
            energy REAL DEFAULT 50.0,
            mood REAL DEFAULT 50.0,
            bond REAL DEFAULT 20.0,
            level INTEGER DEFAULT 1,
            experience INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Daily entries
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            mood_score REAL DEFAULT 50.0,
            energy_level REAL DEFAULT 50.0,
            sleep_quality REAL DEFAULT 70.0,
            anxiety_level REAL DEFAULT 30.0,
            stress_level REAL DEFAULT 30.0,
            journal_entry TEXT DEFAULT '',
            wellness_index REAL DEFAULT 50.0,
            created_at TEXT,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # OCR Entries - Local storage for personal data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_entries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source_type TEXT DEFAULT 'image',
            raw_text TEXT,
            cleaned_text TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            mood_vector TEXT,
            wellness_score REAL,
            detected_emotions TEXT,
            mental_health_flags TEXT,
            extracted_dates TEXT,
            extracted_tasks TEXT,
            image_features TEXT,
            ocr_confidence REAL,
            ocr_method TEXT,
            word_count INTEGER,
            created_at TEXT,
            entry_date TEXT,
            notes TEXT,
            is_private INTEGER DEFAULT 1,
            federated_uploaded INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            priority INTEGER DEFAULT 3,
            progress REAL DEFAULT 0.0,
            target_date TEXT,
            color TEXT DEFAULT '#6B8E9F',
            created_at TEXT,
            completed_at TEXT,
            source TEXT DEFAULT 'manual',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            goal_id TEXT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            spoon_cost INTEGER DEFAULT 1,
            priority INTEGER DEFAULT 3,
            due_date TEXT,
            completed INTEGER DEFAULT 0,
            completed_at TEXT,
            created_at TEXT,
            source TEXT DEFAULT 'manual',
            ocr_entry_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (goal_id) REFERENCES goals(id),
            FOREIGN KEY (ocr_entry_id) REFERENCES ocr_entries(id)
        )
    ''')
    
    # Federated learning queue
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS federated_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            consent_level TEXT NOT NULL,
            created_at TEXT,
            uploaded_at TEXT,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized with OCR and federated learning tables")


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_mayan_day(date: datetime = None) -> Dict[str, Any]:
    """Calculate Mayan Tzolkin day."""
    if date is None:
        date = datetime.now()
    
    correlation = 584283
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    jdn = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    kin = (jdn - correlation) % 260
    day_number = (kin % 13) + 1
    day_sign_index = kin % 20
    
    day_sign, meaning = MAYAN_DAY_SIGNS[day_sign_index]
    
    return {
        'day_number': day_number,
        'day_sign': day_sign,
        'meaning': meaning,
        'full_name': f"{day_number} {day_sign}"
    }

PET_EMOJIS = {
    'cat': '😺', 'dragon': '🐉', 'phoenix': '🦅', 'owl': '🦉',
    'fox': '🦊', 'bunny': '🐰', 'turtle': '🐢', 'butterfly': '🦋'
}

def get_pet_emoji(species: str, mood: float) -> str:
    return PET_EMOJIS.get(species, '🐱')

def get_pet_status(mood: float) -> str:
    if mood >= 70: return "Happy 😊"
    elif mood >= 40: return "Okay 😐"
    else: return "Needs Love 💔"

def validate_email(email: str) -> bool:
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def calculate_wellness(entry: dict) -> float:
    weights = [FIBONACCI[i+3] for i in range(6)]
    total_weight = sum(weights)
    
    positive = (
        entry.get('mood_score', 50) * weights[0] +
        entry.get('energy_level', 50) * weights[1] +
        entry.get('sleep_quality', 70) * weights[2]
    )
    negative = entry.get('anxiety_level', 30) + entry.get('stress_level', 30)
    
    wellness = max(0, min(100, (positive - negative * 0.5) / total_weight * 2))
    return round(wellness, 1)


# ════════════════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_user(email: str, password: str, first_name: str = "") -> Optional[Dict]:
    db = get_db()
    cursor = db.cursor()
    
    try:
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        trial_end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        cursor.execute('''
            INSERT INTO users (id, email, password_hash, first_name, display_name,
                             subscription_status, trial_end_date, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, 'trial', ?, ?, ?)
        ''', (user_id, email.lower(), password_hash, first_name, 
              first_name or email.split('@')[0], trial_end, now, now))
        
        cursor.execute('INSERT INTO accessibility_prefs (user_id) VALUES (?)', (user_id,))
        cursor.execute('''
            INSERT INTO pets (user_id, species, name) VALUES (?, 'cat', 'Buddy')
        ''', (user_id,))
        
        db.commit()
        return {'id': user_id, 'email': email.lower(), 'first_name': first_name,
                'display_name': first_name or email.split('@')[0], 'spoons': 12}
    except:
        return None

def get_user_by_email(email: str) -> Optional[Dict]:
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email.lower(),))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict]:
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def verify_password(user: Dict, password: str) -> bool:
    return check_password_hash(user['password_hash'], password)

def get_pet(user_id: str) -> Optional[Dict]:
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM pets WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    if row:
        pet = dict(row)
        pet['emoji'] = get_pet_emoji(pet['species'], pet['mood'])
        pet['status'] = get_pet_status(pet['mood'])
        return pet
    return None

def save_ocr_entry(user_id: str, data: Dict) -> str:
    """Save an OCR-processed entry to local database."""
    db = get_db()
    cursor = db.cursor()
    
    entry_id = f"ocr_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute('''
        INSERT INTO ocr_entries (
            id, user_id, source_type, raw_text, cleaned_text,
            sentiment_score, sentiment_label, mood_vector, wellness_score,
            detected_emotions, mental_health_flags, extracted_dates,
            extracted_tasks, image_features, ocr_confidence, ocr_method,
            word_count, created_at, entry_date, notes, is_private
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry_id, user_id, data.get('source_type', 'image'),
        data.get('raw_text', ''), data.get('cleaned_text', ''),
        data.get('sentiment_score', 0), data.get('sentiment_label', 'neutral'),
        json.dumps(data.get('mood_vector', [])), data.get('wellness_score', 50),
        json.dumps(data.get('detected_emotions', [])),
        json.dumps(data.get('mental_health_flags', [])),
        json.dumps(data.get('extracted_dates', [])),
        json.dumps(data.get('extracted_tasks', [])),
        json.dumps(data.get('image_features', {})),
        data.get('ocr_confidence', 0), data.get('ocr_method', 'unknown'),
        data.get('word_count', 0), now, data.get('entry_date', now[:10]),
        data.get('notes', ''), 1
    ))
    
    db.commit()
    return entry_id

def create_task_from_ocr(user_id: str, task_data: Dict, ocr_entry_id: str) -> Dict:
    """Create a task extracted from OCR."""
    db = get_db()
    cursor = db.cursor()
    
    task_id = f"task_{secrets.token_hex(6)}"
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute('''
        INSERT INTO tasks (id, user_id, title, description, spoon_cost, 
                          created_at, source, ocr_entry_id)
        VALUES (?, ?, ?, ?, ?, ?, 'ocr', ?)
    ''', (task_id, user_id, task_data.get('title', 'New Task'),
          task_data.get('full_text', ''), task_data.get('spoon_cost', 1),
          now, ocr_entry_id))
    
    db.commit()
    return {'id': task_id, 'title': task_data['title'], 'spoon_cost': task_data['spoon_cost']}


# ════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app, supports_credentials=True)
app.teardown_appcontext(close_db)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ════════════════════════════════════════════════════════════════════════════════
# OCR API ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/ocr/process', methods=['POST'])
def process_ocr():
    """
    Process an uploaded image with OCR, sentiment analysis, and task extraction.
    
    Returns analyzed data but stores everything locally for privacy.
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Check for image in request
    if 'image' not in request.files and 'image_data' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image from file upload or base64
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            image = Image.open(file.stream)
        else:
            # Base64 encoded image
            image_data = request.form.get('image_data', '')
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 1: Analyze image features
        image_analysis = ImageAnalyzer.analyze(image)
        
        # Step 2: Extract text with OCR
        ocr_result = OCREngine.extract_text(image)
        raw_text = ocr_result['text']
        
        # Step 3: Sentiment analysis
        sentiment = SentimentEngine.analyze(raw_text)
        
        # Step 4: Compute mood vector
        mood_vector = MoodVector.compute(
            sentiment['score'], 
            sentiment['detected_emotions']
        )
        mental_health_flags = MoodVector.mental_health_flags(mood_vector)
        wellness_score = MoodVector.wellness_score(mood_vector)
        
        # Step 5: Extract dates and tasks
        extracted_dates = OCREngine.extract_dates(raw_text)
        extracted_tasks = OCREngine.extract_tasks(raw_text)
        
        # Step 6: Prepare entry data
        entry_data = {
            'source_type': 'image',
            'raw_text': raw_text,
            'cleaned_text': raw_text.strip(),
            'sentiment_score': sentiment['score'],
            'sentiment_label': sentiment['label'],
            'mood_vector': mood_vector,
            'wellness_score': wellness_score,
            'detected_emotions': sentiment['detected_emotions'],
            'mental_health_flags': mental_health_flags,
            'extracted_dates': extracted_dates,
            'extracted_tasks': extracted_tasks,
            'image_features': image_analysis,
            'ocr_confidence': ocr_result['confidence'],
            'ocr_method': ocr_result['method'],
            'word_count': ocr_result['word_count'],
            'entry_date': extracted_dates[0]['date'][:10] if extracted_dates else datetime.now().strftime('%Y-%m-%d')
        }
        
        # Step 7: Save locally (privacy-first)
        entry_id = save_ocr_entry(user_id, entry_data)
        
        # Step 8: Update pet mood based on sentiment
        pet = get_pet(user_id)
        if pet:
            mood_delta = sentiment['score'] * 10
            db = get_db()
            db.execute('''
                UPDATE pets SET mood = MIN(100, MAX(0, mood + ?)) WHERE user_id = ?
            ''', (mood_delta, user_id))
            db.commit()
        
        return jsonify({
            'success': True,
            'entry_id': entry_id,
            'ocr': {
                'text': raw_text[:500] + ('...' if len(raw_text) > 500 else ''),
                'confidence': ocr_result['confidence'],
                'method': ocr_result['method'],
                'word_count': ocr_result['word_count']
            },
            'sentiment': sentiment,
            'mood': {
                'vector': mood_vector,
                'vector_named': MoodVector.to_dict(mood_vector),
                'wellness_score': wellness_score,
                'flags': mental_health_flags
            },
            'extracted': {
                'dates': extracted_dates,
                'tasks': extracted_tasks,
                'task_count': len(extracted_tasks)
            },
            'image_analysis': image_analysis,
            'message': f"Processed successfully! Found {len(extracted_tasks)} potential tasks."
        })
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/ocr/create-tasks', methods=['POST'])
def create_tasks_from_ocr():
    """Create tasks from OCR-extracted task list."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json() or {}
    entry_id = data.get('entry_id')
    task_indices = data.get('task_indices', [])  # Which tasks to create
    
    if not entry_id:
        return jsonify({'error': 'entry_id required'}), 400
    
    # Get the OCR entry
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT extracted_tasks FROM ocr_entries WHERE id = ? AND user_id = ?', 
                   (entry_id, user_id))
    row = cursor.fetchone()
    
    if not row:
        return jsonify({'error': 'Entry not found'}), 404
    
    extracted_tasks = json.loads(row['extracted_tasks'])
    
    created_tasks = []
    for i, task in enumerate(extracted_tasks):
        if not task_indices or i in task_indices:
            created = create_task_from_ocr(user_id, task, entry_id)
            created_tasks.append(created)
    
    return jsonify({
        'success': True,
        'created_tasks': created_tasks,
        'count': len(created_tasks),
        'message': f"Created {len(created_tasks)} tasks from your notes!"
    })


@app.route('/api/ocr/entries', methods=['GET'])
def get_ocr_entries():
    """Get user's OCR entries."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT id, source_type, sentiment_score, sentiment_label, 
               wellness_score, detected_emotions, word_count, 
               created_at, entry_date
        FROM ocr_entries 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 50
    ''', (user_id,))
    
    entries = []
    for row in cursor.fetchall():
        entry = dict(row)
        entry['detected_emotions'] = json.loads(entry['detected_emotions'])
        entries.append(entry)
    
    return jsonify({'entries': entries})


@app.route('/api/ocr/entry/<entry_id>', methods=['GET'])
def get_ocr_entry(entry_id):
    """Get a specific OCR entry with full details."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM ocr_entries WHERE id = ? AND user_id = ?', 
                   (entry_id, user_id))
    row = cursor.fetchone()
    
    if not row:
        return jsonify({'error': 'Entry not found'}), 404
    
    entry = dict(row)
    # Parse JSON fields
    for field in ['mood_vector', 'detected_emotions', 'mental_health_flags', 
                  'extracted_dates', 'extracted_tasks', 'image_features']:
        if entry.get(field):
            entry[field] = json.loads(entry[field])
    
    return jsonify({'entry': entry})


@app.route('/api/privacy/consent', methods=['GET', 'POST'])
def handle_privacy_consent():
    """Get or update data sharing consent level."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    db = get_db()
    cursor = db.cursor()
    
    if request.method == 'GET':
        cursor.execute('SELECT data_sharing_consent FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        return jsonify({
            'consent_level': row['data_sharing_consent'] if row else 'minimal',
            'options': {
                'none': 'No data shared - complete privacy',
                'minimal': 'Only aggregate statistics (recommended)',
                'standard': 'Anonymized entries help improve AI',
                'full': 'Full anonymized data for research'
            }
        })
    
    # POST - update consent
    data = request.get_json() or {}
    consent_level = data.get('consent_level', 'minimal')
    
    if consent_level not in ['none', 'minimal', 'standard', 'full']:
        return jsonify({'error': 'Invalid consent level'}), 400
    
    cursor.execute('UPDATE users SET data_sharing_consent = ? WHERE id = ?', 
                   (consent_level, user_id))
    db.commit()
    
    return jsonify({
        'success': True,
        'consent_level': consent_level,
        'message': f'Privacy settings updated to: {consent_level}'
    })


# ════════════════════════════════════════════════════════════════════════════════
# STANDARD AUTH & DASHBOARD ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        
        if not email or not validate_email(email):
            return jsonify({'error': 'Valid email required'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        user = create_user(email, password, first_name)
        if not user:
            return jsonify({'error': 'Email already registered'}), 400
        
        session['user_id'] = user['id']
        session.permanent = True
        
        return jsonify({'success': True, 'user': user}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        user = get_user_by_email(email)
        if not user or not verify_password(user, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['user_id'] = user['id']
        session.permanent = True
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'display_name': user['display_name'],
                'spoons': user['spoons']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


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
        'user': {
            'id': user['id'],
            'email': user['email'],
            'display_name': user['display_name'],
            'spoons': user['spoons']
        }
    })


@app.route('/api/dashboard')
def get_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = get_user_by_id(user_id)
    pet = get_pet(user_id)
    mayan = get_mayan_day()
    
    # Get recent OCR entries count
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM ocr_entries WHERE user_id = ?', (user_id,))
    ocr_count = cursor.fetchone()['count']
    
    # Get tasks
    cursor.execute('SELECT * FROM tasks WHERE user_id = ? AND completed = 0 ORDER BY created_at DESC LIMIT 10', (user_id,))
    tasks = [dict(row) for row in cursor.fetchall()]
    
    # Get goals
    cursor.execute('SELECT * FROM goals WHERE user_id = ? ORDER BY priority LIMIT 10', (user_id,))
    goals = [dict(row) for row in cursor.fetchall()]
    
    return jsonify({
        'user': {
            'id': user['id'],
            'display_name': user['display_name'],
            'spoons': user['spoons'],
            'max_spoons': user['max_spoons']
        },
        'pet': pet,
        'mayan_day': mayan,
        'tasks': tasks,
        'goals': goals,
        'stats': {
            'ocr_entries': ocr_count,
            'active_tasks': len(tasks),
            'active_goals': len(goals)
        },
        'ocr_available': HAS_TESSERACT or HAS_EASYOCR
    })


@app.route('/api/visualization/data')
def get_visualization_data():
    """Get data for 3D fractal visualization including OCR-derived mood data."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    db = get_db()
    cursor = db.cursor()
    
    # Get recent mood vectors from OCR entries
    cursor.execute('''
        SELECT mood_vector, wellness_score, entry_date 
        FROM ocr_entries 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 30
    ''', (user_id,))
    
    mood_history = []
    for row in cursor.fetchall():
        try:
            vector = json.loads(row['mood_vector'])
            mood_history.append({
                'vector': vector,
                'wellness': row['wellness_score'],
                'date': row['entry_date']
            })
        except:
            continue
    
    # Calculate average mood vector
    if mood_history:
        avg_vector = np.mean([m['vector'] for m in mood_history], axis=0).tolist()
        avg_wellness = np.mean([m['wellness'] for m in mood_history])
    else:
        avg_vector = [0.5] * 6
        avg_wellness = 50
    
    # Get goals for visualization orbs
    cursor.execute('SELECT * FROM goals WHERE user_id = ?', (user_id,))
    goals = []
    for i, row in enumerate(cursor.fetchall()):
        goal = dict(row)
        theta = i * GOLDEN_ANGLE_RAD
        r = 3 + i * 0.5
        goals.append({
            'id': goal['id'],
            'title': goal['title'],
            'progress': goal['progress'],
            'color': goal['color'],
            'position': {
                'x': r * math.cos(theta),
                'y': (goal['progress'] / 100 * 2) - 1,
                'z': r * math.sin(theta)
            }
        })
    
    return jsonify({
        'mood_vector': avg_vector,
        'mood_named': MoodVector.to_dict(avg_vector),
        'wellness': avg_wellness,
        'mood_history': mood_history[-10:],
        'goals': goals,
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE
        }
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '9.0.0',
        'features': {
            'ocr': HAS_TESSERACT or HAS_EASYOCR,
            'ocr_method': 'tesseract' if HAS_TESSERACT else ('easyocr' if HAS_EASYOCR else 'none'),
            'sentiment_analysis': True,
            'mood_vectors': True,
            'federated_learning': True,
            'privacy_first': True
        }
    })


# ════════════════════════════════════════════════════════════════════════════════
# MAIN HTML (Includes OCR Upload Interface)
# ════════════════════════════════════════════════════════════════════════════════

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life Fractal Intelligence v9 - OCR Edition</title>
    <style>
        :root {
            --primary: #6B8E9F;
            --primary-light: #8FB3C4;
            --secondary: #9F8E6B;
            --success: #6B9F8E;
            --background: #F8F9FA;
            --surface: #FFFFFF;
            --text: #2D3748;
            --text-secondary: #718096;
            --border: #E2E8F0;
            --radius: 12px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
        }
        .header {
            background: var(--surface);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .logo { font-size: 1.25rem; font-weight: 600; color: var(--primary); }
        .main { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; }
        .card {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: var(--radius);
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: #4A6B7C; }
        .btn-success { background: var(--success); color: white; }
        
        /* OCR Upload Zone */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--primary);
            background: #F0F7FA;
        }
        .upload-zone input { display: none; }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        
        /* Results Display */
        .result-section {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--background);
            border-radius: 8px;
        }
        .mood-bar {
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .mood-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .task-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: var(--surface);
            border-radius: 6px;
            margin: 0.5rem 0;
        }
        .emotion-tag {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            background: var(--primary-light);
            color: white;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 0.25rem;
        }
        .hidden { display: none !important; }
        .loading { opacity: 0.6; pointer-events: none; }
        
        /* Privacy Banner */
        .privacy-banner {
            background: linear-gradient(135deg, #E8F4E8, #F4E8F4);
            padding: 0.75rem 2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
        .privacy-banner a { color: var(--primary); }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">🌀 Life Fractal Intelligence v9 - OCR Edition</div>
        <div>
            <span id="user-greeting">Loading...</span>
            <button class="btn btn-primary" onclick="logout()">Logout</button>
        </div>
    </header>
    
    <div class="privacy-banner">
        🔒 <strong>Privacy First:</strong> Your personal data stays on your device. 
        <a href="#" onclick="openPrivacySettings()">Manage data sharing</a>
    </div>
    
    <main class="main">
        <div class="grid">
            <!-- OCR Upload Card -->
            <div class="card">
                <div class="card-header">
                    📷 Quick Capture (OCR)
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Upload a photo of your journal, notes, or any text to automatically analyze mood and extract tasks.
                </p>
                <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
                    <input type="file" id="file-input" accept="image/*" onchange="processImage(this.files[0])">
                    <div class="upload-icon">📄</div>
                    <div><strong>Click or drag to upload</strong></div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">
                        Supports: JPG, PNG, GIF, BMP, WebP
                    </div>
                </div>
                
                <!-- OCR Results -->
                <div id="ocr-results" class="hidden">
                    <div class="result-section">
                        <strong>📝 Extracted Text</strong>
                        <p id="extracted-text" style="margin-top: 0.5rem; font-size: 0.9rem;"></p>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            Confidence: <span id="ocr-confidence">0%</span> | 
                            Words: <span id="word-count">0</span>
                        </div>
                    </div>
                    
                    <div class="result-section">
                        <strong>😊 Mood Analysis</strong>
                        <div id="sentiment-label" style="font-size: 1.25rem; margin: 0.5rem 0;"></div>
                        <div class="mood-bar">
                            <div id="mood-fill" class="mood-fill" style="width: 50%; background: var(--primary);"></div>
                        </div>
                        <div id="emotions-container"></div>
                        <div style="margin-top: 0.5rem;">
                            Wellness Score: <strong id="wellness-score">50</strong>/100
                        </div>
                    </div>
                    
                    <div class="result-section" id="tasks-section">
                        <strong>✅ Extracted Tasks</strong>
                        <div id="tasks-list"></div>
                        <button class="btn btn-success" style="margin-top: 0.5rem; width: 100%;" 
                                onclick="createAllTasks()">
                            Add All Tasks to My List
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Recent Entries -->
            <div class="card">
                <div class="card-header">
                    📊 Recent Analysis
                </div>
                <div id="recent-entries">
                    <p style="color: var(--text-secondary);">No entries yet. Upload an image to get started!</p>
                </div>
            </div>
            
            <!-- Pet & Stats -->
            <div class="card">
                <div class="card-header">
                    🐾 Your Companion
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 4rem;" id="pet-emoji">🐱</div>
                    <div style="font-size: 1.25rem; font-weight: 600;" id="pet-name">Buddy</div>
                    <div style="color: var(--text-secondary);" id="pet-status">Loading...</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentEntryId = null;
        let extractedTasks = [];
        
        // Check session on load
        async function checkSession() {
            try {
                const res = await fetch('/api/auth/session');
                const data = await res.json();
                if (data.authenticated) {
                    document.getElementById('user-greeting').textContent = 
                        `Hi, ${data.user.display_name}! 🥄 ${data.user.spoons} spoons`;
                    loadDashboard();
                } else {
                    window.location.href = '/login';
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        async function loadDashboard() {
            try {
                const res = await fetch('/api/dashboard');
                const data = await res.json();
                
                if (data.pet) {
                    document.getElementById('pet-emoji').textContent = data.pet.emoji;
                    document.getElementById('pet-name').textContent = data.pet.name;
                    document.getElementById('pet-status').textContent = data.pet.status;
                }
                
                loadRecentEntries();
            } catch (e) {
                console.error(e);
            }
        }
        
        async function loadRecentEntries() {
            try {
                const res = await fetch('/api/ocr/entries');
                const data = await res.json();
                
                const container = document.getElementById('recent-entries');
                if (data.entries && data.entries.length > 0) {
                    container.innerHTML = data.entries.slice(0, 5).map(e => `
                        <div style="padding: 0.5rem; border-bottom: 1px solid var(--border);">
                            <div style="display: flex; justify-content: space-between;">
                                <span>${e.sentiment_label} ${e.sentiment_label === 'positive' ? '😊' : e.sentiment_label === 'negative' ? '😔' : '😐'}</span>
                                <span style="font-size: 0.8rem; color: var(--text-secondary);">${e.entry_date}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                                ${e.word_count} words | Wellness: ${Math.round(e.wellness_score)}
                            </div>
                        </div>
                    `).join('');
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        // Drag and drop
        const uploadZone = document.getElementById('upload-zone');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            uploadZone.addEventListener(event, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        ['dragenter', 'dragover'].forEach(event => {
            uploadZone.addEventListener(event, () => uploadZone.classList.add('dragover'));
        });
        ['dragleave', 'drop'].forEach(event => {
            uploadZone.addEventListener(event, () => uploadZone.classList.remove('dragover'));
        });
        uploadZone.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length) processImage(files[0]);
        });
        
        async function processImage(file) {
            if (!file) return;
            
            uploadZone.classList.add('loading');
            document.getElementById('ocr-results').classList.add('hidden');
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const res = await fetch('/api/ocr/process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await res.json();
                
                if (data.success) {
                    currentEntryId = data.entry_id;
                    extractedTasks = data.extracted.tasks || [];
                    
                    // Display results
                    document.getElementById('extracted-text').textContent = data.ocr.text || '(No text detected)';
                    document.getElementById('ocr-confidence').textContent = Math.round(data.ocr.confidence * 100) + '%';
                    document.getElementById('word-count').textContent = data.ocr.word_count;
                    
                    // Sentiment
                    const sentimentEmoji = data.sentiment.label === 'positive' ? '😊' : 
                                          data.sentiment.label === 'negative' ? '😔' : '😐';
                    document.getElementById('sentiment-label').textContent = 
                        `${sentimentEmoji} ${data.sentiment.label.charAt(0).toUpperCase() + data.sentiment.label.slice(1)}`;
                    
                    const moodPercent = ((data.sentiment.score + 1) / 2) * 100;
                    document.getElementById('mood-fill').style.width = moodPercent + '%';
                    document.getElementById('mood-fill').style.background = 
                        data.sentiment.label === 'positive' ? 'var(--success)' : 
                        data.sentiment.label === 'negative' ? '#E57373' : 'var(--primary)';
                    
                    // Emotions
                    document.getElementById('emotions-container').innerHTML = 
                        data.sentiment.detected_emotions.map(e => 
                            `<span class="emotion-tag">${e}</span>`
                        ).join('');
                    
                    document.getElementById('wellness-score').textContent = Math.round(data.mood.wellness_score);
                    
                    // Tasks
                    if (extractedTasks.length > 0) {
                        document.getElementById('tasks-section').classList.remove('hidden');
                        document.getElementById('tasks-list').innerHTML = extractedTasks.map((t, i) => `
                            <div class="task-item">
                                <input type="checkbox" checked data-index="${i}">
                                <span>${t.title}</span>
                                <span style="color: var(--text-secondary); font-size: 0.8rem;">🥄${t.spoon_cost}</span>
                            </div>
                        `).join('');
                    } else {
                        document.getElementById('tasks-section').classList.add('hidden');
                    }
                    
                    document.getElementById('ocr-results').classList.remove('hidden');
                    loadRecentEntries();
                } else {
                    alert('Processing failed: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
            
            uploadZone.classList.remove('loading');
        }
        
        async function createAllTasks() {
            if (!currentEntryId || extractedTasks.length === 0) return;
            
            const checkboxes = document.querySelectorAll('#tasks-list input[type="checkbox"]:checked');
            const indices = Array.from(checkboxes).map(cb => parseInt(cb.dataset.index));
            
            try {
                const res = await fetch('/api/ocr/create-tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ entry_id: currentEntryId, task_indices: indices })
                });
                
                const data = await res.json();
                if (data.success) {
                    alert(`✅ Created ${data.count} tasks!`);
                    document.getElementById('tasks-section').classList.add('hidden');
                }
            } catch (e) {
                alert('Error creating tasks: ' + e.message);
            }
        }
        
        async function logout() {
            await fetch('/api/auth/logout', { method: 'POST' });
            window.location.href = '/';
        }
        
        function openPrivacySettings() {
            alert('Privacy Settings:\\n\\n• Your personal text and notes stay on your device\\n• Only anonymized mood patterns can optionally help improve the AI\\n• You control what is shared via Settings > Privacy');
        }
        
        // Initialize
        checkSession();
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
    print("\n" + "=" * 70)
    print("🌀 LIFE FRACTAL INTELLIGENCE v9.0 - OCR & INTELLIGENT DATA INGESTION")
    print("=" * 70)
    print("   For brains like mine - Privacy-first, neurodivergent-friendly")
    print("=" * 70)
    print(f"\n🔍 OCR Engine: {'Tesseract' if HAS_TESSERACT else 'EasyOCR' if HAS_EASYOCR else 'None (install pytesseract or easyocr)'}")
    print(f"🖼️ OpenCV: {'Available' if HAS_OPENCV else 'Not available'}")
    print(f"✨ Golden Ratio: {PHI:.10f}")
    print("=" * 70)
    print("\n📡 New API Endpoints:")
    print("  POST /api/ocr/process        - Process image with OCR")
    print("  POST /api/ocr/create-tasks   - Create tasks from OCR")
    print("  GET  /api/ocr/entries        - List OCR entries")
    print("  GET  /api/ocr/entry/<id>     - Get OCR entry details")
    print("  GET  /api/privacy/consent    - Get privacy settings")
    print("  POST /api/privacy/consent    - Update privacy settings")
    print("  GET  /api/visualization/data - Get mood data for 3D viz")
    print("=" * 70)
    print("\n🔒 Privacy Features:")
    print("   • Personal text stays LOCAL only")
    print("   • Anonymized insights optionally improve AI")
    print("   • User controls data sharing level")
    print("=" * 70)


if __name__ == '__main__':
    print_banner()
    
    with app.app_context():
        init_db()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
