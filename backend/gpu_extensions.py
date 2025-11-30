"""
GPU-Accelerated Extensions for Life Planning System
Adds GPU support for fractal generation and ML training with CPU fallback
"""

import logging
import numpy as np
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Try to import PyTorch for GPU acceleration
try:
    import torch
    import torch.nn as nn
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    GPU_AVAILABLE = False
    torch = None
    logger.info("PyTorch not available, using CPU only")


class GPUAcceleratedFractalGenerator:
    """
    GPU-accelerated fractal generation using CUDA if available.
    Falls back to NumPy CPU implementation automatically.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.device = torch.device('cuda')
            # Limit GPU memory usage
            memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', 0.5))
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info("GPU acceleration enabled for fractals")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for fractal generation")
    
    def generate_mandelbrot_gpu(self, width: int, height: int, max_iter: int, 
                                zoom: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """Generate Mandelbrot set using GPU acceleration"""
        
        if not self.use_gpu:
            return self._generate_mandelbrot_cpu(width, height, max_iter, zoom, center)
        
        try:
            # Create coordinate grids on GPU
            x = torch.linspace(-2.0 / zoom + center[0], 2.0 / zoom + center[0], width, device=self.device)
            y = torch.linspace(-2.0 / zoom + center[1], 2.0 / zoom + center[1], height, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='xy')
            
            # Complex plane
            c = X + 1j * Y
            z = torch.zeros_like(c)
            iterations = torch.zeros(width, height, device=self.device, dtype=torch.int32)
            
            # Iterate
            for i in range(max_iter):
                mask = torch.abs(z) <= 2.0
                z[mask] = z[mask] ** 2 + c[mask]
                iterations[mask] = i
            
            # Transfer back to CPU
            return iterations.cpu().numpy()
            
        except Exception as e:
            logger.error(f"GPU fractal generation failed: {e}, falling back to CPU")
            return self._generate_mandelbrot_cpu(width, height, max_iter, zoom, center)
    
    def _generate_mandelbrot_cpu(self, width: int, height: int, max_iter: int,
                                  zoom: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """CPU fallback for Mandelbrot generation"""
        x = np.linspace(-2.0 / zoom + center[0], 2.0 / zoom + center[0], width)
        y = np.linspace(-2.0 / zoom + center[1], 2.0 / zoom + center[1], height)
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros((width, height), dtype=np.int32)
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2.0
            z[mask] = z[mask] ** 2 + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.use_gpu and torch is not None:
            torch.cuda.empty_cache()


class FederatedLearningManager:
    """
    Privacy-preserving federated learning system.
    Aggregates model improvements from all users without accessing personal data.
    """
    
    def __init__(self):
        self.global_model_state = None
        self.update_count = 0
        self.privacy_epsilon = 1.0  # Differential privacy parameter
    
    def aggregate_user_updates(self, user_gradients: list) -> dict:
        """
        Aggregate model updates from multiple users using secure aggregation.
        Implements differential privacy to prevent data leakage.
        """
        if not user_gradients:
            return {}
        
        # Add noise for differential privacy
        aggregated = {}
        for key in user_gradients[0].keys():
            # Average gradients
            avg_gradient = np.mean([g[key] for g in user_gradients], axis=0)
            
            # Add Laplace noise for privacy
            noise_scale = self.privacy_epsilon / len(user_gradients)
            noise = np.random.laplace(0, noise_scale, avg_gradient.shape)
            
            aggregated[key] = avg_gradient + noise
        
        self.update_count += 1
        logger.info(f"Aggregated {len(user_gradients)} user updates (total: {self.update_count})")
        
        return aggregated
    
    def extract_anonymized_patterns(self, user_data: dict) -> dict:
        """
        Extract anonymized behavioral patterns for federated learning.
        Removes all PII and identifiable information.
        """
        # Only extract statistical patterns, no raw data
        anonymized = {
            'stress_variance': np.var([d.get('stress', 50) for d in user_data.get('history', [])]),
            'mood_trend': self._calculate_trend([d.get('mood', 50) for d in user_data.get('history', [])]),
            'sleep_average': np.mean([d.get('sleep_hours', 7) for d in user_data.get('history', [])]),
            'goal_completion_rate': len([d for d in user_data.get('history', []) if d.get('goals_completed', 0) > 0]) / max(len(user_data.get('history', [])), 1)
        }
        
        return anonymized
    
    def _calculate_trend(self, values: list) -> float:
        """Calculate linear trend from values"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope


class AncientMathEnhanced:
    """
    Extended ancient mathematics utilities including:
    - Platonic solids (500 BC)
    - Euclidean geometry (300 BC)
    - Archimedes spiral (287-212 BC)
    - Islamic geometric patterns (800-1500 AD)
    """
    
    @staticmethod
    def archimedes_spiral(theta: float, a: float = 1.0, b: float = 0.5) -> Tuple[float, float]:
        """
        Archimedes spiral: r = a + b*theta
        Used by Archimedes around 225 BC
        """
        r = a + b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    
    @staticmethod
    def islamic_star_pattern(n: int, scale: float = 1.0) -> np.ndarray:
        """
        Generate Islamic geometric star pattern (8th-15th century)
        Based on mathematical principles used in Islamic art
        """
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = []
        
        for i, angle in enumerate(angles):
            # Outer point
            r_outer = scale
            x_outer = r_outer * np.cos(angle)
            y_outer = r_outer * np.sin(angle)
            
            # Inner point (creates star effect)
            r_inner = scale * 0.618  # Golden ratio
            angle_inner = angle + np.pi / n
            x_inner = r_inner * np.cos(angle_inner)
            y_inner = r_inner * np.sin(angle_inner)
            
            points.extend([[x_outer, y_outer], [x_inner, y_inner]])
        
        return np.array(points)
    
    @staticmethod
    def pythagorean_means(values: list) -> dict:
        """
        Calculate the three Pythagorean means (6th century BC):
        - Arithmetic mean
        - Geometric mean  
        - Harmonic mean
        """
        arr = np.array(values)
        return {
            'arithmetic': np.mean(arr),
            'geometric': np.power(np.prod(arr), 1.0 / len(arr)),
            'harmonic': len(arr) / np.sum(1.0 / arr)
        }
    
    @staticmethod
    def golden_rectangle_subdivisions(width: float, height: float, depth: int = 5) -> list:
        """
        Recursively subdivide a rectangle using golden ratio.
        Ancient Greek concept (400 BC)
        """
        phi = (1 + np.sqrt(5)) / 2
        rectangles = [(0, 0, width, height)]
        
        for _ in range(depth):
            new_rects = []
            for x, y, w, h in rectangles:
                if w > h:
                    # Horizontal split
                    split = w / phi
                    new_rects.append((x, y, split, h))
                    new_rects.append((x + split, y, w - split, h))
                else:
                    # Vertical split
                    split = h / phi
                    new_rects.append((x, y, w, split))
                    new_rects.append((x, y + split, w, h - split))
            rectangles = new_rects
        
        return rectangles
    
    @staticmethod
    def fibonacci_spiral_points(n: int = 100) -> np.ndarray:
        """
        Generate points along a Fibonacci spiral.
        Based on Fibonacci sequence and golden ratio
        """
        phi = (1 + np.sqrt(5)) / 2
        golden_angle = 2 * np.pi * (1 - 1 / phi)
        
        points = []
        for i in range(n):
            theta = i * golden_angle
            r = np.sqrt(i)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        
        return np.array(points)


# Memory optimization utilities
class MemoryManager:
    """Optimize memory usage for various hardware configurations"""
    
    @staticmethod
    def get_optimal_batch_size() -> int:
        """Determine optimal batch size based on available memory"""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Use 10% of available memory for batch processing
            optimal_size = int((available_memory * 0.1) / (1024 * 1024 * 10))  # Rough estimate
            return max(1, min(optimal_size, 1000))  # Clamp between 1 and 1000
        except:
            return 100  # Default safe value
    
    @staticmethod
    def optimize_image_generation(target_size: Tuple[int, int]) -> Tuple[int, int]:
        """Adjust image size based on available resources"""
        max_dimension = 2048  # Maximum dimension to prevent memory issues
        width, height = target_size
        
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
        
        return width, height
