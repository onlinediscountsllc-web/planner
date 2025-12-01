"""
PURE PYTHON MATH ENGINE - ZERO DEPENDENCIES
Replaces numpy with built-in Python math for maximum compatibility
Works on ANY Python version, no external dependencies
"""

import math
import cmath
from typing import List, Tuple, Optional, Union

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_INVERSE = PHI - 1
GOLDEN_ANGLE_RAD = math.radians(137.5077640500378)
PI = math.pi
E = math.e
TAU = 2 * math.pi

FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

# ============================================================================
# ARRAY OPERATIONS (numpy replacement)
# ============================================================================

def linspace(start: float, stop: float, num: int = 50) -> List[float]:
    """Create evenly spaced numbers - replaces numpy.linspace"""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def arange(start: float, stop: float, step: float = 1.0) -> List[float]:
    """Create array with step - replaces numpy.arange"""
    result = []
    current = start
    while current < stop:
        result.append(current)
        current += step
    return result


def zeros(n: int) -> List[float]:
    """Create array of zeros - replaces numpy.zeros"""
    return [0.0] * n


def ones(n: int) -> List[float]:
    """Create array of ones - replaces numpy.ones"""
    return [1.0] * n


def mean(arr: List[float]) -> float:
    """Calculate mean - replaces numpy.mean"""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def std(arr: List[float]) -> float:
    """Calculate standard deviation - replaces numpy.std"""
    if not arr:
        return 0.0
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)


def dot(a: List[float], b: List[float]) -> float:
    """Dot product - replaces numpy.dot"""
    return sum(x * y for x, y in zip(a, b))


def normalize(arr: List[float]) -> List[float]:
    """Normalize array to 0-1 range"""
    if not arr:
        return arr
    min_val = min(arr)
    max_val = max(arr)
    if max_val == min_val:
        return [0.5] * len(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]


# ============================================================================
# FFT (Fast Fourier Transform) - Pure Python
# ============================================================================

def fft(x: List[complex]) -> List[complex]:
    """
    Cooley-Tukey FFT algorithm - Pure Python
    Replaces numpy.fft.fft for executive dysfunction detection
    """
    N = len(x)
    
    # Base case
    if N <= 1:
        return x
    
    # Pad to power of 2 if needed
    if N & (N - 1) != 0:
        next_power = 2 ** math.ceil(math.log2(N))
        x = x + [0] * (next_power - N)
        N = next_power
    
    # Divide
    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])
    
    # Conquer
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    
    return [(even[k] + T[k]) for k in range(N // 2)] + \
           [(even[k] - T[k]) for k in range(N // 2)]


def fft_frequencies(n: int, sample_rate: float = 1.0) -> List[float]:
    """Generate FFT frequency bins"""
    return [i * sample_rate / n for i in range(n // 2)]


def fft_power_spectrum(signal: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculate power spectrum from signal
    Returns (frequencies, magnitudes)
    """
    # Convert to complex
    complex_signal = [complex(x, 0) for x in signal]
    
    # Perform FFT
    spectrum = fft(complex_signal)
    
    # Calculate magnitudes (power)
    N = len(spectrum)
    magnitudes = [abs(spectrum[i]) for i in range(N // 2)]
    frequencies = fft_frequencies(N)
    
    return frequencies, magnitudes


# ============================================================================
# FRACTAL GENERATION - Pure Python
# ============================================================================

def mandelbrot(c: complex, max_iter: int = 100) -> int:
    """
    Calculate Mandelbrot set iteration count
    Pure Python - no PIL/Pillow needed for math calculation
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def julia(z: complex, c: complex, max_iter: int = 100) -> int:
    """Calculate Julia set iteration count"""
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def generate_fractal_data(width: int, height: int, 
                         x_min: float = -2.0, x_max: float = 1.0,
                         y_min: float = -1.5, y_max: float = 1.5,
                         max_iter: int = 50) -> List[List[int]]:
    """
    Generate fractal iteration data as 2D list
    Can be used for visualization or analysis
    """
    data = []
    for y in range(height):
        row = []
        for x in range(width):
            # Map pixel to complex plane
            real = x_min + (x / width) * (x_max - x_min)
            imag = y_min + (y / height) * (y_max - y_min)
            c = complex(real, imag)
            
            # Calculate iterations
            iterations = mandelbrot(c, max_iter)
            row.append(iterations)
        data.append(row)
    return data


# ============================================================================
# GOLDEN RATIO & FIBONACCI FUNCTIONS
# ============================================================================

def fibonacci_up_to(n: int) -> List[int]:
    """Generate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci_at(n: int) -> int:
    """Get nth Fibonacci number using Binet's formula"""
    if n < 2:
        return n
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return int((phi**n - psi**n) / sqrt5)


def golden_ratio_partition(total: float, n: int = 2) -> List[float]:
    """Partition a value using golden ratio"""
    if n == 2:
        return [total * PHI_INVERSE, total * (1 - PHI_INVERSE)]
    
    # For n > 2, use Fibonacci-based partitioning
    fib = fibonacci_up_to(n + 1)
    fib_sum = sum(fib[1:n+1])
    
    if fib_sum == 0:
        return [total / n] * n
    
    return [(total * fib[i]) / fib_sum for i in range(1, n + 1)]


def golden_angle_sequence(n: int) -> List[float]:
    """Generate n angles using golden angle"""
    return [i * GOLDEN_ANGLE_RAD for i in range(n)]


# ============================================================================
# DIFFERENTIAL EQUATIONS SOLVER - Pure Python
# ============================================================================

def euler_method(f, y0: float, t0: float, t_end: float, dt: float) -> List[Tuple[float, float]]:
    """
    Solve ODE using Euler method: dy/dt = f(t, y)
    Returns list of (t, y) tuples
    """
    results = [(t0, y0)]
    t = t0
    y = y0
    
    while t < t_end:
        y = y + dt * f(t, y)
        t = t + dt
        results.append((t, y))
    
    return results


def runge_kutta_4(f, y0: float, t0: float, t_end: float, dt: float) -> List[Tuple[float, float]]:
    """
    Solve ODE using 4th order Runge-Kutta: dy/dt = f(t, y)
    More accurate than Euler method
    """
    results = [(t0, y0)]
    t = t0
    y = y0
    
    while t < t_end:
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt/2, y + k1/2)
        k3 = dt * f(t + dt/2, y + k2/2)
        k4 = dt * f(t + dt, y + k3)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + dt
        results.append((t, y))
    
    return results


# ============================================================================
# POLYNOMIAL FITTING - Pure Python
# ============================================================================

def polynomial_fit(x: List[float], y: List[float], degree: int = 2) -> List[float]:
    """
    Fit polynomial to data using least squares
    Returns coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
    Pure Python implementation - replaces numpy.polyfit
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have same length")
    
    # Build Vandermonde matrix
    A = [[x[i]**j for j in range(degree + 1)] for i in range(n)]
    
    # Solve normal equations: A^T A c = A^T y
    # Using Gaussian elimination
    
    # Compute A^T A
    ATA = [[0] * (degree + 1) for _ in range(degree + 1)]
    for i in range(degree + 1):
        for j in range(degree + 1):
            ATA[i][j] = sum(A[k][i] * A[k][j] for k in range(n))
    
    # Compute A^T y
    ATy = [sum(A[k][i] * y[k] for k in range(n)) for i in range(degree + 1)]
    
    # Solve using Gaussian elimination
    return gaussian_elimination(ATA, ATy)


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination"""
    n = len(b)
    
    # Create augmented matrix
    M = [A[i] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        
        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if M[i][i] == 0:
                continue
            c = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= c * M[i][j]
    
    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if M[i][i] == 0:
            x[i] = 0
            continue
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    
    return x


def polynomial_eval(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial at x"""
    return sum(c * (x ** i) for i, c in enumerate(coeffs))


# ============================================================================
# EXPONENTIAL FITTING - Pure Python
# ============================================================================

def exponential_fit(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Fit exponential function y = a * exp(b * x)
    Returns (a, b)
    Uses log transformation and linear regression
    """
    if not x or not y:
        return (1.0, 0.0)
    
    # Take log of y values (skip non-positive)
    log_y = []
    valid_x = []
    for xi, yi in zip(x, y):
        if yi > 0:
            log_y.append(math.log(yi))
            valid_x.append(xi)
    
    if not log_y:
        return (1.0, 0.0)
    
    # Linear fit to log(y) = log(a) + b*x
    n = len(valid_x)
    sum_x = sum(valid_x)
    sum_log_y = sum(log_y)
    sum_x_log_y = sum(xi * lyi for xi, lyi in zip(valid_x, log_y))
    sum_x2 = sum(xi * xi for xi in valid_x)
    
    # Calculate slope b
    b = (n * sum_x_log_y - sum_x * sum_log_y) / (n * sum_x2 - sum_x * sum_x)
    
    # Calculate intercept log(a)
    log_a = (sum_log_y - b * sum_x) / n
    a = math.exp(log_a)
    
    return (a, b)


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y) or not x:
        return 0.0
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denominator_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    
    return numerator / (denominator_x * denominator_y)


def moving_average(data: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if window <= 0 or not data:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        result.append(mean(window_data))
    
    return result


# ============================================================================
# COLOR GENERATION - Pure Python
# ============================================================================

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """
    Convert HSL to RGB
    h: 0-360, s: 0-100, l: 0-100
    Returns: (r, g, b) each 0-255
    """
    h = h % 360
    s = max(0, min(100, s)) / 100
    l = max(0, min(100, l)) / 100
    
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255)
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color string"""
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# SELF-HEALING WRAPPER
# ============================================================================

class SelfHealingMath:
    """
    Wrapper that provides graceful fallbacks for all math operations
    If any function fails, it returns safe defaults
    """
    
    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """Division with zero check"""
        try:
            return a / b if b != 0 else default
        except:
            return default
    
    @staticmethod
    def safe_sqrt(x: float, default: float = 0.0) -> float:
        """Square root with negative check"""
        try:
            return math.sqrt(max(0, x))
        except:
            return default
    
    @staticmethod
    def safe_log(x: float, default: float = 0.0) -> float:
        """Logarithm with positive check"""
        try:
            return math.log(max(1e-10, x))
        except:
            return default
    
    @staticmethod
    def safe_exp(x: float, default: float = 1.0) -> float:
        """Exponential with overflow check"""
        try:
            return math.exp(min(700, x))  # Prevent overflow
        except:
            return default


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constants
    'PHI', 'PHI_INVERSE', 'GOLDEN_ANGLE_RAD', 'FIBONACCI', 'PI', 'E', 'TAU',
    
    # Array operations
    'linspace', 'arange', 'zeros', 'ones', 'mean', 'std', 'dot', 'normalize',
    
    # FFT
    'fft', 'fft_frequencies', 'fft_power_spectrum',
    
    # Fractals
    'mandelbrot', 'julia', 'generate_fractal_data',
    
    # Golden ratio & Fibonacci
    'fibonacci_up_to', 'fibonacci_at', 'golden_ratio_partition', 'golden_angle_sequence',
    
    # Differential equations
    'euler_method', 'runge_kutta_4',
    
    # Polynomial fitting
    'polynomial_fit', 'polynomial_eval', 'gaussian_elimination',
    
    # Exponential fitting
    'exponential_fit',
    
    # Statistics
    'correlation', 'moving_average',
    
    # Colors
    'hsl_to_rgb', 'rgb_to_hex',
    
    # Self-healing
    'SelfHealingMath'
]
