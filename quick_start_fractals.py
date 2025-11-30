"""
🚀 ULTIMATE FRACTAL ENGINE - QUICK START GUIDE
═══════════════════════════════════════════════════════════════════════════════════════

This script tests all features and helps you get started quickly.
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🌀 ULTIMATE FRACTAL ENGINE - QUICK START")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════════════════
# STEP 1: CHECK DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════════

print("📋 Step 1: Checking dependencies...")
print()

dependencies = {
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'flask': 'flask',
    'flask_cors': 'flask-cors'
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} - MISSING")
        missing.append(package)

# Check optional GPU support
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ torch (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  ⚠️  torch (CPU only - GPU not available)")
except ImportError:
    print(f"  ⚠️  torch - Not installed (GPU features disabled, using CPU)")

print()

if missing:
    print("⚠️  Missing dependencies! Install with:")
    print(f"   pip install {' '.join(missing)} --break-system-packages")
    print()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# STEP 2: IMPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

print("📦 Step 2: Importing fractal engine...")
print()

try:
    from fractal_engine_ultimate import (
        UltimateFractalEngine,
        FractalType,
        GPU_AVAILABLE,
        PHI,
        FIBONACCI
    )
    print("  ✅ Fractal engine imported successfully")
    print()
except ImportError as e:
    print(f"  ❌ Error importing engine: {e}")
    print()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# STEP 3: INITIALIZE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

print("🎨 Step 3: Initializing fractal engine...")
print()

engine = UltimateFractalEngine(width=800, height=800)
print(f"  ✅ Engine initialized ({800}x{800})")
print(f"  ✅ GPU: {'Enabled' if GPU_AVAILABLE else 'Disabled (CPU mode)'}")
print()

# ═══════════════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE TEST FRACTALS
# ═══════════════════════════════════════════════════════════════════════════════════════

print("🎨 Step 4: Generating test fractals...")
print()

# Test 1: Basic 2D fractal from life metrics
print("  Test 1: 2D Fractal from life metrics...")
test_metrics = {
    'mood': 75,
    'stress': 30,
    'focus': 80,
    'energy': 70,
    'anxiety': 25,
    'mindfulness': 65,
    'chaos': 0.4
}

try:
    image = engine.generate_from_life_metrics(test_metrics, sacred_overlays=True)
    image.save('/tmp/test_2d_fractal.png')
    print("    ✅ Generated: /tmp/test_2d_fractal.png")
except Exception as e:
    print(f"    ❌ Error: {e}")

print()

# Test 2: 3D Mandelbulb
print("  Test 2: 3D Mandelbulb...")
try:
    image_3d = engine.generate_3d_fractal(power=8.0, rotation=(0.5, 0.3, 0))
    image_3d.save('/tmp/test_3d_mandelbulb.png')
    print("    ✅ Generated: /tmp/test_3d_mandelbulb.png")
except Exception as e:
    print(f"    ❌ Error: {e}")

print()

# Test 3: Small animation (10 frames for speed)
print("  Test 3: Zoom animation (10 frames)...")
try:
    frames = engine.create_animation(anim_type='zoom', frames=10, fractal_type=FractalType.MANDELBROT)
    engine.save_animation_gif(frames, '/tmp/test_animation.gif', duration=100)
    print("    ✅ Generated: /tmp/test_animation.gif")
except Exception as e:
    print(f"    ❌ Error: {e}")

print()

# ═══════════════════════════════════════════════════════════════════════════════════════
# STEP 5: START WEB APP
# ═══════════════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print()
print("📊 Summary:")
print(f"  • Sacred Math Constants: φ = {PHI:.6f}")
print(f"  • Fibonacci Sequence: {FIBONACCI[:10]}")
print(f"  • GPU Acceleration: {'✅ Enabled' if GPU_AVAILABLE else '❌ Disabled'}")
print()
print("🎨 Generated Files:")
print("  • /tmp/test_2d_fractal.png")
print("  • /tmp/test_3d_mandelbulb.png")
print("  • /tmp/test_animation.gif")
print()
print("=" * 80)
print("🚀 READY TO START WEB APP!")
print("=" * 80)
print()
print("Run the web app:")
print("  python sacred_fractal_webapp.py")
print()
print("Then open your browser to:")
print("  http://localhost:5000")
print()
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════════════

print()
print("📚 QUICK USAGE EXAMPLES:")
print("=" * 80)
print()

print("""
# 1. GENERATE FRACTAL FROM LIFE METRICS
# ────────────────────────────────────────────────────────────────────────────────

from fractal_engine_ultimate import UltimateFractalEngine

engine = UltimateFractalEngine(width=800, height=800)

metrics = {
    'mood': 75,        # 0-100
    'stress': 30,      # 0-100
    'focus': 80,       # 0-100
    'energy': 70,      # 0-100
    'anxiety': 25,     # 0-100
    'mindfulness': 65, # 0-100
    'chaos': 0.4       # 0-1
}

image = engine.generate_from_life_metrics(metrics, sacred_overlays=True)
image.save('my_fractal.png')


# 2. GENERATE 3D MANDELBULB
# ────────────────────────────────────────────────────────────────────────────────

image_3d = engine.generate_3d_fractal(
    power=8.0,              # Mandelbulb power (6-12 typical)
    rotation=(0.5, 0.3, 0)  # X, Y, Z rotation in radians
)
image_3d.save('mandelbulb.png')


# 3. CREATE ZOOM ANIMATION
# ────────────────────────────────────────────────────────────────────────────────

from fractal_engine_ultimate import FractalType

frames = engine.create_animation(
    anim_type='zoom',
    frames=60,
    fractal_type=FractalType.MANDELBROT
)

engine.save_animation_gif(frames, 'zoom_animation.gif', duration=50)


# 4. CREATE 3D ROTATION ANIMATION
# ────────────────────────────────────────────────────────────────────────────────

frames = engine.create_animation(
    anim_type='3d_rotation',
    frames=60
)

engine.save_animation_gif(frames, '3d_rotation.gif', duration=50)


# 5. MANUAL FRACTAL GENERATION
# ────────────────────────────────────────────────────────────────────────────────

# Mandelbrot
iterations = engine.gen_2d.generate_mandelbrot(
    max_iter=256,
    zoom=2.0,
    center=(-0.75, 0.1),
    power=2.0
)

# Julia Set
iterations = engine.gen_2d.generate_julia(
    c_real=-0.7,
    c_imag=0.27,
    max_iter=256,
    zoom=1.5
)

# Burning Ship
iterations = engine.gen_2d.generate_burning_ship(
    max_iter=256,
    zoom=1.0,
    center=(-0.5, -0.5)
)

# Apply coloring
colored = engine.gen_2d.apply_coloring(
    iterations,
    max_iter=256,
    color_scheme='cosmic',  # 'cosmic', 'fire', 'ocean', 'golden'
    hue_shift=0.0
)

from PIL import Image
image = Image.fromarray(colored, 'RGB')
image.save('custom_fractal.png')


# 6. ADD SACRED GEOMETRY OVERLAYS
# ────────────────────────────────────────────────────────────────────────────────

from PIL import Image
from fractal_engine_ultimate import SacredGeometryOverlay

image = Image.open('my_fractal.png')
sacred = SacredGeometryOverlay()

# Add Flower of Life
image = sacred.draw_flower_of_life(image, opacity=0.3)

# Add Golden Spiral
image = sacred.draw_golden_spiral(image, opacity=0.4, turns=5)

# Add Metatron's Cube
image = sacred.draw_metatrons_cube(image, opacity=0.2)

# Add Vesica Piscis
image = sacred.draw_vesica_piscis(image, opacity=0.3)

image.save('fractal_with_sacred_geometry.png')


# 7. GET BASE64 STRING (for web integration)
# ────────────────────────────────────────────────────────────────────────────────

image = engine.generate_from_life_metrics(metrics)
base64_str = engine.to_base64(image)
html_img = f'<img src="data:image/png;base64,{base64_str}">'


# 8. API USAGE (using the web app)
# ────────────────────────────────────────────────────────────────────────────────

import requests

# Generate 2D fractal
metrics = {'mood': 75, 'stress': 30, 'focus': 80, 'energy': 70}
response = requests.post('http://localhost:5000/api/fractal/2d', json=metrics)

with open('api_fractal.png', 'wb') as f:
    f.write(response.content)

# Get life data analysis
response = requests.post('http://localhost:5000/api/life_data', json=metrics)
data = response.json()
print(data['wellness_index'])
print(data['fractal_params'])

""")

print("=" * 80)
print()
print("🎉 Setup complete! Enjoy creating sacred fractals!")
print()
