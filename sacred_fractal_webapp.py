"""
🌀 SACRED FRACTAL WEB APP v3.0
═══════════════════════════════════════════════════════════════════════════════════════
Complete integration of fractal engine with life planning metrics
Features:
- Real-time fractal generation from life data
- 2D and 3D fractals
- Smooth animations
- Sacred geometry overlays
- WebGL preview (future)
- API endpoints for all fractal types
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import io
import json
import math
import random
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS

# Import the ultimate fractal engine
from fractal_engine_ultimate import (
    UltimateFractalEngine,
    FractalType,
    GPU_AVAILABLE,
    GPU_NAME,
    PHI,
    GOLDEN_ANGLE,
    FIBONACCI
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize fractal engine
fractal_engine = UltimateFractalEngine(width=800, height=800)

logger.info("🌀 Sacred Fractal Web App v3.0 initialized")
logger.info(f"   GPU: {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ CPU only'}")


# ═══════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_life_metrics_from_request():
    """Extract life metrics from request or generate random ones"""
    if request.method == 'POST' and request.json:
        data = request.json
        return {
            'mood': data.get('mood', 50),
            'stress': data.get('stress', 50),
            'focus': data.get('focus', 50),
            'energy': data.get('energy', 50),
            'anxiety': data.get('anxiety', 30),
            'mindfulness': data.get('mindfulness', 50),
            'chaos': data.get('chaos', random.uniform(0.2, 0.8))
        }
    else:
        # Generate random realistic metrics
        return {
            'mood': random.randint(30, 90),
            'stress': random.randint(20, 70),
            'focus': random.randint(40, 95),
            'energy': random.randint(30, 90),
            'anxiety': random.randint(10, 60),
            'mindfulness': random.randint(30, 85),
            'chaos': random.uniform(0.2, 0.8)
        }


def calculate_wellness_index(metrics):
    """Calculate overall wellness from metrics"""
    positive = (metrics['mood'] + metrics['focus'] + metrics['energy'] + metrics['mindfulness']) / 4
    negative = (metrics['stress'] + metrics['anxiety']) / 2
    return (positive + (100 - negative)) / 2


# ═══════════════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🌀 Sacred Fractal Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            opacity: 0.9;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .gpu-status {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .controls {
            background: rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .control-group {
            margin-bottom: 20px;
        }
        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 0.95em;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        input[type="range"] {
            flex: 1;
            height: 8px;
            border-radius: 5px;
            background: rgba(255,255,255,0.3);
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        .value-display {
            min-width: 50px;
            text-align: right;
            font-weight: bold;
            font-size: 1.1em;
        }
        .button-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .fractal-display {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        #fractalImage {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }
        .loading {
            font-size: 1.2em;
            padding: 40px;
            text-align: center;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .info-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .info-card h3 {
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .info-card ul {
            list-style: none;
            padding: 0;
        }
        .info-card li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .sacred-math {
            background: rgba(255,215,0,0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid gold;
        }
        .wellness-indicator {
            height: 30px;
            background: linear-gradient(90deg, #ff4444, #ffaa44, #44ff44);
            border-radius: 15px;
            position: relative;
            margin: 10px 0;
        }
        .wellness-marker {
            position: absolute;
            width: 4px;
            height: 100%;
            background: white;
            box-shadow: 0 0 10px rgba(255,255,255,0.8);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌀 Sacred Fractal Generator</h1>
        <p class="subtitle">AI-Powered Visualization of Your Life Energy</p>
        
        <div class="gpu-status">
            <strong>System Status:</strong> 
            <span id="gpuStatus">{{ 'GPU Accelerated (' + gpu_name + ')' if gpu_available else 'CPU Mode' }}</span>
            <br>
            <span style="opacity: 0.8; font-size: 0.9em;">
                Golden Ratio (φ): {{ phi }} | Fibonacci Depth: 13
            </span>
        </div>
        
        <div class="controls">
            <h2 style="margin-bottom: 20px;">✨ Life Metrics</h2>
            
            <div class="control-group">
                <label>😊 Mood (Emotional State)</label>
                <div class="slider-container">
                    <input type="range" id="mood" min="0" max="100" value="70" oninput="updateValue('mood')">
                    <span class="value-display" id="moodValue">70</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>⚡ Energy Level</label>
                <div class="slider-container">
                    <input type="range" id="energy" min="0" max="100" value="65" oninput="updateValue('energy')">
                    <span class="value-display" id="energyValue">65</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>🎯 Focus & Clarity</label>
                <div class="slider-container">
                    <input type="range" id="focus" min="0" max="100" value="75" oninput="updateValue('focus')">
                    <span class="value-display" id="focusValue">75</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>😰 Stress Level</label>
                <div class="slider-container">
                    <input type="range" id="stress" min="0" max="100" value="35" oninput="updateValue('stress')">
                    <span class="value-display" id="stressValue">35</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>😨 Anxiety</label>
                <div class="slider-container">
                    <input type="range" id="anxiety" min="0" max="100" value="30" oninput="updateValue('anxiety')">
                    <span class="value-display" id="anxietyValue">30</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>🧘 Mindfulness</label>
                <div class="slider-container">
                    <input type="range" id="mindfulness" min="0" max="100" value="60" oninput="updateValue('mindfulness')">
                    <span class="value-display" id="mindfulnessValue">60</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>🌀 Wellness Index</label>
                <div class="wellness-indicator">
                    <div class="wellness-marker" id="wellnessMarker"></div>
                </div>
                <div style="text-align: center; font-size: 1.3em; font-weight: bold; margin-top: 5px;">
                    <span id="wellnessValue">70</span>%
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="generateFractal('2d')">🎨 Generate 2D Fractal</button>
                <button onclick="generateFractal('3d')">🔮 Generate 3D Fractal</button>
                <button onclick="randomMetrics()">🎲 Random Metrics</button>
                <button onclick="downloadFractal()">💾 Download Image</button>
            </div>
        </div>
        
        <div class="fractal-display">
            <div id="loadingText" class="loading">Click "Generate Fractal" to create your visualization</div>
            <img id="fractalImage" style="display: none;" alt="Fractal Visualization">
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>🎨 Fractal Types</h3>
                <ul>
                    <li><strong>Wellness &lt; 30%:</strong> Burning Ship (Fire)</li>
                    <li><strong>30-50%:</strong> Julia Set (Ocean)</li>
                    <li><strong>50-70%:</strong> Mandelbrot (Cosmic)</li>
                    <li><strong>70-100%:</strong> Golden Mandelbrot</li>
                </ul>
            </div>
            
            <div class="info-card">
                <h3>✨ Sacred Overlays</h3>
                <ul>
                    <li><strong>Wellness &gt; 60%:</strong> Flower of Life</li>
                    <li><strong>Focus &gt; 70%:</strong> Golden Spiral</li>
                    <li><strong>Mood &gt; 80%:</strong> Metatron's Cube</li>
                </ul>
            </div>
            
            <div class="info-card">
                <h3>🔢 Sacred Mathematics</h3>
                <div class="sacred-math">
                    <strong>φ (Phi):</strong> {{ phi }}<br>
                    <strong>Golden Angle:</strong> {{ golden_angle }}°<br>
                    <strong>Fibonacci:</strong> 1, 1, 2, 3, 5, 8, 13...<br>
                    <strong>Zoom:</strong> 1 + (focus/100) × φ × 10
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentImageUrl = '';
        
        function updateValue(id) {
            const value = document.getElementById(id).value;
            document.getElementById(id + 'Value').textContent = value;
            updateWellness();
        }
        
        function updateWellness() {
            const mood = parseInt(document.getElementById('mood').value);
            const energy = parseInt(document.getElementById('energy').value);
            const focus = parseInt(document.getElementById('focus').value);
            const stress = parseInt(document.getElementById('stress').value);
            const anxiety = parseInt(document.getElementById('anxiety').value);
            const mindfulness = parseInt(document.getElementById('mindfulness').value);
            
            const positive = (mood + energy + focus + mindfulness) / 4;
            const negative = (stress + anxiety) / 2;
            const wellness = Math.round((positive + (100 - negative)) / 2);
            
            document.getElementById('wellnessValue').textContent = wellness;
            document.getElementById('wellnessMarker').style.left = wellness + '%';
        }
        
        function getMetrics() {
            return {
                mood: parseInt(document.getElementById('mood').value),
                energy: parseInt(document.getElementById('energy').value),
                focus: parseInt(document.getElementById('focus').value),
                stress: parseInt(document.getElementById('stress').value),
                anxiety: parseInt(document.getElementById('anxiety').value),
                mindfulness: parseInt(document.getElementById('mindfulness').value),
                chaos: Math.random() * 0.6 + 0.2
            };
        }
        
        async function generateFractal(type = '2d') {
            const loadingText = document.getElementById('loadingText');
            const fractalImage = document.getElementById('fractalImage');
            
            loadingText.style.display = 'block';
            loadingText.textContent = '🌀 Generating fractal from your life energy...';
            fractalImage.style.display = 'none';
            
            const metrics = getMetrics();
            
            try {
                const endpoint = type === '3d' ? '/api/fractal/3d' : '/api/fractal/2d';
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(metrics)
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    currentImageUrl = URL.createObjectURL(blob);
                    fractalImage.src = currentImageUrl;
                    fractalImage.style.display = 'block';
                    loadingText.style.display = 'none';
                } else {
                    loadingText.textContent = '❌ Error generating fractal';
                }
            } catch (error) {
                loadingText.textContent = '❌ Connection error: ' + error.message;
            }
        }
        
        function randomMetrics() {
            document.getElementById('mood').value = Math.floor(Math.random() * 60) + 30;
            document.getElementById('energy').value = Math.floor(Math.random() * 60) + 30;
            document.getElementById('focus').value = Math.floor(Math.random() * 60) + 30;
            document.getElementById('stress').value = Math.floor(Math.random() * 60) + 20;
            document.getElementById('anxiety').value = Math.floor(Math.random() * 50) + 20;
            document.getElementById('mindfulness').value = Math.floor(Math.random() * 60) + 30;
            
            ['mood', 'energy', 'focus', 'stress', 'anxiety', 'mindfulness'].forEach(updateValue);
        }
        
        function downloadFractal() {
            if (currentImageUrl) {
                const a = document.createElement('a');
                a.href = currentImageUrl;
                a.download = 'sacred_fractal_' + Date.now() + '.png';
                a.click();
            } else {
                alert('Generate a fractal first!');
            }
        }
        
        // Initialize
        updateWellness();
    </script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main dashboard"""
    return render_template_string(
        DASHBOARD_HTML,
        gpu_available=GPU_AVAILABLE,
        gpu_name=GPU_NAME or 'N/A',
        phi=round(PHI, 15),
        golden_angle=round(GOLDEN_ANGLE, 10)
    )


@app.route('/api/system/info')
def system_info():
    """Get system information"""
    return jsonify({
        'status': 'online',
        'version': '3.0',
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'sacred_math': {
            'phi': PHI,
            'phi_inverse': PHI - 1,
            'golden_angle': GOLDEN_ANGLE,
            'golden_angle_rad': GOLDEN_ANGLE * math.pi / 180,
            'fibonacci': FIBONACCI[:13]
        },
        'fractal_types': [ft.value for ft in FractalType],
        'color_schemes': ['cosmic', 'fire', 'ocean', 'golden']
    })


@app.route('/api/life_data', methods=['GET', 'POST'])
def life_data():
    """Get current life metrics and fractal parameters"""
    metrics = get_life_metrics_from_request()
    wellness = calculate_wellness_index(metrics)
    
    # Calculate fractal parameters using sacred math
    zoom = 1.0 + (metrics['focus'] / 100) * PHI * 10
    max_iter = int(100 + metrics['mood'] * 2)
    
    return jsonify({
        'metrics': metrics,
        'wellness_index': round(wellness, 2),
        'fractal_params': {
            'zoom': round(zoom, 3),
            'max_iterations': max_iter,
            'chaos_factor': round(metrics['chaos'], 3)
        },
        'sacred_overlays': {
            'flower_of_life': wellness > 60,
            'golden_spiral': metrics['focus'] > 70,
            'metatrons_cube': metrics['mood'] > 80
        },
        'recommended_type': (
            'burning_ship' if wellness < 30 else
            'julia' if wellness < 50 else
            'mandelbrot' if wellness < 70 else
            'mandelbrot_golden'
        )
    })


@app.route('/api/fractal/2d', methods=['POST'])
def generate_2d_fractal():
    """Generate 2D fractal from life metrics"""
    try:
        metrics = get_life_metrics_from_request()
        logger.info(f"Generating 2D fractal with metrics: {metrics}")
        
        # Generate fractal
        image = fractal_engine.generate_from_life_metrics(metrics, sacred_overlays=True)
        
        # Return as PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error generating 2D fractal: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fractal/3d', methods=['POST'])
def generate_3d_fractal():
    """Generate 3D Mandelbulb fractal"""
    try:
        metrics = get_life_metrics_from_request()
        logger.info(f"Generating 3D fractal with metrics: {metrics}")
        
        # Map metrics to 3D parameters
        power = 6.0 + (metrics['mood'] / 100) * 4.0  # 6-10
        rotation_y = (metrics['focus'] / 100) * math.pi
        rotation_x = (metrics['energy'] / 100) * math.pi / 2
        
        # Generate 3D fractal
        image = fractal_engine.generate_3d_fractal(
            power=power,
            rotation=(rotation_x, rotation_y, 0)
        )
        
        # Return as PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error generating 3D fractal: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fractal/animation/zoom', methods=['POST'])
def generate_zoom_animation():
    """Generate zoom animation (returns info, actual generation happens async)"""
    try:
        data = request.json or {}
        frames = min(data.get('frames', 30), 60)  # Max 60 frames
        fractal_type = FractalType(data.get('type', 'mandelbrot'))
        
        logger.info(f"Generating {frames}-frame zoom animation")
        
        # Generate animation
        images = fractal_engine.create_animation(
            anim_type='zoom',
            frames=frames,
            fractal_type=fractal_type
        )
        
        # Save as GIF
        filename = f'/tmp/fractal_zoom_{datetime.now().timestamp()}.gif'
        fractal_engine.save_animation_gif(images, filename, duration=50)
        
        return send_file(filename, mimetype='image/gif')
        
    except Exception as e:
        logger.error(f"Error generating animation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fractal/base64', methods=['POST'])
def generate_fractal_base64():
    """Generate fractal and return as base64 string"""
    try:
        metrics = get_life_metrics_from_request()
        image = fractal_engine.generate_from_life_metrics(metrics, sacred_overlays=True)
        
        base64_str = fractal_engine.to_base64(image)
        
        return jsonify({
            'image': f'data:image/png;base64,{base64_str}',
            'metrics': metrics,
            'wellness': calculate_wellness_index(metrics)
        })
        
    except Exception as e:
        logger.error(f"Error generating base64 fractal: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌀 SACRED FRACTAL WEB APP v3.0")
    print("=" * 70)
    print(f"✨ Golden Ratio (φ): {PHI:.15f}")
    print(f"🌻 Golden Angle: {GOLDEN_ANGLE:.10f}°")
    print(f"🔢 Fibonacci: {FIBONACCI[:10]}...")
    print(f"🖥️  GPU: {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ CPU only'}")
    print("=" * 70)
    print("\n📡 API Endpoints:")
    print("  Dashboard:      GET  /")
    print("  System Info:    GET  /api/system/info")
    print("  Life Data:      POST /api/life_data")
    print("  2D Fractal:     POST /api/fractal/2d")
    print("  3D Fractal:     POST /api/fractal/3d")
    print("  Zoom Animation: POST /api/fractal/animation/zoom")
    print("  Base64 Image:   POST /api/fractal/base64")
    print("=" * 70)
    print(f"\n🚀 Starting server at http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
