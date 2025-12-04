# Life Fractal Intelligence v11.0 - Evolved Mathematical Organism
## Deployment Guide & Change Summary

---

## 🎯 CRITICAL FIX: Worker Timeout Issue

### Problem (from error logs):
```
WORKER TIMEOUT (pid:54)
Worker (pid:53) was sent SIGKILL! Perhaps out of memory?
```

The server-side ray marching algorithm for 3D Mandelbulb fractals was causing:
- CPU-intensive computation (thousands of iterations per pixel)
- Gunicorn worker timeouts (30 second limit)
- Memory exhaustion on Render's free tier

### Solution:
**Moved 3D fractal rendering to client-side WebGL/Three.js**

Instead of server-side ray marching:
```python
# OLD (crashed servers):
def _ray_march_mandelbulb(self, x, y, power):
    for _ in range(100):  # Too expensive!
        ...
```

Now the server only provides parameters:
```python
# NEW (lightweight):
def get_3d_parameters(self, wellness_data):
    return {
        'type': 'mandelbulb',
        'power': 8.0 + energy * 4.0,
        'render_mode': 'webgl'  # Client handles rendering
    }
```

The Three.js client renders 3D fractals using GPU acceleration.

---

## 🧬 New Mathematical Algorithm Collection

### 1. Karma-Dharma Scoring Engine
```python
# Core equation: K = Intention × Action × Awareness × φ
karmic_weight = base_weight * harmonic_amplification * (1 + abs(spin) * 0.1)
```

Features:
- `KarmicVector` - Multidimensional karma representation
- `KarmicField` - Superposition of all vectors
- `DharmicPath` - Alignment with cosmic order (D = cos(θ))
- Conservation model: ΣActions = ΣConsequences
- Feedback loop: K(t+1) = K(t) + f(action) - g(consequence)

### 2. Swarm Intelligence System
```python
# Boid rules + karma-dharma
SEPARATION_WEIGHT = 1.5 * PHI_INVERSE
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = PHI_INVERSE
KARMA_ATTRACTION_WEIGHT = 0.5
```

Features:
- Agent roles: Scout, Worker, Leader, Messenger, Guardian, Healer
- Stigmergy field (pheromone trails)
- Particle swarm optimization for planning
- Karma-weighted collective behavior

### 3. Biological Orb System
```python
# Cell types with binding logic
CellType: STEM, NEURON, MEMORY, SENSOR, EFFECTOR, STRUCTURAL, TRANSPORT
```

Features:
- Mitosis (cell division at energy > 0.8)
- Apoptosis (programmed death at energy < 0.1)
- Golden spiral spawning pattern
- L-system fractal growth
- Cell-cell binding with karma compatibility

### 4. Origami Logic Engine
```python
# Rodrigues' rotation formula for fold transformations
rotation = eye(3) + sin(θ) * K + (1 - cos(θ)) * (K @ K)
```

Features:
- 4D transformation matrices
- Karma-based fold calculations
- Crease pattern generation
- Dimensional projection (4D → 3D → 2D)

### 5. Machine Learning Evolution
```python
# Pattern detection and prediction
ml_engine.record_state(state)
patterns = ml_engine.detect_patterns()
predicted_harmony = ml_engine.predict_harmony(current_state)
```

Features:
- Automatic model training after 50 samples
- Karma trend detection
- Harmony prediction
- Cyclical pattern recognition
- Federated learning ready

---

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout

### Goals & Habits
- `GET /api/goals` - List goals
- `POST /api/goals` - Create goal (earns karma)
- `POST /api/goals/<id>/progress` - Update progress (Fibonacci bonuses)
- `GET /api/habits` - List habits
- `POST /api/habits` - Create habit
- `POST /api/habits/<id>/complete` - Complete habit (streak bonuses)

### Wellness
- `POST /api/wellness/checkin` - Daily check-in (earns karma)
- `GET /api/wellness/today` - Today's wellness data

### Organism & Visualization
- `GET /api/organism/state` - Complete organism state
- `GET /api/organism/visualization` - Three.js visualization data
- `POST /api/organism/action` - Process user action
- `GET /api/visualization/fractal-base64/2d` - 2D fractal image
- `GET /api/visualization/fractal-base64/3d` - 3D parameters (WebGL)

### Pet System
- `GET /api/pet/state` - Pet stats
- `POST /api/pet/interact` - Feed, play, pet, rest

### Analytics
- `GET /api/analytics/patterns` - ML-detected patterns
- `GET /api/analytics/karma-history` - Karma history

### Health
- `GET /api/health` - System health check

---

## 🚀 Deployment Steps

### 1. Update Git Repository
```powershell
cd C:\Users\YourUser\planner
# Replace app.py with life_fractal_evolved_v11.py
copy life_fractal_evolved_v11.py app.py
copy requirements.txt requirements.txt

git add -A
git commit -m "v11.0: Evolved mathematical organism with fixed 3D rendering"
git push origin main
```

### 2. Render Configuration
Ensure your `render.yaml` or settings use:
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
- **Python Version**: 3.11+ (3.13 compatible)
- **Environment Variables**:
  - `SECRET_KEY`: (auto-generated if not set)
  - `PORT`: 10000 (Render default)

### 3. Verify Deployment
Check the health endpoint:
```
GET https://planner-1-pyd9.onrender.com/api/health
```

Should return:
```json
{
  "status": "healthy",
  "version": "11.0",
  "organism_mode": "active",
  "gpu": "disabled",
  "ml": "enabled"
}
```

---

## 📐 Sacred Mathematics Reference

| Constant | Value | Usage |
|----------|-------|-------|
| φ (Phi) | 1.618033988749895 | Karma amplification, positioning |
| φ⁻¹ | 0.618033988749895 | Decay rates, damping |
| Golden Angle | 137.5077640500378° | Orb placement, spiral patterns |
| Fibonacci | 0,1,1,2,3,5,8,13,21,34,55,89... | Milestones, breaks, populations |
| Dharma Frequency | 432 Hz | Audio synthesis base |
| Schumann Resonance | 7.83 Hz | Earth frequency reference |
| Planck Karma | 1e-43 | Minimum karmic unit |

---

## 🧪 Testing

Run the test suite:
```python
python -c "
from life_fractal_evolved_v11 import *

# All mathematical systems
engine = KarmaDharmaEngine()
swarm = SwarmCollective()
tissue = OrganicTissue(engine)
fractal = FractalEngine()
origami = OrigamiLogicEngine()
ml = MLEvolutionEngine()
organism = LivingOrganism()

print('All systems initialized successfully!')
"
```

---

## 🎨 Neurodivergent-Friendly Features

- **Swedish minimalist UI** - Reduced visual noise
- **Autism-safe color palettes** - Muted, non-jarring colors
- **Spoon Theory energy tracking** - Daily energy budgeting
- **Zero typing required** - Slider-based inputs
- **Predictable layouts** - Consistent navigation
- **External visualization** - For aphantasia users
- **Compassionate responses** - No shame messaging

---

## 📝 Version History

- **v6.1**: Original production release
- **v10.1**: Added 3D fractals (server-side - crashed)
- **v11.0**: Evolved mathematical organism
  - Fixed 3D rendering (client-side WebGL)
  - Added Karma-Dharma engine
  - Added Swarm Intelligence
  - Added Biological Orbs
  - Added Origami Logic
  - Added ML Evolution
  - Integrated all mathematical algorithms
