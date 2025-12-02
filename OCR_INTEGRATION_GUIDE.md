# 🌀 Life Fractal Intelligence v9.0 - OCR & Intelligent Data Ingestion

## For brains like mine - Privacy-first, neurodivergent-friendly

---

## 🆕 What's New in v9.0

### 📷 OCR Text Recognition
Upload photos of your journal entries, notes, screenshots, or any text and the system will:
- Extract text automatically using Tesseract or EasyOCR
- Work with handwritten and printed text
- Process multiple image formats (JPG, PNG, GIF, BMP, WebP)

### 😊 Automatic Sentiment Analysis
Every piece of text is analyzed for:
- **Sentiment Score** (-1.0 to +1.0)
- **Mood Classification** (positive, neutral, negative)
- **Detected Emotions** (anxiety, joy, burnout, calm, etc.)
- **Neurodivergent-specific patterns** (executive dysfunction, sensory overload, etc.)

### 📊 Six-Dimensional Mood Vectors
Your emotional state is mapped to a mathematical vector:
- **Calm** ← → Anxious
- **Hopeful** ← → Tired  
- **Focused** ← → Scattered

These vectors use the **Golden Ratio (φ)** for beautiful, natural weighting.

### ✅ Auto-Extract Tasks
The system recognizes task patterns in your text:
- TODO lists
- Bullet points
- Numbered lists
- "Need to...", "Have to...", "Remember to..." phrases

Each task is assigned a **spoon cost** based on complexity.

### 🔒 Privacy-First Architecture
Your personal data stays **LOCAL**:
- Raw text is stored only on your device
- No personal content is ever uploaded
- You control what (if anything) is shared

### 🤖 Federated Learning (Optional)
If you choose to help improve the AI:
- Only **anonymized, aggregated** data is shared
- No personal text, names, or specific dates
- Four consent levels: None, Minimal, Standard, Full

---

## 📡 New API Endpoints

### OCR Processing

```
POST /api/ocr/process
```
Upload an image for OCR processing.

**Request:** `multipart/form-data` with `image` file

**Response:**
```json
{
  "success": true,
  "entry_id": "ocr_abc123",
  "ocr": {
    "text": "Extracted text content...",
    "confidence": 0.85,
    "method": "tesseract",
    "word_count": 42
  },
  "sentiment": {
    "score": 0.35,
    "label": "positive",
    "detected_emotions": ["calm", "hope"]
  },
  "mood": {
    "vector": [0.65, 0.15, 0.72, 0.28, 0.25, 0.75],
    "vector_named": {
      "calm": 0.65,
      "anxious": 0.15,
      "hopeful": 0.72,
      "tired": 0.28,
      "scattered": 0.25,
      "focused": 0.75
    },
    "wellness_score": 68.5,
    "flags": ["calm_state", "positive_outlook"]
  },
  "extracted": {
    "dates": [{"date": "2024-01-15", "raw": "January 15, 2024"}],
    "tasks": [
      {"title": "Call doctor", "spoon_cost": 2},
      {"title": "Buy groceries", "spoon_cost": 1}
    ]
  }
}
```

### Create Tasks from OCR

```
POST /api/ocr/create-tasks
```
Create tasks from an OCR entry.

**Request:**
```json
{
  "entry_id": "ocr_abc123",
  "task_indices": [0, 1]  // Optional: which tasks to create
}
```

### Get OCR Entries

```
GET /api/ocr/entries
```
Get list of user's OCR entries.

### Get Single Entry

```
GET /api/ocr/entry/<entry_id>
```
Get full details of an OCR entry.

### Privacy Consent

```
GET /api/privacy/consent
POST /api/privacy/consent
```
View or update data sharing preferences.

---

## 🧠 Sentiment Analysis Details

### Neurodivergent-Aware Vocabulary

The sentiment engine includes specialized vocabulary for:

**Positive States:**
- hyperfocus, flow, special interest
- regulated, grounded, safe
- accommodated, supported, validated
- masking-free, authentic, sensory-friendly

**Challenging States:**
- meltdown, shutdown, burnout
- overstimulated, sensory overload
- executive dysfunction, paralysis
- masking, depleted, no spoons

### Emotion Detection Categories

| Emotion | Trigger Words |
|---------|--------------|
| Anxiety | anxious, worried, nervous, panic, fear |
| Burnout | exhausted, depleted, drained, overwhelmed |
| Joy | happy, excited, thrilled, delighted |
| Calm | peaceful, relaxed, serene, content |
| Executive Dysfunction | forgot, late, stuck, frozen, scattered |
| Sensory Overload | loud, bright, overwhelming, overstimulated |

---

## 📐 Mathematical Foundation

### Mood Vector Computation

The six-dimensional mood vector uses the **Golden Ratio (φ ≈ 1.618)** for natural weighting:

```python
calm = max(0.0, sentiment * PHI_INVERSE + 0.3)
anxious = max(0.0, -sentiment * PHI_INVERSE + 0.2)
hopeful = (sentiment + 1.0) / 2.0
tired = 1.0 - hopeful
scattered = abs(sentiment) * PHI_INVERSE
focused = 1.0 - scattered
```

### Wellness Score

Uses **Fibonacci weighting** for positive traits:

```python
weights = [Fib(8), Fib(5), Fib(7), Fib(4), Fib(3), Fib(6)]
       = [21, 5, 13, 3, 2, 8]
```

Positive traits (calm, hopeful, focused) contribute more to wellness.

---

## 🔒 Privacy & Data Sharing

### Four Consent Levels

| Level | What's Shared |
|-------|--------------|
| **None** | Nothing - complete privacy |
| **Minimal** | Only aggregate statistics (avg sentiment, common emotions) |
| **Standard** | Anonymized individual entries (no text, no dates) |
| **Full** | Full anonymized data with emotion patterns |

### What's NEVER Shared

- Raw text content
- Personal names
- Specific dates (only month buckets)
- Any identifying information
- Images

### Anonymization Process

```python
anonymized = {
    'timestamp_bucket': '2024-01',  # Only month
    'sentiment_score': 0.35,
    'mood_vector': [0.65, 0.15, ...],
    'wellness_score': 68.5,
    'detected_emotions': ['calm', 'hope'],
    # NO personal text
    # NO specific dates
    # NO identifiers
}
```

---

## 🚀 Deployment

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements_v9.txt

# For Tesseract OCR (recommended)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Run Locally

```bash
python life_fractal_v9_with_ocr.py
# Open http://localhost:5000
```

### Deploy to Render

1. Rename file to `app.py`
2. Update `requirements.txt`
3. Add to `render.yaml`:
```yaml
services:
  - type: web
    name: life-fractal-v9
    env: python
    buildCommand: |
      apt-get update && apt-get install -y tesseract-ocr
      pip install -r requirements_v9.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## 🎯 Use Cases

### 1. Journal Entry Processing
Take a photo of your handwritten journal → Get mood analysis and wellness score

### 2. Task Extraction
Photograph a to-do list → Automatically create tasks with spoon costs

### 3. Mood Tracking Over Time
Upload daily notes → Build a visual mood timeline in the 3D fractal

### 4. Pattern Recognition
Over time, the system identifies your emotional patterns and triggers

---

## 🧩 Integration with 3D Visualization

The `/api/visualization/data` endpoint now includes OCR-derived mood data:

```json
{
  "mood_vector": [0.55, 0.25, 0.68, 0.32, 0.30, 0.70],
  "mood_named": {
    "calm": 0.55,
    "anxious": 0.25,
    "hopeful": 0.68,
    "tired": 0.32,
    "scattered": 0.30,
    "focused": 0.70
  },
  "wellness": 62.5,
  "mood_history": [
    {"vector": [...], "wellness": 65, "date": "2024-01-15"},
    {"vector": [...], "wellness": 58, "date": "2024-01-14"}
  ],
  "goals": [
    {"id": "goal_1", "title": "...", "position": {"x": 2.5, "y": 0.5, "z": 1.8}}
  ]
}
```

The 3D fractal visualization uses this data to:
- Color the fractal based on mood vector
- Position goal orbs using the Golden Angle
- Show mood history as visual patterns

---

## 💜 Built for Neurodivergent Minds

This system was designed specifically for:
- **Aphantasia**: Everything is visual/text-based, no mental imagery required
- **ADHD**: Quick capture via photo, task chunking, spoon tracking
- **Autism**: Predictable processing, clear data, no surprises
- **Dyslexia**: OCR handles reading, results are visual
- **Executive Dysfunction**: Tasks extracted automatically, prioritized by spoon cost

**No shame. No judgment. Just understanding.**

---

Built with 💜 by someone with a brain like yours.
