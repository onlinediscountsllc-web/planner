# ⚡ QUICK START - Studio Features & Self-Healing

## 🚀 **5-MINUTE SETUP**

### **Step 1: Install** (Copy-Paste One Command)

```powershell
pip install flask flask-cors numpy pillow scikit-learn requests opencv-python --break-system-packages
```

### **Step 2: Run**

```powershell
py life_planner_unified_ultimate.py
```

### **Step 3: Open Browser**

```
http://localhost:5000
```

### **Step 4: Login**

```
Email: onlinediscountsllc@gmail.com
Password: admin8587037321
```

**Done! All features working!** ✅

---

## 🎯 **TRY THESE NEW FEATURES**

### **1. Create a Detailed Goal** (NEW!)

```bash
curl -X POST http://localhost:5000/api/user/admin_001/goals \
  -H "Content-Type: application/json" \
  -d '{
    "category": "career",
    "title": "Build portfolio website",
    "why_important": "Need to showcase work to clients",
    "difficulty": 6,
    "importance": 8,
    "subtasks": ["Choose platform", "Select 3 projects", "Write case studies"],
    "obstacles": ["Limited time", "Design skills"],
    "success_criteria": ["Live site", "3 projects shown", "Contact form works"]
  }'
```

**What's different?**
- OLD: Just title + description
- NEW: 14 fields with rich context!

---

### **2. Add a Rich Journal Entry** (NEW!)

```bash
curl -X POST http://localhost:5000/api/user/admin_001/journal \
  -H "Content-Type: application/json" \
  -d '{
    "mood": 7,
    "energy": 6,
    "gratitude": ["Friend helped me", "Beautiful day"],
    "wins": ["Finished project", "Went for a run"],
    "lessons_learned": ["Starting early reduces stress"],
    "journal_text": "Today was really productive..."
  }'
```

**What's different?**
- OLD: 4 basic fields
- NEW: 20+ fields with deep reflection!

---

### **3. Generate Vision Board Image** (NEW!)

```bash
curl -X POST http://localhost:5000/api/user/admin_001/vision-board/generate-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "cozy apartment with plants, peaceful, warm colors"}'
```

**Self-Healing Magic:**
- ✅ ComfyUI running? → Beautiful AI image
- ❌ ComfyUI not running? → Placeholder image (still works!)

---

### **4. Create Progress Video** (NEW!)

```bash
curl -X POST http://localhost:5000/api/user/admin_001/video/progress \
  -H "Content-Type: application/json" \
  -d '{"duration": 10.0}'
```

Downloads MP4 animation of your milestone journey!

---

### **5. Get AI Mood Prediction** (NEW!)

```bash
curl -X POST http://localhost:5000/api/user/admin_001/predictions/mood \
  -H "Content-Type: application/json" \
  -d '{"energy": 7, "sleep_quality": 8, "stress": 3}'
```

**Response:**
```json
{
  "prediction": 72.5,
  "confidence": 0.85,
  "method": "ml_model",
  "recommendation": "Good mood predicted! Great time to tackle important goals."
}
```

---

## 🔧 **SELF-HEALING IN ACTION**

### **Scenario 1: Missing OpenCV**

**You:** "Create video"

**System (graceful):**
```json
{
  "error": "Video generation requires OpenCV",
  "cv2_available": false,
  "install_command": "pip install opencv-python --break-system-packages"
}
```

**NOT a crash - clear guidance! ✅**

---

### **Scenario 2: ComfyUI Down**

**You:** "Generate vision image"

**System (automatic fallback):**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "is_placeholder": true,
  "message": "ComfyUI not connected - showing placeholder"
}
```

**Still get an image - just a placeholder! ✅**

---

### **Scenario 3: Network Timeout**

**You:** API request times out

**System (automatic retry):**
- Attempt 1: ❌ Timeout
- Wait 1 second
- Attempt 2: ✅ Success
- **You never knew it failed!**

---

## 📊 **CHECK SYSTEM HEALTH**

```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "gpu_available": true,
  "features": {
    "gpu_fractals": true,
    "video_generation": true,
    "ml_predictions": true,
    "comfyui": false  // ← This tells you what's working!
  }
}
```

---

## 🆘 **TRIGGER SELF-HEAL**

If something seems off:

```bash
curl -X POST http://localhost:5000/api/system/self-heal
```

**System automatically:**
- ✅ Reconnects to ComfyUI
- ✅ Backs up all data
- ✅ Retrains ML models
- ✅ Clears error states

---

## 💡 **TIPS**

### **1. Don't Worry About Missing Libraries**

```python
# OLD (v3.1):
import cv2  # ← Crash if not installed!

# NEW (v4.0):
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False  # ← System still runs!
```

**Just install what you need, skip the rest!**

---

### **2. Data Auto-Backs Up**

Every 5 minutes, your data is saved to:
```
life_planner_data/backup_*.json
```

Last 10 backups kept automatically.

---

### **3. ML Auto-Trains**

Every 5 journal entries, the system retrains itself:
```
[INFO] ML model trained with 25 samples
```

Gets smarter over time!

---

## 📝 **DETAILED GOAL EXAMPLE**

**Full goal with all fields:**

```json
{
  "id": "goal_1_a3f9c2e1",
  "category": "financial",
  "title": "Build $1000 emergency fund",
  "description": "Save consistently to reach financial safety",
  
  "difficulty": 6,
  "importance": 10,
  "energy_required": 5,
  "estimated_hours": 0,
  "actual_hours": 0,
  
  "why_important": "Peace of mind and financial security during uncertain times",
  
  "subtasks": [
    "Set up automatic transfer ($100/month)",
    "Track all expenses in app",
    "Cut 2 unnecessary subscriptions",
    "Sell unused items"
  ],
  
  "resources_needed": [
    "Budgeting app (YNAB or Mint)",
    "Separate savings account"
  ],
  
  "obstacles": [
    "Irregular income",
    "Unexpected expenses",
    "Impulse purchases"
  ],
  
  "support_needed": "Accountability partner for monthly check-ins",
  
  "success_criteria": [
    "Save $100/month consistently",
    "Reach $500 in 5 months",
    "Reach $1000 in 10 months",
    "No withdrawals unless true emergency"
  ],
  
  "tags": ["essential", "2025", "stability"],
  
  "notes": "This is my #1 priority. Everything else can wait.",
  
  "progress_percentage": 25.0,
  "completed": false,
  "created_date": "2025-11-01T10:30:00Z"
}
```

**Powerful, detailed, actionable!**

---

## 🎉 **YOU'RE READY!**

1. ✅ Server running
2. ✅ Features working (with graceful fallbacks)
3. ✅ Auto-backup enabled
4. ✅ Self-healing active

**Start creating detailed goals and journal entries!**

---

## 🆘 **TROUBLESHOOTING**

### **"ImportError: No module named 'flask'"**

```powershell
pip install flask --break-system-packages
```

### **"ComfyUI not connected"**

**Not a problem!** System uses placeholders automatically.

To enable ComfyUI:
1. Install ComfyUI
2. Start ComfyUI server
3. Enable in user settings

### **"Video generation failed"**

Install OpenCV:
```powershell
pip install opencv-python --break-system-packages
```

### **"ML predictions using rule_based"**

Add more journal entries! System auto-trains after 10+ entries.

---

## 📚 **FULL DOCUMENTATION**

See: **ULTIMATE_SYSTEM_V4_GUIDE.md**

**Everything works. Everything has fallbacks. Build your life!** 🌟
