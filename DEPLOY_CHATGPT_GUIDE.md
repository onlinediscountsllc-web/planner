# 🌀 LIFE FRACTAL INTELLIGENCE - DEPLOYMENT GUIDE
## ChatGPT Ultimate Edition v2.0

---

## 🎯 WHAT THIS FIXES

Your ChatGPT was getting **404 errors** because:
- The old code had endpoints like `/api/register`
- ChatGPT's OpenAPI spec expected `/chatgpt/health`, `/chatgpt/fractals/generate`, etc.

**This new version has ALL the ChatGPT endpoints!**

---

## 🚀 QUICK DEPLOY (5 minutes)

### Step 1: Replace Your Code

1. Open your local `planner` folder
2. **Replace** `app.py` (or whatever your main file is) with `life_fractal_chatgpt_ultimate.py`
3. Rename it to `app.py`

```powershell
# In PowerShell, navigate to your folder:
cd C:\Users\Luke\Desktop\planner

# Backup old file (optional)
mv app.py app_old_backup.py

# Copy new file (or download from Claude output)
# Then rename:
mv life_fractal_chatgpt_ultimate.py app.py
```

### Step 2: Update requirements.txt

Replace your `requirements.txt` with the new one provided.

### Step 3: Push to GitHub

```powershell
git add .
git commit -m "Add ChatGPT endpoints and full pet system"
git push origin main
```

### Step 4: Wait for Render

Render will auto-deploy in 2-3 minutes. Watch the logs at:
https://dashboard.render.com

### Step 5: Test

Open these URLs:
- https://planner-1-pyd9.onrender.com/health
- https://planner-1-pyd9.onrender.com/chatgpt/health
- https://planner-1-pyd9.onrender.com/chatgpt/fractals/list

---

## 🤖 UPDATE CHATGPT

### Replace the OpenAPI Schema

1. Go to your GPT at https://chat.openai.com
2. Click "Edit GPT"
3. Go to "Configure" → "Actions"
4. Delete the old schema
5. Paste the new `openapi_chatgpt_v2.yaml` content
6. Click "Save"

---

## 📋 NEW FEATURES

### 🐾 Full Pet System
- **8 species**: cat, dog, dragon, phoenix, owl, fox, unicorn, butterfly
- **Feed pets**: golden_apple, fibonacci_berries, fractal_treat, cosmic_kibble, sacred_nectar
- **Play activities**: explore_mandelbrot, chase_fractals, meditate_sacred, learn_geometry, dream_fractals
- **Level up system**: XP, abilities, evolution!

### 🎨 12+ Fractal Types
Each with meaning:
- `mandelbrot` - Infinite complexity from simple rules
- `julia` - Different paths lead to different beauty
- `golden_spiral` - Nature's growth pattern
- `flower_of_life` - Sacred interconnection
- `burning_ship` - Rising above challenges
- `phoenix` - Rebirth and renewal
- `sierpinski` - Power of repetition
- `dragon` - Complexity from simplicity
- `koch_snowflake` - Infinite within finite
- `fibonacci` - Compound growth
- `lorenz` - Small actions matter
- `metatron` - Universal structure

### 🧠 Neurodivergent Support
- `/chatgpt/aphantasia/help` - External visualization guide
- `/chatgpt/spoons/check` - Energy management

### 🖼️ Media Generation
- `/chatgpt/poster/create` - Printable posters with goals

---

## 🔧 CHATGPT INSTRUCTIONS

Update your GPT's instructions to include these capabilities:

```
You are Fractal Explorer, a friendly guide who transforms goals into beautiful fractal art!

🐾 PET SYSTEM:
- Ask users to pick a pet: cat 🐱, dog 🐕, dragon 🐉, phoenix 🔥, owl 🦉, fox 🦊, unicorn 🦄, butterfly 🦋
- Create pet with: POST /chatgpt/pet/create {"species": "dragon", "name": "Ember"}
- Feed pet: POST /chatgpt/pet/feed {"pet": <pet_data>, "food": "golden_apple"}
- Play: POST /chatgpt/pet/play {"pet": <pet_data>, "activity": "explore_mandelbrot"}
- IMPORTANT: Save the pet data and pass it back for feed/play!

🎨 FRACTALS:
- List types: GET /chatgpt/fractals/list
- Generate: POST /chatgpt/fractals/generate {"type": "mandelbrot", "size": 800}
- Visualize goal: POST /chatgpt/goals/visualize {"goal": "Learn meditation"}

🧠 FOR NEURODIVERGENT USERS:
- Explain that fractals ARE external visualization
- Use Spoon Theory: POST /chatgpt/spoons/check {"spoons": 5}
- Be gentle, no shame about energy levels

📐 SACRED MATH:
- Explain: GET /chatgpt/sacred/explain
- Calculate: POST /chatgpt/sacred/calculate {"type": "fibonacci", "n": 10}

CONVERSATION FLOW:
1. Greet warmly, offer pet choice
2. Create their pet
3. Ask about their goal
4. Suggest and generate fractal
5. Explain the meaning
6. Offer to create poster
7. Pet interactions throughout!

Always display images when returned!
```

---

## 🧪 TESTING ENDPOINTS

### Test Health
```bash
curl https://planner-1-pyd9.onrender.com/chatgpt/health
```

### Test Create Pet
```bash
curl -X POST https://planner-1-pyd9.onrender.com/chatgpt/pet/create \
  -H "Content-Type: application/json" \
  -d '{"species": "dragon", "name": "Spira"}'
```

### Test Generate Fractal
```bash
curl -X POST https://planner-1-pyd9.onrender.com/chatgpt/fractals/generate \
  -H "Content-Type: application/json" \
  -d '{"type": "golden_spiral", "size": 512}'
```

### Test Visualize Goal
```bash
curl -X POST https://planner-1-pyd9.onrender.com/chatgpt/goals/visualize \
  -H "Content-Type: application/json" \
  -d '{"goal": "Learn to meditate daily"}'
```

---

## ⚠️ TROUBLESHOOTING

### "Not Found" errors
- Make sure you deployed the NEW file
- Check Render logs for errors
- Visit /chatgpt/health to verify

### "Connection refused"
- Render free tier sleeps after 15 min
- Visit the URL in browser first to wake it

### Pet not working
- You must pass the FULL pet object back
- Save pet data from create, send it to feed/play

### Images not showing in ChatGPT
- The image_data contains base64 PNG
- ChatGPT should auto-display it
- If not, the response includes the data

---

## 📁 FILES PROVIDED

1. **life_fractal_chatgpt_ultimate.py** - Main backend (deploy this!)
2. **openapi_chatgpt_v2.yaml** - ChatGPT schema (paste in GPT settings)
3. **requirements.txt** - Python dependencies

---

## 🎉 WHAT USERS CAN DO

In ChatGPT, users can now:

1. **Choose a pet companion** from 8 magical species
2. **Feed their pet** with sacred geometry foods
3. **Play activities** like exploring Mandelbrot sets
4. **Watch their pet level up** and unlock abilities
5. **Tell their goals** and see suggested fractals
6. **Generate beautiful fractal art** instantly
7. **Learn why** each fractal matches their goal
8. **Create printable posters** of their visualized goals
9. **Check their energy** with Spoon Theory
10. **Learn about aphantasia** and external visualization

All without leaving ChatGPT!

---

## 💜 BUILT FOR BRAINS LIKE OURS

This app was built specifically for neurodivergent minds:
- **Aphantasia**: External visualization instead of mental imagery
- **ADHD**: Gamification with pets keeps it engaging
- **Executive dysfunction**: Simple one-step interactions
- **Energy management**: Spoon Theory integration

Traditional planning apps weren't built for us. This one is. 💜

---

## 📧 SUPPORT

Questions? Email: onlinediscountsllc@gmail.com

GoFundMe: https://gofund.me/8d9303d27
