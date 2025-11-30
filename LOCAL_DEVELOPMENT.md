# 💻 LOCAL DEVELOPMENT SETUP

Run the app locally before deploying to Heroku.

---

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py
```

The app will run at: **http://localhost:5000**

---

## Full Setup

### 1. Create Virtual Environment (Recommended)

```powershell
# Create venv
python -m venv venv

# Activate
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```powershell
# Copy example
copy .env.example .env

# Edit .env with your settings
notepad .env
```

### 3. Run Application

```powershell
python app.py
```

---

## Local Features

✅ **SQLite Database** - Automatic local database  
✅ **No PostgreSQL needed** - Automatically uses SQLite  
✅ **Hot Reload** - Changes auto-reload  
✅ **Debug Mode** - Detailed error messages  
✅ **Email Logging** - Tokens logged to console  

---

## Testing Locally

### 1. Register Account

```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "first_name": "Test"
  }'
```

### 2. Login

Visit: http://localhost:5000/login

### 3. Create Goal

```bash
curl -X POST http://localhost:5000/api/goals \
  -H "Content-Type: application/json" \
  -H "Cookie: session=YOUR_SESSION" \
  -d '{
    "title": "Test Goal",
    "category": "personal",
    "description": "Testing locally"
  }'
```

### 4. Check Health

Visit: http://localhost:5000/health

---

## Local vs Heroku

| Feature | Local | Heroku |
|---------|-------|--------|
| Database | SQLite | PostgreSQL |
| Email | Logged | Sent (if configured) |
| HTTPS | HTTP | HTTPS |
| Performance | Development | Production |
| Persistence | Local file | Cloud |

---

## Database Location

Local database is created as: **local.db**

To reset database:
```powershell
# Stop app
# Delete database
del local.db
# Restart app (creates new DB)
python app.py
```

---

## Debug Mode

Local runs with debug=True automatically:
- Detailed error messages
- Auto-reload on code changes
- Interactive debugger

**Never use debug=True in production!**

---

## Email Testing Locally

Without SMTP configuration, emails are logged:

```
2024-01-20 10:30:45 [INFO] ✅ Verification email sent to test@example.com
2024-01-20 10:30:45 [INFO] Verification token for test@example.com: abc123...
```

Copy token from logs to test verification:
```
http://localhost:5000/api/auth/verify-email?token=abc123...
```

---

## Port Configuration

Default: **5000**

To change:
```powershell
$env:PORT=8000
python app.py
```

Or in .env:
```
PORT=8000
```

---

## Common Issues

### Port already in use:
```powershell
# Find process on port 5000
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F

# Or use different port
$env:PORT=8000
```

### Database locked:
```powershell
# Stop all Python processes
# Delete local.db
# Restart
```

### Module not found:
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## Development Workflow

1. **Make changes** to app.py
2. **App auto-reloads** (debug mode)
3. **Test locally** at http://localhost:5000
4. **Deploy to Heroku** with `git push heroku master`

---

## Requirements

- Python 3.11+
- pip
- Virtual environment (recommended)

---

## Ready to Deploy?

Once everything works locally:

```powershell
# Deploy to Heroku
.\deploy.ps1
```

See QUICKSTART.md for deployment guide.
