# DEPLOYMENT GUIDE - LIFE FRACTAL INTELLIGENCE v8.0
# ==================================================

## Overview
This guide will help you deploy Life Fractal Intelligence v8.0 with secure authentication to Render.com.

## What's New in v8.0
- ✅ Argon2 password hashing (industry best practice)
- ✅ CAPTCHA protection against bots and fraud
- ✅ Email notifications for trial status
- ✅ Password reset functionality
- ✅ Rate limiting on login attempts
- ✅ Session management
- ✅ Account lockout after failed attempts
- ✅ GoFundMe integration for support
- ✅ Comprehensive bug testing

---

## Pre-Deployment Checklist

### 1. Email Configuration (IMPORTANT!)

You need to set up SMTP for email notifications. For Gmail:

1. Go to your Google Account settings
2. Enable 2-Factor Authentication
3. Generate an App Password:
   - Go to Security → App passwords
   - Select "Mail" and your device
   - Copy the generated 16-character password

### 2. Required Environment Variables

Add these to your Render environment:

```
# Flask
SECRET_KEY=<generate-a-long-random-string>
PORT=8080
DEBUG=False

# SMTP Email Settings
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=onlinediscountsllc@gmail.com
SMTP_PASSWORD=<your-16-char-app-password>

# Optional
PYTHON_VERSION=3.12.0
```

---

## Deployment Steps

### Step 1: Prepare Your Repository

1. Copy these files to your GitHub repository (onlinediscountsllc-web/planner):
   ```
   - secure_auth_module.py
   - life_fractal_v8_secure.py (rename to app.py or main.py)
   - requirements.txt
   ```

2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Deploy v8.0 with secure authentication"
   git push origin main
   ```

### Step 2: Configure Render

1. Go to your Render dashboard (render.com)
2. Find your existing service or create a new Web Service
3. Connect to your GitHub repository
4. Configure:

   **Build Command:**
   ```
   pip install -r requirements.txt
   ```

   **Start Command:**
   ```
   gunicorn life_fractal_v8_secure:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```

### Step 3: Set Environment Variables

In Render dashboard → Environment:

1. Click "Add Environment Variable"
2. Add all variables from the checklist above
3. Generate a SECRET_KEY (use this Python command):
   ```python
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```

### Step 4: Deploy

1. Click "Manual Deploy" → "Deploy latest commit"
2. Wait for deployment (5-10 minutes)
3. Check logs for any errors

---

## Post-Deployment Testing

### Test 1: Health Check
```bash
curl https://planner-1-pyd9.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "8.0",
  "features": [...]
}
```

### Test 2: Run Bug Test Suite
```bash
python test_bugs.py https://planner-1-pyd9.onrender.com
```

This will test:
- CAPTCHA generation
- Registration with validation
- Login security
- Session management
- Rate limiting
- Password reset
- Dashboard access

---

## Email Testing

After deployment, test emails by:

1. Register a new account
2. Check your email for welcome message
3. Verify trial information is included
4. Check GoFundMe link is present

### Expected Emails:

1. **Welcome Email** (on registration)
   - Subject: "Welcome to Life Fractal Intelligence - Your 7-Day Trial Starts Now!"
   - Contains: Trial info, subscription details, GoFundMe link

2. **Trial Ending Soon** (2 days before expiration)
   - Subject: "Your Life Fractal Intelligence Trial Ends in X Days"
   - Contains: Subscription prompt, GoFundMe link

3. **Trial Expired** (after 7 days)
   - Subject: "Your Life Fractal Intelligence Trial Has Ended"
   - Contains: Subscription required message

4. **Password Reset** (when requested)
   - Subject: "Reset Your Life Fractal Intelligence Password"
   - Contains: Reset link (expires in 30 minutes)

---

## Database Management

The app uses SQLite with two databases:

1. **auth_secure.db** - Authentication data
   - Users
   - Sessions
   - Reset tokens
   - Login attempts

2. **Original data store** - User app data
   - Pets
   - Goals
   - Habits
   - Daily entries

### Backup Strategy (Recommended)

Add to your deployment:

```python
# In your cron job or scheduled task
import shutil
from datetime import datetime

def backup_databases():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy('auth_secure.db', f'backups/auth_{timestamp}.db')
```

---

## Security Features

### 1. Password Security
- Argon2id hashing (best-in-class)
- Minimum 8 characters required
- No plain text storage ever

### 2. CAPTCHA Protection
- Math-based challenges
- 5-minute expiration
- Prevents bot registrations

### 3. Rate Limiting
- 5 failed attempts per IP per 15 minutes
- Automatic account lockout after 5 failures
- Login attempt logging

### 4. Session Security
- 24-hour expiration
- Secure tokens (32-byte hex)
- IP tracking
- HttpOnly cookies

### 5. Password Reset
- Secure token generation
- 30-minute expiration
- One-time use only
- Email verification required

---

## Monitoring & Logs

### Check Application Logs in Render:
1. Go to your service dashboard
2. Click "Logs" tab
3. Look for:
   - `User registered: <email>`
   - `User logged in: <email>`
   - `Email sent to <email>: <subject>`
   - Any errors (ERROR level)

### Check Authentication Logs:
File: `auth_system.log` (on your server)

Contains:
- All authentication events
- Failed login attempts
- Password resets
- System errors

---

## Troubleshooting

### Email Not Sending?
1. Check SMTP_PASSWORD is correct (16-char App Password)
2. Verify SMTP_USER matches your Gmail
3. Check Render logs for SMTP errors
4. Test with a simple send:
   ```python
   from secure_auth_module import EmailService
   email_service = EmailService()
   email_service.send_email("test@test.com", "Test", "<h1>Test</h1>")
   ```

### CAPTCHA Not Working?
1. Verify `/api/auth/captcha` endpoint responds
2. Check challenge_id is being passed correctly
3. Clear captcha_challenges dict if memory issue

### Login Failing?
1. Check if account is locked (5 failed attempts)
2. Verify CAPTCHA is correct
3. Check rate limiting (15-minute window)
4. Review logs for specific error

### Database Issues?
1. Ensure write permissions on server
2. Check SQLite version compatibility
3. Verify table creation in logs

---

## Performance Optimization

### For Production:

1. **Use PostgreSQL instead of SQLite**
   ```python
   # Replace SQLite connection with PostgreSQL
   # In secure_auth_module.py
   import psycopg2
   ```

2. **Add Redis for CAPTCHA storage**
   ```python
   # Instead of in-memory dict
   import redis
   r = redis.Redis(host='localhost', port=6379)
   ```

3. **Enable caching**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   ```

4. **Add more workers in gunicorn**
   ```bash
   gunicorn life_fractal_v8_secure:app --workers 4
   ```

---

## API Endpoints Reference

### Authentication
```
POST /api/auth/captcha          - Get CAPTCHA challenge
POST /api/auth/check-email      - Check if email exists
POST /api/auth/register         - Register new user
POST /api/auth/login            - Login user
POST /api/auth/forgot-password  - Request password reset
POST /api/auth/reset-password   - Reset password with token
POST /api/auth/verify-session   - Verify session token
```

### User Data
```
GET  /api/user/<user_id>              - Get user profile
GET  /api/user/<user_id>/dashboard    - Get dashboard data
GET  /api/user/<user_id>/today        - Get today's entry
POST /api/user/<user_id>/today        - Update today's entry
```

### System
```
GET  /health    - Health check
GET  /          - Service info
```

---

## Support & Contact

- Email: onlinediscountsllc@gmail.com
- GoFundMe: https://gofund.me/8d9303d27
- GitHub: onlinediscountsllc-web/planner

---

## Success Checklist

After deployment, verify:

- [ ] Health endpoint returns 200
- [ ] Can register new user
- [ ] Welcome email received
- [ ] CAPTCHA displays correctly
- [ ] Can login with correct credentials
- [ ] Wrong password is rejected
- [ ] Dashboard loads with data
- [ ] Pet appears in dashboard
- [ ] Goals and habits display
- [ ] Trial days countdown works
- [ ] GoFundMe link appears when appropriate
- [ ] Password reset emails work
- [ ] Rate limiting blocks after 5 attempts
- [ ] All tests in test_bugs.py pass

---

## Version History

**v8.0** (Current)
- Secure authentication with Argon2
- CAPTCHA protection
- Email notifications
- Password reset
- Rate limiting
- Comprehensive testing

**v7.1** (Previous)
- Basic auth with pbkdf2
- No CAPTCHA
- No email notifications
- No password reset

---

## Next Steps After Deployment

1. **Test thoroughly** with the bug test suite
2. **Monitor email delivery** for first few users
3. **Check logs daily** for any errors
4. **Set up database backups**
5. **Configure SSL** (Render does this automatically)
6. **Add domain name** if desired
7. **Promote your GoFundMe** to new users
8. **Collect user feedback** on the trial system

---

🎉 **You're all set!** Your Life Fractal Intelligence v8.0 is now secure and ready for users!
