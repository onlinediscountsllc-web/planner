# 🌀 LIFE FRACTAL INTELLIGENCE - HEROKU DEPLOYMENT

Complete production deployment with **self-healing** and **email verification**.

## ✨ Features Included

### Core Features
- ✅ **Self-Healing System** - Automatic error recovery with retry logic
- ✅ **Email Verification** - Secure account activation
- ✅ **PostgreSQL Database** - Production-ready with connection pooling
- ✅ **Complete Security** - Password hashing, session management, HTTPS
- ✅ **Goal Tracking** - Full CRUD operations
- ✅ **Habit Tracking** - Streak management
- ✅ **Virtual Pet System** - Gamification
- ✅ **Fractal Generation** - Personalized visualizations
- ✅ **7-Day Free Trial** - Automatic trial management
- ✅ **Subscription Ready** - Stripe integration ready

### Technical Features
- 🛡️ **Self-Healing Decorators** - `@retry_on_failure` and `@safe_execute`
- 📊 **Health Monitoring** - System health endpoint at `/health`
- 🔄 **Automatic Recovery** - Exponential backoff on errors
- 📧 **Email System** - Verification emails with templates
- 🔐 **Session Management** - Secure cookie handling
- 🗄️ **Database Pooling** - Efficient connection management

---

## 🚀 QUICK DEPLOY (5 Minutes)

### Option 1: Automated PowerShell Script (EASIEST)

1. **Extract this folder** to your computer

2. **Right-click the folder** → "Open PowerShell window here"

3. **Run:**
   ```powershell
   .\deploy.ps1
   ```

4. **Follow prompts:**
   - Login to Heroku (browser opens)
   - Enter app name (e.g., `life-fractal-john-2024`)
   - Configure email (optional)
   - Wait for deployment

5. **Done!** Your app is live at `https://YOUR-APP-NAME.herokuapp.com`

---

### Option 2: Manual Deployment

#### Prerequisites
- Git installed
- Heroku CLI installed ([download](https://devcenter.heroku.com/articles/heroku-cli))
- Heroku account ([free signup](https://signup.heroku.com))

#### Steps

```powershell
# 1. Login to Heroku
heroku login

# 2. Create app (replace YOUR-APP-NAME)
heroku create YOUR-APP-NAME

# 3. Add PostgreSQL database
heroku addons:create heroku-postgresql:essential-0

# 4. Set environment variables
heroku config:set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
heroku config:set ENVIRONMENT=production
heroku config:set APP_URL=https://YOUR-APP-NAME.herokuapp.com

# 5. (Optional) Configure email
heroku config:set SMTP_SERVER=smtp.gmail.com
heroku config:set SMTP_PORT=587
heroku config:set SMTP_USERNAME=your-email@gmail.com
heroku config:set SMTP_PASSWORD=your-app-password
heroku config:set FROM_EMAIL=noreply@yourdomain.com

# 6. Deploy
git init
git add .
git commit -m "Initial deployment"
git push heroku master

# 7. Open your app
heroku open
```

---

## 📧 Email Configuration (Optional but Recommended)

Email verification is **optional** but highly recommended for security.

### For Gmail:

1. **Enable 2-Factor Authentication** on your Gmail account

2. **Generate App Password:**
   - Go to Google Account → Security → 2-Step Verification → App Passwords
   - Generate new app password
   - Copy the 16-character password

3. **Configure Heroku:**
   ```powershell
   heroku config:set SMTP_SERVER=smtp.gmail.com
   heroku config:set SMTP_PORT=587
   heroku config:set SMTP_USERNAME=your-email@gmail.com
   heroku config:set SMTP_PASSWORD=your-app-password
   heroku config:set FROM_EMAIL=noreply@yourdomain.com
   ```

### Without Email Configuration:

- App works perfectly without email
- Verification tokens are logged to Heroku logs
- Users can still register and login
- Check logs with: `heroku logs --tail`

---

## 🛡️ Self-Healing System

The app includes **automatic error recovery**:

### How It Works:

1. **Automatic Retry** - Failed operations retry up to 3 times with exponential backoff
2. **Safe Execution** - Errors are caught and logged without crashing
3. **Health Monitoring** - System health tracked in real-time
4. **Graceful Degradation** - Falls back to safe defaults on failure

### Monitoring:

```bash
# View health status
curl https://YOUR-APP-NAME.herokuapp.com/health

# View detailed logs
heroku logs --tail -a YOUR-APP-NAME
```

### Example Self-Healing:

```python
# Database query fails → Retries 3 times → Falls back to empty array
goals = db.select('goals', {'user_id': user_id})

# Email send fails → Logs error → Returns gracefully
EmailVerificationSystem.send_verification_email(email, token, url)

# Fractal generation fails → Retries → Returns default fractal
fractal = generate_simple_fractal(user_data)
```

---

## 📊 Database Structure

### Tables Created Automatically:

- **users** - User accounts with email verification
- **goals** - Goal tracking with progress
- **habits** - Habit tracking with streaks
- **virtual_pets** - Pet companion system
- **journal_entries** - Daily journal with sentiment

All tables support both PostgreSQL (production) and SQLite (local development).

---

## 🔒 Security Features

✅ **Password Hashing** - Bcrypt with salt  
✅ **Session Management** - Secure HTTP-only cookies  
✅ **HTTPS Enforced** - SSL included with Heroku  
✅ **SQL Injection Prevention** - Parameterized queries  
✅ **Email Verification** - 24-hour expiring tokens  
✅ **Environment Variables** - No secrets in code  
✅ **CORS Protection** - Controlled cross-origin access  

---

## 💰 Heroku Costs

### Free Tier (Eco Dynos - $5/month):
- ✅ Up to 1000 hours/month
- ✅ Sleeps after 30 min inactivity
- ✅ Perfect for testing
- ✅ PostgreSQL Essential-0 (Free with credit card)

### Production Tier (Basic - $7/month):
- ✅ Never sleeps
- ✅ Custom domain support
- ✅ SSL certificates included
- ✅ Better for real users

**Total Cost to Start: $0-$5/month**

---

## 🧪 Testing Your Deployment

### 1. Test Registration:
```bash
curl -X POST https://YOUR-APP-NAME.herokuapp.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123","first_name":"Test"}'
```

### 2. Test Health:
```bash
curl https://YOUR-APP-NAME.herokuapp.com/health
```

### 3. Test Login (in browser):
- Go to `https://YOUR-APP-NAME.herokuapp.com/login`
- Register new account
- Check email for verification
- Login and explore dashboard

---

## 📝 Post-Deployment Checklist

- [ ] App is accessible at Heroku URL
- [ ] Can register new account
- [ ] Verification email received (if configured)
- [ ] Can login successfully
- [ ] Dashboard loads correctly
- [ ] Can create goals
- [ ] Health endpoint works
- [ ] Check logs for errors: `heroku logs --tail`

---

## 🔧 Common Commands

```powershell
# View logs (real-time)
heroku logs --tail -a YOUR-APP-NAME

# Open app in browser
heroku open -a YOUR-APP-NAME

# Check app status
heroku ps -a YOUR-APP-NAME

# View environment variables
heroku config -a YOUR-APP-NAME

# Add custom domain
heroku domains:add www.yourdomain.com -a YOUR-APP-NAME

# Scale dynos (upgrade)
heroku ps:scale web=1:basic -a YOUR-APP-NAME

# Restart app
heroku restart -a YOUR-APP-NAME

# Access database
heroku pg:psql -a YOUR-APP-NAME

# View database info
heroku pg:info -a YOUR-APP-NAME
```

---

## 🆘 Troubleshooting

### App won't start:
```bash
# Check logs
heroku logs --tail -a YOUR-APP-NAME

# Restart
heroku restart -a YOUR-APP-NAME
```

### Database errors:
```bash
# Verify database is attached
heroku addons -a YOUR-APP-NAME

# Check database connection
heroku pg:info -a YOUR-APP-NAME

# Reset database (WARNING: deletes all data)
heroku pg:reset DATABASE_URL -a YOUR-APP-NAME
```

### Email not sending:
- Check SMTP credentials are correct
- Check logs for email errors
- Verify Gmail app password is valid
- Without email, tokens are logged - check logs

### Self-healing not working:
- Check health endpoint: `/health`
- View logs for error recovery attempts
- System automatically retries failed operations

---

## 🎯 Next Steps

### Immediate:
1. ✅ Deploy to Heroku
2. ✅ Test registration and login
3. ✅ Configure email verification
4. ✅ Monitor health endpoint

### This Week:
1. Set up custom domain
2. Configure Stripe for payments
3. Add GoFundMe integration
4. Invite beta users

### This Month:
1. Get first 10 users
2. Collect feedback
3. Add new features
4. Scale as needed

---

## 📚 File Structure

```
heroku_production/
├── app.py                 # Main Flask application (self-healing + email)
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku process configuration
├── runtime.txt           # Python version specification
├── deploy.ps1            # Automated deployment script
└── README.md             # This file
```

---

## 🌟 Features Deep Dive

### Self-Healing Decorators:

```python
@retry_on_failure(max_attempts=3, delay=1.0, component="database")
def query_database():
    # Automatically retries on failure
    # Logs errors to monitoring system
    # Returns fallback on total failure
    pass

@safe_execute(fallback_value=[], component="api")
def get_user_data():
    # Never crashes
    # Returns fallback on error
    # Logs for debugging
    pass
```

### Email Verification Flow:

1. User registers → Token generated
2. Email sent with verification link
3. User clicks link → Token validated
4. Email marked as verified
5. User gains full access

### Health Monitoring:

```json
{
  "overall_health": "excellent",
  "uptime_seconds": 3600,
  "error_counts": {
    "database": 0,
    "email": 0
  },
  "component_status": {
    "database": "healthy",
    "email": "healthy"
  }
}
```

---

## 💡 Pro Tips

1. **Monitor logs daily** - `heroku logs --tail`
2. **Use free tier first** - Test before upgrading
3. **Enable email verification** - Better security
4. **Check health endpoint** - `/health`
5. **Backup database** - Use Heroku automated backups
6. **Add custom domain** - Professional appearance
7. **Use environment variables** - Never hardcode secrets
8. **Scale gradually** - Upgrade as users grow

---

## 📞 Support

- **Heroku Docs:** https://devcenter.heroku.com
- **Flask Docs:** https://flask.palletsprojects.com
- **PostgreSQL Docs:** https://www.postgresql.org/docs
- **View logs:** `heroku logs --tail -a YOUR-APP-NAME`
- **Email:** Check application logs for verification tokens

---

## 🎉 Ready to Deploy?

1. **Extract this folder**
2. **Open PowerShell in folder**
3. **Run:** `.\deploy.ps1`
4. **Wait 5 minutes**
5. **Your app is live!**

**That's it!** Your production-ready app with self-healing and email verification is now live on Heroku! 🚀

---

**Built with ❤️ for Life Fractal Intelligence**

**Questions? Check the logs:** `heroku logs --tail`
#   p l a n n e r  
 