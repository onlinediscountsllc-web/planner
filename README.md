# Life Fractal Intelligence v8.0 - Secure Authentication Update

## 🔒 What's New in v8.0

This is a **major security and authentication update** for Life Fractal Intelligence, adding enterprise-grade security features while maintaining all the neurodivergent-friendly features you love.

### ✨ New Features

1. **Argon2 Password Hashing**
   - Industry best-practice password security
   - Replaces pbkdf2 with more secure Argon2id algorithm
   - Prevents rainbow table and brute-force attacks

2. **CAPTCHA Protection**
   - Math-based challenges to prevent bots
   - Protects registration and login
   - 5-minute expiration for security

3. **Email Notifications**
   - Welcome email with trial information
   - Trial ending warnings (2 days before)
   - Trial expired notifications
   - Password reset emails
   - All emails include GoFundMe link

4. **Password Reset System**
   - Secure token-based reset
   - 30-minute expiration
   - One-time use tokens
   - Email verification required

5. **Rate Limiting**
   - 5 failed attempts per IP per 15 minutes
   - Prevents brute-force attacks
   - Login attempt tracking

6. **Account Security**
   - Automatic lockout after 5 failed attempts
   - Session management with 24-hour expiration
   - Secure token generation
   - IP address tracking

7. **Returning User Check**
   - Separate endpoint to check if email exists
   - Better UX for returning vs new users
   - Prevents accidental duplicate accounts

8. **Comprehensive Testing**
   - Full bug test suite included
   - Tests all authentication flows
   - Validates security features
   - Production-ready verification

---

## 📦 What's Included

```
life-fractal-v8-secure/
├── secure_auth_module.py       # Complete authentication system
├── life_fractal_v8_secure.py   # Enhanced main application
├── requirements.txt             # All dependencies
├── test_bugs.py                 # Comprehensive test suite
├── setup_local.py               # Local setup script
├── DEPLOYMENT_GUIDE.md          # Full deployment instructions
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Option 1: Local Testing (5 minutes)

```bash
# 1. Install dependencies
python setup_local.py

# 2. Edit .env file with your SMTP password
nano .env

# 3. Run the application
python life_fractal_v8_secure.py

# 4. Test it
python test_bugs.py http://localhost:8080
```

### Option 2: Deploy to Render (10 minutes)

```bash
# 1. Copy files to your GitHub repo
cp *.py ../planner/
cp requirements.txt ../planner/
cd ../planner

# 2. Commit and push
git add .
git commit -m "Deploy v8.0 with secure authentication"
git push origin main

# 3. Configure Render (see DEPLOYMENT_GUIDE.md)
# 4. Deploy and test
```

---

## 🔧 Configuration

### Required Environment Variables

```bash
# Flask
SECRET_KEY=<generate-with-secrets.token_hex(32)>
PORT=8080
DEBUG=False

# SMTP (for emails)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=onlinediscountsllc@gmail.com
SMTP_PASSWORD=<your-gmail-app-password>
```

### Gmail App Password Setup

1. Go to Google Account → Security
2. Enable 2-Factor Authentication
3. Generate App Password:
   - Security → App passwords
   - Select "Mail" and your device
   - Copy the 16-character password
4. Use this password as SMTP_PASSWORD

---

## 🧪 Testing

### Run Full Test Suite

```bash
python test_bugs.py http://localhost:8080
```

Tests include:
- ✅ Health check
- ✅ CAPTCHA generation
- ✅ Registration validation
- ✅ Email checking
- ✅ User registration
- ✅ Duplicate prevention
- ✅ Login security
- ✅ CAPTCHA verification
- ✅ Password validation
- ✅ Session management
- ✅ Rate limiting
- ✅ Password reset
- ✅ Dashboard access
- ✅ CORS headers

### Expected Output

```
==================================================
LIFE FRACTAL INTELLIGENCE - BUG TEST SUITE
==================================================

[TEST] Health Check
  ✓ PASS: Server is healthy

[TEST] CAPTCHA Generation
  ✓ PASS: CAPTCHA generated: Security Check: What is 47 + 83?

...

==================================================
TEST SUMMARY
==================================================

Total Tests: 15
Passed: 15
Failed: 0
Pass Rate: 100.0%

🎉 ALL TESTS PASSED! System is ready for deployment.
```

---

## 📧 Email Templates

### 1. Welcome Email (Registration)
- **Subject:** "Welcome to Life Fractal Intelligence - Your 7-Day Trial Starts Now!"
- **Includes:**
  - Trial information (7 days)
  - Feature list
  - Subscription details ($20/month)
  - GoFundMe link

### 2. Trial Ending Soon (2 days before)
- **Subject:** "Your Life Fractal Intelligence Trial Ends in X Days"
- **Includes:**
  - Days remaining
  - Subscription prompt
  - GoFundMe alternative

### 3. Trial Expired
- **Subject:** "Your Life Fractal Intelligence Trial Has Ended"
- **Includes:**
  - Subscription requirement
  - Data preservation notice
  - GoFundMe link

### 4. Password Reset
- **Subject:** "Reset Your Life Fractal Intelligence Password"
- **Includes:**
  - Reset link (30-minute expiration)
  - Security notice

---

## 🔐 Security Features

### Password Security
- Argon2id hashing (best-in-class)
- Minimum 8 characters required
- No plaintext storage
- Secure password reset flow

### Account Protection
- Rate limiting (5 attempts per 15 min)
- Account lockout after 5 failures
- CAPTCHA on registration and login
- Session expiration (24 hours)

### Data Security
- Parameterized queries (SQL injection prevention)
- Secure session tokens
- IP address tracking
- Login attempt logging

---

## 📊 API Endpoints

### Authentication
```
GET  /api/auth/captcha          - Generate CAPTCHA challenge
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

---

## 🐛 Troubleshooting

### Email Not Sending?
- Check SMTP_PASSWORD is correct (16-char App Password)
- Verify SMTP_USER matches your Gmail
- Check Render logs for SMTP errors

### CAPTCHA Not Working?
- Clear browser cache
- Check /api/auth/captcha endpoint
- Verify challenge_id is being passed

### Login Failing?
- Check if account is locked (5 failed attempts)
- Verify CAPTCHA answer
- Check rate limiting status

### Database Issues?
- Ensure write permissions
- Check SQLite compatibility
- Verify table creation in logs

---

## 💡 Key Improvements Over v7.1

| Feature | v7.1 | v8.0 |
|---------|------|------|
| Password Hashing | pbkdf2 | Argon2id ✅ |
| CAPTCHA | None | Math-based ✅ |
| Email Notifications | None | Full system ✅ |
| Password Reset | None | Secure tokens ✅ |
| Rate Limiting | None | IP-based ✅ |
| Account Lockout | None | After 5 attempts ✅ |
| Session Management | Basic | Token-based ✅ |
| Security Testing | None | Comprehensive ✅ |

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] Set up Gmail App Password
- [ ] Configure all environment variables
- [ ] Test email delivery
- [ ] Run full bug test suite
- [ ] Verify CAPTCHA works
- [ ] Test password reset
- [ ] Confirm rate limiting active
- [ ] Check trial notifications send
- [ ] Verify GoFundMe links work
- [ ] Test on mobile devices
- [ ] Set up database backups
- [ ] Monitor logs for errors

---

## 🌟 User Experience Flow

### New User Journey
1. Visit site, see registration form
2. Solve CAPTCHA challenge
3. Register with email/password
4. Receive welcome email with trial info
5. Login with CAPTCHA
6. Access dashboard with virtual pet
7. Use all features for 7 days
8. Receive trial ending email (day 5)
9. Subscribe or see trial expired notice

### Returning User Journey
1. Visit site, enter email
2. System recognizes returning user
3. Solve CAPTCHA challenge
4. Login successfully
5. Check trial days remaining
6. Continue using features

### Password Reset Journey
1. Click "Forgot Password"
2. Enter email address
3. Receive reset email
4. Click reset link
5. Enter new password
6. Login with new password

---

## 📞 Support

- **Email:** onlinediscountsllc@gmail.com
- **GoFundMe:** https://gofund.me/8d9303d27
- **GitHub:** onlinediscountsllc-web/planner

---

## 🙏 Supporting Development

Life Fractal Intelligence is built by someone with autism, ADHD, and aphantasia - for others like us. If you'd like to support continued development:

**GoFundMe:** https://gofund.me/8d9303d27

Your support helps us:
- Add more neurodivergent-friendly features
- Improve accessibility
- Expand pet species and behaviors
- Create more visualization options
- Keep the service affordable

---

## 📝 License & Attribution

Life Fractal Intelligence v8.0
© 2025 Online Discounts LLC
All rights reserved.

Built with:
- Sacred mathematics and ancient geometry
- Compassion for neurodivergent minds
- Industry best-practice security
- Love for virtual pets 🐱🐉🦊

---

## 🎉 Thank You!

Thank you for using Life Fractal Intelligence! Your trust in our secure authentication system means everything. We're committed to protecting your data while providing the best neurodivergent-focused planning experience.

**Ready to deploy?** Follow the DEPLOYMENT_GUIDE.md for step-by-step instructions!

---

*Version 8.0 - December 2025*
*"Sacred Mathematics for Neurodivergent Minds"*
