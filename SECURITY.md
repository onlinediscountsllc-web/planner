# Security Best Practices & Implementation Guide

## 🔒 Security Overview

This document outlines all security measures implemented in the Life Planner application to protect user data and ensure safe operations.

---

## 1. Authentication & Authorization

### A. Password Security

**Implementation**:
- Passwords hashed using PBKDF2-SHA256 (Werkzeug default)
- Minimum 8 characters required
- Passwords never stored in plain text
- Secure password reset with time-limited tokens

**Code Location**: `models/database.py` - User.set_password()

```python
# Password hashing
user.set_password(password)  # Automatically hashes

# Password verification
user.check_password(password)  # Returns True/False
```

### B. JWT Token Management

**Implementation**:
- Access tokens: 1 hour expiry
- Refresh tokens: 30 days expiry
- Tokens stored client-side (localStorage)
- Automatic refresh mechanism

**Security Features**:
- JWT_SECRET_KEY separate from SECRET_KEY
- Tokens signed and verified
- No sensitive data in payload
- Revocation on logout

**Best Practices**:
```python
# Always validate tokens
@jwt_required()
def protected_endpoint():
    user_id = get_jwt_identity()
    # Process request
```

### C. Session Security

**Configuration** (.env):
```
SESSION_COOKIE_SECURE=True      # HTTPS only
SESSION_COOKIE_HTTPONLY=True    # No JavaScript access
SESSION_COOKIE_SAMESITE=Lax     # CSRF protection
PERMANENT_SESSION_LIFETIME=86400 # 24 hours
```

---

## 2. Data Protection

### A. Database Security

**PostgreSQL Hardening**:
```sql
-- Create limited user
CREATE USER life_planner WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE life_planner_db TO life_planner;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO life_planner;

-- Disable superuser access from application
REVOKE ALL ON SCHEMA public FROM public;
```

**SQL Injection Prevention**:
- SQLAlchemy ORM (parameterized queries)
- No raw SQL execution
- Input validation on all endpoints

```python
# Safe: Using ORM
User.query.filter_by(email=email).first()

# Unsafe: Never do this
# db.execute(f"SELECT * FROM users WHERE email='{email}'")
```

### B. Personal Data Encryption

**User Data Storage**:
- Personal data stored locally on user's machine (via browser)
- Server only stores: email, hashed password, subscription status
- Pet data: non-identifiable statistics
- Activity data: aggregated, anonymized patterns

**Privacy-Preserving ML**:
```python
# Extract only anonymized patterns
anonymized = {
    'stress_variance': np.var(stress_values),
    'mood_trend': calculate_trend(mood_values),
    # No raw data, no PII
}
```

### C. Federated Learning

**Implementation** (backend/gpu_extensions.py):
```python
class FederatedLearningManager:
    def aggregate_user_updates(self, user_gradients):
        # Add differential privacy noise
        noise = np.random.laplace(0, epsilon, shape)
        aggregated = avg_gradient + noise
        # Individual contributions cannot be recovered
```

**Privacy Guarantees**:
- Differential privacy (ε = 1.0)
- No access to raw user data
- Only aggregated model updates shared
- Individual user patterns indistinguishable

---

## 3. Network Security

### A. HTTPS/TLS

**Requirements**:
- TLS 1.2+ only
- Strong cipher suites
- HSTS header enabled
- Certificate from trusted CA (Let's Encrypt)

**Nginx Configuration**:
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
ssl_prefer_server_ciphers on;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### B. CORS Protection

**Configuration**:
```python
CORS(app, 
     origins=os.getenv('CORS_ORIGINS').split(','),
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization']
)
```

**Production .env**:
```
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### C. Rate Limiting

**Implementation**:
```python
# Default: 100 requests per hour
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Strict limits on sensitive endpoints
@app.route('/api/auth/register')
@limiter.limit("5 per hour")  # Prevent spam
def register():
    pass

@app.route('/api/auth/login')
@limiter.limit("10 per minute")  # Brute force protection
def login():
    pass
```

---

## 4. Input Validation

### A. Backend Validation

**Always Validate**:
```python
# Email validation
from email_validator import validate_email

try:
    v = validate_email(email)
    email = v.email  # Normalized
except EmailNotValidError:
    return jsonify({'error': 'Invalid email'}), 400

# Numeric ranges
stress = float(data.get('stress', 50))
if not 0 <= stress <= 100:
    return jsonify({'error': 'Stress must be 0-100'}), 400
```

### B. XSS Prevention

**Output Encoding**:
- Flask auto-escapes HTML in templates
- JSON responses (no HTML injection)
- CSP headers

```python
# Safe JSON responses
return jsonify({'message': user_input})  # Auto-sanitized

# If rendering HTML:
from markupsafe import escape
safe_text = escape(user_input)
```

### C. CSRF Protection

**Measures**:
- SameSite cookie attribute
- Token-based authentication (JWT)
- Origin header validation

---

## 5. Secrets Management

### A. Environment Variables

**Never Hardcode**:
❌ `SECRET_KEY = "my-secret-key"`
✅ `SECRET_KEY = os.getenv('SECRET_KEY')`

**Generate Strong Keys**:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### B. Stripe Keys

**Key Types**:
- `sk_test_*`: Development (in .env)
- `sk_live_*`: Production (in .env, never commit)
- `pk_*`: Publishable (safe for client)
- `whsec_*`: Webhook secret (server only)

**Rotation**:
- Rotate keys quarterly
- Rotate immediately if compromised
- Update .env and restart service

### C. Database Credentials

**Storage**:
```bash
# .env file (chmod 600)
DATABASE_URL=postgresql://user:password@localhost/db

# Never in version control
echo ".env" >> .gitignore
```

---

## 6. Audit Logging

### A. What We Log

**Security Events**:
```python
log_audit(user_id, 'login_success', 'auth', 'success')
log_audit(user_id, 'password_reset_requested', 'auth', 'success')
log_audit(user_id, 'subscription_activated', 'subscription', 'success')
log_audit(None, 'login_failed', 'auth', 'failure', {'email': email})
```

**Logged Information**:
- User ID (if authenticated)
- Action performed
- Resource accessed
- Status (success/failure/warning)
- IP address
- User agent
- Timestamp
- Additional details (JSON)

### B. Audit Log Review

```sql
-- Recent failed login attempts
SELECT * FROM audit_logs 
WHERE action = 'login_failed' 
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- User activity timeline
SELECT * FROM audit_logs 
WHERE user_id = 123 
ORDER BY timestamp DESC 
LIMIT 50;
```

### C. Log Retention

**Policy**:
- Keep 90 days of audit logs
- Archive older logs
- Automated cleanup job

```python
# Cleanup script (run monthly)
from datetime import datetime, timedelta

cutoff = datetime.utcnow() - timedelta(days=90)
AuditLog.query.filter(AuditLog.timestamp < cutoff).delete()
db.session.commit()
```

---

## 7. Payment Security

### A. Stripe Integration

**PCI Compliance**:
- Never store card details
- Stripe.js handles card input
- Tokens used for processing
- Webhooks verify authenticity

**Webhook Verification**:
```python
try:
    event = stripe.Webhook.construct_event(
        payload, 
        sig_header, 
        WEBHOOK_SECRET
    )
except stripe.error.SignatureVerificationError:
    return jsonify({'error': 'Invalid signature'}), 400
```

### B. Subscription Security

**Validation**:
```python
def has_active_subscription(user):
    # Check trial
    if user.is_trial_active():
        return True
    
    # Check paid subscription
    if user.subscription_status == 'active':
        if user.subscription_end_date > datetime.utcnow():
            return True
    
    return False
```

**Protection**:
- All protected endpoints check subscription
- Access denied if expired
- Clear error messages
- Redirect to payment page

---

## 8. File Upload Security

### A. Restrictions

**Implementation**:
```python
# File size limit
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### B. Storage

**Secure Path**:
```python
import os
from werkzeug.utils import secure_filename

filename = secure_filename(user_file.filename)
filepath = os.path.join(UPLOAD_FOLDER, user_id, filename)
# Never use user input directly in path
```

---

## 9. Dependency Security

### A. Regular Updates

**Weekly Check**:
```bash
pip list --outdated
pip install --upgrade -r requirements.txt
```

### B. Vulnerability Scanning

**Tools**:
```bash
# Install safety
pip install safety

# Check for vulnerabilities
safety check

# Check specific packages
pip-audit
```

### C. Dependency Pinning

```txt
# requirements.txt - pin versions
Flask==3.0.0  # Not Flask>=3.0.0
```

---

## 10. Incident Response

### A. Security Breach Protocol

**If Compromised**:

1. **Immediate Actions**:
   - Disable affected systems
   - Rotate all secrets (API keys, DB passwords)
   - Review audit logs
   - Identify affected users

2. **Investigation**:
   - Determine breach scope
   - Check what data was accessed
   - Identify vulnerability

3. **Notification**:
   - Email affected users
   - Provide timeline and details
   - Offer remediation steps

4. **Remediation**:
   - Patch vulnerability
   - Force password reset
   - Revoke all tokens
   - Update security measures

### B. Monitoring

**Automated Alerts**:
```python
# Set up monitoring for:
- Failed login attempts > 5 in 5 minutes
- Unusual payment activity
- Database connection failures
- Disk space < 10%
- High error rates
```

---

## 11. GDPR Compliance

### A. User Rights

**Implementation**:

1. **Right to Access**:
```python
@app.route('/api/user/data-export')
@jwt_required()
def export_user_data():
    # Return all user data in JSON
    pass
```

2. **Right to Deletion**:
```python
@app.route('/api/user/delete-account')
@jwt_required()
def delete_account():
    # Anonymize/delete all user data
    pass
```

3. **Right to Portability**:
- Export in JSON format
- Include all personal data
- Machine-readable

### B. Data Retention

**Policy**:
- Keep data while subscription active
- 30 days after cancellation
- Delete on user request
- Audit logs: 90 days

---

## 12. Security Checklist

### Pre-Launch

- [ ] All secrets in environment variables
- [ ] .env file in .gitignore
- [ ] HTTPS certificate installed
- [ ] Strong password policy enforced
- [ ] Rate limiting enabled
- [ ] SQL injection prevention verified
- [ ] XSS protection tested
- [ ] CSRF tokens implemented
- [ ] Input validation on all endpoints
- [ ] Audit logging enabled
- [ ] Error messages don't leak info
- [ ] Database user has minimal permissions
- [ ] Backup system configured
- [ ] Monitoring alerts configured
- [ ] Security headers set

### Post-Launch

- [ ] Weekly log reviews
- [ ] Monthly security audits
- [ ] Quarterly dependency updates
- [ ] Annual penetration testing
- [ ] Regular backup restoration tests

---

## 13. Contact

**Security Issues**:
Email: onlinediscountsllc@gmail.com
Subject: [SECURITY] Life Planner Security Issue

**Response Time**: 24 hours for critical issues

---

## 🔐 Remember

1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Minimal permissions needed
3. **Assume Breach**: Plan for when (not if) something fails
4. **User Privacy First**: Protect user data above all
5. **Regular Updates**: Security is ongoing, not one-time

**Security is everyone's responsibility!**
