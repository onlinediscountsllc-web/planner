# Life Planner Application - Deployment Guide

## 🚀 Production Deployment Checklist

### Prerequisites
- Python 3.9 or higher
- PostgreSQL 13+ (or SQLite for development)
- Redis 6+ (for rate limiting and caching)
- Stripe account with API keys
- Email service (Gmail/SMTP)
- SSL certificate (Let's Encrypt recommended)

---

## 1. Environment Setup

### A. Clone and Install Dependencies

```bash
# Clone the repository (or extract the files)
cd life_planner_app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### B. Configure Environment Variables

```bash
# Copy template and edit
cp .env.template .env
nano .env  # or use your preferred editor
```

**CRITICAL: Update these values in .env:**

1. **Secret Keys** (generate strong random strings):
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. **Database URL**:
```
# For PostgreSQL (recommended):
DATABASE_URL=postgresql://username:password@localhost:5432/life_planner_db

# For SQLite (development only):
DATABASE_URL=sqlite:///life_planner.db
```

3. **Stripe Configuration**:
   - Login to [Stripe Dashboard](https://dashboard.stripe.com/)
   - Get API keys from Developers > API keys
   - Create a subscription product and get the price ID
   - Set up webhook endpoint and get webhook secret

4. **Email Configuration**:
   - For Gmail: Enable 2FA and create an App Password
   - Update MAIL_USERNAME and MAIL_PASSWORD

---

## 2. Database Setup

### A. PostgreSQL Installation (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql

postgres=# CREATE DATABASE life_planner_db;
postgres=# CREATE USER your_username WITH PASSWORD 'strong_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE life_planner_db TO your_username;
postgres=# \q
```

### B. Initialize Database

```bash
# Run migrations
python app.py

# This will:
# - Create all tables
# - Create admin user (onlinediscountsllc@gmail.com / admin8587037321)
# - Initialize system settings
```

### C. Verify Database

```bash
# Connect to database
psql -U your_username -d life_planner_db

# Check tables
\dt

# Should see: users, pets, user_activities, ml_data, system_settings, audit_logs
```

---

## 3. Redis Setup

### A. Install Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Verify
redis-cli ping
# Should return: PONG
```

---

## 4. Stripe Integration

### A. Setup Steps

1. **Create Subscription Product**:
   - Go to Stripe Dashboard > Products
   - Create new product: "Life Planner Monthly"
   - Set price: $20/month recurring
   - Copy the Price ID (starts with price_...)

2. **Setup Webhook**:
   - Go to Developers > Webhooks
   - Add endpoint: `https://yourdomain.com/api/subscription/webhook`
   - Select events: 
     - `checkout.session.completed`
     - `customer.subscription.deleted`
     - `invoice.payment_failed`
   - Copy Webhook Signing Secret

3. **Update .env**:
```
STRIPE_SECRET_KEY=sk_live_your_secret_key
STRIPE_PUBLISHABLE_KEY=pk_live_your_publishable_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
STRIPE_PRICE_ID=price_your_price_id
```

---

## 5. Security Configuration

### A. Generate Secure Admin Password

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('admin8587037321'))"
```

Update `ADMIN_PASSWORD_HASH` in .env with output.

### B. SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### C. Firewall Configuration

```bash
# Allow SSH, HTTP, HTTPS
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

---

## 6. Application Deployment

### A. Using Gunicorn + Nginx

1. **Install Nginx**:
```bash
sudo apt install nginx
```

2. **Create Nginx Configuration**:
```bash
sudo nano /etc/nginx/sites-available/life_planner
```

Add configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name yourdomain.com www.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    client_max_body_size 16M;
}
```

3. **Enable Site**:
```bash
sudo ln -s /etc/nginx/sites-available/life_planner /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

4. **Create Systemd Service**:
```bash
sudo nano /etc/systemd/system/life_planner.service
```

Add:
```ini
[Unit]
Description=Life Planner Application
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/life_planner_app
Environment="PATH=/path/to/life_planner_app/venv/bin"
ExecStart=/path/to/life_planner_app/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
```

5. **Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl start life_planner
sudo systemctl enable life_planner
sudo systemctl status life_planner
```

---

## 7. Monitoring & Logging

### A. Log Files

Logs are stored in `logs/life_planner.log`

```bash
# View logs
tail -f logs/life_planner.log

# Rotate logs (cron job)
sudo nano /etc/logrotate.d/life_planner
```

Add:
```
/path/to/life_planner_app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### B. Health Monitoring

```bash
# Check application health
curl https://yourdomain.com/api/health
```

---

## 8. Backup Strategy

### A. Database Backups

```bash
# Create backup script
nano backup.sh
```

Add:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/life_planner"
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U your_username life_planner_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +30 -delete
```

```bash
chmod +x backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * /path/to/backup.sh
```

---

## 9. Scaling Considerations

### A. For Multiple Servers

1. **Use PostgreSQL** (not SQLite)
2. **Shared Redis** instance
3. **Load Balancer** (Nginx/HAProxy)
4. **Separate workers** for background tasks (Celery)

### B. GPU Optimization

For users with GPU:
- CUDA 11.0+ required
- PyTorch will auto-detect and use GPU
- Set `USE_GPU=True` in .env

For CPU-only:
- Set `USE_GPU=False`
- Performance is still good for fractals

---

## 10. Testing Before Launch

```bash
# Run all tests
pytest tests/

# Test authentication
curl -X POST https://yourdomain.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"testpass"}'

# Test health endpoint
curl https://yourdomain.com/api/health
```

---

## 11. Admin Dashboard Access

**URL**: `https://yourdomain.com`

**Admin Credentials**:
- Email: onlinediscountsllc@gmail.com
- Password: admin8587037321

**IMPORTANT**: Change the admin password immediately after first login!

---

## 12. Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check PostgreSQL is running: `sudo systemctl status postgresql`
   - Verify DATABASE_URL in .env
   - Check database permissions

2. **Email Not Sending**:
   - Verify SMTP settings
   - For Gmail: Enable 2FA and use App Password
   - Check firewall allows port 587

3. **Stripe Webhook Failing**:
   - Verify webhook URL is accessible
   - Check STRIPE_WEBHOOK_SECRET is correct
   - View webhook attempts in Stripe Dashboard

4. **Rate Limiting Issues**:
   - Check Redis is running: `redis-cli ping`
   - Verify REDIS_URL in .env

---

## 13. Maintenance

### Regular Tasks

1. **Weekly**:
   - Review logs for errors
   - Check database size
   - Monitor subscription revenue

2. **Monthly**:
   - Update dependencies: `pip install --upgrade -r requirements.txt`
   - Review and clean audit logs
   - Analyze user engagement metrics

3. **Quarterly**:
   - Security audit
   - Performance optimization
   - Backup restoration test

---

## 14. Support & Updates

For issues:
1. Check logs: `logs/life_planner.log`
2. Review audit logs in database
3. Contact: onlinediscountsllc@gmail.com

---

## 🎉 Launch Checklist

- [ ] Environment variables configured
- [ ] Database initialized with admin user
- [ ] Redis running
- [ ] Stripe configured and tested
- [ ] SSL certificate installed
- [ ] Email sending verified
- [ ] Backup system configured
- [ ] Monitoring enabled
- [ ] Test subscription flow
- [ ] Review security settings
- [ ] GoFundMe link verified
- [ ] Admin access confirmed

---

**Your Life Planner is ready to launch!** 🚀

Users will get:
- 7-day free trial
- GoFundMe banner during trial
- $20/month subscription after trial
- AI-powered life planning
- Virtual pet companion
- Beautiful fractal art
- Ancient mathematics insights
