# 🌟 Life Planner - AI-Powered Life Companion

A comprehensive life planning application powered by ancient mathematics, modern AI, and a virtual pet companion. Built with security, privacy, and scalability in mind.

## 📋 Overview

Life Planner combines:
- **Ancient Mathematics**: Golden ratio, Fibonacci sequences, chaos theory
- **Modern AI**: Machine learning predictions, federated learning
- **Virtual Pet System**: Gamified life planning with an evolving companion
- **Beautiful Fractals**: Personalized art generated from your life data
- **Subscription Model**: 7-day free trial, then $20/month

## ✨ Features

### Core Features
- 🎯 **AI-Powered Guidance**: Predictive mood analysis and personalized recommendations
- 🐉 **Virtual Pet Companion**: Grows and evolves based on your progress
- 🎨 **Fractal Art Generation**: Unique artwork reflecting your life patterns
- 📊 **Progress Tracking**: Monitor stress, mood, sleep, and goals
- 🔮 **Fuzzy Logic Engine**: Human-like reasoning for better advice
- 📈 **Ancient Math Integration**: 500+ year old mathematical wisdom

### Technical Features
- 🔒 **Enterprise Security**: JWT authentication, bcrypt passwords, audit logging
- 💳 **Stripe Integration**: Secure payment processing
- 🚀 **GPU Acceleration**: Fast fractal generation with CPU fallback
- 🤖 **Federated Learning**: Privacy-preserving AI that learns from all users
- 📧 **Email System**: Password reset, verification, notifications
- 🔄 **Rate Limiting**: DDoS protection and abuse prevention
- 🗄️ **PostgreSQL Database**: Scalable, reliable data storage
- 🎨 **Modern UI**: Responsive design with beautiful gradients

## 🏗️ Architecture

```
life_planner_app/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.template              # Environment configuration template
├── init_db.py                 # Database setup script
│
├── models/
│   └── database.py            # SQLAlchemy models
│
├── backend/
│   ├── life_planning_core.py  # Core planning system
│   └── gpu_extensions.py      # GPU acceleration & ML
│
├── templates/
│   └── index.html             # Frontend interface
│
├── logs/                      # Application logs
├── DEPLOYMENT.md              # Deployment guide
├── SECURITY.md                # Security documentation
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 13+ (or SQLite for testing)
- Redis 6+
- Stripe account
- Email service (Gmail/SMTP)

### Installation

1. **Clone/Extract the project**
```bash
cd life_planner_app
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.template .env
# Edit .env with your settings
```

5. **Initialize database**
```bash
python init_db.py
# Choose option 1: Initialize Database
```

6. **Run the application**
```bash
python app.py
```

7. **Access the app**
```
Open browser: http://localhost:5000
```

## ⚙️ Configuration

### Essential Environment Variables

```env
# Application
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@localhost/life_planner_db

# JWT
JWT_SECRET_KEY=your-jwt-secret-here

# Stripe
STRIPE_SECRET_KEY=sk_live_your_key
STRIPE_PUBLISHABLE_KEY=pk_live_your_key
STRIPE_PRICE_ID=price_your_price_id
SUBSCRIPTION_PRICE=20.00
TRIAL_DAYS=7

# Email
MAIL_SERVER=smtp.gmail.com
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password

# Admin
ADMIN_EMAIL=onlinediscountsllc@gmail.com
ADMIN_PASSWORD=admin8587037321

# GoFundMe
GOFUNDME_URL=https://gofund.me/8d9303d27
```

See `.env.template` for complete configuration options.

## 📚 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete production deployment guide
- **[SECURITY.md](SECURITY.md)** - Security best practices and implementation
- **API Documentation** - Available at `/api/docs` (when enabled)

## 🔐 Security Features

### Authentication
- PBKDF2-SHA256 password hashing
- JWT tokens with refresh mechanism
- Email verification
- Password reset with time-limited tokens
- Rate limiting on sensitive endpoints

### Data Protection
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection (auto-escaping)
- CSRF protection (SameSite cookies)
- HTTPS/TLS encryption
- Audit logging for all actions

### Privacy
- Personal data stored locally
- Federated learning (no raw data sharing)
- Differential privacy (ε = 1.0)
- GDPR compliance ready
- User data export/deletion

## 💳 Subscription Flow

1. **Registration**: User signs up → 7-day free trial starts
2. **Trial Period**: Full access + GoFundMe banner shown
3. **Trial Ends**: User must subscribe for $20/month to continue
4. **Active Subscription**: Full access, no ads
5. **Cancellation**: Access until end of billing period

## 🎨 Virtual Pet System

### Species Available
- 🐱 Cat (balanced, friendly)
- 🐉 Dragon (powerful, chaotic)
- 🔥 Phoenix (resilient, passionate)
- 🦉 Owl (wise, calm)
- 🦊 Fox (clever, energetic)

### Pet Mechanics
- **Hunger**: Feed your pet regularly
- **Energy**: Play and rest management
- **Mood**: Influenced by your mood
- **Growth**: Levels up with your progress
- **Bond**: Strengthens through interaction

## 🧮 Ancient Mathematics

### Implemented Algorithms
- **Golden Ratio (Φ)**: Divine proportion (1.618...)
- **Fibonacci Sequence**: Natural growth patterns
- **Logistic Map**: Chaos theory dynamics
- **Archimedes Spiral**: Ancient Greek geometry
- **Islamic Star Patterns**: 8th-15th century art
- **Pythagorean Means**: Arithmetic, geometric, harmonic

### Applications
- Fractal generation parameters
- Pet behavior modulation
- Guidance system weighting
- Art composition rules

## 🤖 Machine Learning

### Predictive Models
- Decision tree regressor for mood prediction
- Fuzzy logic for personalized guidance
- Time series analysis for trend detection

### Federated Learning
- Privacy-preserving aggregation
- Differential privacy noise injection
- No raw data ever leaves user's control
- Global model improves from all users

## 🎨 Fractal Art Generator

### Techniques
- Mandelbrot set computation
- Julia set rendering
- Noise injection for variation
- Radial symmetry (kaleidoscope effect)
- Species-specific modifiers
- Behavioral state mapping

### Parameters
- Zoom level (based on growth)
- Iteration depth (based on goals)
- Color palette (based on mood)
- Symmetry (based on balance)

## 📊 Admin Dashboard

Access at: `https://yourdomain.com` (login as admin)

**Default Credentials**:
- Email: `onlinediscountsllc@gmail.com`
- Password: `admin8587037321`

**⚠️ CHANGE PASSWORD IMMEDIATELY AFTER FIRST LOGIN**

### Admin Features
- User statistics
- Revenue tracking
- Recent signups
- Activity monitoring
- Audit log review

## 🚦 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh token
- `POST /api/auth/forgot-password` - Password reset request
- `POST /api/auth/reset-password` - Reset password

### Subscription
- `POST /api/subscription/create-checkout` - Create Stripe checkout
- `POST /api/subscription/webhook` - Stripe webhook handler
- `GET /api/subscription/status` - Get subscription status

### Life Planner
- `POST /api/planner/update` - Update daily check-in
- `POST /api/planner/fractal` - Generate fractal art
- `GET /api/pet` - Get pet information
- `POST /api/pet/feed` - Feed pet
- `POST /api/pet/play` - Play with pet

### Admin
- `GET /api/admin/dashboard` - Admin dashboard data

### Health
- `GET /api/health` - Application health check

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Manual Testing
```bash
# Test authentication
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"testpass"}'
```

## 📈 Scaling

### For Growth
1. **Database**: Use PostgreSQL with read replicas
2. **Cache**: Redis for session storage
3. **Load Balancer**: Nginx or HAProxy
4. **Workers**: Gunicorn with multiple processes
5. **CDN**: CloudFlare for static assets
6. **Monitoring**: Prometheus + Grafana

### Performance Tips
- Enable GPU acceleration for fractal generation
- Use Redis for rate limiting
- Optimize database queries with indexes
- Enable caching for static content
- Use connection pooling

## 🔧 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify DATABASE_URL in .env
```

**Stripe Webhook Failing**
```bash
# Verify webhook URL is accessible
# Check STRIPE_WEBHOOK_SECRET
# Review webhook logs in Stripe Dashboard
```

**Email Not Sending**
```bash
# For Gmail: Enable 2FA + App Password
# Check MAIL_USERNAME and MAIL_PASSWORD
# Verify port 587 is not blocked
```

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 License

This software is proprietary. All rights reserved.

Owner: Luke Smith (onlinediscountsllc@gmail.com)

## 🤝 Support

For issues, questions, or feature requests:

**Email**: onlinediscountsllc@gmail.com
**GoFundMe**: https://gofund.me/8d9303d27

## 🎯 Roadmap

### Version 1.1 (Planned)
- [ ] Mobile app (iOS/Android)
- [ ] Social features (friend pets)
- [ ] Advanced analytics dashboard
- [ ] More pet species
- [ ] Custom fractal parameters
- [ ] Export reports as PDF

### Version 2.0 (Future)
- [ ] AI chat companion
- [ ] Integration with fitness trackers
- [ ] Group challenges
- [ ] Pet battles/minigames
- [ ] Marketplace for custom items
- [ ] API for third-party apps

## 🙏 Acknowledgments

- Ancient mathematicians for timeless wisdom
- Anthropic for Claude AI assistance
- Open source community for amazing tools
- Early users for valuable feedback

## 📞 Contact

**Owner**: Luke Smith
**Email**: onlinediscountsllc@gmail.com
**Support Our Mission**: https://gofund.me/8d9303d27

---

**Built with ❤️ using ancient mathematics and modern AI**

*Empowering users to plan better lives, one day at a time.*
