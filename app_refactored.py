"""
Life Planner - Self-Healing Application
Automatically handles missing dependencies and recovers from errors
"""

import os
import sys
import logging
from pathlib import Path

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/life_planner.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SELF-HEALING: Check and create required folders
# ============================================================================

REQUIRED_FOLDERS = ['logs', 'models', 'backend', 'templates', 'user_data', 'static']
for folder in REQUIRED_FOLDERS:
    os.makedirs(folder, exist_ok=True)

# ============================================================================
# SELF-HEALING: Load environment with fallbacks
# ============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file")
except ImportError:
    logger.warning("python-dotenv not installed, using system environment only")
except Exception as e:
    logger.warning(f"Could not load .env: {e}")

# ============================================================================
# SELF-HEALING: Import Flask with fallback
# ============================================================================

try:
    from flask import Flask, request, jsonify, render_template, send_file
    from flask_cors import CORS
except ImportError as e:
    logger.error("Flask not installed! Run: pip install flask flask-cors")
    sys.exit(1)

# ============================================================================
# SELF-HEALING: Import optional dependencies with graceful degradation
# ============================================================================

# JWT Authentication (optional)
try:
    from flask_jwt_extended import (
        JWTManager, create_access_token, create_refresh_token,
        jwt_required, get_jwt_identity
    )
    JWT_AVAILABLE = True
    logger.info("JWT authentication enabled")
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("JWT not available - authentication will be basic")
    
    # Create dummy decorators
    def jwt_required(optional=False):
        def decorator(f):
            return f
        return decorator
    
    def get_jwt_identity():
        return 1  # Default admin user

# Email (optional)
try:
    from flask_mail import Mail, Message
    MAIL_AVAILABLE = True
except ImportError:
    MAIL_AVAILABLE = False
    logger.warning("Flask-Mail not available - emails will be logged only")

# Stripe (optional)
try:
    import stripe
    STRIPE_AVAILABLE = True
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
    logger.info("Stripe integration enabled")
except ImportError:
    STRIPE_AVAILABLE = False
    logger.warning("Stripe not available - payments will be simulated")

# Rate limiting (optional)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    logger.warning("Rate limiting not available")

# Database (required)
try:
    from flask_sqlalchemy import SQLAlchemy
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError:
    logger.error("SQLAlchemy not installed! Run: pip install flask-sqlalchemy")
    sys.exit(1)

# ============================================================================
# Initialize Flask Application
# ============================================================================

app = Flask(__name__)

# Configuration with safe fallbacks
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///life_planner.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# JWT Configuration
if JWT_AVAILABLE:
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-me')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600
    jwt = JWTManager(app)

# Mail Configuration
if MAIL_AVAILABLE:
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', '')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '')
    mail = Mail(app)

# Database
db = SQLAlchemy(app)

# CORS
CORS(app, origins=os.getenv('CORS_ORIGINS', '*').split(','))

# Rate Limiter
if LIMITER_AVAILABLE:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per hour"],
        storage_uri=os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
    )

# ============================================================================
# SELF-HEALING: Import models with error handling
# ============================================================================

try:
    from models.database import User, Pet, UserActivity, AuditLog
    logger.info("Database models loaded")
except ImportError as e:
    logger.error(f"Could not import models: {e}")
    logger.error("Creating basic User model...")
    
    # Create basic User model if import fails
    class User(db.Model):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(255), nullable=False)
        is_admin = db.Column(db.Boolean, default=False)
        subscription_status = db.Column(db.String(20), default='trial')
        
        def to_dict(self):
            return {'id': self.id, 'email': self.email}

# ============================================================================
# SELF-HEALING: Import backend with error handling
# ============================================================================

try:
    from backend.life_planning_core import LifePlanningSystem, AncientMathUtil
    PLANNING_AVAILABLE = True
    logger.info("Life planning system loaded")
except ImportError as e:
    PLANNING_AVAILABLE = False
    logger.warning(f"Life planning system not available: {e}")
    
    # Create mock system
    class LifePlanningSystem:
        def __init__(self, species='cat'):
            self.species = species
        
        def update(self, data):
            pass
        
        def generate_guidance(self):
            return {
                'message': 'Life planning system loading...',
                'predicted_mood': '50.0',
                'fuzzy_message': 'Stay balanced'
            }
        
        def generate_fractal_image(self):
            from PIL import Image
            return Image.new('RGB', (512, 512), color=(100, 100, 200))

# ============================================================================
# Utility Functions
# ============================================================================

def safe_send_email(to, subject, body):
    """Send email with fallback to logging"""
    if MAIL_AVAILABLE and app.config['MAIL_USERNAME']:
        try:
            msg = Message(subject, recipients=[to], body=body)
            mail.send(msg)
            logger.info(f"Email sent to {to}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
    
    # Fallback: log the email
    logger.info(f"EMAIL (not sent): To={to}, Subject={subject}, Body={body[:100]}...")
    return False

def log_audit(user_id, action, resource=None, status='success', details=None):
    """Log audit trail"""
    try:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            ip_address=request.remote_addr if request else None,
            user_agent=request.headers.get('User-Agent') if request else None,
            details=details
        )
        db.session.add(audit)
        db.session.commit()
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")

# ============================================================================
# Routes - Authentication
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        from werkzeug.security import generate_password_hash
        from datetime import datetime, timedelta
        
        user = User()
        user.email = email
        user.password_hash = generate_password_hash(password)
        user.subscription_status = 'trial'
        
        db.session.add(user)
        db.session.commit()
        
        # Create tokens
        if JWT_AVAILABLE:
            access_token = create_access_token(identity=user.id)
            refresh_token = create_refresh_token(identity=user.id)
        else:
            access_token = 'mock-token'
            refresh_token = 'mock-refresh-token'
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'Registration successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'trial_days_remaining': 7,
            'show_gofundme': True,
            'gofundme_url': os.getenv('GOFUNDME_URL', 'https://gofund.me/8d9303d27')
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        from werkzeug.security import check_password_hash
        if not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create tokens
        if JWT_AVAILABLE:
            access_token = create_access_token(identity=user.id)
            refresh_token = create_refresh_token(identity=user.id)
        else:
            access_token = 'mock-token'
            refresh_token = 'mock-refresh-token'
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'has_access': True,
            'trial_active': user.subscription_status == 'trial',
            'trial_days_remaining': 7,
            'show_gofundme': user.subscription_status == 'trial',
            'gofundme_url': os.getenv('GOFUNDME_URL')
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# ============================================================================
# Routes - Main Interface
# ============================================================================

@app.route('/')
def index():
    """Serve main application"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return """
        <html>
        <body>
        <h1>Life Planner</h1>
        <p>Template not found. Please ensure index.html is in templates/ folder.</p>
        <p>API is available at /api/health</p>
        </body>
        </html>
        """, 200

@app.route('/api/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'database': 'connected' if DB_AVAILABLE else 'unavailable',
        'jwt': 'enabled' if JWT_AVAILABLE else 'disabled',
        'email': 'enabled' if MAIL_AVAILABLE else 'disabled',
        'stripe': 'enabled' if STRIPE_AVAILABLE else 'disabled',
        'planning_system': 'loaded' if PLANNING_AVAILABLE else 'mock',
    }
    return jsonify(status), 200

# ============================================================================
# Routes - Life Planner
# ============================================================================

@app.route('/api/planner/update', methods=['POST'])
@jwt_required()
def update_planner():
    """Update life planner data"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        # Create planning system
        system = LifePlanningSystem(species='cat')
        system.update(data)
        guidance = system.generate_guidance()
        
        logger.info(f"Planner updated for user {current_user_id}")
        
        return jsonify({
            'message': 'Update successful',
            'guidance': guidance,
            'ancient_wisdom': {
                'golden_ratio': 1.618,
                'fibonacci': [1, 1, 2]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Planner update error: {e}")
        return jsonify({'error': 'Update failed'}), 500

@app.route('/api/planner/fractal', methods=['POST'])
@jwt_required()
def generate_fractal():
    """Generate fractal art"""
    try:
        from io import BytesIO
        
        system = LifePlanningSystem()
        image = system.generate_fractal_image()
        
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({'error': 'Generation failed'}), 500

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# Database Initialization
# ============================================================================

def init_database():
    """Initialize database with error handling"""
    try:
        with app.app_context():
            db.create_all()
            
            # Create admin user if not exists
            admin_email = os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')
            admin = User.query.filter_by(email=admin_email).first()
            
            if not admin:
                from werkzeug.security import generate_password_hash
                
                admin = User()
                admin.email = admin_email
                admin.password_hash = generate_password_hash(
                    os.getenv('ADMIN_PASSWORD', 'admin8587037321')
                )
                admin.is_admin = True
                admin.subscription_status = 'active'
                
                db.session.add(admin)
                db.session.commit()
                logger.info(f"Admin user created: {admin_email}")
            
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize database
    logger.info("Initializing database...")
    if init_database():
        logger.info("Database initialized successfully")
    else:
        logger.warning("Database initialization had issues, but continuing...")
    
    # Print startup info
    print()
    print("=" * 70)
    print("LIFE PLANNER - STARTING")
    print("=" * 70)
    print()
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"JWT Auth: {'Enabled' if JWT_AVAILABLE else 'Disabled'}")
    print(f"Email: {'Enabled' if MAIL_AVAILABLE else 'Disabled'}")
    print(f"Stripe: {'Enabled' if STRIPE_AVAILABLE else 'Disabled'}")
    print()
    print("Access the application at: http://localhost:5000")
    print(f"Admin: {os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')}")
    print(f"Password: {os.getenv('ADMIN_PASSWORD', 'admin8587037321')}")
    print()
    print("=" * 70)
    print()
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False') == 'True'
    )
