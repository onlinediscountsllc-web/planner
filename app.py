"""
Life Planner Application - Main Flask Application
Includes authentication, payment processing, and security features
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import stripe
import os
import logging
from dotenv import load_dotenv
import json
from io import BytesIO

# Load environment variables
load_dotenv()

# Import models and core functionality
from models.database import (
    db, User, Pet, UserActivity, MLData, AuditLog, SystemSettings
)
from backend.life_planning_core import LifePlanningSystem, AncientMathUtil
from backend.gpu_extensions import (
    GPUAcceleratedFractalGenerator, FederatedLearningManager,
    AncientMathEnhanced, MemoryManager
)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///life_planner.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'change-this-jwt-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Stripe configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID')
SUBSCRIPTION_PRICE = float(os.getenv('SUBSCRIPTION_PRICE', 20.00))
TRIAL_DAYS = int(os.getenv('TRIAL_DAYS', 7))

# GoFundMe configuration
GOFUNDME_URL = os.getenv('GOFUNDME_URL', 'https://gofund.me/8d9303d27')

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)
mail = Mail(app)
CORS(app, origins=os.getenv('CORS_ORIGINS', '*').split(','))

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri=os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
)

# Logging configuration
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'logs/life_planner.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize GPU and federated learning
gpu_generator = GPUAcceleratedFractalGenerator(use_gpu=os.getenv('USE_GPU', 'True') == 'True')
federated_manager = FederatedLearningManager()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_audit(user_id, action, resource=None, status='success', details=None):
    """Log security audit trail"""
    try:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            details=details
        )
        db.session.add(audit)
        db.session.commit()
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")


def send_email(to, subject, body):
    """Send email notification"""
    try:
        msg = Message(subject, recipients=[to], body=body)
        mail.send(msg)
        logger.info(f"Email sent to {to}: {subject}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    """Register new user with 7-day free trial"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        # Validation
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            subscription_status='trial',
            trial_start_date=datetime.utcnow(),
            trial_end_date=datetime.utcnow() + timedelta(days=TRIAL_DAYS)
        )
        user.set_password(password)
        verification_token = user.generate_verification_token()
        
        db.session.add(user)
        db.session.commit()
        
        # Create default pet
        pet = Pet(user_id=user.id, name='Buddy', species='cat')
        db.session.add(pet)
        db.session.commit()
        
        # Send verification email
        verification_link = f"{request.host_url}api/auth/verify/{verification_token}"
        send_email(
            to=email,
            subject='Welcome to Life Planner - Verify Your Email',
            body=f"""Welcome to Life Planner!

Your 7-day free trial has started. Click the link below to verify your email:

{verification_link}

After your trial ends, continue for just ${SUBSCRIPTION_PRICE}/month.

Best regards,
Life Planner Team
"""
        )
        
        log_audit(user.id, 'user_registered', 'user', 'success')
        
        # Create tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        
        return jsonify({
            'message': 'Registration successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'trial_days_remaining': TRIAL_DAYS,
            'show_gofundme': True,
            'gofundme_url': GOFUNDME_URL
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            log_audit(None, 'login_failed', 'auth', 'failure', {'email': email})
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account disabled'}), 403
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Check subscription status
        has_access = user.has_active_subscription()
        trial_active = user.is_trial_active()
        days_remaining = 0
        
        if trial_active:
            days_remaining = (user.trial_end_date - datetime.utcnow()).days
        
        # Create tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        
        log_audit(user.id, 'login_success', 'auth', 'success')
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(include_sensitive=True),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'has_access': has_access,
            'trial_active': trial_active,
            'trial_days_remaining': days_remaining,
            'show_gofundme': trial_active,
            'gofundme_url': GOFUNDME_URL if trial_active else None
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = get_jwt_identity()
    access_token = create_access_token(identity=current_user_id)
    return jsonify({'access_token': access_token}), 200


@app.route('/api/auth/forgot-password', methods=['POST'])
@limiter.limit("3 per hour")
def forgot_password():
    """Request password reset"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            reset_token = user.generate_reset_token(expires_in=3600)
            db.session.commit()
            
            reset_link = f"{request.host_url}reset-password/{reset_token}"
            send_email(
                to=email,
                subject='Life Planner - Password Reset Request',
                body=f"""You requested a password reset.

Click the link below to reset your password (expires in 1 hour):

{reset_link}

If you didn't request this, please ignore this email.

Best regards,
Life Planner Team
"""
            )
            
            log_audit(user.id, 'password_reset_requested', 'auth', 'success')
        
        # Always return success to prevent email enumeration
        return jsonify({'message': 'If email exists, reset link sent'}), 200
        
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        return jsonify({'error': 'Request failed'}), 500


@app.route('/api/auth/reset-password', methods=['POST'])
@limiter.limit("5 per hour")
def reset_password():
    """Reset password with token"""
    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('password')
        
        if not token or not new_password:
            return jsonify({'error': 'Token and password required'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        user = User.query.filter_by(reset_token=token).first()
        
        if not user or not user.verify_reset_token(token):
            return jsonify({'error': 'Invalid or expired token'}), 400
        
        user.set_password(new_password)
        user.reset_token = None
        user.reset_token_expires = None
        db.session.commit()
        
        log_audit(user.id, 'password_reset_completed', 'auth', 'success')
        
        return jsonify({'message': 'Password reset successful'}), 200
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return jsonify({'error': 'Reset failed'}), 500


# ============================================================================
# SUBSCRIPTION & PAYMENT ENDPOINTS
# ============================================================================

@app.route('/api/subscription/create-checkout', methods=['POST'])
@jwt_required()
def create_checkout_session():
    """Create Stripe checkout session for subscription"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Create or retrieve Stripe customer
        if not user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={'user_id': user.id}
            )
            user.stripe_customer_id = customer.id
            db.session.commit()
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer=user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1
            }],
            mode='subscription',
            success_url=f"{request.host_url}subscription-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{request.host_url}subscription-cancel",
            metadata={'user_id': user.id}
        )
        
        log_audit(user.id, 'checkout_session_created', 'subscription', 'success')
        
        return jsonify({
            'checkout_url': checkout_session.url,
            'session_id': checkout_session.id
        }), 200
        
    except Exception as e:
        logger.error(f"Checkout session error: {e}")
        return jsonify({'error': 'Checkout creation failed'}), 500


@app.route('/api/subscription/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
        )
    except ValueError as e:
        logger.error(f"Invalid webhook payload: {e}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {e}")
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle different event types
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session['metadata'].get('user_id')
        
        if user_id:
            user = User.query.get(int(user_id))
            if user:
                user.subscription_status = 'active'
                user.subscription_start_date = datetime.utcnow()
                user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
                user.stripe_subscription_id = session.get('subscription')
                db.session.commit()
                logger.info(f"Subscription activated for user {user_id}")
                log_audit(user_id, 'subscription_activated', 'subscription', 'success')
    
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        user = User.query.filter_by(stripe_subscription_id=subscription['id']).first()
        if user:
            user.subscription_status = 'cancelled'
            db.session.commit()
            logger.info(f"Subscription cancelled for user {user.id}")
            log_audit(user.id, 'subscription_cancelled', 'subscription', 'success')
    
    elif event['type'] == 'invoice.payment_failed':
        invoice = event['data']['object']
        user = User.query.filter_by(stripe_customer_id=invoice['customer']).first()
        if user:
            user.subscription_status = 'past_due'
            db.session.commit()
            logger.warning(f"Payment failed for user {user.id}")
            log_audit(user.id, 'payment_failed', 'subscription', 'warning')
    
    return jsonify({'status': 'success'}), 200


@app.route('/api/subscription/status', methods=['GET'])
@jwt_required()
def subscription_status():
    """Get current subscription status"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    trial_active = user.is_trial_active()
    has_access = user.has_active_subscription()
    
    days_remaining = 0
    if trial_active:
        days_remaining = (user.trial_end_date - datetime.utcnow()).days
    
    return jsonify({
        'subscription_status': user.subscription_status,
        'trial_active': trial_active,
        'trial_days_remaining': days_remaining,
        'has_access': has_access,
        'show_gofundme': trial_active,
        'gofundme_url': GOFUNDME_URL if trial_active else None,
        'subscription_price': SUBSCRIPTION_PRICE
    }), 200


# ============================================================================
# LIFE PLANNER CORE ENDPOINTS
# ============================================================================

@app.route('/api/planner/update', methods=['POST'])
@jwt_required()
def update_planner():
    """Update life planner with new user data"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check access
        if not user.has_active_subscription():
            return jsonify({
                'error': 'Subscription required',
                'trial_expired': not user.is_trial_active(),
                'subscription_url': '/api/subscription/create-checkout'
            }), 403
        
        data = request.get_json()
        
        # Save activity
        activity = UserActivity(
            user_id=user.id,
            stress=data.get('stress'),
            mood=data.get('mood'),
            energy=data.get('energy', 50),
            goals_completed=data.get('goals_completed', 0),
            sleep_hours=data.get('sleep_hours'),
            nutrition_score=data.get('nutrition_score'),
            period=data.get('period', 'daily')
        )
        db.session.add(activity)
        
        # Update pet
        pet = user.pet
        if pet:
            pet.hunger = min(100, pet.hunger + 5)
            pet.energy = max(0, pet.energy - 2)
            pet.mood = data.get('mood', pet.mood)
            pet.stress = data.get('stress', pet.stress)
            
            # Gain experience
            if data.get('goals_completed', 0) > 0:
                pet.experience += data.get('goals_completed', 0) * 10
                pet.total_tasks_completed += data.get('goals_completed', 0)
                
                # Level up
                while pet.experience >= (pet.level + 1) * 100:
                    pet.experience -= (pet.level + 1) * 100
                    pet.level += 1
            
            pet.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Generate guidance using ancient math
        phi = AncientMathUtil.golden_ratio()
        fib = AncientMathUtil.fibonacci_sequence(5)
        
        # Create life planning system instance
        system = LifePlanningSystem(species=pet.species if pet else 'cat')
        system.update(data)
        guidance = system.generate_guidance()
        
        log_audit(user.id, 'planner_updated', 'planner', 'success')
        
        return jsonify({
            'message': 'Update successful',
            'guidance': guidance,
            'pet': pet.to_dict() if pet else None,
            'ancient_wisdom': {
                'golden_ratio': phi,
                'fibonacci': fib[:3]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Planner update error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Update failed'}), 500


@app.route('/api/planner/fractal', methods=['POST'])
@jwt_required()
def generate_fractal():
    """Generate personalized fractal artwork"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user or not user.has_active_subscription():
            return jsonify({'error': 'Access denied'}), 403
        
        # Get recent activities
        recent = UserActivity.query.filter_by(user_id=user.id).order_by(
            UserActivity.timestamp.desc()
        ).limit(10).all()
        
        if not recent:
            return jsonify({'error': 'No activity data'}), 400
        
        # Prepare data for fractal generation
        user_data = {
            'stress': recent[0].stress,
            'mood': recent[0].mood,
            'goals_completed': recent[0].goals_completed,
            'sleep_hours': recent[0].sleep_hours,
            'nutrition_score': recent[0].nutrition_score,
            'period': recent[0].period
        }
        
        pet = user.pet
        pet_state = pet.to_dict() if pet else {}
        
        # Generate fractal
        system = LifePlanningSystem(species=pet.species if pet else 'cat')
        system.update(user_data)
        image = system.generate_fractal_image()
        
        # Convert to bytes
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        log_audit(user.id, 'fractal_generated', 'planner', 'success')
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Fractal generation error: {e}")
        return jsonify({'error': 'Generation failed'}), 500


# ============================================================================
# PET ENDPOINTS
# ============================================================================

@app.route('/api/pet', methods=['GET'])
@jwt_required()
def get_pet():
    """Get pet information"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    pet = user.pet
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    return jsonify(pet.to_dict()), 200


@app.route('/api/pet/feed', methods=['POST'])
@jwt_required()
def feed_pet():
    """Feed the virtual pet"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user or not user.has_active_subscription():
            return jsonify({'error': 'Access denied'}), 403
        
        pet = user.pet
        if not pet:
            return jsonify({'error': 'Pet not found'}), 404
        
        # Feed pet
        pet.hunger = max(0, pet.hunger - 20)
        pet.mood = min(100, pet.mood + 5)
        pet.last_fed = datetime.utcnow()
        pet.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        log_audit(user.id, 'pet_fed', 'pet', 'success')
        
        return jsonify({
            'message': 'Pet fed successfully',
            'pet': pet.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Pet feed error: {e}")
        return jsonify({'error': 'Feed failed'}), 500


@app.route('/api/pet/play', methods=['POST'])
@jwt_required()
def play_with_pet():
    """Play with the virtual pet"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user or not user.has_active_subscription():
            return jsonify({'error': 'Access denied'}), 403
        
        pet = user.pet
        if not pet:
            return jsonify({'error': 'Pet not found'}), 404
        
        # Play with pet
        if pet.energy < 20:
            return jsonify({'error': 'Pet too tired to play'}), 400
        
        pet.energy = max(0, pet.energy - 15)
        pet.mood = min(100, pet.mood + 10)
        pet.bond = min(100, pet.bond + 2)
        pet.last_played = datetime.utcnow()
        pet.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        log_audit(user.id, 'pet_played', 'pet', 'success')
        
        return jsonify({
            'message': 'Play time successful',
            'pet': pet.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Pet play error: {e}")
        return jsonify({'error': 'Play failed'}), 500


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.route('/api/admin/dashboard', methods=['GET'])
@jwt_required()
def admin_dashboard():
    """Admin dashboard - owner only"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get statistics
    total_users = User.query.count()
    active_subscriptions = User.query.filter_by(subscription_status='active').count()
    trial_users = User.query.filter_by(subscription_status='trial').count()
    
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    recent_activities = UserActivity.query.order_by(
        UserActivity.timestamp.desc()
    ).limit(20).all()
    
    return jsonify({
        'statistics': {
            'total_users': total_users,
            'active_subscriptions': active_subscriptions,
            'trial_users': trial_users,
            'monthly_revenue': active_subscriptions * SUBSCRIPTION_PRICE
        },
        'recent_users': [u.to_dict() for u in recent_users],
        'recent_activities': [a.to_dict() for a in recent_activities]
    }), 200


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'gpu_available': gpu_generator.use_gpu
    }), 200


# ============================================================================
# ERROR HANDLERS
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
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database and create admin user"""
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin_email = os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')
        admin = User.query.filter_by(email=admin_email).first()
        
        if not admin:
            admin = User(
                email=admin_email,
                first_name='Luke',
                last_name='Smith',
                is_admin=True,
                is_active=True,
                email_verified=True,
                subscription_status='active'
            )
            admin.set_password(os.getenv('ADMIN_PASSWORD', 'admin8587037321'))
            db.session.add(admin)
            db.session.commit()
            logger.info(f"Admin user created: {admin_email}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize database
    init_database()
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False') == 'True'
    )
