"""
LIFE FRACTAL INTELLIGENCE v8.0 - PRODUCTION WITH SECURE AUTH
=============================================================
Complete neurodivergent-focused life planning with:
- Secure Argon2 authentication
- CAPTCHA fraud prevention
- Email notifications for trials
- Password reset functionality
- Rate limiting
- Session management
- All original features intact
"""

import os
import sys

# Add secure auth module to path
sys.path.insert(0, os.path.dirname(__file__))

from secure_auth_module import SecureAuthManager, CaptchaGenerator, EmailService
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, session as flask_session

# Import from original application
import sys
sys.path.insert(0, '/mnt/project')
from life_planner_unified_master import (
    User, PetState, DataStore, LifePlanningSystem,
    PHI, FIBONACCI, GOLDEN_ANGLE, logger,
    DailyEntry, Habit, Goal
)

# ===========================================================
# FLASK APPLICATION WITH SECURE AUTH
# ===========================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'life-fractal-secret-key-2025-secure')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

from flask_cors import CORS
CORS(app, supports_credentials=True)

# Initialize systems
store = DataStore()
auth_manager = SecureAuthManager()
email_service = EmailService()
captcha_gen = CaptchaGenerator()

# Configuration
SUBSCRIPTION_PRICE = 20.00
TRIAL_DAYS = 7
GOFUNDME_URL = 'https://gofund.me/8d9303d27'

# Store CAPTCHA challenges in memory (production: use Redis)
captcha_challenges = {}


# ===========================================================
# AUTHENTICATION ROUTES
# ===========================================================

@app.route('/api/auth/captcha', methods=['GET'])
def get_captcha():
    """Generate CAPTCHA challenge."""
    try:
        question, answer = captcha_gen.generate()
        challenge_id = os.urandom(16).hex()
        captcha_challenges[challenge_id] = {
            'answer': answer,
            'expires': datetime.now(timezone.utc) + timedelta(minutes=5)
        }
        
        return jsonify({
            'challenge_id': challenge_id,
            'question': question
        }), 200
        
    except Exception as e:
        logger.error(f"CAPTCHA generation error: {e}")
        return jsonify({'error': 'Failed to generate CAPTCHA'}), 500


@app.route('/api/auth/check-email', methods=['POST'])
def check_email():
    """Check if email is already registered (returning user check)."""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        
        is_returning = auth_manager.check_returning_user(email)
        
        return jsonify({
            'is_returning_user': is_returning,
            'message': 'Welcome back!' if is_returning else 'New user'
        }), 200
        
    except Exception as e:
        logger.error(f"Email check error: {e}")
        return jsonify({'error': 'Email check failed'}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user with CAPTCHA and email notification."""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        challenge_id = data.get('challenge_id', '')
        captcha_answer = data.get('captcha_answer', '')
        
        # Validate CAPTCHA
        if challenge_id not in captcha_challenges:
            return jsonify({'error': 'Invalid CAPTCHA challenge'}), 400
        
        challenge = captcha_challenges[challenge_id]
        if datetime.now(timezone.utc) > challenge['expires']:
            del captcha_challenges[challenge_id]
            return jsonify({'error': 'CAPTCHA expired'}), 400
        
        if str(captcha_answer).strip() != str(challenge['answer']):
            return jsonify({'error': 'CAPTCHA verification failed'}), 400
        
        # Clean up used CAPTCHA
        del captcha_challenges[challenge_id]
        
        # Get IP address for rate limiting
        ip_address = request.remote_addr or 'unknown'
        
        # Register in auth system
        result = auth_manager.register_user(
            email, password, first_name, last_name, ip_address
        )
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Registration failed')}), 400
        
        # Create user in main system
        user = store.create_user(email, password, first_name, last_name)
        if not user:
            return jsonify({'error': 'Failed to create user account'}), 500
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'Registration successful! Check your email for trial information.',
            'user': user.to_dict(),
            'access_token': user.id,
            'trial_days_remaining': TRIAL_DAYS,
            'show_gofundme': True,
            'gofundme_url': GOFUNDME_URL
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login with CAPTCHA and trial status check."""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        challenge_id = data.get('challenge_id', '')
        captcha_answer = data.get('captcha_answer', '')
        
        # Validate CAPTCHA
        if challenge_id not in captcha_challenges:
            return jsonify({'error': 'Invalid CAPTCHA challenge'}), 400
        
        challenge = captcha_challenges[challenge_id]
        if datetime.now(timezone.utc) > challenge['expires']:
            del captcha_challenges[challenge_id]
            return jsonify({'error': 'CAPTCHA expired'}), 400
        
        expected_answer = challenge['answer']
        del captcha_challenges[challenge_id]
        
        # Get IP address
        ip_address = request.remote_addr or 'unknown'
        
        # Login via auth system
        result = auth_manager.login_user(
            email, password, captcha_answer, expected_answer, ip_address
        )
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Login failed')}), 401
        
        # Get user from main system
        user = store.get_user(email)
        if not user:
            return jsonify({'error': 'User account not found'}), 404
        
        # Update last login
        user.last_login = datetime.now(timezone.utc).isoformat()
        
        # Check trial status
        has_access = user.has_active_subscription()
        trial_active = user.is_trial_active()
        days_remaining = user.days_remaining_trial()
        
        # Send warning emails if trial ending soon
        if trial_active and days_remaining <= 2 and days_remaining > 0:
            email_service.send_trial_ending_soon(email, user.first_name or "Friend", days_remaining)
        elif not has_access and user.subscription_status == 'trial':
            email_service.send_trial_expired(email, user.first_name or "Friend")
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': user.id,
            'session_token': result['session_token'],
            'has_access': has_access,
            'trial_active': trial_active,
            'days_remaining': days_remaining,
            'needs_subscription': not has_access,
            'gofundme_url': GOFUNDME_URL if not has_access else None
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset."""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        
        result = auth_manager.request_password_reset(email)
        
        # Always return success to prevent email enumeration
        return jsonify({
            'message': 'If an account exists with this email, you will receive reset instructions.'
        }), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Request failed'}), 500


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token."""
    try:
        data = request.get_json()
        token = data.get('token', '')
        new_password = data.get('new_password', '')
        
        result = auth_manager.reset_password(token, new_password)
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Reset failed')}), 400
        
        return jsonify({
            'message': 'Password reset successful. You can now login.'
        }), 200
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return jsonify({'error': 'Password reset failed'}), 500


@app.route('/api/auth/verify-session', methods=['POST'])
def verify_session():
    """Verify session token."""
    try:
        data = request.get_json()
        session_token = data.get('session_token', '')
        
        user_id = auth_manager.verify_session(session_token)
        
        if not user_id:
            return jsonify({'error': 'Invalid or expired session'}), 401
        
        user = store.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'valid': True,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Session verification error: {e}")
        return jsonify({'error': 'Verification failed'}), 500


# ===========================================================
# ACCESS CONTROL MIDDLEWARE
# ===========================================================

def require_active_subscription(user_id: str) -> tuple:
    """Check if user has active subscription or trial."""
    user = store.get_user(user_id)
    if not user:
        return None, 'User not found', 404
    
    if not user.has_active_subscription():
        days_since_trial = 0
        if user.trial_end_date:
            try:
                end = datetime.fromisoformat(user.trial_end_date.replace('Z', '+00:00'))
                delta = datetime.now(timezone.utc) - end
                days_since_trial = delta.days
            except:
                pass
        
        return None, {
            'error': 'Subscription required',
            'message': 'Your trial has ended. Please subscribe to continue.',
            'subscription_price': SUBSCRIPTION_PRICE,
            'gofundme_url': GOFUNDME_URL,
            'trial_expired': True,
            'days_since_trial_end': days_since_trial
        }, 403
    
    return user, None, 200


# ===========================================================
# USER ROUTES (WITH ACCESS CONTROL)
# ===========================================================

@app.route('/api/user/<user_id>')
def get_user(user_id):
    """Get user profile."""
    user, error, status = require_active_subscription(user_id)
    if error:
        return jsonify(error), status
    return jsonify(user.to_dict(include_sensitive=True))


@app.route('/api/user/<user_id>/dashboard')
def get_dashboard(user_id):
    """Get comprehensive dashboard data."""
    user, error, status = require_active_subscription(user_id)
    if error:
        return jsonify(error), status
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    today_entry = user.daily_entries.get(today, DailyEntry(date=today))
    
    # Calculate stats
    entries = list(user.daily_entries.values())
    avg_wellness = sum(e.wellness_index for e in entries) / max(1, len(entries))
    
    return jsonify({
        'user': user.to_dict(),
        'today': today_entry.to_dict(),
        'pet': user.pet.to_dict() if user.pet else None,
        'habits': [h.to_dict() for h in user.habits.values()],
        'goals': [g.to_dict() for g in user.goals.values()],
        'stats': {
            'wellness_index': round(today_entry.wellness_index, 1),
            'average_wellness': round(avg_wellness, 1),
            'current_streak': user.current_streak,
            'total_entries': len(entries),
            'habits_completed_today': sum(1 for c in today_entry.habits_completed.values() if c),
            'active_goals': sum(1 for g in user.goals.values() if not g.is_completed),
            'goals_progress': round(sum(g.progress for g in user.goals.values()) / max(1, len(user.goals)), 1)
        },
        'sacred_math': {
            'phi': PHI,
            'golden_angle': GOLDEN_ANGLE,
            'fibonacci': FIBONACCI[:13]
        },
        'trial_info': {
            'days_remaining': user.days_remaining_trial(),
            'trial_active': user.is_trial_active(),
            'subscription_status': user.subscription_status
        }
    })


@app.route('/api/user/<user_id>/today', methods=['GET', 'POST'])
def handle_today(user_id):
    """Get or update today's entry."""
    user, error, status = require_active_subscription(user_id)
    if error:
        return jsonify(error), status
    
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    if request.method == 'GET':
        entry = user.daily_entries.get(today, DailyEntry(date=today))
        return jsonify(entry.to_dict())
    
    # POST - update
    data = request.get_json()
    
    if today not in user.daily_entries:
        user.daily_entries[today] = DailyEntry(date=today)
    
    entry = user.daily_entries[today]
    
    # Update fields
    for field in ['mood_level', 'mood_score', 'energy_level', 'focus_clarity',
                  'anxiety_level', 'stress_level', 'mindfulness_score',
                  'gratitude_level', 'sleep_quality', 'sleep_hours',
                  'nutrition_score', 'social_connection', 'emotional_stability',
                  'self_compassion', 'journal_entry', 'goals_completed_count']:
        if field in data:
            setattr(entry, field, data[field])
    
    if 'habits_completed' in data:
        entry.habits_completed.update(data['habits_completed'])
    
    entry.calculate_wellness()
    
    # Update history
    user.history.append(entry.to_dict())
    
    # Update life planning system
    system = store.get_system(user_id)
    system.update(entry.to_dict())
    
    return jsonify(entry.to_dict())


# ===========================================================
# HEALTH CHECK
# ===========================================================

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'version': '8.0',
        'features': [
            'secure_authentication',
            'captcha_protection',
            'email_notifications',
            'password_reset',
            'rate_limiting',
            'trial_management'
        ]
    }), 200


@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        'service': 'Life Fractal Intelligence',
        'version': '8.0',
        'status': 'running',
        'auth': 'secure',
        'gofundme': GOFUNDME_URL
    }), 200


# ===========================================================
# ERROR HANDLERS
# ===========================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ===========================================================
# RUN APPLICATION
# ===========================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Life Fractal Intelligence v8.0 on port {port}")
    logger.info(f"Secure authentication: ENABLED")
    logger.info(f"CAPTCHA protection: ENABLED")
    logger.info(f"Email notifications: ENABLED")
    logger.info(f"GoFundMe: {GOFUNDME_URL}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
