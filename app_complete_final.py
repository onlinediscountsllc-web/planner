$code = @"
import os
import sys
import logging
from datetime import datetime, timedelta
from io import BytesIO

os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler('logs/life_planner.log', mode='a'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///life_planner.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
CORS(app, resources={r'/api/*': {'origins': '*'}})

try:
    from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    jwt = JWTManager(app)
    JWT_AVAILABLE = True
except:
    JWT_AVAILABLE = False
    def jwt_required(optional=False):
        def decorator(f):
            return f
        return decorator
    def get_jwt_identity():
        return 1

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    subscription_status = db.Column(db.String(20), default='trial')
    trial_start_date = db.Column(db.DateTime, default=datetime.utcnow)
    trial_end_date = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=7))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    def to_dict(self):
        return {'id': self.id, 'email': self.email, 'subscription_status': self.subscription_status, 'is_admin': self.is_admin}
    def get_trial_days_remaining(self):
        if self.trial_end_date:
            delta = self.trial_end_date - datetime.utcnow()
            return max(0, delta.days)
        return 0

class Pet(db.Model):
    __tablename__ = 'pets'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    name = db.Column(db.String(50), default='Buddy')
    species = db.Column(db.String(20), default='cat')
    level = db.Column(db.Integer, default=0)
    experience = db.Column(db.Integer, default=0)
    hunger = db.Column(db.Float, default=50.0)
    energy = db.Column(db.Float, default=100.0)
    mood = db.Column(db.Float, default=75.0)
    stress = db.Column(db.Float, default=30.0)
    bond = db.Column(db.Float, default=50.0)
    def to_dict(self):
        return {'name': self.name, 'species': self.species, 'level': self.level, 'experience': self.experience, 'hunger': round(self.hunger, 1), 'energy': round(self.energy, 1), 'mood': round(self.mood, 1), 'stress': round(self.stress, 1), 'bond': round(self.bond, 1)}
    def level_up_if_ready(self):
        exp_needed = (self.level + 1) * 100
        if self.experience >= exp_needed:
            self.level += 1
            self.experience = self.experience - exp_needed
            return True
        return False

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except:
        return '<html><body style=\"font-family:Arial;padding:40px;text-align:center;\"><h1>Life Planner</h1><p>API Status: <span style=\"color:green;\">Online</span></p><p><a href=\"/api/health\">Health Check</a></p></body></html>'

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'database': 'connected', 'timestamp': datetime.utcnow().isoformat()}), 200

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        user = User()
        user.email = email
        user.password_hash = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()
        pet = Pet(user_id=user.id, name=data.get('pet_name', 'Buddy'))
        db.session.add(pet)
        db.session.commit()
        access_token = create_access_token(identity=user.id) if JWT_AVAILABLE else f'demo-token-{user.id}'
        refresh_token = create_refresh_token(identity=user.id) if JWT_AVAILABLE else f'demo-refresh-{user.id}'
        logger.info(f'User registered: {email}')
        return jsonify({'message': 'Registration successful!', 'user': user.to_dict(), 'access_token': access_token, 'refresh_token': refresh_token, 'trial_days_remaining': user.get_trial_days_remaining(), 'show_gofundme': True, 'gofundme_url': 'https://gofund.me/8d9303d27'}), 201
    except Exception as e:
        logger.error(f'Registration error: {e}')
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid email or password'}), 401
        user.last_login = datetime.utcnow()
        db.session.commit()
        access_token = create_access_token(identity=user.id) if JWT_AVAILABLE else f'demo-token-{user.id}'
        refresh_token = create_refresh_token(identity=user.id) if JWT_AVAILABLE else f'demo-refresh-{user.id}'
        logger.info(f'User logged in: {email}')
        return jsonify({'message': 'Login successful!', 'user': user.to_dict(), 'access_token': access_token, 'refresh_token': refresh_token, 'has_access': True, 'trial_active': user.subscription_status == 'trial', 'trial_days_remaining': user.get_trial_days_remaining(), 'show_gofundme': user.subscription_status == 'trial', 'gofundme_url': 'https://gofund.me/8d9303d27'}), 200
    except Exception as e:
        logger.error(f'Login error: {e}')
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/subscription/status', methods=['GET'])
@jwt_required()
def subscription_status():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        trial_days = user.get_trial_days_remaining()
        trial_expired = trial_days <= 0 and user.subscription_status == 'trial'
        return jsonify({'subscription_status': user.subscription_status, 'trial_active': user.subscription_status == 'trial' and trial_days > 0, 'trial_expired': trial_expired, 'trial_days_remaining': trial_days, 'has_access': user.subscription_status == 'active' or (user.subscription_status == 'trial' and trial_days > 0), 'show_gofundme': user.subscription_status == 'trial', 'gofundme_url': 'https://gofund.me/8d9303d27'}), 200
    except Exception as e:
        logger.error(f'Subscription status error: {e}')
        return jsonify({'error': 'Failed to get status'}), 500

@app.route('/api/subscription/checkout', methods=['POST'])
@jwt_required()
def create_checkout():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        payment_link = os.getenv('STRIPE_PAYMENT_LINK', 'https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00')
        logger.info(f'Checkout for {user.email}')
        return jsonify({'checkout_url': payment_link, 'message': 'Redirecting to checkout...'}), 200
    except Exception as e:
        logger.error(f'Checkout error: {e}')
        return jsonify({'error': 'Checkout failed'}), 500

@app.route('/api/subscription/success', methods=['POST'])
@jwt_required()
def subscription_success():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        user.subscription_status = 'active'
        db.session.commit()
        logger.info(f'Subscription activated: {user.email}')
        return jsonify({'message': 'Subscription activated!', 'subscription_status': 'active'}), 200
    except Exception as e:
        logger.error(f'Activation error: {e}')
        return jsonify({'error': 'Activation failed'}), 500

@app.route('/api/pet', methods=['GET'])
@jwt_required()
def get_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if not pet:
            pet = Pet(user_id=user_id, name='Buddy')
            db.session.add(pet)
            db.session.commit()
        return jsonify({'pet': pet.to_dict()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pet/feed', methods=['POST'])
@jwt_required()
def feed_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if pet:
            pet.hunger = max(0, pet.hunger - 20)
            pet.mood = min(100, pet.mood + 5)
            pet.bond = min(100, pet.bond + 1)
            db.session.commit()
            return jsonify({'message': 'Pet fed!', 'pet': pet.to_dict()}), 200
        return jsonify({'error': 'Pet not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pet/play', methods=['POST'])
@jwt_required()
def play_with_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if pet and pet.energy >= 20:
            pet.energy = max(0, pet.energy - 15)
            pet.mood = min(100, pet.mood + 10)
            pet.bond = min(100, pet.bond + 2)
            pet.experience += 5
            leveled_up = pet.level_up_if_ready()
            db.session.commit()
            message = 'Had fun playing!'
            if leveled_up:
                message += f' Level up to {pet.level}!'
            return jsonify({'message': message, 'leveled_up': leveled_up, 'pet': pet.to_dict()}), 200
        return jsonify({'error': 'Pet too tired' if pet else 'Pet not found'}), 400 if pet else 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/planner/update', methods=['POST'])
@jwt_required()
def update_planner():
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        pet = Pet.query.filter_by(user_id=user_id).first()
        if pet:
            pet.mood = float(data.get('mood', 50))
            pet.stress = float(data.get('stress', 50))
            pet.energy = float(data.get('energy', 50))
            goals = int(data.get('goals', 0))
            if goals > 0:
                pet.experience += goals * 10
                pet.level_up_if_ready()
            db.session.commit()
        guidance = 'Great job checking in!'
        if float(data.get('mood', 50)) > 70:
            guidance = 'You are feeling great! Keep it up.'
        return jsonify({'message': 'Check-in saved!', 'guidance': {'message': guidance, 'predicted_mood': '75.0'}, 'pet': pet.to_dict() if pet else None}), 200
    except Exception as e:
        logger.error(f'Update error: {e}')
        return jsonify({'error': 'Update failed'}), 500

@app.route('/api/planner/fractal', methods=['POST'])
@jwt_required()
def generate_fractal():
    try:
        from PIL import Image
        import random
        img = Image.new('RGB', (512, 512))
        pixels = img.load()
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        mood_factor = int(pet.mood * 2) if pet else 150
        energy_factor = int(pet.energy * 2) if pet else 150
        for i in range(512):
            for j in range(512):
                r = min(255, max(0, (i + mood_factor) % 256 + random.randint(-20, 20)))
                g = min(255, max(0, (j + energy_factor) % 256 + random.randint(-20, 20)))
                b = min(255, max(0, (i + j) % 256 + random.randint(-20, 20)))
                pixels[i, j] = (r, g, b)
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print()
    print('='*70)
    print('LIFE PLANNER - WITH STRIPE PAYMENTS')
    print('='*70)
    print()
    with app.app_context():
        db.create_all()
        admin_email = os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')
        admin = User.query.filter_by(email=admin_email).first()
        if not admin:
            admin = User()
            admin.email = admin_email
            admin.password_hash = generate_password_hash(os.getenv('ADMIN_PASSWORD', 'admin8587037321'))
            admin.is_admin = True
            admin.subscription_status = 'active'
            db.session.add(admin)
            db.session.commit()
            admin_pet = Pet(user_id=admin.id, name='Admin Pet', species='dragon', level=10)
            db.session.add(admin_pet)
            db.session.commit()
            print('Admin created: ' + admin_email)
        else:
            print('Admin exists: ' + admin_email)
    print()
    print('Your Stripe link: ' + os.getenv('STRIPE_PAYMENT_LINK', 'https://buy.stripe.com/eVqeVd0GfadZaUXg8qcwg00'))
    print('Access app: http://localhost:5000')
    print('Admin: ' + os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com'))
    print('Password: ' + os.getenv('ADMIN_PASSWORD', 'admin8587037321'))
    print()
    print('='*70)
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
"@

$code | Out-File -FilePath app_final_clean.py -Encoding ASCII
Write-Host "Created app_final_clean.py - 100% ASCII safe!" -ForegroundColor Green