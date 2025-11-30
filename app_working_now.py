import os
code = '''import os
import sys
import logging
from datetime import datetime, timedelta

os.makedirs(\"logs\", exist_ok=True)
logging.basicConfig(level=logging.INFO, format=\"%(asctime)s [%(levelname)s] %(message)s\", handlers=[logging.FileHandler(\"logs/life_planner.log\", mode=\"a\"), logging.StreamHandler()])
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
app.config[\"SECRET_KEY\"] = os.getenv(\"SECRET_KEY\", \"dev-key\")
app.config[\"SQLALCHEMY_DATABASE_URI\"] = os.getenv(\"DATABASE_URL\", \"sqlite:///life_planner.db\")
app.config[\"SQLALCHEMY_TRACK_MODIFICATIONS\"] = False

db = SQLAlchemy(app)
CORS(app)

try:
    from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
    app.config[\"JWT_SECRET_KEY\"] = os.getenv(\"JWT_SECRET_KEY\", \"jwt-key\")
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
    __tablename__ = \"users\"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    subscription_status = db.Column(db.String(20), default=\"trial\")
    trial_end_date = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=7))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    def to_dict(self):
        return {\"id\": self.id, \"email\": self.email, \"subscription_status\": self.subscription_status}

class Pet(db.Model):
    __tablename__ = \"pets\"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(\"users.id\"))
    name = db.Column(db.String(50), default=\"Buddy\")
    species = db.Column(db.String(20), default=\"cat\")
    level = db.Column(db.Integer, default=0)
    hunger = db.Column(db.Float, default=50.0)
    energy = db.Column(db.Float, default=50.0)
    mood = db.Column(db.Float, default=50.0)
    def to_dict(self):
        return {\"name\": self.name, \"species\": self.species, \"level\": self.level, \"hunger\": self.hunger, \"energy\": self.energy, \"mood\": self.mood}

@app.route(\"/\")
def index():
    try:
        return render_template(\"index.html\")
    except:
        return \"<html><body style=\\\"font-family:Arial;padding:40px;text-align:center;\\\"><h1>Life Planner</h1><p>API Status: <span style=\\\"color:green;\\\">Online</span></p></body></html>\"

@app.route(\"/api/health\")
def health():
    return jsonify({\"status\": \"healthy\"})

@app.route(\"/api/auth/register\", methods=[\"POST\"])
def register():
    try:
        data = request.get_json() or {}
        email = data.get(\"email\", \"\").lower().strip()
        password = data.get(\"password\", \"\")
        if not email or not password:
            return jsonify({\"error\": \"Email and password required\"}), 400
        if len(password) < 8:
            return jsonify({\"error\": \"Password must be at least 8 characters\"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({\"error\": \"Email already registered\"}), 400
        user = User()
        user.email = email
        user.password_hash = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()
        pet = Pet(user_id=user.id)
        db.session.add(pet)
        db.session.commit()
        access_token = create_access_token(identity=user.id) if JWT_AVAILABLE else \"demo-token\"
        refresh_token = create_refresh_token(identity=user.id) if JWT_AVAILABLE else \"demo-refresh\"
        logger.info(f\"User registered: {email}\")
        return jsonify({\"message\": \"Registration successful!\", \"user\": user.to_dict(), \"access_token\": access_token, \"refresh_token\": refresh_token, \"trial_days_remaining\": 7, \"show_gofundme\": True, \"gofundme_url\": \"https://gofund.me/8d9303d27\"}), 201
    except Exception as e:
        logger.error(f\"Registration error: {e}\")
        db.session.rollback()
        return jsonify({\"error\": str(e)}), 500

@app.route(\"/api/auth/login\", methods=[\"POST\"])
def login():
    try:
        data = request.get_json() or {}
        email = data.get(\"email\", \"\").lower().strip()
        password = data.get(\"password\", \"\")
        if not email or not password:
            return jsonify({\"error\": \"Email and password required\"}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({\"error\": \"Invalid credentials\"}), 401
        user.last_login = datetime.utcnow()
        db.session.commit()
        access_token = create_access_token(identity=user.id) if JWT_AVAILABLE else \"demo-token\"
        refresh_token = create_refresh_token(identity=user.id) if JWT_AVAILABLE else \"demo-refresh\"
        logger.info(f\"User logged in: {email}\")
        return jsonify({\"message\": \"Login successful!\", \"user\": user.to_dict(), \"access_token\": access_token, \"refresh_token\": refresh_token, \"trial_active\": True, \"trial_days_remaining\": 7, \"show_gofundme\": True, \"gofundme_url\": \"https://gofund.me/8d9303d27\"}), 200
    except Exception as e:
        logger.error(f\"Login error: {e}\")
        return jsonify({\"error\": str(e)}), 500

@app.route(\"/api/pet\", methods=[\"GET\"])
@jwt_required()
def get_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if not pet:
            pet = Pet(user_id=user_id)
            db.session.add(pet)
            db.session.commit()
        return jsonify({\"pet\": pet.to_dict()}), 200
    except Exception as e:
        return jsonify({\"error\": str(e)}), 500

@app.route(\"/api/pet/feed\", methods=[\"POST\"])
@jwt_required()
def feed_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if pet:
            pet.hunger = max(0, pet.hunger - 20)
            pet.mood = min(100, pet.mood + 5)
            db.session.commit()
            return jsonify({\"message\": \"Pet fed!\", \"pet\": pet.to_dict()}), 200
        return jsonify({\"error\": \"Pet not found\"}), 404
    except Exception as e:
        return jsonify({\"error\": str(e)}), 500

@app.route(\"/api/pet/play\", methods=[\"POST\"])
@jwt_required()
def play_pet():
    try:
        user_id = get_jwt_identity()
        pet = Pet.query.filter_by(user_id=user_id).first()
        if pet and pet.energy >= 20:
            pet.energy = max(0, pet.energy - 15)
            pet.mood = min(100, pet.mood + 10)
            db.session.commit()
            return jsonify({\"message\": \"Had fun!\", \"pet\": pet.to_dict()}), 200
        return jsonify({\"error\": \"Pet too tired\" if pet else \"Pet not found\"}), 400 if pet else 404
    except Exception as e:
        return jsonify({\"error\": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({\"error\": \"Not found\"}), 404

@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return jsonify({\"error\": \"Server error\"}), 500

if __name__ == \"__main__\":
    print(\"\\n\" + \"=\"*70)
    print(\"LIFE PLANNER - WORKING VERSION\")
    print(\"=\"*70 + \"\\n\")
    with app.app_context():
        db.create_all()
        admin_email = os.getenv(\"ADMIN_EMAIL\", \"onlinediscountsllc@gmail.com\")
        admin = User.query.filter_by(email=admin_email).first()
        if not admin:
            admin = User()
            admin.email = admin_email
            admin.password_hash = generate_password_hash(os.getenv(\"ADMIN_PASSWORD\", \"admin8587037321\"))
            admin.is_admin = True
            admin.subscription_status = \"active\"
            db.session.add(admin)
            db.session.commit()
            admin_pet = Pet(user_id=admin.id, name=\"Admin Pet\", species=\"dragon\", level=10)
            db.session.add(admin_pet)
            db.session.commit()
            print(f\"Admin created: {admin_email}\")
        else:
            print(f\"Admin exists: {admin_email}\")
    print(f\"\\nhttp://localhost:5000\")
    print(f\"Admin: {os.getenv(\\\"ADMIN_EMAIL\\\", \\\"onlinediscountsllc@gmail.com\\\")}\")
    print(f\"Password: {os.getenv(\\\"ADMIN_PASSWORD\\\", \\\"admin8587037321\\\")}\")
    print(\"\\n\" + \"=\"*70 + \"\\n\")
    app.run(host=\"0.0.0.0\", port=5000, debug=True)
'''
with open('app_clean.py', 'w') as f:
    f.write(code)
print('Created app_clean.py')
"