"""
Database Models for Life Planner Application
Includes User management, Subscriptions, Pet data, and Activity tracking
"""

from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

db = SQLAlchemy()


class User(db.Model):
    """User account model with subscription and authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Profile
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Subscription & Trial
    trial_start_date = db.Column(db.DateTime, default=datetime.utcnow)
    trial_end_date = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=7))
    subscription_status = db.Column(db.String(20), default='trial')  # trial, active, cancelled, expired
    subscription_start_date = db.Column(db.DateTime, nullable=True)
    subscription_end_date = db.Column(db.DateTime, nullable=True)
    stripe_customer_id = db.Column(db.String(100), unique=True, nullable=True)
    stripe_subscription_id = db.Column(db.String(100), unique=True, nullable=True)
    
    # Security
    email_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    pet = db.relationship('Pet', backref='owner', uselist=False, cascade='all, delete-orphan')
    activities = db.relationship('UserActivity', backref='user', cascade='all, delete-orphan')
    ml_data = db.relationship('MLData', backref='user', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def generate_verification_token(self):
        """Generate email verification token"""
        self.verification_token = secrets.token_urlsafe(32)
        return self.verification_token
    
    def generate_reset_token(self, expires_in=3600):
        """Generate password reset token"""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return self.reset_token
    
    def verify_reset_token(self, token):
        """Verify password reset token"""
        if self.reset_token != token:
            return False
        if self.reset_token_expires < datetime.utcnow():
            return False
        return True
    
    def is_trial_active(self):
        """Check if trial period is still active"""
        return datetime.utcnow() < self.trial_end_date and self.subscription_status == 'trial'
    
    def has_active_subscription(self):
        """Check if user has active subscription or trial"""
        if self.is_trial_active():
            return True
        if self.subscription_status == 'active':
            if self.subscription_end_date and datetime.utcnow() < self.subscription_end_date:
                return True
        return False
    
    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'subscription_status': self.subscription_status,
            'trial_end_date': self.trial_end_date.isoformat() if self.trial_end_date else None,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        if include_sensitive:
            data['email_verified'] = self.email_verified
            data['is_admin'] = self.is_admin
        return data


class Pet(db.Model):
    """Virtual pet data for each user"""
    __tablename__ = 'pets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    
    # Pet Attributes
    name = db.Column(db.String(50), default='Buddy')
    species = db.Column(db.String(20), default='cat')  # cat, dragon, phoenix, etc.
    
    # State (0-100 scale)
    hunger = db.Column(db.Float, default=50.0)
    energy = db.Column(db.Float, default=50.0)
    mood = db.Column(db.Float, default=50.0)
    stress = db.Column(db.Float, default=50.0)
    growth = db.Column(db.Float, default=1.0)
    level = db.Column(db.Integer, default=0)
    experience = db.Column(db.Integer, default=0)
    
    # Evolution & Behavior
    behavior = db.Column(db.String(20), default='idle')  # idle, happy, sad, tired
    evolution_stage = db.Column(db.Integer, default=0)
    
    # Stats
    bond = db.Column(db.Float, default=0.0)
    total_tasks_completed = db.Column(db.Integer, default=0)
    total_goals_achieved = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_fed = db.Column(db.DateTime, nullable=True)
    last_played = db.Column(db.DateTime, nullable=True)
    
    def to_dict(self):
        """Convert pet to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'species': self.species,
            'hunger': self.hunger,
            'energy': self.energy,
            'mood': self.mood,
            'stress': self.stress,
            'growth': self.growth,
            'level': self.level,
            'experience': self.experience,
            'behavior': self.behavior,
            'evolution_stage': self.evolution_stage,
            'bond': self.bond,
            'total_tasks_completed': self.total_tasks_completed,
            'total_goals_achieved': self.total_goals_achieved,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class UserActivity(db.Model):
    """Track user activities and behaviors for ML training"""
    __tablename__ = 'user_activities'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Activity Data
    stress = db.Column(db.Float)
    mood = db.Column(db.Float)
    energy = db.Column(db.Float)
    goals_completed = db.Column(db.Integer, default=0)
    sleep_hours = db.Column(db.Float)
    nutrition_score = db.Column(db.Float)
    period = db.Column(db.String(20), default='daily')  # daily, weekly, monthly, yearly
    
    # Predictions
    predicted_mood = db.Column(db.Float, nullable=True)
    actual_next_mood = db.Column(db.Float, nullable=True)
    
    # Metadata
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        """Convert activity to dictionary"""
        return {
            'id': self.id,
            'stress': self.stress,
            'mood': self.mood,
            'energy': self.energy,
            'goals_completed': self.goals_completed,
            'sleep_hours': self.sleep_hours,
            'nutrition_score': self.nutrition_score,
            'period': self.period,
            'predicted_mood': self.predicted_mood,
            'timestamp': self.timestamp.isoformat()
        }


class MLData(db.Model):
    """Privacy-preserving aggregated ML data"""
    __tablename__ = 'ml_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Anonymized features (no PII)
    feature_vector = db.Column(JSON)  # Aggregated non-identifiable patterns
    model_version = db.Column(db.String(20))
    
    # Privacy flags
    consent_given = db.Column(db.Boolean, default=True)
    anonymized = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemSettings(db.Model):
    """Global system settings"""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Text)
    description = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(db.Model):
    """Security audit trail"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    
    # Event details
    action = db.Column(db.String(100), nullable=False)
    resource = db.Column(db.String(100))
    status = db.Column(db.String(20))  # success, failure, warning
    
    # Request metadata
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    
    # Additional data
    details = db.Column(JSON)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
