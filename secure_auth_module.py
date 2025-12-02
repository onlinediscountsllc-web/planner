"""
SECURE AUTHENTICATION MODULE FOR LIFE FRACTAL INTELLIGENCE
===========================================================
Features:
- Argon2 password hashing
- CAPTCHA verification
- Email notifications for trials
- Rate limiting
- Password reset
- Login attempt tracking
- Session management
"""

import os
import sqlite3
import secrets
import time
import random
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from argon2 import PasswordHasher
from typing import Optional, Dict, Tuple

# Configure logging
logging.basicConfig(
    filename='auth_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Password hasher (Argon2id - best practice)
ph = PasswordHasher()


# ===========================================================
# CAPTCHA SYSTEM
# ===========================================================

class CaptchaGenerator:
    """Simple math CAPTCHA for fraud prevention."""
    
    def generate(self) -> Tuple[str, int]:
        """Generate CAPTCHA question and answer."""
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        answer = a + b
        question = f"Security Check: What is {a} + {b}?"
        return question, answer


# ===========================================================
# EMAIL SERVICE
# ===========================================================

class EmailService:
    """Email notifications for authentication and trials."""
    
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', 'onlinediscountsllc@gmail.com')
        self.smtp_pass = os.getenv('SMTP_PASSWORD', '')
        self.from_email = 'onlinediscountsllc@gmail.com'
        self.gofundme_url = 'https://gofund.me/8d9303d27'
    
    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email via SMTP."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_pass:
                    server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Email send failed to {to_email}: {e}")
            return False
    
    def send_welcome_trial(self, to_email: str, first_name: str) -> bool:
        """Send welcome email with 7-day trial information."""
        subject = "Welcome to Life Fractal Intelligence - Your 7-Day Trial Starts Now!"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; color: white;">
                <h1 style="margin: 0;">Welcome to Life Fractal Intelligence!</h1>
            </div>
            
            <div style="padding: 30px; background: #f9fafb; border-radius: 10px; margin-top: 20px;">
                <h2 style="color: #667eea;">Hi {first_name},</h2>
                
                <p style="font-size: 16px; line-height: 1.6; color: #374151;">
                    Thank you for joining Life Fractal Intelligence - the neurodivergent-focused life planning system 
                    designed for brains like yours!
                </p>
                
                <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #667eea;">Your 7-Day Free Trial</h3>
                    <p style="margin-bottom: 0;">
                        You have <strong>7 days</strong> of full access to explore all features:
                    </p>
                    <ul style="color: #374151;">
                        <li>Virtual pet companions with emotional AI</li>
                        <li>Sacred geometry fractal visualizations</li>
                        <li>Spoon theory energy management</li>
                        <li>Fibonacci task scheduling</li>
                        <li>Shame-free progress tracking</li>
                        <li>GPU-accelerated 3D fractals</li>
                        <li>Therapeutic soundscapes</li>
                    </ul>
                </div>
                
                <div style="background: #fef3c7; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #92400e;">After Your Trial</h3>
                    <p style="color: #78350f;">
                        After 7 days, a subscription is required to continue using Life Fractal Intelligence.
                        <br><br>
                        <strong>Subscription: $20/month</strong>
                        <br><br>
                        We'll send you reminders before your trial ends!
                    </p>
                </div>
                
                <div style="background: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #1e40af;">Support Our Mission</h3>
                    <p style="color: #1e3a8a;">
                        Life Fractal Intelligence is built by someone with autism, ADHD, and aphantasia - 
                        for others like us. If you'd like to support development:
                    </p>
                    <p style="text-align: center;">
                        <a href="{self.gofundme_url}" 
                           style="display: inline-block; background: #667eea; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; font-weight: bold;">
                            Visit Our GoFundMe
                        </a>
                    </p>
                </div>
                
                <p style="font-size: 14px; color: #6b7280; margin-top: 30px;">
                    Questions? Reply to this email or contact us at {self.from_email}
                </p>
            </div>
            
            <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
                <p>Life Fractal Intelligence - Sacred Mathematics for Neurodivergent Minds</p>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_email, subject, body)
    
    def send_trial_ending_soon(self, to_email: str, first_name: str, days_left: int) -> bool:
        """Send trial ending warning."""
        subject = f"Your Life Fractal Intelligence Trial Ends in {days_left} Days"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #dc2626 100%); padding: 30px; border-radius: 10px; color: white;">
                <h1 style="margin: 0;">Trial Ending Soon!</h1>
            </div>
            
            <div style="padding: 30px; background: #f9fafb; border-radius: 10px; margin-top: 20px;">
                <h2 style="color: #dc2626;">Hi {first_name},</h2>
                
                <p style="font-size: 16px; line-height: 1.6; color: #374151;">
                    Your 7-day trial of Life Fractal Intelligence ends in <strong>{days_left} days</strong>.
                </p>
                
                <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #dc2626; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #dc2626;">Continue Your Journey</h3>
                    <p>
                        To keep accessing your virtual pet, fractals, and all the neurodivergent-friendly features, 
                        subscribe for just $20/month.
                    </p>
                    <p style="text-align: center; margin-top: 20px;">
                        <a href="https://planner-1-pyd9.onrender.com" 
                           style="display: inline-block; background: #667eea; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; font-weight: bold;">
                            Subscribe Now
                        </a>
                    </p>
                </div>
                
                <div style="background: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="color: #1e3a8a; margin: 0;">
                        Can't afford a subscription right now? Consider supporting our GoFundMe:
                        <br><br>
                        <a href="{self.gofundme_url}" style="color: #667eea; font-weight: bold;">
                            {self.gofundme_url}
                        </a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_email, subject, body)
    
    def send_trial_expired(self, to_email: str, first_name: str) -> bool:
        """Send trial expired notification."""
        subject = "Your Life Fractal Intelligence Trial Has Ended"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #374151; padding: 30px; border-radius: 10px; color: white;">
                <h1 style="margin: 0;">Trial Ended</h1>
            </div>
            
            <div style="padding: 30px; background: #f9fafb; border-radius: 10px; margin-top: 20px;">
                <h2>Hi {first_name},</h2>
                
                <p style="font-size: 16px; line-height: 1.6; color: #374151;">
                    Your 7-day trial of Life Fractal Intelligence has ended.
                </p>
                
                <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #667eea;">Subscribe to Continue</h3>
                    <p>
                        To regain access to your account and all features, subscribe for $20/month.
                    </p>
                    <p style="text-align: center; margin-top: 20px;">
                        <a href="https://planner-1-pyd9.onrender.com" 
                           style="display: inline-block; background: #667eea; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; font-weight: bold;">
                            Subscribe Now
                        </a>
                    </p>
                </div>
                
                <p style="color: #6b7280;">
                    Your data is safely stored and will be available once you subscribe!
                </p>
                
                <p style="color: #374151;">
                    Support our development: <a href="{self.gofundme_url}" style="color: #667eea;">{self.gofundme_url}</a>
                </p>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_email, subject, body)
    
    def send_password_reset(self, to_email: str, token: str) -> bool:
        """Send password reset email."""
        reset_link = f"https://planner-1-pyd9.onrender.com/reset-password?token={token}"
        subject = "Reset Your Life Fractal Intelligence Password"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #667eea; padding: 30px; border-radius: 10px; color: white;">
                <h1 style="margin: 0;">Password Reset Request</h1>
            </div>
            
            <div style="padding: 30px; background: #f9fafb; border-radius: 10px; margin-top: 20px;">
                <p style="font-size: 16px; line-height: 1.6; color: #374151;">
                    You requested to reset your password. Click the button below to set a new password:
                </p>
                
                <p style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}" 
                       style="display: inline-block; background: #667eea; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 5px; font-weight: bold;">
                        Reset Password
                    </a>
                </p>
                
                <p style="font-size: 14px; color: #6b7280;">
                    This link expires in 30 minutes. If you didn't request this, please ignore this email.
                </p>
                
                <p style="font-size: 12px; color: #9ca3af; margin-top: 20px;">
                    Or copy this link: {reset_link}
                </p>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_email, subject, body)


# ===========================================================
# DATABASE MANAGER
# ===========================================================

class AuthDatabase:
    """Secure database for authentication with rate limiting."""
    
    def __init__(self, db_path: str = "auth_secure.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    created_at REAL,
                    last_login REAL,
                    failed_attempts INTEGER DEFAULT 0,
                    is_locked INTEGER DEFAULT 0,
                    is_verified INTEGER DEFAULT 0
                )
            """)
            
            # Password reset tokens
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reset_tokens (
                    token TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    expires REAL NOT NULL,
                    used INTEGER DEFAULT 0
                )
            """)
            
            # Login attempts (rate limiting)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    ip_address TEXT,
                    email TEXT,
                    attempt_time REAL,
                    success INTEGER
                )
            """)
            
            # Sessions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL,
                    expires_at REAL,
                    ip_address TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def execute(self, query: str, params: tuple = (), fetchone: bool = False, fetchall: bool = False):
        """Execute database query with error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(query, params)
            
            if fetchone:
                result = cur.fetchone()
            elif fetchall:
                result = cur.fetchall()
            else:
                result = None
            
            conn.commit()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return None
    
    def check_rate_limit(self, ip_address: str, max_attempts: int = 5, window_minutes: int = 15) -> bool:
        """Check if IP is rate limited."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        attempts = self.execute(
            "SELECT COUNT(*) FROM login_attempts WHERE ip_address = ? AND attempt_time > ? AND success = 0",
            (ip_address, cutoff_time),
            fetchone=True
        )
        
        if attempts and attempts[0] >= max_attempts:
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return False
        return True
    
    def log_login_attempt(self, ip_address: str, email: str, success: bool):
        """Log login attempt for rate limiting."""
        self.execute(
            "INSERT INTO login_attempts (ip_address, email, attempt_time, success) VALUES (?,?,?,?)",
            (ip_address, email, time.time(), 1 if success else 0)
        )


# ===========================================================
# SECURE AUTHENTICATION MANAGER
# ===========================================================

class SecureAuthManager:
    """Complete authentication system with all security features."""
    
    def __init__(self):
        self.db = AuthDatabase()
        self.email_service = EmailService()
        self.captcha = CaptchaGenerator()
    
    def register_user(self, email: str, password: str, first_name: str, 
                     last_name: str, ip_address: str = "unknown") -> Dict:
        """Register new user with security checks."""
        try:
            # Input validation
            email = email.lower().strip()
            if not email or '@' not in email:
                return {'success': False, 'error': 'Invalid email format'}
            
            if len(password) < 8:
                return {'success': False, 'error': 'Password must be at least 8 characters'}
            
            # Check if email exists
            existing = self.db.execute(
                "SELECT user_id FROM users WHERE email = ?",
                (email,), fetchone=True
            )
            if existing:
                return {'success': False, 'error': 'Email already registered'}
            
            # Hash password with Argon2
            password_hash = ph.hash(password)
            user_id = f"user_{secrets.token_hex(16)}"
            created_at = time.time()
            
            # Insert user
            self.db.execute(
                """INSERT INTO users 
                   (user_id, email, password_hash, first_name, last_name, created_at, is_verified) 
                   VALUES (?,?,?,?,?,?,1)""",
                (user_id, email, password_hash, first_name, last_name, created_at)
            )
            
            # Send welcome email
            self.email_service.send_welcome_trial(email, first_name or "Friend")
            
            logger.info(f"User registered: {email}")
            
            return {
                'success': True,
                'user_id': user_id,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'message': 'Registration successful! Check your email for trial information.'
            }
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {'success': False, 'error': 'Registration failed'}
    
    def login_user(self, email: str, password: str, captcha_answer: str, 
                   captcha_expected: int, ip_address: str = "unknown") -> Dict:
        """Login with CAPTCHA and rate limiting."""
        try:
            # Rate limiting
            if not self.db.check_rate_limit(ip_address):
                return {'success': False, 'error': 'Too many login attempts. Please wait 15 minutes.'}
            
            # CAPTCHA verification
            if str(captcha_answer).strip() != str(captcha_expected):
                self.db.log_login_attempt(ip_address, email, False)
                return {'success': False, 'error': 'CAPTCHA verification failed'}
            
            # Get user
            user = self.db.execute(
                """SELECT user_id, password_hash, failed_attempts, is_locked, first_name, last_name 
                   FROM users WHERE email = ?""",
                (email.lower().strip(),), fetchone=True
            )
            
            if not user:
                self.db.log_login_attempt(ip_address, email, False)
                return {'success': False, 'error': 'Invalid credentials'}
            
            user_id, password_hash, failed_attempts, is_locked, first_name, last_name = user
            
            # Check if locked
            if is_locked:
                return {'success': False, 'error': 'Account locked. Contact support.'}
            
            # Verify password
            try:
                ph.verify(password_hash, password)
                
                # Success - reset failed attempts
                self.db.execute(
                    "UPDATE users SET failed_attempts = 0, last_login = ? WHERE user_id = ?",
                    (time.time(), user_id)
                )
                
                # Create session
                session_token = secrets.token_hex(32)
                expires_at = time.time() + (24 * 60 * 60)  # 24 hours
                
                self.db.execute(
                    "INSERT INTO sessions (session_token, user_id, created_at, expires_at, ip_address) VALUES (?,?,?,?,?)",
                    (session_token, user_id, time.time(), expires_at, ip_address)
                )
                
                self.db.log_login_attempt(ip_address, email, True)
                logger.info(f"User logged in: {email}")
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'email': email,
                    'first_name': first_name,
                    'last_name': last_name,
                    'session_token': session_token,
                    'message': 'Login successful'
                }
                
            except Exception:
                # Password incorrect
                new_failed = failed_attempts + 1
                if new_failed >= 5:
                    self.db.execute(
                        "UPDATE users SET is_locked = 1, failed_attempts = ? WHERE user_id = ?",
                        (new_failed, user_id)
                    )
                    logger.warning(f"Account locked: {email}")
                    return {'success': False, 'error': 'Account locked due to too many failed attempts'}
                else:
                    self.db.execute(
                        "UPDATE users SET failed_attempts = ? WHERE user_id = ?",
                        (new_failed, user_id)
                    )
                
                self.db.log_login_attempt(ip_address, email, False)
                return {'success': False, 'error': 'Invalid credentials'}
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {'success': False, 'error': 'Login failed'}
    
    def request_password_reset(self, email: str) -> Dict:
        """Send password reset email."""
        try:
            user = self.db.execute(
                "SELECT user_id, first_name FROM users WHERE email = ?",
                (email.lower().strip(),), fetchone=True
            )
            
            if not user:
                # Don't reveal if email exists
                return {'success': True, 'message': 'If account exists, reset email sent'}
            
            # Generate reset token
            token = secrets.token_hex(32)
            expires = time.time() + (30 * 60)  # 30 minutes
            
            self.db.execute(
                "INSERT INTO reset_tokens (token, email, expires) VALUES (?,?,?)",
                (token, email.lower().strip(), expires)
            )
            
            # Send email
            self.email_service.send_password_reset(email, token)
            
            logger.info(f"Password reset requested: {email}")
            return {'success': True, 'message': 'If account exists, reset email sent'}
            
        except Exception as e:
            logger.error(f"Password reset request error: {e}")
            return {'success': False, 'error': 'Reset request failed'}
    
    def reset_password(self, token: str, new_password: str) -> Dict:
        """Reset password using token."""
        try:
            # Verify token
            record = self.db.execute(
                "SELECT email, expires, used FROM reset_tokens WHERE token = ?",
                (token,), fetchone=True
            )
            
            if not record:
                return {'success': False, 'error': 'Invalid reset token'}
            
            email, expires, used = record
            
            if used:
                return {'success': False, 'error': 'Reset token already used'}
            
            if time.time() > expires:
                return {'success': False, 'error': 'Reset token expired'}
            
            # Validate new password
            if len(new_password) < 8:
                return {'success': False, 'error': 'Password must be at least 8 characters'}
            
            # Hash new password
            new_hash = ph.hash(new_password)
            
            # Update password
            self.db.execute(
                "UPDATE users SET password_hash = ?, failed_attempts = 0, is_locked = 0 WHERE email = ?",
                (new_hash, email)
            )
            
            # Mark token as used
            self.db.execute(
                "UPDATE reset_tokens SET used = 1 WHERE token = ?",
                (token,)
            )
            
            logger.info(f"Password reset completed: {email}")
            return {'success': True, 'message': 'Password reset successful'}
            
        except Exception as e:
            logger.error(f"Password reset error: {e}")
            return {'success': False, 'error': 'Password reset failed'}
    
    def verify_session(self, session_token: str) -> Optional[str]:
        """Verify session token and return user_id."""
        try:
            session = self.db.execute(
                "SELECT user_id, expires_at FROM sessions WHERE session_token = ?",
                (session_token,), fetchone=True
            )
            
            if not session:
                return None
            
            user_id, expires_at = session
            
            if time.time() > expires_at:
                # Session expired
                self.db.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
                return None
            
            return user_id
            
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return None
    
    def check_returning_user(self, email: str) -> bool:
        """Check if email is already registered."""
        user = self.db.execute(
            "SELECT user_id FROM users WHERE email = ?",
            (email.lower().strip(),), fetchone=True
        )
        return user is not None
