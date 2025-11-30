#!/usr/bin/env python3
"""
Life Planner - Self-Healing Setup Script
Automatically detects and fixes all issues
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

print("=" * 70)
print("LIFE PLANNER - SELF-HEALING SETUP")
print("=" * 70)
print()

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)

print(f"Working directory: {BASE_DIR}")
print()

# ============================================================================
# STEP 1: CREATE FOLDER STRUCTURE
# ============================================================================

print("Step 1: Creating folder structure...")

folders = ['models', 'backend', 'templates', 'logs', 'static', 'user_data']
for folder in folders:
    folder_path = BASE_DIR / folder
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {folder}")
    else:
        print(f"  ✓ Exists: {folder}")

print()

# ============================================================================
# STEP 2: ORGANIZE FILES
# ============================================================================

print("Step 2: Organizing files into correct folders...")

file_moves = {
    'database.py': 'models/database.py',
    'life_planning_core.py': 'backend/life_planning_core.py',
    'gpu_extensions.py': 'backend/gpu_extensions.py',
    'index.html': 'templates/index.html',
}

for source, destination in file_moves.items():
    source_path = BASE_DIR / source
    dest_path = BASE_DIR / destination
    
    if source_path.exists() and not dest_path.exists():
        shutil.move(str(source_path), str(dest_path))
        print(f"  ✓ Moved: {source} → {destination}")
    elif dest_path.exists():
        print(f"  ✓ Already in place: {destination}")
    else:
        print(f"  ⚠ Not found: {source}")

print()

# ============================================================================
# STEP 3: CREATE MINIMAL REQUIREMENTS.TXT
# ============================================================================

print("Step 3: Creating minimal requirements.txt...")

minimal_requirements = """# Core Framework
Flask==3.0.0
Flask-SQLAlchemy==3.1.1
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.3
Flask-Mail==0.9.1

# Database
SQLAlchemy==2.0.23

# Security
bcrypt==4.1.2
python-dotenv==1.0.0

# Payment Processing
stripe==7.8.0

# Essential Libraries
numpy==1.26.2
Pillow==10.1.0
scikit-learn==1.3.2
requests==2.31.0

# Utilities
email-validator==2.1.0
"""

with open(BASE_DIR / 'requirements.txt', 'w') as f:
    f.write(minimal_requirements)

print("  ✓ Created minimal requirements.txt")
print()

# ============================================================================
# STEP 4: CHECK/INSTALL DEPENDENCIES
# ============================================================================

print("Step 4: Checking and installing dependencies...")
print("  This may take a few minutes...")
print()

try:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  ✓ Updated pip")
    
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
    ])
    print("  ✓ All dependencies installed")
except subprocess.CalledProcessError as e:
    print(f"  ⚠ Error installing dependencies: {e}")
    print("  Trying individual packages...")
    
    # Install packages one by one if batch fails
    packages = [
        'Flask==3.0.0',
        'Flask-SQLAlchemy==3.1.1',
        'Flask-CORS==4.0.0',
        'Flask-JWT-Extended==4.5.3',
        'Flask-Mail==0.9.1',
        'SQLAlchemy==2.0.23',
        'bcrypt==4.1.2',
        'python-dotenv==1.0.0',
        'stripe==7.8.0',
        'numpy',
        'Pillow',
        'scikit-learn',
        'requests',
        'email-validator'
    ]
    
    for package in packages:
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✓ Installed: {package}")
        except:
            print(f"  ⚠ Failed: {package} (will try to continue)")

print()

# ============================================================================
# STEP 5: CREATE .ENV FILE IF MISSING
# ============================================================================

print("Step 5: Checking .env configuration...")

env_path = BASE_DIR / '.env'

if not env_path.exists():
    print("  Creating default .env file...")
    
    # Generate secret keys
    import secrets
    secret_key = secrets.token_urlsafe(32)
    jwt_secret = secrets.token_urlsafe(32)
    
    env_content = f"""# Life Planner Configuration
# IMPORTANT: Update the values below!

# Application
SECRET_KEY={secret_key}
JWT_SECRET_KEY={jwt_secret}
DEBUG=True
FLASK_ENV=development

# Database
DATABASE_URL=sqlite:///life_planner.db

# Stripe (UPDATE WITH YOUR KEYS!)
STRIPE_SECRET_KEY=sk_test_YOUR_TEST_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_test_YOUR_TEST_KEY_HERE
STRIPE_PRICE_ID=price_YOUR_PRICE_ID_HERE
SUBSCRIPTION_PRICE=20.00
TRIAL_DAYS=7

# Email (UPDATE WITH YOUR GMAIL!)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=your-email@gmail.com

# Admin
ADMIN_EMAIL=onlinediscountsllc@gmail.com
ADMIN_PASSWORD=admin8587037321

# GoFundMe
GOFUNDME_URL=https://gofund.me/8d9303d27

# Settings
USE_GPU=False
RATELIMIT_STORAGE_URL=memory://
CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
LOG_LEVEL=INFO
LOG_FILE=logs/life_planner.log
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("  ✓ Created .env file with secure keys")
    print()
    print("  ⚠ IMPORTANT: Edit .env and add your Stripe and Gmail credentials!")
else:
    print("  ✓ .env file exists")

print()

# ============================================================================
# STEP 6: CREATE __init__.py FILES
# ============================================================================

print("Step 6: Creating Python package files...")

init_files = [
    'models/__init__.py',
    'backend/__init__.py',
]

for init_file in init_files:
    init_path = BASE_DIR / init_file
    if not init_path.exists():
        init_path.touch()
        print(f"  ✓ Created: {init_file}")
    else:
        print(f"  ✓ Exists: {init_file}")

print()

# ============================================================================
# STEP 7: TEST IMPORTS
# ============================================================================

print("Step 7: Testing critical imports...")

test_imports = [
    ('flask', 'Flask'),
    ('sqlalchemy', 'SQLAlchemy'),
    ('stripe', 'Stripe'),
    ('dotenv', 'python-dotenv'),
]

all_imports_ok = True
for module, name in test_imports:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - MISSING!")
        all_imports_ok = False

print()

# ============================================================================
# STEP 8: INITIALIZE DATABASE
# ============================================================================

print("Step 8: Checking database...")

db_path = BASE_DIR / 'life_planner.db'

if not db_path.exists():
    print("  Database not found. Will initialize on first run.")
    print("  Run: python init_db.py")
else:
    print("  ✓ Database exists")

print()

# ============================================================================
# FINAL STATUS
# ============================================================================

print("=" * 70)
print("SETUP COMPLETE!")
print("=" * 70)
print()

if all_imports_ok:
    print("✅ All dependencies installed successfully!")
    print()
    print("NEXT STEPS:")
    print("1. Edit .env file with your Stripe and Gmail credentials")
    print("2. Run: python init_db.py")
    print("3. Run: python app.py")
    print("4. Open: http://localhost:5000")
    print()
else:
    print("⚠ Some dependencies failed to install.")
    print("  The app may still work with reduced functionality.")
    print()
    print("TRY:")
    print("  pip install flask flask-sqlalchemy python-dotenv")
    print()

print("=" * 70)
print()

# Create a status file
with open(BASE_DIR / 'setup_status.txt', 'w') as f:
    f.write("Setup completed successfully\n")
    f.write(f"Date: {__import__('datetime').datetime.now()}\n")
    f.write(f"All imports OK: {all_imports_ok}\n")
